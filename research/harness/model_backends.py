"""Pluggable inference backends for vision-language-action (VLA) agents.

The harness needs to run the same agent (currently AutoVLA) through several
serving stacks, chosen at launch time:

* ``torch``  - in-process Hugging Face / PyTorch model (the historical path,
  supports activation hooks for KMNC/LSA coverage).
* ``http``   - a batched planning server exposing AutoVLA's exact video prompt
  contract (e.g. ``research/serving/autovla/plan_server.py`` over vLLM).
* ``openai`` - any OpenAI-compatible chat-completions server that hosts the
  converted AutoVLA checkpoint (vLLM ``serve``, SGLang server, llama.cpp
  ``llama-server``, ...).

A backend's only job is to turn the harness feature dict into a planned
trajectory. Sensors, routing, and low-level control stay in the agent adapter
(``autovla_bridge.AutoVlaAdapter``), so all backends get identical tracking and
the experiment differs only in how the model is served.

The remote backends deliberately depend only on the standard library + numpy so
they work in the lightweight harness environment (no torch required).
"""
from __future__ import annotations

import base64
import json
import math
import pathlib
import pickle
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

ACTION_START_ID = 151665
NBINS = 2048
NUM_POSES = 10
DEFAULT_CODEBOOK = (
    pathlib.Path(__file__).resolve().parents[1]
    / "models"
    / "AutoVLA"
    / "codebook_cache"
    / "agent_vocab.pkl"
)


class BackendUnavailable(RuntimeError):
    """Raised when a backend cannot serve inference (or lacks instrumentation)."""


class PlanBackend(ABC):
    """Turns harness features into a planned trajectory."""

    name = "backend"
    supports_activation_hooks = False

    @abstractmethod
    def plan(self, features: dict[str, Any]) -> tuple[np.ndarray, str]:
        """Return ``(poses, cot_text)`` where ``poses`` is ``(N, 3)`` [x, y, head]."""

    def close(self) -> None:  # pragma: no cover - optional cleanup
        pass

    @property
    def torch_model(self) -> Any:
        raise BackendUnavailable(f"backend '{self.name}' does not expose a torch model")


# --------------------------------------------------------------------------- utils
def _as_scalar(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(math.hypot(float(value[0]), float(value[1]))) if len(value) >= 2 else float(value[0])
    return float(value)


def load_codebook(path: pathlib.Path | str = DEFAULT_CODEBOOK) -> np.ndarray:
    with open(path, "rb") as f:
        return np.asarray(pickle.load(f)["token_all"]["veh"], dtype=np.float64)


def text_to_bins(text: str) -> list[int]:
    import re

    return [int(m) for m in re.findall(r"<action_(\d+)>", text)]


def bins_to_trajectory(bins: Sequence[int], code_book: np.ndarray) -> np.ndarray:
    """Numpy port of AutoVLA.ActionTokenizer.decode_token_ids_to_trajectory.

    Returns ``(len(bins), 3)`` ego-frame poses, mirroring the torch rollout
    (including the leading-origin trim).
    """
    bins = list(bins)
    if len(bins) > NUM_POSES:
        bins = bins[:NUM_POSES]
    elif len(bins) < NUM_POSES:
        bins = bins + [0] * (NUM_POSES - len(bins))

    tokens = code_book[np.asarray(bins, dtype=np.int64)]  # (T, 6, 4, 2)
    pos_a = np.zeros((1, 2), dtype=np.float64)
    head_a = np.zeros((1,), dtype=np.float64)
    positions = [pos_a.copy()]
    headings = [head_a.copy()]
    for t in range(tokens.shape[0]):
        local = tokens[t].reshape(-1, 2)  # (24, 2)
        c, s = math.cos(float(head_a[0])), math.sin(float(head_a[0]))
        rot = np.array([[c, s], [-s, c]], dtype=np.float64)
        glob = local @ rot + pos_a[0]
        glob = glob.reshape(6, 4, 2)
        pos_next = glob[:, -1, :].mean(axis=0)
        diff = glob[0, -1] - glob[3, -1]
        head_next = math.atan2(float(diff[1]), float(diff[0]))
        pos_a = np.asarray([pos_next], dtype=np.float64)
        head_a = np.asarray([head_next], dtype=np.float64)
        positions.append(pos_a.copy())
        headings.append(head_a.copy())
    pos = np.concatenate(positions, axis=0)  # (T+1, 2)
    head = np.concatenate(headings, axis=0)  # (T+1,)
    traj = np.concatenate([pos, head[:, None]], axis=1)
    return traj[1:]  # drop origin, matching upstream AutoVLA


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - configured endpoint
        return json.load(resp)


def _b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# --------------------------------------------------------------------- torch backend
class TorchAutoVLABackend(PlanBackend):
    """In-process Hugging Face AutoVLA (supports activation hooks)."""

    name = "torch"
    supports_activation_hooks = True

    def __init__(
        self,
        repo_path: pathlib.Path | None = None,
        checkpoint_dir: pathlib.Path | None = None,
        device: str = "cuda",
    ) -> None:
        import sys

        default_repo = pathlib.Path(__file__).resolve().parents[1] / "models" / "AutoVLA"
        repo = pathlib.Path(repo_path) if repo_path else default_repo
        checkpoint = pathlib.Path(checkpoint_dir) if checkpoint_dir else repo / "checkpoints" / "AutoVLA-hf"
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        import torch  # noqa: E402  (deferred: only the torch backend needs torch)
        from models.autovla import AutoVLA  # noqa: E402

        self._torch = torch
        config = {
            "model": {
                "use_cot": False,
                "pretrained_model_path": str(checkpoint),
                "train_vision_backbone": False,
                "train_lm_backbone": True,
                "codebook_cache_path": str(repo / "codebook_cache" / "agent_vocab.pkl"),
                "trajectory": {"num_poses": NUM_POSES, "interval_length": 0.5, "time_horizon": 5.0},
                "tokens": {"action_start_id": ACTION_START_ID, "ignore_index": -100, "assistant_id": [151644, 77091]},
                "video": {"min_pixels": 109760, "max_pixels": 109760},
            },
            "inference": {"sample": {"max_length": 2048, "temperature": 0.01, "top_k": 0, "top_p": 1.0}},
        }
        self._model = AutoVLA(config, inference=True, device=device)
        self._model.eval()

    def plan(self, features: dict[str, Any]) -> tuple[np.ndarray, str]:
        poses, cot = self._model.predict(features)
        if hasattr(poses, "detach"):
            poses = poses.detach().float().cpu().numpy()
        return np.asarray(poses), str(cot)

    @property
    def torch_model(self) -> Any:
        return self._model.vlm


# ---------------------------------------------------------------------- http backend
class HttpPlanBackend(PlanBackend):
    """AutoVLA ``/plan`` server (batched vLLM/SGLang over the video contract)."""

    name = "http"

    def __init__(self, endpoint: str, timeout: float = 60.0, n_poses: int = NUM_POSES) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = float(timeout)
        self.n_poses = int(n_poses)

    def plan(self, features: dict[str, Any]) -> tuple[np.ndarray, str]:
        images = features["images"]
        frames = {cam: [_b64_file(p) for p in paths] for cam, paths in images.items()}
        payload = {
            "frames": frames,
            "velocity": _as_scalar(features.get("vehicle_velocity")),
            "acceleration": _as_scalar(features.get("vehicle_acceleration")),
            "command": str(features.get("driving_command", "forward")),
            "n_poses": self.n_poses,
        }
        result = _post_json(f"{self.endpoint}/plan", payload, self.timeout)
        traj = result.get("trajectory")
        if not traj:
            raise BackendUnavailable(f"http backend returned no trajectory: {result.get('action_bins')}")
        return np.asarray(traj, dtype=np.float64), str(result.get("text", ""))


# -------------------------------------------------------------------- openai backend
class OpenAIChatBackend(PlanBackend):
    """OpenAI chat-completions server hosting the converted AutoVLA checkpoint.

    Uses the image modality (the OpenAI schema has no multi-frame video), which
    costs ~2x the vision tokens of the native video prompt but works with stock
    vLLM/SGLang/llama.cpp servers.
    """

    name = "openai"

    def __init__(
        self,
        endpoint: str,
        model: str = "autovla",
        timeout: float = 60.0,
        max_tokens: int = 64,
        temperature: float = 0.01,
        codebook_path: pathlib.Path | str = DEFAULT_CODEBOOK,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/chat/completions"):
            self.endpoint = f"{self.endpoint}/v1/chat/completions"
        self.model = model
        self.timeout = float(timeout)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self._code_book = load_codebook(codebook_path)

    def _messages(self, features: dict[str, Any]) -> list[dict[str, Any]]:
        images = features["images"]
        labels = {
            "front_camera": "the front view",
            "front_left_camera": "the front-left view",
            "front_right_camera": "the front-right view",
        }
        ordinals = ("first", "second", "third")
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "The autonomous vehicle is equipped with three cameras mounted at the front, "
                    "left, and right, enabling a comprehensive perception of the surrounding environment."
                ),
            }
        ]
        for ordinal, cam in zip(ordinals, labels):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"The {ordinal} video presents {labels[cam]} of the vehicle, "
                        "comprising four sequential frames sampled at 2 Hz."
                    ),
                }
            )
            for path in images[cam]:
                content.append(
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64_file(path)}}
                )
        velocity = _as_scalar(features.get("vehicle_velocity"))
        acceleration = _as_scalar(features.get("vehicle_acceleration"))
        command = str(features.get("driving_command", "forward")).lower()
        content.append(
            {
                "type": "text",
                "text": (
                    f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current "
                    f"acceleration is {acceleration:.3f} m/s\u00b2. The driving instruction is: "
                    f"{command}. Based on this information, plan the action trajectory for the "
                    "autonomous vehicle over the next five seconds."
                ),
            }
        )
        system = (
            "You are an Advanced Driver Assistance and Full Self-Driving System. You will be "
            "provided with video observations from the ego vehicle\u2019s surrounding cameras, along "
            "with the vehicle\u2019s current dynamic states. Your task is to predict the most "
            "appropriate driving action for the next five seconds."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    def plan(self, features: dict[str, Any]) -> tuple[np.ndarray, str]:
        payload = {
            "model": self.model,
            "messages": self._messages(features),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        result = _post_json(self.endpoint, payload, self.timeout)
        text = result["choices"][0]["message"]["content"]
        bins = text_to_bins(text)
        if not bins:
            raise BackendUnavailable(f"openai backend produced no action tokens: {text[:200]!r}")
        return bins_to_trajectory(bins, self._code_book), text


def make_backend(
    kind: str,
    *,
    endpoint: str | None = None,
    repo_path: pathlib.Path | None = None,
    checkpoint_dir: pathlib.Path | None = None,
    device: str = "cuda",
    timeout: float = 60.0,
    model: str = "autovla",
) -> PlanBackend:
    kind = (kind or "torch").lower()
    if kind == "torch":
        return TorchAutoVLABackend(repo_path=repo_path, checkpoint_dir=checkpoint_dir, device=device)
    if kind in {"http", "plan", "vllm", "sglang"}:
        if not endpoint:
            raise ValueError(f"backend '{kind}' requires an endpoint URL (--autovla-endpoint)")
        return HttpPlanBackend(endpoint, timeout=timeout)
    if kind in {"openai", "openai-chat", "llamacpp", "gguf"}:
        if not endpoint:
            raise ValueError(f"backend '{kind}' requires an endpoint URL (--autovla-endpoint)")
        return OpenAIChatBackend(endpoint, model=model, timeout=timeout)
    raise ValueError(f"unknown backend kind '{kind}' (expected torch, http, or openai)")
