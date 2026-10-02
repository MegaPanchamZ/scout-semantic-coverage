"""Engine-agnostic AutoVLA serving contract.

Mirrors the inference path in the upstream AutoVLA repo:
  research/models/AutoVLA/models/autovla.py  (AutoVLA.get_prompt / predict)
  research/models/AutoVLA/models/action_tokenizer.py (decode_token_ids_to_trajectory)
so that an external engine (vLLM / SGLang / HF) can reproduce one inference step
exactly: build the 3-video x 4-frame prompt, run generation, map generated token
ids in [ACTION_START_ID, ...] through the 2048-bin codebook to a 10-pose trajectory.

No engine imports here on purpose; only numpy + torch.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

# ---- constants pulled from config/eval/qwen2.5-vl-3B-nusc-sft-eval.yaml ----
ACTION_START_ID = 151665          # <action_0>
NBINS = 2048                      # <action_0> .. <action_2047> -> 151665..153712
NUM_POSES = 10                    # future poses
INTERVAL_LENGTH = 0.5             # s
TIME_HORIZON = 5.0                # s
MAX_LENGTH = 2048
TEMPERATURE = 0.01
TOP_K = 0
TOP_P = 1.0
MIN_PIXELS = 109760
MAX_PIXELS = 109760

CAMERA_TYPES = ("front_camera", "front_left_camera", "front_right_camera")
FRAMES_PER_VIDEO = 4

# research/models/AutoVLA unless AUTOVLA_ROOT points elsewhere.
AUTOVLA_ROOT = os.environ.get(
    "AUTOVLA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "AutoVLA")
)

DEFAULT_CODEBOOK = (
    os.path.join(AUTOVLA_ROOT, "codebook_cache", "agent_vocab.pkl")
)


@dataclass
class StepInputs:
    """Everything AutoVLA.get_prompt needs for one planning step."""

    frames: dict[str, list[str]]      # camera -> [4 image paths]
    velocity: float                   # m/s (scalar; repo also accepts (x, y))
    acceleration: float               # m/s^2
    command: str                      # e.g. "forward", "turn left"
    sensor_data_path: str | None = None  # prefix for relative frame paths


SYSTEM_TEXT = (
    "You are an Advanced Driver Assistance and Full Self-Driving System. "
    "You will be provided with video observations from the ego vehicle\u2019s "
    "surrounding cameras, along with the vehicle\u2019s current dynamic states. "
    "Your task is to predict the most appropriate driving action for the next five seconds."
)


def build_messages(step: StepInputs, use_cot: bool = False) -> list[dict[str, Any]]:
    """Reproduce AutoVLA.get_prompt()'s message list verbatim.

    Includes the system message (the non-CoT variant by default, matching
    config/eval/qwen2.5-vl-3B-nusc-sft-eval.yaml: use_cot=false).
    """
    if use_cot:
        raise NotImplementedError("CoT system prompt not ported (campaign uses use_cot=false)")
    velocity = step.velocity
    acceleration = step.acceleration
    instruction = step.command.lower()

    def uri(i: int, camera: str) -> str:
        img = step.frames[camera][i]
        if step.sensor_data_path:
            img = os.path.join(step.sensor_data_path, img)
        return f"file://{img}"

    videos = {
        "front_camera": "the front view",
        "front_left_camera": "the front-left view",
        "front_right_camera": "the front-right view",
    }
    ordinals = ("first", "second", "third")

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "The autonomous vehicle is equipped with three cameras mounted at "
                "the front, left, and right, enabling a comprehensive perception of "
                "the surrounding environment."
            ),
        }
    ]
    for ordinal, camera in zip(ordinals, CAMERA_TYPES):
        content.append(
            {
                "type": "text",
                "text": (
                    f"The {ordinal} video presents {videos[camera]} of the vehicle, "
                    "comprising four sequential frames sampled at 2 Hz."
                ),
            }
        )
        content.append(
            {
                "type": "video",
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
                "video": [uri(i, camera) for i in range(FRAMES_PER_VIDEO)],
            }
        )
    content.append(
        {
            "type": "text",
            "text": (
                f"The current velocity of the vehicle is {velocity:.3f} m/s, and the "
                f"current acceleration is {acceleration:.3f} m/s\u00b2. The driving "
                f"instruction is: {instruction}. Based on this information, plan the "
                "action trajectory for the autonomous vehicle over the next five seconds."
            ),
        }
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_TEXT}]},
        {"role": "user", "content": content},
    ]


def build_prompt(processor: Any, step: StepInputs, use_cot: bool = False) -> str:
    """Render the exact prompt text AutoVLA feeds to the processor."""
    return processor.apply_chat_template(
        build_messages(step, use_cot=use_cot),
        tokenize=False,
        add_generation_prompt=True,
        add_vision_id=True,
    )


# ---------------------------------------------------------------------------
# trajectory decode (verbatim port of action_tokenizer.py)
# ---------------------------------------------------------------------------
def _transform_to_global(pos_local, head_local, pos_now, head_now):
    cos, sin = head_now.cos(), head_now.sin()
    rot_mat = torch.zeros((head_now.shape[0], 2, 2), device=head_now.device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = sin
    rot_mat[:, 1, 0] = -sin
    rot_mat[:, 1, 1] = cos
    pos_global = torch.bmm(pos_local, rot_mat)
    pos_global = pos_global + pos_now.unsqueeze(1)
    head_global = None if head_local is None else head_local + head_now.unsqueeze(1)
    return pos_global, head_global


def load_codebook(path: str = DEFAULT_CODEBOOK) -> torch.Tensor:
    with open(path, "rb") as f:
        code_book = pickle.load(f)["token_all"]["veh"]
    return torch.tensor(code_book)  # (2048, 6, 4, 2)


def bins_to_trajectory(bins: Sequence[int], code_book: torch.Tensor) -> np.ndarray:
    """Roll a list of codebook-bin indices into a global trajectory.

    Returns array shaped (len(bins), 3): x, y, heading (ego frame at t0).
    Mirrors AutoVLA.decode_token_ids_to_trajectory + rollout, including the
    leading-origin trim ([0, 1:]).
    """
    if len(bins) > NUM_POSES:
        bins = bins[:NUM_POSES]
    elif len(bins) < NUM_POSES:
        bins = list(bins) + [0] * (NUM_POSES - len(bins))

    action_tokens = code_book[torch.tensor(bins, dtype=torch.long)]  # (T, 6, 4, 2)
    time_steps = action_tokens.shape[0]

    pos_a = torch.tensor([[[0.0, 0.0]]])  # [1,1,2]
    head_a = torch.tensor([[0.0]])        # [1,1]
    for t in range(time_steps):
        next_token_traj_all = action_tokens[None, t]  # [1,6,4,2]
        token_traj_global = _transform_to_global(
            pos_local=next_token_traj_all.flatten(1, 2),
            head_local=None,
            pos_now=pos_a[:, t],
            head_now=head_a[:, t],
        )[0].view(*next_token_traj_all.shape)
        pos_a_next = token_traj_global[:, -1].mean(dim=1)
        diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
        head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])
        pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
        head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)

    trajectory = torch.cat([pos_a, head_a.unsqueeze(-1)], dim=-1)  # [1,T+1,3]
    return trajectory[0, 1:].numpy()


def token_ids_to_bins(token_ids: Sequence[int]) -> list[int]:
    """Filter generated ids to action tokens and convert to codebook bins."""
    return [int(t) - ACTION_START_ID for t in token_ids if int(t) >= ACTION_START_ID]


def text_to_bins(text: str) -> list[int]:
    """Recover action bins from detokenized text (`<action_N>` markers)."""
    import re

    return [int(m) for m in re.findall(r"<action_(\d+)>", text)]
