"""Shared SGLang setup for AutoVLA serving (mirrors autovla_serve/contract.py).

The SGLang Engine is a drop-in alternative to vLLM's offline `LLM` for the
AutoVLA planning step (3 cameras x 4 frames, video modality).

Two environment quirks on this box:
  * SGLang JIT-compiles small CUDA kernels with `ninja`; the venv's `ninja`
    must be on PATH, otherwise the scheduler dies at load.
  * DeepGEMM's JIT requires NVCC >= 12.9 but the container ships CUDA 12.8, so
    its JIT must be disabled (`SGLANG_ENABLE_JIT_DEEPGEMM=0`).

No engine imports at module import time: keep this importable from workers.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Same default as contract.AUTOVLA_ROOT (contract is imported lazily below).
AUTOVLA_ROOT = os.environ.get(
    "AUTOVLA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "AutoVLA")
)
MODEL = os.path.join(AUTOVLA_ROOT, "checkpoints", "AutoVLA-hf")
FRAMES = os.environ.get("AUTOVLA_FRAMES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames"))

os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "0")
if os.environ.get("SGLANG_ENV_BIN"):
    os.environ["PATH"] = os.environ["SGLANG_ENV_BIN"] + ":" + os.environ.get("PATH", "")

CONTEXT_LENGTH = 4096
MEM_FRACTION_STATIC = 0.5

# Autodetected worker count is 1 for Qwen-VL (the fast image processor contends
# with the serving GPU), which serialises video preprocessing and caps the
# engine at ~3.6 steps/s. Overriding it is the single biggest SGLang win here:
# 2-16 workers all give ~29-30 steps/s at batch 128.
MM_PROCESSOR_WORKER_NUM = int(os.environ.get("SGLANG_MM_WORKERS", "4"))


def make_engine(
    context_length: int = CONTEXT_LENGTH,
    mem_fraction_static: float = MEM_FRACTION_STATIC,
    mm_processor_worker_num: int = MM_PROCESSOR_WORKER_NUM,
    **kwargs,
):
    """Create an SGLang Engine configured like the vLLM AutoVLA setup."""
    import sglang as sgl

    from contract import MAX_PIXELS, MIN_PIXELS

    cfg = dict(
        model_path=MODEL,
        context_length=context_length,
        mem_fraction_static=mem_fraction_static,
        mm_processor_worker_num=mm_processor_worker_num,
        mm_process_config={
            "image": {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
            "video": {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        },
        log_level="warning",
    )
    cfg.update(kwargs)
    return sgl.Engine(**cfg)


def load_frames():
    """3 cameras x 4 frames as PIL images (the AutoVLA video modality)."""
    from PIL import Image

    from contract import CAMERA_TYPES

    return {
        cam: [Image.open(f"{FRAMES}/{cam}_{i}.jpg").convert("RGB") for i in range(4)]
        for cam in CAMERA_TYPES
    }


def frame_paths():
    from contract import CAMERA_TYPES

    return {cam: [f"{FRAMES}/{cam}_{i}.jpg" for i in range(4)] for cam in CAMERA_TYPES}


def video_list(frames):
    """SGLang `video_data` item for one request: one video per camera."""
    from contract import CAMERA_TYPES

    return [frames[cam] for cam in CAMERA_TYPES]


def build_prompt(processor, velocity: float = 5.0, command: str = "forward") -> str:
    from contract import StepInputs, build_prompt as _contract_build_prompt

    step = StepInputs(
        frames=frame_paths(), velocity=velocity, acceleration=0.0, command=command
    )
    # Exactly the contract prompt (includes add_vision_id=True), so SGLang and
    # vLLM tokenize the same 952-token input.
    return _contract_build_prompt(processor, step)


def sampling_params(max_new_tokens: int = 64):
    from contract import TEMPERATURE, TOP_P

    # vLLM TOP_K=0 means "all tokens"; SGLang spells that -1.
    return {
        "temperature": TEMPERATURE,
        "top_k": -1,
        "top_p": TOP_P,
        "max_new_tokens": max_new_tokens,
    }
