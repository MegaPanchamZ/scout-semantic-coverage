"""AutoVLA planning server: batched vLLM (video modality) behind HTTP.

Uses the offline vLLM engine with the faithful AutoVLA video prompt and a small
queue so concurrent requests to /plan are merged into one llm.generate batch
(vLLM's continuous batching). One model load serves all CARLA workers.

Run:
    VLLM_USE_FLASHINFER_SAMPLER=0 <vllm-env>/bin/python plan_server.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import queue
import os
import sys
import time
from concurrent.futures import Future

import numpy as np
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contract import (  # noqa: E402
    CAMERA_TYPES,
    MAX_PIXELS,
    MIN_PIXELS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    StepInputs,
    bins_to_trajectory,
    build_prompt,
    load_codebook,
    text_to_bins,
    AUTOVLA_ROOT,
)

MODEL = os.path.join(AUTOVLA_ROOT, "checkpoints", "AutoVLA-hf")
HOST, PORT = "127.0.0.1", 8101
MAX_BATCH = 64
WINDOW_S = 0.004
MAX_TOKENS = 64

app = FastAPI()
_ready = {"ok": False}
LLM = None
PROCESSOR = None
CODEBOOK = load_codebook()
_req_q: "queue.Queue[tuple[StepReq, Future]]" = queue.Queue()


class StepReq(BaseModel):
    frames: dict[str, list[str]]  # camera -> 4 base64 JPEG (no data: prefix)
    velocity: float = 5.0
    acceleration: float = 0.0
    command: str = "forward"
    n_poses: int = 10


def _pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _build_input(req: StepReq):
    paths = {cam: [f"mem://{cam}_{i}" for i in range(4)] for cam in CAMERA_TYPES}
    step = StepInputs(
        frames=paths, velocity=req.velocity, acceleration=req.acceleration, command=req.command
    )
    prompt = build_prompt(PROCESSOR, step)
    videos = {cam: [_pil(b) for b in req.frames[cam]] for cam in CAMERA_TYPES}
    return {"prompt": prompt, "multi_modal_data": {"video": [videos[c] for c in CAMERA_TYPES]}}


def _to_result(out) -> dict:
    ids = list(out.outputs[0].token_ids)
    text = out.outputs[0].text
    bins = [i - 151665 for i in ids if i >= 151665] or text_to_bins(text)
    res = {
        "action_bins": bins,
        "prompt_tokens": len(out.prompt_token_ids),
        "completion_tokens": len(ids),
        "text": text,
    }
    if bins:
        res["trajectory"] = bins_to_trajectory(bins, CODEBOOK).tolist()
    return res


def _worker() -> None:
    from vllm import SamplingParams

    sp = SamplingParams(temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, max_tokens=MAX_TOKENS)
    while True:
        first = _req_q.get()
        batch = [first]
        t0 = time.time()
        while len(batch) < MAX_BATCH and (time.time() - t0) < WINDOW_S:
            try:
                batch.append(_req_q.get_nowait())
            except queue.Empty:
                break
        try:
            inputs = [_build_input(r) for r, _ in batch]
            outs = LLM.generate(inputs, sampling_params=sp)
            for (_, fut), out in zip(batch, outs):
                fut.set_result(_to_result(out))
        except Exception as exc:  # noqa: BLE001
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc)


@app.get("/health")
def health() -> dict:
    return {"ready": _ready["ok"], "queue": _req_q.qsize()}


@app.post("/plan")
async def plan(req: StepReq) -> dict:
    fut: Future = Future()
    _req_q.put((req, fut))
    return await asyncio.wrap_future(fut)


def _init_engine() -> None:
    global LLM, PROCESSOR
    from transformers import AutoProcessor
    from vllm import LLM as _LLM

    PROCESSOR = AutoProcessor.from_pretrained(MODEL)
    LLM = _LLM(
        model=MODEL,
        max_model_len=4096,
        gpu_memory_utilization=0.4,
        limit_mm_per_prompt={"image": 12, "video": 3},
        mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        disable_log_stats=True,
    )
    import threading

    threading.Thread(target=_worker, daemon=True).start()
    _ready["ok"] = True


if __name__ == "__main__":
    import uvicorn

    print("initializing engine ...", flush=True)
    _init_engine()
    print(f"engine ready, serving on {HOST}:{PORT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
