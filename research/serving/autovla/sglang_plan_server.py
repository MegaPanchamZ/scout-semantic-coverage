"""AutoVLA planning server: batched SGLang engine behind HTTP (mirrors
autovla_serve/plan_server.py, but on the SGLang offline Engine).

SGLang's offline Engine is asyncio-based (`Engine.async_generate`), so unlike the
vLLM server we batch *inside the event loop*: concurrent `/plan` requests are
collected for a few milliseconds and issued as one `async_generate` call.

Run:
    <sglang-env>/bin/python sglang_plan_server.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import sys
import time
from dataclasses import dataclass

from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contract import (  # noqa: E402
    CAMERA_TYPES,
    StepInputs,
    bins_to_trajectory,
    build_prompt,
    load_codebook,
    text_to_bins,
)
from sglang_common import make_engine, sampling_params  # noqa: E402

HOST, PORT = "127.0.0.1", 8201
MAX_BATCH = 64
WINDOW_S = 0.004
MAX_TOKENS = 64

app = FastAPI()
_ready = {"ok": False}
ENGINE = None
PROCESSOR = None
CODEBOOK = load_codebook()
_req_q: "asyncio.Queue[Job]" = asyncio.Queue()


class StepReq(BaseModel):
    frames: dict[str, list[str]]  # camera -> 4 base64 JPEG (no data: prefix)
    velocity: float = 5.0
    acceleration: float = 0.0
    command: str = "forward"
    n_poses: int = 10


@dataclass
class Job:
    req: StepReq
    future: asyncio.Future


def _pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _build_prompt(req: StepReq) -> str:
    paths = {cam: [f"mem://{cam}_{i}" for i in range(4)] for cam in CAMERA_TYPES}
    step = StepInputs(
        frames=paths, velocity=req.velocity, acceleration=req.acceleration, command=req.command
    )
    return build_prompt(PROCESSOR, step)


def _to_result(out) -> dict:
    text = out["text"]
    meta = out.get("meta_info", {})
    bins = text_to_bins(text)
    res = {
        "action_bins": bins,
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "text": text,
    }
    if bins:
        res["trajectory"] = bins_to_trajectory(bins, CODEBOOK).tolist()
    return res


async def _batcher() -> None:
    sp = sampling_params(max_new_tokens=MAX_TOKENS)
    while True:
        batch = [await _req_q.get()]
        t0 = time.time()
        while len(batch) < MAX_BATCH and (time.time() - t0) < WINDOW_S:
            try:
                batch.append(_req_q.get_nowait())
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.0005)
        try:
            prompts = [_build_prompt(j.req) for j in batch]
            videos = [
                [[_pil(b) for b in j.req.frames[cam]] for cam in CAMERA_TYPES]
                for j in batch
            ]
            outs = await ENGINE.async_generate(
                prompt=prompts, video_data=videos, sampling_params=sp
            )
            if isinstance(outs, dict):
                outs = [outs]
            for job, out in zip(batch, outs):
                if not job.future.done():
                    job.future.set_result(_to_result(out))
        except Exception as exc:  # noqa: BLE001
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(exc)


@app.on_event("startup")
async def _startup() -> None:
    global ENGINE, PROCESSOR
    from transformers import AutoProcessor

    from sglang_common import MODEL

    PROCESSOR = AutoProcessor.from_pretrained(MODEL)
    ENGINE = make_engine(context_length=4096)
    asyncio.create_task(_batcher())
    _ready["ok"] = True


@app.get("/health")
def health() -> dict:
    return {"ready": _ready["ok"], "queue": _req_q.qsize()}


@app.post("/plan")
async def plan(req: StepReq) -> dict:
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    await _req_q.put(Job(req, fut))
    return await fut


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
