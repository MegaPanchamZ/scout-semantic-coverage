"""Bounded, synchronized in-memory camera histories for AutoVLA."""
from __future__ import annotations

from collections import OrderedDict
from threading import Condition
import time
from typing import Any


class CameraHistory:
    def __init__(self, names: list[str], stride: int) -> None:
        self.stride = stride
        self._frames: dict[str, OrderedDict[int, Any]] = {name: OrderedDict() for name in names}
        self._condition = Condition()

    def add(self, name: str, frame: int, image: Any) -> None:
        if frame % self.stride:
            return
        with self._condition:
            frames = self._frames[name]
            frames[frame] = image
            while len(frames) > 8:
                frames.popitem(last=False)
            self._condition.notify_all()

    def snapshot(self, current_frame: int, wait_seconds: float = 0.5) -> tuple[dict[str, list[Any]], list[int]] | None:
        newest = current_frame - current_frame % self.stride
        frame_ids = [newest - i * self.stride for i in (3, 2, 1, 0)]
        with self._condition:
            # Don't wait through the initial 1.5 s of history collection.
            if any(not frames or next(iter(frames)) > frame_ids[0] for frames in self._frames.values()):
                return None
            deadline = time.monotonic() + wait_seconds
            while not all(all(frame in frames for frame in frame_ids) for frames in self._frames.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return {name: [frames[frame] for frame in frame_ids]
                    for name, frames in self._frames.items()}, frame_ids

    def clear(self) -> None:
        with self._condition:
            for frames in self._frames.values():
                frames.clear()
