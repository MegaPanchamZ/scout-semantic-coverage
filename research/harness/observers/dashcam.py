from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from time import monotonic
from typing import Any

from research.harness.models import RunResult, ScenarioSpec


@dataclass(slots=True)
class DashcamObserverConfig:
    output_root_dir: Path
    stream_every_ticks: int = 1
    image_width: int = 960
    image_height: int = 540
    field_of_view_deg: float = 90.0


class DashcamObserver:
    def __init__(self, config: DashcamObserverConfig) -> None:
        self.config = config
        self._carla: Any | None = None
        self._camera: Any | None = None
        self._status = "not-started"
        self._notes: list[str] = []
        self._run_output_dir: Path | None = None
        self._captured_frames: list[dict[str, Any]] = []
        self._camera_frames_by_world_frame: dict[int, Any] = {}
        self._camera_frame_queue: Queue[int] = Queue()
        self._latest_sensor_frame: int | None = None
        self._missing_capture_ticks: list[int] = []

    def on_run_start(self, scenario: ScenarioSpec, context: dict[str, Any]) -> None:
        ego_vehicle = context.get("ego_vehicle")
        world = context.get("world")
        carla = context.get("carla")
        if ego_vehicle is None or world is None or carla is None:
            self._status = "context-missing"
            self._notes = ["World, ego vehicle, or CARLA module missing from context."]
            context["dashcam"] = self
            return

        self._carla = carla
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._run_output_dir = (self.config.output_root_dir / f"{scenario.scenario_id}-{timestamp}").resolve()
        self._run_output_dir.mkdir(parents=True, exist_ok=True)
        self._captured_frames = []
        self._camera_frames_by_world_frame = {}
        self._camera_frame_queue = Queue()
        self._latest_sensor_frame = None
        self._missing_capture_ticks = []
        self._notes = []

        blueprint_library = world.get_blueprint_library()
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.config.image_width))
        camera_bp.set_attribute("image_size_y", str(self.config.image_height))
        camera_bp.set_attribute("fov", str(self.config.field_of_view_deg))
        camera_transform = self._carla.Transform(
            self._carla.Location(x=1.5, z=2.4),
            self._carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        self._camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        self._camera.listen(self._on_image)
        self._status = "collecting"
        self._notes.append(f"Dashcam capture writing to {self._run_output_dir}.")
        context["dashcam"] = self

    def on_tick(self, tick_index: int, context: dict[str, Any]) -> None:
        if self._status != "collecting" or self._run_output_dir is None:
            return

        logical_tick = int(context.get("telemetry", {}).get("tick") or (tick_index + 1))
        if self.config.stream_every_ticks > 1 and ((logical_tick - 1) % self.config.stream_every_ticks) != 0:
            return

        world = context.get("world")
        if world is None:
            self._status = "world-missing"
            self._notes.append("World missing from context during dashcam capture.")
            return

        world_frame = int(world.get_snapshot().frame)
        image, matched_world_frame, capture_mode = self._get_image_for_world_frame(world_frame)
        if image is None:
            self._missing_capture_ticks.append(logical_tick)
            self._prune_frame_cache(world_frame)
            return

        output_path = self._run_output_dir / f"tick_{logical_tick:05d}.png"
        image.save_to_disk(str(output_path))
        self._captured_frames.append(
            {
                "tick": logical_tick,
                "world_frame": world_frame,
                "matched_world_frame": matched_world_frame,
                "sensor_frame": int(getattr(image, "frame", matched_world_frame)),
                "capture_mode": capture_mode,
                "path": str(output_path),
            }
        )
        self._prune_frame_cache(world_frame)

    def on_run_end(self, result: RunResult, context: dict[str, Any]) -> None:
        del context
        result.metadata["dashcam"] = {
            "status": self._status,
            "notes": list(self._notes),
            "output_dir": str(self._run_output_dir) if self._run_output_dir is not None else None,
            "frame_count": len(self._captured_frames),
            "stream_every_ticks": self.config.stream_every_ticks,
            "image_width": self.config.image_width,
            "image_height": self.config.image_height,
            "frames": list(self._captured_frames),
            "missing_capture_ticks": list(self._missing_capture_ticks),
        }
        self._destroy()

    def _on_image(self, image: Any) -> None:
        frame = int(getattr(image, "frame", -1))
        if frame >= 0:
            self._camera_frames_by_world_frame[frame] = image
            self._latest_sensor_frame = frame
            self._camera_frame_queue.put(frame)

    def _get_image_for_world_frame(self, world_frame: int) -> tuple[Any | None, int | None, str]:
        image = self._camera_frames_by_world_frame.pop(world_frame, None)
        if image is not None:
            return image, world_frame, "exact"

        deadline = monotonic() + 0.25
        while monotonic() < deadline:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            try:
                self._camera_frame_queue.get(timeout=min(remaining, 0.02))
            except Empty:
                continue
            image = self._camera_frames_by_world_frame.pop(world_frame, None)
            if image is not None:
                return image, world_frame, "awaited-exact"

        fallback_world_frame = self._latest_eligible_world_frame(world_frame)
        if fallback_world_frame is None:
            latest_frame = self._latest_sensor_frame
            if latest_frame is None:
                self._notes.append(
                    f"No dashcam image was available for logical/world tick {world_frame}; sensor callback produced no frames yet."
                )
            else:
                self._notes.append(
                    f"No dashcam image was available for world frame {world_frame}; latest sensor frame was {latest_frame}."
                )
            return None, None, "missing"

        image = self._camera_frames_by_world_frame.pop(fallback_world_frame, None)
        if image is None:
            return None, None, "missing"
        if fallback_world_frame != world_frame:
            self._notes.append(
                f"Used latest available dashcam image from world frame {fallback_world_frame} for requested world frame {world_frame}."
            )
        return image, fallback_world_frame, "fallback-latest"

    def _latest_eligible_world_frame(self, world_frame: int) -> int | None:
        eligible_frames = [frame for frame in self._camera_frames_by_world_frame if frame <= world_frame]
        if not eligible_frames:
            return None
        return max(eligible_frames)

    def _prune_frame_cache(self, world_frame: int) -> None:
        stale_frames = [frame for frame in self._camera_frames_by_world_frame if frame < (world_frame - 8)]
        for frame in stale_frames:
            self._camera_frames_by_world_frame.pop(frame, None)

    def _destroy(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            try:
                self._camera.destroy()
            except Exception:
                pass
        self._camera = None
        self._carla = None
        self._camera_frames_by_world_frame = {}
        self._camera_frame_queue = Queue()
        self._latest_sensor_frame = None
