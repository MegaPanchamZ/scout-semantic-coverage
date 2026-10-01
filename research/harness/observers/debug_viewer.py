from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from research.harness.models import RunResult, ScenarioSpec


@dataclass(slots=True)
class DebugViewerObserverConfig:
    image_width: int = 960
    image_height: int = 540
    topdown_height_m: float = 35.0
    enable_window: bool = True
    capture_sensor: str = "topdown"
    capture_targets: dict[int, Path] | None = None


class DebugViewerObserver:
    def __init__(self, config: DebugViewerObserverConfig) -> None:
        self.config = config
        self._pygame: Any | None = None
        self._carla: Any | None = None
        self._screen: Any | None = None
        self._font: Any | None = None
        self._front_camera: Any | None = None
        self._topdown_camera: Any | None = None
        self._latest_front_image: Any | None = None
        self._latest_topdown_image: Any | None = None
        self._latest_front: np.ndarray | None = None
        self._latest_topdown: np.ndarray | None = None
        self._status = "not-started"
        self._notes: list[str] = []
        self._capture_targets = dict(self.config.capture_targets or {})
        self._captured_paths: dict[int, str] = {}

    def on_run_start(self, scenario: ScenarioSpec, context: dict[str, Any]) -> None:
        del scenario
        ego_vehicle = context.get("ego_vehicle")
        world = context.get("world")
        carla = context.get("carla")
        if ego_vehicle is None or world is None or carla is None:
            self._status = "context-missing"
            self._notes = ["World, ego vehicle, or CARLA module missing from context."]
            context["debug_viewer"] = self
            return

        self._carla = carla
        if self.config.enable_window:
            try:
                import pygame  # type: ignore
            except ModuleNotFoundError:
                self._notes = ["pygame is not installed in the harness environment; continuing in capture-only mode."]
            else:
                self._pygame = pygame
                pygame.init()
                self._screen = pygame.display.set_mode((self.config.image_width * 2, self.config.image_height))
                pygame.display.set_caption("CARLA Harness Debug Viewer")
                self._font = pygame.font.SysFont("consolas", 20)
        self._spawn_cameras(world, ego_vehicle)
        if self._pygame is not None:
            self._status = "collecting"
        elif self._capture_targets:
            self._status = "capture-only"
        else:
            self._status = "headless"
        context["debug_viewer"] = self

    def on_tick(self, tick_index: int, context: dict[str, Any]) -> None:
        logical_tick = int(context.get("telemetry", {}).get("tick") or (tick_index + 1))
        self._capture_frame(logical_tick)

        if self._status != "collecting" or self._pygame is None or self._screen is None:
            return

        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self._notes.append("Debug viewer window was closed by the user.")
                self._status = "closed"
                return

        self._screen.fill((0, 0, 0))
        telemetry = context.get("telemetry", {})
        if self._latest_front is not None:
            self._draw_frame(self._latest_front, 0)
        if self._latest_topdown is not None:
            self._draw_frame(self._latest_topdown, self.config.image_width)
        self._draw_overlay(telemetry)
        self._pygame.display.flip()

    def on_run_end(self, result: RunResult, context: dict[str, Any]) -> None:
        del context
        missing_ticks = sorted(set(self._capture_targets) - set(self._captured_paths))
        result.metadata["debug_viewer"] = {
            "status": self._status,
            "notes": list(self._notes),
            "capture_sensor": self.config.capture_sensor,
            "captured_paths": {str(tick): path for tick, path in sorted(self._captured_paths.items())},
            "missing_capture_ticks": missing_ticks,
        }
        self._destroy()

    def _spawn_cameras(self, world: Any, ego_vehicle: Any) -> None:
        assert self._carla is not None
        blueprint_library = world.get_blueprint_library()

        front_bp = blueprint_library.find("sensor.camera.rgb")
        front_bp.set_attribute("image_size_x", str(self.config.image_width))
        front_bp.set_attribute("image_size_y", str(self.config.image_height))
        front_bp.set_attribute("fov", "90")
        front_transform = self._carla.Transform(
            self._carla.Location(x=1.5, z=2.4),
            self._carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        self._front_camera = world.spawn_actor(front_bp, front_transform, attach_to=ego_vehicle)
        self._front_camera.listen(self._on_front_image)

        top_bp = blueprint_library.find("sensor.camera.rgb")
        top_bp.set_attribute("image_size_x", str(self.config.image_width))
        top_bp.set_attribute("image_size_y", str(self.config.image_height))
        top_bp.set_attribute("fov", "90")
        top_transform = self._carla.Transform(
            self._carla.Location(x=0.0, z=self.config.topdown_height_m),
            self._carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )
        self._topdown_camera = world.spawn_actor(top_bp, top_transform, attach_to=ego_vehicle)
        self._topdown_camera.listen(self._on_topdown_image)

    def _on_front_image(self, image: Any) -> None:
        self._latest_front_image = image
        self._latest_front = self._image_to_array(image)

    def _on_topdown_image(self, image: Any) -> None:
        self._latest_topdown_image = image
        self._latest_topdown = self._image_to_array(image)

    def _capture_frame(self, tick: int) -> None:
        target_path = self._capture_targets.get(tick)
        if target_path is None or tick in self._captured_paths:
            return
        image = self._latest_image_for_capture()
        if image is None:
            self._notes.append(f"No {self.config.capture_sensor} frame was available at tick {tick} for capture.")
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        image.save_to_disk(str(target_path))
        self._captured_paths[tick] = str(target_path)

    def _latest_image_for_capture(self) -> Any | None:
        if self.config.capture_sensor == "front":
            return self._latest_front_image
        return self._latest_topdown_image

    def _image_to_array(self, image: Any) -> np.ndarray:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        return array.reshape((image.height, image.width, 4))[:, :, :3]

    def _draw_frame(self, frame: np.ndarray, x_offset: int) -> None:
        assert self._pygame is not None and self._screen is not None
        surface = self._pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
        self._screen.blit(surface, (x_offset, 0))

    def _draw_overlay(self, telemetry: dict[str, Any]) -> None:
        if self._font is None or self._screen is None or self._pygame is None:
            return
        route_progress = telemetry.get("agent_step", {}).get("route_progress", {})
        lines = [
            f"Tick: {telemetry.get('tick')}",
            f"Speed: {float(telemetry.get('speed_kph', 0.0)):.2f} kph",
            f"Throttle: {float(telemetry.get('control', {}).get('throttle', 0.0)):.2f}",
            f"Brake: {float(telemetry.get('control', {}).get('brake', 0.0)):.2f}",
            f"Steer: {float(telemetry.get('control', {}).get('steer', 0.0)):.2f}",
            f"Goal Dist: {float(telemetry.get('distance_to_goal_m', 0.0)):.2f} m",
            f"Cross-track: {float(route_progress.get('cross_track_error_m', 0.0) or 0.0):.2f} m",
            f"Heading Err: {float(route_progress.get('heading_error_deg', 0.0) or 0.0):.2f} deg",
        ]
        y = 10
        for line in lines:
            text = self._font.render(line, True, (255, 255, 255))
            bg = self._pygame.Surface((text.get_width() + 8, text.get_height() + 4))
            bg.set_alpha(150)
            bg.fill((0, 0, 0))
            self._screen.blit(bg, (10, y))
            self._screen.blit(text, (14, y + 2))
            y += text.get_height() + 8

    def _destroy(self) -> None:
        for sensor in [self._front_camera, self._topdown_camera]:
            if sensor is None:
                continue
            try:
                sensor.stop()
            except Exception:
                pass
            try:
                sensor.destroy()
            except Exception:
                pass
        self._front_camera = None
        self._topdown_camera = None
        if self._pygame is not None:
            self._pygame.quit()
        self._screen = None
        self._font = None
        self._pygame = None