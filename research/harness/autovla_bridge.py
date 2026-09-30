"""CARLA adapter for AutoVLA (UCLA), a vision-language-action driving model.

Mirrors the AutoVLA inference interface used by the original navsim agent:
three front-facing cameras sampled at 2 Hz (4-frame history per camera),
ego velocity/acceleration, and a route-derived driving command. The model
returns an ego-frame trajectory which is tracked with pure-pursuit steering
and a speed controller. The model is exposed through the harness agent
boundary (set_destination / run_step / done).
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
import tempfile
from collections import deque
from typing import Any

import numpy as np


DEFAULT_REPO = pathlib.Path("/mnt/DevDrive/development/MRES/research/models/AutoVLA")
DEFAULT_CHECKPOINT = DEFAULT_REPO / "checkpoints" / "AutoVLA-hf"
CODEBOOK = DEFAULT_REPO / "codebook_cache" / "agent_vocab.pkl"

CAMERA_SPECS = (
    ("front_camera", 2.2, 0.0, 0.0),
    ("front_left_camera", 2.0, -0.9, -55.0),
    ("front_right_camera", 2.0, 0.9, 55.0),
)

# model coordinate frame: x forward, y left. CARLA steer positive = right.
MAX_STEER_RAD = 0.65
WHEELBASE_M = 2.9
INFERENCE_INTERVAL_TICKS = 20  # 2 s at 10 Hz
CAPTURE_INTERVAL_TICKS = 5     # 2 Hz frame sampling
TARGET_SPEED_MPS = 5.5


class AutoVlaAdapter:
    def __init__(
        self,
        world: Any,
        ego_vehicle: Any,
        client: Any,
        repo_path: pathlib.Path | None = None,
        checkpoint_dir: pathlib.Path | None = None,
        device: str = "cuda",
        target_speed_mps: float = TARGET_SPEED_MPS,
    ) -> None:
        self._world = world
        self._ego = ego_vehicle
        self._client = client
        self._device = device
        self._target_speed = target_speed_mps
        repo = pathlib.Path(repo_path) if repo_path else DEFAULT_REPO
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        import torch  # noqa: E402
        from models.autovla import AutoVLA  # noqa: E402

        self._torch = torch
        config = {
            "model": {
                "use_cot": False,
                "pretrained_model_path": str(checkpoint_dir or DEFAULT_CHECKPOINT),
                "train_vision_backbone": False,
                "train_lm_backbone": True,
                "codebook_cache_path": str(CODEBOOK) if pathlib.Path(CODEBOOK).is_absolute() else str(repo / "codebook_cache" / "agent_vocab.pkl"),
                "trajectory": {"num_poses": 10, "interval_length": 0.5, "time_horizon": 5.0},
                "tokens": {"action_start_id": 151665, "ignore_index": -100, "assistant_id": [151644, 77091]},
                "video": {"min_pixels": 109760, "max_pixels": 109760},
            },
            "inference": {"sample": {"max_length": 2048, "temperature": 0.01, "top_k": 0, "top_p": 1.0}},
        }
        self._model = AutoVLA(config, inference=True, device=device)
        self._model.eval()

        self._frame_dir = pathlib.Path(tempfile.mkdtemp(prefix="autovla-frames-"))
        self._cameras: dict[str, Any] = {}
        self._frames: dict[str, deque[str]] = {name: deque(maxlen=4) for name, *_ in CAMERA_SPECS}
        self._tick = 0
        self._last_poses: Any = None
        self._last_inference_tick = -10_000
        self._route: list[Any] = []
        self._route_index = 0
        self._last_speed = 0.0
        self._last_command = "forward"
        self.last_step_info: dict[str, Any] = {}
        self._setup_cameras()

    # ------------------------------------------------------------------ sensors
    def _setup_cameras(self) -> None:
        import carla

        bp_lib = self._world.get_blueprint_library()
        for name, x, y, yaw in CAMERA_SPECS:
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", "800")
            bp.set_attribute("image_size_y", "450")
            bp.set_attribute("fov", "70")
            transform = carla.Transform(
                carla.Location(x=x, y=y, z=1.5),
                carla.Rotation(yaw=yaw),
            )
            cam = self._world.spawn_actor(bp, transform, attach_to=self._ego)
            cam.listen(self._make_callback(name))
            self._cameras[name] = cam

    def _make_callback(self, name: str):
        def _cb(image: Any) -> None:
            if self._tick % CAPTURE_INTERVAL_TICKS != 0:
                return
            cam_dir = self._frame_dir / name
            cam_dir.mkdir(parents=True, exist_ok=True)
            path = cam_dir / f"{self._tick:05d}.png"
            image.save_to_disk(str(path))
            self._frames[name].append(str(path))
        return _cb

    # ------------------------------------------------------------------ routing
    def set_destination(self, goal_location: Any) -> None:
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        carla = __import__("carla")
        world_map = self._world.get_map()
        planner = GlobalRoutePlanner(world_map, 2.0)
        try:
            planner.setup()
        except Exception:
            pass
        start = self._ego.get_location()
        trace = planner.trace_route(start, goal_location)
        self._route = [wp for wp, _ in trace]
        self._route_index = 0

    def _advance_route(self) -> None:
        loc = self._ego.get_location()
        n = len(self._route)
        if n == 0:
            return
        best = self._route_index
        best_d = float("inf")
        for i in range(self._route_index, min(n, self._route_index + 30)):
            d = self._route[i].transform.location.distance(loc)
            if d < best_d:
                best_d = d
                best = i
        self._route_index = best

    def _driving_command(self) -> str:
        if not self._route:
            return "forward"
        self._advance_route()
        idx = min(self._route_index + 8, len(self._route) - 1)  # ~16 m ahead at 2 m spacing
        target = self._route[idx].transform.location
        ego_tf = self._ego.get_transform()
        yaw = math.radians(ego_tf.rotation.yaw)
        dx = target.x - ego_tf.location.x
        dy = target.y - ego_tf.location.y
        # CARLA y is right; heading delta positive = target to the left
        heading_delta = math.degrees(math.atan2(-(dx * math.sin(yaw) - dy * math.cos(yaw)), dx * math.cos(yaw) + dy * math.sin(yaw)))
        if heading_delta > 30:
            return "right"
        if heading_delta < -30:
            return "left"
        return "forward"

    # ------------------------------------------------------------------ control
    def _control_from_poses(self, poses: Any) -> Any:
        carla = __import__("carla")
        import numpy as _np

        pts = poses.detach().float().cpu().numpy() if hasattr(poses, "detach") else _np.asarray(poses)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return carla.VehicleControl(throttle=0.0, brake=0.3)
        lookahead_idx = min(2, pts.shape[0] - 1)  # ~1.0 s ahead
        x, y = float(pts[lookahead_idx, 0]), float(pts[lookahead_idx, 1])
        dist = math.hypot(x, y)
        if dist < 0.5:
            steer = 0.0
        else:
            alpha = math.atan2(y, x)  # model frame: +y is left
            delta = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), dist)
            steer = max(-1.0, min(1.0, -delta / MAX_STEER_RAD))  # CARLA steer positive = right
        v = self._ego.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        target_speed = self._target_speed
        err = target_speed - speed
        throttle = max(0.0, min(0.8, 0.35 + 0.12 * err))
        brake = max(0.0, min(0.6, 0.25 * (-err))) if err < -0.5 else 0.0
        if brake > 0:
            throttle = 0.0
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

    # ------------------------------------------------------------------ harness API
    def run_step(self) -> Any:
        self._tick += 1
        carla = __import__("carla")
        if self._route:
            self._advance_route()
        if self._tick - self._last_inference_tick >= INFERENCE_INTERVAL_TICKS:
            if all(len(self._frames[name]) >= 4 for name, *_ in CAMERA_SPECS):
                v = self._ego.get_velocity()
                speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
                accel = (speed - self._last_speed) / 0.1  # 10 Hz
                self._last_speed = speed
                command = self._driving_command()
                features = {
                    "images": {
                        "front_camera": list(self._frames["front_camera"]),
                        "front_left_camera": list(self._frames["front_left_camera"]),
                        "front_right_camera": list(self._frames["front_right_camera"]),
                    },
                    "vehicle_velocity": [v.x, v.y],
                    "vehicle_acceleration": [0.0, accel],
                    "driving_command": command,
                    "dataset_name": "nuscenes",
                    "sensor_data_path": None,
                }
                try:
                    import time as _time
                    _t0 = _time.time()
                    poses, cot = self._model.predict(features)
                    if not getattr(self, "_logged_first", False):
                        print(f"[autovla] first inference ok: {_time.time()-_t0:.2f}s poses={None if poses is None else tuple(poses.shape)}", flush=True)
                        self._logged_first = True
                    self._last_poses = poses
                    self._last_inference_tick = self._tick
                    self._last_command = command
                    self.last_step_info = {"command": command, "cot": str(cot)[:200]}
                except Exception as exc:
                    import traceback as _tb
                    if not getattr(self, "_logged_error", False):
                        print(f"[autovla] inference error: {exc!r}", flush=True)
                        _tb.print_exc()
                        self._logged_error = True
                    self.last_step_info = {"error": repr(exc)[:300]}
                    return carla.VehicleControl(throttle=0.0, brake=0.5)
        if not getattr(self, "_logged_frames", False) and self._tick > 25:
            print("[autovla] frames:", {k: len(v) for k, v in self._frames.items()}, flush=True)
            self._logged_frames = True
        if self._last_poses is not None:
            return self._control_from_poses(self._last_poses)
        return carla.VehicleControl(throttle=0.0, brake=0.3)

    # ------------------------------------------------------- instrumentation hooks
    @property
    def torch_model(self):
        return self._model.vlm

    def inspect_loaded_model(self, max_modules: int = 24) -> dict:
        model = self.torch_model
        leaves = [(n, m) for n, m in model.named_modules() if n and len(list(m.children())) == 0]
        return {
            "agent_name": "autovla",
            "status": "ok",
            "torch_model_available": True,
            "candidate_layers": [{"name": n, "module_type": type(m).__name__} for n, m in leaves[:max_modules]],
        }

    def _resolve_target_module(self, layer_name: str | None = None):
        model = self.torch_model
        if layer_name:
            for n, m in model.named_modules():
                if n == layer_name:
                    return n, m
        lm = model.model.language_model
        idx = len(lm.layers) - 1
        return f"model.language_model.layers.{idx}", lm.layers[-1]

    def register_activation_hook(self, callback, layer_name: str | None = None) -> dict:
        name, module = self._resolve_target_module(layer_name)
        handle = module.register_forward_hook(callback)
        if not hasattr(self, "_hook_handles"):
            self._hook_handles = []
        self._hook_handles.append(handle)
        return {"handle": handle, "layer_name": name, "module_type": type(module).__name__,
                "model_summary": self.inspect_loaded_model()}

    def clear_activation_hooks(self) -> None:
        for h in list(getattr(self, "_hook_handles", [])):
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles = []

    def done(self) -> bool:
        return False

    def destroy(self) -> None:
        for cam in self._cameras.values():
            try:
                cam.stop()
                cam.destroy()
            except Exception:
                pass
        self._cameras.clear()
