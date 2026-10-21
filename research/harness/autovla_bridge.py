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
import pathlib
import sys
from collections import OrderedDict
from typing import Any

import numpy as np

from research.harness.autovla_inference import AutoVlaInference, validate_trajectory
from research.harness.autovla_sensors import CameraHistory
from research.harness.route_completion import route_goal_reached


DEFAULT_REPO = pathlib.Path(__file__).resolve().parents[1] / "models" / "AutoVLA"
DEFAULT_CHECKPOINT = DEFAULT_REPO / "checkpoints" / "AutoVLA-hf"
CODEBOOK = DEFAULT_REPO / "codebook_cache" / "agent_vocab.pkl"

# The released nuPlan feature builder selects CAM_L1/CAM_R1 (side views),
# not CAM_L0/CAM_R0. See AutoVLAAgentFeatureBuilder.compute_features.
CAMERA_SPECS = (
    ("front_camera", 2.2, 0.0, 0.0),
    ("front_left_camera", 2.0, -0.9, -90.0),
    ("front_right_camera", 2.0, 0.9, 90.0),
)

# model coordinate frame: x forward, y left. CARLA steer positive = right.
MAX_STEER_RAD = 0.65
WHEELBASE_M = 2.9
INFERENCE_INTERVAL_SECONDS = 0.5
CAPTURE_INTERVAL_SECONDS = 0.5
TARGET_SPEED_MPS = 5.5


class AutoVlaAdapter:
    def __init__(
        self,
        world: Any,
        ego_vehicle: Any,
        client: Any,
        repo_path: pathlib.Path | None = None,
        checkpoint_dir: pathlib.Path | None = None,
        device: str | None = None,
        target_speed_mps: float = TARGET_SPEED_MPS,
        decoding_mode: str = "adaptive",
    ) -> None:
        self._world = world
        self._ego = ego_vehicle
        self._configure_vehicle_geometry()
        self._client = client
        self._target_speed = target_speed_mps  # upper speed limit, never a forced cruise speed
        self._fixed_delta_seconds = float(world.get_settings().fixed_delta_seconds or 0.1)
        repo = pathlib.Path(repo_path) if repo_path else DEFAULT_REPO
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        import torch  # noqa: E402
        from models.autovla import AutoVLA  # noqa: E402

        self._torch = torch
        if device is None:
            if torch.cuda.is_available():
                index = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
                device = f"cuda:{index}"
            else:
                device = "cpu"
        self._device = device
        config = {
            "model": {
                "use_cot": True,
                "pretrained_model_path": str(checkpoint_dir or DEFAULT_CHECKPOINT),
                "train_vision_backbone": False,
                "train_lm_backbone": True,
                "codebook_cache_path": str(repo / "codebook_cache" / "agent_vocab.pkl"),
                "trajectory": {"num_poses": 10, "interval_length": 0.5, "time_horizon": 5.0},
                "tokens": {"action_start_id": 151665, "ignore_index": -100, "assistant_id": [151644, 77091]},
                "video": {"min_pixels": 109760, "max_pixels": 109760},
            },
            "inference": {"sample": {"max_length": 2048, "temperature": 0.01, "top_k": 0, "top_p": 1.0}},
        }
        self._model = AutoVLA(config, inference=True, device=device)
        self._model.eval()

        self._predictor = AutoVlaInference(self._model, decoding_mode=decoding_mode)
        self._capture_stride = max(1, round(CAPTURE_INTERVAL_SECONDS / self._fixed_delta_seconds))
        if abs(self._capture_stride * self._fixed_delta_seconds - CAPTURE_INTERVAL_SECONDS) > 1e-9:
            raise ValueError("AutoVLA timestep must divide its 0.5-second camera interval")
        self._inference_interval_ticks = self._capture_stride
        self._history = CameraHistory([name for name, *_ in CAMERA_SPECS], self._capture_stride)
        self._state_history: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._cameras: dict[str, Any] = {}
        self._tick = 0
        self._last_poses: Any = None
        self._trajectory_origin: tuple[float, float, float] | None = None
        self._last_inference_tick = -10_000
        self._last_attempt_tick = -10_000
        self._last_inference_frame = -1
        self._trajectory_timestamp: float | None = None
        self._inference_count = 0
        self._goal_location: Any = None
        self._route: list[Any] = []
        self._route_options: list[Any] = []
        self._route_index = 0
        self._last_speed = 0.0
        self._speed_error_integral = 0.0
        self._last_command = "forward"
        self.last_step_info: dict[str, Any] = {}
        self._setup_cameras()

    # ------------------------------------------------------------------ sensors
    def _configure_vehicle_geometry(self) -> None:
        physics = self._ego.get_physics_control()
        wheels = physics.wheels
        if len(wheels) != 4:
            raise ValueError("AutoVLA trajectory tracking requires a four-wheel vehicle")
        self._max_steer_rad = math.radians(max(wheel.max_steer_angle for wheel in wheels[:2]))
        # CARLA reports wheel positions in world-space centimetres, so axle
        # separation must be rotation invariant, rather than an x difference.
        front = np.mean([[wheel.position.x, wheel.position.y] for wheel in wheels[:2]], axis=0)
        rear = np.mean([[wheel.position.x, wheel.position.y] for wheel in wheels[2:]], axis=0)
        self._wheelbase_m = float(np.linalg.norm(front - rear) / 100.0)
        if self._max_steer_rad <= 0 or self._wheelbase_m <= 0:
            raise ValueError("Invalid vehicle steering geometry")
        self._steering_curve = sorted((point.x, point.y) for point in physics.steering_curve)

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
            if image.frame % self._capture_stride:
                return
            from PIL import Image
            pixels = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
            rgb = Image.fromarray(pixels[:, :, 2::-1].copy())
            self._history.add(name, image.frame, rgb)
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
        self._goal_location = goal_location
        start = self._ego.get_location()
        trace = planner.trace_route(start, goal_location)
        self._route = [wp for wp, _ in trace]
        self._route_options = [option for _, option in trace]
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
            return "keep forward"
        self._advance_route()
        # Preserve the planner's actual junction command. Inferring a turn
        # from a short local bearing confused lane drift with route intent
        # and announced real turns too late.
        velocity = self._ego.get_velocity()
        preview_distance = max(16.0, math.hypot(velocity.x, velocity.y) * 4.0)
        distance = 0.0
        for index in range(self._route_index, len(self._route)):
            if index > self._route_index:
                distance += self._route[index].transform.location.distance(self._route[index - 1].transform.location)
            if distance > preview_distance:
                break
            if index < len(self._route_options):
                name = self._route_options[index].name
                if name in {"LEFT", "CHANGELANELEFT"}:
                    return "turn left"
                if name in {"RIGHT", "CHANGELANERIGHT"}:
                    return "turn right"
        if self._route_options:
            return "keep forward"
        idx = min(self._route_index + 8, len(self._route) - 1)  # ~16 m ahead at 2 m spacing
        target = self._route[idx].transform.location
        ego_tf = self._ego.get_transform()
        yaw = math.radians(ego_tf.rotation.yaw)
        dx = target.x - ego_tf.location.x
        dy = target.y - ego_tf.location.y
        # CARLA y is right; a positive heading delta is a right turn.
        heading_delta = math.degrees(math.atan2(-(dx * math.sin(yaw) - dy * math.cos(yaw)), dx * math.cos(yaw) + dy * math.sin(yaw)))
        if heading_delta > 30:
            return "turn right"
        if heading_delta < -30:
            return "turn left"
        return "keep forward"

    # ------------------------------------------------------------------ control
    def _control_from_poses(self, poses: Any) -> Any:
        carla = __import__("carla")
        import numpy as _np

        pts = poses.detach().float().cpu().numpy() if hasattr(poses, "detach") else _np.asarray(poses)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2 or not _np.isfinite(pts[:, :2]).all():
            return carla.VehicleControl(throttle=0.0, brake=0.3)
        if getattr(self, "_trajectory_timestamp", None) is not None:
            elapsed = max(0.0, self._world.get_snapshot().timestamp.elapsed_seconds - self._trajectory_timestamp)
        else:
            elapsed = max(0.0, (self._tick - self._last_inference_tick) * self._fixed_delta_seconds)
        horizon = len(pts) * 0.5
        if elapsed >= horizon or self._trajectory_origin is None:
            return carla.VehicleControl(throttle=0.0, brake=0.5)
        # Poses are in the inference-time ego frame (+y left), sampled every .5 s.
        # Advance in time and transform the target through that original frame.
        times = _np.arange(len(pts) + 1) * 0.5
        xy = _np.vstack((_np.zeros((1, 2)), pts[:, :2]))
        # Speed planning uses two seconds to include the launch phase of tokenized plans.
        # One-second previews repeatedly saw only the initial stop tokens at
        # 2 Hz replanning, leaving a stationary car unable to start moving.
        target_time = min(elapsed + 2.0, horizon)
        # Track near-term lateral intent separately from launch/speed planning.
        steering_time = min(elapsed + 1.0, horizon)
        target = _np.array([_np.interp(steering_time, times, xy[:, i]) for i in (0, 1)])
        origin_x, origin_y, origin_yaw = self._trajectory_origin
        c, s = math.cos(origin_yaw), math.sin(origin_yaw)
        world_x = origin_x + target[0] * c + target[1] * s
        world_y = origin_y + target[0] * s - target[1] * c
        current = self._ego.get_transform()
        yaw = math.radians(current.rotation.yaw)
        dx, dy = world_x - current.location.x, world_y - current.location.y
        x = dx * math.cos(yaw) + dy * math.sin(yaw)
        y = dx * math.sin(yaw) - dy * math.cos(yaw)
        if x < -0.5:
            return carla.VehicleControl(throttle=0.0, brake=0.5)
        dist = math.hypot(x, y)
        v = self._ego.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        if dist < 0.5:
            steer = 0.0
        else:
            alpha = math.atan2(y, x)  # model frame: +y is left
            delta = math.atan2(2.0 * getattr(self, "_wheelbase_m", WHEELBASE_M) * math.sin(alpha), dist)
            curve = getattr(self, "_steering_curve", [])
            scale = float(np.interp(speed * 3.6, [point[0] for point in curve], [point[1] for point in curve])) if curve else 1.0
            max_angle = getattr(self, "_max_steer_rad", MAX_STEER_RAD) * max(scale, 0.01)
            steer = max(-1.0, min(1.0, -delta / max_angle))  # CARLA steer positive = right
        distances = _np.concatenate(([0.0], _np.cumsum(_np.linalg.norm(_np.diff(xy, axis=0), axis=1))))
        planned_distance = _np.interp(target_time, times, distances) - _np.interp(elapsed, times, distances)
        target_speed = min(self._target_speed, float(planned_distance / (target_time - elapsed)))
        curvature = abs(2.0 * y / max(dist * dist, 0.25))
        if curvature > 1e-6:
            target_speed = min(target_speed, math.sqrt(2.0 / curvature))
        if target_speed < 0.1:
            self._speed_error_integral = 0.0
            return carla.VehicleControl(throttle=0.0, steer=steer, brake=max(0.3, min(1.0, speed * 0.25)))
        err = target_speed - speed
        self._speed_error_integral = max(0.0, min(2.0,
            getattr(self, "_speed_error_integral", 0.0) + err * self._fixed_delta_seconds))
        throttle = max(0.0, min(0.8, 0.35 * err + 0.12 * self._speed_error_integral))
        brake = max(0.0, min(0.6, 0.25 * (-err))) if err < -0.5 else 0.0
        if brake > 0:
            throttle = 0.0
            self._speed_error_integral = 0.0
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

    # ------------------------------------------------------------------ harness API
    def run_step(self) -> Any:
        self._tick += 1
        carla = __import__("carla")
        snapshot = self._world.get_snapshot()
        transform = self._ego.get_transform()
        velocity = self._ego.get_velocity()
        acceleration = self._ego.get_acceleration()
        self._state_history[snapshot.frame] = {
            "timestamp": float(snapshot.timestamp.elapsed_seconds),
            "velocity": [velocity.x, velocity.y],
            "acceleration": [acceleration.x, acceleration.y],
            "origin": (transform.location.x, transform.location.y, math.radians(transform.rotation.yaw)),
        }
        while len(self._state_history) > 64:
            self._state_history.popitem(last=False)
        if self._route:
            self._advance_route()
        history = self._history.snapshot(snapshot.frame)
        if history is None:
            self.last_step_info = {"mode": "sensor-wait", "sensor_ready": False}
            return carla.VehicleControl(throttle=0.0, brake=0.5)
        images, frame_ids = history
        state = self._state_history.get(frame_ids[-1])
        if state is None:
            self.last_step_info = {"mode": "sensor-wait", "sensor_ready": False}
            return carla.VehicleControl(throttle=0.0, brake=0.5)
        if (self._tick - self._last_attempt_tick >= self._inference_interval_ticks
                and frame_ids[-1] != self._last_inference_frame):
            self._last_attempt_tick = self._tick
            command = self._driving_command()
            features = {
                "images": images,
                "vehicle_velocity": state["velocity"],
                "vehicle_acceleration": state["acceleration"],
                "driving_command": command,
                "dataset_name": "nuscenes", "sensor_data_path": None,
            }
            try:
                poses, cot = self._predictor.predict(features)
                self._last_poses = validate_trajectory(poses)
                self._trajectory_origin = state["origin"]
                self._trajectory_timestamp = state["timestamp"]
                self._last_inference_tick = self._tick
                self._last_inference_frame = frame_ids[-1]
                self._last_command = command
                self._inference_count += 1
                if self._inference_count == 1:
                    print(f"[autovla] ready on {self._device}: "
                          f"{self._predictor.last_metrics['inference_seconds']:.3f}s, "
                          "10 action tokens, synchronized 2 Hz cameras", flush=True)
                self.last_step_info = {
                    "mode": "trajectory", "sensor_ready": True, "command": command,
                    "camera_frame_ids": frame_ids, "inference_count": self._inference_count,
                    "trajectory": self._last_poses.tolist(),
                    **self._predictor.last_metrics,
                }
            except Exception as exc:
                self._last_poses = None
                self._trajectory_origin = None
                self._trajectory_timestamp = None
                self.last_step_info = {"mode": "inference-error", "error": repr(exc)[:300]}
                self._speed_error_integral = 0.0
                return carla.VehicleControl(throttle=0.0, brake=0.5)
        if self._last_poses is not None:
            return self._control_from_poses(self._last_poses)
        return carla.VehicleControl(throttle=0.0, brake=0.5)

    def get_debug_state(self) -> dict[str, Any]:
        info = dict(self.last_step_info)
        if self._route:
            self._advance_route()
            location = self._ego.get_location()
            index = self._route_index
            start = self._route[index].transform.location
            error = float(location.distance(start))
            for segment in range(max(0, index - 2), min(len(self._route) - 1, index + 3)):
                a, b = [self._route[i].transform.location for i in (segment, segment + 1)]
                dx, dy = b.x - a.x, b.y - a.y
                length_sq = dx * dx + dy * dy
                projection = max(0.0, min(1.0, ((location.x - a.x) * dx + (location.y - a.y) * dy) / length_sq)) if length_sq else 0.0
                error = min(error, math.hypot(location.x - a.x - projection * dx, location.y - a.y - projection * dy))
            heading = self._route[index].transform.rotation.yaw
            info["route_progress"] = {
                "next_waypoint_index": index,
                "remaining_waypoints": len(self._route) - index,
                "distance_to_next_waypoint_m": float(location.distance(start)),
                "cross_track_error_m": error,
                "heading_error_deg": (self._ego.get_transform().rotation.yaw - heading + 180) % 360 - 180,
            }
        return info

    def get_route_preview(self) -> list[dict[str, Any]]:
        return self.get_route_waypoints()[:12]

    def get_route_waypoints(self) -> list[dict[str, Any]]:
        return [{"index": index, "location": {"x": wp.transform.location.x,
                 "y": wp.transform.location.y, "z": wp.transform.location.z}}
                for index, wp in enumerate(self._route)]

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
        self._advance_route()
        return route_goal_reached(self._ego, self._goal_location,
                                  [wp.transform.location for wp in self._route], self._route_index)

    def destroy(self) -> None:
        self.clear_activation_hooks()
        for cam in self._cameras.values():
            try:
                cam.stop()
                cam.destroy()
            except Exception:
                pass
        self._cameras.clear()
        self._history.clear()
        self._state_history.clear()
