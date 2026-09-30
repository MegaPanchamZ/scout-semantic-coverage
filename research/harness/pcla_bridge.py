from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
import typing
from typing import Any
from xml.dom import minidom


def _apply_pcla_import_aliases() -> None:
    """Work around vendored PCLA agent code that is not importable as-is.

    - ``transfuserv6`` annotates one function with ``torch.Union[Tensor, None]``
      (a typo for ``typing.Union``); ``beartype`` resolves the hint at import
      time and aborts. Exposing ``typing.Union`` on the torch module keeps the
      agent importable without editing vendored model code.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency of the harness env
        return
    if not hasattr(torch, "Union"):
        torch.Union = typing.Union  # type: ignore[attr-defined]


def ensure_pcla_repo_path(repo_path: Path | None) -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    resolved_repo = (repo_path or (workspace_root / "research" / "models" / "PCLA")).resolve()
    repo_str = str(resolved_repo)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)
    _apply_pcla_import_aliases()
    return resolved_repo


class PclaAdapter:
    def __init__(self, world: Any, ego_vehicle: Any, client: Any, repo_path: Path | None, agent_name: str) -> None:
        if not agent_name:
            raise ValueError("PCLA agents require a configured agent name.")

        self._world = world
        self._ego_vehicle = ego_vehicle
        self._client = client
        self._repo_path = ensure_pcla_repo_path(repo_path)
        self._agent_name = agent_name
        self._goal_location = None
        self._route_waypoints: list[Any] = []
        self._route_locations: list[Any] = []
        self._route_progress_index = 0
        self._route_preview_limit = 12
        self._route_waypoint_limit = 600
        self._route_xml_path: Path | None = None
        self._pcla = None
        self.torch_model: Any | None = None
        self.model: Any | None = None
        self._registered_hook_handles: list[Any] = []
        self._instrumentation_summary: dict[str, Any] = {}
        self.last_step_info: dict[str, Any] = {}

    def set_destination(self, goal_location: Any) -> None:
        self._goal_location = goal_location
        if self._pcla is not None:
            self.destroy()

        location_to_waypoint_mod = importlib.import_module("pcla_functions.location_to_waypoint")
        route_maker_mod = importlib.import_module("pcla_functions.route_maker")
        pcla_mod = importlib.import_module("PCLA")

        workspace_root = Path(__file__).resolve().parents[2]
        route_dir = (workspace_root / "research" / "logs" / "pcla_routes").resolve()
        route_dir.mkdir(parents=True, exist_ok=True)
        route_name = f"pcla-{self._agent_name}-{self._ego_vehicle.id}.xml"
        self._route_xml_path = (route_dir / route_name).resolve()

        self._route_waypoints = location_to_waypoint_mod.location_to_waypoint(
            self._client,
            self._ego_vehicle.get_location(),
            goal_location,
        )
        if len(self._route_waypoints) <= 1:
            self._route_waypoints = self._build_minimal_route_waypoints(goal_location)
        self._route_locations = [waypoint.transform.location for waypoint in self._route_waypoints]
        self._route_progress_index = 0
        route_maker_mod.route_maker(self._route_waypoints, str(self._route_xml_path))
        if not self._route_xml_path.exists():
            self._write_route_xml(self._route_waypoints, self._route_xml_path)
        self._pcla = pcla_mod.PCLA(self._agent_name, self._ego_vehicle, str(self._route_xml_path), self._client)
        self._refresh_instrumentation_targets()

    def run_step(self) -> Any:
        if self._pcla is None:
            raise RuntimeError("PCLA agent has not been initialized with a route yet.")
        control = self._pcla.get_action()
        snapshot = self._world.get_snapshot()
        timestamp = float(snapshot.timestamp.elapsed_seconds) if snapshot is not None else None
        self.last_step_info = {
            "sensor_ready": True,
            "mode": "pcla-get-action",
            "timestamp": timestamp,
        }
        return control

    def done(self) -> bool:
        if self._goal_location is None:
            return False
        return self._ego_vehicle.get_location().distance(self._goal_location) < 5.0

    def destroy(self) -> None:
        self.clear_activation_hooks()
        if self._pcla is not None:
            preserved_vehicle = self._pcla.vehicle
            try:
                self._pcla.vehicle = None
                self._pcla.cleanup()
            finally:
                self._pcla.vehicle = preserved_vehicle
                self._pcla = None
        if self._route_xml_path is not None:
            try:
                self._route_xml_path.unlink(missing_ok=True)
            except TypeError:
                if self._route_xml_path.exists():
                    self._route_xml_path.unlink()
            self._route_xml_path = None
        self.torch_model = None
        self.model = None
        self._instrumentation_summary = {}

    def inspect_loaded_model(self, max_modules: int = 24) -> dict[str, Any]:
        if self._pcla is None:
            return {
                "agent_name": self._agent_name,
                "status": "not-initialized",
                "torch_model_available": False,
                "candidate_layers": [],
            }

        self._refresh_instrumentation_targets()
        if self.torch_model is None:
            summary = {
                "agent_name": self._agent_name,
                "status": "no-torch-model",
                "torch_model_available": False,
                "agent_instance_type": type(getattr(self._pcla, "agent_instance", None)).__name__,
                "candidate_layers": [],
            }
            self._instrumentation_summary = summary
            return summary

        named_modules = list(self.torch_model.named_modules())
        parameter_count = sum(int(parameter.numel()) for parameter in self.torch_model.parameters())
        leaf_modules = [
            (name, module)
            for name, module in named_modules
            if name and len(list(module.children())) == 0
        ]
        candidate_layers = [
            {
                "name": name,
                "type": type(module).__name__,
                "parameter_count": sum(int(parameter.numel()) for parameter in module.parameters(recurse=False)),
            }
            for name, module in leaf_modules[-max_modules:]
        ]
        preferred_name, preferred_module = self._resolve_target_module(None)
        summary = {
            "agent_name": self._agent_name,
            "status": "ready",
            "torch_model_available": True,
            "agent_instance_type": type(self._pcla.agent_instance).__name__,
            "model_type": type(self.torch_model).__name__,
            "parameter_count": parameter_count,
            "named_module_count": len(named_modules),
            "preferred_hook_layer": preferred_name,
            "preferred_hook_layer_type": type(preferred_module).__name__ if preferred_module is not None else None,
            "candidate_layers": candidate_layers,
        }
        self._instrumentation_summary = summary
        return summary

    def register_activation_hook(self, callback: Any, layer_name: str | None = None) -> dict[str, Any]:
        self._refresh_instrumentation_targets()
        module_name, module = self._resolve_target_module(layer_name)
        if module is None:
            raise RuntimeError("Unable to resolve a PyTorch layer for activation hooks.")
        handle = module.register_forward_hook(callback)
        self._registered_hook_handles.append(handle)
        return {
            "handle": handle,
            "layer_name": module_name,
            "module_type": type(module).__name__,
            "model_summary": self.inspect_loaded_model(),
        }

    def clear_activation_hooks(self) -> None:
        while self._registered_hook_handles:
            handle = self._registered_hook_handles.pop()
            try:
                handle.remove()
            except Exception:
                pass

    def _build_minimal_route_waypoints(self, goal_location: Any) -> list[Any]:
        carla_map = self._world.get_map()
        start_waypoint = carla_map.get_waypoint(self._ego_vehicle.get_location(), project_to_road=True)
        goal_waypoint = carla_map.get_waypoint(goal_location, project_to_road=True)
        waypoints = [waypoint for waypoint in (start_waypoint, goal_waypoint) if waypoint is not None]
        if len(waypoints) == 2 and self._waypoint_distance(waypoints[0], waypoints[1]) < 0.5:
            return [waypoints[0], waypoints[0].next(2.0)[0] if waypoints[0].next(2.0) else waypoints[1]]
        return waypoints

    def _write_route_xml(self, waypoints: list[Any], save_path: Path) -> None:
        if not waypoints:
            raise RuntimeError("Cannot write a PCLA route XML without at least one waypoint.")

        document = minidom.Document()
        root = document.createElement("route")
        root.setAttribute("id", "_")
        root.setAttribute("town", "_")
        document.appendChild(root)

        for waypoint in waypoints:
            transform = waypoint.transform
            child = document.createElement("waypoint")
            child.setAttribute("pitch", str(transform.rotation.pitch))
            child.setAttribute("roll", str(transform.rotation.roll))
            child.setAttribute("x", str(transform.location.x))
            child.setAttribute("y", str(transform.location.y))
            child.setAttribute("yaw", str(transform.rotation.yaw))
            child.setAttribute("z", str(transform.location.z))
            root.appendChild(child)

        save_path.write_text(document.toprettyxml(indent="\t"), encoding="utf-8")

    def _waypoint_distance(self, first: Any, second: Any) -> float:
        return float(first.transform.location.distance(second.transform.location))

    def get_debug_state(self) -> dict[str, Any]:
        debug_state = dict(self.last_step_info)
        debug_state["route_progress"] = self._route_debug_snapshot()
        if self._instrumentation_summary:
            debug_state["instrumentation"] = {
                "preferred_hook_layer": self._instrumentation_summary.get("preferred_hook_layer"),
                "torch_model_available": self._instrumentation_summary.get("torch_model_available"),
            }
        return debug_state

    def _refresh_instrumentation_targets(self) -> None:
        self.torch_model = self._resolve_torch_model()
        self.model = self.torch_model

    def _resolve_torch_model(self) -> Any | None:
        if self._pcla is None:
            return None
        agent_instance = getattr(self._pcla, "agent_instance", None)
        if agent_instance is None:
            return None

        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError:
            return None

        direct_candidates = [
            getattr(agent_instance, "net", None),
            getattr(agent_instance, "model", None),
            getattr(agent_instance, "network", None),
            getattr(agent_instance, "policy", None),
        ]

        for inference_attr in ("closed_loop_inference", "open_loop_inference"):
            inference_obj = getattr(agent_instance, inference_attr, None)
            if inference_obj is None:
                continue
            direct_candidates.extend(
                [
                    getattr(inference_obj, "net", None),
                    getattr(inference_obj, "model", None),
                    getattr(inference_obj, "network", None),
                    getattr(inference_obj, "policy", None),
                ]
            )
            nested_ensemble = getattr(inference_obj, "nets", None)
            if isinstance(nested_ensemble, (list, tuple)) and nested_ensemble:
                direct_candidates.append(nested_ensemble[0])

        for candidate in direct_candidates:
            if isinstance(candidate, torch.nn.Module):
                return candidate

        ensemble = getattr(agent_instance, "nets", None)
        if isinstance(ensemble, (list, tuple)) and ensemble:
            first_model = ensemble[0]
            if isinstance(first_model, torch.nn.Module):
                return first_model

        if isinstance(agent_instance, torch.nn.Module):
            return agent_instance
        return None

    def _resolve_target_module(self, layer_name: str | None) -> tuple[str | None, Any | None]:
        if self.torch_model is None:
            return None, None

        modules = dict(self.torch_model.named_modules())
        if layer_name is not None:
            return layer_name, modules.get(layer_name)

        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError:
            torch = None

        named_modules = list(self.torch_model.named_modules())
        if torch is not None:
            for name, module in reversed(named_modules):
                if isinstance(module, torch.nn.Linear):
                    return name, module
        for name, module in reversed(named_modules):
            if name and len(list(module.children())) == 0:
                parameters = list(module.parameters(recurse=False))
                if parameters:
                    return name, module
        return None, None

    def get_route_preview(self) -> list[dict[str, Any]]:
        preview = []
        for index, location in enumerate(self._route_locations[: self._route_preview_limit]):
            preview.append(
                {
                    "index": index,
                    "location": self._location_to_dict(location),
                }
            )
        return preview

    def get_route_waypoints(self) -> list[dict[str, float]]:
        """The full planned route as compact ``{"x","y"}`` points.

        Capped at ``_route_waypoint_limit`` entries (600) so the persisted
        ``ego_route`` stays small; the offline coverage engine extends the
        observed ego track with it so crossings beyond the 2 s projection are
        still credited.
        """
        return [
            {"x": round(float(location.x), 3), "y": round(float(location.y), 3)}
            for location in self._route_locations[: self._route_waypoint_limit]
        ]

    def _route_debug_snapshot(self) -> dict[str, Any]:
        if not self._route_locations:
            return {
                "route_waypoint_count": 0,
                "next_waypoint_index": None,
                "distance_to_next_waypoint_m": None,
                "nearest_route_waypoint_index": None,
                "nearest_route_waypoint_distance_m": None,
                "cross_track_error_m": None,
                "ego_yaw_deg": None,
                "route_heading_deg": None,
                "heading_error_deg": None,
                "next_route_waypoint_location": None,
                "nearest_route_waypoint_location": None,
                "nearest_driving_waypoint_location": None,
                "nearest_driving_waypoint_yaw_deg": None,
                "remaining_waypoints": 0,
            }

        ego_transform = self._ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_yaw_deg = float(ego_transform.rotation.yaw)
        nearest_index, nearest_distance = self._nearest_route_waypoint(ego_location, self._route_progress_index)
        if nearest_index is not None and nearest_index > self._route_progress_index:
            self._route_progress_index = nearest_index
        while self._route_progress_index < len(self._route_locations):
            current_waypoint_distance = ego_location.distance(self._route_locations[self._route_progress_index])
            if current_waypoint_distance > 4.0:
                break
            self._route_progress_index += 1

        next_waypoint_index = self._route_progress_index if self._route_progress_index < len(self._route_locations) else None
        next_waypoint_distance = (
            float(ego_location.distance(self._route_locations[next_waypoint_index]))
            if next_waypoint_index is not None
            else None
        )
        route_heading_deg = self._route_heading_deg(next_waypoint_index, nearest_index)
        driving_waypoint = self._nearest_driving_waypoint(ego_location)
        return {
            "route_waypoint_count": len(self._route_locations),
            "next_waypoint_index": next_waypoint_index,
            "distance_to_next_waypoint_m": next_waypoint_distance,
            "nearest_route_waypoint_index": nearest_index,
            "nearest_route_waypoint_distance_m": nearest_distance,
            "cross_track_error_m": self._cross_track_error(ego_location),
            "ego_yaw_deg": ego_yaw_deg,
            "route_heading_deg": route_heading_deg,
            "heading_error_deg": self._heading_error_deg(ego_yaw_deg, route_heading_deg),
            "next_route_waypoint_location": self._location_to_dict(self._route_locations[next_waypoint_index]) if next_waypoint_index is not None else None,
            "nearest_route_waypoint_location": self._location_to_dict(self._route_locations[nearest_index]) if nearest_index is not None else None,
            "nearest_driving_waypoint_location": self._location_to_dict(driving_waypoint.transform.location) if driving_waypoint is not None else None,
            "nearest_driving_waypoint_yaw_deg": float(driving_waypoint.transform.rotation.yaw) if driving_waypoint is not None else None,
            "remaining_waypoints": max(len(self._route_locations) - self._route_progress_index, 0),
        }

    def _nearest_route_waypoint(self, ego_location: Any, start_index: int) -> tuple[int | None, float | None]:
        best_index = None
        best_distance = None
        for index in range(start_index, len(self._route_locations)):
            distance = float(ego_location.distance(self._route_locations[index]))
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index, best_distance

    def _cross_track_error(self, ego_location: Any) -> float | None:
        if len(self._route_locations) == 1:
            return float(ego_location.distance(self._route_locations[0]))
        if len(self._route_locations) < 2:
            return None

        px = float(ego_location.x)
        py = float(ego_location.y)
        best_distance = None
        for start, end in zip(self._route_locations, self._route_locations[1:]):
            distance = self._distance_point_to_segment_2d(
                px,
                py,
                float(start.x),
                float(start.y),
                float(end.x),
                float(end.y),
            )
            if best_distance is None or distance < best_distance:
                best_distance = distance
        return best_distance

    def _route_heading_deg(self, next_waypoint_index: int | None, nearest_index: int | None) -> float | None:
        if not self._route_locations:
            return None

        heading_index = next_waypoint_index if next_waypoint_index is not None else nearest_index
        if heading_index is None:
            return None
        if heading_index >= len(self._route_locations) - 1:
            if heading_index == 0:
                return None
            start = self._route_locations[heading_index - 1]
            end = self._route_locations[heading_index]
        else:
            start = self._route_locations[heading_index]
            end = self._route_locations[heading_index + 1]

        delta_x = float(end.x - start.x)
        delta_y = float(end.y - start.y)
        if abs(delta_x) < 1e-6 and abs(delta_y) < 1e-6:
            return None
        return math.degrees(math.atan2(delta_y, delta_x))

    def _nearest_driving_waypoint(self, ego_location: Any) -> Any:
        try:
            carla = importlib.import_module("carla")
            return self._world.get_map().get_waypoint(
                ego_location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception:
            return None

    @staticmethod
    def _heading_error_deg(ego_heading_deg: float | None, route_heading_deg: float | None) -> float | None:
        if ego_heading_deg is None or route_heading_deg is None:
            return None
        return ((route_heading_deg - ego_heading_deg + 180.0) % 360.0) - 180.0

    @staticmethod
    def _distance_point_to_segment_2d(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        denom = abx * abx + aby * aby
        if denom == 0.0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        closest_x = ax + t * abx
        closest_y = ay + t * aby
        return math.hypot(px - closest_x, py - closest_y)

    @staticmethod
    def _location_to_dict(location: Any) -> dict[str, float]:
        return {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        }