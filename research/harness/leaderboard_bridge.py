from __future__ import annotations

import importlib
import inspect
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


def ensure_repo_path(repo_path: Path | None) -> None:
    if repo_path is None or not repo_path.exists():
        return

    repo_path = repo_path.resolve()
    candidate_paths: list[Path] = []
    seen: set[str] = set()

    def _append_if_exists(path: Path) -> None:
        candidate_str = str(path)
        if not path.exists() or candidate_str in seen:
            return
        seen.add(candidate_str)
        candidate_paths.append(path)

    project_root = repo_path
    if repo_path.name in {"team_code", "team_code_transfuser"}:
        if repo_path.parent.name == "leaderboard":
            project_root = repo_path.parent.parent
        else:
            project_root = repo_path.parent
    elif repo_path.name == "leaderboard":
        project_root = repo_path.parent

    _append_if_exists(project_root)
    _append_if_exists(project_root / "leaderboard")
    _append_if_exists(project_root / "scenario_runner")
    _append_if_exists(project_root / "leaderboard" / "team_code")
    _append_if_exists(project_root / "team_code_transfuser")
    _append_if_exists(repo_path)

    for candidate_path in reversed(candidate_paths):
        candidate_str = str(candidate_path)
        if candidate_str in sys.path:
            sys.path.remove(candidate_str)
        sys.path.insert(0, candidate_str)


def ensure_local_carla_agents_on_path() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    compat_root = workspace_root / "research" / "harness" / "compat"
    packaged_carla_roots = [
        Path(r"H:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla"),
        Path(r"H:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla"),
        Path(r"H:\CARLA_0.9.16\WindowsNoEditor\PythonAPI\carla"),
    ]
    local_agents_root = workspace_root / "CarlaUE" / "PythonAPI" / "carla"
    preferred_paths = [compat_root]
    preferred_paths.extend(path for path in packaged_carla_roots if path.exists())
    if local_agents_root.exists():
        preferred_paths.append(local_agents_root)
    preferred_paths = [path for path in preferred_paths if path.exists()]
    for preferred_path in reversed(preferred_paths):
        preferred_str = str(preferred_path)
        if preferred_str in sys.path:
            sys.path.remove(preferred_str)
        sys.path.insert(0, preferred_str)


class LeaderboardSensorSuite:
    def __init__(self, world: Any, ego_vehicle: Any) -> None:
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.sensor_actors: list[Any] = []
        self.latest_data: dict[str, tuple[float, Any]] = {}

    def setup(self, sensor_specs: list[dict[str, Any]]) -> None:
        for spec in sensor_specs:
            sensor_type = spec["type"]
            if sensor_type == "sensor.camera.rgb":
                self._spawn_camera(spec)
            elif sensor_type == "sensor.lidar.ray_cast":
                self._spawn_lidar(spec)
            elif sensor_type == "sensor.other.gnss":
                self._spawn_gnss(spec)
            elif sensor_type == "sensor.other.imu":
                self._spawn_imu(spec)
            elif sensor_type in {"sensor.speedometer", "sensor.opendrive_map"}:
                continue
            else:
                raise NotImplementedError(f"Unsupported leaderboard sensor type '{sensor_type}'")

    def read_input_data(self, timestamp: float) -> dict[str, tuple[float, Any]]:
        input_data = dict(self.latest_data)
        input_data.update(
            {
                spec_id: (timestamp, self._read_pseudosensor(spec_id, sensor_type))
                for spec_id, sensor_type in self._iter_pseudosensors()
            }
        )
        return input_data

    def destroy(self) -> None:
        for actor in reversed(self.sensor_actors):
            try:
                actor.stop()
            except Exception:
                pass
            try:
                actor.destroy()
            except Exception:
                pass
        self.sensor_actors.clear()

    def _iter_pseudosensors(self) -> list[tuple[str, str]]:
        return getattr(self, "_pseudosensors", [])

    def has_data_for(self, sensor_ids: list[str]) -> bool:
        return all(sensor_id in self.latest_data for sensor_id in sensor_ids)

    def _register_pseudosensor(self, spec_id: str, sensor_type: str) -> None:
        if not hasattr(self, "_pseudosensors"):
            self._pseudosensors: list[tuple[str, str]] = []
        self._pseudosensors.append((spec_id, sensor_type))

    def _spawn_camera(self, spec: dict[str, Any]) -> None:
        carla = importlib.import_module("carla")
        blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(spec.get("width", 800)))
        blueprint.set_attribute("image_size_y", str(spec.get("height", 600)))
        blueprint.set_attribute("fov", str(spec.get("fov", 90)))
        transform = carla.Transform(
            carla.Location(x=spec.get("x", 0.0), y=spec.get("y", 0.0), z=spec.get("z", 0.0)),
            carla.Rotation(
                roll=spec.get("roll", 0.0),
                pitch=spec.get("pitch", 0.0),
                yaw=spec.get("yaw", 0.0),
            ),
        )
        sensor = self.world.spawn_actor(blueprint, transform, attach_to=self.ego_vehicle)
        sensor_id = spec["id"]

        def _on_image(image: Any) -> None:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))[:, :, :3]
            self.latest_data[sensor_id] = (float(image.timestamp), array)

        sensor.listen(_on_image)
        self.sensor_actors.append(sensor)

    def _spawn_gnss(self, spec: dict[str, Any]) -> None:
        carla = importlib.import_module("carla")
        blueprint = self.world.get_blueprint_library().find("sensor.other.gnss")
        if "sensor_tick" in spec:
            blueprint.set_attribute("sensor_tick", str(spec["sensor_tick"]))
        transform = carla.Transform(carla.Location(x=spec.get("x", 0.0), y=spec.get("y", 0.0), z=spec.get("z", 0.0)))
        sensor = self.world.spawn_actor(blueprint, transform, attach_to=self.ego_vehicle)
        sensor_id = spec["id"]

        def _on_gnss(event: Any) -> None:
            self.latest_data[sensor_id] = (
                float(event.timestamp),
                np.array([event.latitude, event.longitude, event.altitude], dtype=np.float32),
            )

        sensor.listen(_on_gnss)
        self.sensor_actors.append(sensor)

    def _spawn_imu(self, spec: dict[str, Any]) -> None:
        carla = importlib.import_module("carla")
        blueprint = self.world.get_blueprint_library().find("sensor.other.imu")
        if "sensor_tick" in spec:
            blueprint.set_attribute("sensor_tick", str(spec["sensor_tick"]))
        transform = carla.Transform(
            carla.Location(x=spec.get("x", 0.0), y=spec.get("y", 0.0), z=spec.get("z", 0.0)),
            carla.Rotation(
                roll=spec.get("roll", 0.0),
                pitch=spec.get("pitch", 0.0),
                yaw=spec.get("yaw", 0.0),
            ),
        )
        sensor = self.world.spawn_actor(blueprint, transform, attach_to=self.ego_vehicle)
        sensor_id = spec["id"]

        def _on_imu(event: Any) -> None:
            gyroscope = np.array(
                [
                    math.degrees(event.gyroscope.x),
                    math.degrees(event.gyroscope.y),
                    math.degrees(event.gyroscope.z),
                ],
                dtype=np.float32,
            )
            self.latest_data[sensor_id] = (
                float(event.timestamp),
                np.array(
                    [
                        event.accelerometer.x,
                        event.accelerometer.y,
                        event.accelerometer.z,
                        gyroscope[0],
                        gyroscope[1],
                        gyroscope[2],
                        float(event.compass),
                    ],
                    dtype=np.float32,
                ),
            )

        sensor.listen(_on_imu)
        self.sensor_actors.append(sensor)

    def _spawn_lidar(self, spec: dict[str, Any]) -> None:
        carla = importlib.import_module("carla")
        blueprint = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        lidar_attributes = {
            "channels": spec.get("channels", 64),
            "range": spec.get("range", 85),
            "points_per_second": spec.get("points_per_second", 600000),
            "rotation_frequency": spec.get("rotation_frequency", 10),
            "upper_fov": spec.get("upper_fov", 10),
            "lower_fov": spec.get("lower_fov", -30),
        }
        for attribute_name, attribute_value in lidar_attributes.items():
            blueprint.set_attribute(attribute_name, str(attribute_value))
        transform = carla.Transform(
            carla.Location(x=spec.get("x", 0.0), y=spec.get("y", 0.0), z=spec.get("z", 0.0)),
            carla.Rotation(
                roll=spec.get("roll", 0.0),
                pitch=spec.get("pitch", 0.0),
                yaw=spec.get("yaw", 0.0),
            ),
        )
        sensor = self.world.spawn_actor(blueprint, transform, attach_to=self.ego_vehicle)
        sensor_id = spec["id"]

        def _on_lidar(point_cloud: Any) -> None:
            array = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
            array = array.reshape((-1, 4))
            self.latest_data[sensor_id] = (float(point_cloud.timestamp), array)

        sensor.listen(_on_lidar)
        self.sensor_actors.append(sensor)

    def _read_pseudosensor(self, spec_id: str, sensor_type: str) -> Any:
        if sensor_type == "sensor.speedometer":
            velocity = self.ego_vehicle.get_velocity()
            return {
                "speed": float((velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5),
            }
        if sensor_type == "sensor.opendrive_map":
            return self.world.get_map().to_opendrive()
        raise NotImplementedError(f"Unsupported pseudosensor '{sensor_type}'")


class LeaderboardModuleAdapter:
    def __init__(self, world: Any, ego_vehicle: Any, agent_module: Any, agent_class_name: str, config_path: str | None = None) -> None:
        agent_class = getattr(agent_module, agent_class_name)
        self._world = world
        self._ego_vehicle = ego_vehicle
        self._goal_location = None
        self._route_locations: list[Any] = []
        self._route_progress_index = 0
        self._route_preview_limit = 12
        self.last_step_info: dict[str, Any] = {}
        init_signature = inspect.signature(agent_class)
        init_parameters = list(init_signature.parameters.values())
        if len(init_parameters) == 0:
            self._agent = agent_class()
            should_call_setup = hasattr(self._agent, "setup")
        else:
            self._agent = agent_class(config_path)
            should_call_setup = False
        self._sensor_suite = LeaderboardSensorSuite(world, ego_vehicle)

        if should_call_setup:
            self._agent.setup(config_path)
        if hasattr(self._agent, "bind_harness"):
            self._agent.bind_harness(world=world, ego_vehicle=ego_vehicle)

        sensor_specs = []
        if hasattr(self._agent, "sensors"):
            sensor_specs = self._agent.sensors() or []
        self._required_sensor_ids = [
            spec["id"]
            for spec in sensor_specs
            if spec["type"] not in {"sensor.speedometer", "sensor.opendrive_map"}
        ]
        self._sensor_suite.setup(sensor_specs)
        for spec in sensor_specs:
            if spec["type"] in {"sensor.speedometer", "sensor.opendrive_map"}:
                self._sensor_suite._register_pseudosensor(spec["id"], spec["type"])

    def set_destination(self, goal_location: Any) -> None:
        self._goal_location = goal_location
        if hasattr(self._agent, "set_destination"):
            self._agent.set_destination(goal_location)
            return

        gps_plan, world_plan = self._build_global_plans(goal_location)
        self._route_locations = [transform.location for transform, _road_option in world_plan]
        self._route_progress_index = 0
        if hasattr(self._agent, "set_global_plan"):
            self._agent.set_global_plan(gps_plan, world_plan)
        setattr(self._agent, "_global_plan_world_coord", world_plan)
        setattr(self._agent, "_global_plan", gps_plan)

    def run_step(self) -> Any:
        snapshot = self._world.get_snapshot()
        timestamp = float(snapshot.timestamp.elapsed_seconds)
        if not self._sensor_suite.has_data_for(self._required_sensor_ids):
            carla = importlib.import_module("carla")
            control = carla.VehicleControl()
            control.brake = 1.0
            self.last_step_info = {
                "sensor_ready": False,
                "mode": "sensor-wait",
                "required_sensor_ids": list(self._required_sensor_ids),
                "available_sensor_ids": sorted(self._sensor_suite.latest_data.keys()),
                "timestamp": timestamp,
            }
            return control
        input_data = self._sensor_suite.read_input_data(timestamp)
        self.last_step_info = {
            "sensor_ready": True,
            "mode": "agent-run-step",
            "required_sensor_ids": list(self._required_sensor_ids),
            "available_sensor_ids": sorted(input_data.keys()),
            "timestamp": timestamp,
        }
        signature = inspect.signature(self._agent.run_step)
        if len(signature.parameters) >= 2:
            return self._agent.run_step(input_data, timestamp)
        if len(signature.parameters) == 1:
            return self._agent.run_step(input_data)
        return self._agent.run_step()

    def get_debug_state(self) -> dict[str, Any]:
        debug_state = dict(self.last_step_info)
        debug_state["route_progress"] = self._route_debug_snapshot()
        return debug_state

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

    def done(self) -> bool:
        if hasattr(self._agent, "done"):
            return bool(self._agent.done())
        if self._goal_location is None:
            return False
        return self._ego_vehicle.get_location().distance(self._goal_location) < 5.0

    def destroy(self) -> None:
        self._sensor_suite.destroy()
        if hasattr(self._agent, "destroy"):
            self._agent.destroy()

    def _build_global_plans(self, goal_location: Any) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
        ensure_local_carla_agents_on_path()
        grp_mod = importlib.import_module("agents.navigation.global_route_planner")
        planner_class = grp_mod.GlobalRoutePlanner
        init_signature = inspect.signature(planner_class)
        if len(init_signature.parameters) == 1:
            dao_mod = importlib.import_module("agents.navigation.global_route_planner_dao")
            dao = dao_mod.GlobalRoutePlannerDAO(self._world.get_map(), 2.0)
            planner = planner_class(dao)
            if hasattr(planner, "setup"):
                planner.setup()
        else:
            planner = planner_class(self._world.get_map(), 2.0)
        raw_plan = planner.trace_route(self._ego_vehicle.get_location(), goal_location)
        world_plan = [(waypoint.transform, road_option) for waypoint, road_option in raw_plan]

        route_mod = importlib.import_module("leaderboard.utils.route_manipulation")
        lat_ref, lon_ref = route_mod._get_latlon_ref(self._world)
        gps_plan = route_mod.location_route_to_gps(world_plan, lat_ref, lon_ref)
        return gps_plan, world_plan

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
        if next_waypoint_index is not None and next_waypoint_index + 1 < len(self._route_locations):
            start = self._route_locations[next_waypoint_index]
            end = self._route_locations[next_waypoint_index + 1]
            return self._bearing_deg(start, end)
        if nearest_index is not None and nearest_index > 0:
            start = self._route_locations[nearest_index - 1]
            end = self._route_locations[nearest_index]
            return self._bearing_deg(start, end)
        return None

    def _nearest_driving_waypoint(self, ego_location: Any) -> Any | None:
        try:
            return self._world.get_map().get_waypoint(ego_location, project_to_road=True)
        except RuntimeError:
            return None

    @staticmethod
    def _location_to_dict(location: Any) -> dict[str, float]:
        return {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        }

    @staticmethod
    def _bearing_deg(start: Any, end: Any) -> float:
        return math.degrees(math.atan2(float(end.y) - float(start.y), float(end.x) - float(start.x)))

    @staticmethod
    def _heading_error_deg(ego_yaw_deg: float | None, route_heading_deg: float | None) -> float | None:
        if ego_yaw_deg is None or route_heading_deg is None:
            return None
        delta = route_heading_deg - ego_yaw_deg
        while delta <= -180.0:
            delta += 360.0
        while delta > 180.0:
            delta -= 360.0
        return delta

    @staticmethod
    def _distance_point_to_segment_2d(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        abx = bx - ax
        aby = by - ay
        ab_squared = abx * abx + aby * aby
        if ab_squared <= 1e-12:
            return math.hypot(px - ax, py - ay)
        projection = ((px - ax) * abx + (py - ay) * aby) / ab_squared
        projection = max(0.0, min(1.0, projection))
        closest_x = ax + projection * abx
        closest_y = ay + projection * aby
        return math.hypot(px - closest_x, py - closest_y)


def load_leaderboard_agent(world: Any, ego_vehicle: Any, repo_path: Path | None, module_name: str, config_path: str | None) -> LeaderboardModuleAdapter:
    ensure_local_carla_agents_on_path()
    ensure_repo_path(repo_path)
    agent_module = importlib.import_module(module_name)
    if not hasattr(agent_module, "get_entry_point"):
        raise AttributeError(f"Leaderboard module '{module_name}' does not define get_entry_point().")
    class_name = agent_module.get_entry_point()
    return LeaderboardModuleAdapter(world, ego_vehicle, agent_module, class_name, config_path=config_path)