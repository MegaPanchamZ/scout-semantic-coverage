from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from research.harness.models import ScenarioSpec


@dataclass(slots=True)
class ScenarioController:
    spawned_actors: list[Any] = field(default_factory=list)

    def setup(self, context: dict[str, Any]) -> None:
        return None

    def on_tick(self, tick_index: int, context: dict[str, Any]) -> None:
        return None

    def teardown(self, context: dict[str, Any]) -> None:
        return None


class RouteOnlyController(ScenarioController):
    pass


class RelativeJaywalkerController(ScenarioController):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__()
        self.params = {
            "forward_offset": 14.0,
            "right_offset": 4.5,
            "walker_speed": 1.8,
            "yaw_offset": -90.0,
            "crossing_direction": -1.0,
        }
        self.params.update(params)
        self._walker: Any | None = None
        self._walker_control: Any | None = None

    def setup(self, context: dict[str, Any]) -> None:
        world = context["world"]
        ego_vehicle = context["ego_vehicle"]
        carla = context["carla"]

        ego_transform = ego_vehicle.get_transform()
        forward = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()
        spawn_location = ego_transform.location + carla.Location(
            x=forward.x * self.params["forward_offset"] + right.x * self.params["right_offset"],
            y=forward.y * self.params["forward_offset"] + right.y * self.params["right_offset"],
            z=0.8,
        )
        spawn_rotation = carla.Rotation(yaw=ego_transform.rotation.yaw + self.params["yaw_offset"])
        walker_bp = world.get_blueprint_library().filter("walker.pedestrian.*")[0]
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")
        walker = world.try_spawn_actor(walker_bp, carla.Transform(spawn_location, spawn_rotation))
        if walker is None:
            context.setdefault("scenario_notes", []).append("Failed to spawn jaywalker actor.")
            return

        direction = carla.Vector3D(
            x=right.x * self.params["crossing_direction"],
            y=right.y * self.params["crossing_direction"],
            z=0.0,
        )
        self._walker = walker
        self._walker_control = carla.WalkerControl(direction=direction, speed=self.params["walker_speed"])
        self.spawned_actors.append(walker)
        context["jaywalker"] = walker

    def on_tick(self, tick_index: int, context: dict[str, Any]) -> None:
        if self._walker is None or self._walker_control is None:
            return
        self._walker.apply_control(self._walker_control)


class RelativeOccludedJaywalkerController(RelativeJaywalkerController):
    def setup(self, context: dict[str, Any]) -> None:
        super().setup(context)
        world = context["world"]
        ego_vehicle = context["ego_vehicle"]
        carla = context["carla"]

        ego_transform = ego_vehicle.get_transform()
        forward = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()
        occluder_location = ego_transform.location + carla.Location(
            x=forward.x * (self.params.get("forward_offset", 14.0) - 2.0) + right.x * 2.2,
            y=forward.y * (self.params.get("forward_offset", 14.0) - 2.0) + right.y * 2.2,
            z=0.3,
        )
        truck_bp = world.get_blueprint_library().filter("vehicle.carlamotors.firetruck")[0]
        truck_bp.set_attribute("role_name", "scenario_occluder")
        truck = world.try_spawn_actor(truck_bp, carla.Transform(occluder_location, ego_transform.rotation))
        if truck is None:
            context.setdefault("scenario_notes", []).append("Failed to spawn occluder vehicle.")
            return

        truck.set_simulate_physics(False)
        self.spawned_actors.append(truck)
        context["occluder_vehicle"] = truck


def _location_from_dict(carla: Any, payload: dict[str, Any]) -> Any:
    return carla.Location(
        x=float(payload["x"]),
        y=float(payload["y"]),
        z=float(payload.get("z", 0.0)),
    )


def _rotation_from_dict(carla: Any, payload: dict[str, Any] | None) -> Any:
    payload = payload or {}
    return carla.Rotation(
        pitch=float(payload.get("pitch", 0.0)),
        yaw=float(payload.get("yaw", 0.0)),
        roll=float(payload.get("roll", 0.0)),
    )


def _normalize_vector(carla: Any, start: Any, end: Any) -> Any:
    delta_x = float(end.x - start.x)
    delta_y = float(end.y - start.y)
    delta_z = float(end.z - start.z)
    norm = sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
    if norm <= 1e-6:
        return carla.Vector3D(x=0.0, y=0.0, z=0.0)
    return carla.Vector3D(x=delta_x / norm, y=delta_y / norm, z=delta_z / norm)


class ThresholdCrossingAdversaryController(ScenarioController):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__()
        self.params = dict(params)
        self._triggered = False
        self._actor: Any | None = None
        self._walker_control: Any | None = None
        self._vehicle_control: Any | None = None

    def setup(self, context: dict[str, Any]) -> None:
        world = context["world"]
        carla = context["carla"]
        notes = context.setdefault("scenario_notes", [])

        adversary_kind = str(self.params.get("adversary_kind", "walker"))
        spawn_transform_payload = dict(self.params["spawn_transform"])
        destination_payload = dict(self.params["destination_location"])
        spawn_location = _location_from_dict(carla, dict(spawn_transform_payload["location"]))
        spawn_rotation = _rotation_from_dict(carla, spawn_transform_payload.get("rotation"))
        spawn_transform = carla.Transform(spawn_location, spawn_rotation)
        destination_location = _location_from_dict(carla, destination_payload)

        if adversary_kind == "walker":
            blueprint_filter = str(self.params.get("blueprint_filter", "walker.pedestrian.*"))
            walker_bps = world.get_blueprint_library().filter(blueprint_filter)
            if not walker_bps:
                notes.append(f"No walker blueprints matched '{blueprint_filter}'.")
                return
            walker_bp = walker_bps[0]
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            actor = world.try_spawn_actor(walker_bp, spawn_transform)
            if actor is None:
                notes.append("Failed to spawn threshold-triggered walker adversary.")
                return
            self._walker_control = carla.WalkerControl(
                direction=_normalize_vector(carla, spawn_location, destination_location),
                speed=0.0,
            )
            actor.apply_control(self._walker_control)
            context["jaywalker"] = actor
        elif adversary_kind == "vehicle":
            blueprint_filter = str(self.params.get("blueprint_filter", "vehicle.*"))
            vehicle_bps = world.get_blueprint_library().filter(blueprint_filter)
            if not vehicle_bps:
                notes.append(f"No vehicle blueprints matched '{blueprint_filter}'.")
                return
            vehicle_bp = vehicle_bps[0]
            vehicle_bp.set_attribute("role_name", "scenario_crossing_adversary")
            actor = world.try_spawn_actor(vehicle_bp, spawn_transform)
            if actor is None:
                notes.append("Failed to spawn threshold-triggered vehicle adversary.")
                return
            actor.set_autopilot(False)
            self._vehicle_control = carla.VehicleControl(
                throttle=float(self.params.get("vehicle_throttle", 0.45)),
                steer=float(self.params.get("vehicle_steer", 0.0)),
                brake=1.0,
                hand_brake=False,
            )
            actor.apply_control(self._vehicle_control)
        else:
            raise ValueError(f"Unsupported threshold adversary kind '{adversary_kind}'")

        self._actor = actor
        self.spawned_actors.append(actor)
        context["threshold_adversary"] = actor
        context["threshold_adversary_kind"] = adversary_kind
        context["threshold_adversary_active"] = False
        context["threshold_adversary_triggered"] = False
        context["threshold_trigger_location"] = dict(self.params["trigger_location"])

    def on_tick(self, tick_index: int, context: dict[str, Any]) -> None:
        if self._actor is None:
            return

        ego_vehicle = context["ego_vehicle"]
        carla = context["carla"]
        trigger_location = _location_from_dict(carla, dict(self.params["trigger_location"]))
        trigger_radius_m = float(self.params.get("trigger_radius_m", 8.0))
        ego_location = ego_vehicle.get_location()
        trigger_distance = float(ego_location.distance(trigger_location))
        context["threshold_trigger_distance_m"] = trigger_distance

        trigger_tick_raw = self.params.get("trigger_tick")
        trigger_tick = int(trigger_tick_raw) if trigger_tick_raw is not None else None
        tick_deadline_reached = trigger_tick is not None and (tick_index + 1) >= trigger_tick

        if not self._triggered and (trigger_distance <= trigger_radius_m or tick_deadline_reached):
            self._triggered = True
            context["threshold_adversary_active"] = True
            context["threshold_adversary_triggered"] = True
            if trigger_distance <= trigger_radius_m:
                trigger_reason = f"distance {trigger_distance:.2f} m"
            else:
                trigger_reason = f"tick deadline {trigger_tick}"
            context.setdefault("scenario_notes", []).append(
                f"Threshold adversary triggered at tick {tick_index + 1} ({trigger_reason})."
            )

        if not self._triggered:
            return

        adversary_kind = str(self.params.get("adversary_kind", "walker"))
        if adversary_kind == "walker" and self._walker_control is not None:
            self._walker_control.speed = float(self.params.get("speed", 1.8))
            self._actor.apply_control(self._walker_control)
            return

        if adversary_kind == "vehicle" and self._vehicle_control is not None:
            self._vehicle_control.brake = 0.0
            self._vehicle_control.throttle = float(self.params.get("vehicle_throttle", 0.45))
            self._actor.apply_control(self._vehicle_control)


class LaneDepartureAdversaryController(ScenarioController):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__()
        self.params = dict(params)
        self._triggered = False
        self._actor: Any | None = None

    def setup(self, context: dict[str, Any]) -> None:
        world = context["world"]
        carla = context["carla"]
        notes = context.setdefault("scenario_notes", [])

        vehicle_bps = world.get_blueprint_library().filter(str(self.params.get("blueprint_filter", "vehicle.*")))
        if not vehicle_bps:
            notes.append("No vehicle blueprints matched the lane-departure adversary filter.")
            return
        vehicle_bp = vehicle_bps[0]
        vehicle_bp.set_attribute("role_name", "scenario_lane_departure_adversary")
        staging_transform = carla.Transform(
            _location_from_dict(carla, dict(self.params["staging_transform"]["location"])),
            _rotation_from_dict(carla, self.params["staging_transform"].get("rotation")),
        )
        actor = world.try_spawn_actor(vehicle_bp, staging_transform)
        if actor is None:
            notes.append("Failed to spawn lane-departure adversary vehicle.")
            return
        actor.set_simulate_physics(False)
        self._actor = actor
        self.spawned_actors.append(actor)
        context["lane_departure_obstacle"] = actor
        context["lane_departure_adversary_active"] = False
        context["lane_departure_trigger_location"] = dict(self.params["trigger_location"])

    def on_tick(self, tick_index: int, context: dict[str, Any]) -> None:
        if self._actor is None or self._triggered:
            return
        ego_vehicle = context["ego_vehicle"]
        carla = context["carla"]
        trigger_location = _location_from_dict(carla, dict(self.params["trigger_location"]))
        trigger_radius_m = float(self.params.get("trigger_radius_m", 8.0))
        ego_location = ego_vehicle.get_location()
        trigger_distance = float(ego_location.distance(trigger_location))
        context["lane_departure_trigger_distance_m"] = trigger_distance
        if trigger_distance > trigger_radius_m:
            return

        active_transform = carla.Transform(
            _location_from_dict(carla, dict(self.params["active_transform"]["location"])),
            _rotation_from_dict(carla, self.params["active_transform"].get("rotation")),
        )
        self._actor.set_transform(active_transform)
        self._triggered = True
        context["lane_departure_adversary_active"] = True
        context.setdefault("scenario_notes", []).append(
            f"Lane-departure adversary activated at tick {tick_index + 1} (distance {trigger_distance:.2f} m)."
        )


class LeadVehicleBrakingController(ScenarioController):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__()
        self.params = dict(params)
        self._triggered = False
        self._actor: Any | None = None
        self._pre_brake_control: Any | None = None
        self._post_brake_control: Any | None = None

    def setup(self, context: dict[str, Any]) -> None:
        world = context["world"]
        carla = context["carla"]
        notes = context.setdefault("scenario_notes", [])

        vehicle_bps = world.get_blueprint_library().filter(str(self.params.get("blueprint_filter", "vehicle.*")))
        if not vehicle_bps:
            notes.append("No vehicle blueprints matched the lead-vehicle adversary filter.")
            return
        vehicle_bp = vehicle_bps[0]
        vehicle_bp.set_attribute("role_name", "scenario_lead_vehicle")
        spawn_transform = carla.Transform(
            _location_from_dict(carla, dict(self.params["spawn_transform"]["location"])),
            _rotation_from_dict(carla, self.params["spawn_transform"].get("rotation")),
        )
        actor = world.try_spawn_actor(vehicle_bp, spawn_transform)
        if actor is None:
            notes.append("Failed to spawn lead-vehicle braking adversary.")
            return
        actor.set_autopilot(False)
        self._actor = actor
        self.spawned_actors.append(actor)
        self._pre_brake_control = carla.VehicleControl(
            throttle=float(self.params.get("pre_brake_throttle", 0.3)),
            brake=0.0,
            steer=0.0,
            hand_brake=False,
        )
        self._post_brake_control = carla.VehicleControl(
            throttle=0.0,
            brake=float(self.params.get("post_trigger_brake", 1.0)),
            steer=0.0,
            hand_brake=False,
        )
        actor.apply_control(self._pre_brake_control)
        context["lead_vehicle"] = actor
        context["lead_vehicle_braking_active"] = False
        context["lead_vehicle_trigger_location"] = dict(self.params["trigger_location"])

    def on_tick(self, tick_index: int, context: dict[str, Any]) -> None:
        if self._actor is None:
            return
        ego_vehicle = context["ego_vehicle"]
        carla = context["carla"]
        trigger_location = _location_from_dict(carla, dict(self.params["trigger_location"]))
        trigger_radius_m = float(self.params.get("trigger_radius_m", 8.0))
        trigger_distance = float(ego_vehicle.get_location().distance(trigger_location))
        context["lead_vehicle_trigger_distance_m"] = trigger_distance
        if not self._triggered and trigger_distance <= trigger_radius_m:
            self._triggered = True
            context["lead_vehicle_braking_active"] = True
            context.setdefault("scenario_notes", []).append(
                f"Lead-vehicle braking adversary triggered at tick {tick_index + 1} (distance {trigger_distance:.2f} m)."
            )
        if self._triggered:
            if self._post_brake_control is not None:
                self._actor.apply_control(self._post_brake_control)
            return
        if self._pre_brake_control is not None:
            self._actor.apply_control(self._pre_brake_control)


def build_scenario_controller(scenario: ScenarioSpec) -> ScenarioController:
    if scenario.controller == "route_only":
        return RouteOnlyController()
    if scenario.controller == "jaywalker":
        return RelativeJaywalkerController(scenario.controller_params)
    if scenario.controller == "occluded_jaywalker":
        return RelativeOccludedJaywalkerController(scenario.controller_params)
    if scenario.controller == "threshold_crossing_adversary":
        return ThresholdCrossingAdversaryController(scenario.controller_params)
    if scenario.controller == "lane_departure_adversary":
        return LaneDepartureAdversaryController(scenario.controller_params)
    if scenario.controller == "lead_vehicle_braking":
        return LeadVehicleBrakingController(scenario.controller_params)
    if scenario.controller == "ptop_adversarial":
        from research.harness.ptop_bridge import PtoPAdversarialController
        return PtoPAdversarialController(scenario.controller_params)
    raise ValueError(f"Unknown scenario controller '{scenario.controller}'")