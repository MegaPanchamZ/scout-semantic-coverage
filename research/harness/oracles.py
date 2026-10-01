from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from research.harness.models import CollisionEvent, SafetyMetrics


@dataclass(slots=True)
class SafetyThresholds:
    """Detection thresholds for the live safety oracle.

    These are engineering proxies for physical risk, not validated labels; the
    named safety *outcomes* derived from them are defined separately in
    ``research/harness/safety_outcomes.py`` so reporting thresholds can move
    without changing what the simulator records.
    """

    near_collision_distance_m: float = 2.5
    near_collision_ttc_s: float = 1.5
    unsafe_pedestrian_distance_m: float = 2.0
    unsafe_vehicle_distance_m: float = 2.5
    lane_offset_warn_m: float = 2.0
    harsh_deceleration_mps2: float = 3.0
    emergency_deceleration_mps2: float = 6.0
    stationary_speed_mps: float = 0.1
    stuck_seconds: float = 8.0
    scan_radius_m: float = 50.0


class SafetyOracle:
    """Records raw safety evidence for one episode, tick by tick.

    Detects low time-to-collision, entries into a near-collision band, minimum
    gaps to vehicles and pedestrians, red-light violations, a lane-centre offset
    (lane-departure proxy), stationary streaks (stuck), and harsh/emergency
    longitudinal deceleration.  Instrumentation must never abort an episode, so
    every read of the simulator is defensively wrapped.
    """

    def __init__(self, tick_seconds: float = 0.1, thresholds: SafetyThresholds | None = None) -> None:
        self.metrics = SafetyMetrics()
        self.tick_seconds = max(float(tick_seconds), 1e-6)
        self.thresholds = thresholds or SafetyThresholds()
        self._stationary_streak = 0
        self._near_collision_ids: set[int] = set()
        self._prev_speed: float | None = None
        self._harsh_active = False
        self._emergency_active = False
        self._lane_departure_active = False
        self._last_snapshot: dict[str, Any] = {}

    def tick(self, world: Any, ego_vehicle: Any, goal_location: Any | None = None) -> None:
        try:
            self._tick_impl(world, ego_vehicle, goal_location)
        except Exception:
            # Safety instrumentation must never abort the run.
            pass

    def snapshot(self) -> dict[str, Any]:
        """Most recent per-tick evidence, for persisting into the log trace."""
        return dict(self._last_snapshot)

    def _tick_impl(self, world: Any, ego_vehicle: Any, goal_location: Any | None) -> None:
        del goal_location  # reserved for progress-aware stuck detection
        th = self.thresholds
        ego_vel = ego_vehicle.get_velocity()
        speed = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2)
        ego_loc = ego_vehicle.get_location()

        decel = 0.0
        if self._prev_speed is not None:
            decel = (self._prev_speed - speed) / self.tick_seconds
        self._prev_speed = speed
        if decel > self.metrics.max_deceleration_mps2:
            self.metrics.max_deceleration_mps2 = decel

        control = None
        try:
            control = ego_vehicle.get_control()
        except Exception:
            control = None
        brake = float(getattr(control, "brake", 0.0)) if control is not None else 0.0
        steer = abs(float(getattr(control, "steer", 0.0))) if control is not None else 0.0

        if decel >= th.harsh_deceleration_mps2:
            if not self._harsh_active:
                self.metrics.harsh_braking_events += 1
                self._harsh_active = True
        else:
            self._harsh_active = False

        emergency = decel >= th.emergency_deceleration_mps2 or (brake >= 0.9 and steer >= 0.3)
        if emergency:
            if not self._emergency_active:
                self.metrics.emergency_manoeuvre_events += 1
                self._emergency_active = True
        else:
            self._emergency_active = False

        if speed < th.stationary_speed_mps:
            self._stationary_streak += 1
        else:
            self._stationary_streak = 0
        if self._stationary_streak > self.metrics.max_stationary_streak_ticks:
            self.metrics.max_stationary_streak_ticks = self._stationary_streak
        stuck_threshold_ticks = int(round(th.stuck_seconds / self.tick_seconds))
        if self._stationary_streak > stuck_threshold_ticks:
            self.metrics.stuck_frames += 1

        lane_offset = self._lane_offset(world, ego_loc)
        if lane_offset is not None:
            if lane_offset > self.metrics.max_lane_offset_m:
                self.metrics.max_lane_offset_m = lane_offset
            if speed > 1.0 and lane_offset > th.lane_offset_warn_m:
                if not self._lane_departure_active:
                    self.metrics.lane_departure_events += 1
                    self._lane_departure_active = True
            else:
                self._lane_departure_active = False

        try:
            if ego_vehicle.is_at_traffic_light():
                tl = ego_vehicle.get_traffic_light()
                if tl is not None and str(tl.get_state()).lower().endswith("red") and speed > 2.0:
                    self.metrics.red_light_violations += 1
                    self.metrics.rule_violations += 1
        except Exception:
            pass

        nearest_dist = float("inf")
        nearest_type: str | None = None
        current_near: set[int] = set()
        for actor in self._iter_actors(world):
            if actor.id == ego_vehicle.id:
                continue
            try:
                other_loc = actor.get_location()
                other_vel = actor.get_velocity()
            except Exception:
                continue
            dx = float(other_loc.x) - float(ego_loc.x)
            dy = float(other_loc.y) - float(ego_loc.y)
            dz = float(other_loc.z) - float(ego_loc.z)
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist <= 0.0 or dist > th.scan_radius_m:
                continue
            type_id = str(getattr(actor, "type_id", ""))
            alias = "pedestrian" if "walker" in type_id else "vehicle"

            if dist < nearest_dist:
                nearest_dist, nearest_type = dist, alias
            if dist < self.metrics.min_actor_distance_m:
                self.metrics.min_actor_distance_m = dist
            if alias == "pedestrian":
                if dist < self.metrics.min_pedestrian_distance_m:
                    self.metrics.min_pedestrian_distance_m = dist
            elif dist < self.metrics.min_vehicle_distance_m:
                self.metrics.min_vehicle_distance_m = dist

            rel_vx = float(other_vel.x) - float(ego_vel.x)
            rel_vy = float(other_vel.y) - float(ego_vel.y)
            rel_vz = float(other_vel.z) - float(ego_vel.z)
            closing = -(dx * rel_vx + dy * rel_vy + dz * rel_vz) / dist
            if closing > 0.1:
                ttc = dist / closing
                if ttc < self.metrics.min_ttc:
                    self.metrics.min_ttc = ttc

            if dist < th.near_collision_distance_m:
                current_near.add(int(actor.id))
                self.metrics.near_collision_ticks += 1

        new_entries = current_near - self._near_collision_ids
        if new_entries:
            self.metrics.near_collisions += len(new_entries)
        self._near_collision_ids = current_near

        self._last_snapshot = {
            "speed_mps": speed,
            "deceleration_mps2": decel,
            "nearest_actor_type": nearest_type,
            "nearest_actor_distance_m": None if math.isinf(nearest_dist) else nearest_dist,
            "lane_offset_m": lane_offset,
            "min_ttc_s": None if math.isinf(self.metrics.min_ttc) else self.metrics.min_ttc,
        }

    @staticmethod
    def _iter_actors(world: Any) -> list[Any]:
        try:
            actors = world.get_actors()
        except Exception:
            return []
        nearby: list[Any] = []
        for actor in actors:
            type_id = str(getattr(actor, "type_id", ""))
            if type_id.startswith("vehicle.") or type_id.startswith("walker."):
                nearby.append(actor)
        return nearby

    @staticmethod
    def _lane_offset(world: Any, ego_loc: Any) -> float | None:
        try:
            carla_map = world.get_map()
            waypoint = carla_map.get_waypoint(ego_loc, project_to_road=True)
        except Exception:
            return None
        try:
            if bool(getattr(waypoint, "is_junction", False)):
                return None
            center = waypoint.transform.location
            return math.hypot(float(ego_loc.x) - float(center.x), float(ego_loc.y) - float(center.y))
        except Exception:
            return None


class CollisionOracle:
    def __init__(self) -> None:
        self.events: list[CollisionEvent] = []
        self.injected_actor_ids: set[int] = set()
        self._sensor: Any | None = None

    def attach(self, world: Any, ego_vehicle: Any) -> Any:
        blueprint = world.get_blueprint_library().find("sensor.other.collision")
        sensor = world.spawn_actor(blueprint, self._identity_transform(world), attach_to=ego_vehicle)
        weak_self = self

        def _on_collision(event: Any) -> None:
            impulse = event.normal_impulse
            intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            other_actor = event.other_actor
            weak_self.events.append(
                CollisionEvent(
                    frame=event.frame,
                    actor_id=other_actor.id,
                    actor_type=other_actor.type_id,
                    intensity=float(intensity),
                    injected_actor=other_actor.id in weak_self.injected_actor_ids,
                )
            )

        sensor.listen(_on_collision)
        self._sensor = sensor
        return sensor

    def destroy(self) -> None:
        if self._sensor is not None:
            try:
                self._sensor.stop()
            except RuntimeError:
                pass
            try:
                self._sensor.destroy()
            except RuntimeError:
                pass
            self._sensor = None

    @staticmethod
    def _identity_transform(world: Any) -> Any:
        carla = type(world).__module__.split(".")[0]
        module = __import__(carla)
        return module.Transform()
