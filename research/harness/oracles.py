from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
from typing import Any

from research.harness.models import CollisionEvent, SafetyMetrics
from research.harness.outcome_models import ActorState, FrameEvidence, SensorEvidence


@dataclass(slots=True)
class SafetyThresholds:
    """Detection thresholds for the live safety oracle.

    These are engineering proxies for physical risk, not validated labels; the
    named safety *outcomes* derived from them are defined separately in
    ``research/harness/legacy_outcomes.py`` so reporting thresholds can move
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
        nearest_actor_id: int | None = None
        tick_min_ttc = float("inf")
        tick_min_ttc_actor_id: int | None = None
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
                nearest_dist, nearest_type, nearest_actor_id = dist, alias, int(actor.id)
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
                if ttc < tick_min_ttc:
                    tick_min_ttc = ttc
                    tick_min_ttc_actor_id = int(actor.id)
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
            "nearest_actor_id": nearest_actor_id,
            "nearest_actor_type": nearest_type,
            "nearest_actor_distance_m": None if math.isinf(nearest_dist) else nearest_dist,
            "lane_offset_m": lane_offset,
            "tick_min_ttc_s": None if math.isinf(tick_min_ttc) else tick_min_ttc,
            "min_ttc_actor_id": tick_min_ttc_actor_id,
            "min_ttc_s": None if math.isinf(tick_min_ttc) else tick_min_ttc,
            "run_min_ttc_s": None if math.isinf(self.metrics.min_ttc) else self.metrics.min_ttc,
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
    def _actor_to_state(actor: Any, *, role: str | None = None) -> ActorState | None:
        try:
            transform = actor.get_transform()
            loc = transform.location
            rot = transform.rotation
            vel = actor.get_velocity()
            bbox = getattr(actor, "bounding_box", None)
            if bbox is None:
                return None  # Never fabricate geometry when bounding_box is unavailable
            ext = bbox.extent
            b_loc = bbox.location
            b_rot = getattr(bbox, "rotation", None)
            half_l = float(ext.x)
            half_w = float(ext.y)
            half_h = float(ext.z)
            if half_l <= 0.0 or half_w <= 0.0 or half_h <= 0.0:
                return None
            off_x = float(b_loc.x)
            off_y = float(b_loc.y)
            off_z = float(b_loc.z)
            b_yaw = math.radians(float(getattr(b_rot, "yaw", 0.0))) if b_rot is not None else 0.0
            type_id = str(getattr(actor, "type_id", "unknown"))
            actor_type = "pedestrian" if "walker" in type_id else ("vehicle" if "vehicle" in type_id else type_id or "unknown")
            z_center = float(loc.z) + off_z
            return ActorState(
                actor_id=str(getattr(actor, "id", "unknown")),
                actor_type=actor_type,
                x=float(loc.x),
                y=float(loc.y),
                yaw_rad=math.radians(float(rot.yaw)),
                half_length_m=half_l,
                half_width_m=half_w,
                vx=float(vel.x),
                vy=float(vel.y),
                z_min_m=z_center - half_h,
                z_max_m=z_center + half_h,
                role=role,
                bbox_offset_x=off_x,
                bbox_offset_y=off_y,
                bbox_yaw_rad=b_yaw,
            )
        except Exception:
            return None

    def capture_frame_evidence(
        self,
        world: Any,
        ego_vehicle: Any,
        *,
        run_id: str,
        sim_frame: int,
        sim_time_s: float,
        phase: str = "evaluation",
        scan_radius_m: float = 50.0,
        control: Any | None = None,
        logical_tick: int | None = None,
        substep: int | None = None,
        actor_roles: dict[int, str] | None = None,
        road_index: Any | None = None,
        route_tracker: Any | None = None,
        prev_ego_state: ActorState | None = None,
        objective_active: bool | None = True,
        sensor_wait: bool | None = False,
        discontinuity: bool = False,
    ) -> tuple[FrameEvidence | None, dict[str, str]]:
        """Capture one normalized v2 FrameEvidence with per-channel error isolation."""
        errors: dict[str, str] = {}
        available: list[str] = []
        roles = actor_roles or {}

        ego_role = roles.get(int(getattr(ego_vehicle, "id", -1)), "ego")
        ego_state = self._actor_to_state(ego_vehicle, role=ego_role)
        if ego_state is None:
            errors["ego_motion"] = "ego_state_read_failed"
            return None, errors
        available.append("ego_motion")

        # Longitudinal acceleration projected onto ego forward axis
        long_accel: float | None = None
        try:
            accel = ego_vehicle.get_acceleration()
            fwd_x, fwd_y = math.cos(ego_state.yaw_rad), math.sin(ego_state.yaw_rad)
            val = float(accel.x) * fwd_x + float(accel.y) * fwd_y
            if math.isfinite(val):
                long_accel = val
                available.append("acceleration")
        except Exception as exc:
            errors["acceleration"] = f"{type(exc).__name__}:{exc}"

        # Applied controls
        brake_val: float | None = None
        throttle_val: float | None = None
        steer_val: float | None = None
        reverse_val: bool | None = None
        try:
            ctrl = control if control is not None else ego_vehicle.get_control()
            if ctrl is not None:
                brake_val = max(0.0, min(1.0, float(getattr(ctrl, "brake", 0.0))))
                throttle_val = max(0.0, min(1.0, float(getattr(ctrl, "throttle", 0.0))))
                steer_val = max(-1.0, min(1.0, float(getattr(ctrl, "steer", 0.0))))
                reverse_val = bool(getattr(ctrl, "reverse", False))
                available.append("controls")
        except Exception as exc:
            errors["controls"] = f"{type(exc).__name__}:{exc}"

        # Nearby actors scan (distinguishes failed/incomplete actor scan from successful empty scan)
        nearby_states: list[ActorState] = []
        dropped_actors: list[str] = []
        actors_ok = False
        try:
            raw_actors = world.get_actors()
            ego_id = getattr(ego_vehicle, "id", None)
            for actor in raw_actors:
                if getattr(actor, "id", None) == ego_id:
                    continue
                tid = str(getattr(actor, "type_id", ""))
                if not (tid.startswith("vehicle.") or tid.startswith("walker.")):
                    continue
                # Pre-filter distant world actors before checking bounding_box so an out-of-range
                # actor missing geometry cannot poison the in-range scan channel.
                try:
                    aloc = actor.get_location()
                    pre_dist = math.hypot(float(aloc.x) - ego_state.x, float(aloc.y) - ego_state.y)
                except Exception:
                    dropped_actors.append(str(getattr(actor, "id", "unknown")))
                    continue
                if not math.isfinite(pre_dist):
                    dropped_actors.append(str(getattr(actor, "id", "unknown")))
                    continue
                if pre_dist > scan_radius_m + 10.0:
                    continue
                a_role = roles.get(int(getattr(actor, "id", -1)))
                state = self._actor_to_state(actor, role=a_role)
                if state is None:
                    dropped_actors.append(str(getattr(actor, "id", "unknown")))
                    continue
                dist = math.hypot(state.x - ego_state.x, state.y - ego_state.y)
                if dist <= scan_radius_m:
                    nearby_states.append(state)
            if dropped_actors:
                errors["actors"] = f"unreadable_actor_geometry:{','.join(dropped_actors[:5])}"
            else:
                available.append("actors")
                actors_ok = True
        except Exception as exc:
            errors["actors"] = f"{type(exc).__name__}:{exc}"

        # Route progress
        route_prog = None
        if route_tracker is not None:
            try:
                route_prog = route_tracker.update(ego_state, sim_time_s, discontinuity=discontinuity)
                if route_prog.status == "available":
                    available.append("route_progress")
            except Exception as exc:
                errors["route_progress"] = f"{type(exc).__name__}:{exc}"

        # Road and task context
        road_ctx = None
        task_ctx = None
        if road_index is not None:
            try:
                road_ctx, task_ctx, ctx_errors = road_index.sample(
                    ego_vehicle,
                    ego_state,
                    prev_ego_state,
                    nearby_states,
                    objective_active=objective_active,
                    sensor_wait=sensor_wait,
                    actors_available=actors_ok,
                )
                errors.update(ctx_errors)
                if ctx_errors:
                    errors.setdefault("road_context", ";".join(sorted(ctx_errors)))
                elif road_ctx is not None:
                    available.append("road_context")
                if task_ctx is not None:
                    available.append("task_context")
            except Exception as exc:
                errors["road_context"] = f"{type(exc).__name__}:{exc}"

        frame_ev = FrameEvidence(
            run_id=run_id,
            sim_frame=int(sim_frame),
            sim_time_s=float(sim_time_s),
            ego=ego_state,
            actors=tuple(nearby_states),
            available_channels=tuple(available),
            phase=phase,
            longitudinal_acceleration_mps2=long_accel,
            brake=brake_val,
            discontinuity=discontinuity,
            task_context=task_ctx,
            route_progress=route_prog,
            road_context=road_ctx,
            reverse=reverse_val,
            logical_tick=logical_tick,
            substep=substep,
            throttle=throttle_val,
            steer=steer_val,
        )
        return frame_ev, errors

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
        self._lock = threading.Lock()
        self._queue: deque[dict[str, Any]] = deque()
        self._phase: str = "evaluation"
        self._seq: int = 0
        self.attached: bool = False
        self.attach_error: str | None = None
        self.destroy_error: str | None = None

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def attach(self, world: Any, ego_vehicle: Any) -> Any:
        try:
            blueprint = world.get_blueprint_library().find("sensor.other.collision")
            sensor = world.spawn_actor(blueprint, self._identity_transform(world), attach_to=ego_vehicle)
        except Exception as exc:
            self.attach_error = f"{type(exc).__name__}:{exc}"
            self.attached = False
            raise
        weak_self = self

        def _on_collision(event: Any) -> None:
            impulse = event.normal_impulse
            ix, iy, iz = float(impulse.x), float(impulse.y), float(impulse.z)
            intensity = math.sqrt(ix ** 2 + iy ** 2 + iz ** 2)
            other_actor = event.other_actor
            timestamp = getattr(event, "timestamp", None)
            ts_val = float(timestamp) if isinstance(timestamp, (int, float)) and math.isfinite(timestamp) else None
            with weak_self._lock:
                weak_self._seq += 1
                seq = weak_self._seq
                phase = weak_self._phase
                weak_self.events.append(
                    CollisionEvent(
                        frame=int(event.frame),
                        actor_id=int(other_actor.id),
                        actor_type=str(other_actor.type_id),
                        intensity=float(intensity),
                        injected_actor=int(other_actor.id) in weak_self.injected_actor_ids,
                        timestamp_s=ts_val,
                        impulse_vector=(ix, iy, iz),
                        phase=phase,
                    )
                )
                weak_self._queue.append(
                    {
                        "frame": int(event.frame),
                        "timestamp_s": ts_val,
                        "actor_id": str(other_actor.id),
                        "actor_type": str(other_actor.type_id),
                        "impulse_ns": (ix, iy, iz),
                        "phase": phase,
                        "callback_id": f"col-{seq}-{int(event.frame)}",
                    }
                )

        sensor.listen(_on_collision)
        self._sensor = sensor
        self.attached = True
        return sensor

    def has_evaluation_collision(self, frame_phases: dict[int, str] | None = None) -> bool:
        phases = frame_phases or {}
        with self._lock:
            for ev in self.events:
                if phases.get(ev.frame, ev.phase) == "evaluation":
                    return True
        return False

    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    def stop_sensor(self) -> None:
        if self._sensor is not None:
            try:
                self._sensor.stop()
            except Exception as exc:
                self.destroy_error = f"stop:{type(exc).__name__}:{exc}"

    def drain_events(
        self,
        run_id: str,
        *,
        frame_times: dict[int, float] | None = None,
        frame_phases: dict[int, str] | None = None,
        fallback_time_s: float = 0.0,
    ) -> list[SensorEvidence]:
        items: list[dict[str, Any]] = []
        with self._lock:
            while self._queue:
                items.append(self._queue.popleft())
        times = frame_times or {}
        phases = frame_phases or {}
        out: list[SensorEvidence] = []
        for item in items:
            # Prefer frame-aligned simulation timestamp & phase to prevent async callback drift
            t_s = times.get(item["frame"])
            if t_s is None:
                t_s = item["timestamp_s"] if item["timestamp_s"] is not None else fallback_time_s
            phase = phases.get(item["frame"], item["phase"])
            out.append(
                SensorEvidence(
                    run_id=run_id,
                    sim_frame=item["frame"],
                    sim_time_s=float(max(0.0, t_s)),
                    channel="collision",
                    actor_id=item["actor_id"],
                    phase=phase,
                    impulse_ns=item["impulse_ns"],
                    callback_id=item["callback_id"],
                    metadata={"actor_type": item["actor_type"]},
                )
            )
        return out

    def destroy(self) -> None:
        if self._sensor is not None:
            try:
                self._sensor.stop()
            except Exception as exc:
                self.destroy_error = f"stop:{type(exc).__name__}:{exc}"
            try:
                self._sensor.destroy()
            except Exception as exc:
                self.destroy_error = f"destroy:{type(exc).__name__}:{exc}"
            self._sensor = None

    @staticmethod
    def _identity_transform(world: Any) -> Any:
        carla = type(world).__module__.split(".")[0]
        module = __import__(carla)
        return module.Transform()


class LaneInvasionOracle:
    """Captures immutable lane-invasion callbacks without querying later world state."""

    def __init__(self) -> None:
        self._sensor: Any | None = None
        self._lock = threading.Lock()
        self._queue: deque[dict[str, Any]] = deque()
        self._phase: str = "evaluation"
        self._seq: int = 0
        self.attached: bool = False
        self.attach_error: str | None = None
        self.destroy_error: str | None = None
        self._ego_actor_id: str = "ego"

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def attach(self, world: Any, ego_vehicle: Any) -> Any:
        self._ego_actor_id = str(getattr(ego_vehicle, "id", "ego"))
        try:
            blueprint = world.get_blueprint_library().find("sensor.other.lane_invasion")
            sensor = world.spawn_actor(blueprint, CollisionOracle._identity_transform(world), attach_to=ego_vehicle)
        except Exception as exc:
            self.attach_error = f"{type(exc).__name__}:{exc}"
            self.attached = False
            raise
        weak_self = self

        def _on_invasion(event: Any) -> None:
            timestamp = getattr(event, "timestamp", None)
            ts_val = float(timestamp) if isinstance(timestamp, (int, float)) and math.isfinite(timestamp) else None
            raw_markings = getattr(event, "crossed_lane_markings", ()) or ()
            markings = tuple(str(getattr(m, "type", m)).split(".")[-1] for m in raw_markings)
            with weak_self._lock:
                weak_self._seq += 1
                seq = weak_self._seq
                phase = weak_self._phase
                weak_self._queue.append(
                    {
                        "frame": int(event.frame),
                        "timestamp_s": ts_val,
                        "markings": markings,
                        "phase": phase,
                        "callback_id": f"lane-{seq}-{int(event.frame)}",
                    }
                )

        sensor.listen(_on_invasion)
        self._sensor = sensor
        self.attached = True
        return sensor

    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    def stop_sensor(self) -> None:
        if self._sensor is not None:
            try:
                self._sensor.stop()
            except Exception as exc:
                self.destroy_error = f"stop:{type(exc).__name__}:{exc}"

    def drain_events(
        self,
        run_id: str,
        *,
        frame_times: dict[int, float] | None = None,
        frame_phases: dict[int, str] | None = None,
        fallback_time_s: float = 0.0,
    ) -> list[SensorEvidence]:
        items: list[dict[str, Any]] = []
        with self._lock:
            while self._queue:
                items.append(self._queue.popleft())
        times = frame_times or {}
        phases = frame_phases or {}
        out: list[SensorEvidence] = []
        for item in items:
            t_s = times.get(item["frame"])
            if t_s is None:
                t_s = item["timestamp_s"] if item["timestamp_s"] is not None else fallback_time_s
            phase = phases.get(item["frame"], item["phase"])
            out.append(
                SensorEvidence(
                    run_id=run_id,
                    sim_frame=item["frame"],
                    sim_time_s=float(max(0.0, t_s)),
                    channel="lane_invasion",
                    actor_id=self._ego_actor_id,
                    phase=phase,
                    callback_id=item["callback_id"],
                    lane_markings=item["markings"],
                )
            )
        return out

    def destroy(self) -> None:
        if self._sensor is not None:
            try:
                self._sensor.stop()
            except Exception as exc:
                self.destroy_error = f"stop:{type(exc).__name__}:{exc}"
            try:
                self._sensor.destroy()
            except Exception as exc:
                self.destroy_error = f"destroy:{type(exc).__name__}:{exc}"
            self._sensor = None
