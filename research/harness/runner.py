from __future__ import annotations

import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.harness.agents import make_agent
from research.harness.carla_utils import (
    apply_world_settings,
    connect_client,
    ensure_output_dir,
    import_carla,
    load_world,
    resolve_weather,
    restore_world_settings,
)
from research.harness.config import AppConfig
from research.harness.interaction_geometry import footprint_clearance
from research.harness.models import RunResult, ScenarioSpec
from research.harness.observers.base import NoOpObserver, RunObserver
from research.harness.oracles import CollisionOracle, LaneInvasionOracle, SafetyOracle
from research.harness.outcome_config import DEFAULT_PROFILE, load_outcome_profile
from research.harness.outcome_io import EvidenceWriter, atomic_json, iter_evidence, write_outcome
from research.harness.outcome_models import CaptureManifest, ChannelCoverage, FrameEvidence, ScenarioChecks, SensorEvidence
from research.harness.road_context import RoadContextIndex
from research.harness.route_progress import RouteProgressTracker
from research.harness.safety_outcomes import OutcomeEvaluator
from research.harness.scenario_runtime import build_scenario_controller


def _vehicle_control_to_dict(control: Any) -> dict[str, Any]:
    return {
        "throttle": float(getattr(control, "throttle", 0.0)),
        "steer": float(getattr(control, "steer", 0.0)),
        "brake": float(getattr(control, "brake", 0.0)),
        "hand_brake": bool(getattr(control, "hand_brake", False)),
        "reverse": bool(getattr(control, "reverse", False)),
        "manual_gear_shift": bool(getattr(control, "manual_gear_shift", False)),
        "gear": int(getattr(control, "gear", 0)),
    }


def _speed_mps(ego_vehicle: Any) -> float:
    velocity = ego_vehicle.get_velocity()
    return float((velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5)


def _full_brake_control(carla: Any) -> Any:
    control = carla.VehicleControl()
    control.brake = 1.0
    return control


def _control_cadence(config: AppConfig) -> tuple[float, int]:
    logical_delta = config.harness.fixed_delta_seconds
    native_pcla = config.agent.kind == "pcla" and config.agent.pcla_agent_name in {
        "if_if", "lav_lav", "lav_fast", "simlingo_simlingo",
    }
    native_transfuser = config.agent.kind == "leaderboard-module" and config.agent.module_name in {"submission_agent", "transfuser_agent"}
    if not (native_pcla or native_transfuser):
        return logical_delta, 1
    # These agents' PID, tracking and stop timers assume the upstream 20 Hz
    # cadence. Preserve campaign/observer ticks while advancing native steps.
    native_delta = 0.05
    substeps = round(logical_delta / native_delta)
    if substeps < 1 or abs(substeps * native_delta - logical_delta) > 1e-9:
        raise ValueError("Native model logical timestep must be a positive multiple of 0.05 seconds")
    if not config.harness.synchronous_mode:
        raise ValueError("Native models require synchronous simulation at 20 Hz")
    return native_delta, substeps


def _spawn_actor_with_retries(world: Any, blueprint: Any, transform: Any, attempts: int = 10, ticks_between_attempts: int = 2) -> Any:
    actor = world.try_spawn_actor(blueprint, transform)
    if actor is not None:
        return actor

    retry_transform = transform
    retry_transform.location.z += 0.25
    for _ in range(attempts - 1):
        for _ in range(ticks_between_attempts):
            world.tick()
        actor = world.try_spawn_actor(blueprint, retry_transform)
        if actor is not None:
            return actor
    return None


def _clear_dynamic_actors_near_spawn(world: Any, spawn_location: Any, radius_m: float = 4.0) -> None:
    for actor in world.get_actors():
        type_id = str(getattr(actor, "type_id", ""))
        if not (type_id.startswith("vehicle.") or type_id.startswith("walker.")):
            continue
        try:
            actor_location = actor.get_location()
        except RuntimeError:
            continue
        if actor_location.distance(spawn_location) > radius_m:
            continue
        try:
            actor.destroy()
        except RuntimeError:
            continue


def _extract_route_points(route_waypoints: Any) -> list[tuple[float, float]]:
    if not isinstance(route_waypoints, (list, tuple)):
        return []
    pts: list[tuple[float, float]] = []
    for item in route_waypoints:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                pts.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
        elif isinstance(item, dict):
            src = item.get("location") if isinstance(item.get("location"), dict) else item
            if "x" in src and "y" in src:
                try:
                    pts.append((float(src["x"]), float(src["y"])))
                except (TypeError, ValueError):
                    continue
    return pts


def _initial_route_anchor(points: list[tuple[float, float]], ego_x: float, ego_y: float, max_offset_m: float = 3.0) -> tuple[float | None, float | None]:
    """Find initial nearest along-track s_m on a polyline within max_offset_m."""
    best_dist = float("inf")
    best_s: float | None = None
    total = 0.0
    for a, b in zip(points, points[1:]):
        seg_len = math.dist(a, b)
        if seg_len <= 0.0:
            continue
        dx, dy = b[0] - a[0], b[1] - a[1]
        t = max(0.0, min(1.0, ((ego_x - a[0]) * dx + (ego_y - a[1]) * dy) / (seg_len * seg_len)))
        dist = math.hypot(ego_x - (a[0] + t * dx), ego_y - (a[1] + t * dy))
        if dist < best_dist:
            best_dist = dist
            best_s = total + t * seg_len
        total += seg_len
    if best_s is not None and best_dist <= max_offset_m:
        return best_s, best_dist
    return None, (best_dist if math.isfinite(best_dist) else None)


def _snapshot_stamp(world: Any, tick_frame: Any, fallback_frame: int, fallback_time_s: float) -> tuple[int, float, bool]:
    """Match world.tick() frame ID to world.get_snapshot(); report whether fallback clock was used."""
    expected_frame = int(tick_frame) if isinstance(tick_frame, int) and not isinstance(tick_frame, bool) and tick_frame >= 0 else None
    try:
        snap = world.get_snapshot()
        snap_frame = int(getattr(snap, "frame", -1))
        ts = getattr(snap, "timestamp", None)
        elapsed = getattr(ts, "elapsed_seconds", None) if ts is not None else None
        if snap_frame >= 0 and elapsed is not None and math.isfinite(float(elapsed)) and float(elapsed) >= 0.0:
            frame_id = expected_frame if expected_frame is not None else snap_frame
            mismatch = expected_frame is not None and snap_frame != expected_frame
            return frame_id, float(elapsed), mismatch
    except Exception:
        pass
    return (expected_frame if expected_frame is not None else fallback_frame), fallback_time_s, True


def _check_initial_geometry(ego_vehicle: Any, spawned_actors: list[Any], collision_oracle: CollisionOracle) -> tuple[bool | None, list[str]]:
    """Verify non-overlapping initial bounding-box geometry and absence of setup contact."""
    refs: list[str] = []
    ego_state = SafetyOracle._actor_to_state(ego_vehicle, role="ego")
    if ego_state is None:
        return None, ["ego_initial_geometry_unreadable"]
    if collision_oracle.attached and collision_oracle.events:
        return False, ["setup_collision_observed"]
    for idx, actor in enumerate(spawned_actors):
        other = SafetyOracle._actor_to_state(actor, role=f"spawned_{idx}")
        if other is None:
            return None, [f"spawned_actor_{idx}_geometry_unreadable"]
        gap = footprint_clearance(ego_state, other)
        if gap is not None and gap <= 0.0:
            return False, [f"initial_overlap_with_{other.actor_id}"]
    refs.append("initial_footprint_clearance_verified")
    if collision_oracle.attached:
        refs.append("collision_sensor_clear_at_setup")
    return True, refs


class _LiveOutcomeCapture:
    """Coordinates physics-step v2 evidence capture, streaming JSONL and replay-verified finalization."""

    def __init__(self, config: AppConfig, scenario: ScenarioSpec, run_id: str, physics_delta_s: float) -> None:
        self.config = config
        self.scenario = scenario
        self.run_id = run_id
        self.physics_delta_s = float(physics_delta_s)
        self.profile = load_outcome_profile(config.outcomes.profile_path or DEFAULT_PROFILE)
        out_dir = Path(config.outcomes.evidence_output_dir or (config.harness.output_dir / "evidence"))
        out_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = out_dir / f"{run_id}.jsonl"
        self.outcome_path = out_dir / f"{run_id}.outcome.json"
        self.writer = EvidenceWriter(self.trace_path, run_id)
        self.road_index: RoadContextIndex | None = None
        self.route_tracker: RouteProgressTracker | None = None
        self._route_points: list[tuple[float, float]] = []
        self.frame_times: dict[int, float] = {}
        self.frame_phases: dict[int, str] = {}
        self.eval_start_s: float | None = None
        self.eval_end_s: float | None = None
        self.eval_frame_count: int = 0
        self.eval_step_attempts: int = 0
        self._prev_eval_time_s: float | None = None
        self.observed_max_dt_s: float = 0.0
        self.last_frame_id: int = -1
        self.last_time_s: float = -1.0
        self.prev_ego_state = None
        self.latest_frame: FrameEvidence | None = None
        self.channel_errors: dict[str, list[str]] = {}
        self.channel_intervals: dict[str, list[float]] = {}
        self.channel_frame_counts: dict[str, int] = {}
        self.sensor_write_errors: dict[str, list[str]] = {}
        self.sensor_unindexed_channels: set[str] = set()
        self.eval_sensor_min_s: float | None = None
        self.eval_sensor_max_s: float | None = None
        self.errors: list[dict[str, Any]] = []
        self.instrumentation_wall_seconds: float = 0.0
        self.finalized: bool = False
        self.manifest_path: Path | None = None
        self.route_anchor_provenance: dict[str, Any] = {}
        self.scenario_checks = ScenarioChecks(
            configuration_feasible=True,
            initial_geometry_feasible=None,
            required_actors_spawned=None,
            route_feasible=None,
            controller_ready=None,
        )

    def init_world_context(self, world: Any, route_waypoints: Any, ego_vehicle: Any | None = None) -> None:
        pts = _extract_route_points(route_waypoints)
        self._route_points = pts
        if self.config.outcomes.capture_road_context:
            self.road_index = RoadContextIndex.from_world(world, route=pts or None)
        if self.config.outcomes.capture_route_progress and len(pts) >= 2:
            try:
                initial_s = 0.0
                if ego_vehicle is not None:
                    loc = ego_vehicle.get_location()
                    anchor_s, anchor_err = _initial_route_anchor(pts, float(loc.x), float(loc.y))
                    if anchor_s is not None:
                        initial_s = anchor_s
                    self.route_anchor_provenance = {"initial_s_m": initial_s, "anchor_offset_m": anchor_err}
                self.route_tracker = RouteProgressTracker(
                    pts,
                    initial_s_m=initial_s,
                    max_sample_interval_s=max(self.profile.value("max_sample_interval_s"), self.physics_delta_s + 0.01),
                )
            except Exception as exc:
                self.errors.append({"channel": "route_progress", "error": f"{type(exc).__name__}:{exc}"})

    def anchor_route_for_evaluation(self, ego_vehicle: Any) -> None:
        """Re-anchor route tracker to ego's exact position right before evaluation starts."""
        if self.route_tracker is None or len(self._route_points) < 2:
            return
        try:
            loc = ego_vehicle.get_location()
            anchor_s, anchor_err = _initial_route_anchor(self._route_points, float(loc.x), float(loc.y))
            if anchor_s is not None:
                self.route_tracker.reset(anchor_s)
                self.route_anchor_provenance = {
                    "anchored_at_evaluation_start": True,
                    "initial_s_m": anchor_s,
                    "anchor_offset_m": anchor_err,
                }
        except Exception as exc:
            self.errors.append({"channel": "route_progress_anchor", "error": f"{type(exc).__name__}:{exc}"})

    def _next_stamp(self, world: Any, tick_frame: Any = None) -> tuple[int, float, bool]:
        fallback_frame = max(0, self.last_frame_id + 1)
        fallback_time = 0.0 if self.last_time_s < 0 else round(self.last_time_s + self.physics_delta_s, 8)
        frame, sim_time, clock_anomaly = _snapshot_stamp(world, tick_frame, fallback_frame, fallback_time)
        if self.last_frame_id >= 0 and frame <= self.last_frame_id:
            frame = self.last_frame_id + 1
            clock_anomaly = True
        if self.last_time_s >= 0 and sim_time <= self.last_time_s:
            sim_time = round(self.last_time_s + self.physics_delta_s, 8)
            clock_anomaly = True
        self.last_frame_id = frame
        self.last_time_s = sim_time
        self.frame_times[frame] = sim_time
        return frame, sim_time, clock_anomaly

    def observe_physics_step(
        self,
        world: Any,
        ego_vehicle: Any,
        safety_oracle: SafetyOracle,
        collision_oracle: CollisionOracle,
        lane_oracle: LaneInvasionOracle,
        *,
        phase: str,
        tick_frame: Any = None,
        control: Any = None,
        logical_tick: int | None = None,
        substep: int | None = None,
        actor_roles: dict[int, str] | None = None,
        objective_active: bool = True,
        sensor_wait: bool = False,
        discontinuity: bool = False,
    ) -> tuple[int, float]:
        t0 = time.perf_counter()
        frame, sim_time, clock_anomaly = self._next_stamp(world, tick_frame=tick_frame)
        self.frame_phases[frame] = phase
        if phase == "evaluation":
            self.eval_step_attempts += 1
        step_discontinuity = bool(discontinuity or clock_anomaly)
        if clock_anomaly:
            self.errors.append({"phase": phase, "sim_frame": frame, "error": "snapshot_clock_fallback_or_mismatch"})
        try:
            fe, errs = safety_oracle.capture_frame_evidence(
                world,
                ego_vehicle,
                run_id=self.run_id,
                sim_frame=frame,
                sim_time_s=sim_time,
                phase=phase,
                scan_radius_m=self.profile.value("scan_radius_m"),
                control=control,
                logical_tick=logical_tick,
                substep=substep,
                actor_roles=actor_roles,
                road_index=self.road_index if self.config.outcomes.capture_road_context else None,
                route_tracker=self.route_tracker if self.config.outcomes.capture_route_progress else None,
                prev_ego_state=self.prev_ego_state,
                objective_active=objective_active if self.config.outcomes.capture_task_context else None,
                sensor_wait=sensor_wait if self.config.outcomes.capture_task_context else None,
                discontinuity=step_discontinuity,
            )
            for ch, msg in errs.items():
                self.channel_errors.setdefault(ch, []).append(msg)
            if fe is not None:
                # Persist to JSONL FIRST before updating in-memory state so live & replay agree
                self.writer.append(fe)
                self.prev_ego_state = fe.ego
                self.latest_frame = fe
                if phase == "evaluation":
                    self.eval_frame_count += 1
                    if self.eval_start_s is None:
                        self.eval_start_s = sim_time
                    if self._prev_eval_time_s is not None:
                        self.observed_max_dt_s = max(self.observed_max_dt_s, sim_time - self._prev_eval_time_s)
                    self._prev_eval_time_s = sim_time
                    self.eval_end_s = sim_time
                    for ch in fe.available_channels:
                        bounds = self.channel_intervals.get(ch)
                        if bounds is None:
                            self.channel_intervals[ch] = [sim_time, sim_time]
                        else:
                            bounds[1] = sim_time
                        self.channel_frame_counts[ch] = self.channel_frame_counts.get(ch, 0) + 1
            self.drain_sensors(collision_oracle, lane_oracle, fallback_time_s=sim_time)
        except Exception as exc:
            self.errors.append({"phase": phase, "sim_frame": frame, "error": f"{type(exc).__name__}:{exc}"})
        finally:
            self.instrumentation_wall_seconds += time.perf_counter() - t0
        return frame, sim_time

    def drain_sensors(
        self,
        collision_oracle: CollisionOracle,
        lane_oracle: LaneInvasionOracle,
        *,
        fallback_time_s: float,
    ) -> int:
        drained = 0
        for ch_name, oracle in (("collision", collision_oracle), ("lane_invasion", lane_oracle)):
            for ev in oracle.drain_events(
                self.run_id,
                frame_times=self.frame_times,
                frame_phases=self.frame_phases,
                fallback_time_s=fallback_time_s,
            ):
                try:
                    self.writer.append(ev)
                except Exception as exc:
                    msg = f"sensor_append_failed:{type(exc).__name__}:{exc}"
                    self.sensor_write_errors.setdefault(ch_name, []).append(msg)
                    self.errors.append({"phase": ev.phase, "channel": ch_name, "error": msg})
                    continue
                drained += 1
                if ev.phase == "evaluation":
                    if ev.sim_frame not in self.frame_times:
                        self.sensor_unindexed_channels.add(ch_name)
                    if self.eval_sensor_min_s is None or ev.sim_time_s < self.eval_sensor_min_s:
                        self.eval_sensor_min_s = ev.sim_time_s
                    if self.eval_sensor_max_s is None or ev.sim_time_s > self.eval_sensor_max_s:
                        self.eval_sensor_max_s = ev.sim_time_s
        return drained

    def _settle_and_drain_sensors(
        self,
        collision_oracle: CollisionOracle,
        lane_oracle: LaneInvasionOracle,
    ) -> bool:
        """Stop sensors and perform bounded quiescent drain without advancing physics."""
        collision_oracle.stop_sensor()
        lane_oracle.stop_sensor()
        timeout_s = max(0.0, float(self.config.outcomes.callback_drain_timeout_s))
        deadline = time.perf_counter() + timeout_s
        fallback_t = max(0.0, self.last_time_s)
        late_unsettled = False
        self.drain_sensors(collision_oracle, lane_oracle, fallback_time_s=fallback_t)
        if timeout_s > 0.0 and (collision_oracle.attached or lane_oracle.attached):
            # Drain across the full bounded callback_drain_timeout_s window after stop_sensor()
            # so delayed async callbacks arriving within the configured timeout are captured.
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                time.sleep(min(0.005, remaining))
                self.drain_sensors(collision_oracle, lane_oracle, fallback_time_s=fallback_t)
            self.drain_sensors(collision_oracle, lane_oracle, fallback_time_s=fallback_t)
        if collision_oracle.pending_count() > 0 or lane_oracle.pending_count() > 0:
            late_unsettled = True
        return late_unsettled

    def finalize(
        self,
        *,
        execution_status: str,
        termination_reason: str,
        reached_goal: bool | None,
        collision_oracle: CollisionOracle,
        lane_oracle: LaneInvasionOracle,
        extra_provenance: dict[str, Any] | None = None,
    ) -> tuple[Path | None, Path | None, dict[str, Any] | None]:
        if self.finalized:
            return self.manifest_path, self.outcome_path, None
        late_unsettled = False
        try:
            late_unsettled = self._settle_and_drain_sensors(collision_oracle, lane_oracle)
        except Exception as exc:
            self.errors.append({"phase": "finalize_drain", "error": f"{type(exc).__name__}:{exc}"})
            late_unsettled = True

        frame_start_s = self.eval_start_s if self.eval_start_s is not None else 0.0
        frame_end_s = self.eval_end_s if self.eval_end_s is not None else frame_start_s
        start_s = frame_start_s
        end_s = frame_end_s
        sensor_expanded_bounds = False
        if self.eval_sensor_min_s is not None:
            if self.eval_frame_count == 0 or self.eval_sensor_min_s < start_s:
                start_s = self.eval_sensor_min_s
                if self.eval_frame_count > 0:
                    sensor_expanded_bounds = True
        if self.eval_sensor_max_s is not None:
            if self.eval_frame_count == 0 or self.eval_sensor_max_s > end_s:
                end_s = self.eval_sensor_max_s
                if self.eval_frame_count > 0:
                    sensor_expanded_bounds = True
        has_eval_frames = self.eval_frame_count > 0

        channels: dict[str, ChannelCoverage] = {}
        for ch, bounds in self.channel_intervals.items():
            has_err = bool(self.channel_errors.get(ch))
            all_steps_present = (
                self.channel_frame_counts.get(ch, 0) == self.eval_step_attempts
                and self.eval_frame_count == self.eval_step_attempts
            )
            if (
                has_eval_frames
                and not has_err
                and all_steps_present
                and bounds[0] <= start_s + 1e-9
                and bounds[1] >= end_s - 1e-9
            ):
                channels[ch] = ChannelCoverage("complete", start_s, end_s)
            else:
                reasons = list(self.channel_errors.get(ch, ()))
                if not all_steps_present:
                    reasons.append("missing_frame_samples")
                if sensor_expanded_bounds:
                    reasons.append("sensor_timestamp_outside_frame_interval")
                channels[ch] = ChannelCoverage(
                    "partial",
                    bounds[0],
                    bounds[1],
                    reasons=tuple(reasons[:4]) or ("partial_interval",),
                )

        for ch_name, oracle in (("collision", collision_oracle), ("lane_invasion", lane_oracle)):
            write_errs = self.sensor_write_errors.get(ch_name, ())
            unindexed = ch_name in self.sensor_unindexed_channels
            if not has_eval_frames or execution_status in {"failed", "not_executed", "dry_run"}:
                if self.eval_sensor_min_s is not None and oracle.attached:
                    channels[ch_name] = ChannelCoverage(
                        "partial", start_s, end_s,
                        reasons=("no_evaluation_frames_observed",),
                    )
                else:
                    channels[ch_name] = ChannelCoverage("unavailable", reasons=("no_evaluation_frames_observed",))
            elif (
                oracle.attached
                and oracle.attach_error is None
                and oracle.destroy_error is None
                and not late_unsettled
                and not write_errs
                and not unindexed
                and not sensor_expanded_bounds
            ):
                channels[ch_name] = ChannelCoverage("complete", start_s, end_s)
            elif oracle.attached and oracle.attach_error is None:
                reasons = [
                    r for r in (
                        oracle.destroy_error,
                        "late_or_unsettled_callbacks" if late_unsettled else None,
                        "unindexed_sensor_callback" if unindexed else None,
                        "sensor_timestamp_outside_frame_interval" if sensor_expanded_bounds else None,
                        *write_errs[:2],
                    ) if r
                ]
                channels[ch_name] = ChannelCoverage(
                    "partial", start_s, end_s,
                    reasons=tuple(reasons) or ("partial_sensor_coverage",),
                )
            else:
                channels[ch_name] = ChannelCoverage(
                    "unavailable",
                    reasons=(oracle.attach_error or f"{ch_name}_sensor_not_attached",),
                )

        observed_interval = self.observed_max_dt_s if self.observed_max_dt_s > 0.0 else self.physics_delta_s
        prov: dict[str, Any] = {
            "capture_source": "carla_live_runner",
            "physics_delta_s": self.physics_delta_s,
            "logical_delta_s": float(self.config.harness.fixed_delta_seconds),
            "scenario_id": self.scenario.scenario_id,
            "town": self.scenario.town,
            "agent_kind": self.config.agent.kind,
            "seed": int(self.config.harness.seed),
            "eval_step_attempts": self.eval_step_attempts,
            "eval_frame_count": self.eval_frame_count,
            "instrumentation_wall_seconds": self.instrumentation_wall_seconds,
            "capture_errors": list(self.errors),
        }
        scripted_case_id = (self.config.agent.init_kwargs or {}).get("case_id")
        if isinstance(scripted_case_id, str) and scripted_case_id:
            prov["case_id"] = scripted_case_id
        if self.route_tracker is not None:
            prov["route_tracker"] = {**self.route_tracker.provenance(), **self.route_anchor_provenance}
        if extra_provenance:
            prov.update(extra_provenance)

        manifest = CaptureManifest(
            run_id=self.run_id,
            start_s=start_s,
            end_s=end_s,
            execution_status=execution_status,
            termination_reason=termination_reason,
            channels=channels,
            reached_goal=reached_goal,
            completion_deadline_s=self.config.outcomes.completion_deadline_s,
            scan_radius_m=self.profile.value("scan_radius_m"),
            projection_horizon_s=self.profile.value("projection_horizon_s"),
            max_sample_interval_s=observed_interval,
            provenance=prov,
            scenario_checks=self.scenario_checks,
        )
        # Finish trace file, then evaluate the PERSISTED records so live == replay strictly!
        self.manifest_path = self.writer.finish(manifest)
        evaluator = OutcomeEvaluator(self.profile)
        for rec in iter_evidence(self.trace_path):
            if isinstance(rec, FrameEvidence):
                evaluator.observe(rec)
            elif isinstance(rec, SensorEvidence):
                evaluator.record_sensor_event(rec)
        outcome = evaluator.finish(manifest)
        write_outcome(self.outcome_path, outcome)
        self.finalized = True
        return self.manifest_path, self.outcome_path, outcome.to_dict()


class HarnessRunner:
    def __init__(self, config: AppConfig, observers: list[RunObserver] | None = None) -> None:
        self.config = config
        self.observers = observers or [NoOpObserver()]

    def run(self, scenario: ScenarioSpec) -> RunResult:
        if self.config.outcomes.enabled and self.config.outcomes.schema_version != 2:
            raise ValueError(f"Outcome capture requires schema_version=2, got {self.config.outcomes.schema_version}")
        run_id = f"{scenario.scenario_id}-{uuid.uuid4().hex[:12]}"
        if self.config.run.dry_run:
            return RunResult(
                scenario_id=scenario.scenario_id,
                town=scenario.town,
                agent_kind=self.config.agent.kind,
                succeeded=True,
                dry_run=True,
                ticks_executed=0,
                reached_goal=False,
                terminated_by_collision=False,
                collision_count=0,
                notes=["Dry-run completed. No CARLA connection attempted."],
                metadata={"weather_preset": scenario.weather_preset},
                run_id=run_id,
                execution_status="dry_run",
                termination_reason="dry_run",
                observed_duration_s=0.0,
            )

        return self._run_live(scenario, run_id=run_id)

    def _run_live(self, scenario: ScenarioSpec, *, run_id: str | None = None) -> RunResult:
        run_id = run_id or f"{scenario.scenario_id}-{uuid.uuid4().hex[:12]}"
        run_started = time.perf_counter()
        native_delta, control_substeps = _control_cadence(self.config)
        capture_enabled = bool(
            self.config.outcomes.schema_version == 2
            and (self.config.outcomes.enabled or self.config.outcomes.evidence_output_dir is not None)
        )
        capture: _LiveOutcomeCapture | None = None

        collision_oracle = CollisionOracle()
        lane_oracle = LaneInvasionOracle()
        safety_oracle = SafetyOracle(tick_seconds=self.config.harness.fixed_delta_seconds)
        scenario_controller = None
        actors: list[Any] = []
        actor_roles: dict[int, str] = {}
        reached_goal = False
        terminated_by_collision = False
        ticks_executed = 0
        physics_steps_executed = 0
        notes: list[str] = []
        world = None
        original_settings = None
        context: dict[str, Any] = {
            "scenario": scenario,
            "run_id": run_id,
            "actor_roles": actor_roles,
            "scenario_trigger_events": [],
        }
        termination_reason = "horizon_exhausted"
        execution_status = "completed"
        phase_stage = "setup"

        try:
            if capture_enabled:
                capture = _LiveOutcomeCapture(self.config, scenario, run_id, native_delta)
            scenario_controller = build_scenario_controller(scenario)
            client = connect_client(
                self.config.harness.host,
                self.config.harness.port,
                self.config.harness.timeout_seconds,
            )
            world = load_world(client, scenario.town)
            if self.config.harness.reload_world:
                world = client.reload_world(False) if self.config.harness.keep_synchronous else client.reload_world()
            import random
            import numpy as np
            random.seed(self.config.harness.seed)
            np.random.seed(self.config.harness.seed)
            if self.config.agent.kind in {"autovla", "pcla", "leaderboard-module"}:
                import torch
                torch.manual_seed(self.config.harness.seed)
            original_settings = apply_world_settings(
                world,
                self.config.harness.synchronous_mode,
                native_delta,
            )
            context.update({"client": client, "world": world})

            resolve_weather(world, scenario.weather_preset)
            carla = import_carla()
            spawn_points = world.get_map().get_spawn_points()
            start_transform = spawn_points[scenario.ego_spawn_index]
            goal_transform = spawn_points[scenario.goal_spawn_index]
            _clear_dynamic_actors_near_spawn(world, start_transform.location)

            blueprint = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
            ego_vehicle = _spawn_actor_with_retries(world, blueprint, start_transform)
            if ego_vehicle is None:
                if capture is not None:
                    capture.scenario_checks = ScenarioChecks(
                        configuration_feasible=True,
                        initial_geometry_feasible=False,
                        required_actors_spawned=False,
                        execution_error="ego_spawn_failed",
                    )
                termination_reason = "scenario_setup_failed"
                raise RuntimeError("Failed to spawn ego vehicle for harness shakedown")
            actors.append(ego_vehicle)
            actor_roles[int(ego_vehicle.id)] = "ego"

            collision_oracle.set_phase("warmup")
            lane_oracle.set_phase("warmup")
            if self.config.sensors.attach_collision_sensor:
                actors.append(collision_oracle.attach(world, ego_vehicle))
            if capture is not None and self.config.sensors.attach_lane_invasion_sensor:
                try:
                    actors.append(lane_oracle.attach(world, ego_vehicle))
                except Exception as exc:
                    notes.append(f"Lane invasion sensor unavailable: {exc}")

            for _ in range(self.config.run.warmup_ticks):
                tf = world.tick()
                if capture is not None:
                    capture.observe_physics_step(
                        world, ego_vehicle, safety_oracle, collision_oracle, lane_oracle,
                        phase="warmup", tick_frame=tf, actor_roles=actor_roles,
                    )

            phase_stage = "agent"
            agent = make_agent(world, ego_vehicle, self.config.agent, client=client)
            agent.set_destination(goal_transform.location)
            phase_stage = "setup"
            route_preview = agent.get_route_preview() if hasattr(agent, "get_route_preview") else []
            route_waypoints = (
                agent.get_route_waypoints()
                if hasattr(agent, "get_route_waypoints")
                else route_preview
            )
            if capture is not None:
                capture.init_world_context(world, route_waypoints, ego_vehicle=ego_vehicle)

            context.update(
                {
                    "ego_vehicle": ego_vehicle,
                    "agent": agent,
                    "collision_oracle": collision_oracle,
                    "lane_invasion_oracle": lane_oracle,
                    "carla": carla,
                    "goal_location": goal_transform.location,
                    "route_preview": route_preview,
                    "route_waypoints": route_waypoints,
                    "scenario_notes": notes,
                    "actor_roles": actor_roles,
                }
            )
            scenario_controller.setup(context)
            collision_oracle.injected_actor_ids = {actor.id for actor in scenario_controller.spawned_actors}
            for idx, spawned in enumerate(scenario_controller.spawned_actors):
                sid = int(spawned.id)
                if sid not in actor_roles:
                    role_name = getattr(spawned, "attributes", {}).get("role_name") if hasattr(spawned, "attributes") else None
                    actor_roles[sid] = str(role_name) if role_name else f"scenario_actor_{idx}"
            actors.extend(scenario_controller.spawned_actors)

            if capture is not None:
                geom_ok, geom_refs = _check_initial_geometry(ego_vehicle, scenario_controller.spawned_actors, collision_oracle)
                req_spawned = bool(
                    scenario_controller.setup_succeeded
                    and len(scenario_controller.spawned_actors) >= scenario_controller.expected_actor_count
                )
                route_ok = True if len(_extract_route_points(route_waypoints)) >= 2 else None
                capture.scenario_checks = ScenarioChecks(
                    configuration_feasible=True,
                    initial_geometry_feasible=geom_ok,
                    required_actors_spawned=req_spawned,
                    route_feasible=route_ok,
                    controller_ready=bool(scenario_controller.setup_succeeded),
                    evidence_refs=tuple(["live_setup_checks", *geom_refs, *scenario_controller.spawn_failures]),
                )

            for observer in self.observers:
                observer.on_run_start(scenario, context)

            startup_hold_ticks = max(int(self.config.run.startup_hold_ticks), 0)
            if startup_hold_ticks:
                collision_oracle.set_phase("startup_hold")
                lane_oracle.set_phase("startup_hold")
                hold_control = _full_brake_control(carla)
                context["startup_hold_ticks"] = startup_hold_ticks
                context["startup_hold_applied"] = True
                for _ in range(startup_hold_ticks):
                    ego_vehicle.apply_control(hold_control)
                    tf = world.tick()
                    if capture is not None:
                        capture.observe_physics_step(
                            world, ego_vehicle, safety_oracle, collision_oracle, lane_oracle,
                            phase="startup_hold", tick_frame=tf, control=hold_control, actor_roles=actor_roles,
                        )
            else:
                context["startup_hold_ticks"] = 0
                context["startup_hold_applied"] = False

            if capture is not None:
                capture.anchor_route_for_evaluation(ego_vehicle)

            phase_stage = "evaluation"
            collision_oracle.set_phase("evaluation")
            lane_oracle.set_phase("evaluation")
            driving_started = time.perf_counter()
            control_wall_seconds = physics_wall_seconds = observer_wall_seconds = 0.0
            last_sim_frame: int | None = None
            last_sim_time_s: float | None = None

            for tick in range(scenario.max_ticks):
                for substep_idx in range(control_substeps):
                    step_started = time.perf_counter()
                    phase_stage = "agent"
                    control = agent.run_step()
                    phase_stage = "evaluation"
                    control_wall_seconds += time.perf_counter() - step_started
                    step_started = time.perf_counter()
                    ego_vehicle.apply_control(control)
                    tf = world.tick()
                    physics_steps_executed += 1
                    physics_wall_seconds += time.perf_counter() - step_started
                    teleport_disc = bool(context.pop("actor_teleport_discontinuity", False))
                    if capture is not None:
                        step_info = (
                            agent.get_debug_state()
                            if hasattr(agent, "get_debug_state")
                            else dict(getattr(agent, "last_step_info", {}))
                        )
                        is_sensor_wait = isinstance(step_info, dict) and step_info.get("mode") == "sensor-wait"
                        last_sim_frame, last_sim_time_s = capture.observe_physics_step(
                            world,
                            ego_vehicle,
                            safety_oracle,
                            collision_oracle,
                            lane_oracle,
                            phase="evaluation",
                            tick_frame=tf,
                            control=control,
                            logical_tick=tick,
                            substep=substep_idx,
                            actor_roles=actor_roles,
                            objective_active=not reached_goal,
                            sensor_wait=is_sensor_wait,
                            discontinuity=teleport_disc,
                        )
                    else:
                        last_sim_frame, last_sim_time_s, _ = _snapshot_stamp(
                            world,
                            tf,
                            (tick * control_substeps) + substep_idx,
                            round(((tick * control_substeps) + substep_idx) * native_delta, 8),
                        )
                    if (
                        self.config.run.stop_on_first_collision
                        and collision_oracle.has_evaluation_collision(capture.frame_phases if capture else None)
                    ):
                        break
                observer_started = time.perf_counter()
                context["last_control"] = _vehicle_control_to_dict(control)
                context["sim_frame"] = last_sim_frame
                context["sim_time_s"] = last_sim_time_s
                if capture is not None and capture.latest_frame is not None:
                    context["latest_frame_evidence"] = capture.latest_frame
                ticks_executed = tick + 1
                safety_oracle.tick(world, ego_vehicle, goal_transform.location)
                speed_mps = _speed_mps(ego_vehicle)
                ego_location = ego_vehicle.get_location()
                agent_step_info = (
                    agent.get_debug_state()
                    if hasattr(agent, "get_debug_state")
                    else dict(getattr(agent, "last_step_info", {}))
                )
                route_prog_dict = None
                if capture is not None and capture.route_tracker is not None:
                    rp = capture.route_tracker.summary()
                    route_prog_dict = {
                        "status": rp.status,
                        "s_m": rp.s_m,
                        "high_water_m": rp.high_water_m,
                        "total_m": rp.total_m,
                        "cross_track_m": rp.cross_track_m,
                        "continuity_id": rp.continuity_id,
                    }
                context["telemetry"] = {
                    "tick": ticks_executed,
                    "sim_frame": last_sim_frame,
                    "sim_time_s": last_sim_time_s,
                    "control": context["last_control"],
                    "speed_mps": speed_mps,
                    "speed_kph": speed_mps * 3.6,
                    "distance_to_goal_m": float(ego_location.distance(goal_transform.location)),
                    "location": {
                        "x": float(ego_location.x),
                        "y": float(ego_location.y),
                        "z": float(ego_location.z),
                    },
                    "agent_step": agent_step_info,
                    "safety": safety_oracle.snapshot(),
                    "route_arc_progress": route_prog_dict,
                }
                scenario_controller.on_tick(tick, context)

                for observer in self.observers:
                    observer.on_tick(tick, context)

                if (
                    self.config.run.stop_on_first_collision
                    and collision_oracle.has_evaluation_collision(capture.frame_phases if capture else None)
                ):
                    terminated_by_collision = True
                    termination_reason = "collision_stop"
                    eval_cols = [
                        e for e in collision_oracle.events
                        if (capture.frame_phases.get(e.frame, e.phase) if capture else e.phase) == "evaluation"
                    ]
                    first_collision = eval_cols[0] if eval_cols else collision_oracle.events[0]
                    notes.append(
                        f"Terminated on first collision at frame {first_collision.frame} with {first_collision.actor_type}."
                    )
                    break

                observer_wall_seconds += time.perf_counter() - observer_started
                if agent.done():
                    reached_goal = True
                    termination_reason = "goal_reached"
                    break

            succeeded = True
            if not reached_goal:
                notes.append("Goal not reached within allotted ticks.")

            trigger_events = list(context.get("scenario_trigger_events") or ())
            manifest_path_str: str | None = None
            outcome_path_str: str | None = None
            outcome_summary: dict[str, Any] | None = None
            observed_duration_s: float | None = None
            if capture is not None:
                m_path, o_path, outcome_summary = capture.finalize(
                    execution_status=execution_status,
                    termination_reason=termination_reason,
                    reached_goal=reached_goal,
                    collision_oracle=collision_oracle,
                    lane_oracle=lane_oracle,
                    extra_provenance={"scenario_trigger_events": trigger_events},
                )
                manifest_path_str = str(m_path) if m_path else None
                outcome_path_str = str(o_path) if o_path else None
                if capture.eval_start_s is not None and capture.eval_end_s is not None:
                    observed_duration_s = capture.eval_end_s - capture.eval_start_s

            result = RunResult(
                scenario_id=scenario.scenario_id,
                town=scenario.town,
                agent_kind=self.config.agent.kind,
                succeeded=succeeded,
                dry_run=False,
                ticks_executed=ticks_executed,
                reached_goal=reached_goal,
                terminated_by_collision=terminated_by_collision,
                collision_count=len(collision_oracle.events),
                collisions=list(collision_oracle.events),
                safety_metrics=safety_oracle.metrics,
                notes=notes,
                metadata={
                    "wall_seconds": time.perf_counter() - run_started,
                    "driving_wall_seconds": time.perf_counter() - driving_started,
                    "control_wall_seconds": control_wall_seconds,
                    "physics_wall_seconds": physics_wall_seconds,
                    "observer_wall_seconds": observer_wall_seconds,
                    "instrumentation_wall_seconds": capture.instrumentation_wall_seconds if capture else 0.0,
                    "physics_steps_executed": physics_steps_executed,
                    "weather_preset": scenario.weather_preset,
                    "goal_spawn_index": scenario.goal_spawn_index,
                    "ego_spawn_index": scenario.ego_spawn_index,
                    "startup_hold_ticks": startup_hold_ticks,
                    "logical_delta_seconds": self.config.harness.fixed_delta_seconds,
                    "physics_delta_seconds": native_delta,
                    "control_substeps": control_substeps,
                    "execution_seed": self.config.harness.seed,
                    "keep_synchronous": self.config.harness.keep_synchronous,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "scenario_trigger_events": trigger_events,
                    **({"outcome_v2": outcome_summary} if outcome_summary is not None else {}),
                },
                run_id=run_id,
                execution_status=execution_status,
                termination_reason=termination_reason,
                observed_duration_s=observed_duration_s,
                outcome_path=outcome_path_str,
                capture_manifest_path=manifest_path_str,
                errors=list(capture.errors) if capture else [],
            )

            for observer in self.observers:
                observer.on_run_end(result, context)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                termination_reason = "cancelled"
            elif termination_reason == "horizon_exhausted":
                if phase_stage == "setup":
                    termination_reason = "scenario_setup_failed"
                elif phase_stage == "agent":
                    termination_reason = "agent_error"
                else:
                    termination_reason = "simulator_error"
            ran_any_eval = physics_steps_executed > 0 or ticks_executed > 0
            execution_status = "partial" if ran_any_eval else "failed"
            m_path = o_path = outcome_summary = None
            err_list: list[dict[str, Any]] = [{"phase": phase_stage, "error": f"{type(exc).__name__}:{exc}"}]
            if capture is not None:
                capture.errors.extend(err_list)
                try:
                    m_path, o_path, outcome_summary = capture.finalize(
                        execution_status=execution_status,
                        termination_reason=termination_reason,
                        reached_goal=reached_goal if ran_any_eval else None,
                        collision_oracle=collision_oracle,
                        lane_oracle=lane_oracle,
                        extra_provenance={"scenario_trigger_events": list(context.get("scenario_trigger_events") or ())},
                    )
                except Exception as fin_exc:
                    capture.errors.append({"phase": "finalize_failure", "error": f"{type(fin_exc).__name__}:{fin_exc}"})
                err_list = list(capture.errors)
            # Always write an atomic failure envelope even if trace finalization failed!
            if self.config.result.write_json:
                try:
                    partial_result = RunResult(
                        scenario_id=scenario.scenario_id,
                        town=scenario.town,
                        agent_kind=self.config.agent.kind,
                        succeeded=False,
                        dry_run=False,
                        ticks_executed=ticks_executed,
                        reached_goal=reached_goal,
                        terminated_by_collision=terminated_by_collision,
                        collision_count=len(collision_oracle.events),
                        collisions=list(collision_oracle.events),
                        safety_metrics=safety_oracle.metrics,
                        notes=[*notes, f"Partial/failed run: {type(exc).__name__}: {exc}"],
                        metadata={
                            "wall_seconds": time.perf_counter() - run_started,
                            "physics_steps_executed": physics_steps_executed,
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                            "scenario_trigger_events": list(context.get("scenario_trigger_events") or ()),
                            **({"outcome_v2": outcome_summary} if outcome_summary is not None else {}),
                        },
                        run_id=run_id,
                        execution_status=execution_status,
                        termination_reason=termination_reason,
                        observed_duration_s=(
                            capture.eval_end_s - capture.eval_start_s
                            if capture is not None and capture.eval_start_s is not None and capture.eval_end_s is not None
                            else None
                        ),
                        outcome_path=str(o_path) if o_path else None,
                        capture_manifest_path=str(m_path) if m_path else None,
                        errors=err_list,
                    )
                    self.write_result(partial_result, self.config.harness.output_dir)
                except Exception:
                    pass
            raise
        finally:
            if scenario_controller is not None:
                try:
                    scenario_controller.teardown(context)
                except Exception:
                    pass
            if "agent" in context and hasattr(context["agent"], "destroy"):
                try:
                    context["agent"].destroy()
                except Exception:
                    pass
            collision_oracle.destroy()
            lane_oracle.destroy()
            if self.config.run.cleanup_actors:
                # Include any controller-spawned actors even if setup() raised mid-way
                all_cleanup = list(actors)
                seen_ids = {id(a) for a in all_cleanup}
                if scenario_controller is not None:
                    for spawned in scenario_controller.spawned_actors:
                        if id(spawned) not in seen_ids:
                            all_cleanup.append(spawned)
                for actor in reversed(all_cleanup):
                    try:
                        actor.destroy()
                    except Exception:
                        pass
            if world is not None and original_settings is not None:
                try:
                    if not self.config.harness.keep_synchronous:
                        restore_world_settings(world, original_settings)
                except Exception:
                    pass
            if capture is not None and not capture.finalized:
                try:
                    capture.writer.close()
                except Exception:
                    pass

        if self.config.result.write_json:
            self.write_result(result, self.config.harness.output_dir)
        return result

    @staticmethod
    def write_result(result: RunResult, output_dir: Path) -> Path:
        target_dir = ensure_output_dir(output_dir)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"-{result.run_id[-8:]}" if result.run_id else ""
        out_path = target_dir / f"{result.scenario_id}-{result.agent_kind}{suffix}-{timestamp}.json"
        atomic_json(out_path, result.to_dict())
        return out_path
