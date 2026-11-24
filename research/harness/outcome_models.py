"""Simulator-independent schema-2 evidence and assessments.

Times are simulation seconds, coordinates/metres and velocities/metres per second.
Missing values are None; availability is explicit, never inferred from zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import json
import math
from typing import Any, Iterable


CATEGORIES = {
    "safety": ("collision", "near_miss", "unsafe_lane_departure", "hazard_associated_braking"),
    "behaviour": ("hard_braking", "lane_crossing", "close_proximity"),
    "rules": ("red_light_violation", "stop_sign_violation", "speeding", "illegal_lane_change", "wrong_way"),
    "task": ("prolonged_immobilization", "stuck", "no_progress", "route_incomplete", "timeout"),
    "quality": ("invalid_scenario", "low_interaction"),
}
DEFINITION_VERSION = "scout-outcomes-2.2-draft"


class AssessmentStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: getattr(obj, f.name) for f in fields(obj)}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def strict_json(value: Any, *, indent: int | None = None) -> str:
    """Reject non-finite values rather than silently converting bad evidence."""
    try:
        return json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
            default=_json_default,
        )
    except RecursionError as exc:
        raise ValueError("JSON serialization nesting depth exceeded") from exc


def finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    try:
        is_fin = math.isfinite(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not is_fin:
        raise ValueError(f"{name} must be finite")


def _str_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of strings")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must contain only strings")
    return tuple(value)


@dataclass(frozen=True)
class MetricAssessment:
    status: AssessmentStatus = AssessmentStatus.UNKNOWN
    event_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    assessed_duration_s: float | None = None
    expected_duration_s: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "status", AssessmentStatus(self.status))
        object.__setattr__(self, "event_ids", _str_tuple(self.event_ids, "event_ids"))
        object.__setattr__(self, "reasons", _str_tuple(self.reasons, "reasons"))
        if self.status == AssessmentStatus.ABSENT and not self.reasons:
            raise ValueError("An absent assessment requires a coverage justification")
        for name in ("assessed_duration_s", "expected_duration_s"):
            value = getattr(self, name)
            if value is not None:
                finite(value, name)
                if value < 0:
                    raise ValueError(f"{name} cannot be negative")

    def __bool__(self):
        raise TypeError("Use assessment_is_present(); assessments are not booleans")


def assessment_is_present(value: MetricAssessment | dict) -> bool:
    status = value.status if isinstance(value, MetricAssessment) else value.get("status")
    return status == AssessmentStatus.PRESENT


def aggregate_assessments(values: Iterable[MetricAssessment]) -> MetricAssessment:
    values = tuple(values)
    present = tuple(v for v in values if v.status == AssessmentStatus.PRESENT)
    if present:
        return MetricAssessment(AssessmentStatus.PRESENT,
                                tuple(sorted({e for v in present for e in v.event_ids})))
    applicable = tuple(v for v in values if v.status != AssessmentStatus.NOT_APPLICABLE)
    if not applicable:
        return MetricAssessment(AssessmentStatus.NOT_APPLICABLE)
    if all(v.status == AssessmentStatus.ABSENT for v in applicable):
        return MetricAssessment(AssessmentStatus.ABSENT, reasons=("all_applicable_constituents_absent",))
    return MetricAssessment(reasons=("one_or_more_constituents_unknown",))


@dataclass(frozen=True)
class ActorState:
    actor_id: str
    actor_type: str
    x: float
    y: float
    yaw_rad: float
    half_length_m: float
    half_width_m: float
    vx: float
    vy: float
    z_min_m: float
    z_max_m: float
    role: str | None = None
    bbox_offset_x: float = 0.0
    bbox_offset_y: float = 0.0
    bbox_yaw_rad: float = 0.0

    def __post_init__(self):
        if not isinstance(self.actor_id, str) or not self.actor_id:
            raise ValueError("actor_id must be a nonempty run-local string")
        if not isinstance(self.actor_type, str) or not self.actor_type:
            raise ValueError("actor_type must be a nonempty string")
        if self.role is not None and (not isinstance(self.role, str) or not self.role):
            raise ValueError("role must be a nonempty stable string or None")
        for name in ("x", "y", "yaw_rad", "half_length_m", "half_width_m", "vx", "vy",
                     "z_min_m", "z_max_m", "bbox_offset_x", "bbox_offset_y", "bbox_yaw_rad"):
            finite(getattr(self, name), name)
        if self.half_length_m <= 0 or self.half_width_m <= 0 or self.z_min_m >= self.z_max_m:
            raise ValueError("Invalid actor bounding box")


@dataclass(frozen=True)
class TaskContext:
    """Observed waiting reasons; None is unobserved, not False."""
    objective_active: bool | None = None
    red_light_stop: bool | None = None
    stop_sign_stop: bool | None = None
    blocked_route: bool | None = None
    yielding: bool | None = None
    sensor_wait: bool | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("objective_active", "red_light_stop", "stop_sign_stop", "blocked_route", "yielding", "sensor_wait"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool or None")
        object.__setattr__(self, "evidence_refs", _str_tuple(self.evidence_refs, "evidence_refs"))

    def eligibility(self) -> tuple[str, str]:
        if self.objective_active is False:
            return "excluded", "objective_inactive"
        if self.sensor_wait is True:
            return "unknown", "execution_sensor_wait"
        for name in ("red_light_stop", "stop_sign_stop", "blocked_route", "yielding"):
            if getattr(self, name) is True:
                return "excluded", name
        if self.objective_active is None or any(getattr(self, name) is None for name in
                ("sensor_wait", "red_light_stop", "stop_sign_stop", "blocked_route", "yielding")):
            return "unknown", "waiting_context_incomplete"
        return "eligible", "no_legitimate_wait_observed"


@dataclass(frozen=True)
class RouteProgress:
    route_id: str
    status: str
    total_m: float
    s_m: float | None = None
    high_water_m: float | None = None
    cross_track_m: float | None = None
    reason: str | None = None
    continuity_id: int = 0

    def __post_init__(self):
        if (
            not isinstance(self.route_id, str)
            or not self.route_id
            or not isinstance(self.status, str)
            or self.status not in {"available", "unknown"}
        ):
            raise ValueError("Invalid route progress identity/status")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError("Route progress reason must be a string or None")
        if isinstance(self.continuity_id, bool) or not isinstance(self.continuity_id, int) or self.continuity_id < 0:
            raise ValueError("Route continuity_id must be a nonnegative integer")
        finite(self.total_m, "total_m")
        if self.total_m <= 0:
            raise ValueError("Route length must be positive")
        for name in ("s_m", "high_water_m", "cross_track_m"):
            value = getattr(self, name)
            if value is not None:
                finite(value, name)
                if value < 0:
                    raise ValueError(f"{name} cannot be negative")
        if self.status == "available":
            if self.s_m is None or self.high_water_m is None or self.cross_track_m is None:
                raise ValueError("Available progress needs all measurements")
            if not self.s_m <= self.high_water_m <= self.total_m:
                raise ValueError("Route progress/high water outside route")


@dataclass(frozen=True)
class ScenarioChecks:
    """Setup evidence. Execution errors alone do not establish infeasibility."""
    configuration_feasible: bool | None = None
    initial_geometry_feasible: bool | None = None
    required_actors_spawned: bool | None = None
    route_feasible: bool | None = None
    controller_ready: bool | None = None
    execution_error: str | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("configuration_feasible", "initial_geometry_feasible", "required_actors_spawned", "route_feasible", "controller_ready"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool or None")
        if self.execution_error is not None and not isinstance(self.execution_error, str):
            raise ValueError("execution_error must be str or None")
        object.__setattr__(self, "evidence_refs", _str_tuple(self.evidence_refs, "evidence_refs"))


@dataclass(frozen=True)
class RoadRuleContext:
    """Per-frame map and traffic-control context; None means unobserved."""
    lane_id: str | None = None
    lane_type: str | None = None
    in_junction: bool | None = None
    lane_change_permission: str | None = None
    permitted_travel_yaw_rad: float | None = None
    speed_limit_mps: float | None = None
    red_light_state: str | None = None
    stop_line_applicable: bool | None = None
    stop_line_crossed: bool | None = None
    stop_sign_applicable: bool | None = None
    stop_sign_approach: bool | None = None
    stop_sign_line_crossed: bool | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        if self.lane_id is not None and (not isinstance(self.lane_id, str) or not self.lane_id):
            raise ValueError("lane_id must be a nonempty string or None")
        if self.lane_type is not None and (
            not isinstance(self.lane_type, str)
            or self.lane_type not in {
                "driving", "shoulder", "sidewalk", "bidirectional", "parking", "biking",
                "restricted", "median", "entry", "exit", "offramp", "onramp", "offroad", "unknown",
            }
        ):
            raise ValueError(f"Unsupported lane_type: {self.lane_type}")
        if self.lane_change_permission is not None and (
            not isinstance(self.lane_change_permission, str)
            or self.lane_change_permission not in {"none", "left", "right", "both", "unknown"}
        ):
            raise ValueError(f"Unsupported lane_change_permission: {self.lane_change_permission}")
        if self.red_light_state is not None and (
            not isinstance(self.red_light_state, str)
            or self.red_light_state not in {"red", "yellow", "green", "off", "unknown"}
        ):
            raise ValueError(f"Unsupported red_light_state: {self.red_light_state}")
        for name in (
            "in_junction", "stop_line_applicable", "stop_line_crossed",
            "stop_sign_applicable", "stop_sign_approach", "stop_sign_line_crossed",
        ):
            val = getattr(self, name)
            if val is not None and not isinstance(val, bool):
                raise ValueError(f"{name} must be bool or None")
        if self.permitted_travel_yaw_rad is not None:
            finite(self.permitted_travel_yaw_rad, "permitted_travel_yaw_rad")
        if self.speed_limit_mps is not None:
            finite(self.speed_limit_mps, "speed_limit_mps")
            if self.speed_limit_mps <= 0:
                raise ValueError("speed_limit_mps must be positive")
        object.__setattr__(self, "evidence_refs", _str_tuple(self.evidence_refs, "evidence_refs"))


@dataclass(frozen=True)
class FrameEvidence:
    run_id: str
    sim_frame: int
    sim_time_s: float
    ego: ActorState
    actors: tuple[ActorState, ...] = ()
    available_channels: tuple[str, ...] = ()
    phase: str = "evaluation"
    longitudinal_acceleration_mps2: float | None = None
    brake: float | None = None
    discontinuity: bool = False
    task_context: TaskContext | None = None
    route_progress: RouteProgress | None = None
    road_context: RoadRuleContext | None = None
    reverse: bool | None = None
    logical_tick: int | None = None
    substep: int | None = None
    throttle: float | None = None
    steer: float | None = None

    def __post_init__(self):
        validate_stamp(self.run_id, self.sim_frame, self.sim_time_s, self.phase)
        if not isinstance(self.ego, ActorState):
            raise ValueError("FrameEvidence.ego must be an ActorState")
        if not isinstance(self.actors, (list, tuple)) or any(not isinstance(a, ActorState) for a in self.actors):
            raise ValueError("FrameEvidence.actors must be a sequence of ActorState")
        object.__setattr__(self, "actors", tuple(self.actors))
        object.__setattr__(self, "available_channels", _str_tuple(self.available_channels, "available_channels"))
        if not isinstance(self.discontinuity, bool):
            raise ValueError("discontinuity must be boolean")
        if self.task_context is not None and not isinstance(self.task_context, TaskContext):
            raise ValueError("task_context must be normalized TaskContext")
        if self.route_progress is not None and not isinstance(self.route_progress, RouteProgress):
            raise ValueError("route_progress must be normalized RouteProgress")
        if self.road_context is not None and not isinstance(self.road_context, RoadRuleContext):
            raise ValueError("road_context must be normalized RoadRuleContext")
        if self.reverse is not None and not isinstance(self.reverse, bool):
            raise ValueError("reverse must be bool or None")
        for name in ("logical_tick", "substep"):
            val = getattr(self, name)
            if val is not None and (isinstance(val, bool) or not isinstance(val, int) or val < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        ids = [self.ego.actor_id, *(a.actor_id for a in self.actors)]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate actor ID in frame")
        for name in ("longitudinal_acceleration_mps2", "brake", "throttle", "steer"):
            if getattr(self, name) is not None:
                finite(getattr(self, name), name)
        if self.brake is not None and not 0 <= self.brake <= 1:
            raise ValueError("brake must be in [0, 1]")
        if self.throttle is not None and not 0 <= self.throttle <= 1:
            raise ValueError("throttle must be in [0, 1]")
        if self.steer is not None and not -1 <= self.steer <= 1:
            raise ValueError("steer must be in [-1, 1]")

    @classmethod
    def from_dict(cls, value: dict) -> FrameEvidence:
        if not isinstance(value, dict):
            raise ValueError("FrameEvidence payload must be a JSON object")
        data = dict(value)
        if not isinstance(data.get("ego"), dict):
            raise ValueError("FrameEvidence.ego must be an object")
        data["ego"] = ActorState(**data["ego"])
        raw_actors = data.get("actors", ())
        if not isinstance(raw_actors, (list, tuple)) or any(not isinstance(a, dict) for a in raw_actors):
            raise ValueError("FrameEvidence.actors must be a list of actor objects")
        data["actors"] = tuple(ActorState(**a) for a in raw_actors)
        raw_channels = data.get("available_channels", ())
        if not isinstance(raw_channels, (list, tuple)) or any(not isinstance(ch, str) for ch in raw_channels):
            raise ValueError("FrameEvidence.available_channels must be a list of strings")
        data["available_channels"] = tuple(raw_channels)
        if data.get("task_context") is not None:
            if not isinstance(data["task_context"], dict):
                raise ValueError("FrameEvidence.task_context must be an object or null")
            data["task_context"] = TaskContext(**data["task_context"])
        if data.get("route_progress") is not None:
            if not isinstance(data["route_progress"], dict):
                raise ValueError("FrameEvidence.route_progress must be an object or null")
            data["route_progress"] = RouteProgress(**data["route_progress"])
        if data.get("road_context") is not None:
            if not isinstance(data["road_context"], dict):
                raise ValueError("FrameEvidence.road_context must be an object or null")
            data["road_context"] = RoadRuleContext(**data["road_context"])
        return cls(**data)


def validate_stamp(run_id, frame, time_s, phase):
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id is required")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError("sim_frame must be a nonnegative integer")
    finite(time_s, "sim_time_s")
    if time_s < 0 or not isinstance(phase, str) or phase not in {"setup", "warmup", "startup_hold", "evaluation"}:
        raise ValueError("Invalid time or observation phase")


@dataclass(frozen=True)
class SensorEvidence:
    run_id: str
    sim_frame: int
    sim_time_s: float
    channel: str
    actor_id: str
    phase: str = "evaluation"
    impulse_ns: tuple[float, float, float] | None = None
    callback_id: str | None = None
    lane_markings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        validate_stamp(self.run_id, self.sim_frame, self.sim_time_s, self.phase)
        if not isinstance(self.actor_id, str) or not self.actor_id or not isinstance(self.channel, str) or not self.channel:
            raise ValueError("Sensor channel and actor identity are required")
        if self.callback_id is not None and (not isinstance(self.callback_id, str) or not self.callback_id):
            raise ValueError("callback_id must be a nonempty string or None")
        if self.impulse_ns is not None:
            if not isinstance(self.impulse_ns, (list, tuple)):
                raise ValueError("Impulse must be a sequence of three finite numbers")
            object.__setattr__(self, "impulse_ns", tuple(self.impulse_ns))
            if len(self.impulse_ns) != 3:
                raise ValueError("Impulse requires three components")
            for v in self.impulse_ns:
                finite(v, "impulse_ns")
        object.__setattr__(self, "lane_markings", _str_tuple(self.lane_markings, "lane_markings"))
        if not isinstance(self.metadata, dict):
            raise ValueError("Sensor metadata must be a mapping")
        try:
            strict_json(self.metadata)
        except RecursionError as exc:
            raise ValueError("Sensor metadata exceeds JSON serialization recursion limit") from exc


@dataclass(frozen=True)
class ChannelCoverage:
    status: str = "unavailable"
    start_s: float | None = None
    end_s: float | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "reasons", _str_tuple(self.reasons, "reasons"))
        if not isinstance(self.status, str) or self.status not in {"complete", "partial", "unavailable"}:
            raise ValueError("Invalid channel coverage status")
        for v in (self.start_s, self.end_s):
            if v is not None:
                finite(v, "coverage time")
        if self.status == "complete" and (self.start_s is None or self.end_s is None):
            raise ValueError("Complete coverage needs an explicit interval")
        if self.start_s is not None and self.end_s is not None and self.end_s < self.start_s:
            raise ValueError("Reversed coverage interval")

    def covers(self, start: float, end: float) -> bool:
        return (self.status == "complete" and self.start_s is not None and self.end_s is not None
                and self.start_s <= start and self.end_s >= end)


@dataclass(frozen=True)
class CaptureManifest:
    run_id: str
    start_s: float
    end_s: float
    execution_status: str
    termination_reason: str
    channels: dict[str, ChannelCoverage] = field(default_factory=dict)
    schema_version: int = 2
    reached_goal: bool | None = None
    completion_deadline_s: float | None = None
    scan_radius_m: float | None = None
    projection_horizon_s: float | None = None
    max_sample_interval_s: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    scenario_checks: ScenarioChecks | None = None

    def __post_init__(self):
        finite(self.start_s, "start_s")
        finite(self.end_s, "end_s")
        if (
            not isinstance(self.run_id, str)
            or not self.run_id
            or type(self.schema_version) is not int
            or self.schema_version != 2
            or self.start_s < 0
            or self.end_s < self.start_s
        ):
            raise ValueError("Invalid capture identity, version or interval")
        if not isinstance(self.termination_reason, str):
            raise ValueError("termination_reason must be a string")
        if not isinstance(self.execution_status, str) or self.execution_status not in {
            "completed", "partial", "failed", "not_executed", "dry_run", "unknown",
        }:
            raise ValueError("Invalid execution status")
        if not isinstance(self.channels, dict) or any(
            not isinstance(k, str) or not isinstance(v, ChannelCoverage) for k, v in self.channels.items()
        ):
            raise ValueError("channels must be a mapping of string keys to ChannelCoverage")
        if self.reached_goal is not None and not isinstance(self.reached_goal, bool):
            raise ValueError("reached_goal must be bool or None")
        for name in ("completion_deadline_s", "scan_radius_m", "projection_horizon_s", "max_sample_interval_s"):
            value = getattr(self, name)
            if value is not None:
                finite(value, name)
                if value <= 0:
                    raise ValueError(f"{name} must be positive")
        if not isinstance(self.provenance, dict):
            raise ValueError("CaptureManifest provenance must be a mapping")
        try:
            strict_json(self.provenance)
        except RecursionError as exc:
            raise ValueError("CaptureManifest provenance exceeds JSON serialization recursion limit") from exc
        if self.scenario_checks is not None and not isinstance(self.scenario_checks, ScenarioChecks):
            raise ValueError("scenario_checks must be normalized ScenarioChecks")

    @classmethod
    def from_dict(cls, value: dict) -> CaptureManifest:
        if not isinstance(value, dict):
            raise ValueError("CaptureManifest payload must be a JSON object")
        data = dict(value)
        raw_channels = data.get("channels", {})
        if not isinstance(raw_channels, dict) or any(not isinstance(v, dict) for v in raw_channels.values()):
            raise ValueError("CaptureManifest.channels must be a mapping of channel objects")
        data["channels"] = {k: ChannelCoverage(**v) for k, v in raw_channels.items()}
        if data.get("scenario_checks") is not None:
            if not isinstance(data["scenario_checks"], dict):
                raise ValueError("CaptureManifest.scenario_checks must be an object or null")
            data["scenario_checks"] = ScenarioChecks(**data["scenario_checks"])
        return cls(**data)


@dataclass(frozen=True)
class OutcomeEvent:
    event_id: str
    label: str
    actor_ids: tuple[str, ...]
    start_s: float
    end_s: float
    state: str
    evidence_frames: tuple[int, ...]
    measurements: dict[str, Any] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


@dataclass
class RunOutcome:
    run_id: str
    execution_status: str
    termination_reason: str
    profile_digest: str
    categories: dict[str, dict[str, MetricAssessment]]
    events: list[OutcomeEvent] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 2
    definition_version: str = DEFINITION_VERSION

    def to_dict(self) -> dict:
        result = {f.name: getattr(self, f.name) for f in fields(self)}
        result["aggregates"] = {
            "safety_event": aggregate_assessments(self.categories["safety"].values()),
            "traffic_rule_violation": aggregate_assessments(self.categories["rules"].values()),
        }
        return json.loads(strict_json(result))


def unknown_categories(reason: str) -> dict[str, dict[str, MetricAssessment]]:
    return {group: {name: MetricAssessment(reasons=(reason,)) for name in names}
            for group, names in CATEGORIES.items()}
