"""Named safety outcomes derived from recorded simulator logs.

The paper's utility question must not rest on collision alone.  This module
turns a single run log (the JSON written by :class:`~research.harness.runner.HarnessRunner`,
including its ``safety_metrics`` and aggregate telemetry) into a fixed set of
named *outcomes*:

===========================  =================================================
outcome                      evidence
===========================  =================================================
``collision``                collision sensor contact (existing ground truth)
``near_collision``           near-collision band entry or low time-to-collision
``unsafe_proximity``         pedestrian/vehicle gap below a safety threshold
``red_light_violation``      ego crosses a red light above a speed threshold
``lane_departure``           lane-centre offset above a threshold while moving
``traffic_rule_violation``   ``red_light_violation`` or ``lane_departure``
``stuck``                    stationary streak past a time threshold
``no_progress``              episode ended short with negligible route progress
``route_incomplete``         goal not reached within the tick budget
``harsh_braking``            longitudinal deceleration above a threshold
``emergency_manoeuvre``      very hard deceleration or hard brake plus hard steer
===========================  =================================================

``unsafe`` is the disjunction of all of the above.  The thresholds are
engineering proxies, not validated labels; keep them in one place so a campaign
can report sensitivity to them.  Legacy run logs without schema-2
``safety_metrics`` classify to ``False`` for the new outcomes rather than
guessing, and the raw evidence is carried through for inspection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SAFETY_OUTCOME_SCHEMA_VERSION = 1

OUTCOME_NAMES: tuple[str, ...] = (
    "collision",
    "near_collision",
    "unsafe_proximity",
    "red_light_violation",
    "lane_departure",
    "traffic_rule_violation",
    "stuck",
    "no_progress",
    "route_incomplete",
    "harsh_braking",
    "emergency_manoeuvre",
)


@dataclass(slots=True)
class OutcomeThresholds:
    """Classification thresholds (overridable per campaign)."""

    near_collision_ttc_s: float = 1.5
    near_collision_distance_m: float = 2.5
    unsafe_pedestrian_distance_m: float = 2.0
    unsafe_vehicle_distance_m: float = 2.5
    lane_offset_m: float = 2.0
    stationary_speed_mps: float = 0.1
    stuck_seconds: float = 8.0
    harsh_deceleration_mps2: float = 3.0
    emergency_deceleration_mps2: float = 6.0
    tick_seconds: float = 0.1
    no_progress_distance_m: float = 5.0
    no_progress_min_ticks: int = 100


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return None


def classify_run(
    run_payload: dict[str, Any],
    thresholds: OutcomeThresholds | None = None,
) -> dict[str, Any]:
    """Classify one run log payload into named unsafe behaviours."""
    th = thresholds or OutcomeThresholds()
    metrics = run_payload.get("safety_metrics") or {}
    metadata = run_payload.get("metadata") or {}
    telemetry = metadata.get("telemetry") or {}

    reached_goal = bool(run_payload.get("reached_goal"))
    ticks_executed = _integer(run_payload.get("ticks_executed"))
    collision_count = _integer(run_payload.get("collision_count")) or 0
    terminated_by_collision = bool(run_payload.get("terminated_by_collision"))

    min_ttc = _number(metrics.get("min_ttc"))
    min_pedestrian = _number(metrics.get("min_pedestrian_distance_m"))
    min_vehicle = _number(metrics.get("min_vehicle_distance_m"))
    max_lane_offset = _number(metrics.get("max_lane_offset_m"))
    max_deceleration = _number(metrics.get("max_deceleration_mps2"))
    stationary_ticks = _integer(metrics.get("max_stationary_streak_ticks")) or 0
    near_collision_events = _integer(metrics.get("near_collisions")) or 0
    red_light_violations = _integer(metrics.get("red_light_violations")) or 0
    lane_departure_events = _integer(metrics.get("lane_departure_events")) or 0
    harsh_braking_events = _integer(metrics.get("harsh_braking_events")) or 0
    emergency_events = _integer(metrics.get("emergency_manoeuvre_events")) or 0

    route_progress = telemetry.get("route_progress")
    if max_lane_offset is None and isinstance(route_progress, dict):
        max_lane_offset = _number(route_progress.get("max_cross_track_error_m"))
    progress_to_goal = _number(telemetry.get("progress_to_goal_m"))

    stationary_seconds = stationary_ticks * th.tick_seconds

    collision = collision_count > 0 or terminated_by_collision
    near_collision = near_collision_events > 0 or (
        min_ttc is not None and min_ttc < th.near_collision_ttc_s
    )
    unsafe_proximity = (
        (min_pedestrian is not None and min_pedestrian < th.unsafe_pedestrian_distance_m)
        or (min_vehicle is not None and min_vehicle < th.unsafe_vehicle_distance_m)
    )
    red_light_violation = red_light_violations > 0
    lane_departure = lane_departure_events > 0 or (
        max_lane_offset is not None and max_lane_offset > th.lane_offset_m
    )
    traffic_rule_violation = red_light_violation or lane_departure
    stuck = stationary_seconds >= th.stuck_seconds
    route_incomplete = not reached_goal
    no_progress = (
        route_incomplete
        and ticks_executed is not None
        and ticks_executed >= th.no_progress_min_ticks
        and progress_to_goal is not None
        and progress_to_goal < th.no_progress_distance_m
    )
    harsh_braking = harsh_braking_events > 0 or (
        max_deceleration is not None and max_deceleration >= th.harsh_deceleration_mps2
    )
    emergency_manoeuvre = emergency_events > 0 or (
        max_deceleration is not None and max_deceleration >= th.emergency_deceleration_mps2
    )

    outcomes = {
        "collision": collision,
        "near_collision": near_collision,
        "unsafe_proximity": unsafe_proximity,
        "red_light_violation": red_light_violation,
        "lane_departure": lane_departure,
        "traffic_rule_violation": traffic_rule_violation,
        "stuck": stuck,
        "no_progress": no_progress,
        "route_incomplete": route_incomplete,
        "harsh_braking": harsh_braking,
        "emergency_manoeuvre": emergency_manoeuvre,
    }
    reasons = [name for name in OUTCOME_NAMES if outcomes[name]]

    return {
        "schema_version": SAFETY_OUTCOME_SCHEMA_VERSION,
        **outcomes,
        "unsafe": bool(reasons),
        "reasons": reasons,
        "raw": {
            "min_ttc_s": min_ttc,
            "min_pedestrian_distance_m": min_pedestrian,
            "min_vehicle_distance_m": min_vehicle,
            "max_lane_offset_m": max_lane_offset,
            "max_deceleration_mps2": max_deceleration,
            "max_stationary_streak_s": stationary_seconds,
            "progress_to_goal_m": progress_to_goal,
            "ticks_executed": ticks_executed,
        },
    }


def safety_schema() -> dict[str, Any]:
    """Machine-readable description of the outcomes and their thresholds."""
    return {
        "schema_version": SAFETY_OUTCOME_SCHEMA_VERSION,
        "outcomes": list(OUTCOME_NAMES),
        "thresholds": asdict(OutcomeThresholds()),
    }
