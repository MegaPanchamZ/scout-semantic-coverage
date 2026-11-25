"""Scripted ego controls for detector-only validation cases.

Runs driven by these controllers are strictly tagged ``detector_validation``
and must never be interpreted as ADS effectiveness results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScheduledControl:
    start_tick: int
    end_tick: int
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    reverse: bool = False


class ScriptedCaseController:
    """Frame-scheduled ego control sequence for controlled live detector tests."""

    run_purpose = "detector_validation"

    def __init__(self, case_id: str, schedule: list[ScheduledControl]) -> None:
        self.case_id = case_id
        self.schedule = list(schedule)
        self._tick = 0

    def control_for_tick(self, tick: int) -> ScheduledControl:
        for step in self.schedule:
            if step.start_tick <= tick <= step.end_tick:
                return step
        return ScheduledControl(tick, tick, throttle=0.0, brake=0.5, steer=0.0)

    def next_control_dict(self) -> dict[str, Any]:
        step = self.control_for_tick(self._tick)
        self._tick += 1
        return {
            "throttle": step.throttle,
            "brake": step.brake,
            "steer": step.steer,
            "reverse": step.reverse,
            "purpose": self.run_purpose,
            "case_id": self.case_id,
        }


def build_case_controller(case_id: str) -> ScriptedCaseController:
    """Return a distinct deterministic scripted ego schedule per declared validation contrast."""
    schedules: dict[str, list[ScheduledControl]] = {
        "collision_positive_contact": [ScheduledControl(0, 250, throttle=0.75)],
        "collision_negative_close_pass": [
            ScheduledControl(0, 12, throttle=0.45),
            ScheduledControl(13, 35, throttle=0.0, brake=0.8),
            ScheduledControl(36, 170, throttle=0.25),
            ScheduledControl(171, 250, throttle=0.0, brake=1.0),
        ],
        "collision_uncertainty_disabled_sensor": [ScheduledControl(0, 250, throttle=0.45)],
        "near_miss_positive_resolved_conflict": [
            ScheduledControl(0, 30, throttle=0.65),
            ScheduledControl(31, 250, brake=1.0),
        ],
        "near_miss_negative_parallel_adjacent": [ScheduledControl(0, 250, throttle=0.4, steer=0.0)],
        "lane_positive_prohibited_solid_crossing": [
            ScheduledControl(0, 15, throttle=0.5),
            ScheduledControl(16, 32, throttle=0.4, steer=-0.28),
            ScheduledControl(33, 250, throttle=0.4, steer=0.0),
        ],
        "lane_negative_permitted_broken_change": [
            ScheduledControl(0, 18, throttle=0.5),
            ScheduledControl(19, 30, throttle=0.4, steer=0.12),
            ScheduledControl(31, 42, throttle=0.4, steer=-0.12),
            ScheduledControl(43, 70, throttle=0.35, steer=0.0),
            ScheduledControl(71, 250, brake=1.0),
        ],
        "braking_positive_hazard_associated": [
            ScheduledControl(0, 30, throttle=0.65),
            ScheduledControl(31, 250, brake=1.0),
        ],
        "braking_negative_low_speed_and_no_hazard": [
            ScheduledControl(0, 15, throttle=0.2),
            ScheduledControl(16, 30, brake=0.8),
        ],
        "red_light_positive_crossing": [ScheduledControl(0, 250, throttle=0.7)],
        "red_light_negative_approach_stop": [
            ScheduledControl(0, 16, throttle=0.72),
            ScheduledControl(17, 250, brake=1.0),
        ],
        "other_rules_positive_stop_sign_speeding_wrong_way": [
            ScheduledControl(0, 44, throttle=0.65, steer=0.0),
            ScheduledControl(45, 58, throttle=0.0, brake=1.0, steer=0.0),
            ScheduledControl(59, 98, throttle=0.35, steer=-0.85),
            ScheduledControl(99, 160, throttle=0.5, steer=0.0),
        ],
        "other_rules_negative_compliant_and_reverse": [
            ScheduledControl(0, 20, throttle=0.25, reverse=True),
            ScheduledControl(21, 60, brake=1.0),
        ],
        "task_positive_free_route_stuck": [ScheduledControl(0, 250, brake=1.0)],
        "task_negative_legitimate_red_light_wait": [
            ScheduledControl(0, 18, throttle=0.75),
            ScheduledControl(19, 250, brake=1.0),
        ],
        "quality_positive_low_interaction": [
            ScheduledControl(0, 45, throttle=0.25),
            ScheduledControl(46, 250, brake=0.8),
        ],
        "quality_positive_spawn_failure": [ScheduledControl(0, 250, throttle=0.4)],
        "quality_negative_safe_meaningful_interaction": [ScheduledControl(0, 250, throttle=0.4)],
    }
    if case_id not in schedules:
        raise ValueError(f"Unknown scripted validation case_id '{case_id}'; refusing silent fallback schedule")
    return ScriptedCaseController(case_id, schedules[case_id])


class ScriptedEgoAgent:
    """Harness-compatible scripted ego adapter for detector-only live validation."""

    run_purpose = "detector_validation"

    def __init__(self, world: Any, ego_vehicle: Any, controller: ScriptedCaseController) -> None:
        self._world = world
        self._ego = ego_vehicle
        self._controller = controller
        self._destination: Any | None = None
        self._route_preview: list[dict[str, Any]] = []
        self.last_step_info: dict[str, Any] = {"mode": "detector_validation", "case_id": controller.case_id}

    def set_destination(self, destination: Any) -> None:
        self._destination = destination
        try:
            start = self._ego.get_location()
            self._route_preview = [
                {"x": float(start.x), "y": float(start.y)},
                {"x": float(destination.x), "y": float(destination.y)},
            ]
        except Exception:
            self._route_preview = []

    def get_route_preview(self) -> list[dict[str, Any]]:
        return list(self._route_preview)

    def get_route_waypoints(self) -> list[dict[str, Any]]:
        return list(self._route_preview)

    def run_step(self) -> Any:
        step = self._controller.next_control_dict()
        self.last_step_info = {"mode": "detector_validation", "case_id": self._controller.case_id, **step}
        carla_mod = type(self._world).__module__.split(".")[0]
        try:
            carla = __import__(carla_mod)
            return carla.VehicleControl(
                throttle=float(step["throttle"]),
                brake=float(step["brake"]),
                steer=float(step["steer"]),
                reverse=bool(step["reverse"]),
            )
        except Exception:
            from types import SimpleNamespace
            return SimpleNamespace(
                throttle=float(step["throttle"]),
                brake=float(step["brake"]),
                steer=float(step["steer"]),
                reverse=bool(step["reverse"]),
                hand_brake=False,
                manual_gear_shift=False,
                gear=-1 if step["reverse"] else 1,
            )

    def done(self) -> bool:
        if self._destination is None:
            return False
        try:
            return float(self._ego.get_location().distance(self._destination)) <= 2.5
        except Exception:
            return False

    def get_debug_state(self) -> dict[str, Any]:
        return dict(self.last_step_info)
