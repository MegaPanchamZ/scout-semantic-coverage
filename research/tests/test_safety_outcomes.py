"""Tests for the safety-outcome layer (oracle evidence + log classification).

No live CARLA is required: the oracle tests use small fakes that satisfy the
CARLA API shape it reads.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.harness.oracles import SafetyOracle, SafetyThresholds  # noqa: E402
from research.harness.safety_outcomes import (  # noqa: E402
    OUTCOME_NAMES,
    OutcomeThresholds,
    classify_run,
)

_POLICY_PATH = (
    REPO_ROOT / "research" / "experiments" / "EXP-020-policy-comparison"
    / "proof-of-concept" / "policy_search.py"
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _Vec:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class _Loc:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class _Control:
    def __init__(self, brake: float = 0.0, steer: float = 0.0) -> None:
        self.brake, self.steer = brake, steer


class _TrafficLight:
    def __init__(self, state: str = "Red") -> None:
        self._state = state

    def get_state(self) -> str:
        return self._state


class _Actor:
    def __init__(self, actor_id: int, type_id: str, loc: _Loc, vel: _Vec) -> None:
        self.id = actor_id
        self.type_id = type_id
        self._loc = loc
        self._vel = vel

    def get_location(self) -> _Loc:
        return self._loc

    def get_velocity(self) -> _Vec:
        return self._vel


class _Ego(_Actor):
    def __init__(self, loc, vel, control=None, at_light=False, light=None) -> None:
        super().__init__(0, "vehicle.tesla.model3", loc, vel)
        self._control = control or _Control()
        self._at_light = at_light
        self._light = light

    def get_control(self) -> _Control:
        return self._control

    def is_at_traffic_light(self) -> bool:
        return self._at_light

    def get_traffic_light(self):
        return self._light


class _Waypoint:
    def __init__(self, loc: _Loc, is_junction: bool = False) -> None:
        self.transform = type("_Transform", (), {"location": loc})()
        self.is_junction = is_junction


class _FakeMap:
    def __init__(self, lane_center: _Loc | None = None, is_junction: bool = False) -> None:
        self._lane_center = lane_center or _Loc(0.0, 0.0, 0.0)
        self._is_junction = is_junction

    def get_waypoint(self, location, project_to_road: bool = True):  # noqa: ARG002
        return _Waypoint(self._lane_center, self._is_junction)


class _World:
    def __init__(self, actors: list[_Actor], carla_map: _FakeMap | None = None) -> None:
        self._actors = actors
        self._map = carla_map or _FakeMap()

    def get_actors(self) -> list[_Actor]:
        return self._actors

    def get_map(self) -> _FakeMap:
        return self._map


def _ego(speed: float, x: float = 0.0, y: float = 0.0, **kwargs) -> _Ego:
    return _Ego(_Loc(x, y, 0.0), _Vec(speed, 0.0, 0.0), **kwargs)


# --------------------------------------------------------------------------
# Oracle evidence
# --------------------------------------------------------------------------


def test_time_to_collision_and_vehicle_gap():
    ego = _ego(10.0, x=0.0)
    other = _Actor(1, "vehicle.audi.a2", _Loc(20.0, 0.0, 0.0), _Vec(0.0, 0.0, 0.0))
    world = _World([ego, other], _FakeMap())
    oracle = SafetyOracle(tick_seconds=0.1)
    oracle.tick(world, ego)
    assert math.isclose(oracle.metrics.min_ttc, 2.0, rel_tol=1e-6)
    assert math.isclose(oracle.metrics.min_vehicle_distance_m, 20.0, rel_tol=1e-6)
    assert math.isinf(oracle.metrics.min_pedestrian_distance_m)


def test_pedestrian_proximity_and_near_collision_edges():
    ego = _ego(1.0, x=0.0)
    walker = _Actor(2, "walker.pedestrian.0001", _Loc(3.0, 0.0, 0.0), _Vec(0.0, 0.0, 0.0))
    world = _World([ego, walker], _FakeMap())
    oracle = SafetyOracle(tick_seconds=0.1)
    oracle.tick(world, ego)  # 3.0 m: outside the near band
    assert oracle.metrics.near_collisions == 0
    walker._loc = _Loc(1.5, 0.0, 0.0)
    oracle.tick(world, ego)  # 1.5 m: rising edge into the near band
    assert oracle.metrics.near_collisions == 1
    assert oracle.metrics.near_collision_ticks == 1
    assert math.isclose(oracle.metrics.min_pedestrian_distance_m, 1.5, rel_tol=1e-6)
    oracle.tick(world, ego)  # still inside, no new event
    assert oracle.metrics.near_collisions == 1
    assert oracle.metrics.near_collision_ticks == 2


def test_red_light_and_lane_departure_events():
    ego = _ego(5.0, at_light=True, light=_TrafficLight("Red"))
    world = _World([ego], _FakeMap(lane_center=_Loc(0.0, 3.0, 0.0)))
    oracle = SafetyOracle(tick_seconds=0.1)
    oracle.tick(world, ego)
    assert oracle.metrics.red_light_violations == 1
    assert oracle.metrics.rule_violations == 1
    assert oracle.metrics.lane_departure_events == 1
    assert math.isclose(oracle.metrics.max_lane_offset_m, 3.0, rel_tol=1e-6)
    oracle.tick(world, ego)  # sustained departure does not double count
    assert oracle.metrics.lane_departure_events == 1


def test_lane_offset_ignored_at_junction():
    ego = _ego(5.0)
    world = _World([ego], _FakeMap(lane_center=_Loc(0.0, 5.0, 0.0), is_junction=True))
    oracle = SafetyOracle(tick_seconds=0.1)
    oracle.tick(world, ego)
    assert oracle.metrics.lane_departure_events == 0
    assert oracle.metrics.max_lane_offset_m == 0.0


def test_harsh_and_emergency_deceleration():
    ego = _ego(10.0)
    world = _World([ego], _FakeMap())
    oracle = SafetyOracle(tick_seconds=0.1)
    oracle.tick(world, ego)
    # 10 -> 8 m/s in one 0.1 s tick is a 20 m/s^2 deceleration.
    ego._vel = _Vec(8.0, 0.0, 0.0)
    oracle.tick(world, ego)
    assert math.isclose(oracle.metrics.max_deceleration_mps2, 20.0, rel_tol=1e-6)
    assert oracle.metrics.harsh_braking_events == 1
    assert oracle.metrics.emergency_manoeuvre_events == 1


def test_stationary_streak_marks_stuck():
    ego = _ego(0.0)
    world = _World([ego], _FakeMap())
    thresholds = SafetyThresholds(stuck_seconds=1.0, stationary_speed_mps=0.1)
    oracle = SafetyOracle(tick_seconds=0.1, thresholds=thresholds)
    for _ in range(10):
        oracle.tick(world, ego)
    assert oracle.metrics.max_stationary_streak_ticks == 10
    # 1.0 s at 10 Hz is a 10-tick threshold; frames past it are counted.
    assert oracle.metrics.stuck_frames == 0
    oracle.tick(world, ego)
    assert oracle.metrics.stuck_frames == 1


# --------------------------------------------------------------------------
# Log classification
# --------------------------------------------------------------------------


def _payload(**overrides):
    payload = {
        "reached_goal": True,
        "ticks_executed": 120,
        "collision_count": 0,
        "terminated_by_collision": False,
        "safety_metrics": {
            "min_ttc": float("inf"),
            "near_collisions": 0,
            "min_pedestrian_distance_m": float("inf"),
            "min_vehicle_distance_m": float("inf"),
            "red_light_violations": 0,
            "rule_violations": 0,
            "lane_departure_events": 0,
            "max_lane_offset_m": 0.0,
            "max_stationary_streak_ticks": 0,
            "max_deceleration_mps2": 0.0,
            "harsh_braking_events": 0,
            "emergency_manoeuvre_events": 0,
        },
        "metadata": {"telemetry": {"progress_to_goal_m": 40.0}},
    }
    payload.update(overrides)
    return payload


def test_classify_clean_run_is_safe():
    result = classify_run(_payload())
    assert result["unsafe"] is False
    assert result["reasons"] == []
    assert result["collision"] is False


def test_classify_collision_and_route_incomplete():
    payload = _payload(
        collision_count=1,
        terminated_by_collision=True,
        reached_goal=False,
        ticks_executed=500,
        metadata={"telemetry": {"progress_to_goal_m": 0.0}},
    )
    result = classify_run(payload)
    assert result["collision"] is True
    assert result["route_incomplete"] is True
    assert result["no_progress"] is True
    assert result["unsafe"] is True


def test_classify_near_collision_and_unsafe_proximity():
    payload = _payload(
        safety_metrics={
            **_payload()["safety_metrics"],
            "min_ttc": 0.8,
            "min_pedestrian_distance_m": 1.2,
        }
    )
    result = classify_run(payload)
    assert result["near_collision"] is True
    assert result["unsafe_proximity"] is True


def test_classify_rule_violation_lane_departure_and_harsh_braking():
    payload = _payload(
        safety_metrics={
            **_payload()["safety_metrics"],
            "red_light_violations": 1,
            "max_lane_offset_m": 2.6,
            "max_deceleration_mps2": 7.5,
        }
    )
    result = classify_run(payload)
    assert result["red_light_violation"] is True
    assert result["lane_departure"] is True
    assert result["traffic_rule_violation"] is True
    assert result["harsh_braking"] is True
    assert result["emergency_manoeuvre"] is True


def test_classify_stuck_from_stationary_streak():
    payload = _payload(
        reached_goal=False,
        safety_metrics={
            **_payload()["safety_metrics"],
            "max_stationary_streak_ticks": 200,
        },
    )
    result = classify_run(payload)
    assert result["stuck"] is True
    assert result["route_incomplete"] is True


def test_classify_legacy_log_does_not_invent_outcomes():
    legacy = {
        "reached_goal": True,
        "ticks_executed": 50,
        "collision_count": 0,
        "terminated_by_collision": False,
    }
    result = classify_run(legacy)
    assert result["unsafe"] is False
    for name in OUTCOME_NAMES:
        assert result[name] is False, name


def test_lane_offset_falls_back_to_telemetry_cross_track_error():
    payload = _payload(
        safety_metrics={},  # legacy schema
        metadata={"telemetry": {"route_progress": {"max_cross_track_error_m": 3.5}}},
    )
    result = classify_run(payload)
    assert result["lane_departure"] is True
    assert result["unsafe"] is True


def test_custom_thresholds_are_respected():
    payload = _payload(safety_metrics={**_payload()["safety_metrics"], "min_ttc": 2.0})
    assert classify_run(payload)["near_collision"] is False
    strict = OutcomeThresholds(near_collision_ttc_s=3.0)
    assert classify_run(payload, strict)["near_collision"] is True


# --------------------------------------------------------------------------
# Campaign row wiring
# --------------------------------------------------------------------------


def _load_policy():
    spec = importlib.util.spec_from_file_location("safety_policy", _POLICY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_row_carries_safety_outcomes():
    policy = _load_policy()
    payload = _payload(
        collision_count=0,
        safety_metrics={**_payload()["safety_metrics"], "min_pedestrian_distance_m": 1.0},
    )
    row = policy._row_from_run(payload, duration=1.5)
    assert row["safety_unsafe"] is True
    assert row["safety_unsafe_proximity"] is True
    assert "unsafe_proximity" in row["safety_reasons"]
    assert row["safety_metrics"]["min_pedestrian_distance_m"] == 1.0
