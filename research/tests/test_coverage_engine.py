"""Synthetic-fixture tests for research/harness/coverage_engine.py.

The tests use a small hand-built oracle (not the EXP-018 inventory) so that the
mapping, witness, window and cross-trace rules are exercised deterministically.
A final smoke test validates normalization against a real semantic stream when
one is present in the repository.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.harness.coverage_engine import (  # noqa: E402
    DerivationConfig,
    TraceInput,
    compute_cov,
    compute_suite_cov,
    load_ego_route,
    load_oracle,
    load_semantic_trace,
    main,
    normalize_tick,
)


SYNTHETIC_ORACLE = {
    "metadata": {"split": "synthetic"},
    "counts": {"total": {"obligation_count": 18, "defined_predicates": 17}},
    "dimensions": {
        "node": {
            "vocabulary": ["pedestrian", "vehicle", "animal"],
            "obligations": [
                {
                    "dimension": "node",
                    "predicate": "node:pedestrian",
                    "signature": "node(pedestrian)",
                    "grounding": "direct",
                    "node_types": ["pedestrian"],
                },
                {
                    "dimension": "node",
                    "predicate": "node:vehicle",
                    "signature": "node(vehicle)",
                    "grounding": "direct",
                    "node_types": ["vehicle"],
                },
                {
                    "dimension": "node",
                    "predicate": "node:animal",
                    "signature": "node(animal)",
                    "grounding": "direct",
                    "node_types": ["animal"],
                },
            ],
        },
        "attribute": {
            "vocabulary": ["stationary", "moving", "braking", "jaywalking", "lane_changing"],
            "obligations": [
                {
                    "dimension": "attribute",
                    "predicate": "stationary",
                    "signature": "stationary(ego)",
                    "grounding": "derived",
                    "node_types": ["ego"],
                },
                {
                    "dimension": "attribute",
                    "predicate": "moving",
                    "signature": "moving(vehicle)",
                    "grounding": "derived",
                    "node_types": ["vehicle"],
                },
                {
                    "dimension": "attribute",
                    "predicate": "braking",
                    "signature": "braking(vehicle)",
                    "grounding": "derived",
                    "node_types": ["vehicle"],
                },
                {
                    "dimension": "attribute",
                    "predicate": "jaywalking",
                    "signature": "jaywalking(pedestrian)",
                    "grounding": "proxy",
                    "node_types": ["pedestrian"],
                },
                {
                    "dimension": "attribute",
                    "predicate": "lane_changing",
                    "signature": "lane_changing(vehicle)",
                    "grounding": "derived",
                    "node_types": ["vehicle"],
                },
            ],
        },
        "relation": {
            "vocabulary": ["in_front_of", "crossing_path", "same_lane", "adjacent_lane", "oncoming"],
            "obligations": [
                {
                    "dimension": "relation",
                    "predicate": "in_front_of",
                    "signature": "in_front_of(pedestrian,ego)",
                    "grounding": "direct",
                    "node_types": ["pedestrian", "ego"],
                },
                {
                    "dimension": "relation",
                    "predicate": "crossing_path",
                    "signature": "crossing_path(pedestrian,ego)",
                    "grounding": "derived",
                    "node_types": ["pedestrian", "ego"],
                },
                {
                    "dimension": "relation",
                    "predicate": "crossing_path",
                    "signature": "crossing_path(vehicle,ego)",
                    "grounding": "derived",
                    "node_types": ["vehicle", "ego"],
                },
                {
                    "dimension": "relation",
                    "predicate": "same_lane",
                    "signature": "same_lane(vehicle,ego)",
                    "grounding": "direct",
                    "node_types": ["vehicle", "ego"],
                },
                {
                    "dimension": "relation",
                    "predicate": "adjacent_lane",
                    "signature": "adjacent_lane(vehicle,ego)",
                    "grounding": "direct",
                    "node_types": ["vehicle", "ego"],
                },
                {
                    "dimension": "relation",
                    "predicate": "oncoming",
                    "signature": "oncoming(vehicle,ego)",
                    "grounding": "derived",
                    "node_types": ["vehicle", "ego"],
                },
            ],
        },
        "hazard_class": {
            "vocabulary": ["other: pedestrian", "other: lateral", "other: parked_lead", "other: unmapped"],
            "obligations": [
                {
                    "dimension": "hazard_class",
                    "predicate": "other: pedestrian",
                    "signature": "hazard(other: pedestrian)",
                    "grounding": "proxy",
                    "node_types": ["pedestrian"],
                },
                {
                    "dimension": "hazard_class",
                    "predicate": "other: lateral",
                    "signature": "hazard(other: lateral)",
                    "grounding": "derived",
                    "node_types": ["vehicle"],
                },
                {
                    "dimension": "hazard_class",
                    "predicate": "other: parked_lead",
                    "signature": "hazard(other: parked_lead)",
                    "grounding": "derived",
                    "node_types": ["vehicle"],
                },
                {
                    "dimension": "hazard_class",
                    "predicate": "other: unmapped",
                    "signature": "hazard(other: unmapped)",
                    "grounding": "derived",
                    "node_types": ["vehicle"],
                },
            ],
        },
    },
    "hazard_classes": {
        "definitions": [
            {
                "name": "other: pedestrian",
                "required_predicates": ["jaywalking", "crossing_path"],
                "grounding": "proxy",
            },
            {
                "name": "other: lateral",
                "required_predicates": ["lane_changing", "crossing_path"],
                "grounding": "derived",
            },
            {
                "name": "other: parked_lead",
                "required_predicates": ["in_front_of", "stationary"],
                "grounding": "derived",
            },
            {
                "name": "other: unmapped",
                "required_predicates": ["parked", "stopped"],
                "grounding": "derived",
            },
        ]
    },
    "grounding": {"ungrounded": {}},
}


@pytest.fixture()
def oracle(tmp_path: Path):
    path = tmp_path / "synthetic_oracle.json"
    path.write_text(json.dumps(SYNTHETIC_ORACLE), encoding="utf-8")
    return load_oracle(path)


def _actor(
    actor_id: int,
    type_id: str,
    distance: float,
    *,
    in_front: bool = True,
    same_road: bool = True,
    same_lane: bool = False,
    speed: float = 0.0,
    yaw: float = 0.0,
    lane: dict | None = None,
    lane_relation: str | None = None,
    adjacent_lane: bool = False,
    oncoming: bool = False,
    track: tuple[dict, ...] = (),
) -> dict:
    return {
        "id": actor_id,
        "type_id": type_id,
        "distance_to_ego_m": distance,
        "location": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rotation": {"pitch": 0.0, "yaw": yaw, "roll": 0.0},
        "velocity_mps": speed,
        "same_road_as_ego": same_road,
        "same_lane_as_ego": same_lane,
        "is_in_front_of_ego": in_front,
        "lane": lane,
        "lane_relation": lane_relation,
        "adjacent_lane_as_ego": adjacent_lane,
        "oncoming_as_ego": oncoming,
        "waypoint_history": list(track),
    }


def _tick(
    index: int,
    *,
    speed: float = 0.0,
    brake: float = 0.0,
    steer: float = 0.0,
    lane_id: int = 0,
    yaw: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    actors: tuple[dict, ...] = (),
    ego_lane: dict | None = None,
    ego_track: tuple[dict, ...] = (),
) -> dict:
    return {
        "tick": index,
        "scenario_id": "synthetic",
        "town": "Town01",
        "harness_state": {},
        "ego": {
            "id": 1,
            "location": {"x": x, "y": y, "z": 0.0},
            "rotation": {"pitch": 0.0, "yaw": yaw, "roll": 0.0},
            "velocity_mps": speed,
            "waypoint": {"road_id": 10, "lane_id": lane_id, "s": 0.0, "is_junction": False},
            "lane": ego_lane if ego_lane is not None else {"road_id": 10, "lane_id": lane_id},
            "waypoint_history": list(ego_track),
        },
        "telemetry": {
            "speed_mps": speed,
            "control": {"throttle": 0.0, "steer": steer, "brake": brake, "gear": 1},
        },
        "nearby_actors": list(actors),
    }


def _obligation(report: dict, axis: str, signature: str) -> dict:
    for row in report["dimensions"][axis]["obligations"]:
        if row["signature"] == signature:
            return row
    raise AssertionError(f"{signature!r} not found in axis {axis}")


def test_predicate_witnessed_at_tick(oracle):
    report = compute_cov(oracle, [_tick(0, speed=0.0)])

    stationary = _obligation(report, "A", "stationary(ego)")
    assert stationary["mapped"] is True
    assert stationary["covered"] is True
    assert stationary["witness_tick"] == 0
    assert stationary["grounding"] == "derived"
    assert "speed" in stationary["evidence"]


def test_obligation_never_witnessed(oracle):
    report = compute_cov(oracle, [_tick(0, speed=0.0)])

    moving = _obligation(report, "A", "moving(vehicle)")
    assert moving["mapped"] is True
    assert moving["covered"] is False
    assert moving["witness_tick"] is None
    assert "never witnessed" in moving["evidence"]


def test_unmapped_predicate_counted_as_unmapped(oracle):
    trace = [
        _tick(
            0,
            speed=5.0,
            actors=(_actor(7, "walker.pedestrian.0001", 20.0, in_front=False),),
        )
    ]
    report = compute_cov(oracle, trace)

    animal = _obligation(report, "V", "node(animal)")
    assert animal["mapped"] is False
    assert animal["covered"] is False
    assert animal["unmapped_reason"]

    pedestrian = _obligation(report, "V", "node(pedestrian)")
    assert pedestrian["mapped"] is True and pedestrian["covered"] is True

    assert report["dimensions"]["V"]["mapped_obligations"] == 2
    assert report["dimensions"]["V"]["unmapped_obligations"] == 1
    assert report["coverage_of_mapped_subset"]["V"] == pytest.approx(0.5)
    assert report["full_vocabulary_coverage"]["V"] == pytest.approx(1.0 / 3.0)
    assert "NOT full-vocabulary coverage" in report["warning"]


def test_hazard_conjunction_across_two_traces_not_credited(oracle):
    trace_a = [
        _tick(
            0,
            actors=(
                _actor(7, "walker.pedestrian.0001", 20.0, in_front=False, same_road=True, speed=1.0),
            ),
        )
    ]
    trace_b = [
        _tick(
            0,
            actors=(
                _actor(8, "walker.pedestrian.0001", 5.0, in_front=True, same_road=False, speed=1.0),
            ),
        )
    ]
    report = compute_suite_cov(
        oracle,
        [TraceInput(label="a", ticks=trace_a), TraceInput(label="b", ticks=trace_b)],
    )

    hazard = _obligation(report, "H", "hazard(other: pedestrian)")
    assert hazard["mapped"] is True
    assert hazard["covered"] is False, "hazard must not be assembled across two traces"

    jaywalking = _obligation(report, "A", "jaywalking(pedestrian)")
    crossing = _obligation(report, "E", "crossing_path(pedestrian,ego)")
    assert jaywalking["covered"] is True and jaywalking["witness_trace"] == "a"
    assert crossing["covered"] is True and crossing["witness_trace"] == "b"


def test_hazard_satisfied_in_one_trace_credited(oracle):
    trace = [
        _tick(
            0,
            actors=(
                _actor(7, "walker.pedestrian.0001", 5.0, in_front=True, same_road=True, speed=0.0),
            ),
        )
    ]
    report = compute_cov(oracle, trace)

    hazard = _obligation(report, "H", "hazard(other: pedestrian)")
    assert hazard["covered"] is True
    assert hazard["witness_tick"] == 0
    assert hazard["grounding"] == "proxy"
    assert hazard["predicate_ticks"]["jaywalking"] == [0]
    assert hazard["predicate_ticks"]["crossing_path"] == [0]


def test_hazard_window_enforced(oracle):
    trace = [
        _tick(0, actors=(_actor(7, "walker.pedestrian.0001", 20.0, in_front=False, same_road=True, speed=1.0),)),
        _tick(10, actors=(_actor(7, "walker.pedestrian.0001", 5.0, in_front=True, same_road=False, speed=1.0),)),
    ]
    narrow = compute_cov(oracle, trace, config=DerivationConfig(hazard_window_ticks=6))
    wide = compute_cov(oracle, trace, config=DerivationConfig(hazard_window_ticks=20))

    assert _obligation(narrow, "H", "hazard(other: pedestrian)")["covered"] is False
    assert _obligation(wide, "H", "hazard(other: pedestrian)")["covered"] is True


def test_unmapped_hazard_reports_reason(oracle):
    report = compute_cov(oracle, [_tick(0, speed=0.0)])
    unmapped = _obligation(report, "H", "hazard(other: unmapped)")
    assert unmapped["mapped"] is False
    assert "parked" in unmapped["unmapped_reason"]

    lateral = _obligation(report, "H", "hazard(other: lateral)")
    assert lateral["mapped"] is True, "lane_changing(vehicle) is observable with schema v2"


def test_hazard_requires_matching_node_type(oracle):
    sign_trace = [
        _tick(
            0,
            actors=(
                _actor(66, "traffic.speed_limit.30", 5.0, in_front=True, same_lane=True, speed=0.0),
            ),
        )
    ]
    vehicle_trace = [
        _tick(
            0,
            actors=(
                _actor(67, "vehicle.tesla.model3", 5.0, in_front=True, same_lane=True, speed=0.0),
            ),
        )
    ]
    sign_report = compute_cov(oracle, sign_trace)
    vehicle_report = compute_cov(oracle, vehicle_trace)

    assert _obligation(sign_report, "H", "hazard(other: parked_lead)")["covered"] is False
    parked_lead = _obligation(vehicle_report, "H", "hazard(other: parked_lead)")
    assert parked_lead["covered"] is True
    assert parked_lead["witness_actor"].startswith("vehicle#")


def test_grounding_classes_recorded(oracle):
    trace = [
        _tick(
            0,
            speed=0.0,
            actors=(
                _actor(7, "walker.pedestrian.0001", 5.0, in_front=True, same_road=True),
            ),
        )
    ]
    report = compute_cov(oracle, trace)
    assert _obligation(report, "E", "in_front_of(pedestrian,ego)")["grounding"] == "direct"
    assert _obligation(report, "A", "jaywalking(pedestrian)")["grounding"] == "proxy"
    assert _obligation(report, "A", "stationary(ego)")["grounding"] == "derived"


def test_compute_cov_accepts_raw_dicts(oracle):
    report = compute_cov(oracle, [_tick(0, speed=0.0)])
    assert _obligation(report, "A", "stationary(ego)")["covered"] is True


def test_cli_writes_report(oracle, tmp_path: Path):
    stream = tmp_path / "stream.jsonl"
    stream.write_text(json.dumps(_tick(0, speed=0.0)) + "\n", encoding="utf-8")
    oracle_path = tmp_path / "synthetic_oracle.json"
    oracle_path.write_text(json.dumps(SYNTHETIC_ORACLE), encoding="utf-8")
    output = tmp_path / "report.json"

    exit_code = main(
        ["--oracle", str(oracle_path), "--traces", str(stream), "--output", str(output), "--format", "json"]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["Cov_A"] is not None
    assert payload["mapped_obligation_count"] > 0
    assert "warning" in payload


# ---------------------------------------------------------------------------
# Schema v2 lane membership and swept-path crossing
# ---------------------------------------------------------------------------


def test_normalize_tick_reads_v2_lane_and_track():
    actor_track = (
        {"tick": 0, "x": -2.0, "y": 8.0, "yaw_deg": 0.0, "velocity_mps": 5.0},
        {"tick": 1, "x": 1.5, "y": 8.0, "yaw_deg": 0.0, "velocity_mps": 5.0},
    )
    ego_track = (
        {"tick": 0, "x": 1.5, "y": 0.0, "yaw_deg": 90.0, "velocity_mps": 10.0},
        {"tick": 1, "x": 1.5, "y": 1.0, "yaw_deg": 90.0, "velocity_mps": 10.0},
    )
    raw = _tick(
        1,
        speed=10.0,
        lane_id=-1,
        yaw=90.0,
        x=1.5,
        y=1.0,
        actors=(
            _actor(
                7,
                "walker.pedestrian.0001",
                8.0,
                in_front=True,
                lane={"road_id": 10, "lane_id": -2},
                lane_relation="right",
                adjacent_lane=True,
                oncoming=False,
                track=actor_track,
            ),
        ),
        ego_track=ego_track,
    )

    normalized = normalize_tick(raw)

    assert normalized.ego_lane is not None and normalized.ego_lane.lane_id == -1
    assert len(normalized.ego_track) == 2
    actor = normalized.actors[0]
    assert actor.lane is not None and actor.lane.lane_id == -2
    assert actor.lane_relation == "right"
    assert actor.adjacent_lane is True
    assert len(actor.track) == 2


def test_v1_row_normalizes_without_new_fields():
    raw = {
        "tick": 0,
        "scenario_id": "v1",
        "town": "Town01",
        "ego": {
            "id": 1,
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"pitch": 0.0, "yaw": 90.0, "roll": 0.0},
            "velocity_mps": 0.0,
            "waypoint": {"road_id": 10, "lane_id": -1, "s": 0.0, "is_junction": False},
        },
        "telemetry": {"speed_mps": 0.0, "control": {"throttle": 0.0, "steer": 0.0, "brake": 0.0}},
        "nearby_actors": [
            {
                "id": 7,
                "type_id": "walker.pedestrian.0001",
                "distance_to_ego_m": 8.0,
                "location": {"x": 0.0, "y": 8.0, "z": 0.0},
                "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                "velocity_mps": 0.0,
                "same_road_as_ego": True,
                "same_lane_as_ego": False,
                "is_in_front_of_ego": True,
            }
        ],
    }

    normalized = normalize_tick(raw)

    assert normalized.ego_lane is not None and normalized.ego_lane.lane_id == -1
    assert normalized.ego_track == ()
    assert normalized.actors[0].lane is None
    assert normalized.actors[0].track == ()
    assert normalized.actors[0].adjacent_lane is False
    assert normalized.actors[0].oncoming_flag is False


def test_v2_lane_membership_recomputes_same_adjacent_oncoming(oracle):
    follower = _actor(
        7,
        "vehicle.tesla.model3",
        10.0,
        in_front=True,
        same_road=True,
        lane={"road_id": 10, "lane_id": -1},
        yaw=90.0,
        speed=5.0,
    )
    oncoming = _actor(
        8,
        "vehicle.tesla.model3",
        20.0,
        in_front=True,
        same_road=True,
        lane={"road_id": 10, "lane_id": 1},
        yaw=270.0,
        speed=5.0,
    )
    report = compute_cov(oracle, [_tick(0, speed=5.0, lane_id=-1, yaw=90.0, actors=(follower, oncoming))])

    same_lane = _obligation(report, "E", "same_lane(vehicle,ego)")
    adjacent_lane = _obligation(report, "E", "adjacent_lane(vehicle,ego)")
    oncoming = _obligation(report, "E", "oncoming(vehicle,ego)")
    assert same_lane["covered"] is True
    assert adjacent_lane["covered"] is True
    assert oncoming["covered"] is True
    assert "lane" in same_lane["predicate_sources"]["same_lane"]
    assert "adjacency" in adjacent_lane["predicate_sources"]["adjacent_lane"]


def test_v2_lane_change_vehicle_from_persisted_lane_history(oracle):
    lane_a = {"road_id": 10, "lane_id": -1}
    lane_b = {"road_id": 10, "lane_id": -2}
    tick0 = _tick(
        0,
        speed=5.0,
        lane_id=-1,
        actors=(_actor(7, "vehicle.tesla.model3", 10.0, lane=lane_a, yaw=90.0, speed=5.0),),
    )
    tick1 = _tick(
        1,
        speed=5.0,
        lane_id=-1,
        actors=(_actor(7, "vehicle.tesla.model3", 10.0, lane=lane_b, yaw=90.0, speed=5.0),),
    )
    report = compute_cov(oracle, [tick0, tick1])

    row = _obligation(report, "A", "lane_changing(vehicle)")
    assert row["covered"] is True
    assert "changed" in row["predicate_sources"]["lane_changing"]


def test_v2_crossing_path_uses_swept_track_not_corridor_proxy(oracle):
    ego_track = (
        {"tick": 0, "x": 1.5, "y": 0.0, "yaw_deg": 90.0, "velocity_mps": 10.0},
        {"tick": 1, "x": 1.5, "y": 1.0, "yaw_deg": 90.0, "velocity_mps": 10.0},
    )
    actor_track = (
        {"tick": 0, "x": -2.0, "y": 8.0, "yaw_deg": 0.0, "velocity_mps": 5.0},
        {"tick": 1, "x": 1.5, "y": 8.0, "yaw_deg": 0.0, "velocity_mps": 5.0},
    )
    pedestrian = _actor(
        7,
        "walker.pedestrian.0001",
        8.0,
        in_front=True,
        same_road=True,
        lane={"road_id": 10, "lane_id": -1},
        yaw=0.0,
        speed=5.0,
        track=actor_track,
    )
    report = compute_cov(
        oracle,
        [_tick(1, speed=10.0, lane_id=-1, yaw=90.0, x=1.5, y=1.0, actors=(pedestrian,), ego_track=ego_track)],
    )

    row = _obligation(report, "E", "crossing_path(pedestrian,ego)")
    assert row["covered"] is True
    assert "swept-path" in row["predicate_sources"]["crossing_path"]


def test_v1_stream_falls_back_to_documented_proxy_and_heading_rules(oracle):
    tick = _tick(
        0,
        speed=0.0,
        lane_id=-1,
        yaw=90.0,
        actors=(
            _actor(7, "walker.pedestrian.0001", 8.0, in_front=True, same_road=True, speed=0.0),
        ),
    )
    crossing = compute_cov(oracle, [tick])
    crossing_row = _obligation(crossing, "E", "crossing_path(pedestrian,ego)")
    assert crossing_row["covered"] is True
    assert "PROXY fallback" in crossing_row["predicate_sources"]["crossing_path"]

    v1_oncoming_tick = _tick(
        0,
        speed=5.0,
        lane_id=-1,
        yaw=90.0,
        actors=(
            _actor(8, "vehicle.tesla.model3", 20.0, in_front=True, same_road=True, yaw=270.0, speed=5.0),
        ),
    )
    oncoming = compute_cov(oracle, [v1_oncoming_tick])
    oncoming_row = _obligation(oncoming, "E", "oncoming(vehicle,ego)")
    assert oncoming_row["covered"] is True
    assert "v1 fallback" in oncoming_row["predicate_sources"]["oncoming"]


def _slow_crossing_ticks(count: int = 41) -> list[dict]:
    """Ego stopped at the origin; slow walker crosses the route line x=0 at y=35."""
    walker_track = tuple(
        {"tick": tick, "x": -2.0 + 0.1 * tick, "y": 35.0, "yaw_deg": 0.0, "velocity_mps": 0.1}
        for tick in range(count)
    )
    pedestrian = _actor(
        7,
        "walker.pedestrian.0001",
        35.0,
        in_front=True,
        same_road=True,
        speed=0.1,
        track=walker_track,
    )
    return [_tick(tick, speed=0.0, yaw=90.0, actors=(pedestrian,)) for tick in range(count)]


EGO_ROUTE_TO_CROSSING = [{"x": 0.0, "y": 10.0 * step} for step in range(7)]


def test_crossing_path_with_ego_route_when_ego_stopped_short(oracle):
    ticks = _slow_crossing_ticks()

    legacy = compute_cov(oracle, ticks)
    assert _obligation(legacy, "E", "crossing_path(pedestrian,ego)")["covered"] is False

    routed = compute_cov(oracle, ticks, ego_route=EGO_ROUTE_TO_CROSSING)
    row = _obligation(routed, "E", "crossing_path(pedestrian,ego)")
    assert row["covered"] is True
    assert "planned route" in row["predicate_sources"]["crossing_path"]


def test_compute_suite_cov_threads_trace_input_ego_route(oracle):
    ticks = _slow_crossing_ticks()
    report = compute_suite_cov(
        oracle,
        [TraceInput(label="routed", ticks=ticks, ego_route=tuple(EGO_ROUTE_TO_CROSSING))],
    )
    assert _obligation(report, "E", "crossing_path(pedestrian,ego)")["covered"] is True


def test_load_ego_route_reads_first_record_and_legacy_streams_yield_empty(tmp_path: Path):
    route = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
    stream = tmp_path / "routed.jsonl"
    stream.write_text(
        json.dumps({**_tick(0), "ego_route": route}) + "\n" + json.dumps(_tick(1)) + "\n",
        encoding="utf-8",
    )
    assert load_ego_route(stream) == route

    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(json.dumps(_tick(0)) + "\n", encoding="utf-8")
    assert load_ego_route(legacy) == []

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert load_ego_route(empty) == []


def test_cli_auto_reads_ego_route_from_stream(oracle, tmp_path: Path):
    ticks = _slow_crossing_ticks()
    ticks[0]["ego_route"] = EGO_ROUTE_TO_CROSSING
    stream = tmp_path / "routed.jsonl"
    stream.write_text("\n".join(json.dumps(tick) for tick in ticks) + "\n", encoding="utf-8")
    oracle_path = tmp_path / "synthetic_oracle.json"
    oracle_path.write_text(json.dumps(SYNTHETIC_ORACLE), encoding="utf-8")
    output = tmp_path / "report.json"

    exit_code = main(
        ["--oracle", str(oracle_path), "--traces", str(stream), "--output", str(output), "--format", "json"]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert _obligation(payload, "E", "crossing_path(pedestrian,ego)")["covered"] is True


REAL_STREAM = (
    REPO_ROOT
    / "research/logs/semantic/town01_spawn0_goal82_threshold_crossing-pcla-20260315T012013Z-semantic-stream.jsonl"
)


@pytest.mark.skipif(not REAL_STREAM.exists(), reason="real semantic stream not present in this checkout")
def test_real_stream_normalization_and_derivation(oracle):
    ticks = load_semantic_trace(REAL_STREAM)
    assert ticks, "real stream should contain ticks"
    first = ticks[0]
    assert first.scenario_id
    assert first.ego_speed_mps >= 0.0

    report = compute_cov(oracle, ticks, trace_label=str(REAL_STREAM))
    assert report["traces"][0]["ticks"] == len(ticks)
    for axis in ("V", "A", "E", "H"):
        cov = report["coverage_of_mapped_subset"][axis]
        assert cov is None or 0.0 <= cov <= 1.0
