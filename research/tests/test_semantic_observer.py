"""Tests for research/harness/observers/semantic.py schema-v2 lane relations.

The fakes reuse the CARLA map/waypoint mocks from ``test_lane_index`` (straight
two-way road; the docstring there records the API-shape assumptions).  No live
CARLA is required.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.harness.lane_index import SEMANTIC_STREAM_SCHEMA_VERSION  # noqa: E402
from research.harness.coverage_engine import derive_trace, load_ego_route, normalize_tick  # noqa: E402
from research.harness.models import RunResult, ScenarioSpec  # noqa: E402
from research.harness.observers.semantic import SemanticObserver, SemanticObserverConfig  # noqa: E402
from research.tests.test_lane_index import (  # noqa: E402
    FakeLocation,
    FakeMap,
    FakeTransform,
    FakeVector,
    build_two_way_road,
)


class FakeActor:
    def __init__(
        self,
        actor_id: int,
        type_id: str,
        x: float,
        y: float,
        yaw: float,
        speed: float = 0.0,
        alive: bool = True,
    ) -> None:
        self.id = actor_id
        self.type_id = type_id
        self.is_alive = alive
        self._transform = FakeTransform(FakeLocation(x, y), yaw)
        self._speed = speed

    def set_pose(self, x: float, y: float, yaw: float | None = None, speed: float | None = None) -> None:
        if yaw is None:
            yaw = self._transform.rotation.yaw
        if speed is not None:
            self._speed = speed
        self._transform = FakeTransform(FakeLocation(x, y), yaw)

    def get_transform(self) -> FakeTransform:
        return self._transform

    def get_location(self) -> FakeLocation:
        return self._transform.location

    def get_velocity(self) -> FakeVector:
        yaw_rad = math.radians(self._transform.rotation.yaw)
        return FakeVector(
            self._speed * math.cos(yaw_rad),
            self._speed * math.sin(yaw_rad),
            0.0,
        )


class FakeWorld:
    def __init__(self, fake_map: FakeMap | None, actors: list[FakeActor]) -> None:
        self._fake_map = fake_map
        self._actors = actors
        self._no_map = FakeMap([])

    def get_map(self) -> FakeMap:
        return self._fake_map if self._fake_map is not None else self._no_map

    def get_actors(self) -> list[FakeActor]:
        return list(self._actors)


def _scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="synthetic-lane-test",
        town="Town01",
        weather_preset="ClearNoon",
        ego_spawn_index=0,
        goal_spawn_index=1,
        description="lane index observer test",
    )


def _telemetry(tick: int, speed: float = 0.0) -> dict:
    return {
        "tick": tick,
        "speed_mps": speed,
        "speed_kph": speed * 3.6,
        "distance_to_goal_m": 100.0,
        "control": {"throttle": 0.5, "steer": 0.0, "brake": 0.0, "gear": 1},
        "agent_step": {"route_progress": {}},
    }


def _run_observer(
    tmp_path: Path,
    world: FakeWorld,
    ego: FakeActor,
    ticks: list[dict],
    *,
    jaywalker: FakeActor | None = None,
    threshold_active: bool = False,
    tick_hooks: list | None = None,
    route_waypoints: list | None = None,
    waypoint_history_ticks: int | None = None,
) -> tuple[RunResult, list[dict]]:
    config = SemanticObserverConfig(
        source="heuristic",
        stream_output_dir=tmp_path / "semantic",
        trace_output_dir=tmp_path / "semantic",
        stream_every_ticks=1,
        nearby_actor_radius_m=50.0,
    )
    if waypoint_history_ticks is not None:
        config.waypoint_history_ticks = waypoint_history_ticks
    observer = SemanticObserver(config)
    scenario = _scenario()
    context: dict = {
        "world": world,
        "ego_vehicle": ego,
        "scenario": scenario,
        "threshold_adversary_kind": "walker" if jaywalker is not None else None,
        "threshold_adversary": jaywalker,
        "threshold_adversary_active": threshold_active,
        "threshold_adversary_triggered": threshold_active,
        "threshold_trigger_distance_m": None,
        "route_waypoints": route_waypoints,
    }
    observer.on_run_start(scenario, context)
    for tick_index, telemetry in enumerate(ticks):
        if tick_hooks is not None and tick_hooks[tick_index] is not None:
            tick_hooks[tick_index]()
        context["telemetry"] = telemetry
        observer.on_tick(tick_index, context)

    result = RunResult(
        scenario_id=scenario.scenario_id,
        town=scenario.town,
        agent_kind="test",
        succeeded=True,
        dry_run=False,
        ticks_executed=len(ticks),
        reached_goal=False,
        terminated_by_collision=False,
        collision_count=0,
    )
    observer.on_run_end(result, context)

    stream_path = Path(result.metadata["semantic"]["stream_dump_path"])
    frames = [json.loads(line) for line in stream_path.read_text(encoding="utf-8").splitlines() if line]
    return result, frames


def test_stream_schema_v2_persists_lane_membership_and_tracks(tmp_path: Path):
    fake_map = build_two_way_road()
    ego = FakeActor(1, "vehicle.tesla.model3", x=1.5, y=0.0, yaw=90.0)
    follower = FakeActor(2, "vehicle.tesla.model3", x=1.5, y=10.0, yaw=90.0)
    world = FakeWorld(fake_map, [ego, follower])

    result, frames = _run_observer(
        tmp_path,
        world,
        ego,
        ticks=[_telemetry(0), _telemetry(1)],
    )

    assert frames and all(frame["schema_version"] == SEMANTIC_STREAM_SCHEMA_VERSION for frame in frames)
    assert result.metadata["semantic"]["stream_schema_version"] == SEMANTIC_STREAM_SCHEMA_VERSION

    last = frames[-1]
    assert last["ego"]["lane"]["road_id"] == 10
    assert last["ego"]["lane"]["lane_id"] == -1
    assert len(last["ego"]["waypoint_history"]) == 2

    actor_entry = next(item for item in last["nearby_actors"] if item["id"] == 2)
    assert actor_entry["lane"]["road_id"] == 10
    assert actor_entry["lane"]["lane_id"] == -1
    assert actor_entry["lane_relation"] == "same"
    assert actor_entry["same_lane_as_ego"] is True
    assert actor_entry["adjacent_lane_as_ego"] is False
    assert actor_entry["oncoming_as_ego"] is False
    assert len(actor_entry["waypoint_history"]) == 2

    covered = set(result.metadata["semantic"]["covered_predicates"])
    assert {"same_lane", "in_front_of", "obstructing"} <= covered


def test_lane_index_emits_adjacent_and_oncoming(tmp_path: Path):
    fake_map = build_two_way_road()
    ego = FakeActor(1, "vehicle.tesla.model3", x=1.5, y=0.0, yaw=90.0, speed=5.0)
    oncoming = FakeActor(2, "vehicle.tesla.model3", x=-1.5, y=15.0, yaw=-90.0, speed=10.0)
    same_direction_adjacent = FakeActor(3, "vehicle.tesla.model3", x=4.5, y=10.0, yaw=90.0, speed=8.0)
    world = FakeWorld(fake_map, [ego, oncoming, same_direction_adjacent])

    result, frames = _run_observer(
        tmp_path,
        world,
        ego,
        ticks=[_telemetry(0, speed=5.0), _telemetry(1, speed=5.0)],
    )

    by_id = {item["id"]: item for item in frames[-1]["nearby_actors"]}
    assert by_id[2]["lane_relation"] == "left"
    assert by_id[2]["adjacent_lane_as_ego"] is True
    assert by_id[2]["oncoming_as_ego"] is True
    assert by_id[3]["lane_relation"] == "right"
    assert by_id[3]["adjacent_lane_as_ego"] is True
    assert by_id[3]["oncoming_as_ego"] is False

    covered = set(result.metadata["semantic"]["covered_predicates"])
    assert {"adjacent_lane", "oncoming"} <= covered

    ticks = [normalize_tick(frame, index=index) for index, frame in enumerate(frames)]
    derived = derive_trace(ticks)
    predicates = {fact.predicate for fact in derived.facts}
    assert {"adjacent_lane", "oncoming"} <= predicates
    oncoming_facts = [fact for fact in derived.facts if fact.predicate == "oncoming"]
    assert oncoming_facts and "lane membership" in oncoming_facts[0].source


def test_crossing_path_via_swept_track_intersection(tmp_path: Path):
    fake_map = build_two_way_road()
    ego = FakeActor(1, "vehicle.tesla.model3", x=1.5, y=0.0, yaw=90.0, speed=10.0)
    pedestrian = FakeActor(2, "walker.pedestrian.0001", x=-2.0, y=8.0, yaw=0.0, speed=5.0)
    world = FakeWorld(fake_map, [ego, pedestrian])

    def place_at_start() -> None:
        pedestrian.set_pose(-2.0, 8.0, yaw=0.0, speed=5.0)

    def move_across() -> None:
        pedestrian.set_pose(1.5, 8.0, yaw=0.0, speed=5.0)

    result, frames = _run_observer(
        tmp_path,
        world,
        ego,
        ticks=[_telemetry(0, speed=10.0), _telemetry(1, speed=10.0)],
        tick_hooks=[place_at_start, move_across],
    )

    last = frames[-1]
    actor_entry = next(item for item in last["nearby_actors"] if item["id"] == 2)
    assert len(actor_entry["waypoint_history"]) == 2
    assert actor_entry["waypoint_history"][0]["x"] == -2.0
    assert actor_entry["waypoint_history"][1]["x"] == 1.5

    covered = set(result.metadata["semantic"]["covered_predicates"])
    assert "crossing_path" in covered

    ticks = [normalize_tick(frame, index=index) for index, frame in enumerate(frames)]
    derived = derive_trace(ticks)
    crossing_facts = [fact for fact in derived.facts if fact.predicate == "crossing_path"]
    assert crossing_facts, "coverage engine must re-derive crossing_path from the persisted track"
    assert any("swept-path" in fact.source for fact in crossing_facts)


def test_crossing_path_proxy_fallback_without_geometry(tmp_path: Path):
    ego = FakeActor(1, "vehicle.tesla.model3", x=0.0, y=0.0, yaw=90.0, speed=0.0)
    pedestrian = FakeActor(2, "walker.pedestrian.0001", x=0.0, y=6.0, yaw=0.0, speed=0.0)
    world = FakeWorld(None, [ego, pedestrian])  # map with no lanes -> no waypoints

    result, _ = _run_observer(
        tmp_path,
        world,
        ego,
        ticks=[_telemetry(0)],
    )

    covered = set(result.metadata["semantic"]["covered_predicates"])
    assert "crossing_path" in covered, "proxy fallback must fire when no geometry exists"


def test_semantic_stream_writer_keeps_v1_keys(tmp_path: Path):
    fake_map = build_two_way_road()
    ego = FakeActor(1, "vehicle.tesla.model3", x=1.5, y=0.0, yaw=90.0)
    world = FakeWorld(fake_map, [ego])

    _, frames = _run_observer(tmp_path, world, ego, ticks=[_telemetry(0)])

    frame = frames[0]
    for key in ("tick", "trigger_reason", "scenario_id", "town", "harness_state", "ego", "telemetry", "route_context", "nearby_actors"):
        assert key in frame, f"v1 key '{key}' missing from schema-v2 frame"
    assert frame["ego"]["waypoint"]["road_id"] == 10


# ---------------------------------------------------------------------------
# Route persistence and the slow-crossing track window
# ---------------------------------------------------------------------------


def test_first_stream_frame_persists_ego_route_once(tmp_path: Path):
    fake_map = build_two_way_road()
    ego = FakeActor(1, "vehicle.tesla.model3", x=1.5, y=0.0, yaw=90.0)
    world = FakeWorld(fake_map, [ego])
    route = [{"x": 1.5, "y": 0.0}, {"x": 1.5, "y": 10.0}, {"x": 1.5, "y": 20.0}]

    result, frames = _run_observer(
        tmp_path,
        world,
        ego,
        ticks=[_telemetry(0), _telemetry(1)],
        route_waypoints=route,
    )

    assert frames[0]["ego_route"] == route
    assert "ego_route" not in frames[1]

    stream_path = Path(result.metadata["semantic"]["stream_dump_path"])
    assert load_ego_route(stream_path) == route


def test_route_preview_fallback_shape_is_normalized(tmp_path: Path):
    fake_map = build_two_way_road()
    ego = FakeActor(1, "vehicle.tesla.model3", x=1.5, y=0.0, yaw=90.0)
    world = FakeWorld(fake_map, [ego])
    preview = [{"index": 0, "location": {"x": 1.5, "y": 0.0, "z": 0.0}}]

    _, frames = _run_observer(tmp_path, world, ego, ticks=[_telemetry(0)], route_waypoints=preview)

    assert frames[0]["ego_route"] == [{"x": 1.5, "y": 0.0}]


def test_stream_without_route_context_omits_ego_route(tmp_path: Path):
    fake_map = build_two_way_road()
    ego = FakeActor(1, "vehicle.tesla.model3", x=1.5, y=0.0, yaw=90.0)
    world = FakeWorld(fake_map, [ego])

    result, frames = _run_observer(tmp_path, world, ego, ticks=[_telemetry(0)])

    assert "ego_route" not in frames[0]
    stream_path = Path(result.metadata["semantic"]["stream_dump_path"])
    assert load_ego_route(stream_path) == []


def _slow_crossing_hooks(ego: FakeActor, pedestrian: FakeActor, ticks: int) -> list:
    """Ego creeps forward at 0.3 m/tick; walker crosses laterally at 0.1 m/tick."""
    hooks = []
    for tick in range(ticks):

        def hook(t: int = tick) -> None:
            ego.set_pose(1.5, 0.3 * t, yaw=90.0, speed=0.3)
            pedestrian.set_pose(-2.0 + 0.1 * t, 20.0, yaw=0.0, speed=0.1)

        hooks.append(hook)
    return hooks


def test_slow_crossing_credited_with_50_tick_history(tmp_path: Path):
    ticks_count = 80

    wide_map = build_two_way_road()
    wide_ego = FakeActor(1, "vehicle.tesla.model3", x=1.5, y=0.0, yaw=90.0, speed=0.3)
    wide_pedestrian = FakeActor(2, "walker.pedestrian.0001", x=-2.0, y=20.0, yaw=0.0, speed=0.1)
    wide_world = FakeWorld(wide_map, [wide_ego, wide_pedestrian])

    wide_result, wide_frames = _run_observer(
        tmp_path / "wide",
        wide_world,
        wide_ego,
        ticks=[_telemetry(tick, speed=0.3) for tick in range(ticks_count)],
        tick_hooks=_slow_crossing_hooks(wide_ego, wide_pedestrian, ticks_count),
    )

    assert "crossing_path" in set(wide_result.metadata["semantic"]["covered_predicates"])
    assert len(wide_frames[-1]["ego"]["waypoint_history"]) == 50
    assert len(wide_frames[-1]["nearby_actors"][0]["waypoint_history"]) == 50

    narrow_map = build_two_way_road()
    narrow_ego = FakeActor(1, "vehicle.tesla.model3", x=1.5, y=0.0, yaw=90.0, speed=0.3)
    narrow_pedestrian = FakeActor(2, "walker.pedestrian.0001", x=-2.0, y=20.0, yaw=0.0, speed=0.1)
    narrow_world = FakeWorld(narrow_map, [narrow_ego, narrow_pedestrian])

    narrow_result, narrow_frames = _run_observer(
        tmp_path / "narrow",
        narrow_world,
        narrow_ego,
        ticks=[_telemetry(tick, speed=0.3) for tick in range(ticks_count)],
        tick_hooks=_slow_crossing_hooks(narrow_ego, narrow_pedestrian, ticks_count),
        waypoint_history_ticks=10,
    )

    assert "crossing_path" not in set(narrow_result.metadata["semantic"]["covered_predicates"])
    assert len(narrow_frames[-1]["nearby_actors"][0]["waypoint_history"]) == 10
