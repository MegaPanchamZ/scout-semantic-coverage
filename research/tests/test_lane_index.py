"""Synthetic-fixture tests for research/harness/lane_index.py.

No live CARLA is required.  The fakes below mirror the documented shapes of the
CARLA map API surface that ``lane_index`` uses:

* ``Map.get_waypoint(location, project_to_road=True)`` -> ``Waypoint`` or None
  (with ``project_to_road=False`` returning None off-road);
* ``Waypoint.road_id``, ``lane_id``, ``s``, ``is_junction``, ``lane_width``,
  ``lane_type`` and ``transform`` (``location`` + ``rotation.yaw``);
* ``Waypoint.get_left_lane()`` / ``get_right_lane()`` -> ``Waypoint`` or None;
* ``Waypoint.next(distance)`` -> list of ``Waypoint`` (first entry followed).

Assumption recorded for live validation: CARLA lane ids in the fakes follow the
OpenDRIVE ordinal convention (+1 immediately left of the reference line, -1
immediately right) and opposite-direction traffic occupies the opposite side of
the road.  ``get_left_lane`` / ``get_right_lane`` responses are authoritative
when a live waypoint is available; the ordinal fallback is exercised separately.

Road layout used by the fixtures (road 10, positive lanes travel -y, negative
lanes travel +y, i.e. right-hand traffic):

    lane  2   |  lane  1  ||  ref  ||  lane -1  |  lane -2
    x=-4.5       x=-1.5               x=+1.5       x=+4.5
    yaw=-90      yaw=-90              yaw=+90      yaw=+90
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.harness.lane_index import (  # noqa: E402
    CrossingPathConfig,
    LaneRef,
    TrackSample,
    angle_difference_deg,
    classify_lane_relation,
    detect_crossing_from_tracks,
    detect_lane_change,
    detect_path_crossing,
    lane_ref_from_waypoint,
    motion_swept_path,
    neighbour_lane_refs,
    relative_lane_from_lane_ids,
    route_polyline,
    track_polyline,
    waypoint_swept_path,
    wrap_angle_deg,
)


# ---------------------------------------------------------------------------
# Fake CARLA API
# ---------------------------------------------------------------------------


class FakeRotation:
    def __init__(self, yaw: float) -> None:
        self.pitch = 0.0
        self.yaw = float(yaw)
        self.roll = 0.0


class FakeLocation:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def distance(self, other: "FakeLocation") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class FakeVector:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class FakeTransform:
    def __init__(self, location: FakeLocation, yaw: float) -> None:
        self.location = location
        self.rotation = FakeRotation(yaw)

    def get_forward_vector(self) -> FakeVector:
        yaw_rad = math.radians(self.rotation.yaw)
        return FakeVector(math.cos(yaw_rad), math.sin(yaw_rad), 0.0)


class FakeLane:
    def __init__(self, road_id: int, lane_id: int, center_x: float, yaw_deg: float, width: float = 3.5) -> None:
        self.road_id = road_id
        self.lane_id = lane_id
        self.center_x = center_x
        self.yaw_deg = yaw_deg
        self.width = width


class FakeWaypoint:
    def __init__(self, lane: FakeLane, fake_map: "FakeMap", s: float = 0.0, is_junction: bool = False) -> None:
        self._lane = lane
        self._map = fake_map
        self.road_id = lane.road_id
        self.lane_id = lane.lane_id
        self.s = float(s)
        self.is_junction = is_junction
        self.lane_width = lane.width
        self.lane_type = "LaneType.Driving"
        self.transform = FakeTransform(FakeLocation(lane.center_x, s), lane.yaw_deg)

    def next(self, distance: float) -> list["FakeWaypoint"]:
        if self.is_junction or self._lane is None:
            return []
        yaw_rad = math.radians(self._lane.yaw_deg)
        new_s = self.s + distance * math.sin(yaw_rad)
        if abs(new_s) > 1000.0:
            return []
        return [FakeWaypoint(self._lane, self._map, s=new_s)]

    def get_left_lane(self) -> "FakeWaypoint | None":
        neighbour = self._map.spatial_neighbour(self._lane, -1)
        return None if neighbour is None else FakeWaypoint(neighbour, self._map, s=self.s)

    def get_right_lane(self) -> "FakeWaypoint | None":
        neighbour = self._map.spatial_neighbour(self._lane, +1)
        return None if neighbour is None else FakeWaypoint(neighbour, self._map, s=self.s)


class FakeMap:
    """Straight-road map with lanes indexed spatially from left to right."""

    def __init__(self, lanes: list[FakeLane]) -> None:
        self.lanes = list(lanes)

    def _lanes_for_road(self, road_id: int) -> list[FakeLane]:
        lanes = [lane for lane in self.lanes if lane.road_id == road_id]
        return sorted(lanes, key=lambda lane: lane.center_x)  # left (negative x) first

    def spatial_neighbour(self, lane: FakeLane, direction: int) -> FakeLane | None:
        ordered = self._lanes_for_road(lane.road_id)
        for index, candidate in enumerate(ordered):
            if candidate.lane_id == lane.lane_id:
                neighbour_index = index + direction
                if 0 <= neighbour_index < len(ordered):
                    return ordered[neighbour_index]
                return None
        return None

    def lane_for_location(self, location: FakeLocation) -> tuple[FakeLane | None, float]:
        best_lane = None
        best_distance = float("inf")
        for lane in self.lanes:
            distance = abs(location.x - lane.center_x)
            if distance < best_distance:
                best_distance = distance
                best_lane = lane
        return best_lane, best_distance

    def get_waypoint(self, location: FakeLocation, project_to_road: bool = True) -> FakeWaypoint | None:
        lane, distance = self.lane_for_location(location)
        if lane is None:
            return None
        if not project_to_road and distance > lane.width / 2.0:
            return None
        return FakeWaypoint(lane, self, s=location.y)


def build_two_way_road() -> FakeMap:
    return FakeMap(
        [
            FakeLane(road_id=10, lane_id=2, center_x=-4.5, yaw_deg=-90.0),
            FakeLane(road_id=10, lane_id=1, center_x=-1.5, yaw_deg=-90.0),
            FakeLane(road_id=10, lane_id=-1, center_x=1.5, yaw_deg=90.0),
            FakeLane(road_id=10, lane_id=-2, center_x=4.5, yaw_deg=90.0),
        ]
    )


def _relation(
    fake_map: FakeMap,
    ego_lane_id: int,
    actor_lane_id: int,
    actor_yaw: float,
    ego_yaw: float = 90.0,
) -> tuple[LaneRef, LaneRef, "object"]:
    ego_waypoint = fake_map.get_waypoint(FakeLocation(-99.0, 0.0))
    # Fetch lane centers directly so lane ids are unambiguous in the fixtures.
    ego_lane = next(lane for lane in fake_map.lanes if lane.lane_id == ego_lane_id and lane.road_id == 10)
    actor_lane = next(lane for lane in fake_map.lanes if lane.lane_id == actor_lane_id and lane.road_id == 10)
    ego_ref = lane_ref_from_waypoint(FakeWaypoint(ego_lane, fake_map, s=0.0))
    actor_ref = lane_ref_from_waypoint(FakeWaypoint(actor_lane, fake_map, s=10.0))
    relation = classify_lane_relation(
        ego_ref,
        actor_ref,
        actor_yaw_deg=actor_yaw,
        ego_yaw_deg=ego_yaw,
        ego_waypoint=FakeWaypoint(ego_lane, fake_map, s=0.0),
        actor_waypoint=FakeWaypoint(actor_lane, fake_map, s=10.0),
    )
    assert ego_waypoint is not None
    return ego_ref, actor_ref, relation


# ---------------------------------------------------------------------------
# Lane membership and adjacency
# ---------------------------------------------------------------------------


def test_lane_ref_from_waypoint_reads_fields():
    fake_map = build_two_way_road()
    waypoint = fake_map.get_waypoint(FakeLocation(1.5, 12.0))
    lane = lane_ref_from_waypoint(waypoint)

    assert lane is not None
    assert lane.road_id == 10
    assert lane.lane_id == -1
    assert lane.s == pytest.approx(12.0)
    assert lane.is_junction is False
    assert lane.yaw_deg == pytest.approx(90.0)
    assert lane.key == (10, -1)


def test_same_lane_relation():
    fake_map = build_two_way_road()
    _, _, relation = _relation(fake_map, ego_lane_id=-1, actor_lane_id=-1, actor_yaw=90.0)

    assert relation.same_road is True
    assert relation.same_lane is True
    assert relation.adjacent_lane is False
    assert relation.relative_lane == "same"
    assert relation.oncoming is False


def test_adjacent_left_lane_is_oncoming_direction():
    fake_map = build_two_way_road()
    _, actor_lane, relation = _relation(fake_map, ego_lane_id=-1, actor_lane_id=1, actor_yaw=-90.0)

    assert actor_lane.lane_id == 1
    assert relation.same_lane is False
    assert relation.adjacent_lane is True
    assert relation.relative_lane == "left"
    assert relation.oncoming is True, "opposed heading on a shared road"


def test_adjacent_right_lane_same_direction():
    fake_map = build_two_way_road()
    _, _, relation = _relation(fake_map, ego_lane_id=-1, actor_lane_id=-2, actor_yaw=90.0)

    assert relation.adjacent_lane is True
    assert relation.relative_lane == "right"
    assert relation.oncoming is False


def test_lane_two_away_is_not_adjacent():
    fake_map = build_two_way_road()
    _, _, relation = _relation(fake_map, ego_lane_id=2, actor_lane_id=-1, actor_yaw=-90.0)

    assert relation.same_lane is False
    assert relation.adjacent_lane is False
    assert relation.relative_lane is None


def test_oncoming_requires_shared_road():
    ego_lane = LaneRef(road_id=10, lane_id=-1)
    other_road_lane = LaneRef(road_id=11, lane_id=1)
    relation = classify_lane_relation(
        ego_lane,
        other_road_lane,
        actor_yaw_deg=-90.0,
        ego_yaw_deg=90.0,
    )

    assert relation.same_road is False
    assert relation.adjacent_lane is False
    assert relation.oncoming is False
    assert relation.reason == "different-road"


def test_junction_lane_membership_is_not_credited():
    ego_lane = LaneRef(road_id=10, lane_id=-1, is_junction=True)
    actor_lane = LaneRef(road_id=10, lane_id=-1, is_junction=True)
    relation = classify_lane_relation(
        ego_lane,
        actor_lane,
        actor_yaw_deg=90.0,
        ego_yaw_deg=90.0,
    )

    assert relation.same_road is True
    assert relation.same_lane is False
    assert relation.adjacent_lane is False
    assert relation.reason == "junction-membership-unstable"


def test_no_lane_match_returns_unknown():
    fake_map = build_two_way_road()
    off_road = fake_map.get_waypoint(FakeLocation(30.0, 0.0), project_to_road=False)
    assert off_road is None
    assert lane_ref_from_waypoint(off_road) is None

    relation = classify_lane_relation(
        None,
        None,
        actor_yaw_deg=0.0,
        ego_yaw_deg=0.0,
    )
    assert relation.same_lane is False
    assert relation.adjacent_lane is False
    assert relation.oncoming is False
    assert relation.reason == "lane-membership-unknown"


def test_neighbour_lane_refs_from_map_api():
    fake_map = build_two_way_road()
    ego_waypoint = FakeWaypoint(
        next(lane for lane in fake_map.lanes if lane.lane_id == -1), fake_map, s=0.0
    )
    left, right = neighbour_lane_refs(ego_waypoint)

    assert left is not None and left.lane_id == 1
    assert right is not None and right.lane_id == -2


def test_lane_ref_rejects_non_driving_lane_types():
    fake_map = build_two_way_road()
    waypoint = FakeWaypoint(
        next(lane for lane in fake_map.lanes if lane.lane_id == -1), fake_map, s=0.0
    )
    waypoint.lane_type = "LaneType.Sidewalk"
    assert lane_ref_from_waypoint(waypoint) is None

    # A non-driving neighbour must not be reported as left/right adjacent.
    driving_waypoint = FakeWaypoint(
        next(lane for lane in fake_map.lanes if lane.lane_id == -1), fake_map, s=0.0
    )
    original_right = driving_waypoint.get_right_lane

    def sidewalk_right() -> FakeWaypoint:
        neighbour = original_right()
        assert neighbour is not None
        neighbour.lane_type = "LaneType.Sidewalk"
        return neighbour

    driving_waypoint.get_right_lane = sidewalk_right  # type: ignore[method-assign]
    left, right = neighbour_lane_refs(driving_waypoint)
    assert left is not None and left.lane_id == 1
    assert right is None


def test_relative_lane_from_lane_ids_conventions():
    def relative(ego_id: int, actor_id: int) -> str | None:
        return relative_lane_from_lane_ids(LaneRef(10, ego_id), LaneRef(10, actor_id))

    assert relative(-1, -2) == "right"
    assert relative(-2, -1) == "left"
    assert relative(1, 2) == "left"
    assert relative(2, 1) == "right"
    assert relative(-1, 1) == "left", "positive lanes sit left of the reference line"
    assert relative(1, -1) == "right"
    assert relative(-1, 2) is None
    assert relative(-1, -1) == "same"
    assert relative(0, 1) is None


def test_detect_lane_change():
    assert detect_lane_change(LaneRef(10, -1), LaneRef(10, -2)) is True
    assert detect_lane_change(LaneRef(10, -2), LaneRef(10, -1)) is True
    assert detect_lane_change(LaneRef(10, -1), LaneRef(10, 1)) is True
    assert detect_lane_change(LaneRef(10, -1), LaneRef(10, -1)) is False
    assert detect_lane_change(LaneRef(10, -1), LaneRef(11, -1)) is False
    assert detect_lane_change(LaneRef(10, -1), LaneRef(10, 2)) is False
    assert detect_lane_change(None, LaneRef(10, -2)) is False


# ---------------------------------------------------------------------------
# Swept paths and crossing detection
# ---------------------------------------------------------------------------


def test_motion_swept_path_projection():
    path = motion_swept_path(0.0, 0.0, 90.0, 10.0, horizon_s=2.0, step_s=0.5)

    assert len(path) == 5
    assert path[0] == (0.0, 0.0)
    assert path[-1][1] == pytest.approx(20.0)
    assert motion_swept_path(0.0, 0.0, 0.0, 0.0) == ((0.0, 0.0),)


def test_waypoint_swept_path_follows_next():
    fake_map = build_two_way_road()
    waypoint = FakeWaypoint(
        next(lane for lane in fake_map.lanes if lane.lane_id == -1), fake_map, s=0.0
    )
    path = waypoint_swept_path(waypoint, horizon_m=10.0, step_m=2.0)

    assert len(path) == 6
    assert path[0] == pytest.approx((1.5, 0.0))
    assert path[-1][1] == pytest.approx(10.0)
    assert waypoint_swept_path(None) == ()


def test_detect_path_crossing_perpendicular():
    ego_path = [(1.5, 0.0), (1.5, 20.0)]
    actor_path = [(-5.0, 10.0), (5.0, 10.0)]

    assert detect_path_crossing(actor_path, ego_path) is True


def test_detect_path_crossing_parallel_is_not_crossing():
    ego_path = [(1.5, 0.0), (1.5, 20.0)]
    actor_path = [(-1.5, 0.0), (-1.5, 20.0)]

    assert detect_path_crossing(actor_path, ego_path) is False


def test_detect_path_crossing_requires_actor_displacement():
    ego_path = [(1.5, 0.0), (1.5, 20.0)]
    actor_path = [(1.4, 10.0), (1.7, 10.0)]

    assert detect_path_crossing(actor_path, ego_path, min_actor_displacement_m=0.5) is False
    assert detect_path_crossing(actor_path, ego_path, min_actor_displacement_m=0.1) is True


def test_detect_path_crossing_respects_corridor_and_along_path_extent():
    ego_path = [(0.0, 0.0), (0.0, 5.0)]
    near_miss_beyond_path = [(-1.0, 10.0), (1.0, 10.0)]

    assert detect_path_crossing(near_miss_beyond_path, ego_path) is False

    within_corridor = [(-1.0, 4.5), (1.0, 4.5)]
    assert detect_path_crossing(within_corridor, ego_path) is True


def test_detect_path_crossing_vehicle_heading_range():
    ego_path = [(0.0, 0.0), (0.0, 20.0)]
    actor_path = [(-5.0, 10.0), (5.0, 10.0)]

    assert (
        detect_path_crossing(
            actor_path,
            ego_path,
            heading_deltas=[90.0],
            heading_range_deg=(30.0, 150.0),
        )
        is True
    )
    assert (
        detect_path_crossing(
            actor_path,
            ego_path,
            heading_deltas=[180.0],
            heading_range_deg=(30.0, 150.0),
        )
        is False
    )


def test_detect_crossing_from_tracks_vehicle_and_pedestrian():
    ego_track = (TrackSample(tick=1, x=1.5, y=0.0, yaw_deg=90.0, velocity_mps=10.0),)
    vehicle_track = (
        TrackSample(tick=0, x=-5.0, y=10.0, yaw_deg=0.0, velocity_mps=10.0),
        TrackSample(tick=1, x=-3.0, y=10.0, yaw_deg=0.0, velocity_mps=10.0),
    )
    pedestrian_track = (
        TrackSample(tick=0, x=-2.0, y=10.0, yaw_deg=0.0, velocity_mps=5.0),
        TrackSample(tick=1, x=0.0, y=10.0, yaw_deg=0.0, velocity_mps=5.0),
    )

    assert detect_crossing_from_tracks(vehicle_track, ego_track, actor_is_vehicle=True) is True
    assert detect_crossing_from_tracks(pedestrian_track, ego_track, actor_is_vehicle=False) is True


def test_detect_crossing_from_tracks_without_track_is_false():
    ego_track = (TrackSample(tick=1, x=1.5, y=0.0, yaw_deg=90.0, velocity_mps=10.0),)
    assert detect_crossing_from_tracks((), ego_track, actor_is_vehicle=True) is False
    assert detect_crossing_from_tracks(ego_track, (), actor_is_vehicle=True) is False


def test_detect_crossing_from_tracks_uses_ego_route_when_ego_stopped():
    # Ego is stopped 30 m short of the crossing point on its planned route.
    ego_track = (TrackSample(tick=0, x=0.0, y=0.0, yaw_deg=90.0, velocity_mps=0.0),)
    ego_route = ((0.0, 0.0), (0.0, 10.0), (0.0, 20.0), (0.0, 30.0), (0.0, 40.0), (0.0, 50.0))
    # A slow walker crosses the route at y=35 over 40 ticks (0.1 m/tick).
    slow_walker = tuple(
        TrackSample(tick=tick, x=-2.0 + 0.1 * tick, y=35.0, yaw_deg=0.0, velocity_mps=0.1)
        for tick in range(41)
    )

    assert detect_crossing_from_tracks(slow_walker, ego_track, actor_is_vehicle=False) is False
    assert (
        detect_crossing_from_tracks(
            slow_walker,
            ego_track,
            actor_is_vehicle=False,
            ego_route=ego_route,
        )
        is True
    )


def test_detect_crossing_from_tracks_without_route_matches_legacy():
    ego_track = (
        TrackSample(tick=0, x=1.5, y=0.0, yaw_deg=90.0, velocity_mps=10.0),
        TrackSample(tick=1, x=1.5, y=1.0, yaw_deg=90.0, velocity_mps=10.0),
    )
    actor_track = (
        TrackSample(tick=0, x=-2.0, y=8.0, yaw_deg=0.0, velocity_mps=5.0),
        TrackSample(tick=1, x=1.5, y=8.0, yaw_deg=0.0, velocity_mps=5.0),
    )

    assert detect_crossing_from_tracks(actor_track, ego_track, actor_is_vehicle=True) is True
    assert (
        detect_crossing_from_tracks(actor_track, ego_track, actor_is_vehicle=True, ego_route=None)
        is True
    )
    assert detect_crossing_from_tracks(actor_track, ego_track, actor_is_vehicle=True, ego_route=()) is True


def test_route_polyline_accepts_dicts_and_point_pairs():
    assert route_polyline(None) == ()
    assert route_polyline([]) == ()
    assert route_polyline(({"x": 1.0, "y": 2.0}, {"x": 1.0, "y": 2.0}, (3.0, 4.0))) == (
        (1.0, 2.0),
        (3.0, 4.0),
    )


def test_crossing_path_config_serialisation():
    config = CrossingPathConfig(corridor_half_width_m=1.5)
    payload = config.to_dict()

    assert payload["corridor_half_width_m"] == 1.5
    assert payload["vehicle_heading_range_deg"] == [30.0, 150.0]


# ---------------------------------------------------------------------------
# Serialisation and angle helpers
# ---------------------------------------------------------------------------


def test_lane_ref_round_trip_and_legacy_aliases():
    lane = LaneRef(road_id=10, lane_id=-1, s=12.5, is_junction=False, lane_width_m=3.5, yaw_deg=90.0)
    restored = LaneRef.from_dict(lane.to_dict())

    assert restored == lane

    legacy = LaneRef.from_dict({"road_id": 10, "lane_id": -1, "rotation": {"yaw": 90.0}})
    assert legacy is not None and legacy.yaw_deg == 90.0
    assert LaneRef.from_dict({"road_id": 10}) is None


def test_track_sample_round_trip_and_legacy_aliases():
    sample = TrackSample(tick=7, x=1.0, y=2.0, yaw_deg=45.0, velocity_mps=3.0, road_id=10, lane_id=-1)
    restored = TrackSample.from_dict(sample.to_dict())

    assert restored == sample

    legacy = TrackSample.from_dict({"tick": 1, "x": 0.0, "y": 0.0, "yaw": 90.0, "speed_mps": 2.0})
    assert legacy is not None
    assert legacy.yaw_deg == 90.0 and legacy.velocity_mps == 2.0
    assert TrackSample.from_dict({"tick": 1}) is None


def test_track_polyline_deduplicates_consecutive_samples():
    samples = (
        TrackSample(tick=0, x=0.0, y=0.0, yaw_deg=0.0),
        TrackSample(tick=1, x=0.0, y=0.0, yaw_deg=0.0),
        TrackSample(tick=2, x=1.0, y=0.0, yaw_deg=0.0),
    )
    assert track_polyline(samples) == ((0.0, 0.0), (1.0, 0.0))


def test_angle_helpers():
    assert wrap_angle_deg(190.0) == pytest.approx(-170.0)
    assert wrap_angle_deg(-190.0) == pytest.approx(170.0)
    assert angle_difference_deg(10.0, 350.0) == pytest.approx(20.0)
