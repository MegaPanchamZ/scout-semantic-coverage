"""Lane indexing, lane-relative geometry, and swept-path relations.

WHY THIS MODULE EXISTS
----------------------
The EXP-018 nuScenes oracle defines seven relation predicates, but the
simulator-side semantic stream only persisted ``same_road_as_ego`` /
``same_lane_as_ego`` booleans computed live by the observer.  Nothing about
*which* OpenDRIVE lane each actor occupied was persisted, so an offline reader
could not recompute lane geometry, lateral adjacency, or a trajectory
intersection.  This module provides:

* a small serialisable lane membership record (``LaneRef``) that both the
  observer (live CARLA map API) and the offline coverage engine (persisted
  stream) construct;
* lateral adjacency classification (left / right neighbour) from the map API's
  ``get_left_lane`` / ``get_right_lane`` when a live waypoint is available, with
  a documented OpenDRIVE ordinal fallback from ``lane_id`` alone;
* heading-based ``oncoming`` classification on a shared road;
* lane-change detection between adjacent lane memberships;
* swept-path intersection between an actor track / waypoint sweep and the ego
  path, mirroring the oracle's ``crossing_path`` definition (consecutive track
  points on opposite sides of the ego route polyline with closest approach
  <= corridor half width).

PREDICATE GROUNDING CLASSES
---------------------------
The consumer (``research/harness/observers/semantic.py``) documents every
emitted predicate.  The classes used by this module are:

* direct   -- ``same_lane`` / ``adjacent_lane``: read from CARLA's
  OpenDRIVE-backed map API (``road_id`` / ``lane_id`` / left-right neighbours)
  or from the persisted counterpart of exactly those fields.
* derived  -- ``oncoming`` (same road + heading opposition + lane membership;
  callers add the oracle's in-front and moving conditions), ``lane_changing``
  (adjacent lane transition between ticks), ``crossing_path`` (swept-path
  intersection of the actor track and the ego path).
* proxy    -- the legacy in-front-within-N-metres corridor test for
  ``crossing_path`` lives in the observer / coverage engine, not here; it is
  used only when no track or waypoint geometry is available.

CARLA API ASSUMPTIONS (faithful mocks in ``research/tests/test_lane_index.py``)
------------------------------------------------------------------------------
* ``carla.Map.get_waypoint(location, project_to_road=True)`` returns a
  ``carla.Waypoint`` with ``road_id``, ``lane_id``, ``s``, ``is_junction``,
  ``lane_width``, ``lane_type`` and ``transform`` (location + rotation.yaw).
* ``carla.Waypoint.get_left_lane()`` / ``get_right_lane()`` return the
  neighbouring lane waypoint or ``None`` at the road edge; results whose
  ``lane_type`` is known and not ``LaneType.Driving`` are rejected.
* ``carla.Waypoint.next(distance)`` returns a (possibly empty) list of
  waypoints ``distance`` metres ahead; the first entry is followed.
* OpenDRIVE lane ordinals: within one road, ``lane_id`` values run outward from
  the reference line (``+1`` immediately left, ``-1`` immediately right), so
  adjacent lane pairs are same-sign ``|id|`` differing by one, or the
  ``+1``/``-1`` pair across the reference line.  ``get_left_lane`` /
  ``get_right_lane`` remain authoritative when available.

UNVALIDATED WITHOUT LIVE CARLA
------------------------------
Everything here runs against mocks in CI.  What mocks cannot confirm:
``get_left_lane`` / ``get_right_lane`` behaviour at junctions and lane merges,
the ``lane_type`` filter against real enum values, and whether CARLA's
``lane_id`` sign convention matches the OpenDRIVE ordinal fallback on every
Town.  The smallest live validation is a single Town01 run of
``run_diagnostics_suite.py`` on a straight two-way road with a stationary ego
and one oncoming vehicle: the persisted stream should show the ego in one lane,
the oncoming vehicle in the lane with opposite-signed ``lane_id``, and
``oncoming_as_ego == true``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

SEMANTIC_STREAM_SCHEMA_VERSION = 2

Point = tuple[float, float]

DEFAULT_ONCOMING_OPPOSITION_DEG = 135.0
DEFAULT_CORRIDOR_HALF_WIDTH_M = 2.0
DEFAULT_MIN_ACTOR_DISPLACEMENT_M = 0.5
DEFAULT_VEHICLE_HEADING_RANGE_DEG = (30.0, 150.0)

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Serialisable records (also the semantic-stream schema, version 2)
# ---------------------------------------------------------------------------


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class LaneRef:
    """One actor's OpenDRIVE lane membership at one tick (direct grounding)."""

    road_id: int
    lane_id: int
    s: float | None = None
    is_junction: bool | None = None
    lane_width_m: float | None = None
    yaw_deg: float | None = None
    lane_type: str | None = None

    @property
    def key(self) -> tuple[int, int]:
        return (self.road_id, self.lane_id)

    @property
    def direction(self) -> int:
        if self.lane_id > 0:
            return 1
        if self.lane_id < 0:
            return -1
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "road_id": int(self.road_id),
            "lane_id": int(self.lane_id),
            "s": None if self.s is None else round(float(self.s), 4),
            "is_junction": self.is_junction,
            "lane_width_m": None if self.lane_width_m is None else round(float(self.lane_width_m), 4),
            "yaw_deg": None if self.yaw_deg is None else round(float(self.yaw_deg), 4),
            "lane_type": self.lane_type,
        }

    @classmethod
    def from_dict(cls, data: Any) -> LaneRef | None:
        if not isinstance(data, dict):
            return None
        road_id = _maybe_int(data.get("road_id"))
        lane_id = _maybe_int(data.get("lane_id"))
        if road_id is None or lane_id is None:
            return None
        yaw = data.get("yaw_deg")
        if yaw is None:
            rotation = data.get("rotation")
            if isinstance(rotation, dict):
                yaw = rotation.get("yaw")
        lane_width = data.get("lane_width_m", data.get("lane_width"))
        return cls(
            road_id=road_id,
            lane_id=lane_id,
            s=_maybe_float(data.get("s")),
            is_junction=bool(data["is_junction"]) if data.get("is_junction") is not None else None,
            lane_width_m=_maybe_float(lane_width),
            yaw_deg=_maybe_float(yaw),
            lane_type=str(data["lane_type"]) if data.get("lane_type") is not None else None,
        )


@dataclass(frozen=True)
class TrackSample:
    """One compact pose sample of an actor (or ego) at one tick."""

    tick: int
    x: float
    y: float
    yaw_deg: float
    velocity_mps: float = 0.0
    road_id: int | None = None
    lane_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": int(self.tick),
            "x": round(float(self.x), 3),
            "y": round(float(self.y), 3),
            "yaw_deg": round(float(self.yaw_deg), 3),
            "velocity_mps": round(float(self.velocity_mps), 3),
            "road_id": self.road_id,
            "lane_id": self.lane_id,
        }

    @classmethod
    def from_dict(cls, data: Any) -> TrackSample | None:
        if not isinstance(data, dict):
            return None
        x = _maybe_float(data.get("x"))
        y = _maybe_float(data.get("y"))
        if x is None or y is None:
            return None
        yaw = data.get("yaw_deg", data.get("yaw"))
        velocity = data.get("velocity_mps", data.get("speed_mps"))
        return cls(
            tick=int(_maybe_int(data.get("tick")) or 0),
            x=x,
            y=y,
            yaw_deg=_maybe_float(yaw) or 0.0,
            velocity_mps=_maybe_float(velocity) or 0.0,
            road_id=_maybe_int(data.get("road_id")),
            lane_id=_maybe_int(data.get("lane_id")),
        )


@dataclass(frozen=True)
class LaneRelation:
    """Lane-index relation of one actor to the ego at one tick.

    ``oncoming`` here means "same road, known lane membership, opposed heading";
    the oracle additionally requires in-front and moving, which the caller adds
    because they come from pose/velocity rather than the lane index.
    """

    same_road: bool
    same_lane: bool
    adjacent_lane: bool
    relative_lane: str | None
    oncoming: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "same_road": self.same_road,
            "same_lane": self.same_lane,
            "adjacent_lane": self.adjacent_lane,
            "relative_lane": self.relative_lane,
            "oncoming": self.oncoming,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CrossingPathConfig:
    """Thresholds for the oracle-shaped swept-path crossing test."""

    corridor_half_width_m: float = DEFAULT_CORRIDOR_HALF_WIDTH_M
    min_actor_displacement_m: float = DEFAULT_MIN_ACTOR_DISPLACEMENT_M
    vehicle_heading_range_deg: tuple[float, float] = DEFAULT_VEHICLE_HEADING_RANGE_DEG
    ego_horizon_s: float = 2.0
    actor_horizon_s: float = 2.0
    step_s: float = 0.2
    speed_epsilon_mps: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "corridor_half_width_m": self.corridor_half_width_m,
            "min_actor_displacement_m": self.min_actor_displacement_m,
            "vehicle_heading_range_deg": list(self.vehicle_heading_range_deg),
            "ego_horizon_s": self.ego_horizon_s,
            "actor_horizon_s": self.actor_horizon_s,
            "step_s": self.step_s,
            "speed_epsilon_mps": self.speed_epsilon_mps,
        }


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------


def wrap_angle_deg(delta: float) -> float:
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def angle_difference_deg(a: float, b: float) -> float:
    return abs(wrap_angle_deg(a - b))


# ---------------------------------------------------------------------------
# Lane membership from the CARLA map API
# ---------------------------------------------------------------------------


def is_driving_lane(waypoint: Any) -> bool:
    """Accept unknown lane-type representations; reject known non-driving ones."""
    lane_type = getattr(waypoint, "lane_type", None)
    if lane_type is None:
        return True
    name = str(lane_type).lower()
    if "driving" in name:
        return True
    if name.startswith("lanetype"):
        return False
    return True


def lane_ref_from_waypoint(waypoint: Any) -> LaneRef | None:
    """Extract a lane membership record, or None for missing/non-driving lanes.

    Non-driving lane types (sidewalk, shoulder, parking, biking) are rejected so
    that a pedestrian or prop beside the road is not credited with same-lane or
    adjacent-lane membership against a driving ego.  CARLA waypoints whose lane
    type cannot be interpreted are accepted (see ``is_driving_lane``).
    """
    if waypoint is None or not is_driving_lane(waypoint):
        return None
    road_id = _maybe_int(getattr(waypoint, "road_id", None))
    lane_id = _maybe_int(getattr(waypoint, "lane_id", None))
    if road_id is None or lane_id is None:
        return None
    transform = getattr(waypoint, "transform", None)
    yaw = None
    if transform is not None:
        rotation = getattr(transform, "rotation", None)
        yaw = getattr(rotation, "yaw", None)
    lane_type = getattr(waypoint, "lane_type", None)
    return LaneRef(
        road_id=road_id,
        lane_id=lane_id,
        s=_maybe_float(getattr(waypoint, "s", None)),
        is_junction=bool(getattr(waypoint, "is_junction", False)),
        lane_width_m=_maybe_float(getattr(waypoint, "lane_width", None)),
        yaw_deg=_maybe_float(yaw),
        lane_type=str(lane_type) if lane_type is not None else None,
    )


def _neighbour_lane(waypoint: Any, getter_name: str) -> LaneRef | None:
    getter = getattr(waypoint, getter_name, None)
    if not callable(getter):
        return None
    try:
        candidate = getter()
    except RuntimeError:
        return None
    if candidate is None or not is_driving_lane(candidate):
        return None
    return lane_ref_from_waypoint(candidate)


def neighbour_lane_refs(waypoint: Any) -> tuple[LaneRef | None, LaneRef | None]:
    """Return (left, right) driving-lane neighbours of a waypoint, if any."""
    if waypoint is None:
        return (None, None)
    return (_neighbour_lane(waypoint, "get_left_lane"), _neighbour_lane(waypoint, "get_right_lane"))


def relative_lane_from_lane_ids(ego_lane: LaneRef | None, actor_lane: LaneRef | None) -> str | None:
    """OpenDRIVE ordinal fallback for left/right adjacency, valid only same road.

    Returns ``"same"``, ``"left"``, ``"right"`` or ``None`` when the lanes are
    not laterally adjacent (or membership is unusable).  ``get_left_lane`` /
    ``get_right_lane`` are preferred whenever a live waypoint is available.
    """
    if ego_lane is None or actor_lane is None:
        return None
    if ego_lane.road_id != actor_lane.road_id:
        return None
    ego_id, actor_id = ego_lane.lane_id, actor_lane.lane_id
    if ego_id == 0 or actor_id == 0:
        return None
    if ego_id == actor_id:
        return "same"
    if (ego_id > 0) == (actor_id > 0):
        if abs(abs(ego_id) - abs(actor_id)) != 1:
            return None
        return "left" if actor_id > ego_id else "right"
    if abs(ego_id) != 1 or abs(actor_id) != 1:
        return None
    return "left" if actor_id > 0 else "right"


def classify_lane_relation(
    ego_lane: LaneRef | None,
    actor_lane: LaneRef | None,
    *,
    actor_yaw_deg: float,
    ego_yaw_deg: float,
    ego_waypoint: Any | None = None,
    actor_waypoint: Any | None = None,
    oncoming_opposition_deg: float = DEFAULT_ONCOMING_OPPOSITION_DEG,
) -> LaneRelation:
    """Classify one actor's lane-index relation to the ego.

    ``same_lane`` / ``adjacent_lane`` are direct groundings from map lane data.
    ``oncoming`` is derived from a shared road plus heading opposition; callers
    add the oracle's in-front and moving conditions.
    """
    if ego_lane is None or actor_lane is None:
        return LaneRelation(False, False, False, None, False, "lane-membership-unknown")
    if ego_lane.road_id != actor_lane.road_id:
        return LaneRelation(False, False, False, None, False, "different-road")
    if bool(ego_lane.is_junction) or bool(actor_lane.is_junction):
        return LaneRelation(True, False, False, None, False, "junction-membership-unstable")
    if ego_lane.lane_id == actor_lane.lane_id:
        return LaneRelation(True, True, False, "same", False, "same-road-same-lane-id")

    relative: str | None = None
    if ego_waypoint is not None:
        left, right = neighbour_lane_refs(ego_waypoint)
        if left is not None and left.key == actor_lane.key:
            relative = "left"
        elif right is not None and right.key == actor_lane.key:
            relative = "right"
    if relative is None and actor_waypoint is not None:
        # The map API on the actor side is symmetric, so accept it too.
        left, right = neighbour_lane_refs(actor_waypoint)
        if left is not None and left.key == ego_lane.key:
            relative = "right"
        elif right is not None and right.key == ego_lane.key:
            relative = "left"
    if relative is None:
        relative = relative_lane_from_lane_ids(ego_lane, actor_lane)

    adjacent = relative in ("left", "right")
    heading_opposed = angle_difference_deg(actor_yaw_deg, ego_yaw_deg) >= oncoming_opposition_deg
    return LaneRelation(True, False, adjacent, relative, heading_opposed, "lane-adjacency")


def detect_lane_change(
    previous: LaneRef | None,
    current: LaneRef | None,
) -> bool:
    """True when lane membership moves to a laterally adjacent lane."""
    if previous is None or current is None:
        return False
    if previous.road_id != current.road_id or previous.lane_id == current.lane_id:
        return False
    if bool(previous.is_junction) or bool(current.is_junction):
        return False
    return relative_lane_from_lane_ids(previous, current) in ("left", "right")


# ---------------------------------------------------------------------------
# Paths (tracks, motion projection, waypoint sweep)
# ---------------------------------------------------------------------------


def _dedupe(points: Iterable[Point], tolerance: float = 1e-6) -> tuple[Point, ...]:
    deduped: list[Point] = []
    for x, y in points:
        if deduped and abs(x - deduped[-1][0]) <= tolerance and abs(y - deduped[-1][1]) <= tolerance:
            continue
        deduped.append((float(x), float(y)))
    return tuple(deduped)


def path_length(points: Sequence[Point]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def track_polyline(samples: Sequence[TrackSample]) -> tuple[Point, ...]:
    return _dedupe((sample.x, sample.y) for sample in samples)


def motion_swept_path(
    x: float,
    y: float,
    yaw_deg: float,
    velocity_mps: float,
    *,
    horizon_s: float = 2.0,
    step_s: float = 0.2,
    speed_epsilon_mps: float = 0.1,
) -> tuple[Point, ...]:
    """Straight-line forward projection of one pose at constant velocity."""
    points: list[Point] = [(x, y)]
    if abs(velocity_mps) < speed_epsilon_mps or horizon_s <= 0.0:
        return tuple(points)
    yaw_rad = math.radians(yaw_deg)
    vx = velocity_mps * math.cos(yaw_rad)
    vy = velocity_mps * math.sin(yaw_rad)
    step = max(step_s, 1e-3)
    elapsed = step
    while elapsed <= horizon_s + _EPS:
        points.append((x + vx * elapsed, y + vy * elapsed))
        elapsed += step
    return _dedupe(points)


def _waypoint_point(waypoint: Any) -> Point | None:
    transform = getattr(waypoint, "transform", None)
    location = getattr(transform, "location", None)
    if location is None:
        return None
    return (float(location.x), float(location.y))


def waypoint_swept_path(
    waypoint: Any | None,
    *,
    horizon_m: float = 30.0,
    step_m: float = 2.0,
) -> tuple[Point, ...]:
    """Follow ``waypoint.next(step_m)`` for ``horizon_m`` metres (direct map sweep)."""
    if waypoint is None:
        return ()
    start = _waypoint_point(waypoint)
    if start is None:
        return ()
    points: list[Point] = [start]
    current = waypoint
    travelled = 0.0
    step = max(float(step_m), 1e-3)
    while travelled < horizon_m:
        next_fn = getattr(current, "next", None)
        if not callable(next_fn):
            break
        try:
            candidates = next_fn(step)
        except RuntimeError:
            break
        if not candidates:
            break
        current = candidates[0]
        point = _waypoint_point(current)
        if point is None:
            break
        if point == points[-1]:
            break
        points.append(point)
        travelled += step
    return _dedupe(points)


# ---------------------------------------------------------------------------
# Swept-path crossing (oracle crossing_path geometry)
# ---------------------------------------------------------------------------


def _orientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> bool:
    return (
        min(ax, bx) - _EPS <= px <= max(ax, bx) + _EPS
        and min(ay, by) - _EPS <= py <= max(ay, by) + _EPS
    )


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    o1 = _orientation(q1[0], q1[1], q2[0], q2[1], p1[0], p1[1])
    o2 = _orientation(q1[0], q1[1], q2[0], q2[1], p2[0], p2[1])
    o3 = _orientation(p1[0], p1[1], p2[0], p2[1], q1[0], q1[1])
    o4 = _orientation(p1[0], p1[1], p2[0], p2[1], q2[0], q2[1])
    if ((o1 > _EPS and o2 < -_EPS) or (o1 < -_EPS and o2 > _EPS)) and (
        (o3 > _EPS and o4 < -_EPS) or (o3 < -_EPS and o4 > _EPS)
    ):
        return True
    if abs(o1) <= _EPS and _on_segment(q1[0], q1[1], q2[0], q2[1], p1[0], p1[1]):
        return True
    if abs(o2) <= _EPS and _on_segment(q1[0], q1[1], q2[0], q2[1], p2[0], p2[1]):
        return True
    if abs(o3) <= _EPS and _on_segment(p1[0], p1[1], p2[0], p2[1], q1[0], q1[1]):
        return True
    if abs(o4) <= _EPS and _on_segment(p1[0], p1[1], p2[0], p2[1], q2[0], q2[1]):
        return True
    return False


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    length_squared = dx * dx + dy * dy
    if length_squared <= _EPS:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length_squared
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _segment_to_polyline_distance(p1: Point, p2: Point, polyline: Sequence[Point]) -> float:
    best = float("inf")
    for q1, q2 in zip(polyline, polyline[1:]):
        if segments_intersect(p1, p2, q1, q2):
            return 0.0
        best = min(
            best,
            _point_segment_distance(p1[0], p1[1], q1[0], q1[1], q2[0], q2[1]),
            _point_segment_distance(p2[0], p2[1], q1[0], q1[1], q2[0], q2[1]),
        )
    return best


def _side_of_point(point: Point, polyline: Sequence[Point]) -> float:
    """Signed side of the nearest polyline segment (-1/0/+1)."""
    best_distance = float("inf")
    best_side = 0.0
    for a, b in zip(polyline, polyline[1:]):
        distance = _point_segment_distance(point[0], point[1], a[0], a[1], b[0], b[1])
        if distance < best_distance:
            best_distance = distance
            cross = _orientation(a[0], a[1], b[0], b[1], point[0], point[1])
            if abs(cross) <= _EPS:
                best_side = 0.0
            else:
                best_side = 1.0 if cross > 0.0 else -1.0
    return best_side


def detect_path_crossing(
    actor_path: Sequence[Point],
    ego_path: Sequence[Point],
    *,
    corridor_half_width_m: float = DEFAULT_CORRIDOR_HALF_WIDTH_M,
    min_actor_displacement_m: float = DEFAULT_MIN_ACTOR_DISPLACEMENT_M,
    heading_deltas: Sequence[float] | None = None,
    heading_range_deg: tuple[float, float] | None = None,
) -> bool:
    """Oracle-shaped crossing test on two polylines.

    An actor segment counts as crossing when its endpoints lie on opposite
    sides of the ego polyline (or one endpoint lies on it) and the closest
    approach to the ego polyline is within ``corridor_half_width_m``.  The
    actor must displace at least ``min_actor_displacement_m``, and when
    ``heading_range_deg`` is given at least one sampled heading difference must
    fall inside it (the oracle applies ``[30, 150]`` degrees to vehicles).
    """
    actor_path = _dedupe(actor_path)
    ego_path = _dedupe(ego_path)
    if len(actor_path) < 2 or len(ego_path) < 2:
        return False
    if path_length(actor_path) < min_actor_displacement_m:
        return False
    if heading_range_deg is not None and heading_deltas:
        low, high = heading_range_deg
        if not any(low <= delta <= high for delta in heading_deltas):
            return False
    sides = [_side_of_point(point, ego_path) for point in actor_path]
    for index in range(len(actor_path) - 1):
        side_a, side_b = sides[index], sides[index + 1]
        if side_a == 0.0 and side_b == 0.0:
            continue
        crosses_sides = side_a == 0.0 or side_b == 0.0 or (side_a > 0.0) != (side_b > 0.0)
        if not crosses_sides:
            continue
        if _segment_to_polyline_distance(actor_path[index], actor_path[index + 1], ego_path) <= corridor_half_width_m:
            return True
    return False


def route_polyline(route: Sequence[Any] | None) -> tuple[Point, ...]:
    """Coerce a persisted ``ego_route`` (``{"x","y"}`` dicts or point pairs)."""
    if not route:
        return ()
    points: list[Point] = []
    for item in route:
        x: float | None = None
        y: float | None = None
        if isinstance(item, dict):
            x = _maybe_float(item.get("x"))
            y = _maybe_float(item.get("y"))
        else:
            try:
                x = float(item[0])
                y = float(item[1])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
        if x is None or y is None:
            continue
        points.append((x, y))
    return _dedupe(points)


def _route_ahead_of(
    route: Sequence[Point],
    anchor: Point,
) -> tuple[Point, ...]:
    """Route points from the index nearest ``anchor`` onward (plus one for join).

    Slicing here avoids a spurious chord from the route's far end back to the
    observed track: the concatenation stays ordered along the driven direction.
    """
    if not route:
        return ()
    best_index = 0
    best_distance = float("inf")
    for index, point in enumerate(route):
        distance = math.hypot(point[0] - anchor[0], point[1] - anchor[1])
        if distance < best_distance:
            best_distance = distance
            best_index = index
    start = max(best_index - 1, 0)
    return tuple(route[start:])


def detect_crossing_from_tracks(
    actor_track: Sequence[TrackSample],
    ego_track: Sequence[TrackSample],
    *,
    actor_is_vehicle: bool,
    config: CrossingPathConfig | None = None,
    ego_route: Sequence[Any] | None = None,
) -> bool:
    """Swept-path crossing from persisted track samples.

    The actor path is its observed track plus a constant-velocity forward
    projection.  Without ``ego_route`` the ego path is its observed track (the
    oracle's "ego executed trajectory" route proxy) plus a forward projection
    from the latest sample.

    ``ego_route`` (the observer's persisted planned route) is joined to the
    observed track, anchored at the route point nearest the ego's latest
    sample, so the corridor test reaches crossing points beyond the ego's
    observed track and 2 s projection.  An ego stopped short of a crossing can
    therefore still be credited when an actor crosses the planned route ahead;
    without a route this function is bit-for-bit the previous behaviour.
    """
    config = config or CrossingPathConfig()
    actor_track = tuple(actor_track)
    ego_track = tuple(ego_track)
    if not actor_track or not ego_track:
        return False
    actor_last = actor_track[-1]
    actor_polyline = list(track_polyline(actor_track))
    actor_polyline.extend(
        motion_swept_path(
            actor_last.x,
            actor_last.y,
            actor_last.yaw_deg,
            actor_last.velocity_mps,
            horizon_s=config.actor_horizon_s,
            step_s=config.step_s,
            speed_epsilon_mps=config.speed_epsilon_mps,
        )[1:]
    )
    ego_last = ego_track[-1]
    ego_polyline = list(track_polyline(ego_track))
    route_points = route_polyline(ego_route)
    if route_points:
        ego_polyline = list(
            _dedupe([*ego_polyline, *_route_ahead_of(route_points, (ego_last.x, ego_last.y))])
        )
    ego_polyline.extend(
        motion_swept_path(
            ego_last.x,
            ego_last.y,
            ego_last.yaw_deg,
            ego_last.velocity_mps,
            horizon_s=config.ego_horizon_s,
            step_s=config.step_s,
            speed_epsilon_mps=config.speed_epsilon_mps,
        )[1:]
    )
    heading_deltas: list[float] | None = None
    heading_range: tuple[float, float] | None = None
    if actor_is_vehicle:
        ego_yaw_by_tick = {sample.tick: sample.yaw_deg for sample in ego_track}
        heading_deltas = [
            angle_difference_deg(sample.yaw_deg, ego_yaw_by_tick.get(sample.tick, ego_last.yaw_deg))
            for sample in actor_track
        ]
        heading_range = config.vehicle_heading_range_deg
    return detect_path_crossing(
        actor_polyline,
        ego_polyline,
        corridor_half_width_m=config.corridor_half_width_m,
        min_actor_displacement_m=config.min_actor_displacement_m,
        heading_deltas=heading_deltas,
        heading_range_deg=heading_range,
    )
