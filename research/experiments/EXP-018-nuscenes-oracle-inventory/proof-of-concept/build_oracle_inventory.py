#!/usr/bin/env python3
"""Build a deterministic STSG oracle inventory from a nuScenes split.

This is the EXP-018 proof-of-concept builder for the RQ1 oracle inventory
described in the paper section "STSG Oracle Construction from nuScenes".

The script reads nuScenes JSON tables directly (no nuscenes-devkit dependency),
derives slice-level predicate facts with explicit definitions and data sources,
and emits an obligation inventory in four dimensions (node, attribute,
relation, hazard class) with provenance for every obligation.

Everything that cannot be honestly grounded from the mounted data is reported
as ungrounded with a reason instead of being approximated silently.

Example:
    research/.venv/bin/python research/experiments/EXP-018-nuscenes-oracle-inventory/proof-of-concept/build_oracle_inventory.py \
        --dataset-root datasets/NuScenes --split v1.0-mini
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "artifacts"

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:  # pragma: no cover - exercised on machines that have the harness checkout
    from research.harness import dota_symbolic as _dota_symbolic
except Exception:  # pragma: no cover - defensive: inventory must still build
    _dota_symbolic = None


# ---------------------------------------------------------------------------
# Vocabulary tables
# ---------------------------------------------------------------------------

# Category alias normalization. This mirrors the semantics of
# `research/harness/observers/semantic.py::SemanticObserver._actor_alias`:
# anything containing "pedestrian" becomes `pedestrian`, anything containing
# "vehicle" (including bicycles and emergency vehicles) becomes `vehicle`, and
# every other category falls back to its last dotted token. The table is
# explicit (not prefix magic) so that vocabulary revisions are reviewable.
NODE_ALIAS_TABLE: dict[str, str] = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "pedestrian",
    "human.pedestrian.stroller": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "animal",
    "vehicle.car": "vehicle",
    "vehicle.motorcycle": "vehicle",
    "vehicle.bicycle": "vehicle",
    "vehicle.bus.bendy": "vehicle",
    "vehicle.bus.rigid": "vehicle",
    "vehicle.truck": "vehicle",
    "vehicle.construction": "vehicle",
    "vehicle.emergency.ambulance": "vehicle",
    "vehicle.emergency.police": "vehicle",
    "vehicle.trailer": "vehicle",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "trafficcone",
    "movable_object.pushable_pullable": "pushable_pullable",
    "movable_object.debris": "debris",
    "static_object.bicycle_rack": "bicycle_rack",
}

# Nodes that the observer alias table collapses into a coarser token. Recorded
# so the inventory can report the collapse and future vocabulary revisions can
# split them again without touching predicate code.
NODE_ALIAS_COLLAPSES: dict[str, str] = {
    "vehicle.bicycle": "vehicle",
    "vehicle.motorcycle": "vehicle",
    "human.pedestrian.personal_mobility": "pedestrian",
    "human.pedestrian.wheelchair": "pedestrian",
}

STATIC_OBJECT_NODES = {"barrier", "trafficcone", "pushable_pullable", "debris", "bicycle_rack"}
AGENT_NODE_TYPES = {"vehicle", "pedestrian"}

# nuScenes attribute labels lifted to inventory predicates. These are dataset
# labels (direct grounding), not derived quantities.
ATTRIBUTE_LABEL_TABLE: dict[str, str] = {
    "vehicle.moving": "moving",
    "vehicle.stopped": "stopped",
    "vehicle.parked": "parked",
    "cycle.with_rider": "with_rider",
    "cycle.without_rider": "without_rider",
    "pedestrian.sitting_lying_down": "sitting_lying_down",
    "pedestrian.standing": "standing",
    "pedestrian.moving": "moving",
}

# Predicates that appear in the simulator-side vocabulary (SemanticObserver and
# the DoTA archetype bundle) but that cannot be grounded from the mounted
# nuScenes annotation tables. These are reported, never fabricated.
UNGROUNDED_PREDICATES: dict[str, dict[str, str]] = {
    "occluded": {
        "dimension": "attribute",
        "reason": (
            "sample_annotation exposes visibility_token (an image-coverage bin) and point counts, "
            "but no line-of-sight relation between an occluder and the ego sensor. The simulator "
            "observer emits occluded(actor) only for its scripted occluder vehicle."
        ),
    },
    "colliding": {
        "dimension": "relation",
        "reason": (
            "no contact or collision event is recorded anywhere in the nuScenes annotation tables; "
            "an inventory entry would have to be invented."
        ),
    },
    "out_of_control": {
        "dimension": "attribute",
        "reason": "no control-state (steering/throttle/stability) signal exists in the dataset tables.",
    },
    "speeding": {
        "dimension": "attribute",
        "reason": (
            "map expansion lane records carry lane_type but no speed_limit field, so a speed-limit "
            "violation cannot be evaluated without inventing a limit."
        ),
    },
}

# DoTA archetypes from research/harness/dota_symbolic.py that are deliberately
# not emitted, with the reason. Kept in the artifact so the coverage gap is
# auditable rather than silent.
DOTA_CLASSES_NOT_EMITTED: dict[str, str] = {
    "ego: lateral": "requires out_of_control(ego), which is not observable in the dataset tables.",
    "ego: leave_to_left": "requires out_of_control(ego), which is not observable in the dataset tables.",
    "ego: leave_to_right": "requires out_of_control(ego), which is not observable in the dataset tables.",
    "ego: unknown": "requires out_of_control(ego) or colliding(ego, adversary); neither is observable.",
    "ego: moving_ahead_or_waiting": "requires braking(ego) as an anomaly label; the dataset has no ego control state.",
    "ego: start_stop_or_stationary": "requires stationary(ego) as an anomaly label; slice-level ego speed is not an anomaly annotation.",
    "other: unknown": "requires colliding(adversary, ego), which is not observable.",
    "other: leave_to_left": "requires departing_left(adversary); lateral departure intent is not separable from a lane change without control signals.",
    "other: leave_to_right": "requires departing_right(adversary); lateral departure intent is not separable from a lane change without control signals.",
}


# ---------------------------------------------------------------------------
# Predicate definitions (single source of truth for parameters.md and the JSON)
# ---------------------------------------------------------------------------

PREDICATE_DEFINITIONS: dict[str, dict[str, str]] = {
    "stationary": {
        "dimension": "attribute",
        "grounding": "derived",
        "definition": "actor speed < --stationary-threshold-mps (default 0.5 m/s) at >= 1 keyframe in the slice",
        "source": "finite difference of consecutive sample_annotation translations of the same instance",
    },
    "moving": {
        "dimension": "attribute",
        "grounding": "derived",
        "definition": "actor speed >= --stationary-threshold-mps at >= 1 keyframe, or nuScenes attribute *.moving",
        "source": "track finite difference; nuScenes attribute table",
    },
    "waiting": {
        "dimension": "attribute",
        "grounding": "derived",
        "definition": "agent node (vehicle/pedestrian) stationary at >= 2 consecutive keyframes AND in_front_of(actor,ego) at >= 1 keyframe",
        "source": "DoTA/EXP-017 crosswalk derivation (agent_motion_stationary + relation_ahead_of_ego)",
    },
    "stopped": {
        "dimension": "attribute",
        "grounding": "direct",
        "definition": "nuScenes attribute vehicle.stopped on >= 1 annotation in the slice",
        "source": "sample_annotation.attribute_tokens -> attribute.name",
    },
    "parked": {
        "dimension": "attribute",
        "grounding": "direct",
        "definition": "nuScenes attribute vehicle.parked on >= 1 annotation in the slice",
        "source": "sample_annotation.attribute_tokens -> attribute.name",
    },
    "standing": {
        "dimension": "attribute",
        "grounding": "direct",
        "definition": "nuScenes attribute pedestrian.standing on >= 1 annotation in the slice",
        "source": "sample_annotation.attribute_tokens -> attribute.name",
    },
    "sitting_lying_down": {
        "dimension": "attribute",
        "grounding": "direct",
        "definition": "nuScenes attribute pedestrian.sitting_lying_down on >= 1 annotation in the slice",
        "source": "sample_annotation.attribute_tokens -> attribute.name",
    },
    "with_rider": {
        "dimension": "attribute",
        "grounding": "direct",
        "definition": "nuScenes attribute cycle.with_rider on >= 1 annotation in the slice",
        "source": "sample_annotation.attribute_tokens -> attribute.name",
    },
    "without_rider": {
        "dimension": "attribute",
        "grounding": "direct",
        "definition": "nuScenes attribute cycle.without_rider on >= 1 annotation in the slice",
        "source": "sample_annotation.attribute_tokens -> attribute.name",
    },
    "braking": {
        "dimension": "attribute",
        "grounding": "derived",
        "definition": "speed drops by >= --braking-drop-mps (default 1.0 m/s) between the actor's first and last keyframe in the slice while it is moving",
        "source": "track finite difference",
    },
    "turning": {
        "dimension": "attribute",
        "grounding": "derived",
        "definition": "|yaw change| >= --turning-yaw-deg (default 30 deg) between the actor's first and last keyframe in the slice while displaced >= 2 m",
        "source": "sample_annotation rotation quaternion (yaw)",
    },
    "lane_changing": {
        "dimension": "attribute",
        "grounding": "derived",
        "definition": "actor lane polygon changes to a laterally adjacent lane polygon within the slice",
        "source": "map expansion lane polygons + lateral adjacency graph",
    },
    "on_road": {
        "dimension": "attribute",
        "grounding": "derived",
        "definition": "actor position falls inside a lane or lane-connector polygon at >= 1 keyframe",
        "source": "map expansion lane / lane_connector polygons",
    },
    "jaywalking": {
        "dimension": "attribute",
        "grounding": "proxy",
        "definition": "pedestrian inside a lane cell, not inside any ped_crossing polygon, and not inside any carpark_area polygon at that keyframe",
        "source": "map expansion lane + ped_crossing + carpark_area polygons; legality (signal state, right of way) is not grounded",
    },
    "in_front_of": {
        "dimension": "relation",
        "grounding": "direct",
        "definition": "dot(target_position - ego_position, ego_forward) > 0 (forward half-plane test)",
        "source": "ego_pose translation/rotation + annotation translation; same convention as SemanticObserver._is_in_front_of",
    },
    "same_lane": {
        "dimension": "relation",
        "grounding": "direct",
        "definition": "ego and actor share at least one lane-cell polygon token (lane or lane_connector)",
        "source": "map expansion point-in-polygon lane membership",
    },
    "adjacent_lane": {
        "dimension": "relation",
        "grounding": "direct",
        "definition": "ego lane polygon and actor lane polygon are laterally adjacent: they share >= 2 boundary nodes that are lane-divider segment nodes on at least one lane",
        "source": "map expansion lane polygons + lane_divider segments",
    },
    "oncoming": {
        "dimension": "relation",
        "grounding": "derived",
        "definition": (
            "vehicle node moving at that keyframe with heading opposition >= --oncoming-opposition-deg "
            "(default 135 deg) AND in_front_of AND both ego and actor assigned to a lane cell"
        ),
        "source": "ego_pose yaw + annotation yaw + track speed + lane membership",
    },
    "approaching": {
        "dimension": "relation",
        "grounding": "derived",
        "definition": (
            "agent node (vehicle/pedestrian) whose ego distance decreases by >= --approaching-drop-m "
            "(default 0.5 m) between its first and last keyframe in the slice"
        ),
        "source": "ego_pose translations + annotation translations; static objects are excluded because "
        "their distance shrinks merely because the ego drives toward them",
    },
    "crossing_path": {
        "dimension": "relation",
        "grounding": "derived",
        "definition": (
            "agent node (vehicle/pedestrian) whose swept path crosses the ego route corridor: consecutive "
            "track points lie on opposite sides of the ego route polyline with closest approach "
            "<= --crossing-corridor-half-width-m (default 2.0 m), the crossing is at or ahead of the ego's "
            "slice-start route position, the actor moves >= 0.5 m, and vehicles have a heading difference "
            "in [30,150] deg at >= 1 keyframe"
        ),
        "source": "annotation track polyline + ego executed trajectory over the scene as the route polygon proxy",
    },
    "obstructing": {
        "dimension": "relation",
        "grounding": "derived",
        "definition": "on_road(actor) AND in_front_of(actor,ego) AND same_lane(actor,ego)",
        "source": "conjunction of grounded attributes/relations (EXP-017 crosswalk: map_lane_overlap + relation_ahead_of_ego)",
    },
    "occluded": {
        "dimension": "attribute",
        "grounding": "ungrounded",
        "definition": "not emitted",
        "source": "n/a",
    },
    "colliding": {
        "dimension": "relation",
        "grounding": "ungrounded",
        "definition": "not emitted",
        "source": "n/a",
    },
    "out_of_control": {
        "dimension": "attribute",
        "grounding": "ungrounded",
        "definition": "not emitted",
        "source": "n/a",
    },
    "speeding": {
        "dimension": "attribute",
        "grounding": "ungrounded",
        "definition": "not emitted",
        "source": "n/a",
    },
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic STSG oracle inventory (node/attribute/relation/hazard-class) "
            "from a nuScenes split by reading JSON tables directly."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/NuScenes"),
        help="nuScenes dataset root containing the split directory and expansion/.",
    )
    parser.add_argument("--split", default="v1.0-mini", help="Split table directory name.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for oracle_inventory_<split>.json and .md.",
    )
    parser.add_argument("--scene", default=None, help="Restrict to one scene name (debug).")
    parser.add_argument("--scene-limit", type=int, default=None, help="Restrict to first N scenes (debug).")
    parser.add_argument("--slice-keyframes", type=int, default=6, help="Keyframes per slice (2 Hz x 3 s).")
    parser.add_argument("--slice-stride-keyframes", type=int, default=2, help="Slice stride in keyframes.")
    parser.add_argument("--stationary-threshold-mps", type=float, default=0.5)
    parser.add_argument("--braking-drop-mps", type=float, default=1.0)
    parser.add_argument("--turning-yaw-deg", type=float, default=30.0)
    parser.add_argument("--oncoming-opposition-deg", type=float, default=135.0)
    parser.add_argument("--approaching-drop-m", type=float, default=0.5)
    parser.add_argument("--crossing-corridor-half-width-m", type=float, default=2.0)
    parser.add_argument("--max-witnesses", type=int, default=5, help="Witnesses retained per obligation.")
    parser.add_argument(
        "--generated-at",
        default=None,
        help="Override the generated_at timestamp (deterministic reruns).",
    )
    return parser.parse_args(argv)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def iter_json_array(path: Path, project: Callable[[dict[str, Any]], Any] | None = None) -> Iterator[Any]:
    """Stream a top-level JSON array of objects without loading the file whole."""
    decoder = json.JSONDecoder()
    chunk_size = 1 << 20
    trim_threshold = 1 << 22
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        while True:
            buffer += handle.read(chunk_size)
            stripped = buffer.lstrip()
            if stripped:
                if stripped[0] != "[":
                    raise ValueError(f"{path} does not start with a JSON array")
                buffer = stripped[1:]
                break
            if not buffer:
                raise ValueError(f"{path} is empty")
        pos = 0
        while True:
            while True:
                while pos < len(buffer) and buffer[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buffer):
                    break
                if pos:
                    buffer = buffer[pos:]
                    pos = 0
                chunk = handle.read(chunk_size)
                if not chunk:
                    return
                buffer += chunk
            if buffer[pos] == "]":
                return
            try:
                obj, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError:
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise
                buffer = buffer + chunk
                continue
            pos = end
            item = project(obj) if project is not None else obj
            if pos > trim_threshold:
                buffer = buffer[pos:]
                pos = 0
            if item is not None:
                yield item


def load_json_table(table_root: Path, name: str) -> list[dict[str, Any]]:
    return json.loads((table_root / f"{name}.json").read_text(encoding="utf-8"))


def yaw_deg_from_quaternion(rotation: Iterable[float]) -> float:
    w, x, y, z = (float(value) for value in rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def wrap_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def angle_difference_deg(a: float, b: float) -> float:
    return abs(wrap_deg(a - b))


def channel_from_filename(filename: str) -> str:
    parts = Path(filename).name.split("__")
    return parts[1] if len(parts) >= 3 else ""


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def polygon_bbox(polygon: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        if (y1 > py) != (y2 > py):
            x_intersect = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_intersect:
                inside = not inside
    return inside


def _orientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    o1 = _orientation(a[0], a[1], b[0], b[1], c[0], c[1])
    o2 = _orientation(a[0], a[1], b[0], b[1], d[0], d[1])
    o3 = _orientation(c[0], c[1], d[0], d[1], a[0], a[1])
    o4 = _orientation(c[0], c[1], d[0], d[1], b[0], b[1])
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    return False


def nearest_route_projection(
    px: float,
    py: float,
    route: list[tuple[float, float]],
    candidate_segments: list[int],
) -> tuple[float, float, int, float] | None:
    """Return (distance, signed_side, segment_index, t) for the nearest candidate segment."""
    best: tuple[float, float, int, float] | None = None
    for index in candidate_segments:
        ax, ay = route[index]
        bx, by = route[index + 1]
        dx = bx - ax
        dy = by - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-6:
            continue
        t = ((px - ax) * dx + (py - ay) * dy) / length_sq
        t = max(0.0, min(1.0, t))
        distance = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
        side = dx * (py - ay) - dy * (px - ax)
        if best is None or distance < best[0]:
            best = (distance, side, index, t)
    return best


def detect_route_crossing(
    actor_polyline: list[tuple[float, float]],
    route: list[tuple[float, float]],
    route_cumulative: list[float],
    s_ego_start: float,
    half_width_m: float,
) -> tuple[bool, float | None]:
    """Detect whether the actor swept path crosses the ego route corridor ahead of the ego.

    A crossing requires: (a) consecutive actor points lie on opposite sides of the route
    polyline, (b) the closest approach is within `half_width_m`, and (c) the crossing
    arc-length is at or ahead of the ego's position at the start of the slice.
    """
    if len(actor_polyline) < 2 or len(route) < 2:
        return False, None
    xs = [point[0] for point in actor_polyline]
    ys = [point[1] for point in actor_polyline]
    margin = half_width_m + 1.0
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin
    candidates: list[int] = []
    for index in range(len(route) - 1):
        ax, ay = route[index]
        bx, by = route[index + 1]
        if max(ax, bx) < min_x or min(ax, bx) > max_x:
            continue
        if max(ay, by) < min_y or min(ay, by) > max_y:
            continue
        candidates.append(index)
    if not candidates:
        return False, None

    projections: list[tuple[float, float, int, float] | None] = [
        nearest_route_projection(x, y, route, candidates) for x, y in actor_polyline
    ]
    for position in range(len(projections) - 1):
        first = projections[position]
        second = projections[position + 1]
        if first is None or second is None:
            continue
        if first[1] * second[1] >= 0.0:
            continue
        if min(first[0], second[0]) > half_width_m:
            continue
        crossing_s: float | None = None
        for index in candidates:
            if segments_intersect(
                actor_polyline[position],
                actor_polyline[position + 1],
                route[index],
                route[index + 1],
            ):
                ax, ay = route[index]
                bx, by = route[index + 1]
                segment_length = math.hypot(bx - ax, by - ay)
                crossing_s = route_cumulative[index] + segment_length * 0.5
                break
        if crossing_s is None:
            first_s = route_cumulative[first[2]] + first[3] * (
                math.hypot(
                    route[first[2] + 1][0] - route[first[2]][0],
                    route[first[2] + 1][1] - route[first[2]][1],
                )
            )
            second_s = route_cumulative[second[2]] + second[3] * (
                math.hypot(
                    route[second[2] + 1][0] - route[second[2]][0],
                    route[second[2] + 1][1] - route[second[2]][1],
                )
            )
            crossing_s = max(first_s, second_s)
        if crossing_s >= s_ego_start - 1.0:
            return True, crossing_s
    return False, None


# ---------------------------------------------------------------------------
# Map expansion index
# ---------------------------------------------------------------------------


@dataclass
class MapPolygon:
    token: str
    kind: str  # "lane" | "lane_connector" | "ped_crossing" | "drivable_area"
    polygon: list[tuple[float, float]]
    bbox: tuple[float, float, float, float]
    node_tokens: tuple[str, ...]
    divider_nodes: frozenset[str]


class MapIndex:
    """Uniform-grid point-in-polygon index over map expansion polygons."""

    def __init__(self, cell_size: float = 10.0) -> None:
        self.cell_size = cell_size
        self.cells: dict[tuple[int, int], list[int]] = {}
        self.polygons: list[MapPolygon] = []
        self.lateral_neighbors: dict[str, set[str]] = {}

    def add(self, record: MapPolygon) -> None:
        index = len(self.polygons)
        self.polygons.append(record)
        min_x, min_y, max_x, max_y = record.bbox
        x0 = math.floor(min_x / self.cell_size)
        x1 = math.floor(max_x / self.cell_size)
        y0 = math.floor(min_y / self.cell_size)
        y1 = math.floor(max_y / self.cell_size)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 4096:
            return
        for cell_x in range(x0, x1 + 1):
            for cell_y in range(y0, y1 + 1):
                self.cells.setdefault((cell_x, cell_y), []).append(index)

    def query(self, x: float, y: float) -> list[MapPolygon]:
        cell = (math.floor(x / self.cell_size), math.floor(y / self.cell_size))
        found: list[MapPolygon] = []
        for index in self.cells.get(cell, ()):  # deterministic order (insertion order)
            record = self.polygons[index]
            min_x, min_y, max_x, max_y = record.bbox
            if x < min_x or x > max_x or y < min_y or y > max_y:
                continue
            if point_in_polygon(x, y, record.polygon):
                found.append(record)
        return found

    def tokens_at(self, x: float, y: float, kinds: frozenset[str]) -> list[str]:
        return sorted({record.token for record in self.query(x, y) if record.kind in kinds})

    def build_lateral_adjacency(self) -> None:
        """Lane polygons sharing >= 2 boundary nodes that are divider nodes."""
        lane_records = [record for record in self.polygons if record.kind == "lane"]
        lane_by_token = {record.token: record for record in lane_records}
        neighbor_candidates: dict[str, set[str]] = {record.token: set() for record in lane_records}
        for record in lane_records:
            min_x, min_y, max_x, max_y = record.bbox
            x0 = math.floor(min_x / self.cell_size)
            x1 = math.floor(max_x / self.cell_size)
            y0 = math.floor(min_y / self.cell_size)
            y1 = math.floor(max_y / self.cell_size)
            candidates: set[int] = set()
            for cell_x in range(x0, x1 + 1):
                for cell_y in range(y0, y1 + 1):
                    candidates.update(self.cells.get((cell_x, cell_y), ()))
            for index in sorted(candidates):
                other = self.polygons[index]
                if other.kind != "lane" or other.token == record.token:
                    continue
                shared = set(record.node_tokens) & set(other.node_tokens)
                if len(shared) < 2:
                    continue
                if shared & record.divider_nodes and shared & other.divider_nodes:
                    neighbor_candidates[record.token].add(other.token)
                    neighbor_candidates[other.token].add(record.token)
        self.lateral_neighbors = {
            token: neighbors for token, neighbors in neighbor_candidates.items() if neighbors
        }
        return None


def load_map_index(dataset_root: Path, location: str) -> MapIndex | None:
    expansion_path = dataset_root / "expansion" / f"{location}.json"
    if not expansion_path.is_file():
        return None
    payload = json.loads(expansion_path.read_text(encoding="utf-8"))
    nodes = {node["token"]: (float(node["x"]), float(node["y"])) for node in payload.get("node", [])}
    polygons = {polygon["token"]: polygon for polygon in payload.get("polygon", [])}

    def polygon_points(polygon_token: str) -> tuple[list[tuple[float, float]], tuple[str, ...]]:
        record = polygons.get(polygon_token)
        if record is None:
            return [], ()
        node_tokens = tuple(record.get("exterior_node_tokens", []))
        points = [nodes[token] for token in node_tokens if token in nodes]
        if len(points) != len(node_tokens):
            return [], ()
        return points, node_tokens

    index = MapIndex()
    for lane in payload.get("lane", []):
        points, node_tokens = polygon_points(str(lane.get("polygon_token", "")))
        if len(points) < 3:
            continue
        divider_nodes = frozenset(
            str(segment.get("node_token"))
            for segment in list(lane.get("left_lane_divider_segments", []))
            + list(lane.get("right_lane_divider_segments", []))
        )
        index.add(
            MapPolygon(
                token=str(lane["token"]),
                kind="lane",
                polygon=points,
                bbox=polygon_bbox(points),
                node_tokens=node_tokens,
                divider_nodes=divider_nodes,
            )
        )
    for connector in payload.get("lane_connector", []):
        points, node_tokens = polygon_points(str(connector.get("polygon_token", "")))
        if len(points) < 3:
            continue
        index.add(
            MapPolygon(
                token=str(connector["token"]),
                kind="lane_connector",
                polygon=points,
                bbox=polygon_bbox(points),
                node_tokens=node_tokens,
                divider_nodes=frozenset(),
            )
        )
    for crossing in payload.get("ped_crossing", []):
        points, node_tokens = polygon_points(str(crossing.get("polygon_token", "")))
        if len(points) < 3:
            continue
        index.add(
            MapPolygon(
                token=str(crossing["token"]),
                kind="ped_crossing",
                polygon=points,
                bbox=polygon_bbox(points),
                node_tokens=node_tokens,
                divider_nodes=frozenset(),
            )
        )
    for carpark in payload.get("carpark_area", []):
        points, node_tokens = polygon_points(str(carpark.get("polygon_token", "")))
        if len(points) < 3:
            continue
        index.add(
            MapPolygon(
                token=str(carpark["token"]),
                kind="carpark_area",
                polygon=points,
                bbox=polygon_bbox(points),
                node_tokens=node_tokens,
                divider_nodes=frozenset(),
            )
        )
    index.build_lateral_adjacency()
    return index


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@dataclass
class AnnotationRecord:
    token: str
    sample_token: str
    instance_token: str
    x: float
    y: float
    z: float
    yaw_deg: float
    attribute_tokens: tuple[str, ...]


@dataclass
class EgoState:
    sample_token: str
    timestamp: int
    x: float
    y: float
    z: float
    yaw_deg: float
    speed_mps: float | None
    lane_cells: list[str] = field(default_factory=list)
    lane_polygons: list[str] = field(default_factory=list)


@dataclass
class SceneContext:
    token: str
    name: str
    location: str
    description: str
    sample_tokens: list[str]
    sample_index: dict[str, int]
    sample_timestamps: list[int]
    ego: list[EgoState]
    annotations_by_sample: dict[str, list[AnnotationRecord]]
    instance_category: dict[str, str]
    instance_tracks: dict[str, list[AnnotationRecord]]
    map_index: MapIndex | None
    route: list[tuple[float, float]]
    route_cumulative: list[float]
    map_index_is_shared: bool = True


def load_dataset(dataset_root: Path, split: str, scene_name: str | None, scene_limit: int | None) -> tuple[dict[str, Any], list[SceneContext]]:
    table_root = dataset_root / split
    if not table_root.is_dir():
        raise SystemExit(f"Split directory not found: {table_root}")

    scenes = load_json_table(table_root, "scene")
    samples = load_json_table(table_root, "sample")
    instances = load_json_table(table_root, "instance")
    categories = load_json_table(table_root, "category")
    attributes = load_json_table(table_root, "attribute")
    logs = load_json_table(table_root, "log")

    paths = {
        "script": Path(__file__).resolve(),
        "tables": {},
    }

    sample_by_token = {str(sample["token"]): sample for sample in samples}
    category_by_token = {str(category["token"]): category for category in categories}
    attribute_by_token = {str(attribute["token"]): attribute for attribute in attributes}
    log_by_token = {str(log["token"]): log for log in logs}

    scene_records = sorted(scenes, key=lambda scene: (str(scene["name"]), str(scene["token"])))
    if scene_name is not None:
        scene_records = [scene for scene in scene_records if str(scene["name"]) == scene_name]
        if not scene_records:
            raise SystemExit(f"Scene {scene_name} not found in {table_root}")
    if scene_limit is not None:
        scene_records = scene_records[: int(scene_limit)]

    # Sample chains per scene.
    scene_samples: dict[str, list[str]] = {}
    for scene in scene_records:
        chain: list[str] = []
        token = str(scene.get("first_sample_token", ""))
        guard = 0
        while token:
            sample = sample_by_token.get(token)
            if sample is None:
                raise SystemExit(f"Scene {scene['name']} references unknown sample {token}")
            chain.append(token)
            token = str(sample.get("next", ""))
            guard += 1
            if guard > 100000:
                raise SystemExit(f"Sample chain for scene {scene['name']} exceeds guard")
        scene_samples[str(scene["token"])] = chain

    selected_sample_tokens = {token for chain in scene_samples.values() for token in chain}

    # Keyframe ego pose per sample (prefer LIDAR_TOP, mirroring EXP-017).
    keyframe_pose_token: dict[str, str] = {}
    keyframe_pose_priority: dict[str, int] = {}

    def project_sample_data(record: dict[str, Any]) -> dict[str, Any] | None:
        if not bool(record.get("is_key_frame", False)):
            return None
        sample_token = str(record.get("sample_token", ""))
        if sample_token not in selected_sample_tokens:
            return None
        channel = channel_from_filename(str(record.get("filename", "")))
        return {
            "sample_token": sample_token,
            "ego_pose_token": str(record.get("ego_pose_token", "")),
            "channel": channel,
        }

    for record in iter_json_array(table_root / "sample_data.json", project_sample_data):
        sample_token = record["sample_token"]
        channel = record["channel"]
        priority = 0 if channel == "LIDAR_TOP" else 1
        current = keyframe_pose_priority.get(sample_token)
        if current is None or priority < current:
            keyframe_pose_priority[sample_token] = priority
            keyframe_pose_token[sample_token] = record["ego_pose_token"]

    selected_pose_tokens = set(keyframe_pose_token.values())

    def project_ego_pose(record: dict[str, Any]) -> dict[str, Any] | None:
        token = str(record.get("token", ""))
        if token not in selected_pose_tokens:
            return None
        return {
            "token": token,
            "translation": tuple(float(value) for value in record["translation"]),
            "rotation": tuple(float(value) for value in record["rotation"]),
        }

    ego_pose_by_token: dict[str, dict[str, Any]] = {}
    for record in iter_json_array(table_root / "ego_pose.json", project_ego_pose):
        ego_pose_by_token[record["token"]] = record

    def project_annotation(record: dict[str, Any]) -> dict[str, Any] | None:
        sample_token = str(record.get("sample_token", ""))
        if sample_token not in selected_sample_tokens:
            return None
        return {
            "token": str(record.get("token", "")),
            "sample_token": sample_token,
            "instance_token": str(record.get("instance_token", "")),
            "translation": tuple(float(value) for value in record["translation"]),
            "rotation": tuple(float(value) for value in record["rotation"]),
            "attribute_tokens": tuple(str(value) for value in record.get("attribute_tokens", [])),
        }

    annotations_by_sample: dict[str, list[AnnotationRecord]] = {}
    for record in iter_json_array(table_root / "sample_annotation.json", project_annotation):
        annotation = AnnotationRecord(
            token=record["token"],
            sample_token=record["sample_token"],
            instance_token=record["instance_token"],
            x=record["translation"][0],
            y=record["translation"][1],
            z=record["translation"][2],
            yaw_deg=yaw_deg_from_quaternion(record["rotation"]),
            attribute_tokens=record["attribute_tokens"],
        )
        annotations_by_sample.setdefault(annotation.sample_token, []).append(annotation)

    instance_category = {
        str(instance["token"]): str(category_by_token.get(str(instance["category_token"]), {}).get("name", ""))
        for instance in instances
    }
    attribute_name_by_token = {
        token: str(attribute.get("name", "")) for token, attribute in attribute_by_token.items()
    }

    map_index_cache: dict[str, MapIndex | None] = {}
    scene_contexts: list[SceneContext] = []
    for scene in scene_records:
        scene_token = str(scene["token"])
        log = log_by_token.get(str(scene["log_token"]), {})
        location = str(log.get("location", "unknown"))
        if location not in map_index_cache:
            map_index_cache[location] = load_map_index(dataset_root, location)
        map_index = map_index_cache[location]

        sample_tokens = scene_samples[scene_token]
        sample_index = {token: index for index, token in enumerate(sample_tokens)}
        timestamps = [int(sample_by_token[token].get("timestamp", 0)) for token in sample_tokens]

        ego_states: list[EgoState] = []
        for index, token in enumerate(sample_tokens):
            pose_token = keyframe_pose_token.get(token)
            if pose_token is None or pose_token not in ego_pose_by_token:
                raise SystemExit(f"Missing keyframe ego pose for sample {token} in scene {scene['name']}")
            pose = ego_pose_by_token[pose_token]
            x, y, z = pose["translation"]
            yaw_deg = yaw_deg_from_quaternion(pose["rotation"])
            lane_cells = map_index.tokens_at(x, y, frozenset({"lane", "lane_connector"})) if map_index else []
            lane_polygons = (
                map_index.tokens_at(x, y, frozenset({"lane"})) if map_index else []
            )
            ego_states.append(
                EgoState(
                    sample_token=token,
                    timestamp=timestamps[index],
                    x=x,
                    y=y,
                    z=z,
                    yaw_deg=yaw_deg,
                    speed_mps=None,
                    lane_cells=lane_cells,
                    lane_polygons=lane_polygons,
                )
            )
        # Ego speed from keyframe pose finite differences (one-sided at scene ends).
        for index, state in enumerate(ego_states):
            if len(ego_states) < 2:
                continue
            if index == 0:
                other = ego_states[1]
            elif index == len(ego_states) - 1:
                other = ego_states[index - 1]
            else:
                prev_state = ego_states[index - 1]
                next_state = ego_states[index + 1]
                delta_seconds = abs(next_state.timestamp - prev_state.timestamp) / 1e6
                if delta_seconds > 0:
                    state.speed_mps = math.hypot(
                        next_state.x - prev_state.x, next_state.y - prev_state.y
                    ) / delta_seconds
                continue
            delta_seconds = abs(other.timestamp - state.timestamp) / 1e6
            if delta_seconds > 0:
                state.speed_mps = math.hypot(other.x - state.x, other.y - state.y) / delta_seconds

        # Ego route proxy: the executed ego trajectory over the whole scene, with
        # cumulative arc length so slices can ask "is the crossing ahead of me?".
        route = [(state.x, state.y) for state in ego_states]
        route_cumulative = [0.0]
        for index in range(1, len(route)):
            route_cumulative.append(
                route_cumulative[-1]
                + math.hypot(route[index][0] - route[index - 1][0], route[index][1] - route[index - 1][1])
            )

        # Actor tracks per instance (ordered by sample index).
        scene_annotations: dict[str, list[AnnotationRecord]] = {}
        for token in sample_tokens:
            for annotation in annotations_by_sample.get(token, []):
                scene_annotations.setdefault(annotation.instance_token, []).append(annotation)
        tracks: dict[str, list[AnnotationRecord]] = {}
        for instance_token, records in scene_annotations.items():
            tracks[instance_token] = sorted(records, key=lambda record: sample_index[record.sample_token])

        scene_contexts.append(
            SceneContext(
                token=scene_token,
                name=str(scene["name"]),
                location=location,
                description=str(scene.get("description", "")),
                sample_tokens=sample_tokens,
                sample_index=sample_index,
                sample_timestamps=timestamps,
                ego=ego_states,
                annotations_by_sample=annotations_by_sample,
                instance_category=instance_category,
                instance_tracks=tracks,
                map_index=map_index,
                route=route,
                route_cumulative=route_cumulative,
            )
        )

    dataset = {
        "attribute_name_by_token": attribute_name_by_token,
        "scene_count_total": len(scenes),
        "scene_count_selected": len(scene_records),
    }
    return dataset, scene_contexts


# ---------------------------------------------------------------------------
# Slice evaluation
# ---------------------------------------------------------------------------


@dataclass
class SliceActorState:
    instance_token: str
    node_type: str
    nuscenes_category: str
    present_keyframes: list[int]
    sample_tokens: list[str]
    annotation_tokens: list[str]
    speeds: dict[int, float | None]
    stationary_keyframes: list[int]
    moving_keyframes: list[int]
    in_front_keyframes: list[int]
    same_lane_keyframes: list[int]
    adjacent_lane_keyframes: list[int]
    oncoming_keyframes: list[int]
    on_road_keyframes: list[int]
    jaywalking_keyframes: list[int]
    obstructing: bool
    crossing_path: bool
    approaching: bool
    braking: bool
    turning: bool
    lane_changing: bool
    label_predicates: list[str]
    ego_distances: dict[int, float]


@dataclass
class SliceResult:
    slice_index: int
    start_keyframe: int
    sample_tokens: list[str]
    actors: dict[str, SliceActorState]
    ego_stationary: bool
    ego_moving: bool
    ego_braking: bool
    ego_turning: bool
    ego_lane_changing: bool


def actor_speed_at(
    track: list[AnnotationRecord],
    sample_index: dict[str, int],
    sample_timestamps: dict[str, int],
    position: int,
) -> float | None:
    def distance(a: AnnotationRecord, b: AnnotationRecord) -> float:
        return math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)

    if position > 0 and position < len(track) - 1:
        prev_record = track[position - 1]
        next_record = track[position + 1]
        delta_seconds = abs(
            sample_timestamps[next_record.sample_token] - sample_timestamps[prev_record.sample_token]
        ) / 1e6
        if delta_seconds <= 0:
            return None
        return distance(prev_record, next_record) / delta_seconds
    if position > 0:
        prev_record = track[position - 1]
        delta_seconds = abs(
            sample_timestamps[track[position].sample_token] - sample_timestamps[prev_record.sample_token]
        ) / 1e6
        if delta_seconds <= 0:
            return None
        return distance(prev_record, track[position]) / delta_seconds
    if position < len(track) - 1:
        next_record = track[position + 1]
        delta_seconds = abs(
            sample_timestamps[next_record.sample_token] - sample_timestamps[track[position].sample_token]
        ) / 1e6
        if delta_seconds <= 0:
            return None
        return distance(track[position], next_record) / delta_seconds
    return None


def evaluate_slice(
    context: SceneContext,
    dataset: dict[str, Any],
    slice_index: int,
    start: int,
    keyframes: list[int],
    params: argparse.Namespace,
    ungrounded_reasons: dict[str, dict[str, int]],
) -> SliceResult:
    def note_ungrounded(predicate: str, reason: str) -> None:
        ungrounded_reasons.setdefault(predicate, {})
        ungrounded_reasons[predicate][reason] = ungrounded_reasons[predicate].get(reason, 0) + 1

    sample_timestamps = {
        token: context.sample_timestamps[index] for index, token in enumerate(context.sample_tokens)
    }
    attribute_name_by_token = dataset["attribute_name_by_token"]
    ego_states = [context.ego[index] for index in keyframes]
    ego_polyline = [(state.x, state.y) for state in ego_states]
    ego_lane_cells = [set(state.lane_cells) for state in ego_states]
    ego_lane_polygons = [set(state.lane_polygons) for state in ego_states]
    ego_stationary_flags = [
        state.speed_mps is not None and state.speed_mps < params.stationary_threshold_mps
        for state in ego_states
    ]
    ego_moving_flags = [
        state.speed_mps is not None and state.speed_mps >= params.stationary_threshold_mps
        for state in ego_states
    ]

    actors: dict[str, SliceActorState] = {}
    annotation_by_token: dict[str, AnnotationRecord] = {}
    for index in keyframes:
        sample_token = context.sample_tokens[index]
        for annotation in context.annotations_by_sample.get(sample_token, []):
            annotation_by_token[annotation.token] = annotation

    instance_ids = sorted(
        {
            annotation.instance_token
            for index in keyframes
            for annotation in context.annotations_by_sample.get(context.sample_tokens[index], [])
        }
    )
    for instance_token in instance_ids:
        track = context.instance_tracks.get(instance_token, [])
        track_positions = {record.sample_token: position for position, record in enumerate(track)}
        present = [
            (keyframe_position, annotation_by_token[annotation.token])
            for keyframe_position, index in enumerate(keyframes)
            for annotation in context.annotations_by_sample.get(context.sample_tokens[index], [])
            if annotation.instance_token == instance_token
        ]
        present.sort(key=lambda item: item[0])
        if not present:
            continue
        category = context.instance_category.get(instance_token, "")
        node_type = NODE_ALIAS_TABLE.get(category, category.split(".")[-1] if category else "unknown")

        speeds: dict[int, float | None] = {}
        stationary_keyframes: list[int] = []
        moving_keyframes: list[int] = []
        in_front_keyframes: list[int] = []
        same_lane_keyframes: list[int] = []
        adjacent_lane_keyframes: list[int] = []
        oncoming_keyframes: list[int] = []
        on_road_keyframes: list[int] = []
        jaywalking_keyframes: list[int] = []
        lane_polygon_series: dict[int, set[str]] = {}
        yaw_series: dict[int, float] = {}
        label_predicates: set[str] = set()
        for keyframe_position, annotation in present:
            position = track_positions.get(annotation.sample_token)
            speed = None
            if position is not None:
                speed = actor_speed_at(track, context.sample_index, sample_timestamps, position)
            speeds[keyframe_position] = speed
            if speed is None:
                note_ungrounded("stationary", "actor_speed_unavailable")
            elif speed < params.stationary_threshold_mps:
                stationary_keyframes.append(keyframe_position)
            else:
                moving_keyframes.append(keyframe_position)
            if not context.map_index:
                note_ungrounded("same_lane", "no_map_expansion_file")
                note_ungrounded("adjacent_lane", "no_map_expansion_file")
                note_ungrounded("oncoming", "no_map_expansion_file")
                note_ungrounded("on_road", "no_map_expansion_file")
            ego_state = ego_states[keyframe_position]
            forward_x = math.cos(math.radians(ego_state.yaw_deg))
            forward_y = math.sin(math.radians(ego_state.yaw_deg))
            dot = (annotation.x - ego_state.x) * forward_x + (annotation.y - ego_state.y) * forward_y
            if dot > 0.0:
                in_front_keyframes.append(keyframe_position)

            lane_cells: set[str] = set()
            lane_polygons: set[str] = set()
            if context.map_index:
                lane_cells = set(
                    context.map_index.tokens_at(annotation.x, annotation.y, frozenset({"lane", "lane_connector"}))
                )
                lane_polygons = set(
                    context.map_index.tokens_at(annotation.x, annotation.y, frozenset({"lane"}))
                )
                if lane_cells:
                    on_road_keyframes.append(keyframe_position)
                elif node_type != "pedestrian":
                    note_ungrounded("on_road", "actor_not_in_lane_cell")
            lane_polygon_series[keyframe_position] = lane_polygons
            yaw_series[keyframe_position] = annotation.yaw_deg

            if lane_cells and ego_lane_cells[keyframe_position]:
                if lane_cells & ego_lane_cells[keyframe_position]:
                    same_lane_keyframes.append(keyframe_position)
                else:
                    neighbors: set[str] = set()
                    for token in ego_lane_cells[keyframe_position]:
                        neighbors.update(
                            context.map_index.lateral_neighbors.get(token, set())
                            if context.map_index
                            else set()
                        )
                    if lane_cells & neighbors:
                        adjacent_lane_keyframes.append(keyframe_position)
            else:
                reason = "no_map_expansion_file" if not context.map_index else (
                    "ego_not_in_lane_cell" if not ego_lane_cells[keyframe_position] else "actor_not_in_lane_cell"
                )
                note_ungrounded("same_lane", reason)
                note_ungrounded("adjacent_lane", reason)

            actor_moving_here = speed is not None and speed >= params.stationary_threshold_mps
            if (
                node_type == "vehicle"
                and angle_difference_deg(annotation.yaw_deg, ego_state.yaw_deg)
                >= params.oncoming_opposition_deg
                and dot > 0.0
                and actor_moving_here
            ):
                if lane_cells and ego_lane_cells[keyframe_position]:
                    oncoming_keyframes.append(keyframe_position)
                else:
                    reason = "no_map_expansion_file" if not context.map_index else (
                        "ego_not_in_lane_cell"
                        if not ego_lane_cells[keyframe_position]
                        else "actor_not_in_lane_cell"
                    )
                    note_ungrounded("oncoming", reason)

            if node_type == "pedestrian" and context.map_index:
                off_crossing = not context.map_index.tokens_at(
                    annotation.x, annotation.y, frozenset({"ped_crossing"})
                )
                off_carpark = not context.map_index.tokens_at(
                    annotation.x, annotation.y, frozenset({"carpark_area"})
                )
                if lane_cells and off_crossing and off_carpark:
                    jaywalking_keyframes.append(keyframe_position)

            for attribute_token in annotation.attribute_tokens:
                attribute_name = attribute_name_by_token.get(attribute_token, "")
                predicate = ATTRIBUTE_LABEL_TABLE.get(attribute_name)
                if predicate:
                    label_predicates.add(predicate)

        # Slice-level derivations.
        first_keyframe, first_annotation = present[0]
        last_keyframe, last_annotation = present[-1]
        first_speed = speeds.get(first_keyframe)
        last_speed = speeds.get(last_keyframe)
        braking = (
            first_speed is not None
            and last_speed is not None
            and first_speed >= params.stationary_threshold_mps
            and (first_speed - last_speed) >= params.braking_drop_mps
        )
        displacement = math.hypot(
            last_annotation.x - first_annotation.x, last_annotation.y - first_annotation.y
        )
        turning = (
            node_type == "vehicle"
            and displacement >= 2.0
            and angle_difference_deg(yaw_series[last_keyframe], yaw_series[first_keyframe])
            >= params.turning_yaw_deg
        )
        lane_changing = False
        if node_type == "vehicle" and context.map_index and len(lane_polygon_series) >= 2:
            first_lanes = lane_polygon_series[first_keyframe]
            last_lanes = lane_polygon_series[last_keyframe]
            if first_lanes and last_lanes and first_lanes != last_lanes:
                for token in first_lanes:
                    neighbors = context.map_index.lateral_neighbors.get(token, set())
                    if neighbors & last_lanes:
                        lane_changing = True
                        break

        ego_distance_first = math.hypot(
            first_annotation.x - ego_states[first_keyframe].x,
            first_annotation.y - ego_states[first_keyframe].y,
        )
        ego_distance_last = math.hypot(
            last_annotation.x - ego_states[last_keyframe].x,
            last_annotation.y - ego_states[last_keyframe].y,
        )
        approaching = (
            node_type in AGENT_NODE_TYPES
            and (ego_distance_first - ego_distance_last) >= params.approaching_drop_m
        )

        crossing_path = False
        if node_type in AGENT_NODE_TYPES and len(present) >= 2 and displacement >= 0.5:
            actor_polyline = [(annotation.x, annotation.y) for _, annotation in present]
            route_crossing, _ = detect_route_crossing(
                actor_polyline,
                context.route,
                context.route_cumulative,
                context.route_cumulative[start],
                params.crossing_corridor_half_width_m,
            )
            if route_crossing:
                if node_type == "pedestrian":
                    crossing_path = True
                else:
                    heading_ok = any(
                        30.0
                        <= angle_difference_deg(
                            yaw_series[keyframe_position], ego_states[keyframe_position].yaw_deg
                        )
                        <= 150.0
                        for keyframe_position, _ in present
                    )
                    crossing_path = heading_ok

        obstructing = bool(
            on_road_keyframes and in_front_keyframes and same_lane_keyframes
        )

        actors[instance_token] = SliceActorState(
            instance_token=instance_token,
            node_type=node_type,
            nuscenes_category=category,
            present_keyframes=[keyframe_position for keyframe_position, _ in present],
            sample_tokens=[annotation.sample_token for _, annotation in present],
            annotation_tokens=[annotation.token for _, annotation in present],
            speeds=speeds,
            stationary_keyframes=stationary_keyframes,
            moving_keyframes=moving_keyframes,
            in_front_keyframes=in_front_keyframes,
            same_lane_keyframes=same_lane_keyframes,
            adjacent_lane_keyframes=adjacent_lane_keyframes,
            oncoming_keyframes=oncoming_keyframes,
            on_road_keyframes=on_road_keyframes,
            jaywalking_keyframes=jaywalking_keyframes,
            obstructing=obstructing,
            crossing_path=crossing_path,
            approaching=approaching,
            braking=braking,
            turning=turning,
            lane_changing=lane_changing,
            label_predicates=sorted(label_predicates),
            ego_distances={first_keyframe: ego_distance_first, last_keyframe: ego_distance_last},
        )

    def consecutive_stationary(keyframes_present: list[int], stationary: list[int]) -> bool:
        stationary_set = set(stationary)
        run = 0
        for keyframe_position in keyframes_present:
            if keyframe_position in stationary_set:
                run += 1
                if run >= 2:
                    return True
            else:
                run = 0
        return False

    for state in actors.values():
        state.waiting_persistent = consecutive_stationary(state.present_keyframes, state.stationary_keyframes)  # type: ignore[attr-defined]

    # Ego slice-level derivations.
    ego_first_speed = ego_states[0].speed_mps
    ego_last_speed = ego_states[-1].speed_mps
    ego_braking = (
        ego_first_speed is not None
        and ego_last_speed is not None
        and ego_first_speed >= params.stationary_threshold_mps
        and (ego_first_speed - ego_last_speed) >= params.braking_drop_mps
    )
    ego_yaw_start = ego_states[0].yaw_deg
    ego_yaw_end = ego_states[-1].yaw_deg
    ego_displacement = math.hypot(ego_states[-1].x - ego_states[0].x, ego_states[-1].y - ego_states[0].y)
    ego_turning = (
        ego_displacement >= 2.0 and angle_difference_deg(ego_yaw_end, ego_yaw_start) >= params.turning_yaw_deg
    )
    ego_lane_changing = False
    if context.map_index and ego_lane_polygons[0] and ego_lane_polygons[-1] and ego_lane_polygons[0] != ego_lane_polygons[-1]:
        for token in ego_lane_polygons[0]:
            if context.map_index.lateral_neighbors.get(token, set()) & ego_lane_polygons[-1]:
                ego_lane_changing = True
                break

    return SliceResult(
        slice_index=slice_index,
        start_keyframe=start,
        sample_tokens=[context.sample_tokens[index] for index in keyframes],
        actors=actors,
        ego_stationary=any(ego_stationary_flags),
        ego_moving=any(ego_moving_flags),
        ego_braking=ego_braking,
        ego_turning=ego_turning,
        ego_lane_changing=ego_lane_changing,
    )


# ---------------------------------------------------------------------------
# Hazard classes
# ---------------------------------------------------------------------------


@dataclass
class HazardClass:
    name: str
    grounding: str
    definition: str
    dota_ancestor: str | None
    required_predicates: list[str]
    evaluate: Callable[[SliceActorState], bool]


def dota_rule_for(ancestor: str) -> str | None:
    if _dota_symbolic is None:
        return None
    entry = _dota_symbolic.ANOMALY_DESCRIPTIONS.get(ancestor)
    if not entry:
        return None
    rules = entry.get("scallop_rules", [])
    return str(rules[0]) if rules else None


def dota_formula_for(ancestor: str) -> str | None:
    if _dota_symbolic is None:
        return None
    entry = _dota_symbolic.ANOMALY_DESCRIPTIONS.get(ancestor)
    if not entry:
        return None
    return str(entry.get("stsl_formula", "")) or None


def build_hazard_classes() -> list[HazardClass]:
    return [
        HazardClass(
            name="other: ahead_or_waiting",
            grounding="derived",
            definition="vehicle actor that is in front of the ego and stationary for >= 2 consecutive keyframes",
            dota_ancestor="other: ahead_or_waiting",
            required_predicates=["in_front_of", "stationary"],
            evaluate=lambda actor: actor.node_type == "vehicle"
            and bool(actor.in_front_keyframes)
            and bool(getattr(actor, "waiting_persistent", False)),
        ),
        HazardClass(
            name="other: start_stop_or_stationary",
            grounding="derived",
            definition="vehicle actor that is in front of the ego and drops >= 1.0 m/s of speed across the slice",
            dota_ancestor="other: start_stop_or_stationary",
            required_predicates=["in_front_of", "braking"],
            evaluate=lambda actor: actor.node_type == "vehicle"
            and bool(actor.in_front_keyframes)
            and actor.braking,
        ),
        HazardClass(
            name="other: oncoming",
            grounding="derived",
            definition="vehicle actor with opposed heading in front of the ego that is closing distance",
            dota_ancestor="other: oncoming",
            required_predicates=["oncoming", "approaching"],
            evaluate=lambda actor: actor.node_type == "vehicle"
            and bool(actor.oncoming_keyframes)
            and actor.approaching,
        ),
        HazardClass(
            name="other: lateral",
            grounding="derived",
            definition="vehicle actor that changes to a laterally adjacent lane and crosses the ego path",
            dota_ancestor="other: lateral",
            required_predicates=["lane_changing", "crossing_path"],
            evaluate=lambda actor: actor.node_type == "vehicle" and actor.lane_changing and actor.crossing_path,
        ),
        HazardClass(
            name="other: turning",
            grounding="derived",
            definition="vehicle actor that turns >= 30 deg while crossing the ego path",
            dota_ancestor="other: turning",
            required_predicates=["turning", "crossing_path"],
            evaluate=lambda actor: actor.node_type == "vehicle" and actor.turning and actor.crossing_path,
        ),
        HazardClass(
            name="other: obstacle",
            grounding="derived",
            definition="static movable object on the road, in front of the ego, in the ego lane",
            dota_ancestor="other: obstacle",
            required_predicates=["on_road", "obstructing"],
            evaluate=lambda actor: actor.node_type in STATIC_OBJECT_NODES and actor.obstructing,
        ),
        HazardClass(
            name="other: pedestrian",
            grounding="proxy",
            definition="pedestrian on the roadway outside a ped_crossing polygon that crosses the ego path (jaywalking proxy)",
            dota_ancestor="other: pedestrian",
            required_predicates=["jaywalking", "crossing_path"],
            evaluate=lambda actor: actor.node_type == "pedestrian"
            and bool(actor.jaywalking_keyframes)
            and actor.crossing_path,
        ),
        HazardClass(
            name="pedestrian_in_path",
            grounding="derived",
            definition="EXP-018 explicit class: pedestrian in front of the ego whose swept path crosses the ego corridor",
            dota_ancestor=None,
            required_predicates=["in_front_of", "crossing_path"],
            evaluate=lambda actor: actor.node_type == "pedestrian"
            and bool(actor.in_front_keyframes)
            and actor.crossing_path,
        ),
        HazardClass(
            name="oncoming_cut_in",
            grounding="derived",
            definition="EXP-018 explicit class (paper example): oncoming actor that reaches the ego lane later in the slice",
            dota_ancestor=None,
            required_predicates=["oncoming", "same_lane"],
            evaluate=lambda actor: actor.node_type == "vehicle"
            and bool(actor.oncoming_keyframes)
            and any(
                same_lane > oncoming
                for oncoming in actor.oncoming_keyframes
                for same_lane in actor.same_lane_keyframes
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class ObligationAggregator:
    def __init__(self, max_witnesses: int) -> None:
        self.max_witnesses = max_witnesses
        self.records: dict[str, dict[str, Any]] = {}
        self._slice_keys: dict[str, set[tuple[str, int]]] = {}
        self._scene_keys: dict[str, set[str]] = {}
        self._instance_keys: dict[str, set[str]] = {}
        self._witness_keys: dict[str, set[tuple[str, int, str]]] = {}

    def add(
        self,
        *,
        dimension: str,
        predicate: str,
        signature: str,
        node_types: list[str],
        scene: SceneContext,
        slice_index: int,
        sample_tokens: list[str],
        instance_tokens: list[str],
        annotation_tokens: list[str],
        keyframes: list[int],
        sources: list[str],
        node_type: str | None = None,
        nuscenes_category: str | None = None,
        extra: dict[str, Any] | None = None,
        grounding: str | None = None,
    ) -> None:
        record = self.records.get(signature)
        if record is None:
            record = {
                "signature": signature,
                "dimension": dimension,
                "predicate": predicate,
                "node_types": node_types,
                "grounding": grounding
                or PREDICATE_DEFINITIONS.get(predicate, {}).get("grounding", "derived"),
                "support": {"slices": 0, "scenes": 0, "actor_instances": 0, "keyframe_occurrences": 0},
                "witnesses": [],
                "witnesses_omitted": 0,
            }
            if nuscenes_category is not None:
                record["nuscenes_categories"] = []
            self.records[signature] = record
            self._slice_keys[signature] = set()
            self._scene_keys[signature] = set()
            self._instance_keys[signature] = set()
            self._witness_keys[signature] = set()
        if nuscenes_category is not None and nuscenes_category not in record.get("nuscenes_categories", []):
            record.setdefault("nuscenes_categories", []).append(nuscenes_category)
        slice_key = (scene.name, slice_index)
        scene_key = scene.name
        instance_key = f"{scene.name}::{instance_tokens[0]}" if instance_tokens else f"{scene.name}::ego"
        witness_key = (scene.name, slice_index, instance_key)
        self._slice_keys[signature].add(slice_key)
        self._scene_keys[signature].add(scene_key)
        if instance_tokens:
            self._instance_keys[signature].add(instance_key)
        record["support"]["keyframe_occurrences"] += max(len(keyframes), 1)
        if witness_key in self._witness_keys[signature]:
            return
        self._witness_keys[signature].add(witness_key)
        if len(record["witnesses"]) < self.max_witnesses:
            witness: dict[str, Any] = {
                "scene": scene.name,
                "scene_token": scene.token,
                "slice_index": slice_index,
                "sample_tokens": list(sample_tokens),
                "keyframes": list(keyframes),
                "sources": list(sources),
            }
            if node_type is not None:
                witness["node_type"] = node_type
            if nuscenes_category is not None:
                witness["nuscenes_category"] = nuscenes_category
            if instance_tokens:
                witness["instance_tokens"] = list(instance_tokens)
            if annotation_tokens:
                witness["annotation_tokens"] = list(annotation_tokens)
            if extra:
                witness.update(extra)
            record["witnesses"].append(witness)
        else:
            record["witnesses_omitted"] += 1

    def finalize(self) -> list[dict[str, Any]]:
        for signature, record in self.records.items():
            record["support"]["slices"] = len(self._slice_keys[signature])
            record["support"]["scenes"] = len(self._scene_keys[signature])
            record["support"]["actor_instances"] = len(self._instance_keys[signature])
            record["witnesses"].sort(key=lambda item: (item["scene"], item["slice_index"], item.get("node_type", "")))
            if "nuscenes_categories" in record:
                record["nuscenes_categories"].sort()
        return [self.records[signature] for signature in sorted(self.records)]


def accumulate_slice(
    context: SceneContext,
    result: SliceResult,
    aggregator: ObligationAggregator,
    node_instances: dict[str, dict[str, Any]],
    hazard_aggregator: ObligationAggregator,
    predicate_counts: dict[str, dict[str, int]],
    hazard_classes: list[HazardClass],
) -> None:
    def bump(predicate: str, level: str, amount: int = 1) -> None:
        predicate_counts.setdefault(predicate, {"keyframe": 0, "slice_actor": 0})
        predicate_counts[predicate][level] += amount

    def select(tokens: list[str], keyframes: list[int]) -> list[str]:
        return [token for position, token in enumerate(tokens) if position in keyframes]

    all_sample_tokens = result.sample_tokens
    for state in result.actors.values():
        # Node obligation (one per scene actor instance).
        node_key = f"{context.name}::{state.instance_token}"
        record = node_instances.get(node_key)
        if record is None:
            record = {
                "scene": context.name,
                "scene_token": context.token,
                "location": context.location,
                "instance_token": state.instance_token,
                "node_type": state.node_type,
                "nuscenes_category": state.nuscenes_category,
                "slices": set(),
                "sample_tokens": set(),
                "annotation_tokens": set(),
            }
            node_instances[node_key] = record
        record["slices"].add(result.slice_index)
        record["sample_tokens"].update(state.sample_tokens)
        record["annotation_tokens"].update(state.annotation_tokens)
        aggregator.add(
            dimension="node",
            predicate=f"node:{state.node_type}",
            signature=f"node({state.node_type})",
            node_types=[state.node_type],
            scene=context,
            slice_index=result.slice_index,
            sample_tokens=state.sample_tokens,
            instance_tokens=[state.instance_token],
            annotation_tokens=state.annotation_tokens,
            keyframes=state.present_keyframes,
            sources=["nuscenes_category_alias"],
            node_type=state.node_type,
            nuscenes_category=state.nuscenes_category,
            grounding="direct",
        )

        # Attribute obligations.
        attribute_hits: list[tuple[str, list[str], list[int]]] = []
        if state.stationary_keyframes:
            attribute_hits.append(("stationary", ["track_finite_difference"], state.stationary_keyframes))
        if state.moving_keyframes:
            attribute_hits.append(("moving", ["track_finite_difference"], state.moving_keyframes))
        if (
            state.node_type in AGENT_NODE_TYPES
            and getattr(state, "waiting_persistent", False)
            and state.in_front_keyframes
        ):
            attribute_hits.append(("waiting", ["stationary_persistent", "in_front_of"], state.in_front_keyframes))
        if state.braking:
            attribute_hits.append(("braking", ["track_finite_difference"], state.present_keyframes))
        if state.turning:
            attribute_hits.append(("turning", ["track_finite_difference"], state.present_keyframes))
        if state.lane_changing:
            attribute_hits.append(("lane_changing", ["map_lane_adjacency"], state.present_keyframes))
        if state.on_road_keyframes:
            attribute_hits.append(("on_road", ["map_lane_membership"], state.on_road_keyframes))
        if state.jaywalking_keyframes:
            attribute_hits.append(("jaywalking", ["map_lane_membership", "map_ped_crossing"], state.jaywalking_keyframes))
        for predicate in state.label_predicates:
            attribute_hits.append((predicate, ["nuscenes_attribute"], state.present_keyframes))
        attribute_hits.sort(key=lambda item: item[0])
        for predicate, sources, keyframes in attribute_hits:
            aggregator.add(
                dimension="attribute",
                predicate=predicate,
                signature=f"{predicate}({state.node_type})",
                node_types=[state.node_type],
                scene=context,
                slice_index=result.slice_index,
                sample_tokens=select(all_sample_tokens, keyframes),
                instance_tokens=[state.instance_token],
                annotation_tokens=select(state.annotation_tokens, keyframes) or state.annotation_tokens,
                keyframes=keyframes,
                sources=sources,
                node_type=state.node_type,
                nuscenes_category=state.nuscenes_category,
            )
            bump(predicate, "slice_actor")
            bump(predicate, "keyframe", len(keyframes))

        # Relation obligations.
        relation_hits: list[tuple[str, bool, list[int], list[str]]] = [
            ("in_front_of", bool(state.in_front_keyframes), state.in_front_keyframes, ["ego_forward_half_plane"]),
            ("same_lane", bool(state.same_lane_keyframes), state.same_lane_keyframes, ["map_lane_membership"]),
            ("adjacent_lane", bool(state.adjacent_lane_keyframes), state.adjacent_lane_keyframes, ["map_lane_adjacency"]),
            ("oncoming", bool(state.oncoming_keyframes), state.oncoming_keyframes, ["heading_opposition", "map_lane_membership"]),
            ("approaching", state.approaching, state.present_keyframes, ["ego_distance_decrease"]),
            ("crossing_path", state.crossing_path, state.present_keyframes, ["swept_path_corridor_intersection"]),
            ("obstructing", state.obstructing, state.in_front_keyframes, ["on_road", "in_front_of", "same_lane"]),
        ]
        for predicate, holds, keyframes, sources in relation_hits:
            if not holds:
                continue
            aggregator.add(
                dimension="relation",
                predicate=predicate,
                signature=f"{predicate}({state.node_type},ego)",
                node_types=[state.node_type, "ego"],
                scene=context,
                slice_index=result.slice_index,
                sample_tokens=select(all_sample_tokens, keyframes),
                instance_tokens=[state.instance_token],
                annotation_tokens=select(state.annotation_tokens, keyframes) or state.annotation_tokens,
                keyframes=keyframes,
                sources=sources,
                node_type=state.node_type,
                nuscenes_category=state.nuscenes_category,
            )
            bump(predicate, "slice_actor")
            bump(predicate, "keyframe", len(keyframes))

    # Ego attribute obligations.
    ego_hits: list[tuple[str, list[str], list[int]]] = []
    if result.ego_stationary:
        ego_hits.append(("stationary", ["ego_pose_finite_difference"], [0]))
    if result.ego_moving:
        ego_hits.append(("moving", ["ego_pose_finite_difference"], [0]))
    if result.ego_braking:
        ego_hits.append(("braking", ["ego_pose_finite_difference"], [0]))
    if result.ego_turning:
        ego_hits.append(("turning", ["ego_pose_finite_difference"], [0]))
    if result.ego_lane_changing:
        ego_hits.append(("lane_changing", ["map_lane_adjacency"], [0]))
    if any(state.lane_cells for state in context.ego):
        ego_hits.append(("on_road", ["map_lane_membership"], [0]))
    ego_hits.sort(key=lambda item: item[0])
    for predicate, sources, keyframes in ego_hits:
        aggregator.add(
            dimension="attribute",
            predicate=predicate,
            signature=f"{predicate}(ego)",
            node_types=["ego"],
            scene=context,
            slice_index=result.slice_index,
            sample_tokens=all_sample_tokens,
            instance_tokens=[],
            annotation_tokens=[],
            keyframes=keyframes,
            sources=sources,
            node_type="ego",
        )
        bump(predicate, "slice_actor", 1)

    # Hazard-class obligations.
    for hazard in hazard_classes:
        for state in result.actors.values():
            if not hazard.evaluate(state):
                continue
            evidence_keyframes = sorted(
                set(state.in_front_keyframes)
                | set(state.oncoming_keyframes)
                | set(state.same_lane_keyframes)
                | set(state.stationary_keyframes)
            )
            hazard_aggregator.add(
                dimension="hazard_class",
                predicate=hazard.name,
                signature=f"hazard({hazard.name})",
                node_types=[state.node_type],
                scene=context,
                slice_index=result.slice_index,
                sample_tokens=select(all_sample_tokens, evidence_keyframes) or state.sample_tokens,
                instance_tokens=[state.instance_token],
                annotation_tokens=state.annotation_tokens,
                keyframes=evidence_keyframes or state.present_keyframes,
                sources=[f"conjunction:{','.join(hazard.required_predicates)}"],
                node_type=state.node_type,
                nuscenes_category=state.nuscenes_category,
                grounding=hazard.grounding,
            )
            bump(hazard.name, "slice_actor", 1)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_markdown(inventory: dict[str, Any]) -> str:
    metadata = inventory["metadata"]
    coverage = inventory["coverage"]
    counts = inventory["counts"]
    lines: list[str] = []
    lines.append(f"# Oracle Inventory — `{metadata['split']}`")
    lines.append("")
    lines.append(
        "Deterministic STSG oracle inventory built from nuScenes annotation metadata by "
        "`build_oracle_inventory.py` (EXP-018). Every count below is measured on the mounted split; "
        "predicates that cannot be grounded from the available tables are reported as ungrounded."
    )
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Dataset root | `{metadata['dataset_root']}` |")
    lines.append(f"| Split | `{metadata['split']}` |")
    lines.append(f"| Scenes in split | {metadata['scene_count_total']} |")
    lines.append(f"| Scenes selected | {metadata['scene_count_selected']} |")
    lines.append(f"| Slice keyframes / stride | {metadata['parameters']['slice_keyframes']} / {metadata['parameters']['slice_stride_keyframes']} |")
    lines.append(f"| Generated at (UTC) | `{metadata['generated_at']}` |")
    lines.append(f"| Script SHA-256 | `{metadata['script_sha256']}` |")
    lines.append(f"| Determinism hash | `{metadata['determinism_hash']}` |")
    lines.append("")
    lines.append("## Coverage Summary")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("| --- | --- |")
    for key in (
        "scenes_total",
        "scenes_selected",
        "scenes_with_slices",
        "scenes_with_map_expansion",
        "scenes_with_at_least_one_fact",
        "slices_total",
        "slices_with_at_least_one_fact",
        "slices_with_relations",
        "slices_with_hazards",
        "actor_instances_in_slices",
        "actor_keyframe_observations",
        "ego_keyframe_lane_assignments",
        "actor_keyframe_lane_assignments",
        "trailing_keyframes_dropped",
    ):
        lines.append(f"| {key} | {coverage[key]} |")
    lines.append("")
    lines.append("## Dimension Counts")
    lines.append("")
    lines.append(
        "`defined` counts every predicate the builder can emit in this dimension; `supported` counts "
        "predicates with at least one obligation on this split; `obligations` counts distinct "
        "instantiated obligations (signature level) with support >= 1."
    )
    lines.append("")
    lines.append("| Dimension | Defined predicates | Supported predicates | Instantiated obligations |")
    lines.append("| --- | ---: | ---: | ---: |")
    for dimension in ("node", "attribute", "relation", "hazard_class"):
        entry = counts[dimension]
        lines.append(
            f"| {dimension} | {len(entry['defined_predicates'])} | {len(entry['supported_predicates'])} | {entry['obligation_count']} |"
        )
    total = counts["total"]
    lines.append(
        f"| **total** | {total['defined_predicates']} | {total['supported_predicates']} | {total['obligation_count']} |"
    )
    lines.append("")
    lines.append("| Dimension | Defined predicates |")
    lines.append("| --- | --- |")
    for dimension in ("node", "attribute", "relation", "hazard_class"):
        predicates = ", ".join(f"`{item}`" for item in counts[dimension]["defined_predicates"])
        lines.append(f"| {dimension} | {predicates} |")
    lines.append("")

    lines.append("## Grounding Status by Predicate")
    lines.append("")
    lines.append("| Predicate | Dimension | Grounding | Definition | Source | Supported slices |")
    lines.append("| --- | --- | --- | --- | --- | ---: |")
    for predicate in sorted(inventory["grounding"]["predicates"]):
        entry = inventory["grounding"]["predicates"][predicate]
        support = entry.get("supported_slices", 0)
        lines.append(
            "| `{predicate}` | {dimension} | {grounding} | {definition} | {source} | {support} |".format(
                predicate=predicate,
                dimension=entry["dimension"],
                grounding=entry["grounding"],
                definition=entry["definition"],
                source=entry["source"],
                support=support,
            )
        )
    lines.append("")

    lines.append("## Node Dimension")
    lines.append("")
    lines.append("| Obligation | Actor instances | Scenes | Slices | Categories |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for record in inventory["dimensions"]["node"]["obligations"]:
        lines.append(
            "| `{signature}` | {instances} | {scenes} | {slices} | {categories} |".format(
                signature=record["signature"],
                instances=record["support"]["actor_instances"],
                scenes=record["support"]["scenes"],
                slices=record["support"]["slices"],
                categories=", ".join(record.get("nuscenes_categories", [])) or "n/a",
            )
        )
    lines.append("")
    lines.append("## Attribute Dimension")
    lines.append("")
    lines.append("| Obligation | Grounding | Slices | Scenes | Actors | Keyframe occurrences |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for record in inventory["dimensions"]["attribute"]["obligations"]:
        lines.append(
            "| `{signature}` | {grounding} | {slices} | {scenes} | {actors} | {keyframes} |".format(
                signature=record["signature"],
                grounding=record["grounding"],
                slices=record["support"]["slices"],
                scenes=record["support"]["scenes"],
                actors=record["support"]["actor_instances"],
                keyframes=record["support"]["keyframe_occurrences"],
            )
        )
    lines.append("")
    lines.append("## Relation Dimension")
    lines.append("")
    lines.append("| Obligation | Grounding | Slices | Scenes | Actors | Keyframe occurrences |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for record in inventory["dimensions"]["relation"]["obligations"]:
        lines.append(
            "| `{signature}` | {grounding} | {slices} | {scenes} | {actors} | {keyframes} |".format(
                signature=record["signature"],
                grounding=record["grounding"],
                slices=record["support"]["slices"],
                scenes=record["support"]["scenes"],
                actors=record["support"]["actor_instances"],
                keyframes=record["support"]["keyframe_occurrences"],
            )
        )
    lines.append("")
    lines.append("## Hazard-Class Dimension")
    lines.append("")
    lines.append("| Hazard class | Grounding | DoTA ancestor | Definition | Slices | Scenes | Actors |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: |")
    hazard_index = {record["predicate"]: record for record in inventory["dimensions"]["hazard_class"]["obligations"]}
    for definition in inventory["hazard_classes"]["definitions"]:
        record = hazard_index.get(definition["name"])
        slices = record["support"]["slices"] if record else 0
        scenes = record["support"]["scenes"] if record else 0
        actors = record["support"]["actor_instances"] if record else 0
        lines.append(
            "| `{name}` | {grounding} | {ancestor} | {definition} | {slices} | {scenes} | {actors} |".format(
                name=definition["name"],
                grounding=definition["grounding"],
                ancestor=definition["dota_ancestor"] or "n/a",
                definition=definition["definition"],
                slices=slices,
                scenes=scenes,
                actors=actors,
            )
        )
    lines.append("")
    lines.append("## Per-Predicate Fact Counts")
    lines.append("")
    lines.append("| Predicate | Slice-level (actor) | Keyframe-level |")
    lines.append("| --- | ---: | ---: |")
    for predicate in sorted(inventory["predicate_counts"]):
        entry = inventory["predicate_counts"][predicate]
        lines.append(f"| `{predicate}` | {entry['slice_actor']} | {entry['keyframe']} |")
    lines.append("")

    lines.append("## Ungrounded Predicates")
    lines.append("")
    lines.append("| Predicate | Dimension | Reason |")
    lines.append("| --- | --- | --- |")
    for predicate, entry in sorted(inventory["grounding"]["ungrounded"].items()):
        lines.append(f"| `{predicate}` | {entry['dimension']} | {entry['reason']} |")
    lines.append("")
    lines.append("## Grounding Miss Reasons (evaluated but ungrounded)")
    lines.append("")
    if inventory["grounding"]["miss_reasons"]:
        lines.append("| Predicate | Reason | Occurrences |")
        lines.append("| --- | --- | ---: |")
        for predicate, reasons in sorted(inventory["grounding"]["miss_reasons"].items()):
            for reason, count in sorted(reasons.items()):
                lines.append(f"| `{predicate}` | `{reason}` | {count} |")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## DoTA Archetypes Not Emitted")
    lines.append("")
    lines.append("| Archetype | Reason |")
    lines.append("| --- | --- |")
    for name, reason in sorted(inventory["hazard_classes"]["dota_not_emitted"].items()):
        lines.append(f"| `{name}` | {reason} |")
    lines.append("")

    lines.append("## Category Alias Normalization")
    lines.append("")
    lines.append("| nuScenes category | Oracle node |")
    lines.append("| --- | --- |")
    for category, node in sorted(inventory["alias_table"].items()):
        collapsed = " (collapsed by observer alias)" if category in inventory["alias_collapses"] else ""
        lines.append(f"| `{category}` | `{node}`{collapsed} |")
    lines.append("")

    lines.append("## Examples (first witnesses)")
    lines.append("")
    for dimension in ("node", "attribute", "relation", "hazard_class"):
        lines.append(f"### {dimension}")
        lines.append("")
        records = inventory["dimensions"][dimension]["obligations"]
        if not records:
            lines.append("- none")
            lines.append("")
            continue
        for record in records:
            witness = record["witnesses"][0] if record["witnesses"] else None
            if witness is None:
                continue
            tokens = ", ".join(witness.get("sample_tokens", [])[:3])
            instances = ", ".join(witness.get("instance_tokens", [])[:2]) or "ego"
            lines.append(
                f"- `{record['signature']}` @ `{witness['scene']}` slice {witness['slice_index']} "
                f"(samples: {tokens}; actors: {instances})"
            )
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    for limitation in inventory["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def build_inventory(
    dataset: dict[str, Any],
    scene_contexts: list[SceneContext],
    params: argparse.Namespace,
    dataset_root: Path,
) -> dict[str, Any]:
    node_aggregator = ObligationAggregator(max_witnesses=params.max_witnesses)
    hazard_aggregator = ObligationAggregator(max_witnesses=params.max_witnesses)
    node_instances: dict[str, dict[str, Any]] = {}
    predicate_counts: dict[str, dict[str, int]] = {}
    ungrounded_reasons: dict[str, dict[str, int]] = {}
    hazard_classes = build_hazard_classes()

    scenes_with_slices = 0
    scenes_with_map = 0
    scenes_with_facts = 0
    slices_total = 0
    slices_with_facts = 0
    slices_with_relations = 0
    slices_with_hazards = 0
    actor_keyframe_observations = 0
    ego_lane_assignments = 0
    actor_lane_assignments = 0
    trailing_keyframes_dropped = 0

    for context in scene_contexts:
        sample_count = len(context.sample_tokens)
        windows: list[tuple[int, list[int]]] = []
        stride = max(int(params.slice_stride_keyframes), 1)
        width = max(int(params.slice_keyframes), 2)
        for start in range(0, sample_count - width + 1, stride):
            windows.append((start, list(range(start, start + width))))
        trailing_keyframes_dropped += max(sample_count - (windows[-1][0] + width), 0) if windows else sample_count
        if windows:
            scenes_with_slices += 1
        if context.map_index is not None:
            scenes_with_map += 1
        scene_has_fact = False
        scene_has_relation = False
        scene_has_hazard = False
        for slice_index, (start, keyframes) in enumerate(windows):
            slices_total += 1
            result = evaluate_slice(context, dataset, slice_index, start, keyframes, params, ungrounded_reasons)
            if result.actors:
                scene_has_fact = True
                slices_with_facts += 1
            actor_keyframe_observations += sum(len(state.present_keyframes) for state in result.actors.values())
            ego_lane_assignments += sum(1 for state in context.ego if state.lane_cells)
            actor_lane_assignments += sum(len(state.on_road_keyframes) for state in result.actors.values())
            accumulate_slice(
                context,
                result,
                node_aggregator,
                node_instances,
                hazard_aggregator,
                predicate_counts,
                hazard_classes,
            )
            if any(
                state.in_front_keyframes or state.same_lane_keyframes or state.adjacent_lane_keyframes or state.oncoming_keyframes
                for state in result.actors.values()
            ):
                scene_has_relation = True
                slices_with_relations += 1
            hazard_hit = any(
                hazard.evaluate(state) for hazard in hazard_classes for state in result.actors.values()
            )
            if hazard_hit:
                scene_has_hazard = True
                slices_with_hazards += 1
        if windows and scene_has_fact:
            scenes_with_facts += 1

    finalized = node_aggregator.finalize()
    node_obligations = [record for record in finalized if record["dimension"] == "node"]
    attribute_obligations = [record for record in finalized if record["dimension"] == "attribute"]
    relation_obligations = [record for record in finalized if record["dimension"] == "relation"]
    hazard_obligations = hazard_aggregator.finalize()

    grounding_predicates: dict[str, dict[str, Any]] = {}
    for predicate, definition in sorted(PREDICATE_DEFINITIONS.items()):
        entry = dict(definition)
        supported = 0
        for record in node_obligations + attribute_obligations + relation_obligations:
            if record["predicate"] == predicate:
                supported = max(supported, record["support"]["slices"])
        entry["supported_slices"] = supported
        grounding_predicates[predicate] = entry

    node_vocabulary = sorted(
        {record["predicate"].split(":", 1)[1] for record in node_obligations if record["predicate"].startswith("node:")}
    )
    attribute_vocabulary = sorted({record["predicate"] for record in attribute_obligations})
    relation_vocabulary = sorted({record["predicate"] for record in relation_obligations})
    hazard_vocabulary = sorted({record["predicate"] for record in hazard_obligations})
    hazard_definitions = build_hazard_classes()
    defined_attribute_predicates = sorted(
        predicate
        for predicate, definition in PREDICATE_DEFINITIONS.items()
        if definition["dimension"] == "attribute" and definition["grounding"] != "ungrounded"
    )
    defined_relation_predicates = sorted(
        predicate
        for predicate, definition in PREDICATE_DEFINITIONS.items()
        if definition["dimension"] == "relation" and definition["grounding"] != "ungrounded"
    )
    defined_hazard_predicates = sorted(hazard.name for hazard in hazard_definitions)

    determinism_payload = {
        "parameters": {
            "slice_keyframes": int(params.slice_keyframes),
            "slice_stride_keyframes": int(params.slice_stride_keyframes),
            "stationary_threshold_mps": float(params.stationary_threshold_mps),
            "braking_drop_mps": float(params.braking_drop_mps),
            "turning_yaw_deg": float(params.turning_yaw_deg),
            "oncoming_opposition_deg": float(params.oncoming_opposition_deg),
            "approaching_drop_m": float(params.approaching_drop_m),
            "crossing_corridor_half_width_m": float(params.crossing_corridor_half_width_m),
        },
        "coverage": {
            "scenes_selected": len(scene_contexts),
            "slices_total": slices_total,
        },
        "dimensions": {
            "node": node_obligations,
            "attribute": attribute_obligations,
            "relation": relation_obligations,
            "hazard_class": hazard_obligations,
        },
    }
    determinism_hash = hashlib.sha256(stable_json(determinism_payload).encode("utf-8")).hexdigest()

    limitations = [
        "The inventory is derived from annotation metadata only (poses, track velocities from finite differences, map expansion polygons). No camera or lidar content is read.",
        "Annotation `size` (box extents) is not used: every predicate is point-and-yaw based, matching the SemanticObserver and EXP-017 conventions. Box-overlap predicates are therefore out of scope for this inventory.",
        "nuScenes sample_annotation in this split carries no velocity field; all speeds are finite differences over adjacent keyframes of the same instance, which underestimates peak speed within a 0.5 s interval.",
        "Slice support counts are tiling-relative. The default tiling is overlapping 6-keyframe (3.0 s) windows with a 2-keyframe (1.0 s) stride; changing the stride changes support counts but not predicate definitions.",
        "The `crossing_path` corridor half-width (2.0 m) and the braking/turning/oncoming thresholds are parameter choices recorded in parameters.md, not dataset labels.",
        "`jaywalking` is a proxy (pedestrian on a lane polygon, off a ped_crossing polygon, outside a carpark_area polygon); right-of-way and signal state are not grounded.",
        "Sample selection is exhaustive over the split; the paper's stratified 156-scene selection is not applied here because v1.0-trainval is not mounted.",
        "Only scenes that contain at least one slice contribute to obligations; trailing keyframes shorter than one slice are dropped and counted.",
        "Overlapping windows credit one physical interaction in several consecutive slices, so slice support is tiling-relative; scene and actor-instance support are the stable counts.",
        "Hazard-class support inherits the corpus mix. The mini split is parking-lot heavy, so `other: ahead_or_waiting` includes parked vehicles occupying the ego lane ahead, which is a genuine in-lane obstruction but not necessarily an active waiting agent.",
        "The `crossing_path` route proxy is the ego's executed trajectory over the whole scene; a self-intersecting route (for example a loop through a car park) can credit a crossing against a geometric branch the ego is not currently driving.",
    ]

    inventory = {
        "metadata": {
            "experiment": "EXP-018-nuscenes-oracle-inventory",
            "dataset_root": str(dataset_root),
            "split": params.split,
            "scene_count_total": dataset["scene_count_total"],
            "scene_count_selected": dataset["scene_count_selected"],
            "generated_at": params.generated_at
            or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_of_file(Path(__file__).resolve()),
            "determinism_hash": determinism_hash,
            "dota_symbolic_available": _dota_symbolic is not None,
            "parameters": {
                "slice_keyframes": int(params.slice_keyframes),
                "slice_stride_keyframes": int(params.slice_stride_keyframes),
                "stationary_threshold_mps": float(params.stationary_threshold_mps),
                "braking_drop_mps": float(params.braking_drop_mps),
                "turning_yaw_deg": float(params.turning_yaw_deg),
                "oncoming_opposition_deg": float(params.oncoming_opposition_deg),
                "approaching_drop_m": float(params.approaching_drop_m),
                "crossing_corridor_half_width_m": float(params.crossing_corridor_half_width_m),
                "max_witnesses": int(params.max_witnesses),
                "scene_filter": params.scene,
                "scene_limit": params.scene_limit,
            },
        },
        "coverage": {
            "scenes_total": dataset["scene_count_total"],
            "scenes_selected": len(scene_contexts),
            "scenes_with_slices": scenes_with_slices,
            "scenes_with_map_expansion": scenes_with_map,
            "scenes_with_at_least_one_fact": scenes_with_facts,
            "slices_total": slices_total,
            "slices_with_at_least_one_fact": slices_with_facts,
            "slices_with_relations": slices_with_relations,
            "slices_with_hazards": slices_with_hazards,
            "actor_instances_in_slices": len(node_instances),
            "actor_keyframe_observations": actor_keyframe_observations,
            "ego_keyframe_lane_assignments": ego_lane_assignments,
            "actor_keyframe_lane_assignments": actor_lane_assignments,
            "trailing_keyframes_dropped": trailing_keyframes_dropped,
        },
        "predicate_counts": predicate_counts,
        "grounding": {
            "predicates": grounding_predicates,
            "ungrounded": UNGROUNDED_PREDICATES,
            "miss_reasons": ungrounded_reasons,
        },
        "alias_table": NODE_ALIAS_TABLE,
        "alias_collapses": NODE_ALIAS_COLLAPSES,
        "dimensions": {
            "node": {"vocabulary": node_vocabulary, "obligations": node_obligations},
            "attribute": {"vocabulary": attribute_vocabulary, "obligations": attribute_obligations},
            "relation": {"vocabulary": relation_vocabulary, "obligations": relation_obligations},
            "hazard_class": {
                "vocabulary": hazard_vocabulary,
                "obligations": hazard_obligations,
            },
        },
        "counts": {
            "node": {
                "defined_predicates": node_vocabulary,
                "supported_predicates": node_vocabulary,
                "obligation_count": len(node_obligations),
            },
            "attribute": {
                "defined_predicates": defined_attribute_predicates,
                "supported_predicates": attribute_vocabulary,
                "obligation_count": len(attribute_obligations),
            },
            "relation": {
                "defined_predicates": defined_relation_predicates,
                "supported_predicates": relation_vocabulary,
                "obligation_count": len(relation_obligations),
            },
            "hazard_class": {
                "defined_predicates": defined_hazard_predicates,
                "supported_predicates": hazard_vocabulary,
                "obligation_count": len(hazard_obligations),
            },
            "total": {
                "defined_predicates": len(node_vocabulary)
                + len(defined_attribute_predicates)
                + len(defined_relation_predicates)
                + len(defined_hazard_predicates),
                "supported_predicates": len(node_vocabulary)
                + len(attribute_vocabulary)
                + len(relation_vocabulary)
                + len(hazard_vocabulary),
                "obligation_count": len(node_obligations)
                + len(attribute_obligations)
                + len(relation_obligations)
                + len(hazard_obligations),
            },
        },
        "hazard_classes": {
            "definitions": [
                {
                    "name": hazard.name,
                    "grounding": hazard.grounding,
                    "definition": hazard.definition,
                    "dota_ancestor": hazard.dota_ancestor,
                    "dota_rule": dota_rule_for(hazard.dota_ancestor) if hazard.dota_ancestor else None,
                    "dota_formula": dota_formula_for(hazard.dota_ancestor) if hazard.dota_ancestor else None,
                    "required_predicates": hazard.required_predicates,
                }
                for hazard in hazard_definitions
            ],
            "dota_not_emitted": DOTA_CLASSES_NOT_EMITTED,
        },
        "limitations": limitations,
    }
    return inventory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def resolve_dataset_root(path: Path) -> Path:
    candidates = [path]
    if not path.is_absolute():
        candidates.append(WORKSPACE_ROOT / path)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise SystemExit(f"Dataset root not found: {path}")


def main(argv: list[str] | None = None) -> int:
    params = parse_args(argv)
    dataset_root = resolve_dataset_root(params.dataset_root)
    if params.scene_limit is None and params.scene is None:
        scene_limit = None
    else:
        scene_limit = params.scene_limit
    dataset, scene_contexts = load_dataset(dataset_root, params.split, params.scene, scene_limit)
    inventory = build_inventory(dataset, scene_contexts, params, dataset_root)

    params.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = params.output_dir / f"oracle_inventory_{params.split}.json"
    md_path = params.output_dir / f"oracle_inventory_{params.split}.md"
    json_path.write_text(stable_json(inventory), encoding="utf-8")
    md_path.write_text(render_markdown(inventory), encoding="utf-8")

    counts = inventory["counts"]
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        "obligations(supported/defined): "
        f"node={counts['node']['obligation_count']}/{len(counts['node']['supported_predicates'])}"
        f" of {len(counts['node']['defined_predicates'])} "
        f"attribute={counts['attribute']['obligation_count']}/{len(counts['attribute']['supported_predicates'])}"
        f" of {len(counts['attribute']['defined_predicates'])} "
        f"relation={counts['relation']['obligation_count']}/{len(counts['relation']['supported_predicates'])}"
        f" of {len(counts['relation']['defined_predicates'])} "
        f"hazard={counts['hazard_class']['obligation_count']}/{len(counts['hazard_class']['supported_predicates'])}"
        f" of {len(counts['hazard_class']['defined_predicates'])}"
    )
    print(
        f"scenes={inventory['coverage']['scenes_with_slices']}/{inventory['coverage']['scenes_selected']} "
        f"slices={inventory['coverage']['slices_total']} "
        f"slices_with_facts={inventory['coverage']['slices_with_at_least_one_fact']}"
    )
    print(f"determinism_hash={inventory['metadata']['determinism_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
