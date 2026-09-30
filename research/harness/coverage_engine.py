#!/usr/bin/env python
"""Offline semantic coverage engine for the MRES ADS testing pipeline.

WHAT THIS IS
------------
An implementation of the paper's coverage criterion

    Cov_k(T) = |{o in O_k : exists tick t, G_t |= o}| / |O_k|
    k in {V (node), A (attribute), E (relation), H (hazard class)}

evaluated offline against the EXP-018 nuScenes STSG oracle inventory
(``research/experiments/EXP-018-nuscenes-oracle-inventory/artifacts/``) and
against the simulator-side semantic streams
(``research/logs/semantic/*.jsonl``).

HONESTY NOTE -- READ BEFORE QUOTING ANY NUMBER
----------------------------------------------
The simulator streams do not persist predicate facts; they persist
per-tick world snapshots (documented below).  This engine re-derives a small,
declared set of simulator-side predicates from those snapshots through the
``CROSSWALK`` table.  The oracle defines 37 predicate names (mini) / 38
(trainval) including 9 hazard classes.  The crosswalk supplies a defended
witness path for 22 of the 29 non-hazard predicate names (7 node types,
8 attribute predicates, 7 relation predicates) and for all 9 of the 9 hazard
classes.  The remaining unmapped predicates are ``parked``, ``standing``,
``sitting_lying_down``, ``with_rider``, ``without_rider``, ``stopped`` and
``animal`` (plus the animal instantiations of ``braking`` / ``moving`` /
``on_road`` / ``stationary``, and ``waiting(vehicle)``), all of which have no
CARLA-side signal in the stream.  On the EXP-018 mini inventory the mapped
subset is 70 of 78 obligations; on trainval it is 86 of 103.

Schema version 2 of the stream (see ``research/harness/lane_index.py``) adds
the per-actor lane membership and compact track history that make
``adjacent_lane``, ``lane_changing(vehicle)`` and swept-path ``crossing_path``
recomputable offline.  Version 1 rows still load: every v2 field is read
defensively and every v1 derivation keeps its documented fallback.

Therefore every report from this engine carries:

* ``Cov_V`` / ``Cov_A`` / ``Cov_E`` / ``Cov_H`` -- ratios over the *mapped*
  obligation subset only, and
* ``coverage_of_mapped_subset`` and ``full_vocabulary_coverage`` side by side,
  plus ``mapped_obligation_count`` / ``total_obligation_count``, plus the full
  per-obligation unmapped list with reasons, plus a ``warning`` string.

Do not present ``Cov_*`` from this engine as full-vocabulary coverage.

REAL SEMANTIC-STREAM SCHEMA (``research/logs/semantic/*.jsonl``, observed 2026-09)
---------------------------------------------------------------------------------
One JSON object per line, one line per captured tick (stream_every_ticks >= 1).
Schema version 1 rows carry the keys below; schema version 2 rows carry every
v1 key unchanged plus the additive lane/track keys marked (v2):

    {
      "schema_version": 2,                      # (v2) absent in v1 rows
      "tick": 0,
      "trigger_reason": null | "persistent-brake" | "heading-error" | ...,
      "scenario_id": "town01_clear_short",
      "town": "Town01",
      "harness_state": {
        "threshold_adversary_kind", "threshold_adversary_active",
        "threshold_adversary_triggered", "threshold_trigger_distance_m"
      },
      "ego": {
        "id", "location" {x,y,z}, "rotation" {pitch,yaw,roll}, "velocity_mps",
        "waypoint": {"road_id","lane_id","s","is_junction","location","rotation"},
        "lane": {"road_id","lane_id","s",...},  # (v2)
        "waypoint_history": [{tick,x,y,yaw_deg,velocity_mps,road_id,lane_id}]
      },
      "telemetry": {
        "speed_mps", "speed_kph", "distance_to_goal_m",
        "control": {throttle, steer, brake, hand_brake, reverse, gear, ...},
        "route_progress": {...}
      },
      "route_context": {...},
      "nearby_actors": [{
        "id", "type_id", "distance_to_ego_m",
        "location", "rotation", "velocity_mps",
        "same_road_as_ego", "same_lane_as_ego", "is_in_front_of_ego",
        "lane": {...},                          # (v2) direct lane membership
        "lane_relation": "same"|"left"|"right"|null,  # (v2)
        "adjacent_lane_as_ego": bool,           # (v2) direct
        "oncoming_as_ego": bool,                # (v2) derived (heading + lane)
        "waypoint_history": [{tick,x,y,yaw,...}]  # (v2) compact track
      }]
    }

Older files omit ``harness_state`` and all (v2) keys; all fields are normalized
defensively by ``load_semantic_trace`` into ``NormalizedTick`` records with ego
speed/brake/steer/yaw/waypoint/lane/track, typed actor records (CARLA
``type_id`` plus the mapped oracle node type), actor geometry/motion flags and
the actor's persisted lane membership and track samples.

Predicates are *not* stored in the stream.  They are emitted in-memory by
``research/harness/observers/semantic.py`` (SemanticObserver) and only the
6-predicate intersection ``{jaywalking, waiting, on_road, in_front_of,
crossing_path, colliding}`` ever reached a logged ``semantic_covered_predicates``
field.  This engine therefore re-derives facts from the raw fields; every fact
keeps a ``source`` string naming the exact field and rule used.

DEPENDENCIES: stdlib plus ``research/harness/lane_index.py`` (itself stdlib
only; shared with the live observer so online and offline geometry agree).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from research.harness.lane_index import (
    CrossingPathConfig,
    LaneRef,
    TrackSample,
    angle_difference_deg,
    classify_lane_relation,
    detect_crossing_from_tracks,
    detect_lane_change,
)

ENGINE_NAME = "coverage_engine"
SCHEMA_VERSION = "2026-09-21"

AXIS_BY_DIMENSION = {
    "node": "V",
    "attribute": "A",
    "relation": "E",
    "hazard_class": "H",
}
DIMENSION_BY_AXIS = {axis: dim for dim, axis in AXIS_BY_DIMENSION.items()}
AXIS_LABELS = {
    "V": "node",
    "A": "attribute",
    "E": "relation",
    "H": "hazard class",
}

GROUNDING_DIRECT = "direct"
GROUNDING_DERIVED = "derived"
GROUNDING_PROXY = "proxy"
GROUNDING_UNMAPPED = "unmapped"

PHYSICAL_NODES = frozenset(
    {"barrier", "bicycle_rack", "debris", "pedestrian", "pushable_pullable", "trafficcone", "vehicle"}
)
AGENT_NODES = frozenset({"pedestrian", "vehicle"})
EGO_NODES = frozenset({"ego"})


# ---------------------------------------------------------------------------
# Declarative vocabulary crosswalk (simulator-side predicate -> oracle predicate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimPredicateMapping:
    sim_predicate: str
    oracle_predicate: str | None
    grounding: str
    nodes: frozenset[str]
    evidence: str
    disallowed_nodes_note: str | None = None
    reason: str | None = None


CROSSWALK: tuple[SimPredicateMapping, ...] = (
    SimPredicateMapping(
        sim_predicate="moving",
        oracle_predicate="moving",
        grounding=GROUNDING_DERIVED,
        nodes=PHYSICAL_NODES | EGO_NODES,
        evidence=(
            "actor speed = nearby_actors[].velocity_mps; ego speed = telemetry.speed_mps "
            "(fallback ego.velocity_mps); credited when speed >= speed_epsilon_mps"
        ),
    ),
    SimPredicateMapping(
        sim_predicate="stationary",
        oracle_predicate="stationary",
        grounding=GROUNDING_DERIVED,
        nodes=PHYSICAL_NODES | EGO_NODES,
        evidence=(
            "same speed source as moving; credited when speed < speed_epsilon_mps. "
            "Oracle uses finite-difference speed over keyframes, simulator uses instantaneous speed"
        ),
    ),
    SimPredicateMapping(
        sim_predicate="braking",
        oracle_predicate="braking",
        grounding=GROUNDING_DERIVED,
        nodes=PHYSICAL_NODES | EGO_NODES,
        evidence=(
            "ego: telemetry.control.brake >= braking_brake_threshold OR ego speed drop "
            ">= braking_speed_drop_mps between adjacent ticks; actors: velocity_mps drop between "
            "adjacent ticks, for every node type the oracle instantiates braking() on"
        ),
        disallowed_nodes_note=(
            "oracle has no braking() obligation for this node type (e.g. animal); simulator "
            "deceleration evidence is derived for all observed physical actors and the ego"
        ),
    ),
    SimPredicateMapping(
        sim_predicate="turning",
        oracle_predicate="turning",
        grounding=GROUNDING_DERIVED,
        nodes=frozenset({"ego", "vehicle"}),
        evidence=(
            "ego: |telemetry.control.steer| >= turning_steer_threshold OR |ego yaw delta| between "
            "adjacent ticks >= turning_yaw_rate_deg_per_tick; vehicle: yaw delta only"
        ),
        disallowed_nodes_note=(
            "simulator heading is only tracked for ego and vehicle actors; oracle turning() "
            "obligations exist only for ego and vehicle"
        ),
    ),
    SimPredicateMapping(
        sim_predicate="lane_changing",
        oracle_predicate="lane_changing",
        grounding=GROUNDING_DERIVED,
        nodes=frozenset({"ego", "vehicle"}),
        evidence=(
            "ego: ego.waypoint.road_id or lane_id changes between adjacent ticks; vehicle: "
            "nearby_actors[].lane (per-actor lane index, schema v2) changes to a laterally "
            "adjacent lane between adjacent ticks. v1 streams can only witness the ego change"
        ),
        disallowed_nodes_note=(
            "oracle lane_changing() obligations exist only for ego and vehicle; v1 streams had "
            "no per-actor lane id (schema v2 persists it), so only the ego witness is derivable"
        ),
    ),
    SimPredicateMapping(
        sim_predicate="on_road",
        oracle_predicate="on_road",
        grounding=GROUNDING_DERIVED,
        nodes=PHYSICAL_NODES | EGO_NODES,
        evidence=(
            "actor: nearby_actors[].same_road_as_ego (CARLA get_waypoint(project_to_road=True) "
            "road_id equality); ego: ego.waypoint present in the tick"
        ),
    ),
    SimPredicateMapping(
        sim_predicate="in_front_of",
        oracle_predicate="in_front_of",
        grounding=GROUNDING_DIRECT,
        nodes=PHYSICAL_NODES,
        evidence=(
            "nearby_actors[].is_in_front_of_ego, computed by SemanticObserver at capture time as "
            "dot(target - ego, ego_forward) > 0"
        ),
    ),
    SimPredicateMapping(
        sim_predicate="same_lane",
        oracle_predicate="same_lane",
        grounding=GROUNDING_DIRECT,
        nodes=PHYSICAL_NODES,
        evidence=(
            "nearby_actors[].same_lane_as_ego, computed at capture time from CARLA waypoint "
            "road_id + lane_id equality; schema v2 also persists nearby_actors[].lane "
            "(road_id/lane_id) so the engine recomputes the equality offline"
        ),
    ),
    SimPredicateMapping(
        sim_predicate="adjacent_lane",
        oracle_predicate="adjacent_lane",
        grounding=GROUNDING_DIRECT,
        nodes=PHYSICAL_NODES,
        evidence=(
            "nearby_actors[].adjacent_lane_as_ego / lane_relation, computed at capture time by "
            "LaneIndex from CARLA get_left_lane/get_right_lane; recomputable offline from "
            "nearby_actors[].lane road_id + lane_id lateral adjacency (schema v2)"
        ),
    ),
    SimPredicateMapping(
        sim_predicate="approaching",
        oracle_predicate="approaching",
        grounding=GROUNDING_DERIVED,
        nodes=AGENT_NODES,
        evidence=(
            "distance_to_ego_m decreases by >= approaching_delta_m between adjacent ticks for the "
            "same actor id crosswalked to this node type"
        ),
        disallowed_nodes_note=(
            "oracle restricts approaching() to agent nodes (pedestrian, vehicle); static props are "
            "not sampled reliably enough for a closing-distance claim"
        ),
    ),
    SimPredicateMapping(
        sim_predicate="oncoming",
        oracle_predicate="oncoming",
        grounding=GROUNDING_DERIVED,
        nodes=frozenset({"vehicle"}),
        evidence=(
            "schema v2: nearby_actors[].oncoming_as_ego (same road + known lane membership + "
            "|yaw delta| >= oncoming_yaw_delta_deg + in front + moving) or offline recomputation "
            "from nearby_actors[].lane plus ego.waypoint; v1 fallback: vehicle same_road_as_ego "
            "AND |wrap(actor yaw - ego yaw)| >= oncoming_yaw_delta_deg"
        ),
        disallowed_nodes_note="oracle oncoming() is a vehicle-only relation",
    ),
    SimPredicateMapping(
        sim_predicate="crossing_path",
        oracle_predicate="crossing_path",
        grounding=GROUNDING_DERIVED,
        nodes=AGENT_NODES,
        evidence=(
            "schema v2: swept-path intersection between nearby_actors[].waypoint_history "
            "(observed track + constant-velocity projection) and the ego forward path, shape-"
            "matched to the oracle (opposite sides of the ego polyline, closest approach <= "
            "crossing_corridor_half_width_m, actor displacement >= 0.5 m, vehicle heading delta "
            "in [30,150] deg). v1 fallback PROXY: in front and distance <= crossing_distance_m "
            "(mirrors SemanticObserver's legacy 12.0 m heuristic)"
        ),
        disallowed_nodes_note="oracle crossing_path() is an agent-only relation",
    ),
    SimPredicateMapping(
        sim_predicate="obstructing",
        oracle_predicate="obstructing",
        grounding=GROUNDING_DERIVED,
        nodes=PHYSICAL_NODES,
        evidence=(
            "nearby_actors[] is_in_front_of_ego AND same_lane_as_ego AND "
            "distance_to_ego_m <= obstructing_distance_m"
        ),
    ),
    SimPredicateMapping(
        sim_predicate="jaywalking",
        oracle_predicate="jaywalking",
        grounding=GROUNDING_PROXY,
        nodes=frozenset({"pedestrian"}),
        evidence=(
            "pedestrian actor with same_road_as_ego == true. PROXY: the stream cannot express "
            "crosswalk legality or walker displacement onto the drivable polygon, which is what "
            "the oracle's jaywalking proxy is built from"
        ),
        disallowed_nodes_note="oracle jaywalking() is a pedestrian-only attribute",
    ),
    SimPredicateMapping(
        sim_predicate="waiting",
        oracle_predicate="waiting",
        grounding=GROUNDING_PROXY,
        nodes=frozenset({"pedestrian"}),
        evidence=(
            "pedestrian same_road_as_ego AND is_in_front_of_ego AND speed < waiting_speed_mps. "
            "PROXY: the observer emits waiting(pedestrian) from scripted threshold-adversary "
            "activation state, which is not persisted per actor"
        ),
        disallowed_nodes_note=(
            "simulator-side waiting is only derived for pedestrians (SemanticObserver emits "
            "waiting(pedestrian) only; waiting(vehicle) has no witness path)"
        ),
    ),
    SimPredicateMapping(
        sim_predicate="colliding",
        oracle_predicate=None,
        grounding=GROUNDING_UNMAPPED,
        nodes=frozenset(),
        evidence="",
        reason=(
            "collision events are produced by a separate collision oracle and never written into "
            "the semantic stream; the nuScenes oracle also declares colliding ungrounded "
            "(no contact event in the annotation tables)"
        ),
    ),
    SimPredicateMapping(
        sim_predicate="occluded",
        oracle_predicate=None,
        grounding=GROUNDING_UNMAPPED,
        nodes=frozenset(),
        evidence="",
        reason=(
            "occluder identity lives in harness context (context['occluder_vehicle']) and is not "
            "persisted in the stream; the nuScenes oracle also declares occluded ungrounded "
            "(no line-of-sight relation in the annotation tables)"
        ),
    ),
    SimPredicateMapping(
        sim_predicate="speeding",
        oracle_predicate=None,
        grounding=GROUNDING_UNMAPPED,
        nodes=frozenset(),
        evidence="",
        reason=(
            "the stream has speed and traffic.speed_limit actors but no speed-limit map "
            "association; the oracle declares speeding ungrounded"
        ),
    ),
    SimPredicateMapping(
        sim_predicate="out_of_control",
        oracle_predicate=None,
        grounding=GROUNDING_UNMAPPED,
        nodes=frozenset(),
        evidence="",
        reason=(
            "no steering/throttle/stability signal is recorded as an anomaly label; the oracle "
            "declares out_of_control ungrounded"
        ),
    ),
)


ACTOR_TYPE_CROSSWALK: tuple[tuple[str, str | None, str | None], ...] = (
    ("walker.pedestrian.*", "pedestrian", None),
    (
        "walker.*",
        None,
        "non-pedestrian CARLA walker classes are not part of the oracle node vocabulary",
    ),
    (
        "vehicle.bicycle*",
        "vehicle",
        "oracle alias_collapses intentionally collapses vehicle.bicycle into vehicle",
    ),
    (
        "vehicle.motorcycle*",
        "vehicle",
        "oracle alias_collapses intentionally collapses vehicle.motorcycle into vehicle",
    ),
    ("vehicle.*", "vehicle", None),
    ("static.prop.streetbarrier*", "barrier", None),
    ("static.prop.barrier*", "barrier", None),
    ("static.prop.trafficcone*", "trafficcone", None),
    ("static.prop.cone*", "trafficcone", None),
    ("static.prop.bicyclerack*", "bicycle_rack", None),
    ("static.prop.debris*", "debris", None),
    ("static.prop.pushable*", "pushable_pullable", None),
    (
        "static.prop.*",
        None,
        "CARLA static props (wall/fence/pole/sidewalk) have no nuScenes annotation category and "
        "no oracle node type",
    ),
    (
        "traffic.*",
        None,
        "traffic infrastructure (lights, signs) is not part of the oracle node vocabulary",
    ),
    ("*", None, "no actor-type crosswalk entry matches this CARLA type_id"),
)


ORACLE_PREDICATE_NOTES: dict[tuple[str, str], str] = {
    ("attribute", "parked"): (
        "the stream carries no parking state; a stationary vehicle cannot be separated from a "
        "parked vehicle without curb/lane context, so it is left unmapped rather than over-credited"
    ),
    ("attribute", "standing"): (
        "no per-pedestrian posture signal is persisted (CARLA walker animation is not in the stream)"
    ),
    ("attribute", "sitting_lying_down"): (
        "no per-pedestrian posture signal is persisted (CARLA walker animation is not in the stream)"
    ),
    ("attribute", "with_rider"): "no rider/passenger relationship is persisted for any actor",
    ("attribute", "without_rider"): "no rider/passenger relationship is persisted for any actor",
    ("attribute", "stopped"): (
        "stopped vs parked vs holding cannot be disambiguated from speed plus brake alone; left "
        "unmapped rather than over-credit"
    ),
    ("attribute", "speeding"): (
        "no speed-limit map association exists in the stream; oracle declares speeding ungrounded"
    ),
    ("attribute", "out_of_control"): (
        "no control-state signal exists in the stream; oracle declares out_of_control ungrounded"
    ),
    ("attribute", "occluded"): (
        "occluder identity is harness context, not stream data; oracle declares occluded ungrounded"
    ),
    ("relation", "colliding"): (
        "collision events live in the collision oracle, not the semantic stream; oracle declares "
        "colliding ungrounded"
    ),
    ("node", "animal"): (
        "no animal actor appears in the observed stream corpus and no CARLA type_id crosswalk "
        "entry can be defended"
    ),
}


@dataclass(frozen=True)
class MappingStatus:
    mapped: bool
    sim_predicate: str | None
    grounding: str | None
    evidence: str
    reason: str | None


def mapping_for(predicate: str, node_type: str) -> MappingStatus:
    """Resolve a (simulator predicate, oracle node type) pair through CROSSWALK."""
    candidates = [m for m in CROSSWALK if m.sim_predicate == predicate]
    if not candidates:
        return MappingStatus(
            mapped=False,
            sim_predicate=None,
            grounding=None,
            evidence="",
            reason=f"no crosswalk entry exists for simulator predicate '{predicate}'",
        )
    for mapping in candidates:
        if node_type in mapping.nodes:
            return MappingStatus(
                mapped=True,
                sim_predicate=mapping.sim_predicate,
                grounding=mapping.grounding,
                evidence=mapping.evidence,
                reason=None,
            )
    mapping = candidates[0]
    return MappingStatus(
        mapped=False,
        sim_predicate=None,
        grounding=None,
        evidence="",
        reason=mapping.disallowed_nodes_note
        or mapping.reason
        or f"crosswalk entry for '{predicate}' does not cover node type '{node_type}'",
    )


def node_type_status(node_type: str) -> MappingStatus:
    """Resolve an oracle node type by asking which CARLA type patterns map to it."""
    patterns = [
        pattern for pattern, oracle_node, _ in ACTOR_TYPE_CROSSWALK if oracle_node == node_type
    ]
    if patterns:
        pattern = patterns[0]
        return MappingStatus(
            mapped=True,
            sim_predicate=f"actor.type_id~{pattern}",
            grounding=GROUNDING_DIRECT,
            evidence=(
                f"nearby_actors[].type_id matching '{pattern}' is crosswalked to oracle node "
                f"type '{node_type}'"
            ),
            reason=None,
        )
    note = ORACLE_PREDICATE_NOTES.get(("node", node_type))
    return MappingStatus(
        mapped=False,
        sim_predicate=None,
        grounding=None,
        evidence="",
        reason=note or f"no actor type crosswalk pattern maps to node type '{node_type}'",
    )


# ---------------------------------------------------------------------------
# Oracle inventory loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleObligation:
    dimension: str
    signature: str
    predicate: str
    grounding: str
    node_types: tuple[str, ...]
    required_predicates: tuple[str, ...] = ()

    @property
    def axis(self) -> str:
        return AXIS_BY_DIMENSION[self.dimension]


@dataclass
class Oracle:
    path: str
    split: str
    metadata: dict[str, Any]
    obligations: list[OracleObligation]
    predicates_by_axis: dict[str, tuple[str, ...]]
    hazard_definitions: dict[str, dict[str, Any]]
    ungrounded: dict[str, dict[str, Any]]
    counts: dict[str, Any]

    def obligations_for_axis(self, axis: str) -> list[OracleObligation]:
        dimension = DIMENSION_BY_AXIS[axis]
        return [o for o in self.obligations if o.dimension == dimension]


def load_oracle(path: str | Path) -> Oracle:
    """Load an EXP-018 oracle inventory JSON into predicate sets per dimension."""
    oracle_path = Path(path)
    raw = json.loads(oracle_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "dimensions" not in raw:
        raise ValueError(
            f"Oracle inventory at {oracle_path} does not look like an EXP-018 payload: "
            "'dimensions' is missing"
        )

    definitions = {
        str(item.get("name")): item
        for item in (raw.get("hazard_classes") or {}).get("definitions") or []
        if isinstance(item, dict) and item.get("name")
    }

    obligations: list[OracleObligation] = []
    predicates: dict[str, tuple[str, ...]] = {}
    for dimension, section in raw["dimensions"].items():
        if not isinstance(section, dict):
            continue
        vocabulary = tuple(str(name) for name in section.get("vocabulary") or ())
        predicates[dimension] = vocabulary
        for item in section.get("obligations") or []:
            if not isinstance(item, dict):
                continue
            predicate = str(item.get("predicate") or "")
            signature = str(item.get("signature") or predicate)
            node_types = tuple(str(node) for node in item.get("node_types") or ())
            if dimension == "hazard_class":
                class_name = predicate or signature.removeprefix("hazard(").removesuffix(")")
                definition = definitions.get(class_name) or {}
                required = tuple(str(name) for name in definition.get("required_predicates") or ())
                obligations.append(
                    OracleObligation(
                        dimension=dimension,
                        signature=signature,
                        predicate=class_name,
                        grounding=str(item.get("grounding") or definition.get("grounding") or ""),
                        node_types=node_types,
                        required_predicates=required,
                    )
                )
            else:
                obligations.append(
                    OracleObligation(
                        dimension=dimension,
                        signature=signature,
                        predicate=predicate,
                        grounding=str(item.get("grounding") or ""),
                        node_types=node_types,
                    )
                )

    ungrounded = {
        str(name): value
        for name, value in ((raw.get("grounding") or {}).get("ungrounded") or {}).items()
    }
    return Oracle(
        path=str(oracle_path),
        split=str((raw.get("metadata") or {}).get("split") or "unknown"),
        metadata=dict(raw.get("metadata") or {}),
        obligations=obligations,
        predicates_by_axis={
            axis: predicates.get(dimension, ()) for axis, dimension in DIMENSION_BY_AXIS.items()
        },
        hazard_definitions=definitions,
        ungrounded=ungrounded,
        counts=dict(raw.get("counts") or {}),
    )


# ---------------------------------------------------------------------------
# Semantic stream loading and normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedActor:
    actor_id: str
    type_id: str
    sim_alias: str
    oracle_node: str | None
    distance_m: float
    yaw_deg: float
    speed_mps: float
    same_road: bool
    same_lane: bool
    in_front: bool
    lane: LaneRef | None = None
    lane_relation: str | None = None
    adjacent_lane: bool = False
    oncoming_flag: bool = False
    track: tuple[TrackSample, ...] = ()


@dataclass(frozen=True)
class NormalizedTick:
    tick: int
    scenario_id: str
    town: str
    ego_id: str
    ego_x: float
    ego_y: float
    ego_yaw_deg: float
    ego_speed_mps: float
    ego_road_id: int | None
    ego_lane_id: int | None
    brake: float
    throttle: float
    steer: float
    gear: int | None
    trigger_reason: str | None
    threshold_kind: str | None
    threshold_active: bool
    actors: tuple[NormalizedActor, ...]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)
    ego_lane: LaneRef | None = None
    ego_track: tuple[TrackSample, ...] = ()


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def resolve_actor_type(type_id: str) -> tuple[str | None, str | None, str]:
    """Crosswalk a CARLA type_id to an oracle node type; also return alias and reason."""
    sim_alias = type_id.split(".")[-1] if type_id else "unknown"
    for pattern, oracle_node, reason in ACTOR_TYPE_CROSSWALK:
        if fnmatch.fnmatchcase(type_id or "", pattern):
            if oracle_node is not None:
                return oracle_node, None, sim_alias
            return None, reason, sim_alias
    return None, "no actor-type crosswalk entry matches this CARLA type_id", sim_alias


def _normalize_track(item: dict[str, Any]) -> tuple[TrackSample, ...]:
    raw_track = item.get("waypoint_history")
    if raw_track is None:
        raw_track = item.get("track")
    if not isinstance(raw_track, list):
        return ()
    samples = []
    for sample in raw_track:
        parsed = TrackSample.from_dict(sample)
        if parsed is not None:
            samples.append(parsed)
    samples.sort(key=lambda sample: sample.tick)
    return tuple(samples)


def normalize_tick(raw: dict[str, Any], index: int = 0) -> NormalizedTick:
    """Normalize one raw JSONL row into a NormalizedTick (schema in module docstring)."""
    ego = raw.get("ego") or {}
    ego_rotation = ego.get("rotation") or {}
    waypoint = ego.get("waypoint") or {}
    telemetry = raw.get("telemetry") or {}
    control = telemetry.get("control") or {}
    harness = raw.get("harness_state") or {}

    speed = _as_float(telemetry.get("speed_mps"))
    if speed is None:
        speed = _as_float(ego.get("velocity_mps"), 0.0) or 0.0

    ego_lane = LaneRef.from_dict(ego.get("lane")) or LaneRef.from_dict(waypoint)
    ego_track = _normalize_track(ego)

    actors: list[NormalizedActor] = []
    for item in raw.get("nearby_actors") or []:
        if not isinstance(item, dict):
            continue
        type_id = str(item.get("type_id") or "")
        oracle_node, _, sim_alias = resolve_actor_type(type_id)
        rotation = item.get("rotation") or {}
        distance = _as_float(item.get("distance_to_ego_m"))
        lane_relation = item.get("lane_relation")
        actors.append(
            NormalizedActor(
                actor_id=str(item.get("id")),
                type_id=type_id,
                sim_alias=sim_alias,
                oracle_node=oracle_node,
                distance_m=float("inf") if distance is None else distance,
                yaw_deg=_as_float(rotation.get("yaw"), 0.0) or 0.0,
                speed_mps=_as_float(item.get("velocity_mps"), 0.0) or 0.0,
                same_road=_as_bool(item.get("same_road_as_ego")),
                same_lane=_as_bool(item.get("same_lane_as_ego")),
                in_front=_as_bool(item.get("is_in_front_of_ego")),
                lane=LaneRef.from_dict(item.get("lane")),
                lane_relation=str(lane_relation) if lane_relation in ("same", "left", "right") else None,
                adjacent_lane=_as_bool(item.get("adjacent_lane_as_ego")),
                oncoming_flag=_as_bool(item.get("oncoming_as_ego")),
                track=_normalize_track(item),
            )
        )

    return NormalizedTick(
        tick=int(raw.get("tick", index)),
        scenario_id=str(raw.get("scenario_id") or ""),
        town=str(raw.get("town") or ""),
        ego_id=str(ego.get("id") or "ego"),
        ego_x=_as_float((ego.get("location") or {}).get("x"), 0.0) or 0.0,
        ego_y=_as_float((ego.get("location") or {}).get("y"), 0.0) or 0.0,
        ego_yaw_deg=_as_float(ego_rotation.get("yaw"), 0.0) or 0.0,
        ego_speed_mps=speed,
        ego_road_id=_as_int(waypoint.get("road_id")),
        ego_lane_id=_as_int(waypoint.get("lane_id")),
        brake=_as_float(control.get("brake"), 0.0) or 0.0,
        throttle=_as_float(control.get("throttle"), 0.0) or 0.0,
        steer=_as_float(control.get("steer"), 0.0) or 0.0,
        gear=_as_int(control.get("gear")),
        trigger_reason=raw.get("trigger_reason"),
        threshold_kind=harness.get("threshold_adversary_kind"),
        threshold_active=_as_bool(harness.get("threshold_adversary_active")),
        actors=tuple(actors),
        raw=raw,
        ego_lane=ego_lane,
        ego_track=ego_track,
    )


def load_semantic_trace(jsonl_path: str | Path) -> list[NormalizedTick]:
    """Load a simulator semantic stream JSONL into normalized tick records."""
    path = Path(jsonl_path)
    ticks: list[NormalizedTick] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object per row")
            ticks.append(normalize_tick(raw, index=len(ticks)))
    return ticks


def load_ego_route(jsonl_path: str | Path) -> list[dict[str, float]]:
    """Read the planned ego route persisted on the first stream record.

    The observer writes the harness context's ``route_waypoints`` as
    ``ego_route`` on the first stream frame when a route is available; older
    streams omit the key entirely and yield ``[]`` so engine behaviour is
    unchanged.  Only the first non-blank record is inspected by design.
    """
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object per row")
            route = raw.get("ego_route")
            if not isinstance(route, list):
                return []
            points: list[dict[str, float]] = []
            for item in route:
                if not isinstance(item, dict):
                    continue
                x = _as_float(item.get("x"))
                y = _as_float(item.get("y"))
                if x is None or y is None:
                    continue
                points.append({"x": x, "y": y})
            return points
    return []


# ---------------------------------------------------------------------------
# Fact derivation
# ---------------------------------------------------------------------------


@dataclass
class DerivationConfig:
    speed_epsilon_mps: float = 0.1
    braking_brake_threshold: float = 0.5
    braking_speed_drop_mps: float = 0.8
    turning_steer_threshold: float = 0.2
    turning_yaw_rate_deg_per_tick: float = 3.0
    approaching_delta_m: float = 0.05
    crossing_distance_m: float = 12.0
    obstructing_distance_m: float = 18.0
    waiting_speed_mps: float = 0.3
    oncoming_yaw_delta_deg: float = 135.0
    hazard_window_ticks: int = 6
    crossing_corridor_half_width_m: float = 2.0
    crossing_min_actor_displacement_m: float = 0.5
    crossing_vehicle_heading_range_deg: tuple[float, float] = (30.0, 150.0)
    crossing_ego_horizon_s: float = 2.0
    crossing_actor_horizon_s: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["crossing_vehicle_heading_range_deg"] = list(self.crossing_vehicle_heading_range_deg)
        return payload

    def crossing_config(self) -> CrossingPathConfig:
        return CrossingPathConfig(
            corridor_half_width_m=self.crossing_corridor_half_width_m,
            min_actor_displacement_m=self.crossing_min_actor_displacement_m,
            vehicle_heading_range_deg=self.crossing_vehicle_heading_range_deg,
            ego_horizon_s=self.crossing_ego_horizon_s,
            actor_horizon_s=self.crossing_actor_horizon_s,
        )


@dataclass(frozen=True)
class Fact:
    predicate: str
    subject: str
    obj: str | None
    tick: int
    actor_key: str
    source: str

    @property
    def signature(self) -> str:
        if self.obj:
            return f"{self.predicate}({self.subject},{self.obj})"
        return f"{self.predicate}({self.subject})"


@dataclass
class DerivedTrace:
    ticks: list[NormalizedTick]
    facts: list[Fact] = field(default_factory=list)
    node_witnesses: dict[str, Fact] = field(default_factory=dict)
    node_instance_counts: dict[str, int] = field(default_factory=dict)
    by_signature: dict[str, list[Fact]] = field(default_factory=dict)
    actor_facts: dict[str, dict[str, list[Fact]]] = field(default_factory=dict)
    unmapped_aliases: dict[str, int] = field(default_factory=dict)

    def witness(self, predicate: str, subject: str, obj: str | None = None) -> Fact | None:
        signature = f"{predicate}({subject},{obj})" if obj else f"{predicate}({subject})"
        candidates = self.by_signature.get(signature) or []
        if not candidates:
            return None
        return min(candidates, key=lambda fact: fact.tick)


def _wrap_angle_deg(delta: float) -> float:
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def derive_trace(
    ticks: Sequence[NormalizedTick],
    config: DerivationConfig | None = None,
    *,
    ego_route: Sequence[Any] | None = None,
) -> DerivedTrace:
    """Re-derive simulator-side predicate facts from normalized ticks.

    Every fact records the tick and the source rule so coverage credit is
    auditable, and actor facts are indexed by actor identity so hazard
    conjunctions cannot be assembled from different actors or different traces.

    ``ego_route`` (``{"x","y"}`` dicts from ``load_ego_route``) extends the
    swept-path ego polyline beyond the observed track and 2 s projection so a
    slow crossing of the planned route ahead is credited even when the ego is
    stopped short of it.  ``None`` reproduces the legacy geometry exactly.
    """
    config = config or DerivationConfig()
    derived = DerivedTrace(ticks=list(ticks))
    previous: dict[str, dict[str, Any]] = {}
    previous_ego_road_lane: tuple[int | None, int | None] | None = None
    ego_track_history: list[TrackSample] = []

    def add(
        predicate: str,
        subject: str,
        obj: str | None,
        tick: int,
        actor_key: str,
        source: str,
    ) -> None:
        fact = Fact(
            predicate=predicate,
            subject=subject,
            obj=obj,
            tick=tick,
            actor_key=actor_key,
            source=source,
        )
        derived.facts.append(fact)
        derived.by_signature.setdefault(fact.signature, []).append(fact)
        derived.actor_facts.setdefault(actor_key, {}).setdefault(predicate, []).append(fact)

    for tick in ticks:
        ego_speed = tick.ego_speed_mps
        previous_ego = previous.get("ego")
        previous_speed = previous_ego["speed"] if previous_ego else None
        previous_yaw = previous_ego["yaw"] if previous_ego else None

        if ego_speed >= config.speed_epsilon_mps:
            add(
                "moving",
                "ego",
                None,
                tick.tick,
                "ego",
                f"telemetry.speed_mps={ego_speed:.3f} >= {config.speed_epsilon_mps}",
            )
        else:
            add(
                "stationary",
                "ego",
                None,
                tick.tick,
                "ego",
                f"telemetry.speed_mps={ego_speed:.3f} < {config.speed_epsilon_mps}",
            )

        if tick.brake >= config.braking_brake_threshold:
            add(
                "braking",
                "ego",
                None,
                tick.tick,
                "ego",
                f"telemetry.control.brake={tick.brake:.3f} >= {config.braking_brake_threshold}",
            )
        elif previous_speed is not None and ego_speed <= previous_speed - config.braking_speed_drop_mps:
            add(
                "braking",
                "ego",
                None,
                tick.tick,
                "ego",
                f"ego speed dropped {previous_speed - ego_speed:.3f} m/s >= {config.braking_speed_drop_mps}",
            )

        if abs(tick.steer) >= config.turning_steer_threshold:
            add(
                "turning",
                "ego",
                None,
                tick.tick,
                "ego",
                f"telemetry.control.steer={tick.steer:.3f} (|.| >= {config.turning_steer_threshold})",
            )
        elif (
            previous_yaw is not None
            and abs(_wrap_angle_deg(tick.ego_yaw_deg - previous_yaw))
            >= config.turning_yaw_rate_deg_per_tick
        ):
            add(
                "turning",
                "ego",
                None,
                tick.tick,
                "ego",
                "|ego yaw delta| >= turning_yaw_rate_deg_per_tick between adjacent ticks",
            )

        if tick.ego_road_id is not None:
            add(
                "on_road",
                "ego",
                None,
                tick.tick,
                "ego",
                "ego.waypoint present (ego projected to a CARLA lane)",
            )

        road_lane = (tick.ego_road_id, tick.ego_lane_id)
        if (
            previous_ego_road_lane is not None
            and road_lane[1] is not None
            and previous_ego_road_lane[1] is not None
            and road_lane != previous_ego_road_lane
        ):
            add(
                "lane_changing",
                "ego",
                None,
                tick.tick,
                "ego",
                f"ego.waypoint changed {previous_ego_road_lane} -> {road_lane} between adjacent ticks",
            )
        previous_ego_road_lane = road_lane

        ego_track_history.append(
            TrackSample(
                tick=tick.tick,
                x=tick.ego_x,
                y=tick.ego_y,
                yaw_deg=tick.ego_yaw_deg,
                velocity_mps=ego_speed,
            )
        )
        ego_track = tick.ego_track if tick.ego_track else tuple(ego_track_history)

        for actor in tick.actors:
            subject = actor.oracle_node or actor.sim_alias
            actor_key = f"{subject}#{actor.actor_id}"
            previous_actor = previous.get(actor.actor_id)

            if actor.oracle_node is not None and actor.oracle_node not in derived.node_witnesses:
                derived.node_witnesses[actor.oracle_node] = Fact(
                    predicate="actor.type_id",
                    subject=actor.oracle_node,
                    obj=None,
                    tick=tick.tick,
                    actor_key=actor_key,
                    source=(
                        f"nearby_actors[] id={actor.actor_id} type_id='{actor.type_id}' "
                        f"crosswalked to node '{actor.oracle_node}'"
                    ),
                )
            derived.node_instance_counts[subject] = derived.node_instance_counts.get(subject, 0) + 1
            if actor.oracle_node is None:
                derived.unmapped_aliases[actor.type_id] = derived.unmapped_aliases.get(actor.type_id, 0) + 1

            same_lane = actor.same_lane
            same_lane_source = "nearby_actors[].same_lane_as_ego"
            adjacent_lane = actor.adjacent_lane
            adjacent_lane_source = "nearby_actors[].adjacent_lane_as_ego"
            heading_opposed: bool | None = None
            if actor.lane_relation in ("left", "right"):
                adjacent_lane = True
                adjacent_lane_source = (
                    f"nearby_actors[].lane_relation={actor.lane_relation} "
                    "(map get_left_lane/get_right_lane)"
                )
            if tick.ego_lane is not None and actor.lane is not None:
                lane_relation = classify_lane_relation(
                    tick.ego_lane,
                    actor.lane,
                    actor_yaw_deg=actor.yaw_deg,
                    ego_yaw_deg=tick.ego_yaw_deg,
                    oncoming_opposition_deg=config.oncoming_yaw_delta_deg,
                )
                same_lane = lane_relation.same_lane
                same_lane_source = (
                    f"nearby_actors[].lane road_id={actor.lane.road_id} "
                    f"lane_id={actor.lane.lane_id} equality with ego lane"
                )
                if actor.lane_relation not in ("left", "right"):
                    adjacent_lane = lane_relation.adjacent_lane
                    adjacent_lane_source = (
                        "nearby_actors[].lane ordinal adjacency (relative_lane="
                        f"{lane_relation.relative_lane})"
                    )
                heading_opposed = lane_relation.oncoming

            if actor.in_front:
                add(
                    "in_front_of",
                    subject,
                    "ego",
                    tick.tick,
                    actor_key,
                    "nearby_actors[].is_in_front_of_ego",
                )
            if same_lane:
                add(
                    "same_lane",
                    subject,
                    "ego",
                    tick.tick,
                    actor_key,
                    same_lane_source,
                )
            if adjacent_lane:
                add(
                    "adjacent_lane",
                    subject,
                    "ego",
                    tick.tick,
                    actor_key,
                    adjacent_lane_source,
                )
            if actor.same_road:
                add(
                    "on_road",
                    subject,
                    None,
                    tick.tick,
                    actor_key,
                    "nearby_actors[].same_road_as_ego",
                )
            if actor.speed_mps >= config.speed_epsilon_mps:
                add(
                    "moving",
                    subject,
                    None,
                    tick.tick,
                    actor_key,
                    f"actor velocity_mps={actor.speed_mps:.3f} >= {config.speed_epsilon_mps}",
                )
            else:
                add(
                    "stationary",
                    subject,
                    None,
                    tick.tick,
                    actor_key,
                    f"actor velocity_mps={actor.speed_mps:.3f} < {config.speed_epsilon_mps}",
                )

            if previous_actor is not None:
                if actor.speed_mps <= previous_actor["speed"] - config.braking_speed_drop_mps:
                    add(
                        "braking",
                        subject,
                        None,
                        tick.tick,
                        actor_key,
                        "actor speed drop >= braking_speed_drop_mps between adjacent ticks",
                    )
            if subject == "vehicle":
                heading_delta = angle_difference_deg(actor.yaw_deg, tick.ego_yaw_deg)
                if actor.oncoming_flag:
                    add(
                        "oncoming",
                        subject,
                        "ego",
                        tick.tick,
                        actor_key,
                        "nearby_actors[].oncoming_as_ego (schema v2: lane membership + opposed "
                        "heading + in front + moving)",
                    )
                elif (
                    heading_opposed is True
                    and actor.in_front
                    and actor.speed_mps >= config.speed_epsilon_mps
                ):
                    add(
                        "oncoming",
                        subject,
                        "ego",
                        tick.tick,
                        actor_key,
                        "persisted lane membership + |yaw delta|="
                        f"{heading_delta:.1f} >= {config.oncoming_yaw_delta_deg} + in front + moving",
                    )
                elif (
                    heading_opposed is None
                    and actor.same_road
                    and heading_delta >= config.oncoming_yaw_delta_deg
                ):
                    add(
                        "oncoming",
                        subject,
                        "ego",
                        tick.tick,
                        actor_key,
                        "v1 fallback: same road and |yaw delta|="
                        f"{heading_delta:.1f} >= {config.oncoming_yaw_delta_deg}",
                    )
                previous_lane = previous_actor.get("lane") if previous_actor else None
                if detect_lane_change(previous_lane, actor.lane):
                    add(
                        "lane_changing",
                        subject,
                        None,
                        tick.tick,
                        actor_key,
                        "nearby_actors[].lane changed "
                        f"{previous_lane.key if previous_lane else None} -> {actor.lane.key} "
                        "between adjacent ticks (laterally adjacent lanes)",
                    )
                if (
                    previous_actor is not None
                    and abs(_wrap_angle_deg(actor.yaw_deg - previous_actor["yaw"]))
                    >= config.turning_yaw_rate_deg_per_tick
                ):
                    add(
                        "turning",
                        subject,
                        None,
                        tick.tick,
                        actor_key,
                        "|actor yaw delta| >= turning_yaw_rate_deg_per_tick",
                    )

            if subject in AGENT_NODES:
                if (
                    previous_actor is not None
                    and actor.distance_m <= previous_actor["distance"] - config.approaching_delta_m
                ):
                    add(
                        "approaching",
                        subject,
                        "ego",
                        tick.tick,
                        actor_key,
                        "distance to ego decreased >= approaching_delta_m between adjacent ticks",
                    )
                if len(actor.track) >= 2 and ego_track:
                    if detect_crossing_from_tracks(
                        actor.track,
                        ego_track,
                        actor_is_vehicle=(subject == "vehicle"),
                        config=config.crossing_config(),
                        ego_route=ego_route,
                    ):
                        path_label = (
                            "ego planned route + observed track"
                            if ego_route
                            else "ego forward path"
                        )
                        add(
                            "crossing_path",
                            subject,
                            "ego",
                            tick.tick,
                            actor_key,
                            f"swept-path intersection: {len(actor.track)}-sample actor track "
                            f"crosses {path_label} (corridor half width "
                            f"{config.crossing_corridor_half_width_m} m)",
                        )
                elif actor.in_front and actor.distance_m <= config.crossing_distance_m:
                    add(
                        "crossing_path",
                        subject,
                        "ego",
                        tick.tick,
                        actor_key,
                        f"PROXY fallback (no track geometry): in front and distance "
                        f"{actor.distance_m:.2f} <= {config.crossing_distance_m}",
                    )

            if subject == "pedestrian":
                if actor.same_road:
                    add(
                        "jaywalking",
                        subject,
                        None,
                        tick.tick,
                        actor_key,
                        "PROXY: pedestrian same_road_as_ego (crosswalk legality unavailable)",
                    )
                if actor.same_road and actor.in_front and actor.speed_mps < config.waiting_speed_mps:
                    add(
                        "waiting",
                        subject,
                        None,
                        tick.tick,
                        actor_key,
                        "PROXY: on-road pedestrian in front with speed < waiting_speed_mps",
                    )

            if actor.in_front and actor.same_lane and actor.distance_m <= config.obstructing_distance_m:
                add(
                    "obstructing",
                    subject,
                    "ego",
                    tick.tick,
                    actor_key,
                    "in front, same lane, distance <= obstructing_distance_m",
                )

            previous[actor.actor_id] = {
                "distance": actor.distance_m,
                "yaw": actor.yaw_deg,
                "speed": actor.speed_mps,
                "lane": actor.lane,
            }

        previous["ego"] = {"speed": ego_speed, "yaw": tick.ego_yaw_deg}

    return derived


# ---------------------------------------------------------------------------
# Coverage computation
# ---------------------------------------------------------------------------


@dataclass
class TraceInput:
    label: str
    ticks: list[NormalizedTick]
    ego_route: Sequence[Any] = ()


def _obligation_mapping_status(obligation: OracleObligation) -> MappingStatus:
    if obligation.dimension == "node":
        node_type = obligation.node_types[0] if obligation.node_types else obligation.predicate
        return node_type_status(node_type)
    node_type = obligation.node_types[0] if obligation.node_types else ""
    return mapping_for(obligation.predicate, node_type)


def _hazard_mapping_status(
    obligation: OracleObligation,
) -> tuple[MappingStatus, dict[str, str], str | None]:
    """A hazard is mapped only if every required predicate is actor-observable."""
    if not obligation.required_predicates:
        return (
            MappingStatus(
                mapped=False,
                sim_predicate=None,
                grounding=None,
                evidence="",
                reason="oracle hazard class records no required_predicates",
            ),
            {},
            None,
        )
    if not obligation.node_types:
        return (
            MappingStatus(
                mapped=False,
                sim_predicate=None,
                grounding=None,
                evidence="",
                reason="oracle hazard class records no node_types; adversary cannot be bound",
            ),
            {},
            None,
        )
    node_type = obligation.node_types[0]
    sim_predicates: dict[str, str] = {}
    groundings: list[str] = []
    for predicate in obligation.required_predicates:
        status = mapping_for(predicate, node_type)
        if not status.mapped:
            return (
                MappingStatus(
                    mapped=False,
                    sim_predicate=None,
                    grounding=None,
                    evidence="",
                    reason=(
                        f"required predicate '{predicate}' has no actor-level witness for a "
                        f"{node_type} adversary: {status.reason}"
                    ),
                ),
                {},
                None,
            )
        sim_predicates[predicate] = status.sim_predicate or predicate
        groundings.append(status.grounding or GROUNDING_DERIVED)
    grounding = (
        GROUNDING_PROXY
        if GROUNDING_PROXY in groundings
        else GROUNDING_DERIVED
        if GROUNDING_DERIVED in groundings
        else GROUNDING_DIRECT
    )
    return (
        MappingStatus(
            mapped=True,
            sim_predicate=",".join(sim_predicates.values()),
            grounding=grounding,
            evidence="conjunction of " + "; ".join(
                f"{pred}<-{sim}" for pred, sim in sim_predicates.items()
            ),
            reason=None,
        ),
        sim_predicates,
        grounding,
    )


def _conjunction_window(
    ticks_by_predicate: dict[str, list[int]], window_ticks: int
) -> tuple[int, int] | None:
    if not ticks_by_predicate or any(not ticks for ticks in ticks_by_predicate.values()):
        return None
    union = sorted({tick for ticks in ticks_by_predicate.values() for tick in ticks})
    if not union:
        return None
    if window_ticks <= 0:
        end = max(min(ticks) for ticks in ticks_by_predicate.values())
        return (min(ticks[0] for ticks in ticks_by_predicate.values()), end)
    for end in union:
        start = end - window_ticks + 1
        if all(any(start <= tick <= end for tick in ticks) for ticks in ticks_by_predicate.values()):
            return (start, end)
    return None


def _hazard_witness(
    derived: DerivedTrace,
    obligation: OracleObligation,
    sim_predicates: dict[str, str],
    config: DerivationConfig,
) -> dict[str, Any] | None:
    """Find one actor of the class node type satisfying the whole conjunction in one window."""
    node_type = obligation.node_types[0] if obligation.node_types else None
    if node_type is None:
        return None
    for actor_key in sorted(derived.actor_facts):
        if actor_key != node_type and not actor_key.startswith(f"{node_type}#"):
            continue
        facts_by_predicate = derived.actor_facts[actor_key]
        ticks_by_predicate: dict[str, list[int]] = {}
        for predicate, sim_predicate in sim_predicates.items():
            ticks = sorted(fact.tick for fact in facts_by_predicate.get(sim_predicate, ()))
            if not ticks:
                ticks_by_predicate = {}
                break
            ticks_by_predicate[predicate] = ticks
        if not ticks_by_predicate:
            continue
        window = _conjunction_window(ticks_by_predicate, config.hazard_window_ticks)
        if window is None:
            continue
        return {
            "witness_tick": window[1],
            "window": [window[0], window[1]],
            "witness_actor": actor_key,
            "predicate_ticks": ticks_by_predicate,
            "predicate_sources": {
                predicate: (
                    facts_by_predicate[sim_predicates[predicate]][0].source
                    if facts_by_predicate.get(sim_predicates[predicate])
                    else ""
                )
                for predicate in sim_predicates
            },
        }
    return None


def _obligation_witness(
    derived: DerivedTrace,
    obligation: OracleObligation,
    status: MappingStatus,
) -> Fact | None:
    if status.sim_predicate is None:
        return None
    if obligation.dimension == "node":
        node_type = obligation.node_types[0] if obligation.node_types else obligation.predicate
        return derived.node_witnesses.get(node_type)
    subject = obligation.node_types[0] if obligation.node_types else ""
    if obligation.dimension == "attribute":
        return derived.witness(status.sim_predicate, subject, None)
    if obligation.dimension == "relation":
        obj = obligation.node_types[1] if len(obligation.node_types) > 1 else "ego"
        return derived.witness(status.sim_predicate, subject, obj)
    return None


def _evaluate(
    oracle: Oracle,
    traces: Sequence[TraceInput],
    config: DerivationConfig,
    ego_route: Sequence[Any] | None = None,
) -> dict[str, Any]:
    derived_traces = [
        derive_trace(trace.ticks, config, ego_route=trace.ego_route or ego_route)
        for trace in traces
    ]

    axis_results: dict[str, dict[str, Any]] = {}
    for axis in ("V", "A", "E", "H"):
        rows: list[dict[str, Any]] = []
        for obligation in oracle.obligations_for_axis(axis):
            if axis == "H":
                status, sim_predicates, hazard_grounding = _hazard_mapping_status(obligation)
            else:
                status = _obligation_mapping_status(obligation)
                sim_predicates, hazard_grounding = {}, None

            witness: dict[str, Any] | None = None
            if status.mapped:
                for trace_index, derived in enumerate(derived_traces):
                    if axis == "H":
                        candidate = _hazard_witness(derived, obligation, sim_predicates, config)
                    else:
                        fact = _obligation_witness(derived, obligation, status)
                        candidate = (
                            {
                                "witness_tick": fact.tick,
                                "witness_actor": fact.actor_key,
                                "predicate_ticks": {obligation.predicate: [fact.tick]},
                                "predicate_sources": {obligation.predicate: fact.source},
                            }
                            if fact is not None
                            else None
                        )
                    if candidate is not None:
                        witness = {
                            **candidate,
                            "witness_trace": traces[trace_index].label,
                            "witness_trace_index": trace_index,
                        }
                        break

            row = {
                "dimension": obligation.dimension,
                "signature": obligation.signature,
                "predicate": obligation.predicate,
                "node_types": list(obligation.node_types),
                "oracle_grounding": obligation.grounding,
                "mapped": status.mapped,
                "covered": witness is not None,
                "sim_predicate": status.sim_predicate,
                "grounding": hazard_grounding or (status.grounding if status.mapped else None),
                "evidence": status.evidence if status.mapped else status.reason,
                "unmapped_reason": None if status.mapped else status.reason,
                "witness_tick": witness["witness_tick"] if witness else None,
                "witness_actor": witness["witness_actor"] if witness else None,
                "witness_trace": witness["witness_trace"] if witness else None,
                "witness_window": witness.get("window") if witness else None,
                "predicate_ticks": witness.get("predicate_ticks") if witness else None,
                "predicate_sources": witness.get("predicate_sources") if witness else None,
            }
            if not status.mapped:
                row["evidence"] = status.reason
            elif witness is None:
                row["evidence"] = f"mapped via {status.sim_predicate} but never witnessed in the retained trace(s)"
            rows.append(row)

        mapped_rows = [row for row in rows if row["mapped"]]
        covered_rows = [row for row in mapped_rows if row["covered"]]
        axis_results[axis] = {
            "axis": axis,
            "label": AXIS_LABELS[axis],
            "total_obligations": len(rows),
            "mapped_obligations": len(mapped_rows),
            "unmapped_obligations": len(rows) - len(mapped_rows),
            "covered_mapped_obligations": len(covered_rows),
            "Cov": (len(covered_rows) / len(mapped_rows)) if mapped_rows else None,
            "obligations": rows,
        }

    total_mapped = sum(result["mapped_obligations"] for result in axis_results.values())
    total_obligations = sum(result["total_obligations"] for result in axis_results.values())
    total_covered = sum(result["covered_mapped_obligations"] for result in axis_results.values())
    mapped_ratios = [
        result["Cov"] for result in axis_results.values() if result["Cov"] is not None
    ]
    full_ratios = [
        (result["covered_mapped_obligations"] / result["total_obligations"])
        if result["total_obligations"]
        else None
        for result in axis_results.values()
    ]
    warning = (
        f"Cov_V/A/E/H are computed over the MAPPED obligation subset only: "
        f"{total_mapped} of {total_obligations} oracle obligations have a simulator-side witness "
        f"path ({100.0 * total_mapped / total_obligations:.1f}%) in this crosswalk. "
        "coverage_of_mapped_subset is NOT full-vocabulary coverage; unmapped obligations are "
        "listed with reasons in dimensions[*].obligations. Do not quote Cov_* as full-vocabulary "
        "coverage."
    )

    unmapped_simulator = [
        {
            "sim_predicate": mapping.sim_predicate,
            "reason": mapping.reason,
            "grounding": mapping.grounding,
        }
        for mapping in CROSSWALK
        if mapping.oracle_predicate is None
    ]
    oracle_status_unmapped: list[dict[str, Any]] = []
    oracle_status_mapped: list[dict[str, Any]] = []
    seen_predicates: set[tuple[str, str]] = set()
    for axis in ("V", "A", "E", "H"):
        for row in axis_results[axis]["obligations"]:
            key = (row["dimension"], row["predicate"] if axis != "H" else f"hazard:{row['predicate']}")
            if key in seen_predicates:
                continue
            seen_predicates.add(key)
            entry = {
                "dimension": row["dimension"],
                "predicate": row["predicate"],
                "axis": axis,
                "sim_predicate": row["sim_predicate"],
                "reason": row["unmapped_reason"],
            }
            if row["mapped"]:
                oracle_status_mapped.append(entry)
            else:
                oracle_status_unmapped.append(entry)

    return {
        "engine": ENGINE_NAME,
        "schema_version": SCHEMA_VERSION,
        "oracle": {
            "path": oracle.path,
            "split": oracle.split,
            "total_obligations": len(oracle.obligations),
            "defined_predicates": oracle.counts.get("total", {}).get(
                "defined_predicates", len(oracle.predicates_by_axis)
            ),
            "predicates_by_axis": {axis: list(names) for axis, names in oracle.predicates_by_axis.items()},
        },
        "config": config.to_dict(),
        "traces": [
            {
                "label": trace.label,
                "ticks": len(trace.ticks),
                "scenario_id": trace.ticks[0].scenario_id if trace.ticks else None,
                "town": trace.ticks[0].town if trace.ticks else None,
                "facts": len(derived.facts),
            }
            for trace, derived in zip(traces, derived_traces)
        ],
        "crosswalk": {
            "mappings": [
                {
                    "sim_predicate": mapping.sim_predicate,
                    "oracle_predicate": mapping.oracle_predicate,
                    "grounding": mapping.grounding,
                    "nodes": sorted(mapping.nodes),
                    "evidence": mapping.evidence,
                    "reason": mapping.reason,
                }
                for mapping in CROSSWALK
            ],
            "unmapped_simulator_predicates": unmapped_simulator,
        },
        "oracle_predicate_status": {
            "mapped": oracle_status_mapped,
            "unmapped": oracle_status_unmapped,
        },
        "dimensions": axis_results,
        "Cov_V": axis_results["V"]["Cov"],
        "Cov_A": axis_results["A"]["Cov"],
        "Cov_E": axis_results["E"]["Cov"],
        "Cov_H": axis_results["H"]["Cov"],
        "coverage_of_mapped_subset": {
            "V": axis_results["V"]["Cov"],
            "A": axis_results["A"]["Cov"],
            "E": axis_results["E"]["Cov"],
            "H": axis_results["H"]["Cov"],
            "macro": (sum(mapped_ratios) / len(mapped_ratios)) if mapped_ratios else None,
            "micro": (total_covered / total_mapped) if total_mapped else None,
        },
        "full_vocabulary_coverage": {
            "V": full_ratios[0],
            "A": full_ratios[1],
            "E": full_ratios[2],
            "H": full_ratios[3],
            "micro": (total_covered / total_obligations) if total_obligations else None,
        },
        "mapped_obligation_count": total_mapped,
        "total_obligation_count": total_obligations,
        "covered_mapped_obligation_count": total_covered,
        "unmapped_simulator_aliases": {
            type_id: count
            for derived in derived_traces
            for type_id, count in derived.unmapped_aliases.items()
        },
        "hazard_window_ticks": config.hazard_window_ticks,
        "warning": warning,
    }


def compute_cov(
    oracle: Oracle,
    trace: Sequence[NormalizedTick] | Sequence[dict[str, Any]],
    *,
    trace_label: str = "trace",
    config: DerivationConfig | None = None,
    ego_route: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compute all four coverage ratios for a single trace.

    ``trace`` may be a list of normalized ticks (``load_semantic_trace``) or a
    list of raw JSONL dicts.  Hazard obligations can only be credited inside
    this one trace; they are never assembled from fragments of other traces.
    ``ego_route`` is the stream's persisted planned route (``load_ego_route``).
    """
    ticks = _coerce_ticks(trace)
    return _evaluate(
        oracle,
        [TraceInput(label=trace_label, ticks=ticks)],
        config or DerivationConfig(),
        ego_route=ego_route,
    )


def compute_suite_cov(
    oracle: Oracle,
    traces: Sequence[TraceInput | tuple[str, Sequence[Any]]],
    *,
    config: DerivationConfig | None = None,
    ego_route: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compute coverage over multiple traces.

    Node/attribute/relation obligations are unioned across traces (the paper's
    suite rule).  Hazard obligations are credited only when a single retained
    trace satisfies the whole conjunction.  ``ego_route`` is the default planned
    route for ``TraceInput`` entries that do not carry their own.
    """
    normalized: list[TraceInput] = []
    for trace in traces:
        if isinstance(trace, TraceInput):
            normalized.append(
                TraceInput(
                    label=trace.label,
                    ticks=_coerce_ticks(trace.ticks, default_label=trace.label),
                    ego_route=trace.ego_route,
                )
            )
        else:
            label, raw_ticks = trace
            normalized.append(TraceInput(label=label, ticks=_coerce_ticks(raw_ticks, default_label=label)))
    return _evaluate(oracle, normalized, config or DerivationConfig(), ego_route=ego_route)


def _coerce_ticks(
    trace: Sequence[NormalizedTick] | Sequence[dict[str, Any]],
    default_label: str = "trace",
) -> list[NormalizedTick]:
    ticks: list[NormalizedTick] = []
    for index, item in enumerate(trace):
        if isinstance(item, NormalizedTick):
            ticks.append(item)
        elif isinstance(item, dict):
            ticks.append(normalize_tick(item, index=index))
        else:
            raise TypeError(
                f"trace item {index} for '{default_label}' is neither a NormalizedTick nor a raw dict"
            )
    return ticks


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Semantic Coverage Report (mapped subset only)",
        "",
        f"> **WARNING**: {report['warning']}",
        "",
        "## Oracle and traces",
        "",
        f"- Oracle: `{report['oracle']['path']}` (split `{report['oracle']['split']}`)",
        f"- Oracle obligations: {report['oracle']['total_obligations']}"
        f" / defined predicates: {report['oracle']['defined_predicates']}",
        f"- Mapped obligations: {report['mapped_obligation_count']}"
        f" / total: {report['total_obligation_count']}",
        f"- Hazard conjunction window: {report['hazard_window_ticks']} ticks"
        " (oracle inventory has no explicit per-hazard window; 6 mirrors its 6-keyframe slices)",
        "",
        "| Trace | Ticks | Scenario | Town | Facts |",
        "|---|---:|---|---|---:|",
    ]
    for trace in report["traces"]:
        lines.append(
            f"| `{trace['label']}` | {trace['ticks']} | {trace['scenario_id']} | "
            f"{trace['town']} | {trace['facts']} |"
        )

    lines += [
        "",
        "## Coverage summary",
        "",
        "| Axis | Dimension | Cov (mapped subset) | Cov (full vocabulary) | Covered | Mapped | Unmapped | Oracle total |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for axis in ("V", "A", "E", "H"):
        result = report["dimensions"][axis]
        cov = result["Cov"]
        full = report["full_vocabulary_coverage"][axis]
        lines.append(
            f"| {axis} | {result['label']} | "
            f"{'n/a' if cov is None else f'{cov:.3f}'} | "
            f"{'n/a' if full is None else f'{full:.3f}'} | "
            f"{result['covered_mapped_obligations']} | {result['mapped_obligations']} | "
            f"{result['unmapped_obligations']} | {result['total_obligations']} |"
        )
    lines.append(
        f"| macro | all | {report['coverage_of_mapped_subset']['macro']:.3f} | "
        f"{report['full_vocabulary_coverage']['micro']:.3f} | "
        f"{report['covered_mapped_obligation_count']} | {report['mapped_obligation_count']} | "
        f"{report['total_obligation_count'] - report['mapped_obligation_count']} | "
        f"{report['total_obligation_count']} |"
    )

    lines += [
        "",
        "## Crosswalk (simulator predicate -> oracle predicate)",
        "",
        "| Simulator predicate | Oracle predicate | Grounding | Node scope | Evidence |",
        "|---|---|---|---|---|",
    ]
    for mapping in report["crosswalk"]["mappings"]:
        oracle_predicate = mapping["oracle_predicate"]
        oracle_cell = "-- unmapped --" if oracle_predicate is None else f"`{oracle_predicate}`"
        lines.append(
            f"| `{mapping['sim_predicate']}` | {oracle_cell} | "
            f"{mapping['grounding']} | {', '.join(mapping['nodes']) or '(none)'} | "
            f"{(mapping['evidence'] or mapping['reason'] or '').replace('|', '/')} |"
        )

    lines += ["", "## Unmapped simulator-side predicates", "", "| Predicate | Reason |", "|---|---|"]
    for mapping in report["crosswalk"]["unmapped_simulator_predicates"]:
        lines.append(f"| `{mapping['sim_predicate']}` | {mapping['reason']} |")

    lines += [
        "",
        "## Oracle predicates without a simulator-side witness",
        "",
        "| Dimension | Predicate | Reason |",
        "|---|---|---|",
    ]
    for entry in report["oracle_predicate_status"]["unmapped"]:
        lines.append(
            f"| {entry['dimension']} | `{entry['predicate']}` | {entry['reason']} |"
        )

    for axis in ("V", "A", "E", "H"):
        result = report["dimensions"][axis]
        lines += [
            "",
            f"## {result['label'].title()} obligations ({axis})",
            "",
            "| Obligation | Mapped | Covered | Sim predicate | Grounding | Witness tick | Witness actor | Evidence |",
            "|---|---|---|---|---|---:|---|---|",
        ]
        for row in result["obligations"]:
            witness_actor = row["witness_actor"]
            if row["witness_trace"] is not None:
                witness_actor = f"{witness_actor} @ {row['witness_trace']}"
            evidence = row["evidence"] or ""
            if row["witness_window"]:
                evidence += f" [window {row['witness_window'][0]}..{row['witness_window'][1]}]"
            lines.append(
                f"| `{row['signature']}` | {'yes' if row['mapped'] else 'no'} | "
                f"{'yes' if row['covered'] else 'no'} | {row['sim_predicate'] or '-'} | "
                f"{row['grounding'] or '-'} | {row['witness_tick'] if row['witness_tick'] is not None else '-'} | "
                f"{witness_actor or '-'} | {evidence.replace('|', '/')} |"
            )

    return "\n".join(lines) + "\n"


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=False) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline semantic coverage engine: compute Cov_V/Cov_A/Cov_E/Cov_H over the "
            "mapped subset of the EXP-018 nuScenes oracle from simulator semantic streams."
        )
    )
    parser.add_argument("--oracle", type=Path, required=True, help="Path to an EXP-018 oracle inventory JSON.")
    parser.add_argument(
        "--traces",
        type=Path,
        nargs="+",
        required=True,
        help="One or more semantic-stream JSONL paths, or directories containing *.jsonl.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output path (defaults to stdout).")
    parser.add_argument("--format", choices=("json", "md"), default="json", help="Output format.")
    parser.add_argument(
        "--hazard-window",
        type=int,
        default=6,
        help="Ticks allowed between the first and last required predicate of a hazard conjunction (0 disables).",
    )
    parser.add_argument(
        "--crossing-distance-m",
        type=float,
        default=12.0,
        help="Distance threshold for the crossing_path derivation (mirrors SemanticObserver's 12.0 m).",
    )
    return parser


def resolve_trace_paths(paths: Iterable[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.jsonl")))
        else:
            resolved.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    oracle = load_oracle(args.oracle)
    trace_paths = resolve_trace_paths(args.traces)
    if not trace_paths:
        print("No semantic stream JSONL files found.", file=sys.stderr)
        return 2

    config = DerivationConfig(
        hazard_window_ticks=args.hazard_window,
        crossing_distance_m=args.crossing_distance_m,
    )
    traces: list[TraceInput] = []
    for path in trace_paths:
        ticks = load_semantic_trace(path)
        route = load_ego_route(path)
        traces.append(TraceInput(label=str(path), ticks=ticks, ego_route=route))
        print(
            f"loaded {len(ticks)} ticks from {path} (ego_route points: {len(route)})",
            file=sys.stderr,
        )

    report = compute_suite_cov(oracle, traces, config=config)
    rendered = render_markdown(report) if args.format == "md" else render_json(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.format} report to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(rendered)

    cov = report["coverage_of_mapped_subset"]
    print(
        "coverage_of_mapped_subset "
        f"V={cov['V'] if cov['V'] is None else round(cov['V'], 4)} "
        f"A={cov['A'] if cov['A'] is None else round(cov['A'], 4)} "
        f"E={cov['E'] if cov['E'] is None else round(cov['E'], 4)} "
        f"H={cov['H'] if cov['H'] is None else round(cov['H'], 4)} "
        f"({report['mapped_obligation_count']}/{report['total_obligation_count']} obligations mapped)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
