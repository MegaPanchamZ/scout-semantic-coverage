"""Heuristic simulator-side semantic observer.

PREDICATE GROUNDING CLASSES
---------------------------
direct
    ``in_front_of``   -- ``dot(target - ego, ego_forward) > 0``, the same
    forward half-plane test the EXP-018 nuScenes oracle uses.
    ``on_road``       -- actor/ego projected onto a CARLA lane
    (``project_to_road=True`` waypoint present).
    ``same_lane``     -- actor and ego occupy the same OpenDRIVE lane
    (``road_id`` + ``lane_id`` equality from the lane index).
    ``adjacent_lane`` -- actor lane is the left or right lateral neighbour of
    the ego lane in the lane index.
    ``colliding``     -- collision-oracle contact event (the oracle itself
    declares ``colliding`` ungrounded, so this is simulator ground truth).

derived
    ``oncoming``      -- same road as ego, known lane membership, opposed
    heading (>= ``oncoming_opposition_deg``), actor in front and moving
    (mirrors the oracle's oncoming definition).
    ``crossing_path`` -- swept-path intersection between the actor track (or
    the actor's map waypoint sweep) and the ego path, shape-matched to the
    oracle: opposite sides of the ego polyline, closest approach within
    ``crossing_corridor_half_width_m``, minimum actor displacement, and the
    ``[30, 150]`` degree heading range for vehicles.
    ``lane_changing`` -- actor lane membership changed to a laterally adjacent
    lane between adjacent ticks.
    ``obstructing``   -- same_lane AND in_front_of (the oracle definition has no
    distance term; the observer's legacy 18 m bound is only used for occluders).
    ``occluded``      -- occluder identity supplied by harness context (the
    oracle declares ``occluded`` ungrounded, so it is not oracle-mappable).

proxy
    ``jaywalking``    -- pedestrian projected onto the roadway; the stream
    cannot express crosswalk legality, matching the oracle's own proxy
    grounding for this predicate.
    ``crossing_path`` -- the legacy "in front and within 12 m" corridor test is
    kept only as a documented fallback for agent actors when neither a track of
    >= 2 samples nor a usable waypoint sweep exists.
    ``waiting``       -- scripted threshold-adversary hold state, not an
    observed pedestrian posture; the coverage engine maps it as a proxy.

SEMANTIC STREAM SCHEMA
----------------------
Schema version 2 (``SEMANTIC_STREAM_SCHEMA_VERSION``) is additive over v1.
Every frame gains ``schema_version``; ``ego`` gains ``lane`` and
``waypoint_history``; each ``nearby_actors[]`` entry gains ``lane``,
``lane_relation``, ``adjacent_lane_as_ego``, ``oncoming_as_ego`` and
``waypoint_history``.  The first stream frame additionally carries
``ego_route`` (the harness context's ``route_waypoints``, a list of
``{"x","y"}`` points, capped by the agent) when a planned route is available,
so the offline coverage engine can extend the swept-path ego polyline beyond
the observed track and 2 s projection; the key is absent when unavailable.
All v1 keys keep their names and meaning, and readers must tolerate missing v2
keys (the coverage engine normalizes defensively).

``waypoint_history_ticks`` defaults to 50 (5 s at 10 Hz) instead of the
original 10 ticks: a slow pedestrian crossing of the ego corridor can span
more than 20 ticks, and no 1 s window then holds track samples on both sides
of the ego path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import math

from research.harness.lane_index import (
    SEMANTIC_STREAM_SCHEMA_VERSION,
    CrossingPathConfig,
    LaneRef,
    TrackSample,
    classify_lane_relation,
    detect_crossing_from_tracks,
    detect_lane_change,
    detect_path_crossing,
    lane_ref_from_waypoint,
    waypoint_swept_path,
)
from research.harness.obligation_credit import compute_obligation_credit
from research.harness.models import RunResult, ScenarioSpec


@dataclass(slots=True)
class SemanticObserverConfig:
    source: str = "heuristic"
    trace_output_dir: Path | None = None
    stream_output_dir: Path | None = None
    stream_every_ticks: int = 1
    anomaly_dump_dir: Path | None = None
    capture_ticks: tuple[int, ...] = ()
    persistent_brake_ticks: int = 10
    heading_error_threshold_deg: float = 20.0
    nearby_actor_radius_m: float = 50.0
    enable_lane_relations: bool = True
    lane_relation_radius_m: float | None = None
    # 5 s at 10 Hz.  A slow pedestrian crossing of the ego corridor can span
    # 20+ ticks; with the old 1 s window no single actor window held samples on
    # both sides of the ego path, so swept-path crossing_path was never
    # credited.  Still configurable per run.
    waypoint_history_ticks: int = 50
    oncoming_opposition_deg: float = 135.0
    crossing_corridor_half_width_m: float = 2.0
    crossing_min_actor_displacement_m: float = 0.5
    crossing_ego_horizon_m: float = 30.0
    crossing_actor_horizon_m: float = 20.0
    crossing_step_m: float = 2.0
    target_predicates: set[str] = field(
        default_factory=lambda: {
            "jaywalking",
            "waiting",
            "occluded",
            "on_road",
            "in_front_of",
            "crossing_path",
            "obstructing",
            "colliding",
            "same_lane",
            "adjacent_lane",
            "oncoming",
            "lane_changing",
        }
    )


@dataclass(slots=True)
class NearbyActorState:
    """One scanned nearby actor plus the derived lane-index relation flags."""

    actor: Any
    entry: dict[str, Any]
    same_lane: bool
    adjacent_lane: bool
    oncoming: bool
    lane_changed: bool
    crossing_path: bool
    crossing_geometry_available: bool


class SemanticObserver:
    def __init__(self, config: SemanticObserverConfig) -> None:
        self.config = config
        self._status = "not-started"
        self._notes: list[str] = []
        self._trace: list[dict[str, Any]] = []
        self._covered_predicates: set[str] = set()
        self._covered_signatures: set[str] = set()
        self._last_collision_frame: int | None = None
        self._persistent_brake_ticks = 0
        self._scene_dumps: list[dict[str, Any]] = []
        self._stream_frames: list[dict[str, Any]] = []
        self._anomaly_dump_index: int | None = None
        self._captured_target_ticks: set[int] = set()
        self._actor_tracks: dict[str, deque[TrackSample]] = {}
        self._actor_lanes: dict[str, LaneRef] = {}
        self._ego_track: deque[TrackSample] = deque(maxlen=max(int(config.waypoint_history_ticks), 1))
        self._ego_route: list[dict[str, float]] = []
        self._world_map: Any | None = None
        self._scan_cache: tuple[int, list[NearbyActorState]] | None = None

    def on_run_start(self, scenario: ScenarioSpec, context: dict[str, Any]) -> None:
        self._trace = []
        self._notes = []
        self._covered_predicates = set()
        self._covered_signatures = set()
        self._last_collision_frame = None
        self._persistent_brake_ticks = 0
        self._scene_dumps = []
        self._stream_frames = []
        self._anomaly_dump_index = None
        self._captured_target_ticks = set()
        self._actor_tracks = {}
        self._actor_lanes = {}
        self._ego_track = deque(maxlen=max(int(self.config.waypoint_history_ticks), 1))
        self._ego_route = self._normalize_route_waypoints(context.get("route_waypoints"))
        self._world_map = None
        self._scan_cache = None
        context["semantic_observer"] = self

        if self.config.source != "heuristic":
            self._status = "source-unavailable"
            self._notes.append(
                f"Semantic source '{self.config.source}' is not implemented yet; using no-op semantics."
            )
            return

        self._status = "collecting"
        self._notes.append(f"Semantic observer started for scenario '{scenario.scenario_id}'.")

    def on_tick(self, tick_index: int, context: dict[str, Any]) -> None:
        if self._status != "collecting":
            return

        ego_vehicle = context.get("ego_vehicle")
        if ego_vehicle is None:
            return

        frame_id = int(tick_index)
        telemetry = context.get("telemetry")
        scan: list[NearbyActorState] = []
        if self.config.enable_lane_relations:
            scan = self._scan_nearby_actors(context, frame_id)
            self._scan_cache = (frame_id, scan)
        self._update_stream_capture(frame_id, context, telemetry)
        self._update_target_tick_capture(frame_id, context, telemetry)
        self._update_anomaly_capture(frame_id, context, telemetry)

        frame_predicates: list[dict[str, Any]] = []
        jaywalker = context.get("jaywalker")
        occluder = context.get("occluder_vehicle")
        collision_oracle = context.get("collision_oracle")

        if jaywalker is not None and self._is_alive(jaywalker):
            frame_predicates.extend(self._infer_jaywalker_predicates(ego_vehicle, jaywalker, context))

        if occluder is not None and self._is_alive(occluder):
            frame_predicates.extend(self._infer_occluder_predicates(ego_vehicle, occluder))

        if scan:
            frame_predicates.extend(self._infer_lane_relation_predicates(scan))

        if collision_oracle is not None and collision_oracle.events:
            latest = collision_oracle.events[-1]
            if self._last_collision_frame != latest.frame:
                self._last_collision_frame = latest.frame
                frame_predicates.append(
                    {
                        "predicate": "colliding",
                        "signature": f"colliding(ego,{latest.actor_type})",
                        "entities": ["ego", latest.actor_type],
                        "confidence": 1.0,
                    }
                )

        frame_predicates = self._merge_predicates(frame_predicates)
        if frame_predicates:
            self._trace.append({"tick": frame_id, "predicates": frame_predicates})
            for item in frame_predicates:
                self._covered_predicates.add(item["predicate"])
                self._covered_signatures.add(item["signature"])

    def on_run_end(self, result: RunResult, context: dict[str, Any]) -> None:
        trace_path: str | None = None
        if self.config.trace_output_dir is not None and self._trace:
            trace_path = str(self._write_trace_dump(result, self.config.trace_output_dir))
        stream_path: str | None = None
        if self.config.stream_output_dir is not None and self._stream_frames:
            stream_path = str(self._write_stream_dump(result, self.config.stream_output_dir))

        coverage_denominator = len(self.config.target_predicates) if self.config.target_predicates else 0
        graph_coverage = (
            len(self._covered_predicates.intersection(self.config.target_predicates)) / coverage_denominator
            if coverage_denominator
            else None
        )
        scene_dump_paths = self._write_scene_dumps(result) if self._scene_dumps and self.config.anomaly_dump_dir is not None else []
        anomaly_dump_path = (
            scene_dump_paths[self._anomaly_dump_index]
            if self._anomaly_dump_index is not None and self._anomaly_dump_index < len(scene_dump_paths)
            else None
        )
        target_tick_dump_paths = {
            str(dump["tick"]): scene_dump_paths[index]
            for index, dump in enumerate(self._scene_dumps)
            if dump.get("trigger_reason", "").startswith("target-tick-") and index < len(scene_dump_paths)
        }

        scenario = context.get("scenario")
        obligation_credit = (
            compute_obligation_credit(
                scenario,
                {
                    "covered_predicates": sorted(self._covered_predicates),
                    "covered_signatures": sorted(self._covered_signatures),
                },
            )
            if isinstance(scenario, ScenarioSpec)
            else None
        )

        result.metadata["semantic"] = {
            "status": self._status,
            "source": self.config.source,
            "notes": list(self._notes),
            "trace_count": len(self._trace),
            "covered_predicates": sorted(self._covered_predicates),
            "covered_signatures": sorted(self._covered_signatures),
            "target_predicates": sorted(self.config.target_predicates),
            "graph_coverage": graph_coverage,
            "capture_ticks": list(self.config.capture_ticks),
            "anomaly_dump_path": anomaly_dump_path,
            "scene_dump_paths": scene_dump_paths,
            "target_tick_dump_paths": target_tick_dump_paths,
            "trace_dump_path": trace_path,
            "stream_dump_path": stream_path,
            "stream_frame_count": len(self._stream_frames),
            "stream_every_ticks": self.config.stream_every_ticks,
            "stream_schema_version": SEMANTIC_STREAM_SCHEMA_VERSION,
            "obligation_credit": obligation_credit,
        }

        scenario_notes = context.get("scenario_notes")
        if isinstance(scenario_notes, list):
            result.notes.extend([note for note in scenario_notes if note not in result.notes])

    def _infer_jaywalker_predicates(self, ego_vehicle: Any, jaywalker: Any, context: dict[str, Any]) -> list[dict[str, Any]]:
        predicates: list[dict[str, Any]] = []
        distance = self._distance_between(ego_vehicle, jaywalker)
        if distance is None:
            return predicates

        threshold_actor = context.get("threshold_adversary")
        threshold_is_pedestrian = context.get("threshold_adversary_kind") == "walker"
        threshold_active = bool(context.get("threshold_adversary_active", False))
        is_threshold_jaywalker = threshold_is_pedestrian and threshold_actor is jaywalker

        if is_threshold_jaywalker and not threshold_active:
            predicates.append(self._make_fact("waiting", "pedestrian", None, 0.9))
        else:
            predicates.append(self._make_fact("jaywalking", "pedestrian", None, 0.9))
        predicates.append(self._make_fact("on_road", "pedestrian", None, 0.7))

        if self._is_in_front_of(ego_vehicle, jaywalker):
            predicates.append(self._make_fact("in_front_of", "pedestrian", "ego", 0.8))

        if is_threshold_jaywalker and threshold_active:
            predicates.append(self._make_fact("crossing_path", "pedestrian", "ego", 0.85))
        elif distance < 12.0:
            predicates.append(self._make_fact("crossing_path", "pedestrian", "ego", 0.75))

        return predicates

    def _infer_occluder_predicates(self, ego_vehicle: Any, occluder: Any) -> list[dict[str, Any]]:
        predicates: list[dict[str, Any]] = []
        distance = self._distance_between(ego_vehicle, occluder)
        if distance is None:
            return predicates

        predicates.append(self._make_fact("occluded", self._actor_alias(occluder), None, 0.85))
        if self._is_in_front_of(ego_vehicle, occluder):
            predicates.append(self._make_fact("in_front_of", self._actor_alias(occluder), "ego", 0.85))
        if distance < 18.0:
            predicates.append(self._make_fact("obstructing", self._actor_alias(occluder), "ego", 0.8))
        return predicates

    def _scan_nearby_actors(self, context: dict[str, Any], frame_id: int) -> list[NearbyActorState]:
        """Index lanes for every nearby actor and derive the lane-relation flags.

        Direct groundings (lane membership, same/adjacent lane) come from the
        CARLA map API; derived groundings (oncoming, lane_changing, crossing
        path) combine the lane index with heading and swept-path geometry.
        """
        world = context.get("world")
        ego_vehicle = context.get("ego_vehicle")
        if world is None or ego_vehicle is None:
            return []
        if self._world_map is None:
            try:
                self._world_map = world.get_map()
            except RuntimeError:
                return []
        world_map = self._world_map

        try:
            ego_transform = ego_vehicle.get_transform()
            ego_location = ego_transform.location
            ego_yaw = float(ego_transform.rotation.yaw)
            ego_waypoint = world_map.get_waypoint(ego_location, project_to_road=True)
        except (RuntimeError, AttributeError):
            return []
        ego_lane = lane_ref_from_waypoint(ego_waypoint)
        self._record_ego_sample(frame_id, ego_vehicle, ego_lane)

        radius = (
            self.config.lane_relation_radius_m
            if self.config.lane_relation_radius_m is not None
            else self.config.nearby_actor_radius_m
        )
        states: list[NearbyActorState] = []
        for actor in world.get_actors():
            if actor.id == ego_vehicle.id or not self._is_alive(actor):
                continue
            type_id = str(getattr(actor, "type_id", ""))
            if type_id.startswith("sensor."):
                continue
            try:
                actor_transform = actor.get_transform()
                actor_location = actor_transform.location
                distance = float(ego_location.distance(actor_location))
                if distance > radius:
                    continue
                actor_yaw = float(actor_transform.rotation.yaw)
            except (RuntimeError, AttributeError):
                continue

            try:
                actor_waypoint = world_map.get_waypoint(actor_location, project_to_road=True)
            except RuntimeError:
                actor_waypoint = None
            actor_lane = lane_ref_from_waypoint(actor_waypoint)
            relation = classify_lane_relation(
                ego_lane,
                actor_lane,
                actor_yaw_deg=actor_yaw,
                ego_yaw_deg=ego_yaw,
                ego_waypoint=ego_waypoint,
                actor_waypoint=actor_waypoint,
                oncoming_opposition_deg=self.config.oncoming_opposition_deg,
            )
            in_front = self._is_in_front_of(ego_vehicle, actor)
            speed = self._velocity_mps(actor)
            lane_changed = detect_lane_change(self._actor_lanes.get(str(actor.id)), actor_lane)
            if actor_lane is not None:
                self._actor_lanes[str(actor.id)] = actor_lane
            track = self._record_actor_sample(frame_id, actor, actor_transform, actor_lane, speed)
            crossing_path, geometry_available = self._actor_crossing_path(
                actor_waypoint,
                ego_waypoint,
                track,
                tuple(self._ego_track),
                actor_is_vehicle="vehicle" in type_id,
            )
            oncoming = (
                relation.oncoming
                and actor_lane is not None
                and in_front
                and speed >= 0.1
            )
            states.append(
                NearbyActorState(
                    actor=actor,
                    entry={
                        "id": int(actor.id),
                        "type_id": type_id,
                        "distance_to_ego_m": distance,
                        "location": self._location_to_dict(actor_location),
                        "rotation": self._rotation_to_dict(actor_transform.rotation),
                        "velocity_mps": speed,
                        "same_road_as_ego": relation.same_road,
                        "same_lane_as_ego": relation.same_lane,
                        "is_in_front_of_ego": in_front,
                        "lane": actor_lane.to_dict() if actor_lane is not None else None,
                        "lane_relation": relation.relative_lane,
                        "lane_relation_reason": relation.reason,
                        "adjacent_lane_as_ego": relation.adjacent_lane,
                        "oncoming_as_ego": oncoming,
                        "waypoint_history": [sample.to_dict() for sample in track],
                    },
                    same_lane=relation.same_lane,
                    adjacent_lane=relation.adjacent_lane,
                    oncoming=oncoming,
                    lane_changed=lane_changed,
                    crossing_path=crossing_path,
                    crossing_geometry_available=geometry_available,
                )
            )
        states.sort(key=lambda state: state.entry["distance_to_ego_m"])
        return states

    def _record_ego_sample(self, frame_id: int, ego_vehicle: Any, ego_lane: LaneRef | None) -> None:
        try:
            transform = ego_vehicle.get_transform()
            location = transform.location
            speed = self._velocity_mps(ego_vehicle)
            yaw = float(transform.rotation.yaw)
        except (RuntimeError, AttributeError):
            return
        self._append_sample(
            self._ego_track,
            TrackSample(
                tick=frame_id,
                x=float(location.x),
                y=float(location.y),
                yaw_deg=yaw,
                velocity_mps=speed,
                road_id=ego_lane.road_id if ego_lane is not None else None,
                lane_id=ego_lane.lane_id if ego_lane is not None else None,
            ),
        )

    def _record_actor_sample(
        self,
        frame_id: int,
        actor: Any,
        actor_transform: Any,
        actor_lane: LaneRef | None,
        speed: float,
    ) -> tuple[TrackSample, ...]:
        key = str(actor.id)
        track = self._actor_tracks.get(key)
        if track is None:
            track = deque(maxlen=max(int(self.config.waypoint_history_ticks), 1))
            self._actor_tracks[key] = track
        location = actor_transform.location
        self._append_sample(
            track,
            TrackSample(
                tick=frame_id,
                x=float(location.x),
                y=float(location.y),
                yaw_deg=float(actor_transform.rotation.yaw),
                velocity_mps=float(speed),
                road_id=actor_lane.road_id if actor_lane is not None else None,
                lane_id=actor_lane.lane_id if actor_lane is not None else None,
            ),
        )
        return tuple(track)

    @staticmethod
    def _append_sample(track: deque[TrackSample], sample: TrackSample) -> None:
        if track and track[-1].tick == sample.tick:
            return
        track.append(sample)

    def _crossing_config(self) -> CrossingPathConfig:
        return CrossingPathConfig(
            corridor_half_width_m=self.config.crossing_corridor_half_width_m,
            min_actor_displacement_m=self.config.crossing_min_actor_displacement_m,
            ego_horizon_s=2.0,
            actor_horizon_s=2.0,
            step_s=0.2,
        )

    def _actor_crossing_path(
        self,
        actor_waypoint: Any,
        ego_waypoint: Any,
        actor_track: tuple[TrackSample, ...],
        ego_track: tuple[TrackSample, ...],
        *,
        actor_is_vehicle: bool,
    ) -> tuple[bool, bool]:
        """Return (crossing, geometry_available); geometry_available gates the proxy."""
        if len(actor_track) >= 2 and ego_track:
            if detect_crossing_from_tracks(
                actor_track,
                ego_track,
                actor_is_vehicle=actor_is_vehicle,
                config=self._crossing_config(),
            ):
                return True, True
        actor_path = waypoint_swept_path(
            actor_waypoint,
            horizon_m=self.config.crossing_actor_horizon_m,
            step_m=self.config.crossing_step_m,
        )
        ego_path = waypoint_swept_path(
            ego_waypoint,
            horizon_m=self.config.crossing_ego_horizon_m,
            step_m=self.config.crossing_step_m,
        )
        if len(actor_path) >= 2 and len(ego_path) >= 2:
            crossing = detect_path_crossing(
                actor_path,
                ego_path,
                corridor_half_width_m=self.config.crossing_corridor_half_width_m,
                min_actor_displacement_m=self.config.crossing_min_actor_displacement_m,
            )
            return crossing, True
        return False, False

    def _infer_lane_relation_predicates(self, states: list[NearbyActorState]) -> list[dict[str, Any]]:
        predicates: list[dict[str, Any]] = []
        for state in states:
            alias = self._actor_alias(state.actor)
            if state.entry["is_in_front_of_ego"]:
                predicates.append(self._make_fact("in_front_of", alias, "ego", 0.85))
            if state.entry["same_road_as_ego"]:
                predicates.append(self._make_fact("on_road", alias, None, 0.75))
            if state.same_lane:
                predicates.append(self._make_fact("same_lane", alias, "ego", 0.9))
            if state.adjacent_lane:
                predicates.append(self._make_fact("adjacent_lane", alias, "ego", 0.85))
            if state.oncoming and alias == "vehicle":
                predicates.append(self._make_fact("oncoming", alias, "ego", 0.85))
            if state.lane_changed and alias == "vehicle":
                predicates.append(self._make_fact("lane_changing", alias, None, 0.8))
            if state.crossing_path:
                predicates.append(self._make_fact("crossing_path", alias, "ego", 0.85))
            elif not state.crossing_geometry_available and alias in ("pedestrian", "vehicle"):
                entry = state.entry
                if entry["is_in_front_of_ego"] and entry["distance_to_ego_m"] <= 12.0:
                    predicates.append(self._make_fact("crossing_path", alias, "ego", 0.6))
            if state.same_lane and state.entry["is_in_front_of_ego"]:
                predicates.append(self._make_fact("obstructing", alias, "ego", 0.9))
        return predicates

    @staticmethod
    def _merge_predicates(predicates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for fact in predicates:
            signature = fact["signature"]
            if signature not in merged:
                merged[signature] = fact
                order.append(signature)
            elif fact["confidence"] > merged[signature]["confidence"]:
                merged[signature] = fact
        return [merged[signature] for signature in order]

    def _make_fact(self, predicate: str, subject: str, obj: str | None, confidence: float) -> dict[str, Any]:
        if obj is None:
            signature = f"{predicate}({subject})"
            entities = [subject]
        else:
            signature = f"{predicate}({subject},{obj})"
            entities = [subject, obj]
        return {
            "predicate": predicate,
            "signature": signature,
            "entities": entities,
            "confidence": confidence,
        }

    def _write_trace_dump(self, result: RunResult, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = output_dir / f"{result.scenario_id}-{result.agent_kind}-{timestamp}-semantic.json"
        out_path.write_text(json.dumps(self._trace, indent=2), encoding="utf-8")
        return out_path

    def _write_stream_dump(self, result: RunResult, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = output_dir / f"{result.scenario_id}-{result.agent_kind}-{timestamp}-semantic-stream.jsonl"
        with out_path.open("w", encoding="utf-8") as stream_file:
            for frame in self._stream_frames:
                stream_file.write(json.dumps(frame))
                stream_file.write("\n")
        return out_path

    def _write_scene_dumps(self, result: RunResult) -> list[str]:
        assert self.config.anomaly_dump_dir is not None
        self.config.anomaly_dump_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_paths: list[str] = []
        for index, dump in enumerate(self._scene_dumps):
            trigger = str(dump.get("trigger_reason", "scene-dump"))
            safe_trigger = "".join(char if char.isalnum() else "-" for char in trigger).strip("-").lower() or "scene-dump"
            tick = int(dump.get("tick", -1))
            out_path = self.config.anomaly_dump_dir / f"{result.scenario_id}-{result.agent_kind}-{timestamp}-tick{tick:04d}-{safe_trigger}-{index + 1}.json"
            out_path.write_text(json.dumps(dump, indent=2), encoding="utf-8")
            out_paths.append(str(out_path))
        return out_paths

    def _update_target_tick_capture(self, frame_id: int, context: dict[str, Any], telemetry: Any) -> None:
        if frame_id not in self.config.capture_ticks:
            return
        if frame_id in self._captured_target_ticks or not isinstance(telemetry, dict):
            return
        self._captured_target_ticks.add(frame_id)
        self._scene_dumps.append(self._capture_scene_dump(frame_id, f"target-tick-{frame_id}", context, telemetry))
        self._notes.append(f"Captured semantic scene dump at target tick {frame_id}.")

    @staticmethod
    def _normalize_route_waypoints(raw: Any) -> list[dict[str, float]]:
        """Coerce harness route context into compact ``{"x","y"}`` points.

        Accepts both ``get_route_waypoints()`` output (``{"x","y"}`` dicts) and
        the ``route_preview`` fallback shape (``{"index","location":{x,y,z}}``).
        Unusable entries are dropped so the persisted route stays well-formed.
        """
        if not isinstance(raw, (list, tuple)):
            return []
        points: list[dict[str, float]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            source = item
            if not isinstance(source.get("x"), (int, float)) and isinstance(
                source.get("location"), dict
            ):
                source = source["location"]
            try:
                x = float(source["x"])
                y = float(source["y"])
            except (KeyError, TypeError, ValueError):
                continue
            points.append({"x": x, "y": y})
        return points

    def _update_stream_capture(self, frame_id: int, context: dict[str, Any], telemetry: Any) -> None:
        if self.config.stream_output_dir is None or not isinstance(telemetry, dict):
            return
        every_ticks = max(int(self.config.stream_every_ticks), 1)
        if (frame_id - 1) % every_ticks != 0:
            return
        frame = self._capture_scene_dump(frame_id, None, context, telemetry)
        if not self._stream_frames and self._ego_route:
            # Persist the planned route exactly once, on the first stream frame.
            frame["ego_route"] = [dict(point) for point in self._ego_route]
        self._stream_frames.append(frame)

    def _update_anomaly_capture(self, frame_id: int, context: dict[str, Any], telemetry: Any) -> None:
        if self._anomaly_dump_index is not None or not isinstance(telemetry, dict):
            return

        control = telemetry.get("control", {})
        throttle = float(control.get("throttle", 0.0))
        brake = float(control.get("brake", 0.0))
        route_progress = telemetry.get("agent_step", {}).get("route_progress", {})
        heading_error = route_progress.get("heading_error_deg")

        if brake >= 0.95:
            self._persistent_brake_ticks += 1
        else:
            self._persistent_brake_ticks = 0

        trigger_reason = None
        if self._persistent_brake_ticks >= self.config.persistent_brake_ticks and throttle <= 0.05:
            trigger_reason = "persistent-brake"
        elif heading_error is not None and abs(float(heading_error)) >= self.config.heading_error_threshold_deg:
            trigger_reason = "heading-error"

        if trigger_reason is None:
            return

        self._scene_dumps.append(self._capture_scene_dump(frame_id, trigger_reason, context, telemetry))
        self._anomaly_dump_index = len(self._scene_dumps) - 1
        self._notes.append(f"Captured anomaly scene dump at tick {frame_id} due to {trigger_reason}.")

    def _capture_scene_dump(self, frame_id: int, trigger_reason: str | None, context: dict[str, Any], telemetry: dict[str, Any]) -> dict[str, Any]:
        world = context.get("world")
        ego_vehicle = context.get("ego_vehicle")
        scenario = context.get("scenario")
        if world is None or ego_vehicle is None or scenario is None:
            return {
                "schema_version": SEMANTIC_STREAM_SCHEMA_VERSION,
                "tick": frame_id,
                "trigger_reason": trigger_reason,
                "error": "World, ego vehicle, or scenario missing from context.",
            }

        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        if self._world_map is None:
            self._world_map = world.get_map()
        world_map = self._world_map
        ego_waypoint = world_map.get_waypoint(ego_location, project_to_road=True)
        ego_lane = lane_ref_from_waypoint(ego_waypoint)
        if self._scan_cache is not None and self._scan_cache[0] == frame_id:
            scan = self._scan_cache[1]
        else:
            scan = self._scan_nearby_actors(context, frame_id)
            self._scan_cache = (frame_id, scan)
        actors = [state.entry for state in scan]

        route_progress = telemetry.get("agent_step", {}).get("route_progress", {})
        return {
            "schema_version": SEMANTIC_STREAM_SCHEMA_VERSION,
            "tick": int(frame_id),
            "trigger_reason": trigger_reason,
            "scenario_id": scenario.scenario_id,
            "town": scenario.town,
            "harness_state": {
                "threshold_adversary_kind": context.get("threshold_adversary_kind"),
                "threshold_adversary_active": bool(context.get("threshold_adversary_active", False)),
                "threshold_adversary_triggered": bool(context.get("threshold_adversary_triggered", False)),
                "threshold_trigger_distance_m": context.get("threshold_trigger_distance_m"),
            },
            "ego": {
                "id": int(ego_vehicle.id),
                "location": self._location_to_dict(ego_location),
                "rotation": self._rotation_to_dict(ego_transform.rotation),
                "velocity_mps": self._velocity_mps(ego_vehicle),
                "waypoint": self._waypoint_to_dict(ego_waypoint),
                "lane": ego_lane.to_dict() if ego_lane is not None else None,
                "waypoint_history": [sample.to_dict() for sample in self._ego_track],
            },
            "telemetry": {
                "speed_mps": telemetry.get("speed_mps"),
                "speed_kph": telemetry.get("speed_kph"),
                "distance_to_goal_m": telemetry.get("distance_to_goal_m"),
                "control": telemetry.get("control"),
                "route_progress": route_progress,
            },
            "route_context": {
                "next_route_waypoint_location": route_progress.get("next_route_waypoint_location"),
                "nearest_route_waypoint_location": route_progress.get("nearest_route_waypoint_location"),
                "nearest_driving_waypoint_location": route_progress.get("nearest_driving_waypoint_location"),
                "nearest_driving_waypoint_yaw_deg": route_progress.get("nearest_driving_waypoint_yaw_deg"),
            },
            "nearby_actors": actors,
        }

    @staticmethod
    def _location_to_dict(location: Any) -> dict[str, float]:
        return {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        }

    @staticmethod
    def _rotation_to_dict(rotation: Any) -> dict[str, float]:
        return {
            "pitch": float(rotation.pitch),
            "yaw": float(rotation.yaw),
            "roll": float(rotation.roll),
        }

    @staticmethod
    def _velocity_mps(actor: Any) -> float:
        velocity = actor.get_velocity()
        return float(math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))

    @staticmethod
    def _waypoint_to_dict(waypoint: Any) -> dict[str, Any] | None:
        if waypoint is None:
            return None
        return {
            "road_id": int(waypoint.road_id),
            "lane_id": int(waypoint.lane_id),
            "s": float(waypoint.s),
            "is_junction": bool(waypoint.is_junction),
            "location": SemanticObserver._location_to_dict(waypoint.transform.location),
            "rotation": SemanticObserver._rotation_to_dict(waypoint.transform.rotation),
        }

    @staticmethod
    def _distance_between(actor_a: Any, actor_b: Any) -> float | None:
        if not SemanticObserver._is_alive(actor_a) or not SemanticObserver._is_alive(actor_b):
            return None
        return actor_a.get_location().distance(actor_b.get_location())

    @staticmethod
    def _is_in_front_of(reference_actor: Any, target_actor: Any) -> bool:
        if not SemanticObserver._is_alive(reference_actor) or not SemanticObserver._is_alive(target_actor):
            return False
        ref_transform = reference_actor.get_transform()
        ref_location = ref_transform.location
        target_location = target_actor.get_location()
        forward = ref_transform.get_forward_vector()
        delta_x = target_location.x - ref_location.x
        delta_y = target_location.y - ref_location.y
        dot = delta_x * forward.x + delta_y * forward.y
        return dot > 0.0

    @staticmethod
    def _actor_alias(actor: Any) -> str:
        type_id = getattr(actor, "type_id", "actor")
        if "pedestrian" in type_id:
            return "pedestrian"
        if "firetruck" in type_id or "vehicle" in type_id:
            return "vehicle"
        return type_id.split(".")[-1]

    @staticmethod
    def _is_alive(actor: Any) -> bool:
        try:
            return actor is not None and actor.is_alive
        except RuntimeError:
            return False
        except AttributeError:
            return actor is not None