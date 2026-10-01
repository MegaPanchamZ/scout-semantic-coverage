from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Callable


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.carla_utils import connect_client, ensure_output_dir, import_carla, load_world, resolve_weather
from research.harness.models import ScenarioSpec


DEFAULT_METADATA_PATH = WORKSPACE_ROOT / "datasets/Detection-of-Traffic-Anomaly/dataset/metadata_val.json"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "research/logs/scenarios/dota"


def _location_to_dict(location: Any) -> dict[str, float]:
    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(location.z),
    }


def _rotation_to_dict(rotation: Any) -> dict[str, float]:
    return {
        "pitch": float(rotation.pitch),
        "yaw": float(rotation.yaw),
        "roll": float(rotation.roll),
    }


def _sample_route(world_map: Any, start_location: Any, goal_location: Any, sampling_resolution: float) -> list[tuple[Any, Any]]:
    from research.harness.compat.agents.navigation.global_route_planner import GlobalRoutePlanner
    from research.harness.compat.agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

    planner = GlobalRoutePlanner(GlobalRoutePlannerDAO(world_map, sampling_resolution))
    return list(planner.trace_route(start_location, goal_location))


def _cumulative_distances(route: list[tuple[Any, Any]]) -> list[float]:
    totals: list[float] = [0.0]
    for index in range(1, len(route)):
        previous = route[index - 1][0].transform.location
        current = route[index][0].transform.location
        totals.append(totals[-1] + float(previous.distance(current)))
    return totals


def _find_trigger_location(route: list[tuple[Any, Any]], cumulative: list[float], anchor_index: int, trigger_lead_distance_m: float) -> Any:
    anchor_distance = cumulative[anchor_index]
    trigger_target = max(anchor_distance - trigger_lead_distance_m, 0.0)
    best_index = 0
    best_delta = abs(cumulative[0] - trigger_target)
    for index in range(anchor_index + 1):
        delta = abs(cumulative[index] - trigger_target)
        if delta < best_delta:
            best_index = index
            best_delta = delta
    return route[best_index][0].transform.location


@dataclass(slots=True)
class DotaClip:
    clip_id: str
    anomaly_class: str
    anomaly_start: int
    anomaly_end: int
    num_frames: int
    subset: str
    video_start: int
    video_end: int

    @property
    def anomaly_progress_ratio(self) -> float:
        if self.num_frames <= 1:
            return 0.5
        return min(max(float(self.anomaly_start) / float(self.num_frames - 1), 0.0), 1.0)

    @property
    def anomaly_duration_ratio(self) -> float:
        if self.num_frames <= 0:
            return 0.0
        duration = max(int(self.anomaly_end) - int(self.anomaly_start), 1)
        return min(max(float(duration) / float(self.num_frames), 0.0), 1.0)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slugify_label(value: str) -> str:
    return value.lower().replace(":", "_").replace(" ", "_")


def normalize_dota_class(raw_label: str) -> str:
    normalized = " ".join(str(raw_label).strip().lower().split())
    aliases = {
        "other:lateral": "other: lateral",
        "other: lateral": "other: lateral",
        "ego:leave_to_left": "ego: leave_to_left",
        "ego: leave_to_left": "ego: leave_to_left",
        "other:ahead_or_waiting": "other: ahead_or_waiting",
        "other: ahead_or_waiting": "other: ahead_or_waiting",
        "other:moving_ahead_or_waiting": "other: ahead_or_waiting",
        "other: moving_ahead_or_waiting": "other: ahead_or_waiting",
    }
    return aliases.get(normalized, normalized)


def supported_dota_classes() -> list[str]:
    return [
        "ego: leave_to_left",
        "other: ahead_or_waiting",
        "other: lateral",
    ]


def load_dota_metadata(path: Path) -> dict[str, DotaClip]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clips: dict[str, DotaClip] = {}
    for clip_id, item in payload.items():
        if not isinstance(item, dict):
            continue
        anomaly_class = normalize_dota_class(str(item.get("anomaly_class", "")))
        clips[str(clip_id)] = DotaClip(
            clip_id=str(clip_id),
            anomaly_class=anomaly_class,
            anomaly_start=int(item.get("anomaly_start") or 0),
            anomaly_end=int(item.get("anomaly_end") or 0),
            num_frames=int(item.get("num_frames") or 0),
            subset=str(item.get("subset") or "unknown"),
            video_start=int(item.get("video_start") or 0),
            video_end=int(item.get("video_end") or 0),
        )
    return clips


def select_dota_clip(metadata: dict[str, DotaClip], dota_class: str, clip_id: str | None = None) -> DotaClip:
    canonical_class = normalize_dota_class(dota_class)
    if clip_id is not None:
        selected = metadata.get(clip_id)
        if selected is None:
            raise KeyError(f"DoTA clip '{clip_id}' was not found in the metadata file.")
        if selected.anomaly_class != canonical_class:
            raise ValueError(
                f"DoTA clip '{clip_id}' has class '{selected.anomaly_class}', not requested class '{canonical_class}'."
            )
        return selected

    matches = sorted((clip for clip in metadata.values() if clip.anomaly_class == canonical_class), key=lambda clip: clip.clip_id)
    if not matches:
        available = ", ".join(sorted({clip.anomaly_class for clip in metadata.values()}))
        raise ValueError(f"No DoTA clips matched class '{canonical_class}'. Available classes include: {available}")
    return matches[0]


def _select_anchor_index(
    route: list[tuple[Any, Any]],
    cumulative: list[float],
    *,
    min_distance_from_start_m: float,
    min_distance_to_goal_m: float,
    target_progress_ratio: float,
    predicate: Callable[[Any], bool],
) -> tuple[int, Any]:
    if not route:
        raise RuntimeError("The sampled route is empty.")
    total_distance = cumulative[-1] if cumulative else 0.0
    target_distance = total_distance * target_progress_ratio
    candidates: list[tuple[float, int, Any]] = []
    fallback: list[tuple[float, int, Any]] = []
    for index, (waypoint, _road_option) in enumerate(route):
        remaining = total_distance - cumulative[index]
        if cumulative[index] < min_distance_from_start_m:
            continue
        if remaining < min_distance_to_goal_m:
            continue
        distance_delta = abs(cumulative[index] - target_distance)
        record = (distance_delta, index, waypoint)
        fallback.append(record)
        if predicate(waypoint):
            candidates.append(record)
    pool = candidates or fallback
    if not pool:
        raise RuntimeError("No eligible route anchor was found for the requested DoTA mapping.")
    _, anchor_index, anchor_waypoint = min(pool, key=lambda item: item[0])
    return anchor_index, anchor_waypoint


def _build_generator_metadata(
    *,
    clip: DotaClip,
    dota_class: str,
    route: list[tuple[Any, Any]],
    cumulative: list[float],
    anchor_index: int,
    anchor_waypoint: Any,
    trigger_location: Any,
    start_transform: Any,
    goal_transform: Any,
    intent: str,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "dota_to_scenario",
        "source": {
            "dataset": "DoTA",
            "clip_id": clip.clip_id,
            "subset": clip.subset,
            "anomaly_class": dota_class,
            "video_start": clip.video_start,
            "video_end": clip.video_end,
            "anomaly_start": clip.anomaly_start,
            "anomaly_end": clip.anomaly_end,
            "num_frames": clip.num_frames,
            "anomaly_progress_ratio": clip.anomaly_progress_ratio,
            "anomaly_duration_ratio": clip.anomaly_duration_ratio,
        },
        "mapping": {
            "intent": intent,
            "route_waypoint_count": len(route),
            "route_length_m": cumulative[-1] if cumulative else 0.0,
            "route_anchor_index": int(anchor_index),
            "route_anchor_location": _location_to_dict(anchor_waypoint.transform.location),
            "route_anchor_is_junction": bool(getattr(anchor_waypoint, "is_junction", False)),
            "trigger_location": _location_to_dict(trigger_location),
            "ego_spawn_location": _location_to_dict(start_transform.location),
            "goal_spawn_location": _location_to_dict(goal_transform.location),
        },
    }


def _build_threshold_crossing_spec(
    *,
    clip: DotaClip,
    town: str,
    weather_preset: str,
    ego_spawn_index: int,
    goal_spawn_index: int,
    anchor_index: int,
    anchor_waypoint: Any,
    trigger_location: Any,
    trigger_radius_m: float,
    walker_speed: float,
    lateral_offset_multiplier: float,
    max_ticks: int,
) -> ScenarioSpec:
    carla = import_carla()

    lane_width = max(float(getattr(anchor_waypoint, "lane_width", 3.5)), 3.5)
    anchor_transform = anchor_waypoint.transform
    right_vector = anchor_transform.get_right_vector()
    crossing_distance = lane_width * lateral_offset_multiplier
    spawn_location = anchor_transform.location + carla.Location(
        x=right_vector.x * crossing_distance,
        y=right_vector.y * crossing_distance,
        z=0.8,
    )
    destination_location = anchor_transform.location + carla.Location(
        x=-right_vector.x * crossing_distance,
        y=-right_vector.y * crossing_distance,
        z=0.8,
    )
    return ScenarioSpec(
        scenario_id=f"{town.lower()}_dota_{clip.clip_id}_{_slugify_label(clip.anomaly_class)}",
        town=town,
        weather_preset=weather_preset,
        ego_spawn_index=ego_spawn_index,
        goal_spawn_index=goal_spawn_index,
        description=(
            f"DoTA-seeded threshold crossing derived from {clip.clip_id} ({clip.anomaly_class}); "
            f"a walker crosses near route waypoint {anchor_index} when the ego enters the trigger zone."
        ),
        max_ticks=max_ticks,
        walker_count=1,
        controller="threshold_crossing_adversary",
        controller_params={
            "adversary_kind": "walker",
            "blueprint_filter": "walker.pedestrian.*",
            "speed": walker_speed,
            "trigger_radius_m": trigger_radius_m,
            "trigger_location": _location_to_dict(trigger_location),
            "spawn_transform": {
                "location": _location_to_dict(spawn_location),
                "rotation": _rotation_to_dict(anchor_transform.rotation),
            },
            "destination_location": _location_to_dict(destination_location),
            "subgraph_type": "dota-other-lateral-threshold-crossing",
            "route_anchor_index": int(anchor_index),
            "route_anchor_location": _location_to_dict(anchor_transform.location),
            "route_anchor_is_junction": bool(getattr(anchor_waypoint, "is_junction", False)),
        },
    )


def _build_lane_departure_spec(
    *,
    clip: DotaClip,
    town: str,
    weather_preset: str,
    ego_spawn_index: int,
    goal_spawn_index: int,
    anchor_index: int,
    anchor_waypoint: Any,
    trigger_location: Any,
    trigger_radius_m: float,
    lateral_offset_multiplier: float,
    max_ticks: int,
) -> ScenarioSpec:
    carla = import_carla()

    lane_width = max(float(getattr(anchor_waypoint, "lane_width", 3.5)), 3.5)
    anchor_transform = anchor_waypoint.transform
    right_vector = anchor_transform.get_right_vector()
    staging_location = anchor_transform.location + carla.Location(
        x=right_vector.x * lane_width * lateral_offset_multiplier,
        y=right_vector.y * lane_width * lateral_offset_multiplier,
        z=0.3,
    )
    active_location = anchor_transform.location + carla.Location(
        x=right_vector.x * lane_width * 0.35,
        y=right_vector.y * lane_width * 0.35,
        z=0.3,
    )
    return ScenarioSpec(
        scenario_id=f"{town.lower()}_dota_{clip.clip_id}_{_slugify_label(clip.anomaly_class)}",
        town=town,
        weather_preset=weather_preset,
        ego_spawn_index=ego_spawn_index,
        goal_spawn_index=goal_spawn_index,
        description=(
            f"DoTA-seeded lane-departure adversary derived from {clip.clip_id} ({clip.anomaly_class}); "
            f"a blocking vehicle intrudes into the ego lane near waypoint {anchor_index}."
        ),
        max_ticks=max_ticks,
        npc_vehicle_count=1,
        controller="lane_departure_adversary",
        controller_params={
            "blueprint_filter": "vehicle.*",
            "trigger_radius_m": trigger_radius_m,
            "trigger_location": _location_to_dict(trigger_location),
            "staging_transform": {
                "location": _location_to_dict(staging_location),
                "rotation": _rotation_to_dict(anchor_transform.rotation),
            },
            "active_transform": {
                "location": _location_to_dict(active_location),
                "rotation": _rotation_to_dict(anchor_transform.rotation),
            },
            "subgraph_type": "dota-ego-leave-to-left-lane-departure",
            "route_anchor_index": int(anchor_index),
            "route_anchor_location": _location_to_dict(anchor_transform.location),
            "route_anchor_is_junction": bool(getattr(anchor_waypoint, "is_junction", False)),
        },
    )


def _build_lead_vehicle_braking_spec(
    *,
    clip: DotaClip,
    town: str,
    weather_preset: str,
    ego_spawn_index: int,
    goal_spawn_index: int,
    anchor_index: int,
    anchor_waypoint: Any,
    trigger_location: Any,
    trigger_radius_m: float,
    pre_brake_throttle: float,
    max_ticks: int,
) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=f"{town.lower()}_dota_{clip.clip_id}_{_slugify_label(clip.anomaly_class)}",
        town=town,
        weather_preset=weather_preset,
        ego_spawn_index=ego_spawn_index,
        goal_spawn_index=goal_spawn_index,
        description=(
            f"DoTA-seeded lead-vehicle braking adversary derived from {clip.clip_id} ({clip.anomaly_class}); "
            f"a same-lane vehicle ahead brakes near waypoint {anchor_index}."
        ),
        max_ticks=max_ticks,
        npc_vehicle_count=1,
        controller="lead_vehicle_braking",
        controller_params={
            "blueprint_filter": "vehicle.*",
            "trigger_radius_m": trigger_radius_m,
            "trigger_location": _location_to_dict(trigger_location),
            "spawn_transform": {
                "location": _location_to_dict(anchor_waypoint.transform.location),
                "rotation": _rotation_to_dict(anchor_waypoint.transform.rotation),
            },
            "pre_brake_throttle": pre_brake_throttle,
            "post_trigger_brake": 1.0,
            "subgraph_type": "dota-other-ahead-or-waiting-lead-vehicle-braking",
            "route_anchor_index": int(anchor_index),
            "route_anchor_location": _location_to_dict(anchor_waypoint.transform.location),
            "route_anchor_is_junction": bool(getattr(anchor_waypoint, "is_junction", False)),
        },
    )


def build_dota_seeded_scenario(
    *,
    metadata_path: Path,
    dota_class: str,
    clip_id: str | None,
    host: str,
    port: int,
    timeout_seconds: float,
    town: str,
    weather_preset: str,
    ego_spawn_index: int,
    goal_spawn_index: int,
    sampling_resolution: float,
    min_distance_from_start_m: float,
    min_distance_to_goal_m: float,
    trigger_lead_distance_m: float,
    trigger_radius_m: float,
    walker_speed: float,
    lateral_offset_multiplier: float,
    pre_brake_throttle: float,
    max_ticks: int,
) -> dict[str, Any]:
    metadata = load_dota_metadata(metadata_path)
    clip = select_dota_clip(metadata, dota_class=dota_class, clip_id=clip_id)
    canonical_class = normalize_dota_class(clip.anomaly_class)
    if canonical_class not in supported_dota_classes():
        raise ValueError(
            f"DoTA class '{canonical_class}' is not yet mapped. Supported classes: {', '.join(supported_dota_classes())}"
        )

    client = connect_client(host, port, timeout_seconds)
    world = load_world(client, town)
    resolve_weather(world, weather_preset)

    spawn_points = world.get_map().get_spawn_points()
    start_transform = spawn_points[ego_spawn_index]
    goal_transform = spawn_points[goal_spawn_index]
    route = _sample_route(world.get_map(), start_transform.location, goal_transform.location, sampling_resolution)
    cumulative = _cumulative_distances(route)

    target_progress_ratio = clip.anomaly_progress_ratio
    if canonical_class == "other: lateral":
        anchor_index, anchor_waypoint = _select_anchor_index(
            route,
            cumulative,
            min_distance_from_start_m=min_distance_from_start_m,
            min_distance_to_goal_m=min_distance_to_goal_m,
            target_progress_ratio=target_progress_ratio,
            predicate=lambda waypoint: bool(getattr(waypoint, "is_junction", False)),
        )
        trigger_location = _find_trigger_location(
            route,
            cumulative,
            anchor_index,
            trigger_lead_distance_m=trigger_lead_distance_m,
        )
        scenario = _build_threshold_crossing_spec(
            clip=clip,
            town=town,
            weather_preset=weather_preset,
            ego_spawn_index=ego_spawn_index,
            goal_spawn_index=goal_spawn_index,
            anchor_index=anchor_index,
            anchor_waypoint=anchor_waypoint,
            trigger_location=trigger_location,
            trigger_radius_m=trigger_radius_m,
            walker_speed=walker_speed,
            lateral_offset_multiplier=lateral_offset_multiplier,
            max_ticks=max_ticks,
        )
        intent = "ThresholdCrossingAdversary"
    elif canonical_class == "ego: leave_to_left":
        anchor_index, anchor_waypoint = _select_anchor_index(
            route,
            cumulative,
            min_distance_from_start_m=min_distance_from_start_m,
            min_distance_to_goal_m=min_distance_to_goal_m,
            target_progress_ratio=target_progress_ratio,
            predicate=lambda waypoint: not bool(getattr(waypoint, "is_junction", False)),
        )
        trigger_location = _find_trigger_location(
            route,
            cumulative,
            anchor_index,
            trigger_lead_distance_m=trigger_lead_distance_m,
        )
        scenario = _build_lane_departure_spec(
            clip=clip,
            town=town,
            weather_preset=weather_preset,
            ego_spawn_index=ego_spawn_index,
            goal_spawn_index=goal_spawn_index,
            anchor_index=anchor_index,
            anchor_waypoint=anchor_waypoint,
            trigger_location=trigger_location,
            trigger_radius_m=trigger_radius_m,
            lateral_offset_multiplier=lateral_offset_multiplier,
            max_ticks=max_ticks,
        )
        intent = "LaneDepartureAdversary"
    else:
        anchor_index, anchor_waypoint = _select_anchor_index(
            route,
            cumulative,
            min_distance_from_start_m=min_distance_from_start_m,
            min_distance_to_goal_m=min_distance_to_goal_m,
            target_progress_ratio=target_progress_ratio,
            predicate=lambda waypoint: not bool(getattr(waypoint, "is_junction", False)),
        )
        trigger_location = _find_trigger_location(
            route,
            cumulative,
            anchor_index,
            trigger_lead_distance_m=trigger_lead_distance_m,
        )
        scenario = _build_lead_vehicle_braking_spec(
            clip=clip,
            town=town,
            weather_preset=weather_preset,
            ego_spawn_index=ego_spawn_index,
            goal_spawn_index=goal_spawn_index,
            anchor_index=anchor_index,
            anchor_waypoint=anchor_waypoint,
            trigger_location=trigger_location,
            trigger_radius_m=trigger_radius_m,
            pre_brake_throttle=pre_brake_throttle,
            max_ticks=max_ticks,
        )
        intent = "LeadVehicleBraking"

    payload = scenario.to_dict()
    payload["generator_metadata"] = _build_generator_metadata(
        clip=clip,
        dota_class=canonical_class,
        route=route,
        cumulative=cumulative,
        anchor_index=anchor_index,
        anchor_waypoint=anchor_waypoint,
        trigger_location=trigger_location,
        start_transform=start_transform,
        goal_transform=goal_transform,
        intent=intent,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a DoTA anomaly class into a CARLA ScenarioSpec JSON seed.")
    parser.add_argument("--dota-class", required=False, default=None)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--town", default="Town01")
    parser.add_argument("--weather-preset", default="ClearNoon")
    parser.add_argument("--ego-spawn-index", type=int, default=0)
    parser.add_argument("--goal-spawn-index", type=int, default=82)
    parser.add_argument("--sampling-resolution", type=float, default=2.0)
    parser.add_argument("--min-distance-from-start-m", type=float, default=20.0)
    parser.add_argument("--min-distance-to-goal-m", type=float, default=20.0)
    parser.add_argument("--trigger-lead-distance-m", type=float, default=12.0)
    parser.add_argument("--trigger-radius-m", type=float, default=8.0)
    parser.add_argument("--walker-speed", type=float, default=1.8)
    parser.add_argument("--lateral-offset-multiplier", type=float, default=1.75)
    parser.add_argument("--pre-brake-throttle", type=float, default=0.3)
    parser.add_argument("--max-ticks", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--list-supported-classes", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list_supported_classes:
        print(json.dumps({"supported_dota_classes": supported_dota_classes()}, indent=2))
        return
    if args.dota_class is None:
        raise SystemExit("--dota-class is required unless --list-supported-classes is used.")

    payload = build_dota_seeded_scenario(
        metadata_path=args.metadata_json,
        dota_class=args.dota_class,
        clip_id=args.clip_id,
        host=args.host,
        port=args.port,
        timeout_seconds=args.timeout_seconds,
        town=args.town,
        weather_preset=args.weather_preset,
        ego_spawn_index=args.ego_spawn_index,
        goal_spawn_index=args.goal_spawn_index,
        sampling_resolution=args.sampling_resolution,
        min_distance_from_start_m=args.min_distance_from_start_m,
        min_distance_to_goal_m=args.min_distance_to_goal_m,
        trigger_lead_distance_m=args.trigger_lead_distance_m,
        trigger_radius_m=args.trigger_radius_m,
        walker_speed=args.walker_speed,
        lateral_offset_multiplier=args.lateral_offset_multiplier,
        pre_brake_throttle=args.pre_brake_throttle,
        max_ticks=args.max_ticks,
    )

    output_path = args.output
    if output_path is None:
        clip_id = str(payload.get("generator_metadata", {}).get("source", {}).get("clip_id") or "unknown")
        label = _slugify_label(str(payload.get("generator_metadata", {}).get("source", {}).get("anomaly_class") or args.dota_class))
        output_path = DEFAULT_OUTPUT_DIR / f"{clip_id}-{label}-{_utc_timestamp()}.json"
    ensure_output_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()