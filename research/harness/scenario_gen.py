from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.carla_utils import connect_client, ensure_output_dir, import_carla, load_world, resolve_weather
from research.harness.compat.agents.navigation.global_route_planner import GlobalRoutePlanner
from research.harness.compat.agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from research.harness.models import ScenarioSpec


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
    planner = GlobalRoutePlanner(GlobalRoutePlannerDAO(world_map, sampling_resolution))
    return list(planner.trace_route(start_location, goal_location))


def _cumulative_distances(route: list[tuple[Any, Any]]) -> list[float]:
    totals: list[float] = [0.0]
    for index in range(1, len(route)):
        previous = route[index - 1][0].transform.location
        current = route[index][0].transform.location
        totals.append(totals[-1] + float(previous.distance(current)))
    return totals


def _find_junction_anchor(route: list[tuple[Any, Any]], min_distance_from_start_m: float, min_distance_to_goal_m: float) -> tuple[int, Any, list[float]]:
    if len(route) < 4:
        raise RuntimeError("Route is too short to place an adversarial crossing scenario.")

    cumulative = _cumulative_distances(route)
    total_distance = cumulative[-1]
    for index, (waypoint, _road_option) in enumerate(route):
        if cumulative[index] < min_distance_from_start_m:
            continue
        if total_distance - cumulative[index] < min_distance_to_goal_m:
            continue
        if bool(getattr(waypoint, "is_junction", False)):
            return index, waypoint, cumulative

    for index, (waypoint, _road_option) in enumerate(route):
        if bool(getattr(waypoint, "is_junction", False)):
            return index, waypoint, cumulative
    raise RuntimeError("No junction waypoint was found along the selected route.")


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


def _build_walker_crossing_spec(
    scenario_id: str,
    town: str,
    weather_preset: str,
    ego_spawn_index: int,
    goal_spawn_index: int,
    route: list[tuple[Any, Any]],
    anchor_index: int,
    anchor_waypoint: Any,
    trigger_location: Any,
    trigger_radius_m: float,
    walker_speed: float,
    lateral_offset_multiplier: float,
    max_ticks: int,
) -> ScenarioSpec:
    carla = import_carla()
    anchor_transform = anchor_waypoint.transform
    lane_width = max(float(getattr(anchor_waypoint, "lane_width", 3.5)), 3.5)
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
    description = (
        f"Generated threshold-triggered walker crossing on {town} near route waypoint {anchor_index}; "
        f"walker begins crossing when ego enters a {trigger_radius_m:.1f} m trigger zone."
    )
    return ScenarioSpec(
        scenario_id=scenario_id,
        town=town,
        weather_preset=weather_preset,
        ego_spawn_index=ego_spawn_index,
        goal_spawn_index=goal_spawn_index,
        description=description,
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
            "subgraph_type": "junction-crossing",
            "route_anchor_index": int(anchor_index),
            "route_anchor_location": _location_to_dict(anchor_transform.location),
            "route_anchor_is_junction": bool(getattr(anchor_waypoint, "is_junction", False)),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Scenic-style asset-placement scenario for the CARLA harness.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--town", default="Town01")
    parser.add_argument("--weather-preset", default="ClearNoon")
    parser.add_argument("--ego-spawn-index", type=int, default=0)
    parser.add_argument("--goal-spawn-index", type=int, default=82)
    parser.add_argument("--sampling-resolution", type=float, default=2.0)
    parser.add_argument("--trigger-lead-distance-m", type=float, default=12.0)
    parser.add_argument("--trigger-radius-m", type=float, default=8.0)
    parser.add_argument("--min-distance-from-start-m", type=float, default=20.0)
    parser.add_argument("--min-distance-to-goal-m", type=float, default=20.0)
    parser.add_argument("--walker-speed", type=float, default=1.8)
    parser.add_argument("--lateral-offset-multiplier", type=float, default=1.75)
    parser.add_argument("--max-ticks", type=int, default=500)
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--output", type=Path, default=Path("research/logs/scenarios/generated-threshold-crossing.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    client = connect_client(args.host, args.port, args.timeout_seconds)
    world = load_world(client, args.town)
    resolve_weather(world, args.weather_preset)

    spawn_points = world.get_map().get_spawn_points()
    start_transform = spawn_points[args.ego_spawn_index]
    goal_transform = spawn_points[args.goal_spawn_index]
    route = _sample_route(
        world.get_map(),
        start_transform.location,
        goal_transform.location,
        args.sampling_resolution,
    )
    anchor_index, anchor_waypoint, cumulative = _find_junction_anchor(
        route,
        min_distance_from_start_m=args.min_distance_from_start_m,
        min_distance_to_goal_m=args.min_distance_to_goal_m,
    )
    trigger_location = _find_trigger_location(
        route,
        cumulative,
        anchor_index,
        trigger_lead_distance_m=args.trigger_lead_distance_m,
    )

    scenario_id = args.scenario_id or (
        f"{args.town.lower()}_spawn{args.ego_spawn_index}_goal{args.goal_spawn_index}_threshold_crossing"
    )
    scenario = _build_walker_crossing_spec(
        scenario_id=scenario_id,
        town=args.town,
        weather_preset=args.weather_preset,
        ego_spawn_index=args.ego_spawn_index,
        goal_spawn_index=args.goal_spawn_index,
        route=route,
        anchor_index=anchor_index,
        anchor_waypoint=anchor_waypoint,
        trigger_location=trigger_location,
        trigger_radius_m=args.trigger_radius_m,
        walker_speed=args.walker_speed,
        lateral_offset_multiplier=args.lateral_offset_multiplier,
        max_ticks=args.max_ticks,
    )

    payload = scenario.to_dict()
    payload["generator_metadata"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "town": args.town,
        "weather_preset": args.weather_preset,
        "route_waypoint_count": len(route),
        "route_length_m": cumulative[-1] if cumulative else 0.0,
        "junction_anchor_index": int(anchor_index),
        "junction_anchor_location": _location_to_dict(anchor_waypoint.transform.location),
        "trigger_location": _location_to_dict(trigger_location),
        "ego_spawn_location": _location_to_dict(start_transform.location),
        "goal_spawn_location": _location_to_dict(goal_transform.location),
    }

    output_path = args.output
    ensure_output_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()