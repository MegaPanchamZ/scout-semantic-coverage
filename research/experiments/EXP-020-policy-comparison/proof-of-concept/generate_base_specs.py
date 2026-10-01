"""Generate one threshold-crossing walker base scenario spec per verified route.

This reproduces the base-scenario template used by the EXP-012 falsification
experiments (see
research/logs/falsification/town01_spawn0_goal82_threshold_crossing-exp-012-overnight-20260315T015916Z/base-scenario.json)
for every route in a verified route list.

The adversary is anchored on the *actual* PCLA route that the ego will drive:
the route is sampled with the PCLA route planner
(``pcla_functions.location_to_waypoint``), the anchor is the first junction
waypoint at least ``--min-distance-from-start-m`` from the start and
``--min-distance-to-goal-m`` from the goal, and the walker crosses
``--lateral-offset-multiplier * lane_width`` to either side of the anchor lane.
The trigger location sits ``--trigger-lead-distance-m`` before the anchor; the
threshold-crossing adversary activates its walker once the ego enters the
``--trigger-radius-m`` trigger zone.

Usage:

    research/.venv/bin/python research/experiments/EXP-020-policy-comparison/proof-of-concept/generate_base_specs.py \
        --port 2510 --routes-file <routes.json> \
        --output-dir research/experiments/EXP-020-policy-comparison/artifacts/base_specs

``routes.json`` is a list of ``{route_id, town, ego_spawn_index, goal_spawn_index}``
objects (``artifacts/route_corpus.json`` is also accepted).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.carla_utils import connect_client, ensure_output_dir, import_carla, load_world, resolve_weather
from research.harness.pcla_bridge import ensure_pcla_repo_path


def _location_to_dict(location: Any) -> dict[str, float]:
    return {"x": float(location.x), "y": float(location.y), "z": float(location.z)}


def _rotation_to_dict(rotation: Any) -> dict[str, float]:
    return {
        "pitch": float(rotation.pitch),
        "yaw": float(rotation.yaw),
        "roll": float(rotation.roll),
    }


def _sample_route_with_pcla(client: Any, start_location: Any, goal_location: Any, sampling_resolution: float) -> list[Any]:
    from pcla_functions import location_to_waypoint

    return list(location_to_waypoint(client, start_location, goal_location, distance=sampling_resolution))


class _OfflineMapWorld:
    def __init__(self, world_map: Any) -> None:
        self._world_map = world_map

    def get_map(self) -> Any:
        return self._world_map


class _OfflineClient:
    """Client shim that serves a carla.Map built from the shipped .xodr file.

    Used when no live CARLA server is available; the PCLA route planner only
    needs ``client.get_world().get_map()`` to build its graph.
    """

    def __init__(self, world_map: Any) -> None:
        self._world = _OfflineMapWorld(world_map)

    def get_world(self) -> Any:
        return self._world


def _load_offline_spawns(spawns_dir: Path, town: str) -> list[dict[str, Any]]:
    payload = json.loads((spawns_dir / f"spawns_{town}.json").read_text(encoding="utf-8"))
    return payload


def _cumulative_distances(route: list[Any]) -> list[float]:
    totals: list[float] = [0.0]
    for index in range(1, len(route)):
        previous = route[index - 1].transform.location
        current = route[index].transform.location
        totals.append(totals[-1] + float(previous.distance(current)))
    return totals


def _find_anchor(route: list[Any], cumulative: list[float], min_from_start_m: float, min_to_goal_m: float) -> tuple[int, bool]:
    if len(route) < 4:
        raise RuntimeError("Route is too short to place an adversarial crossing scenario.")

    total_distance = cumulative[-1]
    for index, waypoint in enumerate(route):
        if cumulative[index] < min_from_start_m:
            continue
        if total_distance - cumulative[index] < min_to_goal_m:
            continue
        if bool(getattr(waypoint, "is_junction", False)):
            return index, True

    for index, waypoint in enumerate(route):
        if bool(getattr(waypoint, "is_junction", False)):
            return index, True

    # Mid-route fallback anchor: closest waypoint to the halfway mark.
    target = total_distance * 0.5
    best_index = min(range(len(route)), key=lambda index: abs(cumulative[index] - target))
    return best_index, False


def _find_trigger_location(route: list[Any], cumulative: list[float], anchor_index: int, trigger_lead_distance_m: float) -> Any:
    anchor_distance = cumulative[anchor_index]
    trigger_target = max(anchor_distance - trigger_lead_distance_m, 0.0)
    best_index = 0
    best_delta = abs(cumulative[0] - trigger_target)
    for index in range(anchor_index + 1):
        delta = abs(cumulative[index] - trigger_target)
        if delta < best_delta:
            best_index = index
            best_delta = delta
    return route[best_index].transform.location


def _effective_route_length(route: list[Any], cumulative: list[float], goal_location: Any) -> float:
    """Route length up to the point where the agent's goal predicate fires.

    The PCLA planner can append a long tail (or loop) after the goal for some
    start/goal pairs, which inflates the raw cumulative length; the agent stops
    once it is within 5 m of the goal (``PclaAdapter.done``), so the descriptive
    length is measured at the first waypoint inside that radius (closest
    approach as fallback).
    """
    for index, waypoint in enumerate(route):
        if float(waypoint.transform.location.distance(goal_location)) <= 5.0:
            return cumulative[index]
    goal_index = min(range(len(route)), key=lambda index: route[index].transform.location.distance(goal_location))
    return cumulative[goal_index]


def build_base_spec(
    *,
    route_entry: dict[str, Any],
    route: list[Any],
    cumulative: list[float],
    effective_length_m: float,
    anchor_index: int,
    anchor_is_junction: bool,
    trigger_location: Any,
    weather_preset: str,
    max_ticks: int,
    trigger_radius_m: float,
    walker_speed: float,
    lateral_offset_multiplier: float,
) -> dict[str, Any]:
    carla = import_carla()

    anchor_waypoint = route[anchor_index]
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

    route_id = str(route_entry["route_id"])
    town = str(route_entry["town"])
    subgraph_type = "junction-crossing" if anchor_is_junction else "mid-route-crossing"
    return {
        "scenario_id": f"{route_id}_threshold_crossing",
        "town": town,
        "weather_preset": weather_preset,
        "ego_spawn_index": int(route_entry["ego_spawn_index"]),
        "goal_spawn_index": int(route_entry["goal_spawn_index"]),
        "description": (
            f"Generated threshold-triggered walker crossing on {town} near route waypoint {anchor_index} "
            f"(route {route_id}, {effective_length_m:.1f} m); walker begins crossing when ego enters a "
            f"{trigger_radius_m:.1f} m trigger zone."
        ),
        "max_ticks": max_ticks,
        "npc_vehicle_count": 0,
        "walker_count": 1,
        "controller": "threshold_crossing_adversary",
        "controller_params": {
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
            "subgraph_type": subgraph_type,
            "route_anchor_index": int(anchor_index),
            "route_anchor_location": _location_to_dict(anchor_transform.location),
            "route_anchor_is_junction": bool(anchor_is_junction),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate threshold-crossing base scenario specs for verified routes.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2510)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--routes-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Build routes against a client-side carla.Map from the shipped .xodr (no CARLA server needed).",
    )
    parser.add_argument(
        "--carla-root",
        type=Path,
        default=Path("/mnt/DevDrive/carla-0.9.16"),
        help="CARLA root, used to locate OpenDrive/TownXX.xodr in offline mode.",
    )
    parser.add_argument(
        "--offline-spawns-dir",
        type=Path,
        default=None,
        help="Directory with spawns_<Town>.json used in offline mode.",
    )
    parser.add_argument("--weather-preset", default="ClearNoon")
    parser.add_argument("--max-ticks", type=int, default=500)
    parser.add_argument("--sampling-resolution", type=float, default=2.0)
    parser.add_argument("--min-distance-from-start-m", type=float, default=20.0)
    parser.add_argument("--min-distance-to-goal-m", type=float, default=20.0)
    parser.add_argument("--trigger-lead-distance-m", type=float, default=12.0)
    parser.add_argument("--trigger-radius-m", type=float, default=20.0)
    parser.add_argument("--walker-speed", type=float, default=1.8)
    parser.add_argument("--lateral-offset-multiplier", type=float, default=1.75)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = json.loads(args.routes_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        routes = payload.get("routes") or payload.get("candidates")
    else:
        routes = payload

    ensure_pcla_repo_path(None)
    ensure_output_dir(args.output_dir)
    carla = import_carla()

    if args.offline:
        if args.offline_spawns_dir is None:
            raise SystemExit("--offline requires --offline-spawns-dir.")
        xodr_dir = args.carla_root / "CarlaUE4/Content/Carla/Maps/OpenDrive"
        client = None
    else:
        client = connect_client(args.host, args.port, args.timeout_seconds)

    current_town: str | None = None
    loaded_world = None
    loaded_spawn_points = None
    summary: list[dict[str, Any]] = []
    for route_entry in routes:
        town = str(route_entry["town"])
        if town != current_town:
            if args.offline:
                world_map = carla.Map(town, (xodr_dir / f"{town}.xodr").read_text(encoding="utf-8"))
                client = _OfflineClient(world_map)
                loaded_spawn_points = [
                    carla.Transform(
                        carla.Location(x=float(sp["x"]), y=float(sp["y"]), z=float(sp["z"])),
                        carla.Rotation(
                            pitch=float(sp.get("pitch", 0.0)),
                            yaw=float(sp.get("yaw", 0.0)),
                            roll=float(sp.get("roll", 0.0)),
                        ),
                    )
                    for sp in _load_offline_spawns(args.offline_spawns_dir, town)
                ]
            else:
                loaded_world = load_world(client, town, load_timeout_seconds=240.0)
                resolve_weather(loaded_world, args.weather_preset)
                loaded_spawn_points = loaded_world.get_map().get_spawn_points()
            current_town = town

        spawn_points = loaded_spawn_points
        start_location = spawn_points[int(route_entry["ego_spawn_index"])].location
        goal_location = spawn_points[int(route_entry["goal_spawn_index"])].location
        route = _sample_route_with_pcla(client, start_location, goal_location, args.sampling_resolution)
        if len(route) < 4:
            raise RuntimeError(f"PCLA route for {route_entry['route_id']} is too short ({len(route)} waypoints).")
        cumulative = _cumulative_distances(route)
        if not (args.min_distance_from_start_m <= cumulative[-1] - args.min_distance_to_goal_m):
            raise RuntimeError(f"Route {route_entry['route_id']} is too short for the requested anchor margins.")
        anchor_index, anchor_is_junction = _find_anchor(
            route, cumulative, args.min_distance_from_start_m, args.min_distance_to_goal_m
        )
        trigger_location = _find_trigger_location(route, cumulative, anchor_index, args.trigger_lead_distance_m)

        effective_length_m = _effective_route_length(route, cumulative, goal_location)
        spec = build_base_spec(
            route_entry=route_entry,
            route=route,
            cumulative=cumulative,
            effective_length_m=effective_length_m,
            anchor_index=anchor_index,
            anchor_is_junction=anchor_is_junction,
            trigger_location=trigger_location,
            weather_preset=args.weather_preset,
            max_ticks=args.max_ticks,
            trigger_radius_m=args.trigger_radius_m,
            walker_speed=args.walker_speed,
            lateral_offset_multiplier=args.lateral_offset_multiplier,
        )
        output_path = args.output_dir / f"{route_entry['route_id']}.json"
        output_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        summary.append(
            {
                "route_id": route_entry["route_id"],
                "town": town,
                "spec_path": str(output_path),
                "route_waypoint_count": len(route),
                "route_length_m": round(effective_length_m, 1),
                "anchor_index": anchor_index,
                "anchor_is_junction": anchor_is_junction,
                "anchor_progress": round(cumulative[anchor_index] / max(effective_length_m, 1.0), 3),
                "anchor_location": _location_to_dict(route[anchor_index].transform.location),
                "trigger_location": _location_to_dict(trigger_location),
            }
        )
        print(f"{route_entry['route_id']}: anchor_index={anchor_index} is_junction={anchor_is_junction} -> {output_path}", flush=True)

    print(json.dumps({"base_specs": summary}, indent=2))
    print(f"wrote {len(summary)} base specs to {args.output_dir}")


if __name__ == "__main__":
    main()
