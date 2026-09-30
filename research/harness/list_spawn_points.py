from __future__ import annotations

import argparse
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.carla_utils import connect_client, load_world


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List CARLA spawn points, optionally ordered by distance to a reference spawn.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town01")
    parser.add_argument("--reference-index", type=int, default=None, help="If set, sort output by distance to this spawn index.")
    parser.add_argument("--limit", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    client = connect_client(args.host, args.port, 20.0)
    world = load_world(client, args.town)
    spawn_points = world.get_map().get_spawn_points()
    if args.reference_index is not None and not (0 <= args.reference_index < len(spawn_points)):
        raise ValueError(f"Reference index {args.reference_index} is out of range for {len(spawn_points)} spawn points.")

    reference_location = None
    if args.reference_index is not None:
        reference_location = spawn_points[args.reference_index].location

    rows: list[tuple[float, int, float, float, float, float]] = []
    for index, transform in enumerate(spawn_points):
        location = transform.location
        distance = 0.0
        if reference_location is not None:
            distance = float(location.distance(reference_location))
        rows.append(
            (
                distance,
                index,
                float(location.x),
                float(location.y),
                float(location.z),
                float(transform.rotation.yaw),
            )
        )

    if reference_location is not None:
        rows.sort(key=lambda item: (item[0], item[1]))
    else:
        rows.sort(key=lambda item: item[1])

    print(f"spawn_count={len(spawn_points)}")
    print("distance_m,index,x,y,z,yaw_deg")
    for distance, index, x, y, z, yaw in rows[: max(args.limit, 1)]:
        print(f"{distance:.3f},{index},{x:.3f},{y:.3f},{z:.3f},{yaw:.3f}")


if __name__ == "__main__":
    main()