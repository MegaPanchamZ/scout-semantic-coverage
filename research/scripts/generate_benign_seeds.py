"""Generate benign, seed-randomised base scenarios (James review, item 1).

Takes the verified per-route threshold-crossing specs and produces benign
variants: the walker starts further back, crosses early (larger trigger
radius), and moves at a moderate speed, so the initial scenario is valid but
does not pre-ordain a collision. Every variant records its seed, and the
manifest fixes one identical seed set for all methods and ADSs.

Usage:
    research/.venv/bin/python research/scripts/generate_benign_seeds.py \
        --base-dir research/experiments/EXP-020-policy-comparison/artifacts/base_specs \
        --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def _unit(dx: float, dy: float) -> tuple[float, float]:
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 0.0, 0.0
    return dx / norm, dy / norm


def benign_variant(spec: dict, seed: int, args: argparse.Namespace) -> dict:
    rng = random.Random(f"{spec.get('scenario_id', 'scenario')}::{seed}")
    params = spec.setdefault("controller_params", {})
    radius = rng.uniform(*args.trigger_radius_range)
    speed = rng.uniform(*args.speed_range)
    spawn = params.get("spawn_transform", {}).get("location", {})
    dest = params.get("destination_location", {})
    back_off = rng.uniform(*args.spawn_back_off_range)
    ux, uy = _unit(spawn.get("x", 0.0) - dest.get("x", 0.0), spawn.get("y", 0.0) - dest.get("y", 0.0))
    variant = json.loads(json.dumps(spec))
    vp = variant["controller_params"]
    vp["trigger_radius_m"] = round(radius, 2)
    vp["speed"] = round(speed, 2)
    vp["spawn_transform"]["location"]["x"] = round(spawn.get("x", 0.0) + ux * back_off, 3)
    vp["spawn_transform"]["location"]["y"] = round(spawn.get("y", 0.0) + uy * back_off, 3)
    variant["scenario_id"] = f"{spec.get('scenario_id', 'scenario')}_benign_seed{seed}"
    variant["benign_seed"] = seed
    variant["benign"] = True
    variant["generation"] = {
        "tool": "generate_benign_seeds.py",
        "seed": seed,
        "trigger_radius_m": vp["trigger_radius_m"],
        "walker_speed_mps": vp["speed"],
        "spawn_back_off_m": round(back_off, 3),
    }
    variant["description"] = (
        f"Benign seed {seed}: {spec.get('description', '')} "
        f"Walker starts {back_off:.2f} m further back, triggers at "
        f"{radius:.1f} m and crosses at {speed:.2f} m/s (seed {seed})."
    )
    return variant


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate benign randomised seed scenarios.")
    ap.add_argument("--base-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--trigger-radius-range", type=float, nargs=2, default=[24.0, 40.0])
    ap.add_argument("--speed-range", type=float, nargs=2, default=[1.4, 2.2])
    ap.add_argument("--spawn-back-off-range", type=float, nargs=2, default=[0.5, 2.0])
    args = ap.parse_args()

    specs = sorted(
        p for p in args.base_dir.glob("*.json")
        if "_lead_braking" not in p.name and "_benign_seed" not in p.name and p.name != "benign_seeds.json"
    )
    manifest: dict = {"seed_set": args.seeds, "routes": {}}
    for spec_path in specs:
        spec = json.loads(spec_path.read_text())
        route = spec_path.stem
        written = []
        for seed in args.seeds:
            variant = benign_variant(spec, seed, args)
            out = args.base_dir / f"{route}_benign_seed{seed}.json"
            out.write_text(json.dumps(variant, indent=2))
            written.append(out.name)
        manifest["routes"][route] = written
        print(f"{route}: {len(written)} benign seeds")
    (args.base_dir / "benign_seeds.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {args.base_dir / 'benign_seeds.json'}")


if __name__ == "__main__":
    main()
