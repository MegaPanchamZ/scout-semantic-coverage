#!/usr/bin/env python
"""Audit base routes for hazard-search suitability and set a per-route tick budget.

The campaigns used one fixed ``--max-ticks`` for every route, but the routes
range from 40 m to 1.3 km. Short routes end before a hazard can trigger (the
lead-braking lead can sit up to 45 m ahead); long routes are guaranteed
``route_incomplete`` at the cap. This tool reads the baked route polyline from
each ``*_lead_braking.json`` and reports, per route:

* ``length_m``
* ``ticks``: ticks needed at ``--speed`` m/s plus ``--margin`` (0.1 s ticks)
* ``suitable``: length within ``[--min-length, --max-length]``

Usage:
    python research/scripts/route_audit.py [--base-dir DIR] [--json OUT] [--ticks ROUTE]
``--ticks ROUTE`` prints just the tick budget (for shell launchers).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_BASE = Path(__file__).resolve().parents[1] / "experiments/EXP-020-policy-comparison/artifacts/base_specs"
LEAD_GAP_MAX_M = 45.0  # LEAD_BRAKING_SPACE upper bound


def polyline_length(poly: list) -> float:
    return sum(math.dist(poly[i][:2], poly[i + 1][:2]) for i in range(len(poly) - 1))


def audit(base_dir: Path, speed: float, margin: float, min_length: float, max_length: float,
          tick_seconds: float = 0.1, min_ticks: int = 300, max_ticks: int = 2400) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(base_dir.glob("*_lead_braking.json")):
        spec = json.loads(path.read_text())
        poly = (spec.get("controller_params") or {}).get("route_polyline") or []
        if len(poly) < 2:
            continue
        length = polyline_length(poly)
        ticks = int(math.ceil(length / speed / tick_seconds * margin))
        ticks = max(min_ticks, min(max_ticks, ticks))
        route = path.name[: -len("_lead_braking.json")]
        reasons = []
        if length < min_length:
            reasons.append(f"shorter than {min_length:.0f} m: hazard cannot trigger before the goal")
        if length > max_length:
            reasons.append(f"longer than {max_length:.0f} m: time cap / drift dominate")
        out[route] = {"length_m": round(length, 1), "ticks": ticks, "suitable": not reasons, "reasons": reasons}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--speed", type=float, default=4.0, help="Conservative ADS cruise speed in m/s.")
    ap.add_argument("--margin", type=float, default=1.5, help="Multiplier on the nominal drive time.")
    ap.add_argument("--min-length", type=float, default=LEAD_GAP_MAX_M + 80.0)
    ap.add_argument("--max-length", type=float, default=700.0)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--ticks", metavar="ROUTE", default=None)
    args = ap.parse_args()
    result = audit(args.base_dir, args.speed, args.margin, args.min_length, args.max_length)
    if args.ticks:
        if args.ticks not in result:
            print(f"unknown route {args.ticks}", file=sys.stderr)
            return 1
        print(result[args.ticks]["ticks"])
        return 0
    for route, info in result.items():
        flag = "ok " if info["suitable"] else "DROP"
        print(f"{flag} {route:34s} {info['length_m']:7.1f} m  ticks={info['ticks']:5d}  {'; '.join(info['reasons'])}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
