"""RQ3 hazard-witness metrics from the EXP-020 policy-search dataset.

Defines a hazard witness as an evaluation whose semantic signatures include the
adversary-walker hazard family. Reports per-policy discovery rate, time to first
witness, and the unique hazard signatures discovered per policy.

Usage:
  research/.venv/bin/python hazard_witness_metrics.py --root research/logs/policy_search \
      --output research/experiments/EXP-020-policy-comparison/artifacts/hazard_witness_metrics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import random
import statistics

HAZARD_SIGNATURES = (
    "crossing_path(pedestrian,ego)",
    "jaywalking(pedestrian)",
    "in_front_of(pedestrian,ego)",
    "colliding(ego,pedestrian)",
)


def is_hazard_witness(row: dict) -> bool:
    signatures = set(row.get("semantic_covered_signatures") or [])
    return any(sig in signatures for sig in HAZARD_SIGNATURES)


def bootstrap_ci(values: list[float], resamples: int = 2000, seed: int = 11) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hazard-witness metrics for the policy search.")
    parser.add_argument("--root", type=Path, default=Path("research/logs/policy_search"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policies = ("random", "lsa", "kmnc", "semantic", "control")
    per_route: dict[str, dict[str, dict]] = {}
    for rows_path in sorted(glob.glob(str(args.root / "*" / "*" / "rows.jsonl"))):
        parts = Path(rows_path).parts
        route, policy = parts[-3], parts[-2]
        if policy not in policies:
            continue
        rows = []
        for line in Path(rows_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ticks_executed") is None:
                continue
            rows.append(row)
        if not rows:
            continue
        rows.sort(key=lambda r: r.get("eval_index") or 0)
        witnesses = [r for r in rows if is_hazard_witness(r)]
        first_index = next((r["eval_index"] for r in rows if is_hazard_witness(r)), None)
        signatures = sorted({sig for r in rows for sig in (r.get("semantic_covered_signatures") or []) if sig in HAZARD_SIGNATURES})
        per_route.setdefault(policy, {})[route] = {
            "n": len(rows),
            "witness_count": len(witnesses),
            "witness_rate": len(witnesses) / len(rows),
            "first_witness_index": first_index,
            "hazard_signatures": signatures,
        }

    summary = {}
    for policy, routes in per_route.items():
        rates = [v["witness_rate"] for v in routes.values()]
        firsts = [v["first_witness_index"] for v in routes.values() if v["first_witness_index"] is not None]
        all_signatures = sorted({sig for v in routes.values() for sig in v["hazard_signatures"]})
        summary[policy] = {
            "routes": len(routes),
            "witness_rate_mean": statistics.fmean(rates),
            "witness_rate_ci95": bootstrap_ci(rates),
            "routes_with_witness": len(firsts),
            "first_witness_index_mean": statistics.fmean(firsts) if firsts else None,
            "unique_hazard_signatures": all_signatures,
            "per_route": routes,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{'policy':<10} {'routes':>6} {'witness rate':>13} {'CI95':>18} {'routes w/ witness':>17} {'mean first idx':>14}")
    for policy, s in summary.items():
        ci = f"[{s['witness_rate_ci95'][0]:.3f}, {s['witness_rate_ci95'][1]:.3f}]"
        first = "-" if s["first_witness_index_mean"] is None else f"{s['first_witness_index_mean']:.1f}"
        print(f"{policy:<10} {s['routes']:>6} {s['witness_rate_mean']:>13.3f} {ci:>18} {s['routes_with_witness']:>17} {first:>14}")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
