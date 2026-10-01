#!/usr/bin/env python
"""Turn the safety campaigns into paper-ready RQ3 numbers.

For each ADS campaign root, this reports per-policy *attributable* safety
issues: an outcome counts only when the generated run exhibits it and its
matched paired control does not (the paper's definition of attributable).
Mean +/- SD is taken across routes, matching the paper's table.

Usage:
    research/.venv/bin/python research/scripts/paper_bundle.py \
        --root interfuser=research/logs/paper_safety_if \
        --root autovla=research/logs/paper_safety_autovla \
        --out-dir research/logs/paper_results
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.harness.safety_outcomes import OUTCOME_NAMES, classify_run  # noqa: E402

POLICIES = ("semantic", "random", "lsa", "kmnc")
POLICY_LABEL = {"semantic": "SCOUT", "random": "Random", "lsa": "LSA", "kmnc": "KMNC"}
# paper columns -> constituent outcome names
COLUMNS = {
    "Collision": ("collision",),
    "Low TTC": ("near_collision",),
    "Unsafe Proximity": ("unsafe_proximity",),
    "Rule Violation": ("red_light_violation", "lane_departure", "traffic_rule_violation"),
}


def _rows(root: Path):
    for path in sorted(root.rglob("rows.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _outcomes(payload: dict) -> dict[str, bool]:
    return classify_run(payload).get("outcomes", {})


def attributable_per_route(root: Path) -> dict[str, dict[str, list[int]]]:
    """policy -> column -> per-route count of attributable issues."""
    per_route: dict[str, dict[str, list[int]]] = {
        p: {c: [] for c in list(COLUMNS) + ["Total"]} for p in POLICIES
    }
    route_rows: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in _rows(root):
        policy = str(row.get("policy") or "")
        if policy not in POLICIES:
            continue
        route = str(row.get("route") or "?")
        control = row.get("paired_control")
        if not isinstance(control, dict):
            continue
        cand = _outcomes(row)
        ctrl = _outcomes(control)
        route_rows[policy][route].append((cand, ctrl))
    for policy, routes in route_rows.items():
        for route, pairs in routes.items():
            # a route contributes 1 if any candidate on it shows an attributable outcome
            for column, names in list(COLUMNS.items()) + [("Total", OUTCOME_NAMES)]:
                hit = any(
                    any(c.get(n) and not k.get(n) for n in names)
                    for c, k in pairs
                )
                per_route[policy][column].append(1 if hit else 0)
    return per_route


def _mean_sd(values: list[int]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    return (statistics.fmean(values), statistics.pstdev(values))


def build_bundle(roots: dict[str, Path], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# RQ3 safety bundle", ""]
    tex: list[str] = []
    for ads, root in roots.items():
        if not root.exists():
            lines.append(f"## {ads}: MISSING root {root}")
            continue
        per_route = attributable_per_route(root)
        lines.append(f"## {ads}  ({root})")
        lines.append("")
        header = "| Policy | " + " | ".join(list(COLUMNS) + ["Total"]) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(COLUMNS) + 2))
        tex.append(f"% {ads} attributable safety issues (mean $\\pm$ sd over routes)")
        tex.append(f"{ads} &")
        for policy in POLICIES:
            cells = []
            for column in list(COLUMNS) + ["Total"]:
                m, s = _mean_sd(per_route[policy][column])
                cells.append(f"{m:.2f}$\\pm${s:.2f}")
            lines.append(f"| {POLICY_LABEL[policy]} | " + " | ".join(cells) + " |")
        lines.append("")
    (out_dir / "BUNDLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "tables.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return out_dir / "BUNDLE.md"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="name=path")
    ap.add_argument("--out-dir", type=Path, default=Path("research/logs/paper_results"))
    return ap


def main() -> int:
    args = build_parser().parse_args()
    roots = {}
    for item in args.root:
        name, _, path = item.partition("=")
        roots[name] = Path(path)
    bundle = build_bundle(roots, args.out_dir)
    print(f"wrote {bundle} and {args.out_dir / 'tables.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
