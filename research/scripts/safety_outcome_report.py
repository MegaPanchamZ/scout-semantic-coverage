#!/usr/bin/env python
"""Report safety-outcome discovery per policy from campaign logs.

Collision alone under-states what a search can find.  This script joins each
evaluation row in a policy-search campaign to its recorded run log, classifies
the run into the named unsafe behaviours in
``research/harness/safety_outcomes.py``, and reports per-policy discovery
rates.  It is the counting half of the utility evaluation; the deeper
obligation-linkage and time-to-first analysis lives in
``research/scripts/safety_utility_report.py``.

Input layouts (both discovered, exactly like ``aggregate_fse_search.py``):

- ``<root>/<route>/<policy>/rows.jsonl`` (nested)
- ``<root>/<policy>/rows.jsonl`` (flat)

A row's ``run_json_path`` is loaded directly, so rows produced before safety
outcomes were added to the row still score.  Rows without a resolvable run log
are counted, never silently dropped.

Usage::

    research/.venv/bin/python research/scripts/safety_outcome_report.py \
        --search-root research/logs/policy_search \
        --out-dir research/logs/safety_report
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.safety_outcomes import (  # noqa: E402
    OUTCOME_NAMES,
    OutcomeThresholds,
    safety_schema,
)
from research.harness.safety_utility import EvalRecord, load_records  # noqa: E402


DEFAULT_SEARCH_ROOT = WORKSPACE_ROOT / "research" / "logs" / "policy_search"


def build_report(
    search_root: Path,
    *,
    thresholds: OutcomeThresholds | None = None,
    include_paired_controls: bool = False,
) -> dict[str, Any]:
    th = thresholds or OutcomeThresholds()
    records, counters = load_records(
        [search_root], thresholds=th, include_paired_controls=include_paired_controls
    )

    per_policy: dict[str, dict[str, Any]] = {}
    per_policy_route: dict[str, dict[str, dict[str, Any]]] = {}
    routes: set[str] = set()

    for record in records:
        routes.add(record.route)
        bucket = per_policy.setdefault(record.policy, _empty_policy(""))
        _accumulate(bucket, record, th)
        route_bucket = per_policy_route.setdefault(record.policy, {}).setdefault(
            record.route, _empty_policy(record.route)
        )
        _accumulate(route_bucket, record, th)

    for bucket in per_policy.values():
        _finalize_bucket(bucket)
    for policy_routes in per_policy_route.values():
        for bucket in policy_routes.values():
            _finalize_bucket(bucket)

    policies = sorted(per_policy)
    return {
        "config": {
            "search_root": str(search_root),
            "include_paired_controls": include_paired_controls,
        },
        "safety_schema": safety_schema(),
        "thresholds": asdict(th),
        "data_state": {
            "n_evaluations": counters["n_rows"],
            "n_scored": len(records),
            "n_missing_run_path": counters["n_missing_run_path"],
            "n_unreadable_run": counters["n_unreadable_run"],
            "policies": policies,
            "routes": sorted(routes),
        },
        "policies": {policy: per_policy[policy] for policy in policies},
        "per_route": {policy: dict(sorted(per_policy_route[policy].items())) for policy in policies},
    }


def _empty_policy(route: str) -> dict[str, Any]:
    return {
        "route": route,
        "n_scored": 0,
        "unsafe_runs": 0,
        "unsafe_rate": None,
        "outcome_counts": {name: 0 for name in OUTCOME_NAMES},
        "outcome_rates": {name: None for name in OUTCOME_NAMES},
        "issue_types_found": set(),
        "first_unsafe_eval": None,
        "max_eval": None,
    }


def _accumulate(bucket: dict[str, Any], record: EvalRecord, thresholds: OutcomeThresholds) -> None:
    if record.eval_index is not None:
        previous = bucket["first_unsafe_eval"]
        if record.outcome.get("unsafe") and (previous is None or record.eval_index < previous):
            bucket["first_unsafe_eval"] = record.eval_index
        total = bucket["max_eval"]
        bucket["max_eval"] = record.eval_index if total is None else max(total, record.eval_index)
    bucket["n_scored"] += 1
    if record.outcome.get("unsafe"):
        bucket["unsafe_runs"] += 1
    for name in OUTCOME_NAMES:
        if record.outcome.get(name):
            bucket["outcome_counts"][name] += 1
    bucket["issue_types_found"].update(record.outcome.get("reasons", []))


def _finalize_bucket(bucket: dict[str, Any]) -> None:
    n = bucket["n_scored"]
    bucket["unsafe_rate"] = (bucket["unsafe_runs"] / n) if n else None
    bucket["outcome_rates"] = {
        name: (count / n) if n else None for name, count in bucket["outcome_counts"].items()
    }
    bucket["issue_types_found"] = sorted(bucket["issue_types_found"])


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def build_markdown(result: dict[str, Any]) -> str:
    state = result["data_state"]
    lines: list[str] = [
        "# Safety-outcome discovery report",
        "",
        "Collision is one outcome, not the evaluation target. This report classifies each "
        "recorded run into the named unsafe behaviours in `research/harness/safety_outcomes.py`.",
        "",
        "## Data state",
        "",
        f"- Search root: `{result['config']['search_root']}`",
        f"- Evaluations read: {state['n_evaluations']}; scored: {state['n_scored']}.",
        f"- Rows without a resolvable run log: {state['n_missing_run_path']}; "
        f"unreadable run logs: {state['n_unreadable_run']}.",
        f"- Policies: {', '.join(state['policies']) or 'none'}; routes: {len(state['routes'])}.",
        "",
        "## Per-policy safety outcomes",
        "",
        "A cell is the number of scored runs in which the outcome was found "
        "(rate in parentheses). `Issues found` is the number of distinct outcome "
        "types the policy discovered.",
        "",
    ]
    header = ["Policy", "Scored", "Unsafe", "Issues found", "First unsafe eval"] + list(OUTCOME_NAMES)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for policy in state["policies"]:
        bucket = result["policies"][policy]
        cells = [
            policy,
            str(bucket["n_scored"]),
            f"{bucket['unsafe_runs']} ({_fmt(bucket['unsafe_rate'], 2)})",
            str(len(bucket["issue_types_found"])),
            _fmt(bucket["first_unsafe_eval"], 0),
        ]
        for name in OUTCOME_NAMES:
            cells.append(
                f"{bucket['outcome_counts'][name]} ({_fmt(bucket['outcome_rates'][name], 2)})"
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("> Thresholds are engineering proxies; report sensitivity alongside these numbers.")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    for key, value in result["thresholds"].items():
        lines.append(f"- `{key}`: {_fmt(value, 3) if isinstance(value, (int, float)) else value}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `safety_report.json` (machine-readable, includes per-route breakdown)")
    lines.append("- `safety_report.md` (this file)")
    lines.append("")
    return "\n".join(lines)


def write_report(result: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "safety_report.json"
    md_path = out_dir / "safety_report.md"
    json_path.write_text(json.dumps(_json_safe(result), indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(result) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search-root", type=Path, default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--include-paired-controls",
        action="store_true",
        help="Also score each row's no-adversary paired control as '<policy>:control'.",
    )
    parser.add_argument("--stuck-seconds", type=float, default=8.0)
    parser.add_argument("--near-collision-ttc-s", type=float, default=1.5)
    parser.add_argument("--lane-offset-m", type=float, default=2.0)
    parser.add_argument("--harsh-deceleration-mps2", type=float, default=3.0)
    return parser


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    search_root = _resolve(args.search_root)
    out_dir = _resolve(args.out_dir) if args.out_dir is not None else search_root
    thresholds = OutcomeThresholds(
        stuck_seconds=args.stuck_seconds,
        near_collision_ttc_s=args.near_collision_ttc_s,
        lane_offset_m=args.lane_offset_m,
        harsh_deceleration_mps2=args.harsh_deceleration_mps2,
    )
    result = build_report(
        search_root,
        thresholds=thresholds,
        include_paired_controls=args.include_paired_controls,
    )
    paths = write_report(result, out_dir)
    state = result["data_state"]
    print(
        "safety report: scored {scored}/{total} evaluations; missing run logs {missing}; "
        "unreadable {bad}".format(
            scored=state["n_scored"],
            total=state["n_evaluations"],
            missing=state["n_missing_run_path"],
            bad=state["n_unreadable_run"],
        )
    )
    for label, path in paths.items():
        print(f"wrote {path} [{label}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
