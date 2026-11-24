#!/usr/bin/env python
"""Report safety-outcome discovery per policy from campaign logs.

Collision alone under-states what a search can find.  This script joins each
evaluation row in a policy-search campaign to its recorded run log, classifies
the run into the named unsafe behaviours in
``research/harness/legacy_outcomes.py``, and reports per-policy discovery
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

from research.harness.legacy_outcomes import (  # noqa: E402
    OUTCOME_NAMES,
    OutcomeThresholds,
    safety_schema,
)
from research.harness.outcome_models import CATEGORIES  # noqa: E402
from research.harness.safety_outcomes import safety_schema as safety_schema_v2  # noqa: E402
from research.harness.safety_utility import EvalRecord, load_records  # noqa: E402


DEFAULT_SEARCH_ROOT = WORKSPACE_ROOT / "research" / "logs" / "policy_search"
V2_METRIC_KEYS = tuple(f"{g}.{m}" for g, names in CATEGORIES.items() for m in names)


def _empty_v2_bucket(route: str) -> dict[str, Any]:
    return {
        "route": route,
        "n_attempted": 0,
        "n_executed": 0,
        "n_completed": 0,
        "n_partial": 0,
        "n_failed_or_missing": 0,
        "n_valid": 0,
        "metrics": {
            key: {
                "present": 0,
                "absent": 0,
                "unknown": 0,
                "not_applicable": 0,
                "episode_count": 0,
                "distinct_actors": set(),
                "assessable_denominator": 0,
                "assessable_rate": None,
                "discovery_per_attempt": None,
            }
            for key in V2_METRIC_KEYS
        },
        "aggregates": {
            key: {
                "present": 0,
                "absent": 0,
                "unknown": 0,
                "not_applicable": 0,
                "assessable_denominator": 0,
                "assessable_rate": None,
                "discovery_per_attempt": None,
            }
            for key in ("safety_event", "traffic_rule_violation")
        },
        "complete_execution_stratum": {
            "n_completed": 0,
            "metrics": {
                key: {"present": 0, "absent": 0, "unknown": 0, "not_applicable": 0, "assessable_rate": None}
                for key in V2_METRIC_KEYS
            },
            "aggregates": {
                key: {"present": 0, "absent": 0, "unknown": 0, "not_applicable": 0, "assessable_rate": None}
                for key in ("safety_event", "traffic_rule_violation")
            },
        },
    }


def _accumulate_v2(bucket: dict[str, Any], record: EvalRecord) -> None:
    bucket["n_attempted"] += 1
    out = record.outcome if isinstance(record.outcome, dict) else {}
    exec_status = str(out.get("execution_status") or record.execution_status or "unknown")
    is_completed = exec_status == "completed"
    if exec_status in ("completed", "partial"):
        bucket["n_executed"] += 1
    if is_completed:
        bucket["n_completed"] += 1
        bucket["complete_execution_stratum"]["n_completed"] += 1
    elif exec_status == "partial":
        bucket["n_partial"] += 1
    else:
        bucket["n_failed_or_missing"] += 1

    raw_categories = out.get("categories")
    categories = raw_categories if isinstance(raw_categories, dict) else {}
    quality = categories.get("quality")
    invalid_sc = quality.get("invalid_scenario") if isinstance(quality, dict) else None
    if isinstance(invalid_sc, dict) and invalid_sc.get("status") == "absent":
        bucket["n_valid"] += 1

    events_by_label: dict[str, list[dict[str, Any]]] = {}
    raw_events = out.get("events")
    if isinstance(raw_events, (list, tuple)):
        for ev in raw_events:
            if isinstance(ev, dict) and ev.get("label"):
                events_by_label.setdefault(str(ev["label"]), []).append(ev)

    run_scope = str(record.run_id or record.attempt_id or record.run_json_path or id(record))
    for grp, names in CATEGORIES.items():
        raw_grp = categories.get(grp)
        grp_map = raw_grp if isinstance(raw_grp, dict) else {}
        for name in names:
            key = f"{grp}.{name}"
            entry = grp_map.get(name)
            raw_status = entry.get("status") if isinstance(entry, dict) else None
            status = (
                raw_status
                if isinstance(raw_status, str) and raw_status in ("present", "absent", "unknown", "not_applicable")
                else "unknown"
            )
            m = bucket["metrics"][key]
            m[status] += 1
            evs = events_by_label.get(name, ())
            m["episode_count"] += len(evs)
            for ev in evs:
                actor_ids = ev.get("actor_ids")
                if isinstance(actor_ids, (list, tuple)):
                    for aid in actor_ids:
                        if isinstance(aid, str) and aid.strip():
                            m["distinct_actors"].add((run_scope, aid.strip()))
                        elif isinstance(aid, int) and not isinstance(aid, bool):
                            m["distinct_actors"].add((run_scope, str(aid)))
            if is_completed:
                bucket["complete_execution_stratum"]["metrics"][key][status] += 1

    raw_aggregates = out.get("aggregates")
    aggregates = raw_aggregates if isinstance(raw_aggregates, dict) else {}
    for agg_name in ("safety_event", "traffic_rule_violation"):
        agg_entry = aggregates.get(agg_name)
        raw_status = agg_entry.get("status") if isinstance(agg_entry, dict) else None
        status = (
            raw_status
            if isinstance(raw_status, str) and raw_status in ("present", "absent", "unknown", "not_applicable")
            else "unknown"
        )
        bucket["aggregates"][agg_name][status] += 1
        if is_completed:
            bucket["complete_execution_stratum"]["aggregates"][agg_name][status] += 1


def _finalize_v2_bucket(bucket: dict[str, Any]) -> None:
    n_att = bucket["n_attempted"]
    for m in bucket["metrics"].values():
        denom = m["present"] + m["absent"]
        m["assessable_denominator"] = denom
        m["assessable_rate"] = (m["present"] / denom) if denom else None
        m["discovery_per_attempt"] = (m["present"] / n_att) if n_att else None
        m["distinct_actors"] = len(m["distinct_actors"])
    for m in bucket["aggregates"].values():
        denom = m["present"] + m["absent"]
        m["assessable_denominator"] = denom
        m["assessable_rate"] = (m["present"] / denom) if denom else None
        m["discovery_per_attempt"] = (m["present"] / n_att) if n_att else None
    for m in bucket["complete_execution_stratum"]["metrics"].values():
        denom = m["present"] + m["absent"]
        m["assessable_rate"] = (m["present"] / denom) if denom else None
    for m in bucket["complete_execution_stratum"]["aggregates"].values():
        denom = m["present"] + m["absent"]
        m["assessable_rate"] = (m["present"] / denom) if denom else None


def build_report(
    search_root: Path,
    *,
    thresholds: OutcomeThresholds | None = None,
    include_paired_controls: bool = False,
    schema_version: int = 1,
) -> dict[str, Any]:
    if schema_version == 2:
        records, counters = load_records(
            [search_root],
            include_paired_controls=include_paired_controls,
            schema_version=2,
        )
        per_policy: dict[str, dict[str, Any]] = {}
        per_policy_route: dict[str, dict[str, dict[str, Any]]] = {}
        per_policy_replicate: dict[str, dict[str, dict[str, Any]]] = {}
        routes: set[str] = set()
        replicates: set[str] = set()
        profile_digests: set[str] = set()
        for record in records:
            routes.add(record.route)
            replicates.add(record.replicate)
            if record.profile_digest:
                profile_digests.add(record.profile_digest)
            bucket = per_policy.setdefault(record.policy, _empty_v2_bucket(""))
            _accumulate_v2(bucket, record)
            rb = per_policy_route.setdefault(record.policy, {}).setdefault(
                record.route, _empty_v2_bucket(record.route)
            )
            _accumulate_v2(rb, record)
            rep_b = per_policy_replicate.setdefault(record.policy, {}).setdefault(
                record.replicate, _empty_v2_bucket(record.replicate)
            )
            _accumulate_v2(rep_b, record)
        for bucket in per_policy.values():
            _finalize_v2_bucket(bucket)
        for policy_routes in per_policy_route.values():
            for bucket in policy_routes.values():
                _finalize_v2_bucket(bucket)
        for policy_reps in per_policy_replicate.values():
            for bucket in policy_reps.values():
                _finalize_v2_bucket(bucket)
        policies = sorted(per_policy)
        return {
            "schema_version": 2,
            "config": {
                "search_root": str(search_root),
                "include_paired_controls": include_paired_controls,
                "schema_version": 2,
            },
            "safety_schema": safety_schema_v2(),
            "data_state": {
                **counters,
                "n_attempted": len(records),
                "policies": policies,
                "routes": sorted(routes),
                "replicates": sorted(replicates),
                "profile_digests": sorted(profile_digests),
                "mixed_profiles": len(profile_digests) > 1,
            },
            "policies": {p: per_policy[p] for p in policies},
            "per_route": {p: dict(sorted(per_policy_route[p].items())) for p in policies},
            "per_replicate": {p: dict(sorted(per_policy_replicate[p].items())) for p in policies},
        }

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
    if result.get("schema_version") == 2:
        state = result["data_state"]
        lines = [
            "# Outcome evaluation report (schema v2)",
            "",
            "Categories are reported separately with explicit `present / absent / unknown / not_applicable` counts.",
            "No global `unsafe_rate` combines safety, rules, task, and quality.",
            "",
            "## Data state",
            "",
            f"- Search root: `{result['config']['search_root']}`",
            f"- Attempted evaluations retained: {state['n_attempted']} (rows read: {state['n_rows']}, malformed: {state['n_malformed_rows']}, duplicates skipped: {state['n_duplicate_rows']}).",
            f"- Missing run logs: {state['n_missing_run_path']}; unreadable: {state['n_unreadable_run']}; failed attempts: {state['n_failed_attempts']}.",
            f"- Policies: {', '.join(state['policies']) or 'none'}; routes: {len(state['routes'])}; replicates: {', '.join(state.get('replicates', ())) or 'none'}.",
            f"- Observed profile digests: {', '.join(state.get('profile_digests', ())) or 'none'} (mixed_profiles={state.get('mixed_profiles', False)}).",
            "",
            "## Explicit aggregates (`safety_event`, `traffic_rule_violation`)",
            "",
            "| Policy | Aggregate | Attempted | Present | Absent | Unknown | N/A | Assessable rate (P/(P+A)) | Per-attempt | Complete-stratum rate |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for policy in state["policies"]:
            bucket = result["policies"][policy]
            for agg_name in ("safety_event", "traffic_rule_violation"):
                a = bucket["aggregates"][agg_name]
                ca = bucket["complete_execution_stratum"]["aggregates"][agg_name]
                lines.append(
                    f"| {policy} | {agg_name} | {bucket['n_attempted']} | {a['present']} | {a['absent']} | "
                    f"{a['unknown']} | {a['not_applicable']} | {_fmt(a['assessable_rate'], 3)} | "
                    f"{_fmt(a['discovery_per_attempt'], 3)} | {_fmt(ca['assessable_rate'], 3)} |"
                )
        lines.append("")
        for grp, names in CATEGORIES.items():
            lines.append(f"## Category: `{grp}`")
            lines.append("")
            lines.append("| Policy | Metric | Attempted | Present | Absent | Unknown | N/A | Assessable rate (P/(P+A)) | Per-attempt | Complete-stratum rate | Episodes |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for policy in state["policies"]:
                bucket = result["policies"][policy]
                for name in names:
                    m = bucket["metrics"][f"{grp}.{name}"]
                    cm = bucket["complete_execution_stratum"]["metrics"][f"{grp}.{name}"]
                    lines.append(
                        f"| {policy} | {name} | {bucket['n_attempted']} | {m['present']} | {m['absent']} | "
                        f"{m['unknown']} | {m['not_applicable']} | {_fmt(m['assessable_rate'], 3)} | "
                        f"{_fmt(m['discovery_per_attempt'], 3)} | {_fmt(cm['assessable_rate'], 3)} | {m['episode_count']} |"
                    )
            lines.append("")
        return "\n".join(lines)

    state = result["data_state"]
    lines: list[str] = [
        "# Safety-outcome discovery report",
        "",
        "Collision is one outcome, not the evaluation target. This report classifies each "
        "recorded run into the named unsafe behaviours in `research/harness/legacy_outcomes.py`.",
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
    parser.add_argument("--outcome-schema", type=int, choices=(1, 2), default=1)
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
        schema_version=args.outcome_schema,
    )
    paths = write_report(result, out_dir)
    state = result["data_state"]
    if result.get("schema_version") == 2:
        print(
            "outcome report v2: attempted {att} (missing={missing}, unreadable={bad}, malformed={mal})".format(
                att=state["n_attempted"],
                missing=state["n_missing_run_path"],
                bad=state["n_unreadable_run"],
                mal=state["n_malformed_rows"],
            )
        )
    else:
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
