#!/usr/bin/env python
"""Utility evaluation: how many, from what, and how fast.

This is the deeper companion to ``safety_outcome_report.py``.  From the same
campaign logs it produces the three discussion quantities for the utility
research question:

1. **How many safety issues** each policy found (named outcomes, not collision
   alone).
2. **Which semantic obligations lead to which safety issues** — per-obligation
   outcome rates and lift against runs that did not witness the obligation.
3. **How quickly the first safety issue is found** — the first evaluation index
   with any issue per independent search arm, summarised as mean/std (plus
   median and Top-1 rate) across repeated runs, with a right-censored value for
   arms that never found one, and paired primary-vs-baseline differences.

Usage::

    research/.venv/bin/python research/scripts/safety_utility_report.py \
        --search-root research/logs/fse_search_gap_v1 \
        --out-dir research/logs/safety_utility

``--search-root`` accepts several roots.  When a root contains ``seed-*``
children they are expanded automatically so each run is a replicate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.safety_outcomes import (  # noqa: E402
    OUTCOME_NAMES,
    OutcomeThresholds,
    safety_schema,
)
from research.harness.safety_utility import (  # noqa: E402
    EvalRecord,
    load_records,
    obligation_association,
    paired_time_to_first,
    rank_associations,
    summarize_time_to_first,
    summarize_time_to_first_by_replicate,
    time_to_first,
)


DEFAULT_SEARCH_ROOT = WORKSPACE_ROOT / "research" / "logs" / "policy_search"
DEFAULT_PRIMARY = "semantic"
DEFAULT_COMPARISONS = ("random", "lsa", "kmnc")


def expand_seed_roots(search_roots: Sequence[Path]) -> list[Path]:
    """Expand each root into its ``seed-*`` children (plus the root itself)."""
    expanded: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        candidates = [root]
        if root.is_dir():
            candidates.extend(sorted(path for path in root.glob("seed-*") if path.is_dir()))
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


def build_utility_report(
    search_roots: Sequence[Path],
    *,
    thresholds: OutcomeThresholds | None = None,
    include_paired_controls: bool = False,
    primary_policy: str = DEFAULT_PRIMARY,
    comparison_policies: Sequence[str] = DEFAULT_COMPARISONS,
    budget: int | None = None,
    association_source: str = "witnessed",
    top_n: int = 20,
) -> dict[str, Any]:
    th = thresholds or OutcomeThresholds()
    roots = expand_seed_roots(list(search_roots))
    records, counters = load_records(
        roots, thresholds=th, include_paired_controls=include_paired_controls
    )
    arms = time_to_first(records, budget=budget)

    policies = sorted({record.policy for record in records})
    comparisons = [policy for policy in comparison_policies if policy in policies and policy != primary_policy]

    # How many issues, per policy (union of outcome types over its runs).
    issues_by_policy: dict[str, Any] = {}
    for policy in policies:
        policy_records = [record for record in records if record.policy == policy]
        types: set[str] = set()
        unsafe_runs = 0
        for record in policy_records:
            types.update(str(name) for name in record.outcome.get("reasons", []))
            if record.outcome.get("unsafe"):
                unsafe_runs += 1
        issues_by_policy[policy] = {
            "n_runs": len(policy_records),
            "unsafe_runs": unsafe_runs,
            "unsafe_rate": (unsafe_runs / len(policy_records)) if policy_records else None,
            "issue_types_found": sorted(types),
        }

    witnessed = obligation_association(records, source="witnessed")
    targeted = obligation_association(records, source="target") if association_source == "target" else None

    return {
        "config": {
            "search_roots": [str(root) for root in roots],
            "include_paired_controls": include_paired_controls,
            "primary_policy": primary_policy,
            "comparison_policies": list(comparison_policies),
            "budget": budget,
            "association_source": association_source,
            "top_n": top_n,
        },
        "safety_schema": safety_schema(),
        "thresholds": asdict(th),
        "data_state": {
            **counters,
            "n_scored": len(records),
            "n_arms": len(arms),
            "policies": policies,
            "replicates": sorted({record.replicate for record in records}),
        },
        "issues_by_policy": issues_by_policy,
        "time_to_first": {
            "summary": summarize_time_to_first(arms),
            "by_replicate": summarize_time_to_first_by_replicate(arms),
            "arms": [asdict(arm) for arm in arms],
        },
        "paired_time_to_first": [
            paired_time_to_first(arms, primary_policy, comparison) for comparison in comparisons
        ],
        "association": {
            "witnessed": witnessed,
            "targeted": targeted,
            "top_witnessed": rank_associations(witnessed, "unsafe", top_n),
            "outcome_leaders": _outcome_leaders(witnessed, top_n),
        },
    }


def _outcome_leaders(association: dict[str, Any], top_n: int) -> dict[str, list[dict[str, Any]]]:
    leaders: dict[str, list[dict[str, Any]]] = {}
    for name in OUTCOME_NAMES:
        if name in {"traffic_rule_violation"}:  # derived aggregate of two outcomes
            continue
        ranked = [item for item in rank_associations(association, name, top_n) if item["lift"] > 0]
        if ranked:
            leaders[name] = ranked
    return leaders


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _risk_ratio(present: Any, absent: Any) -> str:
    if present is None or absent is None:
        return "—"
    if absent <= 0.0:
        return "∞" if present > 0.0 else "—"
    return f"{present / absent:.2f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def build_markdown(result: dict[str, Any]) -> str:
    config = result["config"]
    state = result["data_state"]
    primary = config["primary_policy"]
    lines: list[str] = [
        "# Safety-utility evaluation",
        "",
        "Beyond collision: how many named safety issues each policy finds, which semantic "
        "obligations are associated with which issues, and how quickly the first issue appears.",
        "",
        "## Data state",
        "",
        f"- Roots: {', '.join('`' + root + '`' for root in config['search_roots'])}",
        f"- Scored runs: {state['n_scored']} / {state['n_rows']} rows; "
        f"arms: {state['n_arms']}; replicates: {', '.join(state['replicates'])}.",
        f"- Rows without a resolvable run log: {state['n_missing_run_path']}; "
        f"unreadable: {state['n_unreadable_run']}.",
        f"- Policies: {', '.join(state['policies']) or 'none'}.",
        "",
        "## 1. Safety issues found per policy",
        "",
        "| Policy | Runs | Unsafe runs | Unsafe rate | Distinct issue types | Types |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for policy, entry in sorted(result["issues_by_policy"].items()):
        lines.append(
            "| {policy} | {n} | {unsafe} | {rate} | {kinds} | {types} |".format(
                policy=policy,
                n=entry["n_runs"],
                unsafe=entry["unsafe_runs"],
                rate=_fmt(entry["unsafe_rate"]),
                kinds=len(entry["issue_types_found"]),
                types=", ".join(entry["issue_types_found"]) or "—",
            )
        )
    lines.append("")

    lines.append("## 2. Time to first safety issue")
    lines.append("")
    lines.append(
        "One independent run = one (policy, replicate, route) search arm. `First eval` is the "
        "smallest evaluation index whose run had any safety issue. Censored arms never found "
        "one and are scored as `budget + 1` in the mean/std columns."
    )
    lines.append("")
    lines.append(
        "| Policy | Arms | Found | Found rate | Top-1 rate | Mean first (found) | SD (found) | "
        "Median (found) | Mean first (censored) | SD (censored) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for policy, entry in sorted(result["time_to_first"]["summary"].items()):
        lines.append(
            "| {policy} | {arms} | {found} | {fr} | {top1} | {mf} | {sf} | {med} | {mc} | {sc} |".format(
                policy=policy,
                arms=entry["n_arms"],
                found=entry["n_found"],
                fr=_fmt(entry["found_rate"]),
                top1=_fmt(entry["top1_rate"]),
                mf=_fmt(entry["mean_first_found"]),
                sf=_fmt(entry["std_first_found"]),
                med=_fmt(entry["median_first_found"]),
                mc=_fmt(entry["mean_first_censored"]),
                sc=_fmt(entry["std_first_censored"]),
            )
        )
    lines.append("")

    paired = result["paired_time_to_first"]
    if paired:
        lines.append("### Paired comparison vs baselines (censored iterations; lower is better)")
        lines.append("")
        lines.append(
            "| Primary | Baseline | Pairs | Mean diff [SD] | Median diff | Primary fewer | Ties | "
            "Baseline fewer | Wilcoxon p |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for record in paired:
            if record["n_pairs"] == 0:
                continue
            lines.append(
                "| {p} | {c} | {n} | {md} [{sd}] | {med} | {w} | {t} | {l} | {pval} |".format(
                    p=record["primary"],
                    c=record["comparison"],
                    n=record["n_pairs"],
                    md=_fmt(record["mean_difference"]),
                    sd=_fmt(record["std_difference"]),
                    med=_fmt(record["median_difference"]),
                    w=record["wins_primary_lower"],
                    t=record["ties"],
                    l=record["losses_primary_lower"],
                    pval=_fmt(record["p_value"], 4),
                )
            )
        lines.append("")

    by_replicate = result["time_to_first"]["by_replicate"]
    if any(len(replicates) > 1 for replicates in by_replicate.values()):
        lines.append("### Mean first-issue iteration by replicate (seed)")
        lines.append("")
        lines.append("| Policy | Replicate | Arms | Found | Mean (found) | SD (found) | Mean (censored) |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for policy, replicates in sorted(by_replicate.items()):
            for replicate, entry in sorted(replicates.items()):
                lines.append(
                    "| {policy} | {rep} | {n} | {found} | {mf} | {sf} | {mc} |".format(
                        policy=policy,
                        rep=replicate,
                        n=entry["n_arms"],
                        found=entry["n_found"],
                        mf=_fmt(entry["mean_first_found"]),
                        sf=_fmt(entry["std_first_found"]),
                        mc=_fmt(entry["mean_first_censored"]),
                    )
                )
        lines.append("")

    lines.append("## 3. Semantic obligations associated with safety issues")
    lines.append("")
    source = config["association_source"]
    lines.append(
        f"Association source: **{source}**. `Lift` is the unsafe rate when the obligation is "
        "present minus the rate when it is absent; `RR` is their ratio. This is association, not "
        "causation."
    )
    lines.append("")
    lines.append("### Obligations ranked by overall unsafe lift")
    lines.append("")
    lines.append("| Obligation | Support | Unsafe rate (present) | Unsafe rate (absent) | RR | Lift |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    association = result["association"]["targeted"] if source == "target" else result["association"]["witnessed"]
    obligations = association.get("obligations", {}) if association else {}
    for item in result["association"]["top_witnessed"]:
        entry = obligations.get(item["obligation"], {})
        lines.append(
            "| {ob} | {sup} | {pr} | {ar} | {rr} | {lift} |".format(
                ob=item["obligation"],
                sup=item["support"],
                pr=_fmt(item["present_rate"]),
                ar=_fmt(entry.get("unsafe_rate_absent")),
                rr=_risk_ratio(item["present_rate"], entry.get("unsafe_rate_absent")),
                lift=_fmt(item["lift"]),
            )
        )
    lines.append("")
    lines.append("### Leading obligation per issue type (by lift)")
    lines.append("")
    lines.append("| Issue type | Obligation | Support | Rate when present | Lift |")
    lines.append("| --- | --- | --- | --- | --- |")
    for outcome, leaders in result["association"]["outcome_leaders"].items():
        if not leaders:
            continue
        top = leaders[0]
        lines.append(
            "| {outcome} | {ob} | {sup} | {rate} | {lift} |".format(
                outcome=outcome,
                ob=top["obligation"],
                sup=top["support"],
                rate=_fmt(top["present_rate"]),
                lift=_fmt(top["lift"]),
            )
        )
    lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append(
        "- Censored arms (no issue within budget) enter the censored mean/std as `budget + 1`; "
        "the found-only mean/std ignore them and must be read with the found rate."
    )
    lines.append(
        "- Replicates are inferred from a `seed-*` path component; rows without one are pooled "
        "under `all`. Pass `--budget` to fix the censoring horizon across arms."
    )
    lines.append(
        "- Thresholds are engineering proxies; run the report at a second threshold set to check "
        "conclusions. Contract: never mix schema-1 and schema-2 safety metrics."
    )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `safety_utility.json` (machine-readable, includes every arm and association)")
    lines.append("- `safety_utility.md` (this file)")
    lines.append("")
    return "\n".join(lines)


def write_report(result: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "safety_utility.json"
    md_path = out_dir / "safety_utility.md"
    json_path.write_text(json.dumps(_json_safe(result), indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(result) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--search-root",
        type=Path,
        nargs="+",
        default=[DEFAULT_SEARCH_ROOT],
        help="One or more campaign roots; seed-* children are expanded automatically.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--include-paired-controls", action="store_true")
    parser.add_argument("--primary-policy", default=DEFAULT_PRIMARY)
    parser.add_argument("--comparison-policies", nargs="*", default=list(DEFAULT_COMPARISONS))
    parser.add_argument("--budget", type=int, default=None, help="Fixed censoring horizon (evaluations).")
    parser.add_argument("--association-source", choices=["witnessed", "target"], default="witnessed")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--stuck-seconds", type=float, default=8.0)
    parser.add_argument("--near-collision-ttc-s", type=float, default=1.5)
    parser.add_argument("--lane-offset-m", type=float, default=2.0)
    parser.add_argument("--harsh-deceleration-mps2", type=float, default=3.0)
    return parser


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    search_roots = [_resolve(root) for root in args.search_root]
    out_dir = _resolve(args.out_dir) if args.out_dir is not None else search_roots[0]
    thresholds = OutcomeThresholds(
        stuck_seconds=args.stuck_seconds,
        near_collision_ttc_s=args.near_collision_ttc_s,
        lane_offset_m=args.lane_offset_m,
        harsh_deceleration_mps2=args.harsh_deceleration_mps2,
    )
    result = build_utility_report(
        search_roots,
        thresholds=thresholds,
        include_paired_controls=args.include_paired_controls,
        primary_policy=args.primary_policy,
        comparison_policies=args.comparison_policies,
        budget=args.budget,
        association_source=args.association_source,
        top_n=args.top_n,
    )
    paths = write_report(result, out_dir)
    state = result["data_state"]
    print(
        "safety utility report: scored {scored} runs; arms {arms}; policies {policies}".format(
            scored=state["n_scored"],
            arms=state["n_arms"],
            policies=", ".join(state["policies"]) or "none",
        )
    )
    for label, path in paths.items():
        print(f"wrote {path} [{label}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
