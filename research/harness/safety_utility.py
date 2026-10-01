"""Utility analysis for safety-oriented coverage search.

Three questions, all answered from recorded campaign logs:

1. **How many safety issues were found?**  Handled by
   :mod:`research.harness.safety_outcomes` (named outcomes beyond collision).
2. **Which semantic obligations lead to which safety issues?**
   :func:`obligation_association` joins the obligations witnessed (or targeted)
   in each run to the run's classified outcomes and reports per-obligation
   outcome rates plus a risk-ratio/lift against runs that did not witness it.
3. **How quickly is the first issue found?**  :func:`time_to_first` reduces
   each search arm to the first evaluation index with *any* safety issue, and
   :func:`summarize_time_to_first` reports mean/std (and median, found-rate,
   Top-1 rate) across independent runs for each policy, with an explicit
   right-censored value for arms that never found one.

Replicate inference: a campaign root is expected to contain a path component
``seed-<N>`` (as the FSE launchers write).  Rows in a flat layout without one
are pooled under ``all``.  This is inference from the layout, not a stored
field, because the campaign seed is not written into ``rows.jsonl``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from research.harness.safety_outcomes import OUTCOME_NAMES, OutcomeThresholds, classify_run


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

_SEED_COMPONENT = re.compile(r"^seed[-_]?(\d+)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Loading campaign logs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvalRecord:
    """One evaluation row joined to its run log and safety outcome."""

    policy: str
    route: str
    replicate: str
    eval_index: int | None
    outcome: dict[str, Any]
    obligations: tuple[str, ...]
    target: str | None
    run_json_path: str | None


def discover_row_files(root: Path) -> list[tuple[str | None, str, Path]]:
    """Return ``(route_dir, policy_dir, path)`` for every ``rows.jsonl``.

    Recursive so a campaign root works whether it is laid out flat
    (``<root>/<policy>/rows.jsonl``), nested (``<root>/<route>/<policy>/``) or
    multi-seed (``<root>/seed-<N>/<route>/<policy>/``).  The policy is always
    the file's parent directory; the route is the grandparent when present.
    """
    found: list[tuple[str | None, str, Path]] = []
    for path in sorted(root.rglob("rows.jsonl")):
        parent = path.parent
        grandparent = parent.parent
        route_dir = grandparent.name if grandparent != parent else None
        found.append((route_dir, parent.name, path))
    return found


def load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def resolve_run_path(raw: Any, workspace_root: Path = WORKSPACE_ROOT) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = workspace_root / path
    return path if path.exists() else None


def infer_replicate(row: dict[str, Any], row_file: Path, root: Path) -> str:
    for key in ("replicate", "campaign_seed", "search_seed"):
        value = row.get(key)
        if value is not None:
            return f"seed-{value}"
    for part in (*row_file.parts, *root.parts):
        match = _SEED_COMPONENT.match(str(part))
        if match:
            return f"seed-{match.group(1)}"
    return "all"


def _extract_obligations(row: dict[str, Any]) -> tuple[str, ...]:
    for key in ("engine_run_obligations", "semantic_fulfilled_obligations", "semantic_covered_signatures"):
        values = row.get(key)
        if isinstance(values, list) and values:
            return tuple(sorted({str(value) for value in values if value}))
    return ()


def load_records(
    roots: Sequence[Path] | Path,
    *,
    thresholds: OutcomeThresholds | None = None,
    include_paired_controls: bool = False,
    workspace_root: Path = WORKSPACE_ROOT,
) -> tuple[list[EvalRecord], dict[str, int]]:
    """Read every row under ``roots`` and classify its run log.

    Returns ``(records, counters)``.  Rows without a resolvable/unreadable run
    log are counted, never silently dropped.
    """
    root_list = [roots] if isinstance(roots, Path) else list(roots)
    counters = {"n_rows": 0, "n_missing_run_path": 0, "n_unreadable_run": 0}
    records: list[EvalRecord] = []
    seen_files: set[Path] = set()

    for root in root_list:
        for route_dir, policy_dir, path in discover_row_files(root):
            resolved = path.resolve()
            if resolved in seen_files:  # roots may overlap (parent + seed-* children)
                continue
            seen_files.add(resolved)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                policy = str(row.get("policy") or policy_dir)
                route = str(row.get("route") or route_dir or policy_dir)
                replicate = infer_replicate(row, path, root)
                counters["n_rows"] += 1
                base = _make_record(row, policy, route, replicate, workspace_root, thresholds, counters)
                if base is not None:
                    records.append(base)
                if include_paired_controls:
                    control = row.get("paired_control")
                    if isinstance(control, dict):
                        counts_before = dict(counters)
                        control_record = _make_record(
                            control, f"{policy}:control", route, replicate, workspace_root, thresholds, counters
                        )
                        if control_record is not None:
                            records.append(control_record)
                        else:
                            counters.update(counts_before)  # don't double count the parent row

    return records, counters


def _make_record(
    row: dict[str, Any],
    policy: str,
    route: str,
    replicate: str,
    workspace_root: Path,
    thresholds: OutcomeThresholds | None,
    counters: dict[str, int],
) -> EvalRecord | None:
    run_path = resolve_run_path(row.get("run_json_path"), workspace_root)
    if run_path is None:
        counters["n_missing_run_path"] += 1
        return None
    payload = load_json_file(run_path)
    if not isinstance(payload, dict):
        counters["n_unreadable_run"] += 1
        return None
    return EvalRecord(
        policy=policy,
        route=route,
        replicate=replicate,
        eval_index=_int_or_none(row.get("eval_index")),
        outcome=classify_run(payload, thresholds),
        obligations=_extract_obligations(row),
        target=_str_or_none(row.get("hazard_target")),
        run_json_path=str(run_path),
    )


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return None


def _str_or_none(value: Any) -> str | None:
    return str(value) if value else None


# ---------------------------------------------------------------------------
# Time to first safety issue (Top-1 / time-to-first violation)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ArmFirstIssue:
    policy: str
    replicate: str
    route: str
    first_eval: int | None  # None when the arm never found an issue
    budget: int
    censored: bool
    outcomes: tuple[str, ...] = ()


def time_to_first(records: Iterable[EvalRecord], *, budget: int | None = None) -> list[ArmFirstIssue]:
    """Reduce each (policy, replicate, route) arm to its first unsafe eval.

    An arm is one independent search process.  ``first_eval`` is the smallest
    evaluation index whose run had *any* safety outcome; ``budget`` is the
    largest evaluation index observed (or ``budget`` when supplied), and
    ``censored`` marks arms that never found an issue within it.
    """
    grouped: dict[tuple[str, str, str], list[EvalRecord]] = {}
    for record in records:
        grouped.setdefault((record.policy, record.replicate, record.route), []).append(record)

    arms: list[ArmFirstIssue] = []
    for (policy, replicate, route), group in sorted(grouped.items()):
        indexed = [record for record in group if record.eval_index is not None]
        if not indexed:
            continue
        evaluated = max(record.eval_index for record in indexed)  # type: ignore[arg-type]
        arm_budget = int(budget) if budget is not None else int(evaluated)
        unsafe = sorted(
            (record for record in indexed if record.outcome.get("unsafe")),
            key=lambda record: record.eval_index,
        )
        first = unsafe[0].eval_index if unsafe else None
        reasons: set[str] = set()
        for record in unsafe:
            reasons.update(str(name) for name in record.outcome.get("reasons", []))
        arms.append(
            ArmFirstIssue(
                policy=policy,
                replicate=replicate,
                route=route,
                first_eval=first,  # type: ignore[arg-type]
                budget=arm_budget,
                censored=first is None,
                outcomes=tuple(sorted(reasons)),
            )
        )
    return arms


def _mean_std(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = float(sum(values) / len(values))
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def summarize_time_to_first(arms: Sequence[ArmFirstIssue]) -> dict[str, Any]:
    """Per-policy mean/std of first-issue iteration, found-rate and Top-1 rate."""
    by_policy: dict[str, list[ArmFirstIssue]] = {}
    for arm in arms:
        by_policy.setdefault(arm.policy, []).append(arm)

    summary: dict[str, Any] = {}
    for policy, policy_arms in sorted(by_policy.items()):
        found = [arm for arm in policy_arms if not arm.censored]
        found_values = [float(arm.first_eval) for arm in found]  # type: ignore[arg-type]
        censored_values = [
            float(arm.first_eval) if arm.first_eval is not None else float(arm.budget + 1)
            for arm in policy_arms
        ]
        mean_found, std_found = _mean_std(found_values)
        mean_censored, std_censored = _mean_std(censored_values)
        top1 = sum(1 for arm in policy_arms if arm.first_eval == 0)
        summary[policy] = {
            "n_arms": len(policy_arms),
            "n_found": len(found),
            "found_rate": (len(found) / len(policy_arms)) if policy_arms else None,
            "top1_count": top1,
            "top1_rate": (top1 / len(policy_arms)) if policy_arms else None,
            "mean_first_found": mean_found,
            "std_first_found": std_found,
            "median_first_found": _median(found_values),
            "mean_first_censored": mean_censored,
            "std_first_censored": std_censored,
            "mean_budget": _mean_std([float(arm.budget) for arm in policy_arms])[0],
        }
    return summary


def summarize_time_to_first_by_replicate(arms: Sequence[ArmFirstIssue]) -> dict[str, dict[str, Any]]:
    """Per-policy-per-replicate mean first-issue iteration across routes.

    This is the "mean over repeated runs" view when one wants the run to be a
    whole seeded campaign rather than a single route arm.
    """
    grouped: dict[tuple[str, str], list[ArmFirstIssue]] = {}
    for arm in arms:
        grouped.setdefault((arm.policy, arm.replicate), []).append(arm)
    result: dict[str, dict[str, Any]] = {}
    for (policy, replicate), group in sorted(grouped.items()):
        found = [arm for arm in group if not arm.censored]
        found_values = [float(arm.first_eval) for arm in found]  # type: ignore[arg-type]
        censored_values = [
            float(arm.first_eval) if arm.first_eval is not None else float(arm.budget + 1)
            for arm in group
        ]
        mean_found, std_found = _mean_std(found_values)
        mean_censored, std_censored = _mean_std(censored_values)
        result.setdefault(policy, {})[replicate] = {
            "n_arms": len(group),
            "n_found": len(found),
            "found_rate": (len(found) / len(group)) if group else None,
            "mean_first_found": mean_found,
            "std_first_found": std_found,
            "mean_first_censored": mean_censored,
            "std_first_censored": std_censored,
        }
    return result


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def paired_time_to_first(
    arms: Sequence[ArmFirstIssue],
    primary: str,
    comparison: str,
) -> dict[str, Any]:
    """Paired (by replicate+route) difference in censored first-issue iteration.

    Lower is better: a negative mean difference means the primary policy found
    its first safety issue in fewer evaluations.
    """
    primary_map = {
        (arm.replicate, arm.route): (arm.first_eval if arm.first_eval is not None else arm.budget + 1)
        for arm in arms
        if arm.policy == primary
    }
    comparison_map = {
        (arm.replicate, arm.route): (arm.first_eval if arm.first_eval is not None else arm.budget + 1)
        for arm in arms
        if arm.policy == comparison
    }
    shared = sorted(set(primary_map) & set(comparison_map))
    differences = [float(primary_map[key] - comparison_map[key]) for key in shared]
    wins = sum(1 for value in differences if value < 0)
    ties = sum(1 for value in differences if value == 0)
    losses = len(differences) - wins - ties
    mean_diff, std_diff = _mean_std(differences)
    statistic, p_value = _wilcoxon(differences)
    return {
        "primary": primary,
        "comparison": comparison,
        "n_pairs": len(differences),
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "median_difference": _median(differences),
        "wins_primary_lower": wins,
        "ties": ties,
        "losses_primary_lower": losses,
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
    }


def _wilcoxon(differences: Sequence[float]) -> tuple[float | None, float | None]:
    if len(differences) < 2 or all(value == 0.0 for value in differences):
        return None, None
    try:
        from scipy import stats  # local import: optional analysis dependency

        result = stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
    except Exception:  # pragma: no cover - scipy optional/edge cases
        return None, None
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Obligation -> safety-issue association
# ---------------------------------------------------------------------------


def obligation_association(
    records: Iterable[EvalRecord],
    *,
    source: str = "witnessed",
    policy: str | None = None,
    min_support: int = 1,
) -> dict[str, Any]:
    """Associate semantic obligations with safety outcomes across evaluations.

    ``source="witnessed"`` uses the obligations witnessed in the run
    (``engine_run_obligations`` or the semantic fallbacks); ``source="target"``
    uses the obligation the search was actively targeting.  For every
    obligation and outcome it reports the outcome rate when the obligation is
    present, the rate when absent, the risk ratio and the absolute lift.  These
    are associations, not causal effects.
    """
    if source not in {"witnessed", "target"}:
        raise ValueError("source must be 'witnessed' or 'target'")

    rows = [record for record in records if policy is None or record.policy == policy]
    present_sets: list[set[str]] = []
    for record in rows:
        if source == "target":
            present_sets.append({record.target} if record.target else set())
        else:
            present_sets.append(set(record.obligations))

    obligations: set[str] = set()
    for values in present_sets:
        obligations.update(values)

    table: dict[str, dict[str, Any]] = {}
    unsafe_flags = [bool(record.outcome.get("unsafe")) for record in rows]
    for obligation in sorted(obligations):
        mask = [obligation in values for values in present_sets]
        support = sum(1 for flag in mask if flag)
        if support < min_support:
            continue
        absent = len(rows) - support
        entry: dict[str, Any] = {
            "support": support,
            "n_absent": absent,
            "unsafe_when_present": sum(1 for flag, m in zip(unsafe_flags, mask) if m and flag),
            "unsafe_rate_present": (
                sum(1 for flag, m in zip(unsafe_flags, mask) if m and flag) / support if support else None
            ),
            "unsafe_rate_absent": (
                sum(1 for flag, m in zip(unsafe_flags, mask) if not m and flag) / absent if absent else None
            ),
            "outcomes": {},
        }
        for name in OUTCOME_NAMES:
            present_hits = sum(
                1 for record, m in zip(rows, mask) if m and record.outcome.get(name)
            )
            absent_hits = sum(
                1 for record, m in zip(rows, mask) if not m and record.outcome.get(name)
            )
            present_rate = (present_hits / support) if support else None
            absent_rate = (absent_hits / absent) if absent else None
            entry["outcomes"][name] = {
                "present_hits": present_hits,
                "present_rate": present_rate,
                "absent_hits": absent_hits,
                "absent_rate": absent_rate,
                "lift": (present_rate - absent_rate) if present_rate is not None and absent_rate is not None else None,
            }
        entry["risk_ratio_unsafe"] = _risk_ratio(entry["unsafe_rate_present"], entry["unsafe_rate_absent"])
        entry["lift_unsafe"] = (
            entry["unsafe_rate_present"] - entry["unsafe_rate_absent"]
            if entry["unsafe_rate_present"] is not None and entry["unsafe_rate_absent"] is not None
            else None
        )
        table[obligation] = entry

    return {
        "source": source,
        "policy": policy,
        "n_records": len(rows),
        "n_obligations": len(table),
        "obligations": table,
    }


def _risk_ratio(present: float | None, absent: float | None) -> float | None:
    if present is None or absent is None or absent <= 0.0:
        return None
    return present / absent


def rank_associations(association: dict[str, Any], outcome: str = "unsafe", top: int = 20) -> list[dict[str, Any]]:
    """Obligations ranked by outcome lift, for report display."""
    ranked: list[dict[str, Any]] = []
    for obligation, entry in association.get("obligations", {}).items():
        if outcome == "unsafe":
            lift = entry.get("lift_unsafe")
            present_rate = entry.get("unsafe_rate_present")
            support = entry.get("support")
        else:
            detail = entry.get("outcomes", {}).get(outcome, {})
            lift = detail.get("lift")
            present_rate = detail.get("present_rate")
            support = entry.get("support")
        if lift is None:
            continue
        ranked.append(
            {
                "obligation": obligation,
                "support": support,
                "present_rate": present_rate,
                "lift": lift,
            }
        )
    ranked.sort(key=lambda item: (-float(item["lift"]), -int(item["support"] or 0), str(item["obligation"])))
    return ranked[:top]
