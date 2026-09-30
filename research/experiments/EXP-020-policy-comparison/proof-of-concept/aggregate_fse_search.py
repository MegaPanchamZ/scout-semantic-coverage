#!/usr/bin/env python
"""Aggregate FSE-submission coverage-engine metrics from policy-search rows.

Reads every ``rows.jsonl`` under ``--search-root`` and produces:

- ``fse_summary.json``: every computed number (machine-readable).
- ``fse_summary.md``: human-readable tables.
- ``fse_coverage_growth.csv``: tidy per-(route, policy, dimension, eval) curve
  data for plotting (``route,policy,dimension,eval_index,cov,new_obligations``).
- ``fse_discovery.csv``: per-obligation first-uncover eval index per arm.

Supported layouts (both are discovered, never merged ambiguously):

- ``<root>/<route>/<policy>/rows.jsonl`` (pilot layout; the row's ``route``
  field wins when present);
- ``<root>/<policy>/rows.jsonl`` (flat layout; route taken from the row).

Engine layer (opt-in during the search): rows carry ``engine_cov_v/a/e/h``
(suite-level fraction of the mapped obligation subset after this evaluation),
``engine_new_obligations``, ``engine_run_covered_count``,
``engine_uncovered_count`` and ``engine_first_uncover`` (obligation -> first
eval index in this arm). Legacy pilot rows carry none of these keys.

Skip policy (never silent): arms whose rows all lack ``engine_*`` keys are
listed under ``data_state.skipped_arms`` with a reason; engine rows whose
coverage columns are all null (failed streams) are counted per arm and
excluded from metrics; a legacy row inside an engine arm is counted too.

Metrics
-------
- Per-arm coverage trajectory and AUC per dimension. The AUC is the
  trapezoidal integral of the suite coverage curve (already a fraction of the
  mapped subset) over the arm's own eval indices, divided by the eval span
  (``max(eval) - min(eval)``); a single-point arm returns that value.
- Per-policy cross-route summary: mean +/- bootstrap 95% CI over routes
  (percentile method) per AUC dimension, plus final coverage.
- Discovery: per arm, the merged ``engine_first_uncover`` map gives each
  obligation's first-uncover eval index. Reported are count closed, median and
  mean first-uncover, and the eval index to close 25/50/75% of the obligations
  the arm ever closes (linear-interpolated percentiles of the sorted times).
- Paired comparison: ``semantic`` vs each baseline on per-route paired
  differences. Wilcoxon signed-rank is the primary test (only when
  ``n_pairs >= --min-test-routes``, default 6); Mann-Whitney U on the same
  route values is the cross-check. Holm-Bonferroni correction is applied
  across the three comparisons within each metric family. Effect sizes: the
  paired dominance index (wins + 0.5 * ties) / n and the Mann-Whitney
  Vargha-Delaney A12 (U / (n_a * n_b)); the effect label uses the paired
  index. Bootstrap CI of the mean difference is over routes (seed 0, 2000
  resamples by default).

Determinism: all bootstrap resampling uses ``numpy.random.default_rng(seed)``
and every iteration order is sorted. Never writes to the search root's rows.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SEARCH_ROOT = REPO_ROOT / "research" / "logs" / "policy_search"

DIMENSIONS = ("V", "A", "E", "H")
DIM_LABELS = {
    "V": "node",
    "A": "attribute",
    "E": "relation",
    "H": "hazard",
}
COV_KEYS = {dim: f"engine_cov_{dim.lower()}" for dim in DIMENSIONS}
AUC_METRICS = tuple(f"auc_{dim.lower()}" for dim in DIMENSIONS)
DISCOVERY_METRICS = (
    "n_closed",
    "median_first_uncover",
    "mean_first_uncover",
    "q25_close_index",
    "q50_close_index",
    "q75_close_index",
)
PAIRWISE_METRICS = AUC_METRICS + DISCOVERY_METRICS
METRIC_LABELS = {
    **{f"auc_{dim.lower()}": f"AUC {DIM_LABELS[dim]} ({dim})" for dim in DIMENSIONS},
    "n_closed": "obligations closed",
    "median_first_uncover": "median first-uncover eval",
    "mean_first_uncover": "mean first-uncover eval",
    "q25_close_index": "eval to close 25%",
    "q50_close_index": "eval to close 50%",
    "q75_close_index": "eval to close 75%",
}
DEFAULT_COMPARISON_POLICIES = ("random", "lsa", "kmnc")
DEFAULT_PRIMARY_POLICY = "semantic"

TRAJECTORY_FIELDS = tuple(f"cov_{dim.lower()}" for dim in DIMENSIONS)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass
class EngineRow:
    route: str
    policy: str
    eval_index: int | None
    order: int
    cov: dict[str, float | None]
    new_obligations: int | None
    first_uncover: dict[str, int] | None
    run_covered_count: int | None
    uncovered_count: int | None
    engine_error: str | None
    has_engine_keys: bool
    valid: bool


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return None


def _first_uncover_map(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, int] = {}
    for obligation, eval_index in value.items():
        if not isinstance(obligation, str):
            continue
        parsed = _int_or_none(eval_index)
        if parsed is None or parsed < 0:
            continue
        result[obligation] = parsed
    return result


def discover_row_files(root: Path) -> list[tuple[str | None, str, Path]]:
    """Return ``(route_dir, policy_dir, path)`` for every rows.jsonl under root."""
    found: list[tuple[str | None, str, Path]] = []
    seen: set[Path] = set()
    for path in sorted(root.glob("*/*/rows.jsonl")):
        if path not in seen:
            seen.add(path)
            found.append((path.parent.parent.name, path.parent.name, path))
    for path in sorted(root.glob("*/rows.jsonl")):
        if path not in seen:
            seen.add(path)
            found.append((None, path.parent.name, path))
    return found


def load_engine_rows(root: Path) -> dict[str, Any]:
    """Read every rows.jsonl once; separate engine arms, legacy arms and failures."""
    arms: dict[tuple[str, str], list[EngineRow]] = defaultdict(list)
    row_count_by_arm: Counter[tuple[str, str]] = Counter()
    engine_key_arms: Counter[tuple[str, str]] = Counter()
    legacy_rows = 0
    failed_engine_rows = 0
    malformed_lines = 0
    missing_eval_index = 0
    layouts: set[str] = set()

    for route_dir, policy_dir, path in discover_row_files(root):
        layouts.add("nested" if route_dir is not None else "flat")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for order, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
            if not isinstance(row, dict):
                malformed_lines += 1
                continue
            route = str(row.get("route") or route_dir or policy_dir)
            policy = str(row.get("policy") or policy_dir)
            key = (route, policy)
            row_count_by_arm[key] += 1
            has_engine_keys = any(str(name).startswith("engine_") for name in row)
            if not has_engine_keys:
                legacy_rows += 1
                continue
            engine_key_arms[key] += 1
            eval_index = _int_or_none(row.get("eval_index"))
            cov = {dim: _finite_float(row.get(key_name)) for dim, key_name in COV_KEYS.items()}
            first_uncover = _first_uncover_map(row.get("engine_first_uncover"))
            valid = any(value is not None for value in cov.values()) or first_uncover is not None
            if not valid:
                failed_engine_rows += 1
            if eval_index is None and valid:
                missing_eval_index += 1
            arms[key].append(
                EngineRow(
                    route=route,
                    policy=policy,
                    eval_index=eval_index,
                    order=order,
                    cov=cov,
                    new_obligations=_int_or_none(row.get("engine_new_obligations")),
                    first_uncover=first_uncover,
                    run_covered_count=_int_or_none(row.get("engine_run_covered_count")),
                    uncovered_count=_int_or_none(row.get("engine_uncovered_count")),
                    engine_error=row.get("engine_error") if isinstance(row.get("engine_error"), str) else None,
                    has_engine_keys=True,
                    valid=valid,
                )
            )

    skipped_arms: list[dict[str, Any]] = []
    for key, count in sorted(row_count_by_arm.items()):
        if engine_key_arms.get(key, 0) == 0:
            route, policy = key
            skipped_arms.append(
                {
                    "route": route,
                    "policy": policy,
                    "rows": int(count),
                    "reason": "no engine_* fields (legacy pilot row)",
                }
            )

    empty_engine_arms: list[dict[str, Any]] = []
    for key, rows in sorted(arms.items()):
        if all(not row.valid for row in rows):
            route, policy = key
            empty_engine_arms.append(
                {
                    "route": route,
                    "policy": policy,
                    "rows": len(rows),
                    "reason": "engine_* keys present but every row has null engine metrics",
                }
            )

    return {
        "arms": {key: rows for key, rows in arms.items() if any(row.valid for row in rows)},
        "skipped_arms": skipped_arms,
        "empty_engine_arms": empty_engine_arms,
        "legacy_rows": legacy_rows,
        "failed_engine_rows": failed_engine_rows,
        "malformed_lines": malformed_lines,
        "missing_eval_index": missing_eval_index,
        "layouts": sorted(layouts),
        "n_rows_total": sum(row_count_by_arm.values()),
    }


# ---------------------------------------------------------------------------
# Per-arm metrics
# ---------------------------------------------------------------------------


def trajectory_auc(eval_indices: Sequence[int], values: Sequence[float | None]) -> float | None:
    """Trapezoidal AUC of a coverage curve over its own eval axis, normalised by span."""
    by_eval: dict[int, float] = {}
    for eval_index, value in zip(eval_indices, values):
        if value is None or not math.isfinite(float(value)):
            continue
        by_eval[int(eval_index)] = float(value)
    if not by_eval:
        return None
    xs = sorted(by_eval)
    ys = [by_eval[x] for x in xs]
    if len(xs) == 1:
        return float(ys[0])
    span = float(xs[-1] - xs[0])
    if span <= 0.0:
        return float(np.mean(ys))
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(np.asarray(ys, dtype=float), x=np.asarray(xs, dtype=float)) / span)


def merge_first_uncover(rows: Iterable[EngineRow]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for row in rows:
        if row.first_uncover is None:
            continue
        for obligation, eval_index in row.first_uncover.items():
            previous = merged.get(obligation)
            if previous is None or eval_index < previous:
                merged[obligation] = eval_index
    return merged


def close_index(times: Sequence[int], fraction: float) -> float | None:
    """Linear-interpolated percentile of first-uncover times (eval index axis)."""
    if not times:
        return None
    return float(np.percentile(np.asarray(sorted(times), dtype=float), fraction))


def discovery_curve(times: Sequence[int], max_eval: int) -> tuple[list[int], list[float]]:
    ordered = sorted(int(value) for value in times)
    if not ordered:
        return [], []
    end = max(int(max_eval), ordered[-1])
    xs = list(range(0, end + 1))
    ys = [bisect.bisect_right(ordered, x) / float(len(ordered)) for x in xs]
    return xs, ys


def compute_arm_metrics(rows: Sequence[EngineRow]) -> dict[str, Any]:
    valid = [row for row in rows if row.valid]
    ordered = sorted(
        valid,
        key=lambda row: (
            row.eval_index is None,
            row.eval_index if row.eval_index is not None else 0,
            row.order,
        ),
    )
    by_eval: dict[int, EngineRow] = {}
    for row in ordered:
        if row.eval_index is not None:
            by_eval[row.eval_index] = row
    if by_eval:
        eval_indices = sorted(by_eval)
        cov_series = {dim: [by_eval[index].cov[dim] for index in eval_indices] for dim in DIMENSIONS}
    else:
        eval_indices = list(range(len(ordered)))
        cov_series = {dim: [row.cov[dim] for row in ordered] for dim in DIMENSIONS}

    last_by_eval = [by_eval[index] for index in eval_indices] if by_eval else ordered
    final_cov: dict[str, float | None] = {}
    for dim in DIMENSIONS:
        final_cov[dim] = next(
            (row.cov[dim] for row in reversed(last_by_eval) if row.cov[dim] is not None),
            None,
        )

    merged = merge_first_uncover(valid)
    times = sorted(merged.values())
    max_eval = max(eval_indices) if eval_indices else 0
    curve_x, curve_y = discovery_curve(times, max_eval)
    uncovered = next(
        (row.uncovered_count for row in reversed(ordered) if row.uncovered_count is not None),
        None,
    )
    mapped_total = (uncovered + len(merged)) if uncovered is not None else None

    auc = {
        f"auc_{dim.lower()}": trajectory_auc(eval_indices, cov_series[dim]) for dim in DIMENSIONS
    }
    discovery = {
        "n_closed": len(times),
        "median_first_uncover": close_index(times, 50.0),
        "mean_first_uncover": float(np.mean(times)) if times else None,
        "q25_close_index": close_index(times, 25.0),
        "q50_close_index": close_index(times, 50.0),
        "q75_close_index": close_index(times, 75.0),
    }
    new_by_eval = Counter(times)
    return {
        "eval_indices": eval_indices,
        "cov_series": cov_series,
        "new_obligations_by_eval": {str(index): int(count) for index, count in sorted(new_by_eval.items())},
        "auc": auc,
        "final_cov": final_cov,
        "discovery": discovery,
        "discovery_curve": {"eval_indices": curve_x, "fraction": curve_y},
        "first_uncover": dict(sorted(merged.items())),
        "mapped_total": mapped_total,
        "n_valid_rows": len(valid),
        "n_skipped_null_rows": sum(1 for row in rows if not row.valid),
    }


# ---------------------------------------------------------------------------
# Bootstrap / effect sizes
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(
    values: Sequence[float], rng: np.random.Generator, n_boot: int
) -> tuple[float, float] | None:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    if array.size == 1:
        return float(array[0]), float(array[0])
    indices = rng.integers(0, array.size, size=(n_boot, array.size))
    means = array[indices].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def holm_bonferroni(pvalues: Sequence[float]) -> list[float]:
    count = len(pvalues)
    order = sorted(range(count), key=lambda index: pvalues[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * pvalues[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def effect_label(index: float | None) -> str | None:
    if index is None or not math.isfinite(index):
        return None
    delta = abs(index - 0.5)
    if delta < 0.06:
        return "negligible"
    if delta < 0.14:
        return "small"
    if delta < 0.21:
        return "medium"
    return "large"


def _mann_whitney(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    try:
        result = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    except TypeError:  # pragma: no cover - scipy < 1.12 fallback
        result = stats.mannwhitneyu(x, y, alternative="two-sided", exact=False)
    return float(result.statistic), float(result.pvalue)


def _wilcoxon(
    differences: Sequence[float], min_routes: int
) -> tuple[float | None, float | None, str | None]:
    if len(differences) < min_routes:
        return None, None, f"wilcoxon requires n >= {min_routes} paired routes"
    if all(value == 0.0 for value in differences):
        return 0.0, 1.0, None
    try:
        result = stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
    except ValueError as exc:  # pragma: no cover - defensive
        return None, None, f"wilcoxon failed: {exc}"
    return float(result.statistic), float(result.pvalue), None


def paired_comparison(
    metric: str,
    policy_a: str,
    policy_b: str,
    values_a: dict[str, float],
    values_b: dict[str, float],
    rng: np.random.Generator,
    n_boot: int,
    min_test_routes: int,
) -> dict[str, Any]:
    routes = sorted(route for route in values_a if route in values_b)
    differences = [values_a[route] - values_b[route] for route in routes]
    n_pairs = len(differences)
    record: dict[str, Any] = {
        "metric": metric,
        "policy_a": policy_a,
        "policy_b": policy_b,
        "n_pairs": n_pairs,
        "routes": routes,
        "testable": n_pairs >= min_test_routes,
        "min_test_routes": min_test_routes,
        "median_difference": float(np.median(differences)) if differences else None,
        "mean_difference": float(np.mean(differences)) if differences else None,
        "mean_difference_ci": None,
        "w": None,
        "p_wilcoxon": None,
        "p_wilcoxon_holm": None,
        "u": None,
        "p_mannwhitney": None,
        "p_mannwhitney_holm": None,
        "a12": None,
        "a12_paired": None,
        "effect": None,
        "wins": None,
        "ties": None,
        "losses": None,
        "note": None,
    }
    if not differences:
        record["note"] = "no route has both arms scored"
        return record
    ci = bootstrap_mean_ci(differences, rng, n_boot)
    record["mean_difference_ci"] = [ci[0], ci[1]] if ci else None
    wins = sum(1 for value in differences if value > 0)
    ties = sum(1 for value in differences if value == 0)
    losses = n_pairs - wins - ties
    record["wins"], record["ties"], record["losses"] = wins, ties, losses
    record["a12_paired"] = (wins + 0.5 * ties) / n_pairs
    if n_pairs >= 2:
        values_a_list = [values_a[route] for route in routes]
        values_b_list = [values_b[route] for route in routes]
        u_stat, p_value = _mann_whitney(values_a_list, values_b_list)
        record["u"] = u_stat
        record["p_mannwhitney"] = p_value
        record["a12"] = u_stat / (n_pairs * n_pairs)
    if record["testable"]:
        w_stat, p_value, error = _wilcoxon(differences, min_test_routes)
        record["w"] = w_stat
        record["p_wilcoxon"] = p_value
        if error:
            record["note"] = error
    else:
        record["note"] = (
            f"not testable: {n_pairs} paired route(s), need >= {min_test_routes}"
        )
    record["effect"] = effect_label(record["a12_paired"])
    return record


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def policy_order(policies: Iterable[str], primary: str, comparisons: Sequence[str]) -> list[str]:
    known = {primary, *comparisons}
    ordered = [policy for policy in (primary, *comparisons) if policy in policies]
    ordered.extend(sorted(policy for policy in policies if policy not in known))
    return ordered


def compute_fse_statistics(
    search_root: Path,
    *,
    seed: int = 0,
    n_boot: int = 2000,
    primary_policy: str = DEFAULT_PRIMARY_POLICY,
    comparison_policies: Sequence[str] = DEFAULT_COMPARISON_POLICIES,
    min_test_routes: int = 6,
) -> dict[str, Any]:
    loaded = load_engine_rows(search_root)
    arms: dict[tuple[str, str], list[EngineRow]] = loaded["arms"]
    rng = np.random.default_rng(seed)

    arm_records: list[dict[str, Any]] = []
    metrics_by_arm: dict[tuple[str, str], dict[str, Any]] = {}
    for (route, policy), rows in sorted(arms.items()):
        computed = compute_arm_metrics(rows)
        metrics_by_arm[(route, policy)] = computed
        arm_records.append(
            {
                "route": route,
                "policy": policy,
                "n_valid_rows": computed["n_valid_rows"],
                "n_skipped_null_rows": computed["n_skipped_null_rows"],
                "mapped_total": computed["mapped_total"],
                "auc": computed["auc"],
                "discovery": computed["discovery"],
                "final_cov": computed["final_cov"],
                "eval_indices": computed["eval_indices"],
                "cov_series": computed["cov_series"],
                "new_obligations_by_eval": computed["new_obligations_by_eval"],
                "first_uncover": computed["first_uncover"],
                "discovery_curve": computed["discovery_curve"],
            }
        )

    routes = sorted({route for route, _ in arms})
    policies = policy_order({policy for _, policy in arms}, primary_policy, comparison_policies)
    metrics = [*AUC_METRICS, *DISCOVERY_METRICS]

    # Cross-route summaries per policy.
    cross_route: dict[str, Any] = {}
    for policy in policies:
        policy_routes = [route for route in routes if (route, policy) in metrics_by_arm]
        entry: dict[str, Any] = {
            "n_routes": len(policy_routes),
            "routes": policy_routes,
            "auc_mean": {},
            "auc_ci": {},
            "final_cov_mean": {},
            "final_cov_ci": {},
            "discovery": {},
        }
        for metric in metrics:
            values = [
                metrics_by_arm[(route, policy)][
                    "auc" if metric in AUC_METRICS else "discovery"
                ][metric]
                for route in policy_routes
            ]
            clean = [float(value) for value in values if value is not None]
            mean = float(np.mean(clean)) if clean else None
            ci = bootstrap_mean_ci(clean, rng, n_boot)
            if metric in AUC_METRICS:
                entry["auc_mean"][metric] = mean
                entry["auc_ci"][metric] = [ci[0], ci[1]] if ci else None
            else:
                entry["discovery"][metric] = {
                    "mean": mean,
                    "mean_ci": [ci[0], ci[1]] if ci else None,
                }
        for dim in DIMENSIONS:
            values = [
                metrics_by_arm[(route, policy)]["final_cov"][dim]
                for route in policy_routes
            ]
            clean = [float(value) for value in values if value is not None]
            mean = float(np.mean(clean)) if clean else None
            ci = bootstrap_mean_ci(clean, rng, n_boot)
            entry["final_cov_mean"][dim] = mean
            entry["final_cov_ci"][dim] = [ci[0], ci[1]] if ci else None
        pooled_times = sorted(
            value
            for route in policy_routes
            for value in metrics_by_arm[(route, policy)]["first_uncover"].values()
        )
        entry["discovery"]["pooled"] = {
            "n_obligations": len(pooled_times),
            "median_first_uncover": close_index(pooled_times, 50.0),
            "mean_first_uncover": float(np.mean(pooled_times)) if pooled_times else None,
            "q25_close_index": close_index(pooled_times, 25.0),
            "q50_close_index": close_index(pooled_times, 50.0),
            "q75_close_index": close_index(pooled_times, 75.0),
        }
        cross_route[policy] = entry

    # Paired comparisons (primary vs each baseline) per metric family.
    pairwise: dict[str, list[dict[str, Any]]] = {}
    comparison_targets = [policy for policy in comparison_policies if policy != primary_policy]
    for metric in metrics:
        records: list[dict[str, Any]] = []
        for other in comparison_targets:
            values_a: dict[str, float] = {}
            values_b: dict[str, float] = {}
            for route in routes:
                first = metrics_by_arm.get((route, primary_policy))
                second = metrics_by_arm.get((route, other))
                if first is None or second is None:
                    continue
                value_a = (first["auc"] if metric in AUC_METRICS else first["discovery"]).get(metric)
                value_b = (second["auc"] if metric in AUC_METRICS else second["discovery"]).get(metric)
                if value_a is None or value_b is None:
                    continue
                values_a[route] = float(value_a)
                values_b[route] = float(value_b)
            records.append(
                paired_comparison(
                    metric,
                    primary_policy,
                    other,
                    values_a,
                    values_b,
                    rng,
                    n_boot,
                    min_test_routes,
                )
            )
        for family, key in (
            ("wilcoxon", "p_wilcoxon"),
            ("mannwhitney", "p_mannwhitney"),
        ):
            scorable = [record for record in records if record[key] is not None]
            if scorable:
                adjusted = holm_bonferroni([record[key] for record in scorable])
                for record, value in zip(scorable, adjusted):
                    record[f"p_{family}_holm"] = value
        pairwise[metric] = records

    data_state = {
        "search_root": str(search_root),
        "layouts": loaded["layouts"],
        "n_rows_total": loaded["n_rows_total"],
        "n_arms_discovered": len(loaded["skipped_arms"]) + len(loaded["empty_engine_arms"]) + len(arms),
        "n_arms_with_engine": len(arms),
        "n_arms_skipped_no_engine": len(loaded["skipped_arms"]),
        "n_arms_engine_no_valid_rows": len(loaded["empty_engine_arms"]),
        "skipped_arms": loaded["skipped_arms"],
        "empty_engine_arms": loaded["empty_engine_arms"],
        "legacy_rows": loaded["legacy_rows"],
        "failed_engine_rows": loaded["failed_engine_rows"],
        "malformed_lines": loaded["malformed_lines"],
        "missing_eval_index_rows": loaded["missing_eval_index"],
    }

    config = {
        "search_root": str(search_root),
        "seed": seed,
        "bootstrap": n_boot,
        "primary_policy": primary_policy,
        "comparison_policies": list(comparison_policies),
        "min_test_routes": min_test_routes,
    }
    return {
        "config": config,
        "data_state": data_state,
        "policies": policies,
        "routes": routes,
        "arms": arm_records,
        "metrics": metrics,
        "cross_route": cross_route,
        "pairwise": pairwise,
        "skipped_arms": loaded["skipped_arms"],
        "empty_engine_arms": loaded["empty_engine_arms"],
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_coverage_growth_csv(result: dict[str, Any], path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["route", "policy", "dimension", "eval_index", "cov", "new_obligations"])
        for record in result["arms"]:
            new_by_eval = {int(key): value for key, value in record["new_obligations_by_eval"].items()}
            for position, eval_index in enumerate(record["eval_indices"]):
                for dim in DIMENSIONS:
                    value = record["cov_series"][dim][position]
                    writer.writerow(
                        [
                            record["route"],
                            record["policy"],
                            dim,
                            eval_index,
                            "" if value is None else f"{float(value):.6f}",
                            new_by_eval.get(eval_index, 0),
                        ]
                    )
    return path


def write_discovery_csv(result: dict[str, Any], path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["route", "policy", "obligation", "first_uncover"])
        for record in result["arms"]:
            for obligation, eval_index in record["first_uncover"].items():
                writer.writerow([record["route"], record["policy"], obligation, eval_index])
    return path


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


def _fmt_ci(ci: Any, digits: int = 3) -> str:
    if not ci or len(ci) != 2 or ci[0] is None:
        return "—"
    return f"{_fmt(ci[0], digits)}–{_fmt(ci[1], digits)}"


def build_markdown(result: dict[str, Any]) -> str:
    config = result["config"]
    state = result["data_state"]
    lines: list[str] = []
    lines.append("# FSE policy search: coverage-engine aggregation summary")
    lines.append("")
    lines.append(
        "Generated by `research/experiments/EXP-020-policy-comparison/proof-of-concept/"
        "aggregate_fse_search.py` from `{}`.".format(config["search_root"])
    )
    lines.append("")

    lines.append("## Data state")
    lines.append("")
    lines.append(f"- Layouts discovered: {', '.join(state['layouts']) or 'none'}")
    lines.append(
        f"- Rows read: {state['n_rows_total']}; engine arms with valid rows: {state['n_arms_with_engine']}; "
        f"arms skipped (no `engine_*` fields): {state['n_arms_skipped_no_engine']}; "
        f"engine arms with no valid rows: {state['n_arms_engine_no_valid_rows']}."
    )
    lines.append(
        f"- Legacy rows skipped: {state['legacy_rows']}; engine rows skipped (null metrics): "
        f"{state['failed_engine_rows']}; malformed JSON lines: {state['malformed_lines']}; "
        f"valid engine rows missing `eval_index`: {state['missing_eval_index_rows']}."
    )
    if state["skipped_arms"]:
        lines.append("")
        lines.append("### Arms skipped (no engine metrics)")
        lines.append("")
        lines.append("| Route | Policy | Rows | Reason |")
        lines.append("| --- | --- | --- | --- |")
        for arm in state["skipped_arms"]:
            lines.append(f"| {arm['route']} | {arm['policy']} | {arm['rows']} | {arm['reason']} |")
    if state["empty_engine_arms"]:
        lines.append("")
        lines.append("### Engine arms with no valid rows")
        lines.append("")
        lines.append("| Route | Policy | Rows | Reason |")
        lines.append("| --- | --- | --- | --- |")
        for arm in state["empty_engine_arms"]:
            lines.append(f"| {arm['route']} | {arm['policy']} | {arm['rows']} | {arm['reason']} |")
    lines.append("")

    lines.append("## Per-arm engine metrics")
    lines.append("")
    lines.append(
        "AUC is the trapezoidal integral of the suite coverage curve over the arm's own eval "
        "indices, divided by the eval span (mean covered fraction over the arm)."
    )
    lines.append("")
    lines.append(
        "| Route | Policy | n valid | n null | AUC node | AUC attr | AUC rel | AUC hazard | "
        "Closed | Median first-uncover | q25 close | q50 close | q75 close |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for record in sorted(result["arms"], key=lambda item: (item["route"], item["policy"])):
        auc = record["auc"]
        discovery = record["discovery"]
        lines.append(
            "| {route} | {policy} | {n} | {nulls} | {v} | {a} | {e} | {h} | {closed} | "
            "{median} | {q25} | {q50} | {q75} |".format(
                route=record["route"],
                policy=record["policy"],
                n=record["n_valid_rows"],
                nulls=record["n_skipped_null_rows"],
                v=_fmt(auc["auc_v"]),
                a=_fmt(auc["auc_a"]),
                e=_fmt(auc["auc_e"]),
                h=_fmt(auc["auc_h"]),
                closed=discovery["n_closed"],
                median=_fmt(discovery["median_first_uncover"]),
                q25=_fmt(discovery["q25_close_index"]),
                q50=_fmt(discovery["q50_close_index"]),
                q75=_fmt(discovery["q75_close_index"]),
            )
        )
    lines.append("")

    lines.append("## Cross-route summary per policy")
    lines.append("")
    lines.append(
        f"Mean and bootstrap 95% CI over routes ({config['bootstrap']} resamples, seed "
        f"{config['seed']}); each route weighted equally. Discovery values are route-macro means "
        "of the per-arm metrics."
    )
    lines.append("")
    lines.append("| Policy | Routes | AUC node | AUC attr | AUC rel | AUC hazard |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for policy in result["policies"]:
        entry = result["cross_route"][policy]
        cells = []
        for metric in AUC_METRICS:
            mean = entry["auc_mean"].get(metric)
            ci = entry["auc_ci"].get(metric)
            cells.append(f"{_fmt(mean)} [{_fmt_ci(ci)}]" if mean is not None else "—")
        lines.append(
            f"| {policy} | {entry['n_routes']} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    lines.append("| Policy | Closed (pooled) | Median first-uncover (pooled) | Mean first-uncover (pooled) | "
                 "q25 close (pooled) | q50 close (pooled) | q75 close (pooled) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for policy in result["policies"]:
        pooled = result["cross_route"][policy]["discovery"]["pooled"]
        lines.append(
            "| {policy} | {closed} | {median} | {mean} | {q25} | {q50} | {q75} |".format(
                policy=policy,
                closed=pooled["n_obligations"],
                median=_fmt(pooled["median_first_uncover"]),
                mean=_fmt(pooled["mean_first_uncover"]),
                q25=_fmt(pooled["q25_close_index"]),
                q50=_fmt(pooled["q50_close_index"]),
                q75=_fmt(pooled["q75_close_index"]),
            )
        )
    lines.append("")

    lines.append("## Paired comparisons: {} vs baselines".format(config["primary_policy"]))
    lines.append("")
    lines.append(
        "Wilcoxon signed-rank on per-route paired differences (primary test, requires "
        f">= {config['min_test_routes']} paired routes); Mann-Whitney U on the same route values "
        "as an unpaired cross-check. Holm-Bonferroni within each metric across the three "
        "comparisons. Effect sizes: paired dominance index and Vargha-Delaney A12 (U / (n_a * n_b))."
    )
    lines.append("")
    for metric in result["metrics"]:
        records = result["pairwise"][metric]
        lines.append(f"### {METRIC_LABELS[metric]}")
        lines.append("")
        lines.append(
            "| Comparison | n pairs | Mean diff [95% CI] | Median diff | W | p (Wilcoxon) | "
            "p Holm | A12 paired | A12 MWU | Effect | MWU U | p (MWU) | Notes |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for record in records:
            lines.append(
                "| {a} vs {b} | {n} | {mean} [{ci}] | {median} | {w} | {pw} | {pwh} | {a12p} | "
                "{a12} | {effect} | {u} | {pm} | {note} |".format(
                    a=record["policy_a"],
                    b=record["policy_b"],
                    n=record["n_pairs"],
                    mean=_fmt(record["mean_difference"]),
                    ci=_fmt_ci(record["mean_difference_ci"]),
                    median=_fmt(record["median_difference"]),
                    w=_fmt(record["w"], 1),
                    pw=_fmt(record["p_wilcoxon"], 4),
                    pwh=_fmt(record["p_wilcoxon_holm"], 4),
                    a12p=_fmt(record["a12_paired"]),
                    a12=_fmt(record["a12"]),
                    effect=record["effect"] or "—",
                    u=_fmt(record["u"], 1),
                    pm=_fmt(record["p_mannwhitney"], 4),
                    note=record["note"] or "—",
                )
            )
        lines.append("")

    lines.append("## Method notes")
    lines.append("")
    lines.append(
        "- Trajectories use valid engine rows only (coverage keys present and not all null), "
        "ordered by `eval_index`; duplicate eval indices keep the last row written."
    )
    lines.append(
        "- Discovery merges `engine_first_uncover` across rows by minimum eval index, which "
        "reconstructs an arm's first-uncover map across resume boundaries."
    )
    lines.append(
        "- `time to close X%` is the linear-interpolated percentile of the sorted first-uncover "
        "eval indices over the obligations the arm ever closes (q50 equals the median)."
    )
    lines.append(
        "- Legacy (pilot) rows are never silently dropped: every skipped arm is listed under "
        "`data_state.skipped_arms` and counted."
    )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `fse_summary.json` (machine-readable)")
    lines.append("- `fse_summary.md` (this file)")
    lines.append("- `fse_coverage_growth.csv` (route, policy, dimension, eval_index, cov, new_obligations)")
    lines.append("- `fse_discovery.csv` (route, policy, obligation, first_uncover)")
    lines.append("")
    return "\n".join(lines)


def write_fse_outputs(result: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fse_summary.json"
    md_path = out_dir / "fse_summary.md"
    growth_path = out_dir / "fse_coverage_growth.csv"
    discovery_path = out_dir / "fse_discovery.csv"
    json_path.write_text(
        json.dumps(_json_safe(result), indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(build_markdown(result) + "\n", encoding="utf-8")
    write_coverage_growth_csv(result, growth_path)
    write_discovery_csv(result, discovery_path)
    return {
        "json": json_path,
        "markdown": md_path,
        "coverage_growth_csv": growth_path,
        "discovery_csv": discovery_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate FSE coverage-engine metrics from policy-search rows."
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=DEFAULT_SEARCH_ROOT,
        help="Root containing <route>/<policy>/rows.jsonl or <policy>/rows.jsonl.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for fse_summary.* and CSVs (default: the search root).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed.")
    parser.add_argument(
        "--bootstrap", type=int, default=2000, help="Bootstrap resamples over routes."
    )
    parser.add_argument("--primary-policy", default=DEFAULT_PRIMARY_POLICY)
    parser.add_argument(
        "--comparison-policies",
        nargs="*",
        default=list(DEFAULT_COMPARISON_POLICIES),
        help="Baselines compared against the primary policy.",
    )
    parser.add_argument(
        "--min-test-routes",
        type=int,
        default=6,
        help="Minimum paired routes for the Wilcoxon signed-rank test.",
    )
    return parser


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    search_root = _resolve(args.search_root)
    out_dir = _resolve(args.out_dir) if args.out_dir is not None else search_root
    result = compute_fse_statistics(
        search_root,
        seed=args.seed,
        n_boot=args.bootstrap,
        primary_policy=args.primary_policy,
        comparison_policies=args.comparison_policies,
        min_test_routes=args.min_test_routes,
    )
    paths = write_fse_outputs(result, out_dir)
    state = result["data_state"]
    print(
        "engine arms: {arms}; skipped arms without engine_* fields: {skipped} "
        "(legacy rows: {legacy}); engine rows skipped with null metrics: {failed}".format(
            arms=state["n_arms_with_engine"],
            skipped=state["n_arms_skipped_no_engine"],
            legacy=state["legacy_rows"],
            failed=state["failed_engine_rows"],
        )
    )
    for arm in state["skipped_arms"]:
        print(
            "  skipped arm {route}/{policy}: {reason} ({rows} rows)".format(**arm)
        )
    for label, path in paths.items():
        print(f"wrote {path} [{label}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
