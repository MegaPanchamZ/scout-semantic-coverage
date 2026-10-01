#!/usr/bin/env python
"""Aggregate and statistically summarise the EXP-020 matched-budget policy search.

Reads every ``rows.jsonl`` under ``--root/<route>/<policy>/`` and produces:

- ``policy_search_summary.json``: every computed number.
- ``policy_search_summary.md``: human-readable tables.
- figures under ``research/logs/policy_search/figures/``.

Design notes
------------
- **Re-runnable while the search is writing.** Each ``rows.jsonl`` is read once
  per run; malformed/partial lines are skipped and counted. Rows with
  ``ticks_executed is None`` (failed evaluations) are excluded from metrics but
  counted and reported per arm.
- **Deterministic for a fixed input snapshot.** All bootstrap resampling uses
  ``numpy.random.default_rng(--seed)``, and all iteration order is sorted.
- **Coverage curves** for a (route, policy) are the cumulative union of
  ``semantic_covered_predicates`` (and separately signatures) over successful
  evaluations in ``eval_index`` order, de-duplicated by ``radius`` when a radius
  repeats. Two normalisations are reported: raw union size and union size
  divided by the per-route attainable union (union across the non-control
  policies on that route; the with-control variant is reported as a delta).
- **Coverage AUC** is the trapezoidal integral of the normalised curve over the
  candidate index, divided by ``m - 1`` (``m`` = number of unique candidates),
  i.e. the mean normalised coverage over the search.
- **Pairwise tests** are Mann-Whitney U (asymptotic) on per-route AUCs with a
  Holm-Bonferroni correction across the semantic-vs-others family, plus the
  Vargha-Delaney A effect size. Comparisons with fewer than
  ``--min-test-routes`` (default 4) routes are reported as not testable and no
  p-value is emitted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]

EXPECTED_ROUTES_DEFAULT = (
    "town01_spawn0_goal82",
    "town01_spawn115_goal206",
    "town01_spawn195_goal197",
    "town01_spawn55_goal154",
    "town01_spawn68_goal218",
    "town01_spawn82_goal200",
    "town02_spawn40_goal33",
    "town02_spawn98_goal35",
)
SEARCH_POLICIES_DEFAULT = ("random", "lsa", "kmnc", "semantic")
CONTROL_POLICY = "control"
POLICY_ORDER = ("random", "lsa", "kmnc", "semantic", "control")
POLICY_COLORS = {
    "random": "#4C72B0",
    "lsa": "#DD8452",
    "kmnc": "#55A868",
    "semantic": "#C44E52",
    "control": "#8172B3",
}
ACTOR_CLASSES = ("static.*", "walker.*", "vehicle.*", "other", "no_actor_recorded")
FIG_PREDICATES = "cumulative_predicates_vs_eval.png"
FIG_SIGNATURES = "cumulative_signatures_vs_eval.png"
FIG_AUC = "coverage_auc_by_policy.png"
FIG_TRAJECTORIES = "kmnc_lsa_trajectories.png"
FIG_FAILURES = "failure_taxonomy.png"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    policy: str
    route: str
    eval_index: int | None
    radius: float | None
    ticks: int
    reached_goal: bool
    collision_count: int | None
    terminated_by_collision: bool
    collision_actors: tuple[str, ...]
    kmnc: float | None
    lsa_max: float | None
    lsa_mean: float | None
    fulfilled: tuple[str, ...]
    missing: tuple[str, ...]
    predicates: tuple[str, ...]
    signatures: tuple[str, ...]
    run_error: str | None
    order: int

    @property
    def critical(self) -> bool:
        return bool(self.collision_count and self.collision_count > 0)


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


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if item is not None)


def load_episodes(
    root: Path,
    expected_routes: Sequence[str],
    expected_policies: Sequence[str],
) -> dict[str, Any]:
    """Read every rows.jsonl exactly once and return valid episodes per arm."""
    episodes: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    failed = Counter()
    malformed_lines = 0
    duplicate_eval_indices = 0
    discovered_routes: set[str] = set()
    discovered_policies: set[str] = set()
    arm_paths: dict[tuple[str, str], Path] = {}

    for path in sorted(root.glob("*/*/rows.jsonl")):
        route_dir = path.parent.parent.name
        policy_dir = path.parent.name
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        seen_eval: set[int] = set()
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
            route = str(row.get("route") or route_dir)
            policy = str(row.get("policy") or policy_dir)
            discovered_routes.add(route)
            discovered_policies.add(policy)
            arm_paths[(route, policy)] = path
            eval_index = _int_or_none(row.get("eval_index"))
            if eval_index is not None:
                if eval_index in seen_eval:
                    # Distinct candidates can share an eval_index after a writer
                    # restart; keep both rows (legitimate evaluations) and count
                    # the collision for the integrity report.
                    duplicate_eval_indices += 1
                seen_eval.add(eval_index)
            ticks = _int_or_none(row.get("ticks_executed"))
            if ticks is None:
                failed[(route, policy)] += 1
                continue
            episodes[(route, policy)].append(
                Episode(
                    policy=policy,
                    route=route,
                    eval_index=eval_index,
                    radius=_finite_float(row.get("radius")),
                    ticks=ticks,
                    reached_goal=bool(row.get("reached_goal")),
                    collision_count=_int_or_none(row.get("collision_count")),
                    terminated_by_collision=bool(row.get("terminated_by_collision")),
                    collision_actors=_str_tuple(row.get("collision_actors")),
                    kmnc=_finite_float(row.get("coverage_kmnc")),
                    lsa_max=_finite_float(row.get("coverage_lsa_max")),
                    lsa_mean=_finite_float(row.get("coverage_lsa_mean")),
                    fulfilled=_str_tuple(row.get("semantic_fulfilled_obligations")),
                    missing=_str_tuple(row.get("semantic_missing_obligations")),
                    predicates=_str_tuple(row.get("semantic_covered_predicates")),
                    signatures=_str_tuple(row.get("semantic_covered_signatures")),
                    run_error=row.get("run_error") if isinstance(row.get("run_error"), str) else None,
                    order=order,
                )
            )

    for arm in episodes:
        episodes[arm].sort(
            key=lambda ep: (
                ep.eval_index if ep.eval_index is not None else 10**9,
                ep.order,
            )
        )

    return {
        "episodes": dict(episodes),
        "failed": dict(failed),
        "malformed_lines": malformed_lines,
        "duplicate_eval_indices": duplicate_eval_indices,
        "discovered_routes": discovered_routes,
        "discovered_policies": discovered_policies,
        "arm_paths": arm_paths,
    }


# ---------------------------------------------------------------------------
# Coverage curves / AUC
# ---------------------------------------------------------------------------


def unique_candidate_episodes(episodes: Sequence[Episode]) -> list[Episode]:
    """Drop repeated radii (keep first occurrence in eval_index order)."""
    seen: set[float] = set()
    result: list[Episode] = []
    for episode in episodes:
        if episode.radius is not None:
            key = round(episode.radius, 6)
            if key in seen:
                continue
            seen.add(key)
        result.append(episode)
    return result


def union_of(episodes: Iterable[Episode], field: str) -> set[str]:
    result: set[str] = set()
    for episode in episodes:
        result.update(getattr(episode, field))
    return result


def cumulative_union_curve(episodes: Sequence[Episode], field: str) -> list[int]:
    seen: set[str] = set()
    curve: list[int] = []
    for episode in episodes:
        seen.update(getattr(episode, field))
        curve.append(len(seen))
    return curve


def trapz_auc(curve: Sequence[float]) -> float | None:
    if len(curve) < 2:
        return None
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(np.asarray(curve, dtype=float), dx=1.0) / (len(curve) - 1))


# ---------------------------------------------------------------------------
# Bootstrap helpers
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


def bootstrap_band(
    matrix: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and 95% CI band over rows (routes); NaN entries are ignored per column."""
    n_routes, length = matrix.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(matrix, axis=0)
        if n_routes == 1:
            return mean, mean.copy(), mean.copy()
        indices = rng.integers(0, n_routes, size=(n_boot, n_routes))
        sampled = matrix[indices]
        boot_means = np.nanmean(sampled, axis=1)
        lo = np.nanpercentile(boot_means, 2.5, axis=0)
        hi = np.nanpercentile(boot_means, 97.5, axis=0)
    return mean, lo, hi


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mann_whitney(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    try:
        result = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    except TypeError:  # pragma: no cover - scipy < 1.12 fallback
        result = stats.mannwhitneyu(x, y, alternative="two-sided", exact=False)
    return float(result.statistic), float(result.pvalue)


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


def vargha_delaney_a(u_statistic: float, n_a: int, n_b: int) -> float:
    if n_a == 0 or n_b == 0:
        return float("nan")
    return float(u_statistic) / (n_a * n_b)


def effect_label(a12: float | None) -> str | None:
    if a12 is None or not math.isfinite(a12):
        return None
    delta = abs(a12 - 0.5)
    if delta < 0.06:
        return "negligible"
    if delta < 0.14:
        return "small"
    if delta < 0.21:
        return "medium"
    return "large"


def actor_class(actor: str) -> str:
    if actor.startswith("static."):
        return "static.*"
    if actor.startswith("walker."):
        return "walker.*"
    if actor.startswith("vehicle."):
        return "vehicle.*"
    return "other"


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute_statistics(
    root: Path,
    expected_routes: Sequence[str],
    expected_policies: Sequence[str],
    target_evals: int,
    target_controls: int,
    primary_policy: str,
    rng: np.random.Generator,
    n_boot: int,
    min_arm_evals: int,
    min_test_routes: int,
) -> dict[str, Any]:
    loaded = load_episodes(root, expected_routes, expected_policies)
    episodes: dict[tuple[str, str], list[Episode]] = loaded["episodes"]
    failed: dict[tuple[str, str], int] = loaded["failed"]

    policies = [p for p in POLICY_ORDER if p in loaded["discovered_policies"] or p in expected_policies]
    for policy in sorted(loaded["discovered_policies"]):
        if policy not in policies:
            policies.append(policy)
    routes = sorted(set(expected_routes) | loaded["discovered_routes"])

    def target_for(policy: str) -> int:
        return target_controls if policy == CONTROL_POLICY else target_evals

    # Per-route attainable unions (search policies, and with-control variant).
    attainable: dict[str, dict[str, Any]] = {}
    for route in routes:
        predicate_union: set[str] = set()
        signature_union: set[str] = set()
        control_predicates: set[str] = set()
        control_signatures: set[str] = set()
        for (arm_route, policy), arm_episodes in sorted(episodes.items()):
            if arm_route != route:
                continue
            if policy == CONTROL_POLICY:
                control_predicates.update(union_of(arm_episodes, "predicates"))
                control_signatures.update(union_of(arm_episodes, "signatures"))
            else:
                predicate_union.update(union_of(arm_episodes, "predicates"))
                signature_union.update(union_of(arm_episodes, "signatures"))
        attainable[route] = {
            "predicates": sorted(predicate_union),
            "signatures": sorted(signature_union),
            "n_predicates": len(predicate_union),
            "n_signatures": len(signature_union),
            "control_only_predicates": sorted(control_predicates - predicate_union),
            "control_only_signatures": sorted(control_signatures - signature_union),
            "n_predicates_with_control": len(predicate_union | control_predicates),
            "n_signatures_with_control": len(signature_union | control_signatures),
        }

    # Per-arm curves and AUCs.
    arm_rows: list[dict[str, Any]] = []
    aucs: dict[tuple[str, str], dict[str, float | None]] = {}
    for route in routes:
        for policy in policies:
            key = (route, policy)
            arm_episodes = episodes.get(key, [])
            candidates = unique_candidate_episodes(arm_episodes)
            predicate_curve = cumulative_union_curve(candidates, "predicates")
            signature_curve = cumulative_union_curve(candidates, "signatures")
            n_pred = attainable[route]["n_predicates"]
            n_sig = attainable[route]["n_signatures"]
            predicate_norm = [value / n_pred for value in predicate_curve] if n_pred else []
            signature_norm = [value / n_sig for value in signature_curve] if n_sig else []
            arm_auc = {
                "auc_predicates_normalized": trapz_auc(predicate_norm),
                "auc_predicates_raw": trapz_auc(predicate_curve),
                "auc_signatures_normalized": trapz_auc(signature_norm),
                "auc_signatures_raw": trapz_auc(signature_curve),
            }
            if arm_episodes:
                aucs[key] = arm_auc
            tick_values = [ep.ticks for ep in arm_episodes]
            collision_valid = [ep for ep in arm_episodes if ep.collision_count is not None]
            record = {
                "route": route,
                "policy": policy,
                "n_valid": len(arm_episodes),
                "n_failed": int(failed.get(key, 0)),
                "target": target_for(policy),
                "completeness": f"n={len(arm_episodes)}/{target_for(policy)}",
                "complete": len(arm_episodes) >= target_for(policy),
                "present": key in loaded["arm_paths"] or bool(arm_episodes) or key in failed,
                "n_unique_candidates": len(candidates),
                "n_repeated_radii": len(arm_episodes) - len(candidates) if policy != CONTROL_POLICY else 0,
                "mean_ticks": float(np.mean(tick_values)) if tick_values else None,
                "collision_rate": (
                    sum(1 for ep in collision_valid if ep.critical) / len(collision_valid)
                    if collision_valid
                    else None
                ),
                "goal_rate": (
                    sum(1 for ep in arm_episodes if ep.reached_goal) / len(arm_episodes)
                    if arm_episodes
                    else None
                ),
                "mean_distinct_predicates": (
                    float(np.mean([len(ep.predicates) for ep in arm_episodes])) if arm_episodes else None
                ),
                "mean_distinct_signatures": (
                    float(np.mean([len(ep.signatures) for ep in arm_episodes])) if arm_episodes else None
                ),
                **arm_auc,
                "curve_predicates_raw": predicate_curve,
                "curve_predicates_normalized": predicate_norm,
                "curve_signatures_raw": signature_curve,
                "curve_signatures_normalized": signature_norm,
                "kmnc_series": [ep.kmnc for ep in candidates],
                "lsa_max_series": [ep.lsa_max for ep in candidates],
                "lsa_mean_series": [ep.lsa_mean for ep in candidates],
            }
            arm_rows.append(record)

    # Per-policy summary: macro (route-level) means with bootstrap CIs over routes.
    policy_summary: dict[str, Any] = {}
    for policy in policies:
        route_keys = sorted(route for route in routes if episodes.get((route, policy)))
        n_valid = sum(len(episodes[(route, policy)]) for route in route_keys)
        n_failed = sum(int(failed.get((route, policy), 0)) for route in routes)
        if not route_keys:
            policy_summary[policy] = {
                "policy": policy,
                "n_valid": 0,
                "n_failed": n_failed,
                "target": target_for(policy),
                "completeness": f"n=0/{target_for(policy)}",
                "complete": False,
                "n_routes_complete": 0,
                "n_routes_present": 0,
                "n_routes_auc": 0,
                "routes_present": [],
                "routes_with_auc": [],
                "mean_ticks": None,
                "mean_ticks_ci": None,
                "collision_rate": None,
                "collision_rate_ci": None,
                "goal_rate": None,
                "goal_rate_ci": None,
                "mean_distinct_predicates": None,
                "mean_distinct_predicates_ci": None,
                "mean_distinct_signatures": None,
                "mean_distinct_signatures_ci": None,
                "auc_predicates_normalized_mean": None,
                "auc_predicates_normalized_ci": None,
                "auc_predicates_raw_mean": None,
                "auc_signatures_normalized_mean": None,
                "auc_signatures_normalized_ci": None,
                "auc_signatures_raw_mean": None,
            }
            continue

        def route_mean(route: str, values: Sequence[float]) -> float | None:
            return float(np.mean(values)) if values else None

        ticks_by_route = [route_mean(r, [ep.ticks for ep in episodes[(r, policy)]]) for r in route_keys]
        coll_by_route = [
            route_mean(
                r,
                [1.0 if ep.critical else 0.0 for ep in episodes[(r, policy)] if ep.collision_count is not None],
            )
            for r in route_keys
        ]
        goal_by_route = [
            route_mean(r, [1.0 if ep.reached_goal else 0.0 for ep in episodes[(r, policy)]]) for r in route_keys
        ]
        preds_by_route = [
            route_mean(r, [float(len(ep.predicates)) for ep in episodes[(r, policy)]]) for r in route_keys
        ]
        sigs_by_route = [
            route_mean(r, [float(len(ep.signatures)) for ep in episodes[(r, policy)]]) for r in route_keys
        ]
        auc_pred_routes = [
            (r, aucs[(r, policy)]["auc_predicates_normalized"])
            for r in route_keys
            if aucs.get((r, policy), {}).get("auc_predicates_normalized") is not None
        ]
        auc_sig_routes = [
            (r, aucs[(r, policy)]["auc_signatures_normalized"])
            for r in route_keys
            if aucs.get((r, policy), {}).get("auc_signatures_normalized") is not None
        ]
        auc_pred_values = [value for _, value in auc_pred_routes]
        auc_sig_values = [value for _, value in auc_sig_routes]
        auc_pred_raw_values = [
            aucs[(r, policy)]["auc_predicates_raw"]
            for r in route_keys
            if aucs.get((r, policy), {}).get("auc_predicates_raw") is not None
        ]
        auc_sig_raw_values = [
            aucs[(r, policy)]["auc_signatures_raw"]
            for r in route_keys
            if aucs.get((r, policy), {}).get("auc_signatures_raw") is not None
        ]

        def clean_ci(ci: tuple[float, float] | None) -> list[float] | None:
            return [ci[0], ci[1]] if ci is not None else None

        def clean_values(values: Sequence[float | None]) -> list[float]:
            return [float(value) for value in values if value is not None]

        policy_summary[policy] = {
            "policy": policy,
            "n_valid": n_valid,
            "n_failed": n_failed,
            "target": target_for(policy),
            "completeness": f"n={n_valid}/{target_for(policy) * len(routes)}",
            "complete": all(len(episodes[(route, policy)]) >= target_for(policy) for route in route_keys)
            and len(route_keys) == len(routes),
            "n_routes_complete": sum(
                1 for route in route_keys if len(episodes[(route, policy)]) >= target_for(policy)
            ),
            "n_routes_present": len(route_keys),
            "n_routes_auc": len(auc_pred_values),
            "routes_present": route_keys,
            "routes_with_auc": [r for r, _ in auc_pred_routes],
            "mean_ticks": float(np.mean(clean_values(ticks_by_route))) if clean_values(ticks_by_route) else None,
            "mean_ticks_ci": clean_ci(bootstrap_mean_ci(clean_values(ticks_by_route), rng, n_boot)),
            "collision_rate": (
                float(np.mean(clean_values(coll_by_route))) if clean_values(coll_by_route) else None
            ),
            "collision_rate_ci": clean_ci(bootstrap_mean_ci(clean_values(coll_by_route), rng, n_boot)),
            "goal_rate": float(np.mean(clean_values(goal_by_route))) if clean_values(goal_by_route) else None,
            "goal_rate_ci": clean_ci(bootstrap_mean_ci(clean_values(goal_by_route), rng, n_boot)),
            "mean_distinct_predicates": (
                float(np.mean(clean_values(preds_by_route))) if clean_values(preds_by_route) else None
            ),
            "mean_distinct_predicates_ci": clean_ci(
                bootstrap_mean_ci(clean_values(preds_by_route), rng, n_boot)
            ),
            "mean_distinct_signatures": (
                float(np.mean(clean_values(sigs_by_route))) if clean_values(sigs_by_route) else None
            ),
            "mean_distinct_signatures_ci": clean_ci(
                bootstrap_mean_ci(clean_values(sigs_by_route), rng, n_boot)
            ),
            "auc_predicates_normalized_mean": (
                float(np.mean(auc_pred_values)) if auc_pred_values else None
            ),
            "auc_predicates_normalized_ci": clean_ci(bootstrap_mean_ci(auc_pred_values, rng, n_boot)),
            "auc_predicates_raw_mean": (
                float(np.mean(clean_values(auc_pred_raw_values))) if auc_pred_raw_values else None
            ),
            "auc_signatures_normalized_mean": (
                float(np.mean(auc_sig_values)) if auc_sig_values else None
            ),
            "auc_signatures_normalized_ci": clean_ci(bootstrap_mean_ci(auc_sig_values, rng, n_boot)),
            "auc_signatures_raw_mean": (
                float(np.mean(clean_values(auc_sig_raw_values))) if auc_sig_raw_values else None
            ),
        }

    # Pairwise comparisons: primary vs each other policy, per metric family.
    pairwise: dict[str, list[dict[str, Any]]] = {
        "auc_predicates_normalized": [],
        "auc_signatures_normalized": [],
    }
    comparison_policies = [
        policy for policy in policies if policy != primary_policy and policy != CONTROL_POLICY
    ]
    for metric in pairwise:
        records: list[dict[str, Any]] = []
        for other in comparison_policies:
            included: list[str] = []
            excluded: list[dict[str, str]] = []
            for route in routes:
                primary_arm = episodes.get((route, primary_policy), [])
                other_arm = episodes.get((route, other), [])
                primary_n = len(unique_candidate_episodes(primary_arm))
                other_n = len(unique_candidate_episodes(other_arm))
                primary_auc = aucs.get((route, primary_policy), {}).get(metric)
                other_auc = aucs.get((route, other), {}).get(metric)
                if primary_auc is None or other_auc is None:
                    excluded.append({"route": route, "reason": "arm has fewer than 2 unique candidates"})
                elif primary_n < min_arm_evals or other_n < min_arm_evals:
                    excluded.append(
                        {
                            "route": route,
                            "reason": f"arm below min unique candidates ({primary_n} vs {other_n}, need {min_arm_evals})",
                        }
                    )
                else:
                    included.append(route)
            record: dict[str, Any] = {
                "metric": metric,
                "policy_a": primary_policy,
                "policy_b": other,
                "n_routes": len(included),
                "routes": included,
                "excluded_routes": excluded,
                "min_arm_evals": min_arm_evals,
                "testable": len(included) >= min_test_routes,
                "u": None,
                "p": None,
                "p_holm": None,
                "a12": None,
                "effect": None,
                "note": None,
            }
            if record["testable"]:
                values_a = [aucs[(route, primary_policy)][metric] for route in included]
                values_b = [aucs[(route, other)][metric] for route in included]
                u_stat, p_value = mann_whitney(values_a, values_b)
                record["u"] = u_stat
                record["p"] = p_value
                record["a12"] = vargha_delaney_a(u_stat, len(values_a), len(values_b))
                record["effect"] = effect_label(record["a12"])
            else:
                record["note"] = (
                    f"not testable: {len(included)} route(s) with both arms >= {min_arm_evals} "
                    f"unique candidates (need >= {min_test_routes})"
                )
            records.append(record)
        testable = [record for record in records if record["testable"]]
        if testable:
            adjusted = holm_bonferroni([record["p"] for record in testable])
            for record, value in zip(testable, adjusted):
                record["p_holm"] = value
        pairwise[metric] = records

    # Failure taxonomy.
    taxonomy_counts: dict[str, dict[str, int]] = {}
    taxonomy_meta: dict[str, Any] = {}
    for policy in policies:
        policy_episodes = [
            ep for route in routes for ep in episodes.get((route, policy), [])
        ]
        critical = [ep for ep in policy_episodes if ep.critical]
        counts = Counter()
        cooccurrence = Counter()
        for episode in critical:
            classes = {actor_class(actor) for actor in episode.collision_actors}
            if not classes:
                classes = {"no_actor_recorded"}
            if len(classes) > 1:
                cooccurrence[tuple(sorted(classes))] += 1
            for cls in classes:
                counts[cls] += 1
        taxonomy_counts[policy] = {cls: int(counts.get(cls, 0)) for cls in ACTOR_CLASSES}
        taxonomy_meta[policy] = {
            "n_valid": len(policy_episodes),
            "n_critical": len(critical),
            "n_non_critical": len(policy_episodes) - len(critical),
            "cooccurrence": {"+".join(key): int(value) for key, value in sorted(cooccurrence.items())},
            "actor_histogram": dict(
                sorted(Counter(actor for ep in critical for actor in ep.collision_actors).items())
            ),
        }

    # Top covered signatures among critical rows.
    critical_signature_top: dict[str, Any] = {}
    for policy in policies:
        counter: Counter[str] = Counter()
        for route in routes:
            for episode in episodes.get((route, policy), []):
                if episode.critical:
                    counter.update(set(episode.signatures))
        top = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:10]
        critical_signature_top[policy] = {
            "critical_rows": sum(1 for route in routes for ep in episodes.get((route, policy), []) if ep.critical),
            "top": [{"signature": name, "count": int(count)} for name, count in top],
        }

    # Fulfilled obligations on critical vs non-critical rows.
    obligations: dict[str, Any] = {}
    for policy in policies:
        buckets: dict[str, list[float]] = {"critical": [], "non_critical": []}
        for route in routes:
            for episode in episodes.get((route, policy), []):
                buckets["critical" if episode.critical else "non_critical"].append(
                    float(len(episode.fulfilled))
                )
        entry: dict[str, Any] = {}
        for bucket, values in buckets.items():
            entry[bucket] = {
                "n": len(values),
                "mean_fulfilled": float(np.mean(values)) if values else None,
                "mean_fulfilled_ci": (
                    list(bootstrap_mean_ci(values, rng, n_boot)) if values else None
                ),
            }
        obligations[policy] = entry

    integrity = {
        "malformed_lines": loaded["malformed_lines"],
        "duplicate_eval_indices": loaded["duplicate_eval_indices"],
        "n_valid_rows": sum(len(v) for v in episodes.values()),
        "n_failed_rows": sum(failed.values()),
        "failed_rows_by_arm": {
            f"{route}/{policy}": int(count)
            for (route, policy), count in sorted(failed.items())
        },
    }

    return {
        "episodes": episodes,
        "failed": failed,
        "routes": routes,
        "policies": policies,
        "arm_rows": arm_rows,
        "aucs": aucs,
        "attainable": attainable,
        "policy_summary": policy_summary,
        "pairwise": pairwise,
        "taxonomy_counts": taxonomy_counts,
        "taxonomy_meta": taxonomy_meta,
        "critical_signature_top": critical_signature_top,
        "obligations": obligations,
        "integrity": integrity,
        "expected_routes": list(expected_routes),
        "expected_policies": list(expected_policies),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _policy_curve_matrices(
    stats_result: dict[str, Any], field: str, normalized: bool
) -> dict[str, np.ndarray]:
    policies = stats_result["policies"]
    routes = stats_result["routes"]
    matrices: dict[str, np.ndarray] = {}
    for policy in policies:
        curves = []
        for route in routes:
            record = next(
                (item for item in stats_result["arm_rows"] if item["route"] == route and item["policy"] == policy),
                None,
            )
            if record is None:
                continue
            key = (
                f"curve_{field}_normalized" if normalized else f"curve_{field}_raw"
            )
            curve = record.get(key) or []
            if not curve:
                continue
            curves.append((route, curve))
        if not curves:
            continue
        length = max(len(curve) for _, curve in curves)
        matrix = np.full((len(curves), length), np.nan)
        for index, (_, curve) in enumerate(curves):
            matrix[index, : len(curve)] = curve
            matrix[index, len(curve):] = curve[-1]
        matrices[policy] = matrix
    return matrices


def _series_matrix(stats_result: dict[str, Any], key: str) -> dict[str, np.ndarray]:
    policies = stats_result["policies"]
    routes = stats_result["routes"]
    matrices: dict[str, np.ndarray] = {}
    for policy in policies:
        series = []
        for route in routes:
            record = next(
                (item for item in stats_result["arm_rows"] if item["route"] == route and item["policy"] == policy),
                None,
            )
            if record is None:
                continue
            values = record.get(key) or []
            numeric = [float(value) if value is not None and np.isfinite(value) else np.nan for value in values]
            if numeric and any(np.isfinite(value) for value in numeric):
                series.append(numeric)
        if not series:
            continue
        length = max(len(values) for values in series)
        matrix = np.full((len(series), length), np.nan)
        for index, values in enumerate(series):
            matrix[index, : len(values)] = values
        matrices[policy] = matrix
    return matrices


def _draw_curve_panel(
    ax: plt.Axes,
    matrices: dict[str, np.ndarray],
    rng: np.random.Generator,
    n_boot: int,
    title: str,
    ylabel: str,
) -> None:
    for policy, matrix in matrices.items():
        mean, lo, hi = bootstrap_band(matrix, rng, n_boot)
        x = np.arange(len(mean))
        color = POLICY_COLORS.get(policy, None)
        ax.plot(x, mean, label=f"{policy} (n_routes={matrix.shape[0]})", color=color, linewidth=1.8)
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Evaluation index (unique candidates)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8)


def plot_coverage_curves(
    stats_result: dict[str, Any],
    figures_dir: Path,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, Path]:
    plt.rcParams.update({"font.size": 9, "figure.dpi": 200})
    output: dict[str, Path] = {}

    for field, filename, label in (
        ("predicates", FIG_PREDICATES, "distinct semantic predicates"),
        ("signatures", FIG_SIGNATURES, "distinct semantic signatures"),
    ):
        raw = _policy_curve_matrices(stats_result, field, normalized=False)
        norm = _policy_curve_matrices(stats_result, field, normalized=True)
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        _draw_curve_panel(
            axes[0], raw, rng, n_boot, "Raw cumulative union", f"cumulative distinct {label}"
        )
        _draw_curve_panel(
            axes[1],
            norm,
            rng,
            n_boot,
            "Normalised by per-route attainable union",
            "fraction of attainable union",
        )
        axes[1].set_ylim(0.0, 1.02)
        fig.suptitle(f"Cumulative {label} vs evaluation index (mean and 95% CI over routes)")
        fig.tight_layout()
        path = figures_dir / filename
        fig.savefig(path)
        plt.close(fig)
        output[field] = path
    return output


def plot_auc(
    stats_result: dict[str, Any],
    figures_dir: Path,
    rng: np.random.Generator,
    n_boot: int,
) -> Path:
    policy_summary = stats_result["policy_summary"]
    policies = [policy for policy in stats_result["policies"] if policy_summary.get(policy)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, metric, title in (
        (axes[0], "auc_predicates_normalized", "Semantic predicates"),
        (axes[1], "auc_signatures_normalized", "Semantic signatures"),
    ):
        means: list[float] = []
        errors: list[list[float]] = []
        labels: list[str] = []
        colors: list[str] = []
        hatches: list[str] = []
        for policy in policies:
            summary = policy_summary[policy]
            mean = summary.get(f"{metric}_mean")
            ci = summary.get(f"{metric}_ci")
            n_routes = summary.get("n_routes_auc", 0) or 0
            if mean is None:
                continue
            means.append(mean)
            lower = mean - (ci[0] if ci else mean)
            upper = (ci[1] if ci else mean) - mean
            errors.append([lower, upper])
            labels.append(f"{policy}\n({n_routes} routes)")
            colors.append(POLICY_COLORS.get(policy, "#999999"))
            hatches.append("//" if policy == CONTROL_POLICY else "")
            if policy == CONTROL_POLICY:
                colors[-1] = "#CCCCCC"
        x = np.arange(len(means))
        bars = ax.bar(
            x,
            means,
            yerr=np.asarray(errors).T if errors else None,
            capsize=3.0,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("normalised coverage AUC")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.suptitle("Coverage AUC per policy (mean and bootstrap 95% CI over routes; control is baseline)")
    fig.tight_layout()
    path = figures_dir / FIG_AUC
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_trajectories(
    stats_result: dict[str, Any],
    figures_dir: Path,
    rng: np.random.Generator,
    n_boot: int,
) -> Path:
    kmnc = _series_matrix(stats_result, "kmnc_series")
    lsa = _series_matrix(stats_result, "lsa_max_series")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    _draw_curve_panel(axes[0], kmnc, rng, n_boot, "KMNC per evaluation", "coverage_kmnc")
    _draw_curve_panel(axes[1], lsa, rng, n_boot, "LSA (max) per evaluation", "coverage_lsa_max")
    axes[0].set_ylim(bottom=0.0)
    axes[1].set_ylim(bottom=0.0)
    fig.suptitle("Per-evaluation coverage trajectories (mean and 95% CI over routes)")
    fig.tight_layout()
    path = figures_dir / FIG_TRAJECTORIES
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_failure_taxonomy(stats_result: dict[str, Any], figures_dir: Path) -> Path:
    policies = stats_result["policies"]
    counts = stats_result["taxonomy_counts"]
    classes = list(ACTOR_CLASSES)
    x = np.arange(len(classes))
    width = 0.8 / max(len(policies), 1)
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for index, policy in enumerate(policies):
        values = [counts.get(policy, {}).get(cls, 0) for cls in classes]
        offset = (index - (len(policies) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=f"{policy} (critical rows={stats_result['taxonomy_meta'][policy]['n_critical']})",
            color=POLICY_COLORS.get(policy, "#999999"),
            edgecolor="black",
            linewidth=0.4,
        )
        for bar, value in zip(bars, values):
            if value:
                ax.annotate(
                    str(value),
                    (bar.get_x() + bar.get_width() / 2.0, value),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("critical rows (collision_count > 0)")
    ax.set_title("Failure taxonomy by collision actor class (a row can appear in several classes)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir / FIG_FAILURES
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    if digits == 0:
        return f"{number:.0f}"
    return f"{number:.{digits}f}"


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100.0:.1f}%"


def _fmt_ci(ci: Any, digits: int = 3, transform: Any = None) -> str:
    if not ci or len(ci) != 2 or ci[0] is None:
        return "—"
    low, high = (transform(ci[0]), transform(ci[1])) if transform else (ci[0], ci[1])
    return f"{float(low):.{digits}f}–{float(high):.{digits}f}"


def _arm_table_markdown(stats_result: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    header = (
        "| Route | Policy | n | Status | Fails | Mean ticks | Collision rate | Goal rate | "
        "Predicates/ep | Signatures/ep | AUC pred (norm) | AUC sig (norm) |"
    )
    separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines.append(header)
    lines.append(separator)
    for record in stats_result["arm_rows"]:
        if not record["present"]:
            status = "missing"
        elif record["complete"]:
            status = "complete"
        else:
            status = "partial"
        lines.append(
            "| {route} | {policy} | {n}/{target} | {status} | {fails} | {ticks} | {coll} | {goal} | "
            "{preds} | {sigs} | {aucp} | {aucs} |".format(
                route=record["route"],
                policy=record["policy"],
                n=record["n_valid"],
                target=record["target"],
                status=status,
                fails=record["n_failed"],
                ticks=_fmt(record["mean_ticks"], 1),
                coll=_fmt_rate(record["collision_rate"]),
                goal=_fmt_rate(record["goal_rate"]),
                preds=_fmt(record["mean_distinct_predicates"], 2),
                sigs=_fmt(record["mean_distinct_signatures"], 2),
                aucp=_fmt(record["auc_predicates_normalized"], 3),
                aucs=_fmt(record["auc_signatures_normalized"], 3),
            )
        )
    return lines


def _policy_summary_markdown(stats_result: dict[str, Any]) -> list[str]:
    lines = [
        "| Policy | n valid (all routes) | Routes complete | Mean ticks | Collision rate | Goal rate | "
        "Mean preds/ep | Mean sigs/ep | AUC pred (norm, mean ± CI) | AUC sig (norm, mean ± CI) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for policy in stats_result["policies"]:
        summary = stats_result["policy_summary"].get(policy)
        if not summary:
            continue
        aucp = summary["auc_predicates_normalized_mean"]
        aucs = summary["auc_signatures_normalized_mean"]
        aucp_text = (
            f"{_fmt(aucp, 3)} ± [{_fmt_ci(summary['auc_predicates_normalized_ci'])}]"
            if aucp is not None
            else "—"
        )
        aucs_text = (
            f"{_fmt(aucs, 3)} ± [{_fmt_ci(summary['auc_signatures_normalized_ci'])}]"
            if aucs is not None
            else "—"
        )
        lines.append(
            "| {policy} | {n} | {complete} | {ticks} | {coll} | {goal} | {preds} | {sigs} | {aucp} | {aucs} |".format(
                policy=policy,
                n=summary["n_valid"],
                complete=f"{summary['n_routes_complete']}/{len(stats_result['routes'])}",
                ticks=f"{_fmt(summary['mean_ticks'], 1)} ± [{_fmt_ci(summary['mean_ticks_ci'], 1)}]",
                coll=f"{_fmt_rate(summary['collision_rate'])}",
                goal=_fmt_rate(summary["goal_rate"]),
                preds=_fmt(summary["mean_distinct_predicates"], 2),
                sigs=_fmt(summary["mean_distinct_signatures"], 2),
                aucp=aucp_text,
                aucs=aucs_text,
            )
        )
    return lines


def _pairwise_markdown(stats_result: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for metric, title in (
        ("auc_predicates_normalized", "Normalised predicate AUC"),
        ("auc_signatures_normalized", "Normalised signature AUC"),
    ):
        lines.append(f"### {title}")
        lines.append("")
        lines.append(
            "| Comparison | n routes | Routes | U | p (raw) | p (Holm) | Vargha-Delaney A | Effect |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for record in stats_result["pairwise"][metric]:
            if record["testable"]:
                lines.append(
                    "| {a} vs {b} | {n} | {routes} | {u} | {p} | {ph} | {a12} | {effect} |".format(
                        a=record["policy_a"],
                        b=record["policy_b"],
                        n=record["n_routes"],
                        routes=", ".join(record["routes"]),
                        u=_fmt(record["u"], 1),
                        p=_fmt(record["p"], 4),
                        ph=_fmt(record["p_holm"], 4),
                        a12=_fmt(record["a12"], 3),
                        effect=record["effect"],
                    )
                )
            else:
                lines.append(
                    "| {a} vs {b} | {n} | {routes} | — | — | — | — | {note} |".format(
                        a=record["policy_a"],
                        b=record["policy_b"],
                        n=record["n_routes"],
                        routes=", ".join(record["routes"]) or "none",
                        note=record["note"] or "",
                    )
                )
        lines.append("")
        excluded_all = {
            (item["route"], item["reason"])
            for record in stats_result["pairwise"][metric]
            for item in record["excluded_routes"]
        }
        if excluded_all:
            lines.append("Excluded route/arms: " + "; ".join(
                f"`{route}` ({reason})" for route, reason in sorted(excluded_all)
            ))
            lines.append("")
    return lines


def _taxonomy_markdown(stats_result: dict[str, Any]) -> list[str]:
    policies = stats_result["policies"]
    counts = stats_result["taxonomy_counts"]
    lines = [
        "| Policy | " + " | ".join(ACTOR_CLASSES) + " | Critical rows | Non-critical rows |",
        "| --- | " + " | ".join(["---"] * len(ACTOR_CLASSES)) + " | --- | --- |",
    ]
    for policy in policies:
        meta = stats_result["taxonomy_meta"][policy]
        lines.append(
            "| {policy} | {cells} | {crit} | {non} |".format(
                policy=policy,
                cells=" | ".join(str(counts.get(policy, {}).get(cls, 0)) for cls in ACTOR_CLASSES),
                crit=meta["n_critical"],
                non=meta["n_non_critical"],
            )
        )
    lines.append("")
    lines.append("Actor histograms on critical rows:")
    lines.append("")
    for policy in policies:
        meta = stats_result["taxonomy_meta"][policy]
        histogram = ", ".join(f"{actor} ({count})" for actor, count in meta["actor_histogram"].items())
        cooc = ", ".join(f"{key} ({value})" for key, value in meta["cooccurrence"].items())
        lines.append(
            f"- **{policy}** — actors: {histogram or 'none'}; mixed-class rows: {cooc or 'none'}"
        )
    return lines


def _signature_markdown(stats_result: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for policy in stats_result["policies"]:
        entry = stats_result["critical_signature_top"].get(policy)
        if not entry:
            continue
        lines.append(f"**{policy}** ({entry['critical_rows']} critical rows)")
        lines.append("")
        if entry["top"]:
            lines.append("| Rank | Signature | Count |")
            lines.append("| --- | --- | --- |")
            for rank, item in enumerate(entry["top"], start=1):
                lines.append(f"| {rank} | `{item['signature']}` | {item['count']} |")
        else:
            lines.append("_no critical rows with covered signatures_")
        lines.append("")
    return lines


def _obligation_markdown(stats_result: dict[str, Any]) -> list[str]:
    lines = [
        "| Policy | n critical | Mean fulfilled (critical) | n non-critical | Mean fulfilled (non-critical) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for policy in stats_result["policies"]:
        entry = stats_result["obligations"].get(policy)
        if not entry:
            continue
        lines.append(
            "| {policy} | {nc} | {mc} | {nn} | {mn} |".format(
                policy=policy,
                nc=entry["critical"]["n"],
                mc=_fmt(entry["critical"]["mean_fulfilled"], 3),
                nn=entry["non_critical"]["n"],
                mn=_fmt(entry["non_critical"]["mean_fulfilled"], 3),
            )
        )
    return lines


def build_markdown(
    stats_result: dict[str, Any], figures: dict[str, Path], root: Path, out_dir: Path
) -> str:
    policies = stats_result["policies"]
    routes = stats_result["routes"]
    expected_routes = stats_result["expected_routes"]
    target_evals = stats_result["config"]["target_evals"]
    target_controls = stats_result["config"]["target_controls"]
    primary = stats_result["config"]["primary_policy"]
    lines: list[str] = []
    lines.append("# EXP-020 policy search: aggregation summary")
    lines.append("")
    lines.append(
        "Generated by `research/experiments/EXP-020-policy-comparison/proof-of-concept/aggregate_policy_search.py` "
        "from `{}`. Re-running while the search appends rows is safe: the summary reflects the input snapshot "
        "at read time.".format(root)
    )
    lines.append("")

    integrity = stats_result["integrity"]
    lines.append("## Data state")
    lines.append("")
    lines.append(f"- Routes discovered: {len(routes)} (expected {len(expected_routes)}): {', '.join(f'`{r}`' for r in routes)}")
    missing_routes = [route for route in expected_routes if route not in routes]
    if missing_routes:
        lines.append(f"- Expected routes with no data directory: {', '.join(f'`{r}`' for r in missing_routes)}")
    lines.append(f"- Policies discovered: {', '.join(f'`{p}`' for p in policies)}")
    lines.append(
        f"- Valid rows (ticks_executed not null): {integrity['n_valid_rows']}; "
        f"failed rows: {integrity['n_failed_rows']}; "
        f"malformed JSON lines skipped: {integrity['malformed_lines']}; "
        f"duplicate eval_index occurrences: {integrity['duplicate_eval_indices']} "
        "(rows kept; distinct radii are distinct candidates and are only de-duplicated in curves)"
    )
    if integrity["failed_rows_by_arm"]:
        lines.append(
            "- Failed rows by arm: "
            + ", ".join(f"`{arm}` {count}" for arm, count in integrity["failed_rows_by_arm"].items())
        )
    lines.append("")

    lines.append("## Per-policy summary")
    lines.append("")
    lines.append(
        "Row-level statistics are macro means over routes (each route weighted equally) with bootstrap 95% CIs "
        "over routes; AUC is the mean normalised coverage over the search (trapezoidal integral of the "
        "normalised cumulative-union curve divided by `m - 1`)."
    )
    lines.append("")
    lines.extend(_policy_summary_markdown(stats_result))
    lines.append("")

    lines.append("## Per-arm detail")
    lines.append("")
    lines.append(
        f"Search arms target {target_evals} evaluations per (route, policy); controls target {target_controls}. "
        "`partial`/`missing` arms are search-in-progress states, not failures."
    )
    lines.append("")
    lines.extend(_arm_table_markdown(stats_result))
    lines.append("")

    control_summary = stats_result["policy_summary"].get(CONTROL_POLICY) if CONTROL_POLICY in policies else None
    lines.append("## Controls (no-adversary baselines)")
    lines.append("")
    if control_summary is None:
        lines.append("_No control rows available yet._")
    else:
        lines.append(
            "- Valid rows: {n} across {nr} route(s); failed rows: {nf}.".format(
                n=control_summary["n_valid"],
                nr=control_summary["n_routes_present"],
                nf=control_summary["n_failed"],
            )
        )
        lines.append(
            "- Mean ticks: {ticks}; collision rate: {coll}; goal rate: {goal}; mean distinct predicates/episode: "
            "{preds}; mean distinct signatures/episode: {sigs}.".format(
                ticks=_fmt(control_summary["mean_ticks"], 1),
                coll=_fmt_rate(control_summary["collision_rate"]),
                goal=_fmt_rate(control_summary["goal_rate"]),
                preds=_fmt(control_summary["mean_distinct_predicates"], 2),
                sigs=_fmt(control_summary["mean_distinct_signatures"], 2),
            )
        )
        control_predicates = sorted(
            {
                predicate
                for route in routes
                for (arm_route, policy), arm_episodes in stats_result["episodes"].items()
                if policy == CONTROL_POLICY and arm_route == route
                for episode in arm_episodes
                for predicate in episode.predicates
            }
        )
        lines.append(
            "- Predicates observed under control: "
            + (", ".join(f"`{p}`" for p in control_predicates) if control_predicates else "none")
        )
    lines.append("")

    lines.append("## Pairwise comparisons ({primary} vs others, per-route normalised AUC)".format(primary=primary))
    lines.append("")
    lines.append(
        f"Mann-Whitney U (asymptotic, two-sided) on per-route normalised AUCs; Holm-Bonferroni correction "
        f"within each metric family; Vargha-Delaney A12 (`U / (n_a * n_b)`, > 0.5 favours {primary}). "
        f"A route enters a comparison only when both arms have >= {stats_result['config']['min_arm_evals']} "
        f"unique candidates; comparisons with fewer than {stats_result['config']['min_test_routes']} routes "
        f"are reported without a p-value."
    )
    lines.append("")
    lines.extend(_pairwise_markdown(stats_result))

    lines.append("## Failure taxonomy")
    lines.append("")
    lines.append(
        "Critical rows are valid rows with `collision_count > 0`. A row with actors from several classes is "
        "counted in each class (mixed-class rows are listed below). `other` covers any non-`static`/`walker`/"
        "`vehicle` actor (e.g. `traffic.*`); `no_actor_recorded` means a critical row had an empty actor list."
    )
    lines.append("")
    lines.extend(_taxonomy_markdown(stats_result))
    lines.append("")

    lines.append("### Top semantic signatures among critical rows")
    lines.append("")
    lines.extend(_signature_markdown(stats_result))

    lines.append("## Fulfilled obligations: critical vs non-critical rows")
    lines.append("")
    lines.extend(_obligation_markdown(stats_result))
    lines.append("")

    lines.append("## Per-route attainable unions (normalisation denominators)")
    lines.append("")
    lines.append(
        "The attainable union is computed over the non-control policies (`random`, `lsa`, `kmnc`, `semantic`); "
        "the control arm is a no-adversary baseline and is excluded from the denominator. The last two columns "
        "show what the denominators would be if control were included."
    )
    lines.append("")
    lines.append("| Route | Predicates (search) | Signatures (search) | Predicates (+control) | Signatures (+control) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for route in routes:
        entry = stats_result["attainable"][route]
        lines.append(
            "| {route} | {p} | {s} | {pc} | {sc} |".format(
                route=route,
                p=entry["n_predicates"],
                s=entry["n_signatures"],
                pc=entry["n_predicates_with_control"],
                sc=entry["n_signatures_with_control"],
            )
        )
    lines.append("")

    if figures:
        lines.append("## Figures")
        lines.append("")
        for field, path in figures.items():
            try:
                label = str(path.relative_to(REPO_ROOT))
            except ValueError:
                label = str(path)
            embed = os.path.relpath(path, start=out_dir)
            lines.append(f"- `{label}` ({field})")
            lines.append("")
            lines.append(f"![{field}]({embed})")
            lines.append("")

    lines.append("## Method notes and gaps")
    lines.append("")
    lines.append(
        "- Coverage curves use only successful rows (`ticks_executed` not null), ordered by `eval_index`, with "
        "repeated radii dropped (first occurrence kept). Failed rows are excluded from metrics but counted per arm."
    )
    lines.append(
        "- AUC is trapezoidal over the candidate index divided by `m - 1` (m = unique candidates); with "
        "cumulative-union curves this equals the mean normalised coverage over the search."
    )
    lines.append(
        "- Aggregate curves carry the last observed value forward for arms shorter than the widest arm; "
        "the legend reports the number of contributing routes."
    )
    lines.append(
        "- Bootstrap: {n} resamples over routes (seeded, deterministic) for CIs and bands; percentile method."
        .format(n=stats_result["config"]["bootstrap"])
    )
    lines.append(
        "- Pairwise tests use per-route AUCs as the unit of analysis; n is the number of routes, not rows."
    )
    incomplete = [
        record
        for record in stats_result["arm_rows"]
        if record["present"] and not record["complete"]
    ]
    missing = [record for record in stats_result["arm_rows"] if not record["present"]]
    lines.append("")
    lines.append("### Data gaps at read time")
    lines.append("")
    if incomplete:
        for record in sorted(incomplete, key=lambda item: (item["route"], item["policy"])):
            lines.append(
                f"- `{record['route']}/{record['policy']}` is incomplete: {record['completeness']} "
                f"({record['n_failed']} failed rows)."
            )
    else:
        lines.append("- No partial arms: every present arm has reached its target evaluation count.")
    if missing:
        for record in sorted(missing, key=lambda item: (item["route"], item["policy"])):
            lines.append(f"- `{record['route']}/{record['policy']}` has no rows yet.")
    lines.append("")
    lines.append(
        "Regenerate with: `research/.venv/bin/python "
        "research/experiments/EXP-020-policy-comparison/proof-of-concept/aggregate_policy_search.py`"
    )
    lines.append("")
    return "\n".join(lines)


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate the EXP-020 matched-budget policy search into tables and figures."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "research" / "logs" / "policy_search",
        help="Root directory containing <route>/<policy>/rows.jsonl.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts",
        help="Directory for policy_search_summary.json and .md.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory for figures (default: <root>/figures).",
    )
    parser.add_argument("--target-evals", type=int, default=30, help="Evaluation budget for search policies.")
    parser.add_argument("--target-controls", type=int, default=8, help="Evaluation budget for controls.")
    parser.add_argument(
        "--expected-routes",
        nargs="*",
        default=list(EXPECTED_ROUTES_DEFAULT),
        help="Routes expected by the experiment matrix (for gap reporting).",
    )
    parser.add_argument(
        "--policies",
        nargs="*",
        default=list(POLICY_ORDER),
        help="Policies to look for, in reporting order.",
    )
    parser.add_argument("--primary-policy", default="semantic", help="Policy compared against the others.")
    parser.add_argument(
        "--comparison-policies",
        nargs="*",
        default=list(SEARCH_POLICIES_DEFAULT),
        help="Candidate policies for the pairwise family (control is never tested).",
    )
    parser.add_argument("--seed", type=int, default=20260915, help="Bootstrap RNG seed.")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap resamples.")
    parser.add_argument(
        "--min-arm-evals",
        type=int,
        default=5,
        help="Minimum unique candidates in both arms for a route to enter a pairwise test.",
    )
    parser.add_argument(
        "--min-test-routes",
        type=int,
        default=4,
        help="Minimum routes required to report a pairwise p-value.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (REPO_ROOT / args.out_dir).resolve()
    if args.figures_dir is None:
        figures_dir = root / "figures"
    elif args.figures_dir.is_absolute():
        figures_dir = args.figures_dir
    else:
        figures_dir = (REPO_ROOT / args.figures_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    stats_result = compute_statistics(
        root=root,
        expected_routes=args.expected_routes,
        expected_policies=args.policies,
        target_evals=args.target_evals,
        target_controls=args.target_controls,
        primary_policy=args.primary_policy,
        rng=rng,
        n_boot=args.bootstrap,
        min_arm_evals=args.min_arm_evals,
        min_test_routes=args.min_test_routes,
    )

    figure_paths: dict[str, Path] = {}
    figure_paths.update(plot_coverage_curves(stats_result, figures_dir, rng, args.bootstrap))
    figure_paths["auc"] = plot_auc(stats_result, figures_dir, rng, args.bootstrap)
    figure_paths["trajectories"] = plot_trajectories(stats_result, figures_dir, rng, args.bootstrap)
    figure_paths["failures"] = plot_failure_taxonomy(stats_result, figures_dir)

    stats_result["config"] = {
        "root": str(root),
        "expected_routes": list(args.expected_routes),
        "policies": list(args.policies),
        "target_evals": args.target_evals,
        "target_controls": args.target_controls,
        "primary_policy": args.primary_policy,
        "comparison_policies": list(args.comparison_policies),
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "min_arm_evals": args.min_arm_evals,
        "min_test_routes": args.min_test_routes,
    }
    stats_result["figures"] = {key: str(path) for key, path in figure_paths.items()}

    json_payload = _json_safe(
        {
            key: value
            for key, value in stats_result.items()
            if key not in {"episodes", "failed", "aucs"}
        }
    )
    json_path = out_dir / "policy_search_summary.json"
    json_path.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")

    markdown = build_markdown(stats_result, figure_paths, root, out_dir)
    md_path = out_dir / "policy_search_summary.md"
    md_path.write_text(markdown + "\n", encoding="utf-8")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    for key, path in figure_paths.items():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
