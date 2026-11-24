"""Utility analysis for safety-oriented coverage search.

Three questions, all answered from recorded campaign logs:

1. **How many safety issues were found?**  Handled by
   :mod:`research.harness.legacy_outcomes` (named outcomes beyond collision).
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

from research.harness.legacy_outcomes import OUTCOME_NAMES, OutcomeThresholds, classify_run
from research.harness.outcome_models import CATEGORIES, AssessmentStatus, unknown_categories
from research.harness.safety_outcomes import classify_run as classify_run_v2


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
    attempt_id: str | None = None
    run_id: str | None = None
    execution_status: str | None = None
    termination_reason: str | None = None
    schema_version: int = 1
    source_status: str = "ok"
    candidate_id: str | None = None
    profile_digest: str | None = None
    replicate_provenance: str = "inferred"
    semantic_observed: bool = True


def _v2_unknown_outcome(run_id: str, execution_status: str, reason: str) -> dict[str, Any]:
    cats = unknown_categories(reason)
    return {
        "schema_version": 2,
        "run_id": run_id,
        "execution_status": execution_status,
        "termination_reason": reason,
        "profile_digest": None,
        "categories": {
            g: {k: {"status": v.status.value, "event_ids": list(v.event_ids), "reasons": list(v.reasons)}
                for k, v in metrics.items()}
            for g, metrics in cats.items()
        },
        "aggregates": {
            "safety_event": {"status": "unknown", "event_ids": [], "reasons": [reason]},
            "traffic_rule_violation": {"status": "unknown", "event_ids": [], "reasons": [reason]},
        },
        "events": [],
        "provenance": {"source_status": reason},
    }


_VALID_V2_STATUSES = frozenset({"present", "absent", "unknown", "not_applicable"})


def _normalize_v2_status(value: Any) -> str:
    return value if isinstance(value, str) and value in _VALID_V2_STATUSES else "unknown"


def _v2_present_labels(outcome: dict[str, Any]) -> tuple[str, ...]:
    if not isinstance(outcome, dict):
        return ()
    cats = outcome.get("categories")
    if not isinstance(cats, dict):
        return ()
    labels: list[str] = []
    for grp, metrics in cats.items():
        if not isinstance(metrics, dict):
            continue
        for mname, assess in metrics.items():
            if isinstance(assess, dict) and assess.get("status") == "present":
                labels.append(f"{grp}.{mname}")
    return tuple(sorted(labels))


def _v2_assessment_status(outcome: dict[str, Any], metric: str) -> str:
    """Look up status ('present'|'absent'|'unknown'|'not_applicable') for a v2 metric or aggregate."""
    if not isinstance(outcome, dict):
        return "unknown"
    if metric in ("safety_event", "traffic_rule_violation"):
        aggs = outcome.get("aggregates")
        agg = aggs.get(metric) if isinstance(aggs, dict) else None
        return _normalize_v2_status(agg.get("status") if isinstance(agg, dict) else None)
    cats = outcome.get("categories")
    if not isinstance(cats, dict):
        return "unknown"
    if "." in metric:
        group, name = metric.split(".", 1)
        grp_map = cats.get(group)
        val = grp_map.get(name) if isinstance(grp_map, dict) else None
        return _normalize_v2_status(val.get("status") if isinstance(val, dict) else None)
    for metrics in cats.values():
        if isinstance(metrics, dict) and metric in metrics:
            entry = metrics[metric]
            return _normalize_v2_status(entry.get("status") if isinstance(entry, dict) else None)
    return "unknown"


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


def resolve_run_path(
    raw: Any,
    workspace_root: Path = WORKSPACE_ROOT,
    *,
    manifest_dir: Path | None = None,
    archive_root: Path | None = None,
) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        if path.exists():
            return path
        if archive_root is not None:
            candidate = archive_root / path.name
            if candidate.exists():
                return candidate
        return None
    candidates = []
    if manifest_dir is not None:
        candidates.append(manifest_dir / path)
    if archive_root is not None:
        candidates.append(archive_root / path)
    candidates.append(workspace_root / path)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def infer_replicate_with_provenance(row: dict[str, Any], row_file: Path, root: Path) -> tuple[str, str]:
    for key in ("replicate_id", "replicate", "campaign_seed", "search_seed"):
        value = row.get(key)
        if value is not None:
            text = str(value)
            return (text if text.startswith("seed-") else f"seed-{text}"), f"stored:{key}"
    for part in (*row_file.parts, *root.parts):
        match = _SEED_COMPONENT.match(str(part))
        if match:
            return f"seed-{match.group(1)}", "inferred_from_path"
    return "all", "unmatched_pooled"


def infer_replicate(row: dict[str, Any], row_file: Path, root: Path) -> str:
    rep, _ = infer_replicate_with_provenance(row, row_file, root)
    return rep


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
    schema_version: int = 1,
    retain_missing: bool | None = None,
    archive_root: Path | None = None,
) -> tuple[list[EvalRecord], dict[str, int]]:
    """Read every row under ``roots`` and classify its run log.

    Returns ``(records, counters)``. When ``schema_version == 2`` or
    ``retain_missing=True``, every attempted evaluation (including missing,
    unreadable, or failed runs) is retained in ``records`` with explicit
    ``unknown`` assessments, and duplicate attempt/run IDs from copied files
    are detected and counted without dropping distinct repeat attempts.
    """
    if schema_version not in (1, 2):
        raise ValueError(f"Unsupported schema_version: {schema_version}")
    keep_missing = (schema_version == 2) if retain_missing is None else bool(retain_missing)
    root_list = [roots] if isinstance(roots, Path) else list(roots)
    counters = {
        "n_rows": 0,
        "n_missing_run_path": 0,
        "n_unreadable_run": 0,
        "n_malformed_rows": 0,
        "n_duplicate_rows": 0,
        "n_failed_attempts": 0,
    }
    records: list[EvalRecord] = []
    seen_files: set[Path] = set()
    seen_attempt_ids: set[tuple[str, str, str, str]] = set()

    for root in root_list:
        for route_dir, policy_dir, path in discover_row_files(root):
            resolved = path.resolve()
            if resolved in seen_files:  # roots may overlap (parent + seed-* children)
                continue
            seen_files.add(resolved)
            try:
                text = path.read_bytes().decode("utf-8", errors="replace")
            except OSError:
                continue
            for line_no, raw_line in enumerate(text.split("\n"), 1):
                line = raw_line.rstrip("\r")
                if not line.strip(" \t\r"):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    counters["n_malformed_rows"] += 1
                    continue
                if not isinstance(row, dict):
                    counters["n_malformed_rows"] += 1
                    continue
                policy = str(row.get("policy") or policy_dir)
                route = str(row.get("route") or route_dir or policy_dir)
                replicate, rep_prov = infer_replicate_with_provenance(row, path, root)

                # Deduplicate copied rows carrying the exact same attempt_id or run_id within one arm
                dedup_key = row.get("attempt_id") or row.get("run_id")
                if isinstance(dedup_key, str) and dedup_key:
                    scoped_key = (policy, route, replicate, dedup_key)
                    if scoped_key in seen_attempt_ids:
                        counters["n_duplicate_rows"] += 1
                        continue
                    seen_attempt_ids.add(scoped_key)

                counters["n_rows"] += 1
                base = _make_record(
                    row,
                    policy,
                    route,
                    replicate,
                    workspace_root,
                    thresholds,
                    counters,
                    schema_version=schema_version,
                    retain_missing=keep_missing,
                    manifest_dir=path.parent,
                    archive_root=archive_root,
                    replicate_provenance=rep_prov,
                )
                if base is not None:
                    records.append(base)
                if include_paired_controls:
                    control = row.get("paired_control")
                    if isinstance(control, dict):
                        counts_before = dict(counters)
                        control_record = _make_record(
                            control,
                            f"{policy}:control",
                            route,
                            replicate,
                            workspace_root,
                            thresholds,
                            counters,
                            schema_version=schema_version,
                            retain_missing=keep_missing,
                            manifest_dir=path.parent,
                            archive_root=archive_root,
                            replicate_provenance=rep_prov,
                        )
                        if control_record is not None:
                            records.append(control_record)
                        else:
                            counters.update(counts_before)  # don't double count the parent row

    return records, counters


def _has_semantic_observation(row: dict[str, Any], obligations: tuple[str, ...], *, failed: bool = False) -> bool:
    if obligations:
        return True
    if failed or row.get("run_error") or row.get("engine_error"):
        return False
    for key in ("engine_run_obligations", "semantic_fulfilled_obligations", "semantic_covered_signatures", "semantic_covered_predicates"):
        if isinstance(row.get(key), list):
            return True
    return False


def _make_record(
    row: dict[str, Any],
    policy: str,
    route: str,
    replicate: str,
    workspace_root: Path,
    thresholds: OutcomeThresholds | None,
    counters: dict[str, int],
    *,
    schema_version: int = 1,
    retain_missing: bool = False,
    manifest_dir: Path | None = None,
    archive_root: Path | None = None,
    replicate_provenance: str = "inferred",
) -> EvalRecord | None:
    attempt_id = _str_or_none(row.get("attempt_id"))
    run_id = _str_or_none(row.get("run_id"))
    candidate_id = _str_or_none(row.get("candidate_id"))
    eval_index = _int_or_none(row.get("eval_index"))
    obligations = _extract_obligations(row)
    target = _str_or_none(row.get("hazard_target"))

    # Schema-2 fast path when the row already embeds normalized outcome_v2
    if schema_version == 2 and isinstance(row.get("outcome_v2"), dict):
        outcome_v2 = row["outcome_v2"]
        exec_status = str(outcome_v2.get("execution_status") or row.get("execution_status") or "unknown")
        is_failed = exec_status == "failed" or bool(row.get("run_error"))
        if is_failed:
            counters["n_failed_attempts"] += 1
        return EvalRecord(
            policy=policy,
            route=route,
            replicate=replicate,
            eval_index=eval_index,
            outcome=outcome_v2,
            obligations=obligations,
            target=target,
            run_json_path=_str_or_none(row.get("run_json_path")),
            attempt_id=attempt_id,
            run_id=run_id or _str_or_none(outcome_v2.get("run_id")),
            execution_status=exec_status,
            termination_reason=_str_or_none(outcome_v2.get("termination_reason") or row.get("termination_reason")),
            schema_version=2,
            source_status="embedded_v2",
            candidate_id=candidate_id,
            profile_digest=_str_or_none(outcome_v2.get("profile_digest") or row.get("outcome_profile_digest")),
            replicate_provenance=replicate_provenance,
            semantic_observed=_has_semantic_observation(row, obligations, failed=is_failed),
        )

    run_path = resolve_run_path(
        row.get("run_json_path"),
        workspace_root,
        manifest_dir=manifest_dir,
        archive_root=archive_root,
    )
    if run_path is None:
        counters["n_missing_run_path"] += 1
        if row.get("run_error"):
            counters["n_failed_attempts"] += 1
        if not retain_missing:
            return None
        status = "failed" if row.get("run_error") else "unknown"
        reason = "failed_attempt_without_run_log" if row.get("run_error") else "missing_run_path"
        outcome = _v2_unknown_outcome(run_id or attempt_id or "missing", status, reason) if schema_version == 2 else {"unsafe": False, "reasons": [], "missing": True}
        return EvalRecord(
            policy=policy,
            route=route,
            replicate=replicate,
            eval_index=eval_index,
            outcome=outcome,
            obligations=obligations,
            target=target,
            run_json_path=_str_or_none(row.get("run_json_path")),
            attempt_id=attempt_id,
            run_id=run_id,
            execution_status=status,
            termination_reason=reason,
            schema_version=schema_version,
            source_status="failed_attempt" if row.get("run_error") else "missing_run_path",
            candidate_id=candidate_id,
            replicate_provenance=replicate_provenance,
            semantic_observed=bool(obligations),
        )

    payload = load_json_file(run_path)
    if not isinstance(payload, dict):
        counters["n_unreadable_run"] += 1
        if not retain_missing:
            return None
        outcome = _v2_unknown_outcome(run_id or attempt_id or "unreadable", "unknown", "unreadable_run_json") if schema_version == 2 else {"unsafe": False, "reasons": [], "unreadable": True}
        return EvalRecord(
            policy=policy,
            route=route,
            replicate=replicate,
            eval_index=eval_index,
            outcome=outcome,
            obligations=obligations,
            target=target,
            run_json_path=str(run_path),
            attempt_id=attempt_id,
            run_id=run_id,
            execution_status="unknown",
            termination_reason="unreadable_run_json",
            schema_version=schema_version,
            source_status="unreadable_run",
            candidate_id=candidate_id,
            replicate_provenance=replicate_provenance,
            semantic_observed=bool(obligations),
        )

    if schema_version == 2:
        embedded = (payload.get("metadata") or {}).get("outcome_v2") if isinstance(payload.get("metadata"), dict) else None
        outcome = embedded if isinstance(embedded, dict) and embedded.get("schema_version") == 2 else classify_run_v2(payload)
        exec_status = str(outcome.get("execution_status") or payload.get("execution_status") or "unknown")
        is_failed = exec_status == "failed" or bool(row.get("run_error"))
        if exec_status == "failed":
            counters["n_failed_attempts"] += 1
        return EvalRecord(
            policy=policy,
            route=route,
            replicate=replicate,
            eval_index=eval_index,
            outcome=outcome,
            obligations=obligations,
            target=target,
            run_json_path=str(run_path),
            attempt_id=attempt_id,
            run_id=run_id or _str_or_none(outcome.get("run_id") or payload.get("run_id")),
            execution_status=exec_status,
            termination_reason=_str_or_none(outcome.get("termination_reason") or payload.get("termination_reason")),
            schema_version=2,
            source_status="run_v2" if isinstance(embedded, dict) else "adapted_legacy",
            candidate_id=candidate_id,
            profile_digest=_str_or_none(outcome.get("profile_digest")),
            replicate_provenance=replicate_provenance,
            semantic_observed=_has_semantic_observation(row, obligations, failed=is_failed),
        )

    return EvalRecord(
        policy=policy,
        route=route,
        replicate=replicate,
        eval_index=eval_index,
        outcome=classify_run(payload, thresholds),
        obligations=obligations,
        target=target,
        run_json_path=str(run_path),
        attempt_id=attempt_id,
        run_id=run_id,
        schema_version=1,
        source_status="ok",
        candidate_id=candidate_id,
        replicate_provenance=replicate_provenance,
        semantic_observed=_has_semantic_observation(row, obligations, failed=bool(row.get("run_error"))),
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
    n_attempted: int = 0
    n_assessable: int = 0
    n_unknown_prior: int = 0
    censoring_reason: str = "unknown"
    metric: str = "unsafe"
    exclude_eval_zero: bool = False
    replicate_provenance: str = "inferred"
    first_eligible_eval: int = 0
    top1_eligible: bool | None = None


def time_to_first(
    records: Iterable[EvalRecord],
    *,
    budget: int | None = None,
    metric: str | None = None,
    exclude_eval_zero: bool | None = None,
) -> list[ArmFirstIssue]:
    """Reduce each (policy, replicate, route) arm to its first discovery.

    In legacy mode (``schema_version == 1`` and ``metric is None``), preserves
    the historical index-based calculation while respecting ``budget`` cutoffs.
    In schema-2 mode (or when ``metric`` is explicitly named), retains attempts
    with missing ``eval_index``, enforces strict attempt-count budget cutoffs
    (0-indexed ``0 .. budget - 1`` and at most ``budget`` attempts), defaults
    ``exclude_eval_zero=True`` while charging eval 0 against budget, tracks
    attempted/assessable counts and prior unknowns, and distinguishes clean
    right-censoring from uncertainty-censored arms.
    """
    grouped: dict[tuple[str, str, str], list[EvalRecord]] = {}
    for record in records:
        grouped.setdefault((record.policy, record.replicate, record.route), []).append(record)

    arms: list[ArmFirstIssue] = []
    for (policy, replicate, route), group in sorted(grouped.items()):
        is_v2 = metric is not None or any(r.schema_version == 2 or r.outcome.get("schema_version") == 2 for r in group)
        rep_prov = group[0].replicate_provenance if group else "inferred"

        if not is_v2:
            indexed = sorted(
                (record for record in group if record.eval_index is not None),
                key=lambda record: record.eval_index,  # type: ignore[arg-type]
            )
            if not indexed:
                continue
            cutoff_records = [r for r in indexed if (r.eval_index or 0) <= int(budget)] if budget is not None else indexed
            eff_exclude = bool(exclude_eval_zero) if exclude_eval_zero is not None else False
            eligible = [r for r in cutoff_records if (r.eval_index or 0) > 0] if eff_exclude else cutoff_records
            evaluated = max(record.eval_index for record in cutoff_records) if cutoff_records else 0  # type: ignore[arg-type]
            arm_budget = int(budget) if budget is not None else int(evaluated)
            unsafe = [record for record in eligible if record.outcome.get("unsafe")]
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
                    n_attempted=len(eligible),
                    n_assessable=len(eligible),
                    n_unknown_prior=0,
                    censoring_reason="found" if first is not None else "right_censored_complete",
                    metric="unsafe",
                    exclude_eval_zero=eff_exclude,
                    replicate_provenance=rep_prov,
                )
            )
            continue

        eff_exclude = True if exclude_eval_zero is None else bool(exclude_eval_zero)
        selected_metric = metric or "safety_event"
        # Order ties conservatively with a distinct rank per status
        # (not_applicable -> absent -> unknown -> present) ahead of attempt_id/run_id so swapping
        # IDs never changes cutoff inclusion, first_eval, Top-1, n_unknown_prior, or n_assessable.
        _status_rank = {"not_applicable": 0, "absent": 1, "unknown": 2, "present": 3}
        indexed_v2 = sorted(
            (r for r in group if r.eval_index is not None and r.eval_index >= 0),
            key=lambda r: (
                r.eval_index,  # type: ignore[arg-type]
                _status_rank.get(_v2_assessment_status(r.outcome, selected_metric), 2),
                _v2_present_labels(r.outcome),
                r.attempt_id or r.run_id or "",
            ),
        )
        invalid_negative_v2 = [r for r in group if r.eval_index is not None and r.eval_index < 0]
        unindexed_v2 = sorted(
            (r for r in group if r.eval_index is None),
            key=lambda r: (
                _status_rank.get(_v2_assessment_status(r.outcome, selected_metric), 2),
                _v2_present_labels(r.outcome),
                r.attempt_id or r.run_id or "",
            ),
        )
        if not group:
            continue
        if budget is not None:
            b = int(budget)
            # Reserve room for unindexed attempts only up to budget, while respecting 0..b-1 index ceiling
            in_range_indexed = [r for r in indexed_v2 if 0 <= (r.eval_index or 0) < b]
            cutoff_records = (in_range_indexed + unindexed_v2)[:b]
            overflow_indexed = in_range_indexed[min(len(in_range_indexed), b):]
        else:
            cutoff_records = indexed_v2 + unindexed_v2
            overflow_indexed = []
        start_idx = 1 if eff_exclude else 0
        occupied_indices = {r.eval_index for r in cutoff_records if r.eval_index is not None and r.eval_index >= 0}
        max_observed_idx = max(occupied_indices, default=-1)
        arm_budget = int(budget) if budget is not None else max(len(cutoff_records), max_observed_idx + 1)

        # Assign unindexed attempts ONLY to unoccupied eligible slots (never onto a known indexed slot).
        # When exclude_eval_zero=True and eval 0 is unobserved, conservatively reserve excluded slot 0
        # for the highest-ranked unindexed row so an unindexed baseline cannot claim discovery/Top-1 credit.
        indexed_eligible = [
            r for r in cutoff_records
            if r.eval_index is not None and r.eval_index >= 0 and (not eff_exclude or r.eval_index != 0)
        ]
        unindexed_in_budget = [r for r in cutoff_records if r.eval_index is None]
        slot0_unindexed_reserved = eff_exclude and (0 not in occupied_indices) and bool(unindexed_in_budget)
        unindexed_candidates = unindexed_in_budget[:-1] if slot0_unindexed_reserved else unindexed_in_budget
        if budget is not None:
            free_eligible_slots = [i for i in range(start_idx, arm_budget) if i not in occupied_indices]
            eligible_unindexed = unindexed_candidates[:len(free_eligible_slots)]
            # Conservatively place unindexed rows into the latest free eligible slots so partially
            # observed arms never push an unindexed hit forward into an earlier slot or Top-1 credit.
            assigned_free_slots = free_eligible_slots[-len(eligible_unindexed):] if eligible_unindexed else []
        else:
            free_eligible_slots = []
            slot_cursor = max(start_idx, max_observed_idx + 1)
            while len(free_eligible_slots) < len(unindexed_candidates):
                if slot_cursor not in occupied_indices:
                    free_eligible_slots.append(slot_cursor)
                slot_cursor += 1
            eligible_unindexed = unindexed_candidates
            assigned_free_slots = free_eligible_slots
            if assigned_free_slots and assigned_free_slots[-1] + 1 > arm_budget:
                arm_budget = assigned_free_slots[-1] + 1
        slotted_eligible = sorted(
            [(int(r.eval_index), r) for r in indexed_eligible]  # type: ignore[arg-type]
            + list(zip(assigned_free_slots, eligible_unindexed)),
            key=lambda pair: (pair[0], pair[1].eval_index is None or pair[1].eval_index < 0),
        )
        first_eligible_eval = slotted_eligible[0][0] if slotted_eligible else start_idx
        if unindexed_v2:
            earliest_free_eligible = next(
                (i for i in range(start_idx, arm_budget) if i not in occupied_indices),
                None,
            )
            if earliest_free_eligible is not None:
                first_eligible_eval = min(first_eligible_eval, earliest_free_eligible)

        # A single unindexed attempt occupying the sole free eligible slot (with eval 0 accounted for,
        # actually assigned in cutoff_records, and no truncated competing unindexed rows) has a uniquely
        # determined temporal position: negatives before a hit are NOT missing slots, and unknowns
        # strictly after a hit are NOT prior uncertainty.
        unique_unindexed_slot = (
            budget is not None
            and len(eligible_unindexed) == 1
            and len(unindexed_v2) == 1
            and len(free_eligible_slots) == 1
            and (not eff_exclude or 0 in occupied_indices)
        )
        # When every free eligible slot is filled by unindexed rows and ALL unindexed rows are
        # known non-unknown negatives (absent/not_applicable), every valid slot permutation makes
        # every free eligible slot a confirmed non-unknown negative.
        all_free_slots_known_non_unknown = (
            budget is not None
            and bool(free_eligible_slots)
            and len(eligible_unindexed) == len(free_eligible_slots)
            and all(
                _v2_assessment_status(r.outcome, selected_metric) in ("absent", "not_applicable")
                for r in unindexed_v2
            )
        )
        first_eval: int | None = None
        first_eval_is_indexed = False
        first_eval_is_top1 = False
        n_assessable = 0
        n_unknown_prior = 0
        present_labels: set[str] = set()
        known_slot_indices: set[int] = {r.eval_index for r in indexed_eligible if r.eval_index is not None}
        if unique_unindexed_slot:
            known_slot_indices.add(free_eligible_slots[0])
        elif all_free_slots_known_non_unknown:
            known_slot_indices.update(free_eligible_slots)
        for attempt_pos, (assigned_slot, record) in enumerate(slotted_eligible):
            st = _v2_assessment_status(record.outcome, selected_metric)
            is_indexed_rec = record.eval_index is not None and record.eval_index >= 0
            if st in ("present", "absent"):
                n_assessable += 1
            if first_eval is None:
                if st == "present":
                    first_eval = assigned_slot
                    first_eval_is_indexed = is_indexed_rec
                    # Top-1 requires the hit to be the very first eligible attempt in conservative
                    # order (no earlier non-present attempt sharing the slot) at first_eligible_eval.
                    first_eval_is_top1 = (attempt_pos == 0) and (assigned_slot == first_eligible_eval)
                    if budget is None and first_eval + 1 > arm_budget:
                        arm_budget = first_eval + 1
                elif st == "unknown" and (is_indexed_rec or unique_unindexed_slot):
                    n_unknown_prior += 1
            if st == "present":
                present_labels.update(_v2_present_labels(record.outcome))
        # Unindexed rows can ONLY precede first_eval if the arm is censored or at least one free
        # eligible slot exists strictly BEFORE first_eval. When every eligible slot before first_eval
        # is already occupied by an indexed record, no unindexed row can precede the hit.
        free_slots_before_hit = (
            [idx for idx in range(start_idx, first_eval) if idx not in occupied_indices]
            if first_eval is not None
            else None
        )
        unindexed_can_precede_hit = (first_eval is None) or bool(free_slots_before_hit)
        unindexed_unknowns = (
            0
            if (unique_unindexed_slot or not unindexed_can_precede_hit)
            else sum(
                1 for r in eligible_unindexed
                if _v2_assessment_status(r.outcome, selected_metric) == "unknown"
            )
        )
        # If unindexed rows compete across excluded slot 0 or overflow the free eligible slot(s),
        # and any competing unindexed row is unknown/present, slot assignment is ambiguous -> count
        # prior uncertainty whenever an unindexed row could precede the hit (or on a censored arm).
        if (
            free_eligible_slots
            and (slot0_unindexed_reserved or (budget is not None and len(unindexed_v2) > len(free_eligible_slots)))
            and unindexed_can_precede_hit
            and any(_v2_assessment_status(r.outcome, selected_metric) in ("unknown", "present") for r in unindexed_v2)
        ):
            unindexed_unknowns += 1
        if overflow_indexed:
            unindexed_unknowns += sum(
                1 for r in overflow_indexed
                if r.eval_index is not None
                and r.eval_index >= start_idx
                and _v2_assessment_status(r.outcome, selected_metric) in ("unknown", "present")
                and (first_eval is None or r.eval_index < first_eval)
            )
        n_unknown_prior += unindexed_unknowns
        if first_eval is not None:
            missing_prior_slots = sum(1 for idx in range(start_idx, first_eval) if idx not in known_slot_indices)
            n_unknown_prior += missing_prior_slots
        covered_eligible_slots = {slot for slot, _ in slotted_eligible if start_idx <= slot < arm_budget}
        eligible_slots_needed = max(0, arm_budget - start_idx)
        missing_eligible_slots = max(0, eligible_slots_needed - len(covered_eligible_slots))
        if budget is not None:
            missing_budget_slots = max(max(0, arm_budget - len(cutoff_records)), missing_eligible_slots)
        else:
            missing_budget_slots = missing_eligible_slots
        # Preserve arms whose rows all have invalid negative indices and mark their budget unobserved.
        if invalid_negative_v2 and not cutoff_records:
            n_unknown_prior = max(n_unknown_prior, len(invalid_negative_v2))
            missing_budget_slots = max(missing_budget_slots, len(invalid_negative_v2))
        if first_eval is not None:
            reason = "found_with_unknown_prior" if n_unknown_prior > 0 else "found"
        else:
            if n_unknown_prior > 0 or missing_budget_slots > 0:
                reason = "right_censored_with_unknown"
            else:
                reason = "right_censored_complete"
        arms.append(
            ArmFirstIssue(
                policy=policy,
                replicate=replicate,
                route=route,
                first_eval=first_eval,
                budget=arm_budget,
                censored=first_eval is None,
                outcomes=tuple(sorted(present_labels)),
                n_attempted=len(cutoff_records),
                n_assessable=n_assessable,
                n_unknown_prior=n_unknown_prior,
                censoring_reason=reason,
                metric=selected_metric,
                exclude_eval_zero=eff_exclude,
                replicate_provenance=rep_prov,
                first_eligible_eval=first_eligible_eval,
                top1_eligible=first_eval_is_top1,
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
        top1 = sum(
            1 for arm in policy_arms
            if (
                arm.top1_eligible
                if arm.top1_eligible is not None
                else (arm.first_eval is not None and arm.first_eval == arm.first_eligible_eval)
            )
        )
        summary[policy] = {
            "n_arms": len(policy_arms),
            "n_found": len(found),
            "n_found_with_unknown_prior": sum(1 for a in policy_arms if a.censoring_reason == "found_with_unknown_prior"),
            "n_right_censored_complete": sum(1 for a in policy_arms if a.censoring_reason == "right_censored_complete"),
            "n_right_censored_with_unknown": sum(1 for a in policy_arms if a.censoring_reason == "right_censored_with_unknown"),
            "found_rate": (len(found) / len(policy_arms)) if policy_arms else None,
            "top1_count": top1,
            "top1_rate": (top1 / len(policy_arms)) if policy_arms else None,
            "mean_first_found": mean_found,
            "std_first_found": std_found,
            "median_first_found": _median(found_values),
            "mean_first_censored": mean_censored,
            "std_first_censored": std_censored,
            "mean_first_imputed_budget_plus_one": mean_censored,
            "std_first_imputed_budget_plus_one": std_censored,
            "imputation_note": "censored_arms_imputed_as_budget_plus_one",
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
    its first safety issue in fewer evaluations. In schema-2 mode, refuses to pair
    ``unmatched_pooled`` replicates or arms with mismatched budgets.
    """
    is_v2 = any(arm.metric != "unsafe" for arm in arms)
    primary_arms = {(arm.replicate, arm.route): arm for arm in arms if arm.policy == primary}
    comparison_arms = {(arm.replicate, arm.route): arm for arm in arms if arm.policy == comparison}
    shared = sorted(set(primary_arms) & set(comparison_arms))
    excluded_unmatched = 0
    excluded_budget = 0
    excluded_uncertain_censoring = 0
    differences: list[float] = []
    for key in shared:
        p_arm = primary_arms[key]
        c_arm = comparison_arms[key]
        if is_v2:
            if p_arm.replicate_provenance == "unmatched_pooled" or c_arm.replicate_provenance == "unmatched_pooled":
                excluded_unmatched += 1
                continue
            if p_arm.budget != c_arm.budget:
                excluded_budget += 1
                continue
            if (
                p_arm.censoring_reason == "right_censored_with_unknown"
                or c_arm.censoring_reason == "right_censored_with_unknown"
            ):
                excluded_uncertain_censoring += 1
                continue
        p_val = p_arm.first_eval if p_arm.first_eval is not None else p_arm.budget + 1
        c_val = c_arm.first_eval if c_arm.first_eval is not None else c_arm.budget + 1
        differences.append(float(p_val - c_val))
    wins = sum(1 for value in differences if value < 0)
    ties = sum(1 for value in differences if value == 0)
    losses = len(differences) - wins - ties
    mean_diff, std_diff = _mean_std(differences)
    statistic, p_value = _wilcoxon(differences)
    result = {
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
        "imputation_note": "complete_right_censored_arms_imputed_as_budget_plus_one",
    }
    if is_v2:
        result["n_excluded_unmatched_pooled"] = excluded_unmatched
        result["n_excluded_budget_mismatch"] = excluded_budget
        result["n_excluded_uncertain_censoring"] = excluded_uncertain_censoring
    return result


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
    metric: str | None = None,
) -> dict[str, Any]:
    """Associate semantic obligations with safety outcomes across evaluations.

    In schema-2 mode (or when ``metric`` is supplied), uses per-outcome
    assessable denominators (``present + absent``) and reports ``n_unknown``
    explicitly so unobserved outcomes are never counted as safe negatives.
    These are observational associations, never causal effects.
    """
    if source not in {"witnessed", "target"}:
        raise ValueError("source must be 'witnessed' or 'target'")

    rows = [record for record in records if policy is None or record.policy == policy]
    is_v2 = metric is not None or any(r.schema_version == 2 or r.outcome.get("schema_version") == 2 for r in rows)

    present_sets: list[set[str]] = []
    for record in rows:
        if source == "target":
            present_sets.append({record.target} if record.target else set())
        else:
            present_sets.append(set(record.obligations))

    obligations: set[str] = set()
    for values in present_sets:
        obligations.update(values)

    if is_v2:
        selected = metric or "safety_event"
        v2_metrics = [f"{g}.{m}" for g, names in CATEGORIES.items() for m in names]
        if source == "witnessed":
            obs_pairs = [(r, ps) for r, ps in zip(rows, present_sets) if r.semantic_observed]
        else:
            obs_pairs = [(r, ps) for r, ps in zip(rows, present_sets) if r.target is not None]
        obs_rows = [r for r, _ in obs_pairs]
        obs_sets = [ps for _, ps in obs_pairs]
        n_obligation_unknown = len(rows) - len(obs_rows)
        table_v2: dict[str, dict[str, Any]] = {}
        for obligation in sorted(obligations):
            mask = [obligation in values for values in obs_sets]
            support = sum(1 for flag in mask if flag)
            if support < min_support:
                continue
            absent_count = len(obs_rows) - support
            primary_p = [_v2_assessment_status(r.outcome, selected) for r, m in zip(obs_rows, mask) if m]
            primary_a = [_v2_assessment_status(r.outcome, selected) for r, m in zip(obs_rows, mask) if not m]
            p_hits = sum(1 for s in primary_p if s == "present")
            p_denom = sum(1 for s in primary_p if s in ("present", "absent"))
            a_hits = sum(1 for s in primary_a if s == "present")
            a_denom = sum(1 for s in primary_a if s in ("present", "absent"))
            p_rate = (p_hits / p_denom) if p_denom else None
            a_rate = (a_hits / a_denom) if a_denom else None
            entry: dict[str, Any] = {
                "support": support,
                "n_absent": absent_count,
                "n_obligation_unknown": n_obligation_unknown,
                "metric": selected,
                "present_hits": p_hits,
                "present_assessable": p_denom,
                "present_unknown": sum(1 for s in primary_p if s == "unknown"),
                "present_rate": p_rate,
                "absent_hits": a_hits,
                "absent_assessable": a_denom,
                "absent_unknown": sum(1 for s in primary_a if s == "unknown"),
                "absent_rate": a_rate,
                "lift_primary": (p_rate - a_rate) if (p_rate is not None and a_rate is not None) else None,
                "risk_ratio_primary": _risk_ratio(p_rate, a_rate),
                "outcomes": {},
            }
            for mkey in v2_metrics:
                mp = [_v2_assessment_status(r.outcome, mkey) for r, m in zip(obs_rows, mask) if m]
                ma = [_v2_assessment_status(r.outcome, mkey) for r, m in zip(obs_rows, mask) if not m]
                mph = sum(1 for s in mp if s == "present")
                mpd = sum(1 for s in mp if s in ("present", "absent"))
                mah = sum(1 for s in ma if s == "present")
                mad = sum(1 for s in ma if s in ("present", "absent"))
                mpr = (mph / mpd) if mpd else None
                mar = (mah / mad) if mad else None
                entry["outcomes"][mkey] = {
                    "present_hits": mph,
                    "present_assessable": mpd,
                    "present_unknown": sum(1 for s in mp if s == "unknown"),
                    "present_rate": mpr,
                    "absent_hits": mah,
                    "absent_assessable": mad,
                    "absent_unknown": sum(1 for s in ma if s == "unknown"),
                    "absent_rate": mar,
                    "lift": (mpr - mar) if (mpr is not None and mar is not None) else None,
                }
            table_v2[obligation] = entry
        return {
            "schema_version": 2,
            "source": source,
            "policy": policy,
            "metric": selected,
            "n_records": len(rows),
            "n_obligation_observed": len(obs_rows),
            "n_obligation_unknown": n_obligation_unknown,
            "n_obligations": len(table_v2),
            "interpretation": "observational_association_not_causal_effect",
            "obligations": table_v2,
        }

    table: dict[str, dict[str, Any]] = {}
    unsafe_flags = [bool(record.outcome.get("unsafe")) for record in rows]
    for obligation in sorted(obligations):
        mask = [obligation in values for values in present_sets]
        support = sum(1 for flag in mask if flag)
        if support < min_support:
            continue
        absent = len(rows) - support
        entry = {
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
    primary_metric = association.get("metric")
    for obligation, entry in association.get("obligations", {}).items():
        if outcome == "unsafe":
            lift = entry.get("lift_unsafe")
            present_rate = entry.get("unsafe_rate_present")
            support = entry.get("support")
        elif outcome == primary_metric or outcome == entry.get("metric"):
            lift = entry.get("lift_primary")
            present_rate = entry.get("present_rate")
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
