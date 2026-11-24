"""Semantic-to-outcome linkage and guidance failure-stage diagnostics.

Guards strictly against circular evaluation: outcome-derived predicates (such
as ``colliding``) are separated from antecedent predicates so same-run
collision ground truth is never presented as predictive semantic guidance.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable, Sequence


OUTCOME_DERIVED_PREDICATES = frozenset({"colliding", "collision"})


def _safe_finite_float(val: Any) -> float | None:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    try:
        num = float(val)
    except OverflowError:
        return None
    return num if math.isfinite(num) else None


def _is_outcome_derived(fact: dict[str, Any]) -> bool:
    if fact.get("outcome_derived") is True:
        return True
    pred = str(fact.get("predicate") or "").lower()
    if pred in OUTCOME_DERIVED_PREDICATES:
        return True
    sig = str(fact.get("signature") or "").lower()
    return sig.startswith("colliding(") or sig.startswith("collision(")


def _fact_time(frame_entry: dict[str, Any], fact: dict[str, Any]) -> float | None:
    for src in (fact, frame_entry):
        num = _safe_finite_float(src.get("sim_time_s"))
        if num is not None:
            return num
    return None


def _fact_actors(fact: dict[str, Any]) -> tuple[set[str], set[str]]:
    ids = {str(a) for a in (fact.get("actor_ids") or ()) if a}
    roles = {str(r) for r in (fact.get("actor_roles") or ()) if r}
    return ids, roles


def link_semantics_to_events(
    semantic_frames: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    *,
    antecedent_window_s: float = 3.0,
    actor_role_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Link semantic facts to physical outcome events on the same run/actor/window.

    Requires explicit simulation timestamps, compatible run IDs (when present),
    and compatible actor IDs or stable roles (contradictory actor IDs are never
    bridged via shared role). Outcome-derived predicates (e.g. ``colliding``)
    and simultaneous/post-event facts are separated from antecedent links and
    never counted as predictive evidence.
    """
    role_map = actor_role_map or {}
    antecedent_links: list[dict[str, Any]] = []
    outcome_derived_links: list[dict[str, Any]] = []
    rejected_missing_identity = 0
    rejected_missing_time = 0
    rejected_post_event = 0
    rejected_cross_run = 0

    # Validate semantic facts once so missing time/identity counters do not scale by len(events).
    validated_facts: list[tuple[str | None, float, set[str], set[str], dict[str, Any]]] = []
    for frame_entry in semantic_frames:
        frame_run_id = frame_entry.get("run_id")
        for fact in frame_entry.get("predicates") or ():
            fact_t = _fact_time(frame_entry, fact)
            if fact_t is None:
                rejected_missing_time += 1
                continue
            f_ids, f_roles = _fact_actors(fact)
            if not f_ids and not f_roles:
                rejected_missing_identity += 1
                continue
            fact_run_id = fact.get("run_id") or frame_run_id
            validated_facts.append((str(fact_run_id) if fact_run_id else None, fact_t, f_ids, f_roles, fact))

    for ev in events:
        ev_id = str(ev["event_id"])
        ev_label = str(ev["label"])
        ev_start = float(ev["start_s"])
        ev_end = float(ev["end_s"])
        ev_run_id = str(ev["run_id"]) if ev.get("run_id") else None
        ev_actor_ids = {str(a) for a in (ev.get("actor_ids") or ()) if a}
        ev_roles = {role_map[a] for a in ev_actor_ids if a in role_map}

        non_ego_ev_roles = {r for r in ev_roles if r != "ego"}

        for fact_run_id, fact_t, f_ids, f_roles, fact in validated_facts:
            if ev_run_id is not None and fact_run_id != ev_run_id:
                rejected_cross_run += 1
                continue

            raw_ids = [str(a) for a in (fact.get("actor_ids") or ()) if a]
            raw_roles = [str(r) for r in (fact.get("actor_roles") or ()) if r]
            fact_ego_ids = {"ego"} | {aid for aid, r in role_map.items() if r == "ego"}
            if len(raw_ids) == len(raw_roles):
                fact_ego_ids |= {aid for aid, r in zip(raw_ids, raw_roles) if r == "ego"}

            non_ego_ev_ids_eff = {a for a in ev_actor_ids if a not in fact_ego_ids}
            non_ego_f_ids = {a for a in f_ids if a not in fact_ego_ids}
            non_ego_f_roles = {r for r in f_roles if r != "ego"}

            if non_ego_ev_ids_eff and non_ego_f_ids:
                # Contradictory explicit non-ego actor IDs must never match via shared role.
                actor_match = bool(non_ego_ev_ids_eff & non_ego_f_ids)
            elif non_ego_ev_ids_eff or non_ego_ev_roles:
                actor_match = bool(
                    (non_ego_ev_ids_eff and (non_ego_ev_ids_eff & non_ego_f_ids))
                    or (non_ego_ev_roles and (non_ego_ev_roles & non_ego_f_roles))
                )
            elif ev_actor_ids and f_ids:
                actor_match = bool(ev_actor_ids & f_ids)
            else:
                actor_match = bool((ev_actor_ids & f_ids) or (ev_roles & f_roles))
            if not actor_match:
                continue

            lag_s = ev_start - fact_t
            if _is_outcome_derived(fact):
                if ev_start - 1e-9 <= fact_t <= ev_end + 1e-9 or abs(lag_s) <= antecedent_window_s:
                    outcome_derived_links.append({
                        "event_id": ev_id,
                        "event_label": ev_label,
                        "signature": fact.get("signature"),
                        "predicate": fact.get("predicate"),
                        "fact_time_s": fact_t,
                        "event_start_s": ev_start,
                        "lag_s": round(lag_s, 6),
                        "circular_outcome_derived": True,
                    })
                continue

            if fact_t >= ev_start - 1e-9:
                rejected_post_event += 1
                continue
            if 0.0 < lag_s <= antecedent_window_s + 1e-9:
                antecedent_links.append({
                    "event_id": ev_id,
                    "event_label": ev_label,
                    "signature": fact.get("signature"),
                    "predicate": fact.get("predicate"),
                    "fact_time_s": fact_t,
                    "event_start_s": ev_start,
                    "lag_s": round(lag_s, 6),
                    "matched_actor_ids": sorted(ev_actor_ids & f_ids),
                    "matched_actor_roles": sorted(ev_roles & f_roles),
                    "interpretation": "observational_antecedent_linkage_not_causation",
                })

    return {
        "antecedent_window_s": antecedent_window_s,
        "antecedent_links": antecedent_links,
        "outcome_derived_links": outcome_derived_links,
        "diagnostics": {
            "n_antecedent_links": len(antecedent_links),
            "n_outcome_derived_excluded": len(outcome_derived_links),
            "n_rejected_missing_identity": rejected_missing_identity,
            "n_rejected_missing_time": rejected_missing_time,
            "n_rejected_post_event": rejected_post_event,
            "n_rejected_cross_run": rejected_cross_run,
        },
    }


def evaluate_intended_interaction(
    expectation: dict[str, Any],
    semantic_frames: Sequence[dict[str, Any]],
    physical_evidence: Sequence[Any] = (),
    *,
    ego_actor_id: str | None = None,
    actor_role_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Verify whether the intended actor, predicate conjunction and ordering occurred.

    Conjunctions and predicate ordering are evaluated per actor identity so two
    distinct actors sharing a role cannot be spliced into a false match (including
    when ego has a numeric simulator actor ID). Missing timestamps are never
    coerced to 0.0 and fail ordering checks explicitly. Ordering checks all
    observed timestamps for each predicate so sequences like ``B@0.1, A@0.2, B@0.3``
    satisfy ``A -> B``.
    """
    intended_role = expectation.get("intended_role")
    required = list(expectation.get("required_predicates") or ())
    ordered = list(expectation.get("ordered_predicates") or ())
    window = expectation.get("window_s")
    role_map = actor_role_map or {}

    ego_ids: set[str] = {"ego"}
    if ego_actor_id:
        ego_ids.add(str(ego_actor_id))
    for aid, rname in role_map.items():
        if rname == "ego":
            ego_ids.add(str(aid))

    phys_role_ids: set[str] = set()
    phys_present: bool | None = None
    if physical_evidence:
        for frame in physical_evidence:
            ego_obj = getattr(frame, "ego", None)
            if ego_obj is not None and getattr(ego_obj, "actor_id", None):
                ego_ids.add(str(ego_obj.actor_id))
            if intended_role:
                for a in getattr(frame, "actors", ()):
                    if getattr(a, "role", None) == intended_role and getattr(a, "actor_id", None):
                        phys_role_ids.add(str(a.actor_id))
        if intended_role:
            phys_present = bool(phys_role_ids)

    # Map actor_key -> {token: list[float | None]} (retain all observed timestamps for ordering)
    by_actor: dict[str, dict[str, list[float | None]]] = {}
    for frame_entry in semantic_frames:
        for fact in frame_entry.get("predicates") or ():
            if _is_outcome_derived(fact):
                continue
            pred = str(fact.get("predicate") or "")
            sig = str(fact.get("signature") or "")
            t = _fact_time(frame_entry, fact)
            if window is not None and (t is None or not (window[0] - 1e-9 <= t <= window[1] + 1e-9)):
                continue
            raw_ids = [str(a) for a in (fact.get("actor_ids") or ()) if a]
            raw_roles = [str(r) for r in (fact.get("actor_roles") or ()) if r]
            roles_set = set(raw_roles)
            if intended_role and intended_role not in roles_set:
                continue

            # Bind actor_ids precisely to intended_role when positional pairing or role_map is available.
            actor_keys: list[str] = []
            if intended_role and raw_ids and len(raw_ids) == len(raw_roles):
                actor_keys = sorted({
                    aid for aid, rname in zip(raw_ids, raw_roles)
                    if rname == intended_role and aid not in ego_ids
                })
            elif raw_ids:
                non_ego = [aid for aid in raw_ids if aid not in ego_ids]
                if intended_role and role_map:
                    mapped = [aid for aid in non_ego if role_map.get(aid) == intended_role]
                    actor_keys = sorted(set(mapped or non_ego))
                else:
                    actor_keys = sorted(set(non_ego))
            if not actor_keys:
                if intended_role and not raw_ids:
                    actor_keys = [f"role:{intended_role}"]
                elif not intended_role:
                    actor_keys = ["__ego_or_unidentified__"]
                else:
                    continue

            for actor_key in actor_keys:
                # If physical_evidence observed concrete actor IDs for intended_role, require match
                if phys_role_ids and not actor_key.startswith("role:") and actor_key not in phys_role_ids:
                    continue
                seen_map = by_actor.setdefault(actor_key, {})
                for token in (pred, sig):
                    if not token:
                        continue
                    times_list = seen_map.setdefault(token, [])
                    times_list.append(float(t) if t is not None else None)

    def _eval_actor(seen_map: dict[str, list[float | None]]) -> tuple[list[str], list[str], dict[str, float | None]]:
        missing = [p for p in required if p not in seen_map]
        failures: list[str] = []
        # Check whether there exists a non-decreasing chain of timestamps along `ordered`
        if ordered:
            cur_min_t: float | None = None
            for idx, pred_tok in enumerate(ordered):
                if pred_tok not in seen_map:
                    prev_tok = ordered[idx - 1] if idx > 0 else pred_tok
                    failures.append(f"missing_in_order:{prev_tok}->{pred_tok}")
                    break
                valid_times = sorted(tv for tv in seen_map[pred_tok] if tv is not None)
                if not valid_times:
                    prev_tok = ordered[idx - 1] if idx > 0 else pred_tok
                    failures.append(f"missing_timestamp:{prev_tok}->{pred_tok}")
                    break
                if cur_min_t is None:
                    cur_min_t = valid_times[0]
                else:
                    nxt = next((tv for tv in valid_times if tv + 1e-9 >= cur_min_t), None)
                    if nxt is None:
                        failures.append(f"out_of_order:{ordered[idx - 1]}({cur_min_t})>{pred_tok}({valid_times[-1]})")
                        break
                    cur_min_t = nxt
        first_times: dict[str, float | None] = {}
        for tok, vals in seen_map.items():
            non_null = [v for v in vals if v is not None]
            first_times[tok] = min(non_null) if non_null else None
        return missing, failures, first_times

    best_actor: str | None = None
    best_missing: list[str] = list(required)
    best_failures: list[str] = (
        [f"missing_in_order:{a}->{b}" for a, b in zip(ordered, ordered[1:])]
        if len(ordered) >= 2
        else ([f"missing_in_order:{ordered[0]}"] if len(ordered) == 1 else [])
    )
    best_times: dict[str, float | None] = {}

    for actor_key in sorted(by_actor):
        missing, failures, first_times = _eval_actor(by_actor[actor_key])
        if best_actor is None or (len(missing), len(failures)) < (len(best_missing), len(best_failures)):
            best_actor = actor_key
            best_missing = missing
            best_failures = failures
            best_times = first_times

    order_ok = not best_failures
    witnessed = (not best_missing) and order_ok and (phys_present is not False)
    if phys_present is False:
        stage = "actor_not_instantiated_physically"
    elif best_missing:
        stage = "intended_predicate_not_witnessed"
    elif not order_ok:
        stage = "predicate_order_violated"
    else:
        stage = "witnessed"

    return {
        "witnessed": witnessed,
        "stage": stage,
        "intended_role": intended_role,
        "matched_actor_key": best_actor,
        "physical_actor_observed": phys_present,
        "missing_predicates": best_missing,
        "order_failures": best_failures,
        "first_seen_times_s": dict(sorted(best_times.items())),
    }


def build_guidance_diagnostics(attempts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Trace per-target/template failure stages and search saturation diagnostics."""
    attempts = list(attempts)
    stage_counts: Counter[str] = Counter()
    by_target: dict[str, dict[str, Any]] = {}
    seen_candidates: set[str] = set()
    repeated_candidates = 0
    clipped_mutations = 0
    fallback_selections = 0
    fitness_values: list[float] = []
    seen_clusters: set[str] = set()
    repeat_outcome_clusters = 0
    eligible_for_witness = 0
    eligible_for_outcome = 0

    for row in attempts:
        target = str(row.get("hazard_target") or row.get("template") or "untargeted")
        bucket = by_target.setdefault(
            target,
            {
                "n_attempts": 0,
                "n_execution_failed": 0,
                "n_trigger_activated": 0,
                "n_eligible_witness": 0,
                "n_witness_unavailable": 0,
                "n_target_witnessed": 0,
                "n_new_coverage": 0,
                "n_eligible_outcome": 0,
                "n_outcome_unknown": 0,
                "n_safety_events": 0,
                "stages": Counter(),
            },
        )
        bucket["n_attempts"] += 1

        cand_key = row.get("candidate_key") or row.get("candidate_id")
        if isinstance(cand_key, str) and cand_key:
            if cand_key in seen_candidates:
                repeated_candidates += 1
            seen_candidates.add(cand_key)
        if row.get("mutation_clipped"):
            clipped_mutations += 1
        if row.get("fallback_selection"):
            fallback_selections += 1
        fit_f = _safe_finite_float(row.get("fitness"))
        if fit_f is not None:
            fitness_values.append(fit_f)

        cluster = row.get("outcome_cluster_digest")
        if isinstance(cluster, str) and cluster:
            if cluster in seen_clusters:
                repeat_outcome_clusters += 1
            seen_clusters.add(cluster)

        failed = bool(row.get("run_error") or row.get("execution_status") == "failed")
        triggered = row.get("trigger_activated")
        has_witness_obs = (
            row.get("target_witnessed") is not None
            or isinstance(row.get("engine_run_obligations"), list)
            or isinstance(row.get("semantic_covered_predicates"), list)
            or isinstance(row.get("semantic_fulfilled_obligations"), list)
        )
        witnessed = bool(row.get("target_witnessed") or (target in (row.get("engine_run_obligations") or ())))
        new_cov = (row.get("engine_new_obligations") or 0) > 0
        outcome_v2 = row.get("outcome_v2")
        backend_caps = row.get("backend_capabilities")
        is_v2_row = (
            ("outcome_v2" in row)
            or (row.get("outcome_schema_version") == 2)
            or (row.get("feedback_version") == "v2_normalized")
            or (isinstance(backend_caps, dict) and "native_v2_capture" in backend_caps)
        )
        safety_hit = False
        outcome_assessable = False
        if is_v2_row:
            if (
                isinstance(outcome_v2, dict)
                and outcome_v2.get("schema_version") == 2
                and isinstance(outcome_v2.get("execution_status"), str)
                and outcome_v2.get("execution_status") in {"completed", "partial"}
            ):
                aggs = outcome_v2.get("aggregates")
                se_entry = aggs.get("safety_event") if isinstance(aggs, dict) else None
                se_status = se_entry.get("status") if isinstance(se_entry, dict) else "unknown"
                if se_status == "present":
                    safety_hit = True
                    outcome_assessable = True
                elif se_status == "absent":
                    outcome_assessable = True
        elif "terminated_by_collision" in row:
            outcome_assessable = True
            safety_hit = bool(row.get("terminated_by_collision"))

        if failed:
            bucket["n_execution_failed"] += 1
            stage = "execution_or_setup_failure"
        elif triggered is False:
            stage = "controller_not_triggered"
        elif not has_witness_obs:
            bucket["n_witness_unavailable"] += 1
            stage = "witness_unavailable"
        else:
            eligible_for_witness += 1
            bucket["n_eligible_witness"] += 1
            if not witnessed:
                stage = "target_not_witnessed"
            elif not new_cov:
                stage = "coverage_saturated_no_new_credit"
            elif not outcome_assessable:
                bucket["n_outcome_unknown"] += 1
                stage = "outcome_unknown"
            else:
                eligible_for_outcome += 1
                bucket["n_eligible_outcome"] += 1
                stage = "safety_event_discovered" if safety_hit else "witnessed_without_safety_event"

        if triggered is True:
            bucket["n_trigger_activated"] += 1
        if has_witness_obs and witnessed:
            bucket["n_target_witnessed"] += 1
        if new_cov:
            bucket["n_new_coverage"] += 1
        if safety_hit:
            bucket["n_safety_events"] += 1
        bucket["stages"][stage] += 1
        stage_counts[stage] += 1

    n = len(attempts)
    unique_fitness = len(set(fitness_values))
    tie_rate = (1.0 - unique_fitness / len(fitness_values)) if len(fitness_values) >= 2 else None

    return {
        "n_attempts": n,
        "denominators": {
            "n_attempts": n,
            "eligible_for_witness": eligible_for_witness,
            "eligible_for_outcome": eligible_for_outcome,
            "n_witness_unavailable": stage_counts.get("witness_unavailable", 0),
            "n_outcome_unknown": stage_counts.get("outcome_unknown", 0),
        },
        "failure_stage_counts": dict(sorted(stage_counts.items())),
        "repeated_candidates": repeated_candidates,
        "clipped_mutations": clipped_mutations,
        "fallback_selections": fallback_selections,
        "fitness_tie_rate": tie_rate,
        "repeat_outcome_clusters": repeat_outcome_clusters,
        "by_target": {
            k: {**v, "stages": dict(sorted(v["stages"].items()))}
            for k, v in sorted(by_target.items())
        },
        "note": "Diagnostic stage counts are observational hypotheses for trace inspection, not automatic causal proofs.",
    }
