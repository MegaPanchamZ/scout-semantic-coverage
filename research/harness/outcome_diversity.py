"""Deterministic, coarse observed-event patterns; not unique-bug clustering."""
from collections import Counter
from dataclasses import asdict
import hashlib
from pathlib import Path

from research.harness.outcome_models import FrameEvidence, strict_json


CRITERION = "ordered_event_role_pattern_v1"
ENRICHED_CRITERION = "enriched_event_context_v1"


def _severity_band(label: str, measurements: dict) -> str | None:
    if label == "collision":
        imp = measurements.get("peak_impulse_ns")
        if isinstance(imp, (int, float)) and not isinstance(imp, bool):
            return "high" if imp >= 5000.0 else ("medium" if imp >= 500.0 else "low")
    elif label in ("near_miss", "conflict_exposure"):
        ttc = measurements.get("min_ttc_s")
        if isinstance(ttc, (int, float)) and not isinstance(ttc, bool):
            return "severe" if ttc < 0.5 else ("moderate" if ttc < 1.0 else "mild")
    elif label == "close_proximity":
        gap = measurements.get("min_gap_m")
        if isinstance(gap, (int, float)) and not isinstance(gap, bool):
            return "critical" if gap < 0.5 else ("close" if gap < 1.5 else "marginal")
    return None


def _event_frame_context(evidence, event_frames) -> dict:
    """Extract road context and manoeuvre cues for the first matching evaluation frame."""
    target = set(event_frames or ())
    if not target:
        return {}
    for rec in evidence:
        if not isinstance(rec, FrameEvidence) or rec.phase != "evaluation":
            continue
        if rec.sim_frame not in target:
            continue
        ctx = {}
        if rec.road_context is not None:
            if rec.road_context.lane_type is not None:
                ctx["lane_type"] = rec.road_context.lane_type
            if rec.road_context.in_junction is not None:
                ctx["in_junction"] = rec.road_context.in_junction
        if rec.reverse is True:
            ctx["manoeuvre"] = "reversing"
        elif rec.steer is not None:
            ctx["manoeuvre"] = "turning" if abs(rec.steer) >= 0.15 else "straight"
        return ctx
    return {}


def _actor_context(evidence, run_id):
    context, ambiguous, roles = {}, set(), {}
    for record in evidence:
        if record.run_id != run_id:
            raise ValueError("Diversity evidence cannot mix run identities")
        if not isinstance(record, FrameEvidence) or record.phase != "evaluation":
            continue
        for actor in (record.ego, *record.actors):
            identity = {"role": actor.role, "actor_type": actor.actor_type}
            if not actor.role or not actor.actor_type or actor.actor_id in context and context[actor.actor_id] != identity:
                ambiguous.add(actor.actor_id)
            if actor.role:
                roles.setdefault(actor.role, set()).add(actor.actor_id)
            context[actor.actor_id] = identity
    for ids in roles.values():
        if len(ids) > 1:
            ambiguous.update(ids)
    return {key: value for key, value in context.items() if key not in ambiguous}


def build_event_signature(event, actor_context, *, rich_context: dict | None = None):
    """Build deterministic event signature from recorded label, state and stable actor roles.

    When ``rich_context`` is supplied (for ``enriched_event_context_v1``),
    appends severity band, road context and manoeuvre cues without ever using
    ephemeral actor IDs or absolute timestamps.
    """
    item = asdict(event) if hasattr(event, "event_id") else event
    missing = [actor_id for actor_id in item["actor_ids"] if actor_id not in actor_context or any(
        not isinstance(actor_context[actor_id].get(key), str) or not actor_context[actor_id][key]
        for key in ("role", "actor_type"))]
    source = {"event_id": item["event_id"], "start_s": item["start_s"], "end_s": item["end_s"]}
    if missing or not item["actor_ids"]:
        return {"status": "unknown", "reasons": ["stable_actor_context_unavailable"], "source": source}
    pattern = {"label": item["label"], "state": item["state"],
               "actors": sorted(({key: actor_context[a][key] for key in ("role", "actor_type")}
                                 for a in item["actor_ids"]), key=strict_json)}
    if rich_context is not None:
        band = _severity_band(item["label"], item.get("measurements") or {})
        if band is not None:
            pattern["severity_band"] = band
        for key in ("lane_type", "in_junction", "manoeuvre"):
            if key in rich_context:
                pattern[key] = rich_context[key]
    return {"status": "available", "pattern": pattern, "source": source}


def build_run_signature(outcome, evidence, *, include_rich_context: bool = False):
    data = outcome.to_dict() if hasattr(outcome, "to_dict") else outcome
    if data.get("schema_version") != 2:
        raise ValueError("Diversity requires schema-2 outcomes")
    evidence = tuple(evidence)
    actors = _actor_context(evidence, data["run_id"])
    signatures = [
        build_event_signature(
            event,
            actors,
            rich_context=_event_frame_context(evidence, event.get("evidence_frames")) if include_rich_context else None,
        )
        for event in data["events"]
    ]
    criterion = ENRICHED_CRITERION if include_rich_context else CRITERION
    scope = {"criterion": criterion, "definition_version": data["definition_version"],
             "profile_digest": data["profile_digest"],
             "signature_implementation_digest": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "implementation_digest": data["provenance"].get("implementation_digest")}
    result = {"scope": scope, "run_id": data["run_id"], "event_sources": [s["source"] for s in signatures]}
    if not signatures:
        return {**result, "status": "not_applicable", "reasons": ["no_observed_event_pattern_not_a_safety_claim"]}
    if any(s["status"] != "available" for s in signatures):
        return {**result, "status": "unknown", "reasons": ["stable_actor_context_unavailable"]}
    if not scope["implementation_digest"]:
        return {**result, "status": "unknown", "reasons": ["implementation_provenance_unavailable"]}
    # Retain order, including simultaneous groups; do not break ties using IDs.
    groups = {}
    for signature in signatures:
        groups.setdefault(signature["source"]["start_s"], []).append(signature["pattern"])
    pattern = {"scope": scope, "ordered_events": [sorted(groups[t], key=strict_json) for t in sorted(groups)],
               "assessment_states": {g: {k: v["status"] for k, v in metrics.items()} for g, metrics in data["categories"].items()},
               "execution_status": data["execution_status"], "termination_reason": data["termination_reason"]}
    digest = hashlib.sha256(strict_json(pattern).encode()).hexdigest()
    return {**result, "status": "available", "digest": digest, "pattern": pattern}


def cluster_outcomes(records):
    """Exact signature grouping within profile/definition/implementation scope.

    records carry an explicit analysis record_id and a build_run_signature
    result. These are source records, not inferred distinct campaign attempts.
    """
    seen, groups, unclustered, scopes = set(), {}, [], {}
    for record in records:
        rid = record["record_id"]
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("Analysis record IDs must be unique nonempty strings")
        seen.add(rid)
        signature = record["signature"]
        scope_id = hashlib.sha256(strict_json(signature["scope"]).encode()).hexdigest()
        bucket = scopes.setdefault(scope_id, {"scope": signature["scope"], "n_records": 0, "n_pattern_assessable": 0,
                                              "n_patterns": 0, "n_repeated_pattern_records": 0})
        bucket["n_records"] += 1
        if signature["status"] != "available":
            unclustered.append({"record_id": rid, "status": signature["status"], "reasons": signature["reasons"]})
            continue
        digest = signature["digest"]
        if digest != hashlib.sha256(strict_json(signature["pattern"]).encode()).hexdigest():
            raise ValueError("Signature content/digest mismatch")
        if signature["pattern"]["scope"] != signature["scope"]:
            raise ValueError("Signature scope mismatch")
        bucket["n_pattern_assessable"] += 1
        groups.setdefault(digest, {"signature_digest": digest, "scope_id": scope_id, "members": [],
                                   "pattern": signature["pattern"]})["members"].append(rid)
    for group in groups.values():
        group["members"].sort()
        group["representative"] = group["members"][0]
        bucket = scopes[group["scope_id"]]
        bucket["n_patterns"] += 1
        bucket["n_repeated_pattern_records"] += len(group["members"]) - 1
    for bucket in scopes.values():
        n = bucket["n_pattern_assessable"]
        bucket["repeat_fraction_of_assessable_records"] = bucket["n_repeated_pattern_records"] / n if n else None
    criteria = {bucket["scope"].get("criterion") for bucket in scopes.values() if bucket["scope"].get("criterion")}
    criterion_label = next(iter(criteria)) if len(criteria) == 1 else ("mixed_scopes" if len(criteria) > 1 else CRITERION)
    return {"criterion": criterion_label, "counting_unit": "source_records_not_deduplicated_attempts",
            "interpretation": "Coarse observed-event patterns, not distinct bugs, root causes, or a matched-budget policy comparison.",
            "n_records": len(seen), "scopes": dict(sorted(scopes.items())),
            "clusters": [groups[k] for k in sorted(groups)], "unclustered": sorted(unclustered, key=lambda r: r["record_id"])}


def summarize_diversity(outcomes):
    """Raw event and label counts; missingness stays visible beside them."""
    events, labels, unknown, states = Counter(), Counter(), Counter(), {}
    for outcome in outcomes:
        data = outcome.to_dict() if hasattr(outcome, "to_dict") else outcome
        events.update(event["label"] for event in data["events"])
        for group, metrics in data["categories"].items():
            for name, assessment in metrics.items():
                counts = states.setdefault(f"{group}.{name}", dict.fromkeys(("present", "absent", "unknown", "not_applicable"), 0))
                counts[assessment["status"]] += 1
                if assessment["status"] == "present":
                    labels[f"{group}.{name}"] += 1
                elif assessment["status"] == "unknown":
                    unknown[f"{group}.{name}"] += 1
    return {"event_episode_counts": dict(sorted(events.items())), "present_record_counts": dict(sorted(labels.items())),
            "unknown_record_counts": dict(sorted(unknown.items())), "assessment_state_counts": dict(sorted(states.items()))}


def cumulative_discovery(ordered_records, *, budget: int | None = None, exclude_eval_zero: bool = False):
    """Cumulative unique pattern discovery by attempted evaluation index.

    ``ordered_records`` is an iterable of ``{"record_id", "eval_index", "signature"}``
    sorted by ``eval_index``. Missing/unassessable attempts still consume budget
    without advancing discovery counts. When ``exclude_eval_zero=True``, baseline
    ``eval_index == 0`` consumes budget and seeds ``baseline_digests`` without
    being credited as a search discovery.
    """
    ordered = sorted(ordered_records, key=lambda r: (r.get("eval_index") is None, r.get("eval_index", 0), r["record_id"]))
    if budget is not None:
        ordered = ordered[: int(budget)]
    criteria = {
        rec["signature"].get("scope", {}).get("criterion")
        for rec in ordered
        if isinstance(rec.get("signature"), dict) and rec["signature"].get("scope", {}).get("criterion")
    }
    criterion_label = next(iter(criteria)) if len(criteria) == 1 else ("mixed_scopes" if len(criteria) > 1 else CRITERION)
    baseline_digests: set[str] = set()
    seen_digests: set[str] = set()
    curve = []
    n_assessable = 0
    n_repeated = 0
    for attempt_idx, rec in enumerate(ordered, 1):
        sig = rec["signature"]
        is_baseline = exclude_eval_zero and rec.get("eval_index") == 0
        is_new = False
        if sig.get("status") == "available":
            digest = sig["digest"]
            if is_baseline:
                baseline_digests.add(digest)
            else:
                n_assessable += 1
                if digest not in seen_digests and digest not in baseline_digests:
                    seen_digests.add(digest)
                    is_new = True
                else:
                    n_repeated += 1
        curve.append({
            "attempt": attempt_idx,
            "eval_index": rec.get("eval_index"),
            "record_id": rec["record_id"],
            "signature_status": sig.get("status"),
            "is_baseline_excluded": is_baseline,
            "is_new_pattern": is_new,
            "cumulative_unique_patterns": len(seen_digests),
        })
    return {
        "criterion": criterion_label,
        "exclude_eval_zero": exclude_eval_zero,
        "n_attempted": len(ordered),
        "n_baseline_patterns": len(baseline_digests),
        "n_assessable": n_assessable,
        "n_unique_patterns": len(seen_digests),
        "n_repeated_patterns": n_repeated,
        "repeat_discovery_rate": (n_repeated / n_assessable) if n_assessable else None,
        "curve": curve,
    }
