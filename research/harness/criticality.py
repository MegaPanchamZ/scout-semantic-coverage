"""Scalar criticality of one evaluated run, used to guide the hazard search.

Semantic coverage says *which* situation was realised; criticality says how
close that situation came to failing the ADS. The score is a pure function of
the named safety metrics already stored on every row, so it is cheap and can be
recomputed offline for archived rows.

Range: 0 (benign) .. 2 (collision). Near misses fall in between:
``max(1.5 s / min_ttc, 2.5 m / min_distance)`` clipped to 1.

Peak deceleration is deliberately *not* a term: in the pilot every run, benign
ones included, showed 17-27 m/s^2 from the ADS slamming to a full stop (the
per-tick speed difference), which pinned the score at 1.0 and removed any
gradient. The harsh-braking outcome still reports it separately.
"""

from __future__ import annotations

import math
from typing import Any

TTC_REF_S = 1.5
DIST_REF_M = 2.5
COLLISION_SCORE = 2.0
FEEDBACK_VERSIONS = ("legacy_v1", "v2_normalized")


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        num = float(value)
    except OverflowError:
        return None
    return num if math.isfinite(num) else None


def _criticality_v2(row: dict[str, Any]) -> float | None:
    """Compute criticality from normalized schema-2 evidence; None if unavailable."""
    outcome = row.get("outcome_v2")
    if not isinstance(outcome, dict) or outcome.get("schema_version") != 2:
        return None
    exec_status = outcome.get("execution_status")
    if not isinstance(exec_status, str) or exec_status not in {"completed", "partial"}:
        return None
    raw_categories = outcome.get("categories")
    categories = raw_categories if isinstance(raw_categories, dict) else {}
    raw_safety = categories.get("safety")
    safety = raw_safety if isinstance(raw_safety, dict) else {}
    col = safety.get("collision")
    col_status = col.get("status") if isinstance(col, dict) else None
    if col_status == "present":
        return COLLISION_SCORE

    terms: list[float] = []
    raw_events = outcome.get("events")
    if isinstance(raw_events, (list, tuple)):
        for ev in raw_events:
            if not isinstance(ev, dict):
                continue
            raw_meas = ev.get("measurements")
            meas = raw_meas if isinstance(raw_meas, dict) else {}
            ttc = _num(meas.get("min_ttc_s"))
            if ttc is not None and math.isfinite(ttc):
                terms.append(TTC_REF_S / ttc if ttc > 0 else 1.0)
            gap = _num(meas.get("min_gap_m"))
            if gap is not None and math.isfinite(gap):
                terms.append(DIST_REF_M / max(gap, 0.1))

    raw_prov = outcome.get("provenance")
    prov = raw_prov if isinstance(raw_prov, dict) else {}
    raw_opps = prov.get("interaction_opportunities")
    if isinstance(raw_opps, (list, tuple)):
        for opp in raw_opps:
            if not isinstance(opp, dict):
                continue
            gap = _num(opp.get("min_gap_m"))
            if gap is not None and math.isfinite(gap):
                terms.append(DIST_REF_M / max(gap, 0.1))
            ttc = _num(opp.get("min_ttc_s"))
            if ttc is not None and math.isfinite(ttc):
                terms.append(TTC_REF_S / ttc if ttc > 0 else 1.0)

    # Distinguish complete negative observation (0.0) from unavailable measurements (None).
    # Returning 0.0 requires confirmed collision absence AND no present/unknown near-miss or proximity.
    nm_entry = safety.get("near_miss")
    nm_status = nm_entry.get("status") if isinstance(nm_entry, dict) else None
    raw_beh = categories.get("behaviour")
    beh = raw_beh if isinstance(raw_beh, dict) else {}
    prox_entry = beh.get("close_proximity")
    prox_status = prox_entry.get("status") if isinstance(prox_entry, dict) else None
    confirmed_negative = (
        col_status == "absent"
        and nm_status in ("absent", "not_applicable")
        and prox_status in ("absent", "not_applicable")
        and "absent" in (nm_status, prox_status)
    )

    if terms:
        score = round(min(1.0, max(terms)), 4)
        if score > 0.0:
            return score
        return 0.0 if confirmed_negative else None

    return 0.0 if confirmed_negative else None


def criticality_score(row: dict[str, Any], *, feedback_version: str = "legacy_v1") -> float | None:
    """Criticality of ``row`` in [0, 2].

    ``legacy_v1`` preserves the historical heuristic (returns 0.0 on invalid/crashed
    or missing rows). ``v2_normalized`` reads schema-2 evidence explicitly and
    returns ``None`` when feedback is unavailable rather than fabricating zero risk.
    """
    if feedback_version == "v2_normalized":
        return _criticality_v2(row)
    if feedback_version != "legacy_v1":
        raise ValueError(f"Unsupported feedback_version: {feedback_version}")
    if row.get("run_error") or row.get("safety_valid") is False:
        return 0.0
    if row.get("terminated_by_collision") or (_num(row.get("collision_count")) or 0) > 0:
        return COLLISION_SCORE
    raw = row.get("safety_raw") or {}
    terms = [0.0]
    ttc = _num(raw.get("min_ttc_s"))
    if ttc is not None and ttc > 0:
        terms.append(TTC_REF_S / ttc)
    elif ttc is not None:
        terms.append(1.0)
    for key in ("min_pedestrian_distance_m", "min_vehicle_distance_m", "min_actor_distance_m"):
        dist = _num(raw.get(key))
        if dist is not None and math.isfinite(dist):
            terms.append(DIST_REF_M / max(dist, 0.1))
    gap = _num(row.get("min_gap_m"))
    if gap is not None and math.isfinite(gap):
        terms.append(DIST_REF_M / max(gap, 0.1))
    return round(min(1.0, max(terms)), 4)

