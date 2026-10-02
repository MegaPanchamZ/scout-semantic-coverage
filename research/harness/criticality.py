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

from typing import Any

TTC_REF_S = 1.5
DIST_REF_M = 2.5
COLLISION_SCORE = 2.0


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def criticality_score(row: dict[str, Any]) -> float:
    """Criticality of ``row`` in [0, 2]; invalid/crashed runs score 0."""
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
    for key in ("min_pedestrian_distance_m", "min_vehicle_distance_m"):
        dist = _num(raw.get(key))
        if dist is not None:
            terms.append(DIST_REF_M / max(dist, 0.1))
    return round(min(1.0, max(terms)), 4)
