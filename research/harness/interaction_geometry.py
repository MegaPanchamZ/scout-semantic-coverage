"""Horizontal oriented-box geometry with constant-velocity swept SAT.

No simulator dependency. Heading and vertical intervals are held constant over
the projection horizon. Contact predictions are not collision-sensor events.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from research.harness.outcome_models import ActorState, finite


@dataclass(frozen=True)
class ContactPrediction:
    status: str
    ttc_s: float | None
    horizon_s: float
    model: str = "fixed_heading_constant_velocity_2d_v1"


@dataclass(frozen=True)
class ClosestApproach:
    status: str
    min_clearance_m: float | None
    time_s: float | None
    horizon_s: float
    model: str = "fixed_heading_constant_velocity_2d_v1"


def world_footprint(actor: ActorState) -> tuple[tuple[float, float], ...]:
    c, s = math.cos(actor.yaw_rad), math.sin(actor.yaw_rad)
    x = actor.x + c * actor.bbox_offset_x - s * actor.bbox_offset_y
    y = actor.y + s * actor.bbox_offset_x + c * actor.bbox_offset_y
    c, s = math.cos(actor.yaw_rad + actor.bbox_yaw_rad), math.sin(actor.yaw_rad + actor.bbox_yaw_rad)
    return tuple((x + c * a - s * b, y + s * a + c * b)
                 for a, b in ((-actor.half_length_m, -actor.half_width_m),
                              (actor.half_length_m, -actor.half_width_m),
                              (actor.half_length_m, actor.half_width_m),
                              (-actor.half_length_m, actor.half_width_m)))


def _axes(a: ActorState, b: ActorState):
    for actor in (a, b):
        angle = actor.yaw_rad + actor.bbox_yaw_rad
        yield math.cos(angle), math.sin(angle)
        yield -math.sin(angle), math.cos(angle)


def _projection(points, axis):
    values = [x * axis[0] + y * axis[1] for x, y in points]
    return min(values), max(values)


def _vertical_overlap(a, b):
    return a.z_min_m <= b.z_max_m and b.z_min_m <= a.z_max_m


def constant_velocity_ttc(a: ActorState, b: ActorState, horizon_s: float) -> ContactPrediction:
    finite(horizon_s, "horizon_s")
    if horizon_s <= 0:
        raise ValueError("horizon_s must be positive")
    if not _vertical_overlap(a, b):
        return ContactPrediction("no_contact_within_horizon", None, horizon_s)
    pa, pb = world_footprint(a), world_footprint(b)
    enter, leave = 0.0, horizon_s
    for axis in _axes(a, b):
        lo_a, hi_a = _projection(pa, axis)
        lo_b, hi_b = _projection(pb, axis)
        speed = (b.vx - a.vx) * axis[0] + (b.vy - a.vy) * axis[1]
        if abs(speed) < 1e-12:
            if hi_a < lo_b or hi_b < lo_a:
                return ContactPrediction("no_contact_within_horizon", None, horizon_s)
            continue
        t0, t1 = (lo_a - hi_b) / speed, (hi_a - lo_b) / speed
        enter = max(enter, min(t0, t1))
        leave = min(leave, max(t0, t1))
        if enter > leave:
            return ContactPrediction("no_contact_within_horizon", None, horizon_s)
    return ContactPrediction("already_overlapping" if enter == 0 else "predicted_contact", enter, horizon_s)


def _point_segment_distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    if denom <= 1e-18:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denom))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def footprint_clearance(a: ActorState, b: ActorState) -> float | None:
    """Surface clearance; None means different vertical layers, not zero gap."""
    if not _vertical_overlap(a, b):
        return None
    pa, pb = world_footprint(a), world_footprint(b)
    overlaps = all(not (max(_projection(pa, axis)[0], _projection(pb, axis)[0]) >
                        min(_projection(pa, axis)[1], _projection(pb, axis)[1])) for axis in _axes(a, b))
    if overlaps:
        return 0.0
    return min(_point_segment_distance(p, poly[i], poly[(i + 1) % 4])
               for points, poly in ((pa, pb), (pb, pa)) for p in points for i in range(4))


def _translate_actor(actor: ActorState, t_s: float) -> ActorState:
    if t_s == 0.0:
        return actor
    return ActorState(
        actor.actor_id, actor.actor_type,
        actor.x + actor.vx * t_s, actor.y + actor.vy * t_s,
        actor.yaw_rad, actor.half_length_m, actor.half_width_m,
        actor.vx, actor.vy, actor.z_min_m, actor.z_max_m,
        actor.role, actor.bbox_offset_x, actor.bbox_offset_y, actor.bbox_yaw_rad,
    )


def closest_approach(a: ActorState, b: ActorState, horizon_s: float) -> ClosestApproach:
    """Future minimum oriented-footprint clearance over [0, horizon_s].

    Explicitly distinct from TTC: diagnoses paths that converge closely without
    intersecting. Uses exact SAT contact when paths intersect, and unimodal
    golden-section refinement over [0, horizon_s] otherwise (clearance between two
    translating convex polygons is convex in t).
    """
    finite(horizon_s, "horizon_s")
    if horizon_s <= 0:
        raise ValueError("horizon_s must be positive")
    if not _vertical_overlap(a, b):
        return ClosestApproach("vertical_separation", None, None, horizon_s)
    pred = constant_velocity_ttc(a, b, horizon_s)
    if pred.ttc_s is not None:
        return ClosestApproach(pred.status, 0.0, pred.ttc_s, horizon_s)

    def clearance_at(t: float) -> float:
        gap = footprint_clearance(_translate_actor(a, t), _translate_actor(b, t))
        return float(gap if gap is not None else 0.0)

    # Clearance between two linearly translating convex polygons is convex in t;
    # coarse bracket + golden-section search finds the global minimum reliably.
    steps = 16
    best_i = 0
    best_val = clearance_at(0.0)
    for i in range(1, steps + 1):
        val = clearance_at(horizon_s * i / steps)
        if val < best_val:
            best_val = val
            best_i = i
    lo = horizon_s * max(0, best_i - 1) / steps
    hi = horizon_s * min(steps, best_i + 1) / steps
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - phi * (hi - lo)
    d = lo + phi * (hi - lo)
    fc, fd = clearance_at(c), clearance_at(d)
    for _ in range(36):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - phi * (hi - lo)
            fc = clearance_at(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + phi * (hi - lo)
            fd = clearance_at(d)
    t_star = 0.5 * (lo + hi)
    candidates = ((0.0, clearance_at(0.0)), (t_star, clearance_at(t_star)), (horizon_s, clearance_at(horizon_s)))
    best_t, best_gap = min(candidates, key=lambda pair: (pair[1], pair[0]))
    return ClosestApproach("closest_approach_within_horizon", best_gap, best_t, horizon_s)
