"""Planar route arc-length tracking with explicit continuity and reacquisition.

Routes can loop or cross themselves. A projection may only move within the
declared speed/time bound of its previous route position. After loss, callers
must provide a trusted route anchor with reset(); no global nearest jump occurs.
"""
from __future__ import annotations

import hashlib
import math

from research.harness.outcome_models import ActorState, RouteProgress, finite, strict_json


class RouteProgressTracker:
    def __init__(self, route_points, *, initial_s_m: float = 0, max_offset_m: float = 3,
                 max_speed_mps: float = 60, continuity_slack_m: float = 1,
                 max_sample_interval_s: float = .11):
        self.points = tuple(tuple(point) for point in route_points)
        if len(self.points) < 2:
            raise ValueError("A route requires at least two points")
        for point in self.points:
            if len(point) != 2:
                raise ValueError("Route points must be planar (x, y)")
            for value in point:
                finite(value, "route coordinate")
        self.segments = []
        total = 0.0
        for a, b in zip(self.points, self.points[1:]):
            length = math.dist(a, b)
            if length == 0:
                continue
            self.segments.append((a, b, total, length))
            total += length
        if total == 0:
            raise ValueError("Route length must be positive")
        self.total_m = total
        self.route_id = hashlib.sha256(strict_json({"model": "planar_polyline_v1", "points": self.points}).encode()).hexdigest()
        for name, value in (("max_offset_m", max_offset_m), ("max_speed_mps", max_speed_mps),
                            ("continuity_slack_m", continuity_slack_m), ("max_sample_interval_s", max_sample_interval_s)):
            finite(value, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        self._epoch = -1
        self.reset(initial_s_m)

    def reset(self, s_m: float, *, sim_time_s: float | None = None):
        finite(s_m, "s_m")
        if not 0 <= s_m <= self.total_m:
            raise ValueError("Anchor outside route")
        if sim_time_s is not None:
            finite(sim_time_s, "sim_time_s")
            if sim_time_s < 0:
                raise ValueError("Negative simulation time")
        self._epoch += 1
        self._s = s_m
        self._high_water = s_m
        self._time = sim_time_s
        self._lost = False
        self._last = RouteProgress(self.route_id, "unknown", self.total_m, reason="awaiting_observation", continuity_id=self._epoch)

    def update(self, actor: ActorState, sim_time_s: float, *, discontinuity: bool = False) -> RouteProgress:
        finite(sim_time_s, "sim_time_s")
        if sim_time_s < 0 or (self._time is not None and sim_time_s <= self._time):
            raise ValueError("Route timestamps must increase")
        dt = 0 if self._time is None else sim_time_s - self._time
        self._time = sim_time_s
        if discontinuity or dt > self.max_sample_interval_s + 1e-9:
            self._lost = True
        reason = "continuity_lost_requires_anchor"
        if not self._lost:
            candidates = []
            bound = self.max_speed_mps * dt + self.continuity_slack_m
            for a, b, start, length in self.segments:
                dx, dy = b[0] - a[0], b[1] - a[1]
                t = max(0, min(1, ((actor.x - a[0]) * dx + (actor.y - a[1]) * dy) / length**2))
                s_m = start + t * length
                distance = math.hypot(actor.x - a[0] - t * dx, actor.y - a[1] - t * dy)
                if abs(s_m - self._s) <= bound + 1e-9 and distance <= self.max_offset_m:
                    candidates.append((distance, abs(s_m - self._s), s_m))
            if candidates:
                distance, _, s_m = min(candidates)
                self._s = s_m
                self._high_water = max(self._high_water, s_m)
                self._last = RouteProgress(self.route_id, "available", self.total_m, s_m, self._high_water, distance, continuity_id=self._epoch)
                return self._last
            self._lost = True
            reason = "no_continuous_projection_within_route_corridor"
        self._last = RouteProgress(self.route_id, "unknown", self.total_m, reason=reason, continuity_id=self._epoch)
        return self._last

    def summary(self) -> RouteProgress:
        return self._last

    def provenance(self) -> dict:
        """Attach to capture provenance when this tracker supplies route evidence."""
        return {"model": "planar_polyline_v1", "route_id": self.route_id,
                "total_m": self.total_m, "threshold_status": "experimental",
                "max_offset_m": self.max_offset_m, "max_speed_mps": self.max_speed_mps,
                "continuity_slack_m": self.continuity_slack_m,
                "max_sample_interval_s": self.max_sample_interval_s}
