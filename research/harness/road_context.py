"""Map road-rule and waiting-context adapter plus pure stop-line crossing geometry.

``crosses_directed_line`` is pure numeric geometry with no simulator imports.
``RoadContextIndex`` adapts CARLA map/traffic-control state defensively into
``RoadRuleContext`` and ``TaskContext``; unavailable map geometry yields None
rather than fabricated compliance.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from research.harness.interaction_geometry import world_footprint
from research.harness.lane_index import lane_ref_from_waypoint
from research.harness.outcome_models import ActorState, RoadRuleContext, TaskContext


def crosses_directed_line(
    previous_footprint: Sequence[tuple[float, float]],
    current_footprint: Sequence[tuple[float, float]],
    line_p0: tuple[float, float],
    line_p1: tuple[float, float],
    forward_normal: tuple[float, float],
) -> bool:
    """Directed swept front-footprint crossing test for a finite stop line.

    Computes the exact interpolation fraction ``alpha`` in ``(0, 1]`` where the
    swept leading edge crosses the directed line ``s = 0``, and requires the
    footprint's interpolated lateral span AT ``alpha`` to overlap the finite
    segment ``[line_p0, line_p1]``. Rejects reverse motion, parallel motion,
    diagonal motion crossing the infinite line outside the segment, and
    crossings of unrelated lanes.
    """
    if not previous_footprint or not current_footprint or len(previous_footprint) != len(current_footprint):
        return False
    nx, ny = float(forward_normal[0]), float(forward_normal[1])
    norm = math.hypot(nx, ny)
    if norm <= 1e-9:
        return False
    nx, ny = nx / norm, ny / norm

    x0, y0 = float(line_p0[0]), float(line_p0[1])
    x1, y1 = float(line_p1[0]), float(line_p1[1])
    tx, ty = x1 - x0, y1 - y0
    seg_len = math.hypot(tx, ty)
    if seg_len <= 1e-9:
        return False
    ux, uy = tx / seg_len, ty / seg_len

    prev_s_vals = [(px - x0) * nx + (py - y0) * ny for px, py in previous_footprint]
    curr_s_vals = [(px - x0) * nx + (py - y0) * ny for px, py in current_footprint]
    prev_s = max(prev_s_vals)
    curr_s = max(curr_s_vals)
    if not (prev_s <= 0.0 and curr_s > 0.0):
        return False

    # Exact s=0 intersection points of the swept footprint:
    # 1) Vertex trajectories (prev[i] -> curr[i]) crossing s=0 from s0 <= 0 to s1 > 0
    # 2) Current polygon edges (curr[i] -> curr[j]) straddling s=0
    # 3) Any previous vertices already lying exactly on s=0
    u_hits: list[float] = []
    n = len(previous_footprint)
    for i in range(n):
        s0 = prev_s_vals[i]
        s1 = curr_s_vals[i]
        p0 = previous_footprint[i]
        p1 = current_footprint[i]
        if abs(s0) <= 1e-12:
            u_hits.append((p0[0] - x0) * ux + (p0[1] - y0) * uy)
        if s0 <= 0.0 < s1:
            denom = s1 - s0
            if denom > 1e-12:
                alpha = -s0 / denom
                ix = p0[0] + alpha * (p1[0] - p0[0])
                iy = p0[1] + alpha * (p1[1] - p0[1])
                u_hits.append((ix - x0) * ux + (iy - y0) * uy)

    for i in range(n):
        j = (i + 1) % n
        sa, sb = curr_s_vals[i], curr_s_vals[j]
        if (sa <= 0.0 <= sb or sb <= 0.0 <= sa) and abs(sb - sa) > 1e-12:
            t = -sa / (sb - sa)
            pa, pb = current_footprint[i], current_footprint[j]
            ix = pa[0] + t * (pb[0] - pa[0])
            iy = pa[1] + t * (pb[1] - pa[1])
            u_hits.append((ix - x0) * ux + (iy - y0) * uy)

    if not u_hits:
        return False
    return max(u_hits) >= -1e-9 and min(u_hits) <= seg_len + 1e-9


def _map_lane_type(raw: Any) -> str:
    name = str(raw).split(".")[-1].lower()
    mapping = {
        "driving": "driving",
        "shoulder": "shoulder",
        "sidewalk": "sidewalk",
        "bidirectional": "bidirectional",
        "parking": "parking",
        "biking": "biking",
        "restricted": "restricted",
        "median": "median",
        "entry": "entry",
        "exit": "exit",
        "offramp": "offramp",
        "onramp": "onramp",
        "none": "offroad",
    }
    return mapping.get(name, "unknown")


def _map_lane_change(raw: Any) -> str:
    name = str(raw).split(".")[-1].lower()
    if name in {"none", "left", "right", "both"}:
        return name
    return "unknown"


def _map_light_state(raw: Any) -> str:
    name = str(raw).split(".")[-1].lower()
    if "red" in name:
        return "red"
    if "yellow" in name or "amber" in name:
        return "yellow"
    if "green" in name:
        return "green"
    if "off" in name:
        return "off"
    return "unknown"


def _get_any_lane_waypoint(carla_map: Any, loc: Any, *, project_to_road: bool) -> Any:
    """Query CARLA waypoint across all lane types when supported, with mock fallback."""
    lane_type_any = None
    try:
        carla_mod = type(carla_map).__module__.split(".")[0]
        if carla_mod == "carla":
            carla = __import__(carla_mod)
            lane_type_any = getattr(getattr(carla, "LaneType", None), "Any", None)
    except Exception:
        lane_type_any = None
    if lane_type_any is not None:
        try:
            return carla_map.get_waypoint(loc, project_to_road=project_to_road, lane_type=lane_type_any)
        except TypeError:
            pass
    return carla_map.get_waypoint(loc, project_to_road=project_to_road)


class RoadContextIndex:
    """Caches static map references and samples per-frame road/task context."""

    def __init__(
        self,
        world: Any,
        *,
        route_points: Sequence[tuple[float, float]] | None = None,
        rule_profile: dict[str, Any] | None = None,
    ) -> None:
        self._world = world
        self._map: Any | None = None
        self._route_points = tuple(route_points or ())
        self._rule_profile = dict(rule_profile or {})
        self._stop_sign_actors: list[Any] = []
        self._last_traffic_light: Any | None = None
        self._prev_ego_lane: tuple[int, int] | None = None
        self._init_error: str | None = None
        try:
            self._map = world.get_map()
            actors = world.get_actors() if hasattr(world, "get_actors") else []
            for actor in actors:
                tid = str(getattr(actor, "type_id", "")).lower()
                if "stop" in tid and "traffic" in tid:
                    self._stop_sign_actors.append(actor)
        except Exception as exc:
            self._init_error = f"road_context_init_failed:{type(exc).__name__}"

    @classmethod
    def from_world(
        cls,
        world: Any,
        route: Sequence[tuple[float, float]] | None = None,
        rule_profile: dict[str, Any] | None = None,
    ) -> "RoadContextIndex":
        return cls(world, route_points=route, rule_profile=rule_profile)

    def _stop_line_for_waypoint(self, wp: Any) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
        try:
            loc = wp.transform.location
            yaw_rad = math.radians(float(wp.transform.rotation.yaw))
            width = float(getattr(wp, "lane_width", 3.5))
            nx, ny = math.cos(yaw_rad), math.sin(yaw_rad)
            rx, ry = -ny * (width * 0.5), nx * (width * 0.5)
            p0 = (float(loc.x) - rx, float(loc.y) - ry)
            p1 = (float(loc.x) + rx, float(loc.y) + ry)
            return p0, p1, (nx, ny)
        except Exception:
            return None

    def sample(
        self,
        ego_vehicle: Any,
        ego_state: ActorState,
        prev_ego_state: ActorState | None,
        nearby_actors: Sequence[ActorState],
        *,
        objective_active: bool | None = True,
        sensor_wait: bool | None = False,
        actors_available: bool = True,
    ) -> tuple[RoadRuleContext | None, TaskContext | None, dict[str, str]]:
        errors: dict[str, str] = {}
        if self._map is None:
            errors["road_context"] = self._init_error or "map_unavailable"
            return None, None, errors

        lane_id: str | None = None
        ego_road_id: int | None = None
        ego_lane_num: int | None = None
        lane_type: str | None = None
        in_junction: bool | None = None
        lane_change_perm: str | None = None
        permitted_yaw: float | None = None
        speed_limit_mps: float | None = None
        red_light_state: str | None = None
        stop_line_applicable: bool | None = None
        stop_line_crossed: bool | None = None
        stop_sign_applicable: bool | None = None
        stop_sign_approach: bool | None = None
        stop_sign_line_crossed: bool | None = None
        refs: list[str] = []

        try:
            ego_loc = ego_vehicle.get_location()
            wp = _get_any_lane_waypoint(self._map, ego_loc, project_to_road=False)
            if wp is None:
                wp_proj = _get_any_lane_waypoint(self._map, ego_loc, project_to_road=True)
                lane_type = "offroad"
                in_junction = bool(getattr(wp_proj, "is_junction", False)) if wp_proj is not None else False
                refs.append("offroad_projection")
            else:
                lane_ref = lane_ref_from_waypoint(wp)
                if lane_ref is not None:
                    ego_road_id = lane_ref.road_id
                    ego_lane_num = lane_ref.lane_id
                    lane_id = f"{lane_ref.road_id}:{lane_ref.lane_id}"
                lane_type = _map_lane_type(getattr(wp, "lane_type", "unknown"))
                in_junction = bool(getattr(wp, "is_junction", False))
                lane_change_perm = _map_lane_change(getattr(wp, "lane_change", "unknown"))
                permitted_yaw = math.radians(float(wp.transform.rotation.yaw))
                refs.append(f"wp:{lane_id or 'unidentified'}")
        except Exception as exc:
            errors["road_waypoint"] = f"{type(exc).__name__}:{exc}"

        try:
            if hasattr(ego_vehicle, "get_speed_limit"):
                limit_kph = float(ego_vehicle.get_speed_limit())
                if math.isfinite(limit_kph) and limit_kph > 0:
                    speed_limit_mps = limit_kph / 3.6
                    refs.append(f"speed_limit_kph:{limit_kph:.1f}")
        except Exception as exc:
            errors["speed_limit"] = f"{type(exc).__name__}:{exc}"

        # Match stop waypoints against current OR previous (road_id, lane_id) so crossing a
        # stop line directly into a CARLA junction road_id is never dropped.
        allowed_lanes: set[tuple[int, int]] = set()
        if ego_road_id is not None and ego_lane_num is not None:
            allowed_lanes.add((ego_road_id, ego_lane_num))
        if self._prev_ego_lane is not None:
            allowed_lanes.add(self._prev_ego_lane)

        # Traffic light applicability & lane-matched stop-line crossing
        # Cache last active light across the trigger-box exit boundary so crossing on the exit tick is never dropped.
        try:
            if hasattr(ego_vehicle, "is_at_traffic_light") and hasattr(ego_vehicle, "get_traffic_light"):
                at_tl = bool(ego_vehicle.is_at_traffic_light())
                tl = ego_vehicle.get_traffic_light() if at_tl else None
                active_tl = tl if tl is not None else self._last_traffic_light
                self._last_traffic_light = tl
                if active_tl is not None:
                    stop_wps = active_tl.get_stop_waypoints() if hasattr(active_tl, "get_stop_waypoints") else []
                    # Filter stop waypoints to ego's current or approach road/lane when topology is known
                    if allowed_lanes and stop_wps:
                        matched_wps = [
                            swp for swp in stop_wps
                            if (getattr(swp, "road_id", None), getattr(swp, "lane_id", None)) in allowed_lanes
                        ]
                    else:
                        matched_wps = list(stop_wps)
                    if matched_wps or tl is not None:
                        stop_line_applicable = True
                        red_light_state = _map_light_state(active_tl.get_state())
                        if matched_wps and prev_ego_state is not None:
                            prev_fp = world_footprint(prev_ego_state)
                            curr_fp = world_footprint(ego_state)
                            crossed = False
                            for swp in matched_wps:
                                line = self._stop_line_for_waypoint(swp)
                                if line is not None and crosses_directed_line(prev_fp, curr_fp, *line):
                                    crossed = True
                                    break
                            stop_line_crossed = crossed
                            # Drop cached light if it was carried over from prior tick and wasn't crossed
                            if tl is None and not crossed:
                                stop_line_applicable = False
                                stop_line_crossed = False
                                red_light_state = "off"
                        else:
                            stop_line_crossed = None
                        if stop_line_applicable:
                            refs.append(f"tl:{getattr(active_tl, 'id', 'unknown')}:{red_light_state}")
                    else:
                        stop_line_applicable = False
                        stop_line_crossed = False
                        red_light_state = "off"
                else:
                    stop_line_applicable = False
                    stop_line_crossed = False
                    red_light_state = "off"
        except Exception as exc:
            errors["traffic_light"] = f"{type(exc).__name__}:{exc}"

        # Stop-sign applicability filtered by matching current/approach road/lane & heading alignment
        try:
            nearest_stop = None
            nearest_dist = float("inf")
            nearest_wp = None
            for sign in self._stop_sign_actors:
                loc = sign.get_location()
                dist = math.hypot(float(loc.x) - ego_state.x, float(loc.y) - ego_state.y)
                if dist > 12.0:
                    continue
                swp = self._map.get_waypoint(loc, project_to_road=True)
                if swp is None:
                    continue
                if allowed_lanes and (getattr(swp, "road_id", None), getattr(swp, "lane_id", None)) not in allowed_lanes:
                    continue
                swp_yaw = math.radians(float(swp.transform.rotation.yaw))
                if math.cos(swp_yaw - ego_state.yaw_rad) < 0.5:
                    continue
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_stop = sign
                    nearest_wp = swp
            if nearest_stop is not None and nearest_wp is not None:
                stop_sign_applicable = True
                stop_sign_approach = nearest_dist <= 8.0
                if prev_ego_state is not None:
                    line = self._stop_line_for_waypoint(nearest_wp)
                    if line is not None:
                        stop_sign_line_crossed = crosses_directed_line(
                            world_footprint(prev_ego_state), world_footprint(ego_state), *line
                        )
                refs.append(f"stop_sign:{getattr(nearest_stop, 'id', 'unknown')}")
            else:
                stop_sign_applicable = False
                stop_sign_approach = False
                stop_sign_line_crossed = False
        except Exception as exc:
            errors["stop_sign"] = f"{type(exc).__name__}:{exc}"

        if ego_road_id is not None and ego_lane_num is not None:
            self._prev_ego_lane = (ego_road_id, ego_lane_num)

        road_ctx = RoadRuleContext(
            lane_id=lane_id,
            lane_type=lane_type,
            in_junction=in_junction,
            lane_change_permission=lane_change_perm,
            permitted_travel_yaw_rad=permitted_yaw,
            speed_limit_mps=speed_limit_mps,
            red_light_state=red_light_state,
            stop_line_applicable=stop_line_applicable,
            stop_line_crossed=stop_line_crossed,
            stop_sign_applicable=stop_sign_applicable,
            stop_sign_approach=stop_sign_approach,
            stop_sign_line_crossed=stop_sign_line_crossed,
            evidence_refs=tuple(refs),
        )

        # Conservative waiting-context evidence; never fabricate False when actor scan failed!
        red_stop = (
            True if (stop_line_applicable is True and red_light_state == "red")
            else False if stop_line_applicable is not None
            else None
        )
        sign_stop = (
            True if (stop_sign_applicable is True and stop_sign_approach is True)
            else False if stop_sign_applicable is not None
            else None
        )
        if not actors_available:
            blocked: bool | None = None
            yielding: bool | None = None
        else:
            fwd_x, fwd_y = math.cos(ego_state.yaw_rad), math.sin(ego_state.yaw_rad)
            blocked = False
            yielding = False
            for other in nearby_actors:
                dx, dy = other.x - ego_state.x, other.y - ego_state.y
                along = dx * fwd_x + dy * fwd_y
                lateral = abs(-dx * fwd_y + dy * fwd_x)
                other_speed = math.hypot(other.vx, other.vy)
                if 0.0 < along <= 10.0 and lateral <= 2.2 and other_speed < 1.0:
                    blocked = True
                elif 0.0 < along <= 12.0 and lateral <= 4.5 and other_speed >= 0.5:
                    yielding = True
        task_ctx = TaskContext(
            objective_active=objective_active,
            red_light_stop=red_stop,
            stop_sign_stop=sign_stop,
            blocked_route=blocked,
            yielding=yielding,
            sensor_wait=sensor_wait,
            evidence_refs=tuple(refs),
        )
        return road_ctx, task_ctx, errors
