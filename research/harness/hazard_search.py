"""Hazard-obligation-specific multi-parameter search for SCOUT.

Design: different hazard templates expose different parameter
spaces; the search optimises one uncovered obligation/template at a time and
moves on once it is realised. This module provides:

- ``HAZARD_TEMPLATES``: registry of hazard templates, each with its controller,
  hazard classes, a template-specific parameter space, and an ``apply``
  function that mutates a base ScenarioSpec offline.
- ``TemplateSpace``: sampling/mutation within one template's parameter space.
- ``ObligationScheduler``: chooses the active uncovered obligation/template,
  and advances when the target is credited.
- ``generate_lead_braking_spec``: CARLA-connected base-spec generator for the
  lead-vehicle braking template (route polyline is baked into the spec so the
  search can mutate positions offline).

The crossing template reuses the existing 5-D campaign space and base specs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

WORKSPACE = pathlib.Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))


# ---------------------------------------------------------------------------
# Parameter-space primitives (template-specific; independent of spec_path logic)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Dim:
    name: str
    low: float
    high: float
    default: float
    sigma: float
    kind: str = "float"  # float | int

    def sample(self, rng: random.Random) -> float:
        value = rng.uniform(self.low, self.high)
        return int(round(value)) if self.kind == "int" else round(value, 3)

    def clamp(self, value: float) -> float:
        value = max(self.low, min(self.high, value))
        return int(round(value)) if self.kind == "int" else round(value, 3)


@dataclass
class TemplateSpace:
    name: str
    dims: tuple[Dim, ...]

    def sample(self, rng: random.Random) -> dict[str, float]:
        return {d.name: d.sample(rng) for d in self.dims}

    def mutate(self, candidate: dict[str, float], rng: random.Random, scale: float = 1.0) -> dict[str, float]:
        """Gaussian step around ``candidate``; ``scale`` multiplies every sigma."""
        out = {}
        for d in self.dims:
            base = float(candidate.get(d.name, d.default))
            out[d.name] = d.clamp(rng.gauss(base, d.sigma * scale))
        return out

    def defaults(self) -> dict[str, float]:
        return {d.name: d.default for d in self.dims}


# ---------------------------------------------------------------------------
# Offline spec mutation helpers
# ---------------------------------------------------------------------------

def _polyline_point(spec: dict[str, Any], distance_m: float) -> tuple[float, float, float] | None:
    """Interpolate a point at ``distance_m`` along the baked route polyline."""
    poly = (spec.get("controller_params") or {}).get("route_polyline")
    if not poly:
        return None
    remaining = max(0.0, distance_m)
    for i in range(len(poly) - 1):
        x0, y0 = float(poly[i][0]), float(poly[i][1])
        x1, y1 = float(poly[i + 1][0]), float(poly[i + 1][1])
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 1e-6:
            continue
        if remaining <= seg:
            t = remaining / seg
            yaw = math.degrees(math.atan2(y1 - y0, x1 - x0))
            return x0 + t * (x1 - x0), y0 + t * (y1 - y0), yaw
        remaining -= seg
    x1, y1 = float(poly[-1][0]), float(poly[-1][1])
    return x1, y1, 0.0


def apply_crossing(spec: dict[str, Any], params: dict[str, float]) -> dict[str, Any]:
    """Crossing template: delegate to the campaign space's baked geometry."""
    from research.harness.search_space import make_campaign_space

    space = make_campaign_space()
    candidate = {
        "trigger_radius_m": params.get("trigger_radius_m", 20.0),
        "subject_speed_mps": params.get("subject_speed_mps", 1.8),
        "staging_lateral_offset_m": params.get("staging_lateral_offset_m", 7.0),
        "staging_longitudinal_offset_m": params.get("staging_longitudinal_offset_m", 0.0),
        "trigger_tick": params.get("trigger_tick", 400),
    }
    return space.apply_to_payload(spec, candidate)


# Seconds the lead holds its brake before driving off again.
LEAD_RELEASE_S = 4.0


def apply_lead_braking(spec: dict[str, Any], params: dict[str, float]) -> dict[str, Any]:
    """Lead-braking template: move the lead along the baked route, set trigger/brake."""
    out = json.loads(json.dumps(spec))  # deep copy
    cp = out.setdefault("controller_params", {})
    point = _polyline_point(out, float(params.get("lead_gap_m", 25.0)))
    if point is not None:
        x, y, yaw = point
        cp["spawn_transform"] = {
            "location": {"x": x, "y": y, "z": 0.6},
            "rotation": {"pitch": 0.0, "yaw": yaw, "roll": 0.0},
        }
        cp["trigger_location"] = {"x": x, "y": y, "z": 0.6}
        cp["route_anchor_location"] = {"x": x, "y": y, "z": 0.6}
    cp["trigger_radius_m"] = float(params.get("trigger_radius_m", 8.0))
    cp["post_trigger_brake"] = float(params.get("post_trigger_brake", 1.0))
    cp["pre_brake_throttle"] = float(params.get("pre_brake_throttle", 0.3))
    # The lead drives off after braking so the hazard is a braking event, not
    # a permanent roadblock (SCOUT_LEAD_RELEASE_S=none restores the old hold).
    release = os.environ.get("SCOUT_LEAD_RELEASE_S", str(LEAD_RELEASE_S))
    cp["release_after_s"] = None if release.lower() in ("", "none") else float(release)
    return out


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

@dataclass
class HazardTemplate:
    name: str
    controller: str
    hazard_classes: tuple[str, ...]
    relation_targets: tuple[str, ...]
    space: TemplateSpace
    apply: Callable[[dict[str, Any], dict[str, float]], dict[str, Any]]
    base_spec_glob: str


CROSSING_SPACE = TemplateSpace(
    name="pedestrian_crossing",
    dims=(
        Dim("trigger_radius_m", 5.0, 35.0, 20.0, 3.0),
        Dim("subject_speed_mps", 0.8, 4.0, 1.8, 0.4),
        Dim("staging_lateral_offset_m", 1.0, 14.0, 7.0, 1.5),
        Dim("staging_longitudinal_offset_m", -10.0, 30.0, 0.0, 4.0),
        Dim("trigger_tick", 10, 400, 400, 40.0, kind="int"),
    ),
)

LEAD_BRAKING_SPACE = TemplateSpace(
    name="lead_vehicle_braking",
    dims=(
        Dim("lead_gap_m", 12.0, 45.0, 25.0, 5.0),
        Dim("trigger_radius_m", 4.0, 20.0, 8.0, 2.5),
        Dim("post_trigger_brake", 0.4, 1.0, 1.0, 0.15),
        Dim("pre_brake_throttle", 0.0, 0.6, 0.3, 0.15),
    ),
)

HAZARD_TEMPLATES: dict[str, HazardTemplate] = {
    "pedestrian_crossing": HazardTemplate(
        name="pedestrian_crossing",
        controller="threshold_crossing_adversary",
        hazard_classes=("pedestrian_in_path", "other: pedestrian"),
        relation_targets=("crossing_path(pedestrian,ego)", "jaywalking(pedestrian)"),
        space=CROSSING_SPACE,
        apply=apply_crossing,
        base_spec_glob="*_threshold_crossing.json",
    ),
    "lead_vehicle_braking": HazardTemplate(
        name="lead_vehicle_braking",
        controller="lead_vehicle_braking",
        hazard_classes=("other: ahead_or_waiting", "other: start_stop_or_stationary"),
        relation_targets=("same_lane(vehicle,ego)", "in_front_of(vehicle,ego)"),
        space=LEAD_BRAKING_SPACE,
        apply=apply_lead_braking,
        base_spec_glob="*_lead_braking.json",
    ),
}

# hazard-class / obligation -> template routing used by the scheduler
OBLIGATION_TEMPLATE_MAP: dict[str, str] = {
    "pedestrian_in_path": "pedestrian_crossing",
    "other: pedestrian": "pedestrian_crossing",
    "crossing_path(pedestrian,ego)": "pedestrian_crossing",
    "jaywalking(pedestrian)": "pedestrian_crossing",
    "other: ahead_or_waiting": "lead_vehicle_braking",
    "other: start_stop_or_stationary": "lead_vehicle_braking",
    "same_lane(vehicle,ego)": "lead_vehicle_braking",
    "in_front_of(vehicle,ego)": "lead_vehicle_braking",
    "braking(vehicle)": "lead_vehicle_braking",
}
DEFAULT_TEMPLATE = "pedestrian_crossing"


# ---------------------------------------------------------------------------
# Obligation scheduler
# ---------------------------------------------------------------------------

@dataclass
class ObligationScheduler:
    uncovered: set[str]
    current: str | None = None
    history: list[str] = field(default_factory=list)

    @staticmethod
    def _priority(signature: str) -> int:
        if signature.startswith("hazard("):
            return 0
        if "(" in signature and signature.split("(")[0] in {"crossing_path", "in_front_of", "same_lane", "approaching", "oncoming", "adjacent_lane"}:
            return 1
        return 2

    def next_target(self) -> str | None:
        if not self.uncovered:
            return None
        ranked = sorted(self.uncovered, key=lambda s: (self._priority(s), s))
        return ranked[0]

    def template_for(self, signature: str) -> str:
        base = signature
        if signature.startswith("hazard("):
            base = signature[len("hazard("):-1]
        return OBLIGATION_TEMPLATE_MAP.get(base, OBLIGATION_TEMPLATE_MAP.get(signature, DEFAULT_TEMPLATE))

    def observe(self, covered: set[str]) -> bool:
        """Advance if the current target is now covered. Returns True if advanced."""
        self.uncovered.difference_update(covered)
        previous = self.current
        if previous in covered and previous not in self.history:
            self.history.append(previous)
        if self.current not in self.uncovered:
            self.current = self.next_target()
        return previous != self.current

    def select(self, uncovered: set[str], covered: set[str]) -> tuple[str | None, str]:
        """Return the current target obligation and its template name.

        ``uncovered`` is the remaining obligation set (the caller removes
        credited/stalled targets before calling); ``covered`` is what the latest
        run credited and drives advancement and history. This is the single
        selection path used by ``policy_search._pick_target`` -- it must be given
        the real covered set so a credited target advances.
        """
        self.uncovered = set(uncovered)
        self.observe(covered)
        if self.current is None:
            return None, DEFAULT_TEMPLATE
        return self.current, self.template_for(self.current)


# ---------------------------------------------------------------------------
# Stage-2 exploitation and adaptive mutation
# ---------------------------------------------------------------------------

@dataclass
class ExploitTracker:
    """Keep searching a covered obligation while its runs get more critical.

    Stage 1 covers the obligation. Stage 2 (this tracker) holds the scheduler on
    it until ``patience`` consecutive evaluations fail to raise the best
    criticality, or ``cap`` evaluations have been spent after coverage. Only
    then does the scheduler advance to the next gap.
    """

    patience: int = 3
    cap: int = 8
    target: str | None = None
    best: float = 0.0
    stale: int = 0
    spent: int = 0

    def observe(self, target: str | None, covered: bool, criticality: float) -> None:
        if target is None or self.patience <= 0:
            return
        if target != self.target:
            self.target, self.best, self.stale, self.spent = target, 0.0, 0, 0
        if not covered:
            return
        self.spent += 1
        if self.spent == 1 or criticality > self.best + 1e-9:  # first witness sets the baseline
            self.best, self.stale = criticality, 0
        else:
            self.stale += 1

    def holding(self, target: str | None) -> bool:
        """True while ``target`` is covered but still worth exploiting."""
        if target is None or target != self.target or self.patience <= 0 or self.spent == 0:
            return False
        return self.stale < self.patience and self.spent < self.cap


@dataclass
class AdaptiveMutation:
    """Per-elite step-size control: shrink on improvement, widen on stagnation.

    ``scale`` multiplies the template sigmas. After ``restart_after`` stale
    evaluations the caller should draw a fresh uniform sample instead
    (``should_restart``), and ``epsilon`` mixes in uniform exploration.
    """

    scale: float = 1.0
    stale: int = 0
    shrink: float = 0.7
    grow: float = 1.4
    min_scale: float = 0.25
    max_scale: float = 2.5
    restart_after: int = 6

    def update(self, improved: bool) -> None:
        if improved:
            self.scale, self.stale = max(self.min_scale, self.scale * self.shrink), 0
        else:
            self.scale, self.stale = min(self.max_scale, self.scale * self.grow), self.stale + 1

    def should_restart(self) -> bool:
        return self.stale >= self.restart_after

    def reset(self) -> None:
        self.scale, self.stale = 1.0, 0


# ---------------------------------------------------------------------------
# Lead-braking base-spec generation (CARLA-connected)
# ---------------------------------------------------------------------------

def generate_lead_braking_spec(
    host: str,
    port: int,
    town: str,
    ego_spawn_index: int,
    goal_spawn_index: int,
    output: pathlib.Path,
    lead_gap_m: float = 25.0,
    trigger_radius_m: float = 8.0,
    max_ticks: int = 500,
) -> pathlib.Path:
    import carla  # local import: only required for generation

    from research.harness.leaderboard_bridge import ensure_local_carla_agents_on_path
    ensure_local_carla_agents_on_path()

    from research.harness.scenario_gen import _cumulative_distances, _location_to_dict, _sample_route

    client = carla.Client(host, port)
    client.set_timeout(180.0)
    world = client.get_world()
    if town.lower() not in world.get_map().name.lower():
        world = client.load_world(town)
    world_map = world.get_map()
    spawn_points = world_map.get_spawn_points()
    ego_spawn = spawn_points[ego_spawn_index]
    goal_spawn = spawn_points[goal_spawn_index]
    route = _sample_route(world_map, ego_spawn.location, goal_spawn.location, 2.0)
    cumulative = _cumulative_distances(route)
    # route polyline for offline mutation (every sampled waypoint)
    polyline = [[float(wp.transform.location.x), float(wp.transform.location.y)] for wp, _ in route]
    # lead staging point at lead_gap_m along the route
    target_s = lead_gap_m
    point = None
    for (wp, _), s in zip(route, cumulative):
        if s >= target_s:
            point = wp
            break
    if point is None and route:
        point = route[-1][0]
    spawn_transform = point.transform
    spec = {
        "scenario_id": f"{town.lower()}_spawn{ego_spawn_index}_goal{goal_spawn_index}_lead_braking",
        "town": town,
        "weather_preset": "ClearNoon",
        "ego_spawn_index": ego_spawn_index,
        "goal_spawn_index": goal_spawn_index,
        "description": (
            f"Lead-vehicle braking hazard on {town}: lead spawned {lead_gap_m:.0f} m ahead of the ego route; "
            f"brakes when the ego enters a {trigger_radius_m:.0f} m trigger zone."
        ),
        "max_ticks": max_ticks,
        "npc_vehicle_count": 0,
        "walker_count": 0,
        "controller": "lead_vehicle_braking",
        "controller_params": {
            "blueprint_filter": "vehicle.*",
            "pre_brake_throttle": 0.3,
            "post_trigger_brake": 1.0,
            "trigger_radius_m": trigger_radius_m,
            "spawn_transform": {
                "location": _location_to_dict(spawn_transform.location),
                "rotation": {
                    "pitch": float(spawn_transform.rotation.pitch),
                    "yaw": float(spawn_transform.rotation.yaw),
                    "roll": float(spawn_transform.rotation.roll),
                },
            },
            "trigger_location": _location_to_dict(spawn_transform.location),
            "route_anchor_index": int(len(route) // 2),
            "route_anchor_location": _location_to_dict(spawn_transform.location),
            "route_polyline": polyline,
            "subgraph_type": "lead-vehicle-braking",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2))
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description="Hazard-specific search utilities.")
    ap.add_argument("--list", action="store_true", help="List registered hazard templates and spaces.")
    ap.add_argument("--generate-lead-spec", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town01")
    ap.add_argument("--ego-spawn-index", type=int, default=0)
    ap.add_argument("--goal-spawn-index", type=int, default=82)
    ap.add_argument("--output", type=pathlib.Path, default=None)
    args = ap.parse_args()

    if args.list:
        for name, t in HAZARD_TEMPLATES.items():
            dims = ", ".join(f"{d.name}[{d.low},{d.high}]" for d in t.space.dims)
            print(f"{name}: controller={t.controller} | {dims}")
        return
    if args.generate_lead_spec:
        out = args.output or WORKSPACE / "research" / "experiments" / "EXP-020-policy-comparison" / "artifacts" / "base_specs" / (
            f"{args.town.lower()}_spawn{args.ego_spawn_index}_goal{args.goal_spawn_index}_lead_braking.json"
        )
        path = generate_lead_braking_spec(args.host, args.port, args.town, args.ego_spawn_index, args.goal_spawn_index, out)
        print("wrote", path)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
