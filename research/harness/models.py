from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CollisionEvent:
    frame: int
    actor_id: int
    actor_type: str
    intensity: float
    injected_actor: bool = False


@dataclass(slots=True)
class SafetyMetrics:
    """Raw safety evidence recorded live by ``SafetyOracle`` for one episode.

    Schema 2 keeps the original four fields and adds the signals needed to
    evaluate unsafe behaviour beyond collision.  These are *evidence*, not
    classified outcomes: the named outcomes and their thresholds live in
    ``research/harness/safety_outcomes.py``.
    """

    schema_version: int = 2
    min_ttc: float = float('inf')
    near_collisions: int = 0  # rising-edge entries into the near-collision band
    near_collision_ticks: int = 0
    min_actor_distance_m: float = float('inf')
    min_vehicle_distance_m: float = float('inf')
    min_pedestrian_distance_m: float = float('inf')
    red_light_violations: int = 0
    rule_violations: int = 0
    lane_departure_events: int = 0
    max_lane_offset_m: float = 0.0
    max_stationary_streak_ticks: int = 0
    stuck_frames: int = 0
    max_deceleration_mps2: float = 0.0
    harsh_braking_events: int = 0
    emergency_manoeuvre_events: int = 0

@dataclass(slots=True)
class ScenarioSpec:
    scenario_id: str
    town: str
    weather_preset: str
    ego_spawn_index: int
    goal_spawn_index: int
    description: str
    max_ticks: int = 500
    npc_vehicle_count: int = 0
    walker_count: int = 0
    controller: str = "route_only"
    controller_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScenarioSpec":
        return cls(
            scenario_id=str(payload["scenario_id"]),
            town=str(payload["town"]),
            weather_preset=str(payload["weather_preset"]),
            ego_spawn_index=int(payload["ego_spawn_index"]),
            goal_spawn_index=int(payload["goal_spawn_index"]),
            description=str(payload["description"]),
            max_ticks=int(payload.get("max_ticks", 500)),
            npc_vehicle_count=int(payload.get("npc_vehicle_count", 0)),
            walker_count=int(payload.get("walker_count", 0)),
            controller=str(payload.get("controller", "route_only")),
            controller_params=dict(payload.get("controller_params", {})),
        )


@dataclass(slots=True)
class RunResult:
    scenario_id: str
    town: str
    agent_kind: str
    succeeded: bool
    dry_run: bool
    ticks_executed: int
    reached_goal: bool
    terminated_by_collision: bool
    collision_count: int
    collisions: list[CollisionEvent] = field(default_factory=list)
    safety_metrics: SafetyMetrics = field(default_factory=SafetyMetrics)
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
