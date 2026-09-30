from __future__ import annotations

import json
from pathlib import Path

from research.harness.models import ScenarioSpec


SCENARIO_CATALOG: dict[str, ScenarioSpec] = {
    "town01_straight_micro": ScenarioSpec(
        scenario_id="town01_straight_micro",
        town="Town01",
        weather_preset="ClearNoon",
        ego_spawn_index=0,
        goal_spawn_index=233,
        description="Minimal straight-line Town01 route for nominal PCLA/InterFuser profiling.",
        max_ticks=120,
    ),
    "town01_clear_short": ScenarioSpec(
        scenario_id="town01_clear_short",
        town="Town01",
        weather_preset="ClearNoon",
        ego_spawn_index=0,
        goal_spawn_index=82,
        description="Shakedown route in Town01 under clear weather.",
        max_ticks=400,
    ),
    "town01_rain_short": ScenarioSpec(
        scenario_id="town01_rain_short",
        town="Town01",
        weather_preset="WetCloudyNoon",
        ego_spawn_index=0,
        goal_spawn_index=82,
        description="Weather-perturbed safe route used for false-positive checks.",
        max_ticks=450,
    ),
    "town01_fog_short": ScenarioSpec(
        scenario_id="town01_fog_short",
        town="Town01",
        weather_preset="SoftRainSunset",
        ego_spawn_index=0,
        goal_spawn_index=82,
        description="Temporary fog-like placeholder until custom weather tuples are introduced.",
        max_ticks=450,
    ),
    "town01_jaywalker_crossing": ScenarioSpec(
        scenario_id="town01_jaywalker_crossing",
        town="Town01",
        weather_preset="ClearNoon",
        ego_spawn_index=0,
        goal_spawn_index=82,
        description="Hand-authored jaywalker crossing ahead of the ego route.",
        max_ticks=450,
        walker_count=1,
        controller="jaywalker",
        controller_params={
            "forward_offset": 14.0,
            "right_offset": 4.5,
            "walker_speed": 1.8,
            "crossing_direction": -1.0,
        },
    ),
    "town01_occluded_jaywalker": ScenarioSpec(
        scenario_id="town01_occluded_jaywalker",
        town="Town01",
        weather_preset="ClearNoon",
        ego_spawn_index=0,
        goal_spawn_index=82,
        description="Jaywalker partially occluded by a stopped large vehicle near the crossing path.",
        max_ticks=500,
        walker_count=1,
        npc_vehicle_count=1,
        controller="occluded_jaywalker",
        controller_params={
            "forward_offset": 16.0,
            "right_offset": 5.0,
            "walker_speed": 1.6,
            "crossing_direction": -1.0,
        },
    ),
    # --- PtoP SVGD-generated scenarios ---
    "ptop_town01_svgd_20npc": ScenarioSpec(
        scenario_id="ptop_town01_svgd_20npc",
        town="Town01",
        weather_preset="ClearNoon",
        ego_spawn_index=0,
        goal_spawn_index=82,
        description="PtoP SVGD-generated scenario with 20 adversarial NPCs (mixed vehicle/pedestrian).",
        max_ticks=500,
        npc_vehicle_count=10,
        walker_count=10,
        controller="ptop_adversarial",
        controller_params={
            "ptop_seed": True,
            "npc_count": 20,
            "k_attack": 3,
            "svgd_steps": 8,
        },
    ),
    "ptop_town01_svgd_5npc": ScenarioSpec(
        scenario_id="ptop_town01_svgd_5npc",
        town="Town01",
        weather_preset="ClearNoon",
        ego_spawn_index=0,
        goal_spawn_index=82,
        description="PtoP SVGD-generated lightweight scenario with 5 adversarial NPCs.",
        max_ticks=400,
        npc_vehicle_count=3,
        walker_count=2,
        controller="ptop_adversarial",
        controller_params={
            "ptop_seed": True,
            "npc_count": 5,
            "k_attack": 2,
            "svgd_steps": 5,
        },
    ),
    "ptop_town01_rain_svgd": ScenarioSpec(
        scenario_id="ptop_town01_rain_svgd",
        town="Town01",
        weather_preset="WetCloudyNoon",
        ego_spawn_index=0,
        goal_spawn_index=82,
        description="PtoP SVGD scenario under rain weather for perception-stress testing.",
        max_ticks=500,
        npc_vehicle_count=10,
        walker_count=10,
        controller="ptop_adversarial",
        controller_params={
            "ptop_seed": True,
            "npc_count": 20,
            "k_attack": 3,
            "svgd_steps": 8,
        },
    ),
}


def get_scenario(scenario_id: str) -> ScenarioSpec:
    try:
        return SCENARIO_CATALOG[scenario_id]
    except KeyError as exc:
        known = ", ".join(sorted(SCENARIO_CATALOG))
        raise KeyError(f"Unknown scenario '{scenario_id}'. Known scenarios: {known}") from exc


def resolve_scenario(
    scenario_id: str,
    *,
    town: str | None = None,
    weather_preset: str | None = None,
    ego_spawn_index: int | None = None,
    goal_spawn_index: int | None = None,
    description: str | None = None,
) -> ScenarioSpec:
    if all(value is None for value in (town, weather_preset, ego_spawn_index, goal_spawn_index, description)):
        return get_scenario(scenario_id)

    if None in (town, weather_preset, ego_spawn_index, goal_spawn_index):
        raise ValueError(
            "Scenario overrides require --town, --weather-preset, --ego-spawn-index, and --goal-spawn-index together."
        )

    base = SCENARIO_CATALOG.get(scenario_id)
    max_ticks = base.max_ticks if base is not None else 500
    return ScenarioSpec(
        scenario_id=scenario_id,
        town=str(town),
        weather_preset=str(weather_preset),
        ego_spawn_index=int(ego_spawn_index),
        goal_spawn_index=int(goal_spawn_index),
        description=description or (base.description if base is not None else "Ad hoc scenario override."),
        max_ticks=max_ticks,
        npc_vehicle_count=base.npc_vehicle_count if base is not None else 0,
        walker_count=base.walker_count if base is not None else 0,
        controller=base.controller if base is not None else "route_only",
        controller_params=dict(base.controller_params) if base is not None else {},
    )


def load_scenario_spec(path: Path) -> ScenarioSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenario = ScenarioSpec.from_dict(payload)
    if not scenario.description:
        scenario.description = f"Externally loaded scenario from {path.name}."
    return scenario
