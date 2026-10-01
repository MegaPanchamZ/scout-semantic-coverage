from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.harness.agents import make_agent
from research.harness.carla_utils import (
    apply_world_settings,
    connect_client,
    ensure_output_dir,
    import_carla,
    load_world,
    resolve_weather,
    restore_world_settings,
)
from research.harness.config import AppConfig
from research.harness.models import RunResult, ScenarioSpec
from research.harness.observers.base import NoOpObserver, RunObserver
from research.harness.oracles import CollisionOracle, SafetyOracle
from research.harness.scenario_runtime import build_scenario_controller


def _vehicle_control_to_dict(control: Any) -> dict[str, Any]:
    return {
        "throttle": float(getattr(control, "throttle", 0.0)),
        "steer": float(getattr(control, "steer", 0.0)),
        "brake": float(getattr(control, "brake", 0.0)),
        "hand_brake": bool(getattr(control, "hand_brake", False)),
        "reverse": bool(getattr(control, "reverse", False)),
        "manual_gear_shift": bool(getattr(control, "manual_gear_shift", False)),
        "gear": int(getattr(control, "gear", 0)),
    }


def _speed_mps(ego_vehicle: Any) -> float:
    velocity = ego_vehicle.get_velocity()
    return float((velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5)


def _full_brake_control(carla: Any) -> Any:
    control = carla.VehicleControl()
    control.brake = 1.0
    return control


def _spawn_actor_with_retries(world: Any, blueprint: Any, transform: Any, attempts: int = 10, ticks_between_attempts: int = 2) -> Any:
    actor = world.try_spawn_actor(blueprint, transform)
    if actor is not None:
        return actor

    retry_transform = transform
    retry_transform.location.z += 0.25
    for _ in range(attempts - 1):
        for _ in range(ticks_between_attempts):
            world.tick()
        actor = world.try_spawn_actor(blueprint, retry_transform)
        if actor is not None:
            return actor
    return None


def _clear_dynamic_actors_near_spawn(world: Any, spawn_location: Any, radius_m: float = 4.0) -> None:
    for actor in world.get_actors():
        type_id = str(getattr(actor, "type_id", ""))
        if not (type_id.startswith("vehicle.") or type_id.startswith("walker.")):
            continue
        try:
            actor_location = actor.get_location()
        except RuntimeError:
            continue
        if actor_location.distance(spawn_location) > radius_m:
            continue
        try:
            actor.destroy()
        except RuntimeError:
            continue


class HarnessRunner:
    def __init__(self, config: AppConfig, observers: list[RunObserver] | None = None) -> None:
        self.config = config
        self.observers = observers or [NoOpObserver()]

    def run(self, scenario: ScenarioSpec) -> RunResult:
        if self.config.run.dry_run:
            return RunResult(
                scenario_id=scenario.scenario_id,
                town=scenario.town,
                agent_kind=self.config.agent.kind,
                succeeded=True,
                dry_run=True,
                ticks_executed=0,
                reached_goal=False,
                terminated_by_collision=False,
                collision_count=0,
                notes=["Dry-run completed. No CARLA connection attempted."],
                metadata={"weather_preset": scenario.weather_preset},
            )

        return self._run_live(scenario)

    def _run_live(self, scenario: ScenarioSpec) -> RunResult:
        client = connect_client(
            self.config.harness.host,
            self.config.harness.port,
            self.config.harness.timeout_seconds,
        )
        world = load_world(client, scenario.town)
        if self.config.harness.reload_world:
            world = client.reload_world()
        import random
        import numpy as np
        random.seed(self.config.harness.seed)
        np.random.seed(self.config.harness.seed)
        if self.config.agent.kind in {"autovla", "pcla", "leaderboard-module"}:
            import torch
            torch.manual_seed(self.config.harness.seed)
        original_settings = apply_world_settings(
            world,
            self.config.harness.synchronous_mode,
            self.config.harness.fixed_delta_seconds,
        )
        collision_oracle = CollisionOracle()
        safety_oracle = SafetyOracle(tick_seconds=self.config.harness.fixed_delta_seconds)
        scenario_controller = build_scenario_controller(scenario)
        actors: list[Any] = []
        reached_goal = False
        terminated_by_collision = False
        ticks_executed = 0
        notes: list[str] = []
        context: dict[str, Any] = {"client": client, "world": world, "scenario": scenario}

        try:
            resolve_weather(world, scenario.weather_preset)
            carla = import_carla()
            spawn_points = world.get_map().get_spawn_points()
            start_transform = spawn_points[scenario.ego_spawn_index]
            goal_transform = spawn_points[scenario.goal_spawn_index]
            _clear_dynamic_actors_near_spawn(world, start_transform.location)

            blueprint = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
            ego_vehicle = _spawn_actor_with_retries(world, blueprint, start_transform)
            if ego_vehicle is None:
                raise RuntimeError("Failed to spawn ego vehicle for harness shakedown")
            actors.append(ego_vehicle)

            for _ in range(self.config.run.warmup_ticks):
                world.tick()

            if self.config.sensors.attach_collision_sensor:
                actors.append(collision_oracle.attach(world, ego_vehicle))

            agent = make_agent(world, ego_vehicle, self.config.agent, client=client)
            agent.set_destination(goal_transform.location)
            route_preview = agent.get_route_preview() if hasattr(agent, "get_route_preview") else []
            route_waypoints = (
                agent.get_route_waypoints()
                if hasattr(agent, "get_route_waypoints")
                else route_preview
            )
            context.update(
                {
                    "ego_vehicle": ego_vehicle,
                    "agent": agent,
                    "collision_oracle": collision_oracle,
                    "carla": carla,
                    "goal_location": goal_transform.location,
                    "route_preview": route_preview,
                    "route_waypoints": route_waypoints,
                    "scenario_notes": notes,
                }
            )
            scenario_controller.setup(context)
            collision_oracle.injected_actor_ids = {actor.id for actor in scenario_controller.spawned_actors}
            actors.extend(scenario_controller.spawned_actors)

            for observer in self.observers:
                observer.on_run_start(scenario, context)

            startup_hold_ticks = max(int(self.config.run.startup_hold_ticks), 0)
            if startup_hold_ticks:
                hold_control = _full_brake_control(carla)
                context["startup_hold_ticks"] = startup_hold_ticks
                context["startup_hold_applied"] = True
                for _ in range(startup_hold_ticks):
                    ego_vehicle.apply_control(hold_control)
                    world.tick()
            else:
                context["startup_hold_ticks"] = 0
                context["startup_hold_applied"] = False

            for tick in range(scenario.max_ticks):
                control = agent.run_step()
                context["last_control"] = _vehicle_control_to_dict(control)
                ego_vehicle.apply_control(control)
                world.tick()
                ticks_executed = tick + 1
                safety_oracle.tick(world, ego_vehicle, goal_transform.location)
                speed_mps = _speed_mps(ego_vehicle)
                ego_location = ego_vehicle.get_location()
                agent_step_info = (
                    agent.get_debug_state()
                    if hasattr(agent, "get_debug_state")
                    else dict(getattr(agent, "last_step_info", {}))
                )
                context["telemetry"] = {
                    "tick": ticks_executed,
                    "control": context["last_control"],
                    "speed_mps": speed_mps,
                    "speed_kph": speed_mps * 3.6,
                    "distance_to_goal_m": float(ego_location.distance(goal_transform.location)),
                    "location": {
                        "x": float(ego_location.x),
                        "y": float(ego_location.y),
                        "z": float(ego_location.z),
                    },
                    "agent_step": agent_step_info,
                    "safety": safety_oracle.snapshot(),
                }
                scenario_controller.on_tick(tick, context)

                for observer in self.observers:
                    observer.on_tick(tick, context)

                if collision_oracle.events and self.config.run.stop_on_first_collision:
                    terminated_by_collision = True
                    first_collision = collision_oracle.events[0]
                    notes.append(
                        f"Terminated on first collision at frame {first_collision.frame} with {first_collision.actor_type}."
                    )
                    break

                if agent.done() or ego_vehicle.get_location().distance(goal_transform.location) < 5.0:
                    reached_goal = True
                    break

            succeeded = True
            if not reached_goal:
                notes.append("Goal not reached within allotted ticks.")

            result = RunResult(
                scenario_id=scenario.scenario_id,
                town=scenario.town,
                agent_kind=self.config.agent.kind,
                succeeded=succeeded,
                dry_run=False,
                ticks_executed=ticks_executed,
                reached_goal=reached_goal,
                terminated_by_collision=terminated_by_collision,
                collision_count=len(collision_oracle.events),
                collisions=list(collision_oracle.events),
                safety_metrics=safety_oracle.metrics,
                notes=notes,
                metadata={
                    "weather_preset": scenario.weather_preset,
                    "goal_spawn_index": scenario.goal_spawn_index,
                    "ego_spawn_index": scenario.ego_spawn_index,
                    "startup_hold_ticks": startup_hold_ticks,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            for observer in self.observers:
                observer.on_run_end(result, context)
        finally:
            try:
                scenario_controller.teardown(context)
            except RuntimeError:
                pass
            if "agent" in context and hasattr(context["agent"], "destroy"):
                try:
                    context["agent"].destroy()
                except Exception:
                    pass
            collision_oracle.destroy()
            if self.config.run.cleanup_actors:
                for actor in reversed(actors):
                    try:
                        actor.destroy()
                    except RuntimeError:
                        pass
            try:
                restore_world_settings(world, original_settings)
            except RuntimeError:
                pass

        if self.config.result.write_json:
            self.write_result(result, self.config.harness.output_dir)
        return result

    @staticmethod
    def write_result(result: RunResult, output_dir: Path) -> Path:
        target_dir = ensure_output_dir(output_dir)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = target_dir / f"{result.scenario_id}-{result.agent_kind}-{timestamp}.json"
        out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return out_path
