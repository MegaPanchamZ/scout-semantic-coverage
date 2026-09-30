from __future__ import annotations

import importlib
from typing import Any

from research.harness.leaderboard_bridge import ensure_local_carla_agents_on_path


def get_entry_point() -> str:
    return "ResearchLeaderboardBehaviorAgent"


class ResearchLeaderboardBehaviorAgent:
    def __init__(self) -> None:
        self.track = "SENSORS"
        self._agent: Any | None = None

    def setup(self, path_to_conf_file: str | None) -> None:
        del path_to_conf_file
        self.track = "SENSORS"

    def sensors(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "sensor.speedometer",
                "id": "Speed",
            }
        ]

    def bind_harness(self, world: Any, ego_vehicle: Any) -> None:
        del world
        ensure_local_carla_agents_on_path()
        behavior_mod = importlib.import_module("agents.navigation.behavior_agent")
        self._agent = behavior_mod.BehaviorAgent(ego_vehicle, behavior="normal")
        self._agent.get_local_planner().set_speed(30.0)

    def set_global_plan(self, plan: Any, gps_plan: Any | None = None) -> None:
        del gps_plan
        if self._agent is not None:
            self._agent.set_global_plan(plan)

    def run_step(self, input_data: dict[str, tuple[float, Any]], timestamp: float) -> Any:
        del input_data, timestamp
        if self._agent is None:
            raise RuntimeError("Harness binding has not been completed.")
        return self._agent.run_step()

    def done(self) -> bool:
        return bool(self._agent.done()) if self._agent is not None else False

    def destroy(self) -> None:
        return None