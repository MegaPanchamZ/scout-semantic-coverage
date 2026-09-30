from __future__ import annotations

from typing import Any

from research.harness.backend_models import BackendRequest
from research.harness.backends.base import BackendAdapter


class CarlaBackendAdapter(BackendAdapter):
    name = "carla"

    def build_request(
        self,
        *,
        scenario_id: str,
        description: str,
        language_spec: str,
        bindings: dict[str, str],
        stsg_targets: dict[str, Any],
        base_scenario: dict[str, Any],
        world_model_request: dict[str, Any],
    ) -> BackendRequest:
        payload = {
            "town": str(base_scenario.get("town") or "Town01"),
            "weather_preset": str(bindings.get("weather") or base_scenario.get("weather_preset") or "ClearNoon"),
            "ego_spawn_index": int(base_scenario.get("ego_spawn_index", 0)),
            "goal_spawn_index": int(base_scenario.get("goal_spawn_index", 0)),
            "controller": str(base_scenario.get("controller") or "semantic_variant"),
            "bindings": dict(sorted(bindings.items())),
            "stsg_targets": stsg_targets,
            "world_model_request": world_model_request,
            "notes": [
                "CARLA backend request compiled from semantic DSL.",
                "Execution remains compatible with ScenarioSpec-style routing."
            ],
        }
        return BackendRequest(backend=self.name, scenario_id=scenario_id, payload=payload)