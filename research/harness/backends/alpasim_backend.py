from __future__ import annotations

from typing import Any

from research.harness.backend_models import BackendRequest
from research.harness.backends.base import BackendAdapter


class AlpaSimBackendAdapter(BackendAdapter):
    name = "alpasim"

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
            "scene_id": scenario_id,
            "driver": {
                "model_type": str(base_scenario.get("driver_model") or "alpamayo1"),
                "required_modalities": list(world_model_request.get("expected_outputs", {}).keys()),
            },
            "runtime": {
                "backend": "grpc",
                "needs_route_submission": True,
                "needs_egomotion_submission": True,
                "needs_multiview_images": bool(world_model_request.get("expected_outputs", {}).get("multi_view_rgb", False)),
            },
            "bindings": dict(sorted(bindings.items())),
            "stsg_targets": stsg_targets,
            "world_model_request": world_model_request,
            "notes": [
                "AlpaSim backend request compiled from semantic DSL.",
                "Intended for driver/runtime integration through AlpaSim gRPC services.",
                "First-pass adapter compiles session intent and semantic constraints without assuming a specific scene asset pipeline."
            ],
        }
        return BackendRequest(backend=self.name, scenario_id=scenario_id, payload=payload)