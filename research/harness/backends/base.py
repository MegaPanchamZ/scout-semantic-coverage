from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from research.harness.backend_models import BackendRequest


class BackendAdapter(ABC):
    name: str

    @abstractmethod
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
        raise NotImplementedError