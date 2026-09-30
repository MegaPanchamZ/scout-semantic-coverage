from __future__ import annotations

from typing import Any, Protocol

from research.harness.models import RunResult, ScenarioSpec


class RunObserver(Protocol):
    def on_run_start(self, scenario: ScenarioSpec, context: dict[str, Any]) -> None:
        ...

    def on_tick(self, tick_index: int, context: dict[str, Any]) -> None:
        ...

    def on_run_end(self, result: RunResult, context: dict[str, Any]) -> None:
        ...


class NoOpObserver:
    def on_run_start(self, scenario: ScenarioSpec, context: dict[str, Any]) -> None:
        return None

    def on_tick(self, tick_index: int, context: dict[str, Any]) -> None:
        return None

    def on_run_end(self, result: RunResult, context: dict[str, Any]) -> None:
        return None
