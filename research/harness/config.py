from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class HarnessConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    timeout_seconds: float = 60.0
    traffic_manager_port: int = 8000
    synchronous_mode: bool = True
    fixed_delta_seconds: float = 0.1
    seed: int = 42
    reload_world: bool = False
    output_dir: Path = Path("research/logs/runs")


@dataclass(slots=True)
class AgentConfig:
    kind: str = "behavior"
    behavior: str = "normal"
    target_speed_kph: float = 30.0
    repo_path: Path | None = None
    module_name: str | None = None
    pcla_agent_name: str | None = None
    class_name: str | None = None
    checkpoint_path: Path | None = None
    config_path: Path | None = None
    init_kwargs: dict[str, Any] = field(default_factory=dict)
    # VLA serving backend selection (currently AutoVLA): torch | http | openai
    autovla_backend: str = "torch"
    autovla_endpoint: str | None = None
    autovla_timeout: float = 60.0
    autovla_model: str = "autovla"


@dataclass(slots=True)
class SensorConfig:
    attach_collision_sensor: bool = True


@dataclass(slots=True)
class RunConfig:
    max_ticks: int = 500
    warmup_ticks: int = 15
    startup_hold_ticks: int = 0
    cleanup_actors: bool = True
    dry_run: bool = False
    stop_on_first_collision: bool = True


@dataclass(slots=True)
class ResultConfig:
    write_json: bool = True
    include_tick_trace: bool = False


@dataclass(slots=True)
class AppConfig:
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    run: RunConfig = field(default_factory=RunConfig)
    result: ResultConfig = field(default_factory=ResultConfig)
