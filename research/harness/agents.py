from __future__ import annotations

import importlib
from typing import Any

from research.harness.config import AgentConfig
from research.harness.leaderboard_bridge import (
    ensure_local_carla_agents_on_path,
    load_leaderboard_agent,
)
from research.harness.pcla_bridge import PclaAdapter


def _load_external_agent(agent_config: AgentConfig, world: Any, ego_vehicle: Any) -> Any:
    if agent_config.module_name is None or agent_config.class_name is None:
        raise ValueError(
            "External agent adapters require both module_name and class_name to be configured."
        )

    from research.harness.leaderboard_bridge import ensure_repo_path

    ensure_repo_path(agent_config.repo_path)
    module = importlib.import_module(agent_config.module_name)
    agent_class = getattr(module, agent_config.class_name)

    init_kwargs = dict(agent_config.init_kwargs)
    if agent_config.checkpoint_path is not None:
        init_kwargs.setdefault("checkpoint_path", str(agent_config.checkpoint_path))

    constructor_attempts = [
        {"vehicle": ego_vehicle, "world": world, **init_kwargs},
        {"ego_vehicle": ego_vehicle, "world": world, **init_kwargs},
        {"vehicle": ego_vehicle, **init_kwargs},
        {**init_kwargs},
    ]
    last_error: Exception | None = None
    for kwargs in constructor_attempts:
        try:
            agent = agent_class(**kwargs)
            break
        except TypeError as exc:
            last_error = exc
    else:
        raise TypeError(
            f"Unable to construct external agent {agent_config.module_name}.{agent_config.class_name} "
            f"with the supported constructor patterns."
        ) from last_error

    missing = [name for name in ("run_step", "set_destination", "done") if not hasattr(agent, name)]
    if missing:
        raise TypeError(
            f"External agent {agent_config.module_name}.{agent_config.class_name} is missing required methods: {missing}"
        )
    return agent


def make_agent(world: Any, ego_vehicle: Any, agent_config: AgentConfig, client: Any | None = None) -> Any:
    if agent_config.kind == "behavior":
        ensure_local_carla_agents_on_path()
        behavior_mod = importlib.import_module("agents.navigation.behavior_agent")
        agent = behavior_mod.BehaviorAgent(ego_vehicle, behavior=agent_config.behavior)
        agent.get_local_planner().set_speed(agent_config.target_speed_kph)
        return agent

    if agent_config.kind == "leaderboard-module":
        if agent_config.module_name is None:
            raise ValueError("Leaderboard-module agents require --agent-module.")
        config_path = str(agent_config.config_path) if agent_config.config_path is not None else None
        return load_leaderboard_agent(world, ego_vehicle, agent_config.repo_path, agent_config.module_name, config_path)

    if agent_config.kind == "pcla":
        ensure_local_carla_agents_on_path()
        if client is None:
            raise ValueError("PCLA agents require a CARLA client instance.")
        if agent_config.pcla_agent_name is None:
            raise ValueError("PCLA agents require --pcla-agent.")
        return PclaAdapter(world, ego_vehicle, client, agent_config.repo_path, agent_config.pcla_agent_name)

    if agent_config.kind in {"external-module", "tcp", "interfuser", "transfuser"}:
        return _load_external_agent(agent_config, world, ego_vehicle)

    raise NotImplementedError(
        f"Agent kind '{agent_config.kind}' is not implemented yet. "
        "Supported adapters: behavior, leaderboard-module, pcla, external-module, tcp, interfuser, transfuser."
    )
