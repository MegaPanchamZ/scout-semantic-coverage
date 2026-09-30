from __future__ import annotations

import importlib
from pathlib import Path
import time
from typing import Any


def import_carla() -> Any:
    try:
        return importlib.import_module("carla")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Unable to import 'carla'. Install the CARLA Python API or run from an environment "
            "where the CARLA egg/wheel is available."
        ) from exc


def connect_client(host: str, port: int, timeout_seconds: float) -> Any:
    carla = import_carla()
    client = carla.Client(host, port)
    client.set_timeout(timeout_seconds)
    return client


def _retry(action: Any, attempts: int = 5, sleep_seconds: float = 2.0) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return action()
        except RuntimeError as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(sleep_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry helper failed without capturing an exception.")


def load_world(client: Any, town: str, load_timeout_seconds: float = 60.0) -> Any:
    world = _retry(lambda: client.get_world())
    if world.get_map().name.split("/")[-1] == town:
        return world

    _retry(lambda: client.load_world(town), attempts=3, sleep_seconds=5.0)
    deadline = time.time() + load_timeout_seconds
    while time.time() < deadline:
        world = _retry(lambda: client.get_world())
        if world.get_map().name.split("/")[-1] == town:
            return world
        time.sleep(2.0)
    raise RuntimeError(f"Timed out waiting for CARLA to finish loading town '{town}'.")


def apply_world_settings(world: Any, synchronous_mode: bool, fixed_delta_seconds: float) -> Any:
    settings = world.get_settings()
    original = settings
    settings.synchronous_mode = synchronous_mode
    settings.fixed_delta_seconds = fixed_delta_seconds if synchronous_mode else None
    _retry(lambda: world.apply_settings(settings), attempts=5, sleep_seconds=2.0)
    return original


def restore_world_settings(world: Any, original_settings: Any) -> None:
    _retry(lambda: world.apply_settings(original_settings), attempts=3, sleep_seconds=1.0)


def resolve_weather(world: Any, preset_name: str) -> Any:
    carla = import_carla()
    if not hasattr(carla.WeatherParameters, preset_name):
        raise ValueError(f"Unknown weather preset '{preset_name}'")
    weather = getattr(carla.WeatherParameters, preset_name)
    world.set_weather(weather)
    return weather


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
