from __future__ import annotations

from typing import Any


class GlobalRoutePlannerDAO:
    def __init__(self, world_map: Any, sampling_resolution: float) -> None:
        self._world_map = world_map
        self._sampling_resolution = sampling_resolution

    def get_map(self) -> Any:
        return self._world_map

    def get_resolution(self) -> float:
        return self._sampling_resolution

    def get_topology(self) -> Any:
        return self._world_map.get_topology()

    def get_waypoint(self, location: Any) -> Any:
        return self._world_map.get_waypoint(location)