from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


def _load_modern_global_route_planner() -> Any:
    workspace_root = Path(__file__).resolve().parents[5]
    agents_root = workspace_root / "CarlaUE" / "PythonAPI" / "carla"
    if str(agents_root) not in sys.path:
        sys.path.insert(0, str(agents_root))
    module_path = workspace_root / "CarlaUE" / "PythonAPI" / "carla" / "agents" / "navigation" / "global_route_planner.py"
    spec = importlib.util.spec_from_file_location("_mres_modern_global_route_planner", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load modern GlobalRoutePlanner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_modern_module = _load_modern_global_route_planner()
_ModernGlobalRoutePlanner = _modern_module.GlobalRoutePlanner


class GlobalRoutePlanner:
    def __init__(self, dao_or_wmap: Any, sampling_resolution: float | None = None) -> None:
        if sampling_resolution is None:
            self._dao: Any | None = dao_or_wmap
            self._planner: Any | None = None
        else:
            self._dao = None
            self._planner = _ModernGlobalRoutePlanner(dao_or_wmap, sampling_resolution)

    def setup(self) -> None:
        if self._planner is None and self._dao is not None:
            self._planner = _ModernGlobalRoutePlanner(self._dao.get_map(), self._dao.get_resolution())

    def trace_route(self, origin: Any, destination: Any) -> Any:
        if self._planner is None:
            self.setup()
        return self._planner.trace_route(origin, destination)

    def __getattr__(self, name: str) -> Any:
        if self._planner is None:
            self.setup()
        return getattr(self._planner, name)