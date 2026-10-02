from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from typing import Any


def _candidate_agents_roots() -> list[Path]:
    """Ordered candidate ``.../PythonAPI/carla`` roots that contain ``agents.navigation``.

    The CARLA PythonAPI is not shipped in the pip ``carla`` wheel, so the
    harness resolves the real agents package at runtime. ``CARLA_ROOT`` is the
    documented location (see ``REPRODUCING_SCOUT.md``); the remaining entries
    keep the historical workspace/Windows layouts working.
    """
    workspace_root = Path(__file__).resolve().parents[5]
    candidates: list[Path] = []
    env_root = os.environ.get("CARLA_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "PythonAPI" / "carla")
    candidates.extend(
        [
            Path("/opt/carla/PythonAPI/carla"),
            workspace_root / "CarlaUE" / "PythonAPI" / "carla",
            Path(r"H:\CARLA_0.9.16\WindowsNoEditor\PythonAPI\carla"),
            Path(r"H:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla"),
            Path(r"H:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla"),
        ]
    )
    return candidates


def _find_modern_global_route_planner() -> Path | None:
    for agents_root in _candidate_agents_roots():
        module_path = agents_root / "agents" / "navigation" / "global_route_planner.py"
        if module_path.is_file():
            return module_path
    return None


def _load_modern_global_route_planner() -> Any:
    module_path = _find_modern_global_route_planner()
    if module_path is None:
        searched = ", ".join(str(path) for path in _candidate_agents_roots())
        raise ImportError(
            "Unable to locate CARLA's GlobalRoutePlanner. Set CARLA_ROOT to the "
            "extracted CARLA directory that contains CarlaUE4.sh (the PythonAPI "
            f"agents package is required). Searched: {searched}"
        )
    agents_root = module_path.parents[2]
    if str(agents_root) not in sys.path:
        sys.path.insert(0, str(agents_root))
    spec = importlib.util.spec_from_file_location("_scout_modern_global_route_planner", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load modern GlobalRoutePlanner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ModernGlobalRoutePlanner: Any | None = None


def _modern_global_route_planner_class() -> Any:
    """Resolve CARLA's real GlobalRoutePlanner lazily, so importing this shim is offline-safe."""
    global _ModernGlobalRoutePlanner
    if _ModernGlobalRoutePlanner is None:
        _ModernGlobalRoutePlanner = _load_modern_global_route_planner().GlobalRoutePlanner
    return _ModernGlobalRoutePlanner


class GlobalRoutePlanner:
    def __init__(self, dao_or_wmap: Any, sampling_resolution: float | None = None) -> None:
        if sampling_resolution is None:
            self._dao: Any | None = dao_or_wmap
            self._planner: Any | None = None
        else:
            self._dao = None
            self._planner = _modern_global_route_planner_class()(dao_or_wmap, sampling_resolution)

    def setup(self) -> None:
        if self._planner is None and self._dao is not None:
            self._planner = _modern_global_route_planner_class()(self._dao.get_map(), self._dao.get_resolution())

    def trace_route(self, origin: Any, destination: Any) -> Any:
        if self._planner is None:
            self.setup()
        return self._planner.trace_route(origin, destination)

    def __getattr__(self, name: str) -> Any:
        if self._planner is None:
            self.setup()
        return getattr(self._planner, name)
