from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class BackendRequest:
    backend: str
    scenario_id: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)