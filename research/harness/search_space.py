"""Declarative candidate search space for the EXP-020 policy-comparison campaign.

The 768-evaluation pilot sampled a single scalar,
``controller_params.trigger_radius_m`` in ``[5, 35]``. Coverage saturated and no
policy separated (Holm p >= 0.978). The pre-registered campaign (paper Section
6.2) requires at least four dimensions so that a coverage gap implies a
distinguishable action. This module is the single source of truth for that
space: ``proof-of-concept/policy_search.py`` and ``research/harness/optimiser.py``
both consume it, and the legacy one-dimensional space stays available for
reproducing archived sessions.

Default campaign dimensions and where each value lands in the ScenarioSpec:

====================== =========================================================== ==========================
Dimension              ScenarioSpec path                                           Runtime coverage
====================== =========================================================== ==========================
``trigger_radius_m``   ``controller_params.trigger_radius_m``                      live (distance trigger)
``subject_speed_mps``  ``controller_params.speed``                                 live (walker release speed)
``staging_lateral_offset_m`` ``controller_params.spawn_transform.location`` and     spec-baked: the writer
                       ``controller_params.destination_location``                  recomputes both locations
                                                                                   symmetrically about the
                                                                                   route-anchor crossing axis
``staging_longitudinal_offset_m`` ``controller_params.spawn_transform.location``    spec-baked: the writer
                       and ``controller_params.destination_location``              shifts both locations
                                                                                   together along the route
                                                                                   heading (positive = further
                                                                                   ahead of the ego along the
                                                                                   route direction)
``trigger_tick``       ``controller_params.trigger_tick``                          1-based logical tick at
                                                                                   10 Hz (``seconds =
                                                                                   tick / 10``); additive
                                                                                   runtime support releases
                                                                                   the adversary at the
                                                                                   deadline even if the ego
                                                                                   never enters the radius
====================== =========================================================== ==========================

Runtime notes:

- ``trigger_radius_m``, ``subject_speed_mps`` and ``trigger_tick`` are consumed
  by ``ThresholdCrossingAdversaryController`` in
  ``research/harness/scenario_runtime.py``. No runtime work remains for the
  default campaign dimensions: the controller already read ``speed`` and
  ``trigger_radius_m``, and the ``trigger_tick`` deadline was added alongside
  this module.
- The two staging dimensions need no runtime change because the writer bakes
  them into ``spawn_transform.location`` and ``destination_location``, which the
  controller already consumes: ``spawn = anchor + right * lateral + forward *
  longitudinal`` and ``destination = anchor - right * lateral + forward *
  longitudinal``, where ``right`` is the unit crossing axis and ``forward`` is
  the route heading orthogonalised against it. This requires a spec that
  carries ``spawn_transform`` and ``destination_location`` about the anchor (as
  produced by ``scenario_gen.py`` and ``dota_to_scenario.py``). Both
  ``staging_lateral_offset_m`` and ``staging_longitudinal_offset_m`` keys are
  annotations used for round-trip reads and never reach the simulator.
- A future dimensional extension for ``adversary_kind == "vehicle"`` should map
  subject speed onto ``controller_params.vehicle_throttle``; that runtime path
  exists but is not part of the default walker space.
- Ego speed (checked 2026-09-28): there is no clean spec-level lever, so no
  ``ego_speed_cap_mps`` dimension is declared. ``ScenarioSpec`` has no ego-speed
  field (``research/harness/models.py``), and ``controller_params`` is consumed
  only by ``build_scenario_controller`` in
  ``research/harness/scenario_runtime.py``, which actuates adversary actors. The
  ego is driven exclusively by the ADS agent (``agent.run_step()`` in
  ``research/harness/runner.py``), whose only speed knob is
  ``AgentConfig.target_speed_kph`` (``research/harness/config.py``) set from the
  run-level CLI ``--target-speed-kph``; the built-in ``behavior`` agent applies
  it (``agents.py`` line 62) while PCLA/leaderboard/external agents own their
  speed policy and ignore it. A ``controller_params.ego_speed_cap_mps`` key
  would be silently ignored by the runtime and would fake a dimension.

Bounds policy: the trigger radius keeps the pilot's ``[5, 35]`` range for
continuity; the other bounds are campaign defaults chosen to span the
observable behaviour of the threshold-crossing walker family and are
overridable by rebuilding the dimensions. The longitudinal staging bounds of
``[-10, 30]`` m keep a negative gate inside the generator's 12 m
``trigger_lead_distance_m`` and extend up to roughly the trigger-radius scale
downstream of the junction, so the walker can be staged anywhere the threshold
template can plausibly release it.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from research.harness.models import ScenarioSpec


TICKS_PER_SECOND = 10.0
STAGING_OFFSET_KINDS = ("lateral_offset_m", "longitudinal_offset_m")
DIMENSION_KINDS = ("float", "int") + STAGING_OFFSET_KINDS
LATERAL_OFFSET_ANNOTATION = "staging_lateral_offset_m"
LONGITUDINAL_OFFSET_ANNOTATION = "staging_longitudinal_offset_m"


class SearchSpaceError(ValueError):
    """Raised when a candidate or spec does not match the declarative space."""


@dataclass(frozen=True, slots=True)
class SearchDimension:
    """One declarative axis of the candidate space."""

    name: str
    spec_path: str
    low: float
    high: float
    kind: str = "float"
    step: float | None = None
    decimals: int = 3
    sigma: float | None = None
    default: float | None = None
    runtime: str = "live"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise SearchSpaceError("Search dimension names must be non-empty.")
        if self.kind not in DIMENSION_KINDS:
            raise SearchSpaceError(
                f"Unknown kind '{self.kind}' for dimension '{self.name}'; expected one of {DIMENSION_KINDS}."
            )
        for field_name in ("low", "high", "step", "sigma", "default"):
            raw = getattr(self, field_name)
            if raw is not None:
                object.__setattr__(self, field_name, float(raw))
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise SearchSpaceError(f"Dimension '{self.name}' bounds must be finite.")
        if self.high < self.low:
            raise SearchSpaceError(f"Dimension '{self.name}' has high < low ({self.high} < {self.low}).")
        if self.step is not None and (not math.isfinite(self.step) or self.step <= 0):
            raise SearchSpaceError(f"Dimension '{self.name}' step must be positive and finite.")
        if self.decimals < 0:
            raise SearchSpaceError(f"Dimension '{self.name}' decimals must be non-negative.")

    def coerce(self, raw: Any) -> int | float:
        """Round/clamp-format a raw numeric value to the dimension's grid."""
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise SearchSpaceError(f"Dimension '{self.name}' expects a number, got {raw!r}.") from exc
        if not math.isfinite(value):
            raise SearchSpaceError(f"Dimension '{self.name}' received a non-finite value {raw!r}.")
        if self.step is not None:
            value = self.low + round((value - self.low) / self.step) * self.step
        if self.kind == "int":
            return int(round(value))
        return round(value, self.decimals)

    def in_bounds(self, raw: Any) -> bool:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return False
        value = float(raw)
        if not math.isfinite(value):
            return False
        return self.low - 1e-9 <= value <= self.high + 1e-9

    def validate(self, raw: Any) -> None:
        if not self.in_bounds(raw):
            raise SearchSpaceError(
                f"Dimension '{self.name}' value {raw!r} is outside [{self.low}, {self.high}]."
            )

    def missing_value(self) -> int | float:
        base = self.default if self.default is not None else (self.low + self.high) / 2.0
        return self.coerce(max(self.low, min(self.high, float(base))))

    @property
    def sigma_or_default(self) -> float:
        if self.sigma is not None:
            return float(self.sigma)
        span = self.high - self.low
        return span / 10.0 if span > 0 else 0.0

    def sample_value(self, rng: random.Random) -> int | float:
        if self.kind == "int":
            if self.step is not None and self.step > 1:
                levels = int(math.floor((self.high - self.low) / self.step)) + 1
                raw: float = self.low + rng.randrange(levels) * self.step
            else:
                raw = float(rng.randint(int(math.ceil(self.low)), int(math.floor(self.high))))
        else:
            raw = rng.uniform(self.low, self.high)
        return self.coerce(raw)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec_path": self.spec_path,
            "low": self.low,
            "high": self.high,
            "kind": self.kind,
            "step": self.step,
            "decimals": self.decimals,
            "sigma": self.sigma,
            "default": self.default,
            "runtime": self.runtime,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """A concrete point in a :class:`SearchSpace`."""

    values: dict[str, int | float]

    def to_dict(self) -> dict[str, int | float]:
        return dict(self.values)

    def __getitem__(self, name: str) -> int | float:
        return self.values[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchCandidate":
        return cls(values={str(key): value for key, value in dict(payload).items()})


def _nested_get(payload: Mapping[str, Any], dotted_path: str) -> Any:
    cursor: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _nested_set(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = dotted_path.split(".")
    cursor: dict[str, Any] = payload
    for key in keys[:-1]:
        next_cursor = cursor.get(key)
        if not isinstance(next_cursor, dict):
            raise SearchSpaceError(f"Search path '{dotted_path}' does not exist in the scenario spec.")
        cursor = next_cursor
    cursor[keys[-1]] = value


def _controller_params(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    controller_params = payload.get("controller_params")
    if not isinstance(controller_params, Mapping):
        raise SearchSpaceError("Scenario spec has no 'controller_params' mapping.")
    return controller_params


def _staging_geometry(
    controller_params: Mapping[str, Any],
) -> tuple[dict[str, float], tuple[float, float], tuple[float, float], float]:
    """Return ``(anchor, right, forward, lateral_norm)`` for the staging frame.

    ``right`` is the unit lane-perpendicular crossing axis: the normalised
    spawn-anchor delta, falling back to the spawn yaw's right vector when the
    spawn sits on the anchor. ``forward`` is the unit route-forward axis from
    the spawn yaw, orthogonalised against ``right`` so the two staging axes
    compose independently. ``lateral_norm`` is the original spawn-anchor
    distance, used as the base lateral offset when a space does not search it.
    """
    spawn_transform = controller_params.get("spawn_transform")
    if not isinstance(spawn_transform, Mapping) or not isinstance(spawn_transform.get("location"), Mapping):
        raise SearchSpaceError("Staging offset requires 'controller_params.spawn_transform.location'.")
    destination = controller_params.get("destination_location")
    if not isinstance(destination, Mapping):
        raise SearchSpaceError("Staging offset requires 'controller_params.destination_location'.")
    spawn_location = spawn_transform["location"]

    anchor = controller_params.get("route_anchor_location")
    if not isinstance(anchor, Mapping):
        anchor = {
            "x": (float(spawn_location["x"]) + float(destination["x"])) / 2.0,
            "y": (float(spawn_location["y"]) + float(destination["y"])) / 2.0,
            "z": (float(spawn_location.get("z", 0.8)) + float(destination.get("z", 0.8))) / 2.0,
        }
    anchor_xy = {"x": float(anchor["x"]), "y": float(anchor["y"]), "z": float(anchor.get("z", 0.0))}

    rotation = spawn_transform.get("rotation") if isinstance(spawn_transform, Mapping) else None
    yaw = math.radians(float(rotation.get("yaw", 0.0))) if isinstance(rotation, Mapping) else 0.0

    delta_x = float(spawn_location["x"]) - anchor_xy["x"]
    delta_y = float(spawn_location["y"]) - anchor_xy["y"]
    norm = math.hypot(delta_x, delta_y)
    if norm <= 1e-6:
        right = (-math.sin(yaw), math.cos(yaw))
        norm = 0.0
    else:
        right = (delta_x / norm, delta_y / norm)

    forward = (math.cos(yaw), math.sin(yaw))
    projection = forward[0] * right[0] + forward[1] * right[1]
    forward = (forward[0] - projection * right[0], forward[1] - projection * right[1])
    forward_norm = math.hypot(forward[0], forward[1])
    if forward_norm <= 1e-6:
        forward = (right[1], -right[0])
    else:
        forward = (forward[0] / forward_norm, forward[1] / forward_norm)
    return anchor_xy, right, forward, norm


def _apply_staging_offsets(
    payload: dict[str, Any],
    lateral: float | None,
    longitudinal: float | None,
    *,
    lateral_dimension: SearchDimension | None = None,
    longitudinal_dimension: SearchDimension | None = None,
) -> None:
    """Bake staging offsets into the spawn and destination about the route anchor.

    ``spawn = anchor + right * lateral + forward * longitudinal`` and
    ``destination = anchor - right * lateral + forward * longitudinal``, so the
    crossing axis keeps its length while the pair slides along the route. Axes
    not searched by the active space keep the spec-baked base value (``lateral``
    ``None`` reuses the existing spawn-anchor distance; ``longitudinal`` ``None``
    means zero along-route displacement).
    """
    controller_params = payload.get("controller_params")
    if not isinstance(controller_params, dict):
        raise SearchSpaceError("Scenario spec has no 'controller_params' mapping.")
    anchor, right, forward, base_lateral = _staging_geometry(controller_params)
    lateral_value = base_lateral if lateral is None else float(lateral)
    longitudinal_value = 0.0 if longitudinal is None else float(longitudinal)
    spawn_transform = controller_params["spawn_transform"]
    destination = controller_params["destination_location"]
    spawn_z = float(spawn_transform["location"].get("z", 0.8))
    destination_z = float(destination.get("z", 0.8))
    along_x = forward[0] * longitudinal_value
    along_y = forward[1] * longitudinal_value
    spawn_transform["location"] = {
        "x": anchor["x"] + right[0] * lateral_value + along_x,
        "y": anchor["y"] + right[1] * lateral_value + along_y,
        "z": spawn_z,
    }
    controller_params["destination_location"] = {
        "x": anchor["x"] - right[0] * lateral_value + along_x,
        "y": anchor["y"] - right[1] * lateral_value + along_y,
        "z": destination_z,
    }
    if lateral_dimension is not None:
        controller_params[LATERAL_OFFSET_ANNOTATION] = lateral_dimension.coerce(lateral_value)
    if longitudinal_dimension is not None:
        controller_params[LONGITUDINAL_OFFSET_ANNOTATION] = longitudinal_dimension.coerce(longitudinal_value)


def _read_lateral_offset(payload: Mapping[str, Any], dimension: SearchDimension) -> int | float | None:
    controller_params = payload.get("controller_params")
    if not isinstance(controller_params, Mapping):
        return None
    annotation = controller_params.get(LATERAL_OFFSET_ANNOTATION)
    if annotation is not None:
        try:
            return dimension.coerce(annotation)
        except SearchSpaceError:
            return None
    try:
        _, _, _, norm = _staging_geometry(controller_params)
    except (KeyError, TypeError, SearchSpaceError, ValueError):
        return None
    if norm <= 1e-6:
        return None
    return dimension.coerce(norm)


def _read_longitudinal_offset(payload: Mapping[str, Any], dimension: SearchDimension) -> int | float | None:
    controller_params = payload.get("controller_params")
    if not isinstance(controller_params, Mapping):
        return None
    annotation = controller_params.get(LONGITUDINAL_OFFSET_ANNOTATION)
    if annotation is not None:
        try:
            return dimension.coerce(annotation)
        except SearchSpaceError:
            return None
    try:
        anchor, _, forward, _ = _staging_geometry(controller_params)
        spawn_location = controller_params["spawn_transform"]["location"]
    except (KeyError, TypeError, SearchSpaceError, ValueError):
        return None
    delta_x = float(spawn_location["x"]) - anchor["x"]
    delta_y = float(spawn_location["y"]) - anchor["y"]
    return dimension.coerce(delta_x * forward[0] + delta_y * forward[1])


class SearchSpace:
    """An ordered set of :class:`SearchDimension` objects plus sampling helpers."""

    def __init__(self, dimensions: Sequence[SearchDimension], *, name: str = "campaign", seed: int | None = None) -> None:
        self._dimensions = tuple(dimensions)
        if not self._dimensions:
            raise SearchSpaceError("A search space must have at least one dimension.")
        names = [dimension.name for dimension in self._dimensions]
        if len(set(names)) != len(names):
            raise SearchSpaceError(f"Search dimension names must be unique; got {names}.")
        self.name = str(name)
        self.seed = seed
        self._rng = random.Random(seed)

    def dimensions(self) -> tuple[SearchDimension, ...]:
        return self._dimensions

    def names(self) -> tuple[str, ...]:
        return tuple(dimension.name for dimension in self._dimensions)

    def dimension(self, name: str) -> SearchDimension:
        for dimension in self._dimensions:
            if dimension.name == name:
                return dimension
        raise SearchSpaceError(f"Unknown search dimension '{name}'.")

    def __len__(self) -> int:
        return len(self._dimensions)

    def __contains__(self, name: object) -> bool:
        return any(dimension.name == name for dimension in self._dimensions)

    def signature(self) -> str:
        payload = [dimension.describe() for dimension in self._dimensions]
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "seed": self.seed, "dimensions": [d.describe() for d in self._dimensions]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchSpace":
        dimensions = [
            SearchDimension(
                name=str(item["name"]),
                spec_path=str(item["spec_path"]),
                low=float(item["low"]),
                high=float(item["high"]),
                kind=str(item.get("kind", "float")),
                step=float(item["step"]) if item.get("step") is not None else None,
                decimals=int(item.get("decimals", 3)),
                sigma=float(item["sigma"]) if item.get("sigma") is not None else None,
                default=float(item["default"]) if item.get("default") is not None else None,
                runtime=str(item.get("runtime", "live")),
                description=str(item.get("description", "")),
            )
            for item in payload["dimensions"]
        ]
        seed = payload.get("seed")
        return cls(dimensions, name=str(payload.get("name", "campaign")), seed=int(seed) if seed is not None else None)

    def sample(self, rng: random.Random | None = None) -> SearchCandidate:
        active_rng = self._rng if rng is None else rng
        return SearchCandidate(
            {dimension.name: dimension.sample_value(active_rng) for dimension in self._dimensions}
        )

    def sample_many(self, count: int, rng: random.Random | None = None) -> list[SearchCandidate]:
        if count < 0:
            raise SearchSpaceError("Sample count must be non-negative.")
        active_rng = self._rng if rng is None else rng
        return [self.sample(active_rng) for _ in range(count)]

    def validate(self, values: Mapping[str, Any]) -> SearchCandidate:
        if not isinstance(values, Mapping):
            raise SearchSpaceError(f"Expected a mapping of dimension values, got {type(values).__name__}.")
        expected = set(self.names())
        missing = sorted(expected - set(values.keys()))
        extra = sorted(set(values.keys()) - expected)
        if missing:
            raise SearchSpaceError(f"Candidate is missing dimensions: {', '.join(missing)}.")
        if extra:
            raise SearchSpaceError(f"Candidate has unknown dimensions: {', '.join(extra)}.")
        coerced: dict[str, int | float] = {}
        for dimension in self._dimensions:
            dimension.validate(values[dimension.name])
            coerced[dimension.name] = dimension.coerce(values[dimension.name])
        return SearchCandidate(coerced)

    def clamp(self, values: Mapping[str, Any]) -> SearchCandidate:
        if not isinstance(values, Mapping):
            raise SearchSpaceError(f"Expected a mapping of dimension values, got {type(values).__name__}.")
        expected = set(self.names())
        extra = sorted(set(values.keys()) - expected)
        if extra:
            raise SearchSpaceError(f"Candidate has unknown dimensions: {', '.join(extra)}.")
        coerced: dict[str, int | float] = {}
        for dimension in self._dimensions:
            raw = values.get(dimension.name, dimension.missing_value())
            try:
                numeric = float(raw)
            except (TypeError, ValueError) as exc:
                raise SearchSpaceError(f"Dimension '{dimension.name}' expects a number, got {raw!r}.") from exc
            if not math.isfinite(numeric):
                numeric = float(dimension.missing_value())
            coerced[dimension.name] = dimension.coerce(min(max(numeric, dimension.low), dimension.high))
        return SearchCandidate(coerced)

    def key(self, candidate: SearchCandidate | Mapping[str, Any]) -> tuple[int | float, ...]:
        values = candidate.values if isinstance(candidate, SearchCandidate) else candidate
        return tuple(self.dimension(name).coerce(values[name]) for name in self.names())

    def apply_to_payload(self, payload: Mapping[str, Any], candidate: SearchCandidate | Mapping[str, Any]) -> dict[str, Any]:
        validated = self.validate(candidate.values if isinstance(candidate, SearchCandidate) else candidate)
        result = copy.deepcopy(dict(payload))
        staging_dimensions = [dimension for dimension in self._dimensions if dimension.kind in STAGING_OFFSET_KINDS]
        if staging_dimensions:
            lateral_dimension = next(
                (dimension for dimension in staging_dimensions if dimension.kind == "lateral_offset_m"), None
            )
            longitudinal_dimension = next(
                (dimension for dimension in staging_dimensions if dimension.kind == "longitudinal_offset_m"), None
            )
            lateral = float(validated.values[lateral_dimension.name]) if lateral_dimension is not None else None
            longitudinal = (
                float(validated.values[longitudinal_dimension.name]) if longitudinal_dimension is not None else None
            )
            _apply_staging_offsets(
                result,
                lateral,
                longitudinal,
                lateral_dimension=lateral_dimension,
                longitudinal_dimension=longitudinal_dimension,
            )
        for dimension in self._dimensions:
            if dimension.kind in STAGING_OFFSET_KINDS:
                continue
            _nested_set(result, dimension.spec_path, validated.values[dimension.name])
        return result

    def apply(self, spec: ScenarioSpec, candidate: SearchCandidate | Mapping[str, Any]) -> ScenarioSpec:
        return ScenarioSpec.from_dict(self.apply_to_payload(spec.to_dict(), candidate))

    def from_payload(self, payload: Mapping[str, Any], *, fill_missing: bool = True) -> SearchCandidate | None:
        values: dict[str, int | float] = {}
        for dimension in self._dimensions:
            raw: Any
            if dimension.kind == "lateral_offset_m":
                raw = _read_lateral_offset(payload, dimension)
            elif dimension.kind == "longitudinal_offset_m":
                raw = _read_longitudinal_offset(payload, dimension)
            else:
                raw = _nested_get(payload, dimension.spec_path)
            if raw is None:
                if not fill_missing:
                    return None
                values[dimension.name] = dimension.missing_value()
                continue
            values[dimension.name] = dimension.coerce(raw)
        return SearchCandidate(values)


def make_campaign_space(*, name: str = "campaign-v2", seed: int | None = None) -> SearchSpace:
    """The pre-registered campaign space (five dimensions after the staging extension)."""
    dimensions = (
        SearchDimension(
            name="trigger_radius_m",
            spec_path="controller_params.trigger_radius_m",
            low=5.0,
            high=35.0,
            kind="float",
            decimals=2,
            sigma=3.0,
            default=20.0,
            runtime="live",
            description="Pilot range kept for continuity; walker releases when the ego enters this radius.",
        ),
        SearchDimension(
            name="subject_speed_mps",
            spec_path="controller_params.speed",
            low=0.8,
            high=4.0,
            kind="float",
            decimals=2,
            sigma=0.4,
            default=1.8,
            runtime="live",
            description="Walker release speed in m/s (0.8 stroll to 4.0 run).",
        ),
        SearchDimension(
            name="staging_lateral_offset_m",
            spec_path="controller_params.spawn_transform.location",
            low=1.0,
            high=14.0,
            kind="lateral_offset_m",
            decimals=2,
            sigma=1.5,
            default=7.0,
            runtime="spec-baked",
            description="Lane-perpendicular staging distance from the route anchor to spawn and destination.",
        ),
        SearchDimension(
            name="staging_longitudinal_offset_m",
            spec_path="controller_params.spawn_transform.location",
            low=-10.0,
            high=30.0,
            kind="longitudinal_offset_m",
            decimals=2,
            sigma=4.0,
            default=0.0,
            runtime="spec-baked",
            description=(
                "Along-route staging displacement from the route anchor in m (positive = ahead of the ego); "
                "-10 m stays inside the 12 m generator trigger lead, +30 m covers the downstream junction approach."
            ),
        ),
        SearchDimension(
            name="trigger_tick",
            spec_path="controller_params.trigger_tick",
            low=10,
            high=400,
            kind="int",
            step=10,
            sigma=40.0,
            default=400,
            runtime="live",
            description=(
                "1-based logical tick deadline for adversary release (10 Hz, so seconds = tick / 10); "
                "the distance trigger still fires first when the ego arrives earlier."
            ),
        ),
    )
    return SearchSpace(dimensions, name=name, seed=seed)


def make_legacy_space(
    *,
    spec_path: str = "controller_params.trigger_radius_m",
    low: float = 5.0,
    high: float = 35.0,
    kind: str = "float",
    step: float | None = None,
    decimals: int = 2,
    sigma: float = 4.0,
    seed: int | None = None,
) -> SearchSpace:
    """The pilot's one-dimensional ``trigger_radius_m`` space."""
    dimension = SearchDimension(
        name="trigger_radius_m",
        spec_path=str(spec_path),
        low=float(low),
        high=float(high),
        kind=str(kind),
        step=float(step) if step is not None else None,
        decimals=int(decimals),
        sigma=float(sigma),
        default=None,
        runtime="live",
        description="Legacy pilot dimension; reproduces archived candidate values for a fixed seed.",
    )
    return SearchSpace((dimension,), name="legacy-trigger-radius", seed=seed)
