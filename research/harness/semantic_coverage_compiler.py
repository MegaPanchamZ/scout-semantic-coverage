from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.backends.alpasim_backend import AlpaSimBackendAdapter  # noqa: E402
from research.harness.backends.carla_backend import CarlaBackendAdapter  # noqa: E402
from research.harness.models import ScenarioSpec  # noqa: E402


@dataclass(slots=True)
class CoverageAxis:
    key: str
    values: list[str]
    required: bool = True
    description: str = ""


@dataclass(slots=True)
class SemanticCoveragePlan:
    scenario_id: str
    description: str
    language_spec: str
    stsg_targets: dict[str, Any]
    required_obligations: list[str]
    covered_obligations: list[str]
    missing_obligations: list[str]
    next_test_requests: list[dict[str, Any]]
    variants: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile a semantic DSL into coverage-tracked scenario variants and world-model requests."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to the semantic DSL JSON file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the compiled plan JSON file.")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only emit variants that exercise at least one currently uncovered coverage obligation.",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Optional cap on the number of emitted variants after filtering.",
    )
    return parser


def _slugify(value: str) -> str:
    safe = []
    for char in value.lower():
        if char.isalnum():
            safe.append(char)
        else:
            safe.append("-")
    return "".join(safe).strip("-") or "variant"


def _coerce_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def _parse_axes(payload: dict[str, Any]) -> list[CoverageAxis]:
    axes: list[CoverageAxis] = []
    for item in payload.get("coverage_axes") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        values = [str(value) for value in item.get("values") or [] if str(value).strip()]
        if not key or not values:
            continue
        axes.append(
            CoverageAxis(
                key=key,
                values=values,
                required=bool(item.get("required", True)),
                description=str(item.get("description") or ""),
            )
        )
    return axes


def _count_actor_types(actors: list[dict[str, Any]]) -> tuple[int, int]:
    vehicle_count = 0
    walker_count = 0
    for actor in actors:
        actor_type = str(actor.get("type") or "").lower()
        if actor_type.startswith("vehicle") and actor.get("id") != "ego":
            vehicle_count += 1
        if actor_type.startswith("pedestrian") or actor_type.startswith("walker"):
            walker_count += 1
    return vehicle_count, walker_count


def _binding_to_obligation(key: str, value: str) -> str:
    return f"{key}={value}"


def _build_world_model_request(
    *,
    scenario_id: str,
    description: str,
    language_spec: str,
    bindings: dict[str, str],
    stsg_targets: dict[str, Any],
    world_model: dict[str, Any],
) -> dict[str, Any]:
    base_prompt = str(world_model.get("environment_prompt") or description).strip()
    sensor_modalities = _coerce_string_list(world_model.get("sensor_modalities"))
    control_mode = str(world_model.get("control_mode") or "structured_scene").strip()
    binding_text = ", ".join(f"{key}={value}" for key, value in sorted(bindings.items()))
    prompt_suffix = f" Coverage bindings: {binding_text}." if binding_text else ""
    return {
        "scenario_id": scenario_id,
        "prompt": f"{base_prompt}{prompt_suffix}".strip(),
        "language_spec": language_spec,
        "control_mode": control_mode,
        "sensor_modalities": sensor_modalities,
        "bindings": dict(sorted(bindings.items())),
        "stsg_targets": stsg_targets,
        "expected_outputs": {
            "multi_view_rgb": "rgb_multiview" in sensor_modalities,
            "lidar": "lidar" in sensor_modalities,
            "scene_graph": True,
        },
    }


def _build_backend_requests(
    *,
    scenario_id: str,
    description: str,
    language_spec: str,
    bindings: dict[str, str],
    stsg_targets: dict[str, Any],
    base_scenario: dict[str, Any],
    world_model_request: dict[str, Any],
) -> dict[str, Any]:
    adapters = [CarlaBackendAdapter(), AlpaSimBackendAdapter()]
    return {
        adapter.name: adapter.build_request(
            scenario_id=scenario_id,
            description=description,
            language_spec=language_spec,
            bindings=bindings,
            stsg_targets=stsg_targets,
            base_scenario=base_scenario,
            world_model_request=world_model_request,
        ).to_dict()
        for adapter in adapters
    }


def _build_next_test_requests(axes: list[CoverageAxis], covered_obligations: set[str]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for axis in axes:
        uncovered = [value for value in axis.values if _binding_to_obligation(axis.key, value) not in covered_obligations]
        if not uncovered:
            continue
        requests.append(
            {
                "axis": axis.key,
                "description": axis.description,
                "uncovered_values": uncovered,
                "reason": "STSG coverage obligation has not yet been exercised by a retained run.",
            }
        )
    return requests


def compile_semantic_plan(
    payload: dict[str, Any],
    *,
    only_missing: bool = False,
    max_variants: int | None = None,
) -> dict[str, Any]:
    scenario_id = str(payload.get("scenario_id") or "semantic-scenario").strip()
    description = str(payload.get("description") or "Semantic ADS test scenario.").strip()
    language_spec = str(payload.get("language_spec") or description).strip()
    base_scenario = dict(payload.get("base_scenario") or {})
    stsg_targets = dict(payload.get("stsg_targets") or {})
    world_model = dict(payload.get("world_model") or {})
    actors = [item for item in payload.get("actors") or [] if isinstance(item, dict)]
    axes = _parse_axes(payload)

    covered_obligations = {
        str(item)
        for item in ((payload.get("existing_coverage") or {}).get("covered_obligations") or [])
        if str(item).strip()
    }
    required_obligations = sorted(
        _binding_to_obligation(axis.key, value)
        for axis in axes
        if axis.required
        for value in axis.values
    )
    missing_obligations = sorted(item for item in required_obligations if item not in covered_obligations)

    vehicle_count, walker_count = _count_actor_types(actors)
    variant_bindings = [
        dict(zip([axis.key for axis in axes], combination))
        for combination in product(*[axis.values for axis in axes])
    ] or [{}]

    variants: list[dict[str, Any]] = []
    skipped_variants = 0
    for index, bindings in enumerate(variant_bindings, start=1):
        obligations = [_binding_to_obligation(key, value) for key, value in sorted(bindings.items())]
        uncovered = [item for item in obligations if item not in covered_obligations]
        if only_missing and obligations and not uncovered:
            skipped_variants += 1
            continue

        variant_suffix = "__".join(_slugify(f"{key}-{value}") for key, value in sorted(bindings.items())) or "base"
        compiled_id = f"{scenario_id}__{variant_suffix}"
        weather_preset = str(bindings.get("weather") or base_scenario.get("weather_preset") or "ClearNoon")
        controller_params = dict(base_scenario.get("controller_params") or {})
        controller_params["coverage_bindings"] = dict(sorted(bindings.items()))
        controller_params["coverage_obligations"] = obligations
        variant_stsg_targets = {
            "nodes": list(stsg_targets.get("nodes") or []),
            "attributes": list(stsg_targets.get("attributes") or []),
            "relations": list(stsg_targets.get("relations") or [])
        }
        maneuver = bindings.get("maneuver_type")
        if maneuver == "lane_change":
            if "ego" not in variant_stsg_targets["nodes"]: variant_stsg_targets["nodes"].append("ego")
            variant_stsg_targets["relations"].append("changes_lane(ego)")
        elif maneuver == "overtaking":
            if "ego" not in variant_stsg_targets["nodes"]: variant_stsg_targets["nodes"].append("ego")
            if "npc1" not in variant_stsg_targets["nodes"]: variant_stsg_targets["nodes"].append("npc1")
            variant_stsg_targets["relations"].append("overtakes(ego, npc1)")
        elif maneuver == "merging":
            if "ego" not in variant_stsg_targets["nodes"]: variant_stsg_targets["nodes"].append("ego")
            if "npc1" not in variant_stsg_targets["nodes"]: variant_stsg_targets["nodes"].append("npc1")
            variant_stsg_targets["relations"].append("merges_behind(npc1, ego)")
            
        variant_stsg_targets = {
            "nodes": list(set(variant_stsg_targets["nodes"])),
            "attributes": list(set(variant_stsg_targets["attributes"])),
            "relations": list(set(variant_stsg_targets["relations"]))
        }

        controller_params["stsg_targets"] = variant_stsg_targets

        scenario_spec = ScenarioSpec(
            scenario_id=compiled_id,
            town=str(base_scenario.get("town") or "Town01"),
            weather_preset=weather_preset,
            ego_spawn_index=int(base_scenario.get("ego_spawn_index", 0)),
            goal_spawn_index=int(base_scenario.get("goal_spawn_index", 0)),
            description=f"{description} [{variant_suffix}]",
            max_ticks=int(base_scenario.get("max_ticks", 500)),
            npc_vehicle_count=max(int(base_scenario.get("npc_vehicle_count", 0)), vehicle_count),
            walker_count=max(int(base_scenario.get("walker_count", 0)), walker_count),
            controller=str(base_scenario.get("controller") or "semantic_variant"),
            controller_params=controller_params,
        )
        world_model_request = _build_world_model_request(
            scenario_id=compiled_id,
            description=description,
            language_spec=language_spec,
            bindings=bindings,
            stsg_targets=variant_stsg_targets,
            world_model=world_model,
        )
        variants.append(
            {
                "variant_index": index,
                "variant_id": compiled_id,
                "bindings": dict(sorted(bindings.items())),
                "coverage_obligations": obligations,
                "uncovered_obligations": uncovered,
                "world_model_request": world_model_request,
                "backend_requests": _build_backend_requests(
                    scenario_id=compiled_id,
                    description=description,
                    language_spec=language_spec,
                    bindings=bindings,
                    stsg_targets=variant_stsg_targets,
                    base_scenario=base_scenario,
                    world_model_request=world_model_request,
                ),
                "scenario_spec": scenario_spec.to_dict(),
            }
        )
        if max_variants is not None and len(variants) >= max_variants:
            skipped_variants += max(len(variant_bindings) - index, 0)
            break

    plan = SemanticCoveragePlan(
        scenario_id=scenario_id,
        description=description,
        language_spec=language_spec,
        stsg_targets=stsg_targets,
        required_obligations=required_obligations,
        covered_obligations=sorted(covered_obligations),
        missing_obligations=missing_obligations,
        next_test_requests=_build_next_test_requests(axes, covered_obligations),
        variants=variants,
    )
    return {
        "schema_version": "2026-04-21",
        "compiler": "semantic_coverage_compiler",
        "coverage_axes": [asdict(axis) for axis in axes],
        "actor_count": len(actors),
        "emitted_variant_count": len(variants),
        "skipped_variant_count": skipped_variants,
        **plan.to_dict(),
    }


def main() -> None:
    args = build_parser().parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    compiled = compile_semantic_plan(payload, only_missing=args.only_missing, max_variants=args.max_variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compiled, indent=2), encoding="utf-8")
    print(json.dumps(compiled, indent=2))


if __name__ == "__main__":
    main()