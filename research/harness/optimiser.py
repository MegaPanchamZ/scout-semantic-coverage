from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import random
import subprocess
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.models import ScenarioSpec  # noqa: E402
from research.harness.scenarios import load_scenario_spec  # noqa: E402
from research.harness.dota_to_scenario import DEFAULT_METADATA_PATH, build_dota_seeded_scenario  # noqa: E402
from research.harness.search_space import (  # noqa: E402
    SearchCandidate,
    SearchSpace,
    make_campaign_space,
    make_legacy_space,
)


@dataclass(slots=True)
class CandidatePlan:
    generation: int
    index: int
    values: dict[str, int | float]
    parent_id: str | None
    source: str

    @property
    def candidate_id(self) -> str:
        return f"g{self.generation:02d}-c{self.index:02d}"

    @property
    def value(self) -> float | None:
        raw = self.values.get("trigger_radius_m")
        return None if raw is None else float(raw)


def _workspace_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return (WORKSPACE_ROOT / path).resolve()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_python_executable() -> Path:
    venv_python = WORKSPACE_ROOT / "research/.venv/Scripts/python.exe"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def _nested_get(mapping: dict[str, object], dotted_path: str) -> object:
    cursor: object = mapping
    for key in dotted_path.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            raise KeyError(f"Mutation path '{dotted_path}' does not exist.")
        cursor = cursor[key]
    return cursor


def _coerce_numeric_value(template_value: object, raw_value: float, decimals: int) -> int | float:
    if isinstance(template_value, bool) or not isinstance(template_value, (int, float)):
        raise TypeError("The mutation target must be numeric.")
    if isinstance(template_value, int):
        return int(round(raw_value))
    return round(float(raw_value), decimals)


def _plan_values(parent: dict[str, object]) -> dict[str, float]:
    candidate = parent.get("candidate")
    if isinstance(candidate, dict):
        return {str(key): float(value) for key, value in candidate.items()}
    mutation_value = parent.get("mutation_value")
    if mutation_value is not None:
        return {"trigger_radius_m": float(mutation_value)}
    raise KeyError("Elite evaluation has no candidate values.")


def _resolve_search_space(args: argparse.Namespace, template_value: object) -> SearchSpace:
    if args.search_space == "campaign":
        return make_campaign_space(seed=args.seed)
    if args.mutation_min is None or args.mutation_max is None:
        raise ValueError("Legacy optimiser runs require --mutation-min and --mutation-max.")
    kind = "int" if isinstance(template_value, int) and not isinstance(template_value, bool) else "float"
    return make_legacy_space(
        spec_path=args.mutation_path,
        low=args.mutation_min,
        high=args.mutation_max,
        kind=kind,
        decimals=args.mutation_decimals,
        sigma=args.mutation_sigma,
        seed=args.seed,
    )


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a first-pass evolutionary falsification loop over ScenarioSpec JSON files.")
    parser.add_argument("--base-scenario-spec", type=Path, default=None)
    parser.add_argument("--dota-class", default=None)
    parser.add_argument("--dota-clip-id", default=None)
    parser.add_argument("--dota-metadata-json", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--carla-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--search-space", choices=("legacy", "campaign"), default="legacy")
    parser.add_argument("--mutation-path", default="controller_params.trigger_radius_m")
    parser.add_argument("--mutation-min", type=float, default=None)
    parser.add_argument("--mutation-max", type=float, default=None)
    parser.add_argument("--mutation-sigma", type=float, default=2.0)
    parser.add_argument("--mutation-decimals", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--fitness-metric", default="coverage_lsa_max", choices=["coverage_lsa_max", "coverage_lsa_mean", "coverage_kmnc", "collision_count", "ticks_executed"])
    parser.add_argument("--collision-bonus", type=float, default=1000000.0)
    parser.add_argument("--output-dir", type=Path, default=Path("research/logs/falsification"))
    parser.add_argument("--label", default="optimiser")
    parser.add_argument("--python-executable", type=Path, default=None)
    parser.add_argument("--port", type=int, default=2500)
    parser.add_argument("--agent-kind", default="pcla")
    parser.add_argument("--behavior", default="normal", choices=["cautious", "normal", "aggressive"])
    parser.add_argument("--target-speed-kph", type=float, default=30.0)
    parser.add_argument("--agent-repo-path", type=Path, default=None)
    parser.add_argument("--agent-module", default=None)
    parser.add_argument("--pcla-agent", default=None)
    parser.add_argument("--agent-class", default=None)
    parser.add_argument("--agent-checkpoint", type=Path, default=None)
    parser.add_argument("--agent-config", type=Path, default=None)
    parser.add_argument("--coverage-profile", type=Path, default=None)
    parser.add_argument("--coverage-layer", default=None)
    parser.add_argument("--coverage-k-sections", type=int, default=1000)
    parser.add_argument("--max-ticks", type=int, default=200)
    parser.add_argument("--telemetry-sample-every", type=int, default=1)
    parser.add_argument("--telemetry-deviation-threshold", type=float, default=2.0)
    parser.add_argument("--semantic-stream-every", type=int, default=1)
    parser.add_argument("--semantic-capture-tick", action="append", type=int, default=[])
    parser.add_argument("--semantic-brake-threshold-ticks", type=int, default=10)
    parser.add_argument("--semantic-heading-threshold-deg", type=float, default=20.0)
    parser.add_argument("--startup-hold-ticks", type=int, default=0)
    parser.add_argument("--resolution-x", type=int, default=800)
    parser.add_argument("--resolution-y", type=int, default=600)
    parser.add_argument("--quality-level", default="Low")
    parser.add_argument("--boot-timeout-seconds", type=float, default=60.0)
    return parser


def _build_initial_population(
    *,
    space: SearchSpace,
    rng: random.Random,
    base_candidate: SearchCandidate | None,
    include_base: bool,
    population_size: int,
) -> list[CandidatePlan]:
    plans: list[CandidatePlan] = []
    seen: set[tuple] = set()
    next_index = 0
    if include_base and base_candidate is not None:
        clamped = space.clamp(base_candidate.values)
        plans.append(
            CandidatePlan(generation=0, index=next_index, values=clamped.to_dict(), parent_id=None, source="base")
        )
        seen.add(space.key(clamped))
        next_index += 1

    while len(plans) < population_size:
        candidate = space.sample(rng)
        key = space.key(candidate)
        if key in seen:
            continue
        plans.append(
            CandidatePlan(generation=0, index=next_index, values=candidate.to_dict(), parent_id=None, source="uniform")
        )
        seen.add(key)
        next_index += 1
    return plans


def _build_next_generation(
    *,
    space: SearchSpace,
    rng: random.Random,
    generation: int,
    population_size: int,
    elites: list[dict[str, object]],
) -> list[CandidatePlan]:
    plans: list[CandidatePlan] = []
    seen: set[tuple] = set()
    seed_parents = elites[: max(1, min(len(elites), population_size))]

    for index, parent in enumerate(seed_parents):
        candidate = space.clamp(_plan_values(parent))
        plans.append(
            CandidatePlan(
                generation=generation,
                index=index,
                values=candidate.to_dict(),
                parent_id=str(parent["candidate_id"]),
                source="elite-copy",
            )
        )
        seen.add(space.key(candidate))

    next_index = len(plans)
    attempts = 0
    while len(plans) < population_size:
        attempts += 1
        parent = rng.choice(seed_parents)
        parent_values = _plan_values(parent)
        mutated = {
            dimension.name: float(parent_values.get(dimension.name, dimension.missing_value()))
            + rng.gauss(0.0, dimension.sigma_or_default)
            for dimension in space.dimensions()
        }
        candidate = space.clamp(mutated)
        key = space.key(candidate)
        if key in seen:
            if attempts > population_size * 20:
                candidate = space.sample(rng)
                key = space.key(candidate)
                if key in seen:
                    continue
            else:
                continue
        plans.append(
            CandidatePlan(
                generation=generation,
                index=next_index,
                values=candidate.to_dict(),
                parent_id=str(parent["candidate_id"]),
                source="gaussian-mutation",
            )
        )
        seen.add(key)
        next_index += 1
    return plans


def _build_diagnostics_command(args: argparse.Namespace, scenario_spec_path: Path, candidate_dir: Path, label: str) -> list[str]:
    command = [
        str(args.python_executable),
        str(WORKSPACE_ROOT / "research/harness/run_inter_session_diagnostics.py"),
        "--scenario-spec",
        str(scenario_spec_path),
        "--runs",
        "1",
        "--carla-root",
        str(args.carla_root),
        "--port",
        str(args.port),
        "--output-dir",
        str(candidate_dir / "diagnostics"),
        "--run-output-dir",
        str(candidate_dir / "runs"),
        "--telemetry-output",
        str(candidate_dir / "telemetry"),
        "--coverage-trace-output",
        str(candidate_dir / "coverage"),
        "--semantic-output",
        str(candidate_dir / "semantic"),
        "--semantic-anomaly-output",
        str(candidate_dir / "semantic_dumps"),
        "--agent-kind",
        args.agent_kind,
        "--behavior",
        args.behavior,
        "--target-speed-kph",
        str(args.target_speed_kph),
        "--max-ticks",
        str(args.max_ticks),
        "--telemetry-sample-every",
        str(args.telemetry_sample_every),
        "--telemetry-deviation-threshold",
        str(args.telemetry_deviation_threshold),
        "--semantic-stream-every",
        str(args.semantic_stream_every),
        "--semantic-brake-threshold-ticks",
        str(args.semantic_brake_threshold_ticks),
        "--semantic-heading-threshold-deg",
        str(args.semantic_heading_threshold_deg),
        "--startup-hold-ticks",
        str(args.startup_hold_ticks),
        "--label",
        label,
        "--resolution-x",
        str(args.resolution_x),
        "--resolution-y",
        str(args.resolution_y),
        "--quality-level",
        str(args.quality_level),
        "--boot-timeout-seconds",
        str(args.boot_timeout_seconds),
    ]
    if args.agent_repo_path is not None:
        command.extend(["--agent-repo-path", str(args.agent_repo_path)])
    if args.agent_module is not None:
        command.extend(["--agent-module", str(args.agent_module)])
    if args.pcla_agent is not None:
        command.extend(["--pcla-agent", str(args.pcla_agent)])
    if args.agent_class is not None:
        command.extend(["--agent-class", str(args.agent_class)])
    if args.agent_checkpoint is not None:
        command.extend(["--agent-checkpoint", str(args.agent_checkpoint)])
    if args.agent_config is not None:
        command.extend(["--agent-config", str(args.agent_config)])
    if args.coverage_profile is not None:
        command.extend(["--coverage-profile", str(args.coverage_profile)])
    if args.coverage_layer is not None:
        command.extend(["--coverage-layer", str(args.coverage_layer)])
    if args.coverage_k_sections is not None:
        command.extend(["--coverage-k-sections", str(args.coverage_k_sections)])
    for tick in sorted(set(args.semantic_capture_tick)):
        command.extend(["--semantic-capture-tick", str(tick)])
    return command


def _select_run_json(run_output_dir: Path) -> Path | None:
    candidates = sorted(run_output_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def _extract_metric(result: dict[str, object], metric_name: str) -> float | None:
    raw_value = result.get(metric_name)
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _evaluate_candidate(
    *,
    args: argparse.Namespace,
    space: SearchSpace,
    plan: CandidatePlan,
    base_scenario: ScenarioSpec,
    base_payload: dict[str, object],
    session_root: Path,
) -> dict[str, object]:
    candidate_dir = session_root / "evaluations" / plan.candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    scenario_payload = space.apply_to_payload(base_payload, plan.values)
    scenario_id = f"{base_scenario.scenario_id}-{plan.candidate_id}"
    if space.name == "legacy-trigger-radius":
        parameter_text = f"{args.mutation_path}={plan.value}"
    else:
        parameter_text = f"{space.name} values {json.dumps(plan.values, sort_keys=True)}"
    scenario_payload["scenario_id"] = scenario_id
    scenario_payload["description"] = (
        f"{base_scenario.description} Optimiser candidate {plan.candidate_id} with {parameter_text}."
    )

    candidate_spec_path = candidate_dir / f"{scenario_id}.json"
    _write_json(candidate_spec_path, scenario_payload)

    label = f"{args.label}-{plan.candidate_id}"
    command = _build_diagnostics_command(args, candidate_spec_path, candidate_dir, label)
    completed = subprocess.run(
        command,
        cwd=WORKSPACE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    diagnostics_path = candidate_dir / "diagnostics" / f"{scenario_id}-{args.agent_kind}-{label}-diagnostics.json"
    diagnostics_summary = _load_json(diagnostics_path) if diagnostics_path.exists() else None
    result = diagnostics_summary["results"][0] if diagnostics_summary is not None and diagnostics_summary.get("results") else {}
    run_json_path = _select_run_json(candidate_dir / "runs")
    run_payload = _load_json(run_json_path) if run_json_path is not None else None

    collision_count = int(result.get("collision_count") or 0)
    terminated_by_collision = bool(run_payload.get("terminated_by_collision")) if isinstance(run_payload, dict) else False
    semantic_covered_predicates = []
    semantic_covered_signatures = []
    semantic_fulfilled_obligations = []
    semantic_missing_obligations = []
    semantic_relation_hits = []
    semantic_next_test_requests = []
    if isinstance(run_payload, dict):
        metadata = run_payload.get("metadata", {})
        if isinstance(metadata, dict):
            semantic = metadata.get("semantic", {})
            if isinstance(semantic, dict):
                semantic_covered_predicates = list(semantic.get("covered_predicates", []) or [])
                semantic_covered_signatures = list(semantic.get("covered_signatures", []) or [])
                obligation_credit = semantic.get("obligation_credit", {}) or {}
                if isinstance(obligation_credit, dict):
                    semantic_fulfilled_obligations = list(obligation_credit.get("fulfilled_obligations", []) or [])
                    semantic_missing_obligations = list(obligation_credit.get("missing_obligations", []) or [])
                    semantic_relation_hits = list(obligation_credit.get("relation_obligation_hits", []) or [])
                    semantic_next_test_requests = list(obligation_credit.get("next_test_requests", []) or [])
    semantic_colliding = "colliding" in semantic_covered_predicates
    critical_edge_case = collision_count > 0 or terminated_by_collision or semantic_colliding

    metric_value = _extract_metric(result, args.fitness_metric)
    score = float("-inf") if metric_value is None else metric_value
    if critical_edge_case and not math.isinf(score):
        score += args.collision_bonus
    elif critical_edge_case and math.isinf(score):
        score = args.collision_bonus

    evaluation_summary = {
        "candidate_id": plan.candidate_id,
        "generation": plan.generation,
        "index": plan.index,
        "parent_id": plan.parent_id,
        "source": plan.source,
        "mutation_path": args.mutation_path,
        "mutation_value": plan.value,
        "candidate": plan.values,
        "parameter_values": plan.values,
        "scenario_id": scenario_id,
        "scenario_spec_path": str(candidate_spec_path),
        "diagnostics_summary_path": str(diagnostics_path) if diagnostics_path.exists() else None,
        "run_json_path": str(run_json_path) if run_json_path is not None else None,
        "fitness_metric": args.fitness_metric,
        "fitness_value": metric_value,
        "score": score,
        "critical_edge_case": critical_edge_case,
        "collision_count": collision_count,
        "terminated_by_collision": terminated_by_collision,
        "semantic_colliding": semantic_colliding,
        "semantic_covered_predicates": semantic_covered_predicates,
        "semantic_covered_signatures": semantic_covered_signatures,
        "semantic_fulfilled_obligations": semantic_fulfilled_obligations,
        "semantic_missing_obligations": semantic_missing_obligations,
        "semantic_relation_obligation_hits": semantic_relation_hits,
        "next_test_requests": semantic_next_test_requests,
        "follow_up_requested": bool(semantic_missing_obligations),
        "first_deviation_alert": result.get("first_deviation_alert"),
        "ticks_executed": result.get("ticks_executed"),
        "coverage_kmnc": result.get("coverage_kmnc"),
        "coverage_lsa_mean": result.get("coverage_lsa_mean"),
        "coverage_lsa_max": result.get("coverage_lsa_max"),
        "run_error": (
            result.get("run_error")
            if isinstance(result, dict) and result.get("run_error") is not None
            else (completed.stderr[-4000:] if completed.returncode != 0 and diagnostics_summary is None else None)
        ),
        "subprocess_returncode": completed.returncode,
        "subprocess_stdout_tail": completed.stdout[-4000:],
        "subprocess_stderr_tail": completed.stderr[-4000:],
    }
    _write_json(candidate_dir / "evaluation-summary.json", evaluation_summary)

    if critical_edge_case:
        critical_dir = session_root / "critical_edge_cases"
        critical_dir.mkdir(parents=True, exist_ok=True)
        _write_json(critical_dir / f"{scenario_id}.critical.json", scenario_payload)
        _write_json(critical_dir / f"{scenario_id}.report.json", evaluation_summary)

    return evaluation_summary


def main() -> None:
    args = build_parser().parse_args()
    args.base_scenario_spec = _workspace_path(args.base_scenario_spec) or args.base_scenario_spec
    args.dota_metadata_json = _workspace_path(args.dota_metadata_json) or args.dota_metadata_json
    args.carla_root = _workspace_path(args.carla_root) or args.carla_root
    args.output_dir = _workspace_path(args.output_dir) or args.output_dir
    args.python_executable = _workspace_path(args.python_executable) or args.python_executable or _default_python_executable()
    args.agent_repo_path = _workspace_path(args.agent_repo_path)
    args.agent_checkpoint = _workspace_path(args.agent_checkpoint)
    args.agent_config = _workspace_path(args.agent_config)
    args.coverage_profile = _workspace_path(args.coverage_profile)

    if not Path(args.python_executable).exists():
        raise FileNotFoundError(f"Python executable not found at {args.python_executable}")
    if args.base_scenario_spec is None and args.dota_class is None:
        raise ValueError("Provide either --base-scenario-spec or --dota-class.")
    if args.base_scenario_spec is not None and args.dota_class is not None:
        raise ValueError("Use either --base-scenario-spec or --dota-class, not both.")

    if args.search_space == "legacy" and (args.mutation_min is None or args.mutation_max is None):
        raise ValueError("Legacy optimiser runs require --mutation-min and --mutation-max.")
    if args.mutation_min is not None and args.mutation_max is not None and args.mutation_min > args.mutation_max:
        raise ValueError("--mutation-min must be less than or equal to --mutation-max.")
    if args.population_size < 1:
        raise ValueError("--population-size must be at least 1.")
    if args.generations < 1:
        raise ValueError("--generations must be at least 1.")
    if args.elite_count < 1:
        raise ValueError("--elite-count must be at least 1.")
    if args.agent_kind == "pcla" and args.pcla_agent is None:
        raise ValueError("PCLA optimiser runs require --pcla-agent.")
    if args.agent_kind != "behavior" and args.agent_repo_path is None and args.agent_kind != "pcla":
        raise ValueError(f"Agent kind '{args.agent_kind}' requires --agent-repo-path.")
    if args.agent_kind == "leaderboard-module" and args.agent_module is None:
        raise ValueError("Leaderboard-module optimiser runs require --agent-module.")
    if args.agent_kind == "pcla" and args.agent_repo_path is None:
        args.agent_repo_path = WORKSPACE_ROOT / "research/models/PCLA"

    scenario_source: dict[str, object] = {"mode": "scenario-spec"}
    if args.base_scenario_spec is not None:
        base_scenario = load_scenario_spec(args.base_scenario_spec)
        base_payload = base_scenario.to_dict()
        scenario_source["base_scenario_spec_path"] = str(args.base_scenario_spec)
    else:
        base_payload = build_dota_seeded_scenario(
            metadata_path=args.dota_metadata_json,
            dota_class=str(args.dota_class),
            clip_id=args.dota_clip_id,
            host=args.host,
            port=args.port,
            timeout_seconds=args.boot_timeout_seconds,
            town="Town01",
            weather_preset="ClearNoon",
            ego_spawn_index=0,
            goal_spawn_index=82,
            sampling_resolution=2.0,
            min_distance_from_start_m=20.0,
            min_distance_to_goal_m=20.0,
            trigger_lead_distance_m=12.0,
            trigger_radius_m=8.0,
            walker_speed=1.8,
            lateral_offset_multiplier=1.75,
            pre_brake_throttle=0.3,
            max_ticks=max(int(args.max_ticks), 1),
        )
        base_scenario = ScenarioSpec.from_dict(base_payload)
        scenario_source = {
            "mode": "dota-class",
            "dota_class": str(args.dota_class),
            "dota_clip_id": str(base_payload.get("generator_metadata", {}).get("source", {}).get("clip_id") or args.dota_clip_id or ""),
            "dota_metadata_json": str(args.dota_metadata_json),
            "generator_metadata": base_payload.get("generator_metadata"),
        }

    target_template_value = _nested_get(base_payload, args.mutation_path)
    if isinstance(target_template_value, bool) or not isinstance(target_template_value, (int, float)):
        raise TypeError(f"Mutation path '{args.mutation_path}' must point to a numeric field.")
    base_value = float(target_template_value)

    space = _resolve_search_space(args, target_template_value)
    base_candidate = space.from_payload(base_payload, fill_missing=args.search_space != "legacy")
    if base_candidate is None:
        raise TypeError(f"Mutation path '{args.mutation_path}' does not resolve to a numeric field.")

    session_id = f"{base_scenario.scenario_id}-{args.label}-{_utc_timestamp()}"
    session_root = args.output_dir / session_id
    session_root.mkdir(parents=True, exist_ok=True)
    _write_json(session_root / "base-scenario.json", base_payload)

    rng = random.Random(args.seed)
    all_results: list[dict[str, object]] = []
    if args.include_base:
        initial_population_size = max(1, args.population_size)
    else:
        initial_population_size = args.population_size

    generation_plans = _build_initial_population(
        space=space,
        rng=rng,
        base_candidate=base_candidate,
        include_base=args.include_base,
        population_size=initial_population_size,
    )

    for generation in range(args.generations):
        if generation > 0:
            ranked = sorted(all_results, key=lambda item: float(item["score"]), reverse=True)
            generation_plans = _build_next_generation(
                space=space,
                rng=rng,
                generation=generation,
                population_size=args.population_size,
                elites=ranked[: max(1, min(args.elite_count, len(ranked)))],
            )

        for plan in generation_plans:
            evaluation = _evaluate_candidate(
                args=args,
                space=space,
                plan=plan,
                base_scenario=base_scenario,
                base_payload=base_payload,
                session_root=session_root,
            )
            all_results.append(evaluation)

            progress_payload = {
                "session_id": session_id,
                "completed_evaluations": len(all_results),
                "latest_candidate_id": evaluation["candidate_id"],
                "latest_score": evaluation["score"],
                "critical_edge_case_count": sum(1 for item in all_results if bool(item["critical_edge_case"])),
                "best_candidate_id": max(all_results, key=lambda item: float(item["score"]))["candidate_id"],
            }
            _write_json(session_root / "progress.json", progress_payload)

    ranked_results = sorted(all_results, key=lambda item: float(item["score"]), reverse=True)
    critical_edge_cases = [item for item in ranked_results if bool(item["critical_edge_case"])]
    session_summary = {
        "session_id": session_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_scenario_spec_path": str(args.base_scenario_spec) if args.base_scenario_spec is not None else None,
        "base_scenario_id": base_scenario.scenario_id,
        "scenario_source": scenario_source,
        "mutation_path": args.mutation_path,
        "base_value": _coerce_numeric_value(target_template_value, base_value, args.mutation_decimals),
        "search_space": {
            "name": space.name,
            "signature": space.signature(),
            "dimensions": [dimension.describe() for dimension in space.dimensions()],
            "base_candidate": base_candidate.to_dict(),
        },
        "mutation_range": {
            "min": args.mutation_min,
            "max": args.mutation_max,
            "sigma": args.mutation_sigma,
            "decimals": args.mutation_decimals,
        },
        "search": {
            "population_size": args.population_size,
            "generations": args.generations,
            "elite_count": args.elite_count,
            "seed": args.seed,
            "include_base": args.include_base,
            "fitness_metric": args.fitness_metric,
            "collision_bonus": args.collision_bonus,
        },
        "agent": {
            "python_executable": str(args.python_executable),
            "agent_kind": args.agent_kind,
            "pcla_agent": args.pcla_agent,
            "agent_module": args.agent_module,
            "behavior": args.behavior,
            "target_speed_kph": args.target_speed_kph,
            "coverage_profile": str(args.coverage_profile) if args.coverage_profile is not None else None,
            "coverage_layer": args.coverage_layer,
        },
        "results": ranked_results,
        "best_candidate": ranked_results[0] if ranked_results else None,
        "critical_edge_case_count": len(critical_edge_cases),
        "critical_edge_cases": critical_edge_cases,
    }
    _write_json(session_root / "session-summary.json", session_summary)
    print(json.dumps(session_summary, indent=2))


if __name__ == "__main__":
    main()