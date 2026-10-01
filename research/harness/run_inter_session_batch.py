from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.scenarios import load_scenario_spec


def _workspace_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return (WORKSPACE_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hard-restart diagnostics as separate Python processes and aggregate the results.")
    parser.add_argument("--scenario", default="town01_clear_short")
    parser.add_argument("--scenario-spec", type=Path, default=None)
    parser.add_argument("--town", default=None)
    parser.add_argument("--weather-preset", default=None)
    parser.add_argument("--ego-spawn-index", type=int, default=None)
    parser.add_argument("--goal-spawn-index", type=int, default=None)
    parser.add_argument("--scenario-description", default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--carla-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=2400)
    parser.add_argument("--output-dir", type=Path, default=Path("research/logs/diagnostics"))
    parser.add_argument("--run-output-dir", type=Path, default=Path("research/logs/runs"))
    parser.add_argument("--telemetry-output", type=Path, default=Path("research/logs/telemetry"))
    parser.add_argument("--coverage-profile", type=Path, default=None)
    parser.add_argument("--coverage-layer", default=None)
    parser.add_argument("--coverage-k-sections", type=int, default=1000)
    parser.add_argument("--coverage-trace-output", type=Path, default=Path("research/logs/coverage"))
    parser.add_argument("--semantic-output", type=Path, default=Path("research/logs/semantic"))
    parser.add_argument("--semantic-anomaly-output", type=Path, default=Path("research/logs/semantic_dumps"))
    parser.add_argument("--semantic-stream-every", type=int, default=1)
    parser.add_argument("--semantic-capture-tick", action="append", type=int, default=[])
    parser.add_argument("--child-log-dir", type=Path, default=Path("research/logs/diagnostics"))
    parser.add_argument("--agent-kind", default="leaderboard-module")
    parser.add_argument("--behavior", default="normal")
    parser.add_argument("--target-speed-kph", type=float, default=30.0)
    parser.add_argument("--agent-repo-path", type=Path, default=None)
    parser.add_argument("--agent-module", default=None)
    parser.add_argument("--pcla-agent", default=None)
    parser.add_argument("--agent-class", default=None)
    parser.add_argument("--agent-checkpoint", type=Path, default=None)
    parser.add_argument("--agent-config", type=Path, default=None)
    parser.add_argument("--max-ticks", type=int, default=200)
    parser.add_argument("--telemetry-sample-every", type=int, default=1)
    parser.add_argument("--telemetry-deviation-threshold", type=float, default=2.0)
    parser.add_argument("--semantic-brake-threshold-ticks", type=int, default=10)
    parser.add_argument("--semantic-heading-threshold-deg", type=float, default=20.0)
    parser.add_argument("--startup-hold-ticks", type=int, default=0)
    parser.add_argument("--label", default="restart-batch")
    parser.add_argument("--resolution-x", type=int, default=800)
    parser.add_argument("--resolution-y", type=int, default=600)
    parser.add_argument("--quality-level", default=os.environ.get("SCOUT_CARLA_QUALITY", "Epic"))
    parser.add_argument("--boot-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    return parser


def _child_summary_path(output_dir: Path, scenario: str, agent_kind: str, child_label: str) -> Path:
    return output_dir / f"{scenario}-{agent_kind}-{child_label}-diagnostics.json"


def _cleanup_carla_processes() -> None:
    if os.name == "nt":
        for image_name in ("CarlaUE4.exe", "CarlaUE4-Win64-Shipping.exe"):
            subprocess.run(
                ["taskkill", "/F", "/T", "/IM", image_name],
                check=False,
                capture_output=True,
                text=True,
            )
        return

    subprocess.run(["pkill", "-f", "CarlaUE4"], check=False, capture_output=True, text=True)


def main() -> None:
    args = build_parser().parse_args()
    args.scenario_spec = _workspace_path(args.scenario_spec)
    args.carla_root = _workspace_path(args.carla_root) or args.carla_root
    args.output_dir = _workspace_path(args.output_dir) or args.output_dir
    args.run_output_dir = _workspace_path(args.run_output_dir) or args.run_output_dir
    args.telemetry_output = _workspace_path(args.telemetry_output) or args.telemetry_output
    args.coverage_profile = _workspace_path(args.coverage_profile)
    args.coverage_trace_output = _workspace_path(args.coverage_trace_output) or args.coverage_trace_output
    args.semantic_output = _workspace_path(args.semantic_output) or args.semantic_output
    args.semantic_anomaly_output = _workspace_path(args.semantic_anomaly_output) or args.semantic_anomaly_output
    args.child_log_dir = _workspace_path(args.child_log_dir) or args.child_log_dir
    args.agent_repo_path = _workspace_path(args.agent_repo_path)
    args.agent_checkpoint = _workspace_path(args.agent_checkpoint)
    args.agent_config = _workspace_path(args.agent_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.child_log_dir.mkdir(parents=True, exist_ok=True)

    if args.agent_kind == "pcla":
        if args.pcla_agent is None:
            raise ValueError("PCLA batch diagnostics require --pcla-agent.")
        if args.agent_repo_path is None:
            args.agent_repo_path = Path("research/models/PCLA")
    elif args.agent_kind != "behavior":
        if args.agent_repo_path is None:
            raise ValueError(f"Agent kind '{args.agent_kind}' requires --agent-repo-path.")
        if args.agent_kind == "leaderboard-module" and args.agent_module is None:
            raise ValueError("Leaderboard-module batch diagnostics require --agent-module.")

    if args.scenario_spec is not None:
        if any(value is not None for value in (args.town, args.weather_preset, args.ego_spawn_index, args.goal_spawn_index, args.scenario_description)):
            raise ValueError("--scenario-spec cannot be combined with scenario override flags.")
        scenario_output_key = load_scenario_spec(args.scenario_spec).scenario_id
    else:
        scenario_output_key = args.scenario

    python_exe = Path(sys.executable)
    runner_script = WORKSPACE_ROOT / "research" / "harness" / "run_inter_session_diagnostics.py"
    child_results: list[dict[str, object]] = []

    for run_index in range(args.runs):
        child_label = f"{args.label}-run{run_index + 1}"
        child_log_path = args.child_log_dir / f"{child_label}.log"
        _cleanup_carla_processes()
        time.sleep(args.settle_seconds)
        child_command = [
            str(python_exe),
            str(runner_script),
            "--runs", "1",
            "--carla-root", str(args.carla_root),
            "--port", str(args.port + run_index),
            "--output-dir", str(args.output_dir),
            "--run-output-dir", str(args.run_output_dir),
            "--telemetry-output", str(args.telemetry_output),
            "--coverage-trace-output", str(args.coverage_trace_output),
            "--semantic-output", str(args.semantic_output),
            "--semantic-anomaly-output", str(args.semantic_anomaly_output),
            "--semantic-stream-every", str(args.semantic_stream_every),
            "--agent-kind", args.agent_kind,
            "--behavior", args.behavior,
            "--target-speed-kph", str(args.target_speed_kph),
            "--agent-repo-path", str(args.agent_repo_path),
            "--max-ticks", str(args.max_ticks),
            "--telemetry-sample-every", str(args.telemetry_sample_every),
            "--telemetry-deviation-threshold", str(args.telemetry_deviation_threshold),
            "--coverage-k-sections", str(args.coverage_k_sections),
            "--semantic-brake-threshold-ticks", str(args.semantic_brake_threshold_ticks),
            "--semantic-heading-threshold-deg", str(args.semantic_heading_threshold_deg),
            "--startup-hold-ticks", str(args.startup_hold_ticks),
            "--label", child_label,
            "--resolution-x", str(args.resolution_x),
            "--resolution-y", str(args.resolution_y),
            "--quality-level", args.quality_level,
            "--boot-timeout-seconds", str(args.boot_timeout_seconds),
        ]
        if args.scenario_spec is not None:
            child_command.extend(["--scenario-spec", str(args.scenario_spec)])
        else:
            child_command.extend(["--scenario", args.scenario])
        if args.town:
            child_command.extend(["--town", args.town])
        if args.weather_preset:
            child_command.extend(["--weather-preset", args.weather_preset])
        if args.ego_spawn_index is not None:
            child_command.extend(["--ego-spawn-index", str(args.ego_spawn_index)])
        if args.goal_spawn_index is not None:
            child_command.extend(["--goal-spawn-index", str(args.goal_spawn_index)])
        if args.scenario_description:
            child_command.extend(["--scenario-description", args.scenario_description])
        if args.agent_module:
            child_command.extend(["--agent-module", args.agent_module])
        if args.coverage_profile:
            child_command.extend(["--coverage-profile", str(args.coverage_profile)])
        if args.coverage_layer:
            child_command.extend(["--coverage-layer", args.coverage_layer])
        for capture_tick in sorted(set(args.semantic_capture_tick)):
            child_command.extend(["--semantic-capture-tick", str(capture_tick)])
        if args.pcla_agent:
            child_command.extend(["--pcla-agent", args.pcla_agent])
        if args.agent_class:
            child_command.extend(["--agent-class", args.agent_class])
        if args.agent_checkpoint:
            child_command.extend(["--agent-checkpoint", str(args.agent_checkpoint)])
        if args.agent_config:
            child_command.extend(["--agent-config", str(args.agent_config)])

        completed = subprocess.run(child_command, check=False, capture_output=True, text=True)
        child_log_path.write_text(
            "STDOUT\n"
            f"{completed.stdout}\n\n"
            "STDERR\n"
            f"{completed.stderr}",
            encoding="utf-8",
        )
        _cleanup_carla_processes()
        time.sleep(args.settle_seconds)
        summary_path = _child_summary_path(args.output_dir, scenario_output_key, args.agent_kind, child_label)
        if summary_path.exists():
            child_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            child_result = dict(child_summary["results"][0])
            child_result["child_label"] = child_label
            child_result["returncode"] = completed.returncode
            child_result["child_log_path"] = str(child_log_path)
            child_results.append(child_result)
            continue

        child_results.append(
            {
                "run_index": run_index + 1,
                "port": args.port + run_index,
                "child_label": child_label,
                "returncode": completed.returncode,
                "ticks_executed": None,
                "reached_goal": False,
                "collision_count": None,
                "progress_to_goal_m": None,
                "max_cross_track_error_m": None,
                "max_abs_heading_error_deg": None,
                "full_brake_ticks": None,
                "throttle_active_ticks": None,
                "first_deviation_alert": None,
                "coverage_status": None,
                "coverage_target_layer": None,
                "coverage_hook_calls": None,
                "coverage_trace_dim": None,
                "coverage_trace_dump_path": None,
                "coverage_kmnc": None,
                "coverage_kmnc_neuron_coverage": None,
                "coverage_out_of_range_fraction": None,
                "coverage_lsa_mean": None,
                "coverage_lsa_max": None,
                "coverage_lsa_p95": None,
                "coverage_model_type": None,
                "coverage_named_module_count": None,
                "coverage_preferred_hook_layer": None,
                "semantic_anomaly_dump_path": None,
                "semantic_scene_dump_paths": None,
                "semantic_stream_dump_path": None,
                "semantic_stream_frame_count": None,
                "semantic_target_tick_dump_paths": None,
                "route_preview": None,
                "startup_hold_ticks": args.startup_hold_ticks,
                "child_log_path": str(child_log_path),
                "run_error": (completed.stderr or completed.stdout).strip()[:4000] or f"Child process failed with code {completed.returncode}.",
            }
        )

    summary = {
        "scenario_id": scenario_output_key,
        "scenario_spec_path": str(args.scenario_spec) if args.scenario_spec is not None else None,
        "scenario_overrides": {
            "town": args.town,
            "weather_preset": args.weather_preset,
            "ego_spawn_index": args.ego_spawn_index,
            "goal_spawn_index": args.goal_spawn_index,
            "description": args.scenario_description,
        },
        "runs": args.runs,
        "host": "127.0.0.1",
        "port": args.port,
        "port_strategy": "increment-per-run",
        "agent_kind": args.agent_kind,
        "agent_module": args.agent_module,
        "label": args.label,
        "startup_hold_ticks": args.startup_hold_ticks,
        "quality_level": args.quality_level,
        "results": child_results,
        "aggregate": {
            "goal_reaches": sum(1 for item in child_results if item.get("reached_goal")),
            "deviation_alert_runs": sum(1 for item in child_results if item.get("first_deviation_alert") is not None),
            "failed_runs": sum(1 for item in child_results if item.get("run_error") is not None),
            "mean_progress_to_goal_m": (
                sum(float(item["progress_to_goal_m"] or 0.0) for item in child_results if item.get("progress_to_goal_m") is not None)
                / max(sum(1 for item in child_results if item.get("progress_to_goal_m") is not None), 1)
            ) if child_results else None,
            "max_cross_track_error_m": max(float(item["max_cross_track_error_m"] or 0.0) for item in child_results) if child_results else None,
        },
    }

    out_path = args.output_dir / f"{scenario_output_key}-{args.agent_kind}-{args.label}-diagnostics.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()