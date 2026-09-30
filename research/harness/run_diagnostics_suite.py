from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.config import AppConfig
from research.harness.observers.semantic import SemanticObserver, SemanticObserverConfig
from research.harness.observers.telemetry import TelemetryObserver, TelemetryObserverConfig
from research.harness.runner import HarnessRunner
from research.harness.scenarios import SCENARIO_CATALOG, get_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeated ADS diagnostics for the same scenario and agent.")
    parser.add_argument("--scenario", default="town01_clear_short", choices=sorted(SCENARIO_CATALOG))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2100)
    parser.add_argument("--output-dir", type=Path, default=Path("research/logs/diagnostics"))
    parser.add_argument("--run-output-dir", type=Path, default=Path("research/logs/runs"))
    parser.add_argument("--telemetry-output", type=Path, default=Path("research/logs/telemetry"))
    parser.add_argument("--semantic-output", type=Path, default=Path("research/logs/semantic"))
    parser.add_argument("--semantic-anomaly-output", type=Path, default=Path("research/logs/semantic_dumps"))
    parser.add_argument("--semantic-stream-every", type=int, default=1)
    parser.add_argument("--semantic-capture-tick", action="append", type=int, default=[])
    parser.add_argument("--agent-kind", default="leaderboard-module")
    parser.add_argument("--behavior", default="normal", choices=["cautious", "normal", "aggressive"])
    parser.add_argument("--target-speed-kph", type=float, default=30.0)
    parser.add_argument("--agent-repo-path", type=Path, required=True)
    parser.add_argument("--agent-module", required=True)
    parser.add_argument("--agent-class", default=None)
    parser.add_argument("--agent-checkpoint", type=Path, default=None)
    parser.add_argument("--agent-config", type=Path, default=None)
    parser.add_argument("--max-ticks", type=int, default=200)
    parser.add_argument("--startup-hold-ticks", type=int, default=0)
    parser.add_argument("--telemetry-sample-every", type=int, default=1)
    parser.add_argument("--telemetry-deviation-threshold", type=float, default=2.0)
    parser.add_argument("--semantic-brake-threshold-ticks", type=int, default=10)
    parser.add_argument("--semantic-heading-threshold-deg", type=float, default=20.0)
    parser.add_argument("--label", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict[str, object]] = []
    for run_index in range(args.runs):
        scenario = get_scenario(args.scenario)
        scenario.max_ticks = args.max_ticks

        config = AppConfig()
        config.harness.host = args.host
        config.harness.port = args.port
        config.harness.output_dir = args.run_output_dir
        config.agent.kind = args.agent_kind
        config.agent.behavior = args.behavior
        config.agent.target_speed_kph = args.target_speed_kph
        config.agent.repo_path = args.agent_repo_path
        config.agent.module_name = args.agent_module
        config.agent.class_name = args.agent_class
        config.agent.checkpoint_path = args.agent_checkpoint
        config.agent.config_path = args.agent_config
        config.run.startup_hold_ticks = args.startup_hold_ticks

        observers = [
            TelemetryObserver(
                TelemetryObserverConfig(
                    trace_output_dir=args.telemetry_output,
                    sample_every_ticks=args.telemetry_sample_every,
                    deviation_alert_threshold_m=args.telemetry_deviation_threshold,
                )
            ),
            SemanticObserver(
                SemanticObserverConfig(
                    source="heuristic",
                    trace_output_dir=args.semantic_output,
                    stream_output_dir=args.semantic_output,
                    stream_every_ticks=args.semantic_stream_every,
                    anomaly_dump_dir=args.semantic_anomaly_output,
                    capture_ticks=tuple(sorted(set(args.semantic_capture_tick))),
                    persistent_brake_ticks=args.semantic_brake_threshold_ticks,
                    heading_error_threshold_deg=args.semantic_heading_threshold_deg,
                )
            ),
        ]

        runner = HarnessRunner(config, observers=observers)
        result = runner.run(scenario)
        telemetry = result.metadata.get("telemetry", {})
        semantic = result.metadata.get("semantic", {})
        run_summaries.append(
            {
                "run_index": run_index + 1,
                "ticks_executed": result.ticks_executed,
                "reached_goal": result.reached_goal,
                "collision_count": result.collision_count,
                "progress_to_goal_m": telemetry.get("progress_to_goal_m"),
                "max_cross_track_error_m": telemetry.get("route_progress", {}).get("max_cross_track_error_m"),
                "max_abs_heading_error_deg": telemetry.get("route_progress", {}).get("max_abs_heading_error_deg"),
                "full_brake_ticks": telemetry.get("full_brake_ticks"),
                "throttle_active_ticks": telemetry.get("throttle_active_ticks"),
                "first_deviation_alert": telemetry.get("first_deviation_alert"),
                "semantic_anomaly_dump_path": semantic.get("anomaly_dump_path"),
                "semantic_scene_dump_paths": semantic.get("scene_dump_paths"),
                "semantic_stream_dump_path": semantic.get("stream_dump_path"),
                "semantic_stream_frame_count": semantic.get("stream_frame_count"),
                "semantic_target_tick_dump_paths": semantic.get("target_tick_dump_paths"),
                "route_preview": telemetry.get("route_preview"),
                "startup_hold_ticks": result.metadata.get("startup_hold_ticks"),
            }
        )

    label = args.label or (f"hold{args.startup_hold_ticks}" if args.startup_hold_ticks else "default")
    summary = {
        "scenario_id": args.scenario,
        "runs": args.runs,
        "host": args.host,
        "port": args.port,
        "agent_kind": args.agent_kind,
        "agent_module": args.agent_module,
        "label": label,
        "startup_hold_ticks": args.startup_hold_ticks,
        "results": run_summaries,
        "aggregate": {
            "goal_reaches": sum(1 for item in run_summaries if item["reached_goal"]),
            "deviation_alert_runs": sum(1 for item in run_summaries if item["first_deviation_alert"] is not None),
            "mean_progress_to_goal_m": (
                sum(float(item["progress_to_goal_m"] or 0.0) for item in run_summaries) / len(run_summaries)
                if run_summaries
                else None
            ),
            "max_cross_track_error_m": max(
                float(item["max_cross_track_error_m"] or 0.0) for item in run_summaries
            ) if run_summaries else None,
        },
    }

    out_path = args.output_dir / f"{args.scenario}-{args.agent_kind}-{label}-diagnostics.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()