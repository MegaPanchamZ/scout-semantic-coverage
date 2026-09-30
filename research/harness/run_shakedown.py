from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.config import AppConfig
from research.harness.observers.coverage import CoverageObserver, CoverageObserverConfig
from research.harness.observers.debug_viewer import DebugViewerObserver, DebugViewerObserverConfig
from research.harness.observers.semantic import SemanticObserver, SemanticObserverConfig
from research.harness.observers.telemetry import TelemetryObserver, TelemetryObserverConfig
from research.harness.runner import HarnessRunner
from research.harness.scenarios import SCENARIO_CATALOG, get_scenario, load_scenario_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ADS harness shakedown scenario.")
    parser.add_argument("--scenario", default="town01_clear_short")
    parser.add_argument("--scenario-spec", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("research/logs/runs"))
    parser.add_argument("--agent-kind", default="behavior")
    parser.add_argument("--behavior", default="normal", choices=["cautious", "normal", "aggressive"])
    parser.add_argument("--target-speed-kph", type=float, default=30.0)
    parser.add_argument("--agent-repo-path", type=Path, default=None)
    parser.add_argument("--agent-module", default=None)
    parser.add_argument("--pcla-agent", default=None)
    parser.add_argument("--agent-class", default=None)
    parser.add_argument("--agent-checkpoint", type=Path, default=None)
    parser.add_argument("--agent-config", type=Path, default=None)
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--startup-hold-ticks", type=int, default=0)
    parser.add_argument("--coverage-observer", action="store_true")
    parser.add_argument("--coverage-profile", type=Path, default=None)
    parser.add_argument("--coverage-layer", default=None)
    parser.add_argument("--coverage-k-sections", type=int, default=1000)
    parser.add_argument("--coverage-trace-output", type=Path, default=Path("research/logs/coverage"))
    parser.add_argument("--semantic-observer", action="store_true")
    parser.add_argument("--semantic-source", default="heuristic")
    parser.add_argument("--semantic-trace-output", type=Path, default=Path("research/logs/semantic"))
    parser.add_argument("--semantic-stream-every", type=int, default=1)
    parser.add_argument("--semantic-anomaly-dump", action="store_true")
    parser.add_argument("--semantic-anomaly-output", type=Path, default=Path("research/logs/semantic_dumps"))
    parser.add_argument("--semantic-capture-tick", action="append", type=int, default=[])
    parser.add_argument("--semantic-brake-threshold-ticks", type=int, default=10)
    parser.add_argument("--semantic-heading-threshold-deg", type=float, default=20.0)
    parser.add_argument("--telemetry-observer", action="store_true")
    parser.add_argument("--telemetry-output", type=Path, default=Path("research/logs/telemetry"))
    parser.add_argument("--telemetry-sample-every", type=int, default=1)
    parser.add_argument("--telemetry-deviation-threshold", type=float, default=2.0)
    parser.add_argument("--debug-viewer", action="store_true")
    parser.add_argument("--debug-viewer-width", type=int, default=960)
    parser.add_argument("--debug-viewer-height", type=int, default=540)
    parser.add_argument("--debug-topdown-height", type=float, default=35.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.scenario_spec is not None:
        scenario = load_scenario_spec(args.scenario_spec)
    else:
        if args.scenario not in SCENARIO_CATALOG:
            known = ", ".join(sorted(SCENARIO_CATALOG))
            raise SystemExit(f"Unknown scenario '{args.scenario}'. Known scenarios: {known}")
        scenario = get_scenario(args.scenario)
    if args.max_ticks is not None:
        scenario.max_ticks = args.max_ticks

    config = AppConfig()
    config.harness.host = args.host
    config.harness.port = args.port
    config.harness.output_dir = args.output_dir
    config.agent.kind = args.agent_kind
    config.agent.behavior = args.behavior
    config.agent.target_speed_kph = args.target_speed_kph
    config.agent.repo_path = args.agent_repo_path
    config.agent.module_name = args.agent_module
    config.agent.pcla_agent_name = args.pcla_agent
    config.agent.class_name = args.agent_class
    config.agent.checkpoint_path = args.agent_checkpoint
    config.agent.config_path = args.agent_config
    config.run.dry_run = args.dry_run
    config.run.startup_hold_ticks = args.startup_hold_ticks

    observers = []
    if args.coverage_observer:
        observers.append(
            CoverageObserver(
                CoverageObserverConfig(
                    k_sections=args.coverage_k_sections,
                    layer_name=args.coverage_layer,
                    profile_path=args.coverage_profile,
                    trace_output_dir=args.coverage_trace_output,
                )
            )
        )
    if args.semantic_observer:
        observers.append(
            SemanticObserver(
                SemanticObserverConfig(
                    source=args.semantic_source,
                    trace_output_dir=args.semantic_trace_output,
                    stream_output_dir=args.semantic_trace_output,
                    stream_every_ticks=args.semantic_stream_every,
                    anomaly_dump_dir=args.semantic_anomaly_output if args.semantic_anomaly_dump else None,
                    capture_ticks=tuple(sorted(set(args.semantic_capture_tick))),
                    persistent_brake_ticks=args.semantic_brake_threshold_ticks,
                    heading_error_threshold_deg=args.semantic_heading_threshold_deg,
                )
            )
        )
    if args.telemetry_observer:
        observers.append(
            TelemetryObserver(
                TelemetryObserverConfig(
                    trace_output_dir=args.telemetry_output,
                    sample_every_ticks=args.telemetry_sample_every,
                    deviation_alert_threshold_m=args.telemetry_deviation_threshold,
                )
            )
        )
    if args.debug_viewer:
        observers.append(
            DebugViewerObserver(
                DebugViewerObserverConfig(
                    image_width=args.debug_viewer_width,
                    image_height=args.debug_viewer_height,
                    topdown_height_m=args.debug_topdown_height,
                )
            )
        )

    runner = HarnessRunner(config, observers=observers or None)
    result = runner.run(scenario)
    print(json.dumps(result.to_dict(), indent=2))
    # Some agent stacks (e.g., PCLA/TransFuser) leave non-joinable worker
    # threads that abort the interpreter during teardown after all results
    # have been written. Exit explicitly once the artifacts are on disk.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()