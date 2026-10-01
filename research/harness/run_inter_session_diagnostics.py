from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.config import AppConfig
from research.harness.observers.dashcam import DashcamObserver, DashcamObserverConfig
from research.harness.observers.coverage import CoverageObserver, CoverageObserverConfig
from research.harness.observers.debug_viewer import DebugViewerObserver, DebugViewerObserverConfig
from research.harness.observers.semantic import SemanticObserver, SemanticObserverConfig
from research.harness.observers.telemetry import TelemetryObserver, TelemetryObserverConfig
from research.harness.runner import HarnessRunner
from research.harness.models import ScenarioSpec
from research.harness.scenarios import SCENARIO_CATALOG, load_scenario_spec, resolve_scenario


def _workspace_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return (WORKSPACE_ROOT / path).resolve()


def _scenario_output_key(scenario: ScenarioSpec) -> str:
    return scenario.scenario_id


def _parse_debug_capture_targets(raw_targets: list[str], output_dir: Path | None) -> dict[int, Path]:
    capture_targets: dict[int, Path] = {}
    for raw_target in raw_targets:
        tick_text, separator, filename = raw_target.partition(":")
        if not separator or not filename:
            raise ValueError(
                "Debug capture targets must use the format <tick>:<filename>, "
                f"but received '{raw_target}'."
            )
        try:
            tick = int(tick_text)
        except ValueError as exc:
            raise ValueError(f"Debug capture tick '{tick_text}' is not an integer.") from exc
        target_path = Path(filename)
        if not target_path.is_absolute():
            if output_dir is None:
                raise ValueError(
                    "Relative debug capture filenames require --debug-capture-output-dir."
                )
            target_path = output_dir / target_path
        capture_targets[tick] = target_path.resolve()
    return capture_targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run diagnostics with a fresh CARLA process for each repetition.")
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
    parser.add_argument("--dashcam-capture", action="store_true")
    parser.add_argument("--dashcam-output", type=Path, default=Path("research/logs/perception_frames"))
    parser.add_argument("--dashcam-stream-every", type=int, default=None)
    parser.add_argument("--dashcam-width", type=int, default=960)
    parser.add_argument("--dashcam-height", type=int, default=540)
    parser.add_argument("--agent-kind", default="leaderboard-module")
    parser.add_argument("--behavior", default="normal", choices=["cautious", "normal", "aggressive"])
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
    parser.add_argument("--debug-capture-tick", action="append", default=[])
    parser.add_argument("--debug-capture-output-dir", type=Path, default=None)
    parser.add_argument("--debug-capture-sensor", choices=["front", "topdown"], default="topdown")
    parser.add_argument("--debug-viewer-width", type=int, default=960)
    parser.add_argument("--debug-viewer-height", type=int, default=540)
    parser.add_argument("--debug-topdown-height", type=float, default=35.0)
    parser.add_argument("--startup-hold-ticks", type=int, default=0)
    parser.add_argument("--label", default="restart-sessions")
    parser.add_argument("--resolution-x", type=int, default=800)
    parser.add_argument("--resolution-y", type=int, default=600)
    parser.add_argument("--quality-level", default=os.environ.get("SCOUT_CARLA_QUALITY", "Epic"))
    parser.add_argument("--boot-timeout-seconds", type=float, default=60.0)
    return parser


def wait_for_carla(host: str, port: int, timeout_seconds: float) -> None:
    import carla  # type: ignore

    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            client = carla.Client(host, port)
            client.set_timeout(5.0)
            client.get_world()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2.0)
    raise RuntimeError(f"CARLA server on {host}:{port} did not become ready in time.") from last_error


def launch_carla(carla_root: Path, port: int, resolution_x: int, resolution_y: int, quality_level: str) -> subprocess.Popen[bytes]:
    if os.name == "nt":
        exe_path = carla_root / "CarlaUE4.exe"
        if not exe_path.exists():
            raise FileNotFoundError(f"CARLA executable not found at {exe_path}")
        args = [
            str(exe_path),
            f"-carla-rpc-port={port}",
            f"-quality-level={quality_level}",
            "-windowed",
            f"-ResX={resolution_x}",
            f"-ResY={resolution_y}",
            "-nosound",
        ]
        creationflags = 0
        if hasattr(subprocess, "DETACHED_PROCESS"):
            creationflags |= subprocess.DETACHED_PROCESS
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
        return subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )

    launcher_path = carla_root / "CarlaUE4.sh"
    if not launcher_path.exists():
        raise FileNotFoundError(f"CARLA launcher not found at {launcher_path}")
    args = [
        str(launcher_path),
        f"-carla-rpc-port={port}",
        f"-quality-level={quality_level}",
        "-RenderOffScreen",
        "-opengl",
        "-nosound",
    ]
    graphics_adapter = os.environ.get("MRES_CARLA_GRAPHICS_ADAPTER")
    if graphics_adapter is not None:
        args.append(f"-graphicsadapter={graphics_adapter}")
    return subprocess.Popen(
        args,
        cwd=str(carla_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def stop_carla(carla_process: subprocess.Popen[bytes]) -> None:
    if carla_process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(carla_process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        import signal

        try:
            os.killpg(os.getpgid(carla_process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        carla_process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            import signal

            try:
                os.killpg(os.getpgid(carla_process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        else:
            carla_process.kill()
        carla_process.wait(timeout=10)


def main() -> None:
    args = build_parser().parse_args()
    if args.scenario_spec is None and args.scenario not in SCENARIO_CATALOG and args.town is None:
        known = ", ".join(sorted(SCENARIO_CATALOG))
        raise ValueError(f"Unknown scenario '{args.scenario}'. Known scenarios: {known}")
    args.carla_root = _workspace_path(args.carla_root) or args.carla_root
    args.scenario_spec = _workspace_path(args.scenario_spec)
    args.output_dir = _workspace_path(args.output_dir) or args.output_dir
    args.run_output_dir = _workspace_path(args.run_output_dir) or args.run_output_dir
    args.telemetry_output = _workspace_path(args.telemetry_output) or args.telemetry_output
    args.coverage_profile = _workspace_path(args.coverage_profile)
    args.coverage_trace_output = _workspace_path(args.coverage_trace_output) or args.coverage_trace_output
    args.semantic_output = _workspace_path(args.semantic_output) or args.semantic_output
    args.semantic_anomaly_output = _workspace_path(args.semantic_anomaly_output) or args.semantic_anomaly_output
    args.dashcam_output = _workspace_path(args.dashcam_output) or args.dashcam_output
    args.debug_capture_output_dir = _workspace_path(args.debug_capture_output_dir)
    args.agent_repo_path = _workspace_path(args.agent_repo_path)
    args.agent_checkpoint = _workspace_path(args.agent_checkpoint)
    args.agent_config = _workspace_path(args.agent_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.debug_capture_output_dir is not None:
        args.debug_capture_output_dir.mkdir(parents=True, exist_ok=True)
    debug_capture_targets = _parse_debug_capture_targets(args.debug_capture_tick, args.debug_capture_output_dir)

    if args.agent_kind == "pcla":
        if args.pcla_agent is None:
            raise ValueError("PCLA diagnostics require --pcla-agent.")
        if args.agent_repo_path is None:
            args.agent_repo_path = Path("research/models/PCLA")
    elif args.agent_kind != "behavior":
        if args.agent_repo_path is None:
            raise ValueError(f"Agent kind '{args.agent_kind}' requires --agent-repo-path.")
        if args.agent_kind == "leaderboard-module" and args.agent_module is None:
            raise ValueError("Leaderboard-module diagnostics require --agent-module.")

    if args.scenario_spec is not None:
        if any(value is not None for value in (args.town, args.weather_preset, args.ego_spawn_index, args.goal_spawn_index, args.scenario_description)):
            raise ValueError("--scenario-spec cannot be combined with scenario override flags.")
        base_scenario = load_scenario_spec(args.scenario_spec)
    else:
        base_scenario = resolve_scenario(
            args.scenario,
            town=args.town,
            weather_preset=args.weather_preset,
            ego_spawn_index=args.ego_spawn_index,
            goal_spawn_index=args.goal_spawn_index,
            description=args.scenario_description,
        )
    base_scenario.max_ticks = args.max_ticks
    scenario_output_key = _scenario_output_key(base_scenario)

    run_summaries: list[dict[str, object]] = []
    for run_index in range(args.runs):
        current_port = args.port + run_index
        carla_process = launch_carla(args.carla_root, current_port, args.resolution_x, args.resolution_y, args.quality_level)
        try:
            wait_for_carla("127.0.0.1", current_port, args.boot_timeout_seconds)

            scenario = ScenarioSpec.from_dict(base_scenario.to_dict())

            config = AppConfig()
            config.harness.host = "127.0.0.1"
            config.harness.port = current_port
            config.harness.timeout_seconds = max(float(args.boot_timeout_seconds), config.harness.timeout_seconds)
            config.harness.output_dir = args.run_output_dir
            config.agent.kind = args.agent_kind
            config.agent.behavior = args.behavior
            config.agent.target_speed_kph = args.target_speed_kph
            config.agent.repo_path = args.agent_repo_path
            config.agent.module_name = args.agent_module
            config.agent.pcla_agent_name = args.pcla_agent
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
                CoverageObserver(
                    CoverageObserverConfig(
                        k_sections=args.coverage_k_sections,
                        layer_name=args.coverage_layer,
                        profile_path=args.coverage_profile,
                        trace_output_dir=args.coverage_trace_output,
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
            if args.dashcam_capture:
                dashcam_stream_every = args.dashcam_stream_every or args.semantic_stream_every
                observers.append(
                    DashcamObserver(
                        DashcamObserverConfig(
                            output_root_dir=args.dashcam_output,
                            stream_every_ticks=max(int(dashcam_stream_every), 1),
                            image_width=args.dashcam_width,
                            image_height=args.dashcam_height,
                        )
                    )
                )
            if debug_capture_targets:
                observers.append(
                    DebugViewerObserver(
                        DebugViewerObserverConfig(
                            image_width=args.debug_viewer_width,
                            image_height=args.debug_viewer_height,
                            topdown_height_m=args.debug_topdown_height,
                            enable_window=False,
                            capture_sensor=args.debug_capture_sensor,
                            capture_targets=debug_capture_targets,
                        )
                    )
                )

            runner = HarnessRunner(config, observers=observers)
            try:
                result = runner.run(scenario)
            except Exception as exc:  # noqa: BLE001
                formatted_traceback = traceback.format_exc()
                print(formatted_traceback, file=sys.stderr)
                run_summaries.append(
                    {
                        "run_index": run_index + 1,
                        "port": current_port,
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
                        "dashcam_status": None,
                        "dashcam_output_dir": None,
                        "dashcam_frame_count": None,
                        "dashcam_stream_every": None,
                        "dashcam_frames": None,
                        "dashcam_missing_capture_ticks": None,
                        "debug_capture_status": None,
                        "debug_capture_paths": None,
                        "route_preview": None,
                        "startup_hold_ticks": args.startup_hold_ticks,
                        "run_error": f"{repr(exc)}\n{formatted_traceback}",
                    }
                )
            else:
                telemetry = result.metadata.get("telemetry", {})
                coverage = result.metadata.get("coverage", {})
                coverage_model_info = coverage.get("model_info", {}) if isinstance(coverage.get("model_info"), dict) else {}
                semantic = result.metadata.get("semantic", {})
                dashcam = result.metadata.get("dashcam", {})
                debug_viewer = result.metadata.get("debug_viewer", {})
                run_summaries.append(
                    {
                        "run_index": run_index + 1,
                        "port": current_port,
                        "ticks_executed": result.ticks_executed,
                        "reached_goal": result.reached_goal,
                        "collision_count": result.collision_count,
                        "progress_to_goal_m": telemetry.get("progress_to_goal_m"),
                        "max_cross_track_error_m": telemetry.get("route_progress", {}).get("max_cross_track_error_m"),
                        "max_abs_heading_error_deg": telemetry.get("route_progress", {}).get("max_abs_heading_error_deg"),
                        "full_brake_ticks": telemetry.get("full_brake_ticks"),
                        "throttle_active_ticks": telemetry.get("throttle_active_ticks"),
                        "first_deviation_alert": telemetry.get("first_deviation_alert"),
                        "coverage_status": coverage.get("status"),
                        "coverage_target_layer": coverage.get("resolved_target_layer") or coverage.get("target_layer"),
                        "coverage_hook_calls": coverage.get("hook_calls"),
                        "coverage_trace_dim": coverage.get("trace_dim"),
                        "coverage_trace_dump_path": coverage.get("trace_dump_path"),
                        "coverage_kmnc": coverage.get("kmnc"),
                        "coverage_kmnc_neuron_coverage": coverage.get("kmnc_neuron_coverage"),
                        "coverage_out_of_range_fraction": coverage.get("out_of_range_fraction"),
                        "coverage_lsa_mean": coverage.get("lsa_mean"),
                        "coverage_lsa_max": coverage.get("lsa_max"),
                        "coverage_lsa_p95": coverage.get("lsa_p95"),
                        "coverage_model_type": coverage_model_info.get("model_type"),
                        "coverage_named_module_count": coverage_model_info.get("named_module_count"),
                        "coverage_preferred_hook_layer": coverage_model_info.get("preferred_hook_layer"),
                        "semantic_anomaly_dump_path": semantic.get("anomaly_dump_path"),
                        "semantic_scene_dump_paths": semantic.get("scene_dump_paths"),
                        "semantic_stream_dump_path": semantic.get("stream_dump_path"),
                        "semantic_stream_frame_count": semantic.get("stream_frame_count"),
                        "semantic_target_tick_dump_paths": semantic.get("target_tick_dump_paths"),
                        "dashcam_status": dashcam.get("status"),
                        "dashcam_output_dir": dashcam.get("output_dir"),
                        "dashcam_frame_count": dashcam.get("frame_count"),
                        "dashcam_stream_every": dashcam.get("stream_every_ticks"),
                        "dashcam_frames": dashcam.get("frames"),
                        "dashcam_missing_capture_ticks": dashcam.get("missing_capture_ticks"),
                        "debug_capture_status": debug_viewer.get("status"),
                        "debug_capture_paths": debug_viewer.get("captured_paths"),
                        "route_preview": telemetry.get("route_preview"),
                        "startup_hold_ticks": result.metadata.get("startup_hold_ticks"),
                        "run_error": None,
                    }
                )
        finally:
            stop_carla(carla_process)
            time.sleep(5.0)

    summary = {
        "scenario_id": base_scenario.scenario_id,
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
        "results": run_summaries,
        "aggregate": {
            "goal_reaches": sum(1 for item in run_summaries if item["reached_goal"]),
            "deviation_alert_runs": sum(1 for item in run_summaries if item["first_deviation_alert"] is not None),
            "failed_runs": sum(1 for item in run_summaries if item.get("run_error") is not None),
            "mean_progress_to_goal_m": (
                sum(float(item["progress_to_goal_m"] or 0.0) for item in run_summaries if item.get("progress_to_goal_m") is not None)
                / max(sum(1 for item in run_summaries if item.get("progress_to_goal_m") is not None), 1)
                if run_summaries
                else None
            ),
            "max_cross_track_error_m": max(
                float(item["max_cross_track_error_m"] or 0.0) for item in run_summaries
            ) if run_summaries else None,
        },
    }

    out_path = args.output_dir / f"{scenario_output_key}-{args.agent_kind}-{args.label}-diagnostics.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()