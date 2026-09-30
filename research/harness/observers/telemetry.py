from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from research.harness.models import RunResult, ScenarioSpec


@dataclass(slots=True)
class TelemetryObserverConfig:
    trace_output_dir: Path | None = None
    sample_every_ticks: int = 1
    deviation_alert_threshold_m: float = 2.0


class TelemetryObserver:
    def __init__(self, config: TelemetryObserverConfig) -> None:
        self.config = config
        self._status = "not-started"
        self._trace: list[dict[str, Any]] = []
        self._first_deviation_alert: dict[str, Any] | None = None
        self._route_preview: list[dict[str, Any]] = []

    def on_run_start(self, scenario: ScenarioSpec, context: dict[str, Any]) -> None:
        del scenario
        self._trace = []
        self._status = "collecting"
        self._first_deviation_alert = None
        self._route_preview = list(context.get("route_preview", []))
        context["telemetry_observer"] = self

    def on_tick(self, tick_index: int, context: dict[str, Any]) -> None:
        if self._status != "collecting":
            return
        if self.config.sample_every_ticks > 1 and tick_index % self.config.sample_every_ticks != 0:
            return
        telemetry = context.get("telemetry")
        if not isinstance(telemetry, dict):
            return
        self._trace.append(dict(telemetry))
        if self._first_deviation_alert is None:
            route_progress = telemetry.get("agent_step", {}).get("route_progress", {})
            cross_track_error = route_progress.get("cross_track_error_m")
            if cross_track_error is not None and float(cross_track_error) >= self.config.deviation_alert_threshold_m:
                self._first_deviation_alert = {
                    "threshold_m": self.config.deviation_alert_threshold_m,
                    "tick": int(telemetry.get("tick", tick_index + 1)),
                    "cross_track_error_m": float(cross_track_error),
                    "distance_to_goal_m": float(telemetry.get("distance_to_goal_m", 0.0)),
                    "speed_mps": float(telemetry.get("speed_mps", 0.0)),
                    "speed_kph": float(telemetry.get("speed_kph", 0.0)),
                    "ego_location": telemetry.get("location"),
                    "control": telemetry.get("control"),
                    "route_progress": {
                        "next_waypoint_index": route_progress.get("next_waypoint_index"),
                        "nearest_route_waypoint_index": route_progress.get("nearest_route_waypoint_index"),
                        "remaining_waypoints": route_progress.get("remaining_waypoints"),
                        "distance_to_next_waypoint_m": route_progress.get("distance_to_next_waypoint_m"),
                        "nearest_route_waypoint_distance_m": route_progress.get("nearest_route_waypoint_distance_m"),
                        "ego_yaw_deg": route_progress.get("ego_yaw_deg"),
                        "route_heading_deg": route_progress.get("route_heading_deg"),
                        "heading_error_deg": route_progress.get("heading_error_deg"),
                        "next_route_waypoint_location": route_progress.get("next_route_waypoint_location"),
                        "nearest_route_waypoint_location": route_progress.get("nearest_route_waypoint_location"),
                        "nearest_driving_waypoint_location": route_progress.get("nearest_driving_waypoint_location"),
                        "nearest_driving_waypoint_yaw_deg": route_progress.get("nearest_driving_waypoint_yaw_deg"),
                    },
                }

    def on_run_end(self, result: RunResult, context: dict[str, Any]) -> None:
        del context
        self._status = "completed"
        trace_path: str | None = None
        if self.config.trace_output_dir is not None and self._trace:
            trace_path = str(self._write_trace_dump(result, self.config.trace_output_dir))

        speed_samples = [float(sample.get("speed_mps", 0.0)) for sample in self._trace]
        distance_samples = [float(sample.get("distance_to_goal_m", 0.0)) for sample in self._trace]
        throttle_samples = [float(sample.get("control", {}).get("throttle", 0.0)) for sample in self._trace]
        brake_samples = [float(sample.get("control", {}).get("brake", 0.0)) for sample in self._trace]
        steer_samples = [abs(float(sample.get("control", {}).get("steer", 0.0))) for sample in self._trace]
        waiting_ticks = sum(1 for sample in self._trace if sample.get("agent_step", {}).get("mode") == "sensor-wait")
        conflicting_control_ticks = sum(
            1
            for sample in self._trace
            if float(sample.get("control", {}).get("throttle", 0.0)) > 0.05
            and float(sample.get("control", {}).get("brake", 0.0)) > 0.05
        )
        stalled_throttle_ticks = sum(
            1
            for sample in self._trace
            if float(sample.get("control", {}).get("throttle", 0.0)) > 0.05
            and float(sample.get("speed_mps", 0.0)) < 0.1
        )
        route_progress_samples = [
            sample.get("agent_step", {}).get("route_progress", {})
            for sample in self._trace
            if isinstance(sample.get("agent_step", {}).get("route_progress", {}), dict)
        ]
        next_waypoint_distances = [
            float(route_progress.get("distance_to_next_waypoint_m"))
            for route_progress in route_progress_samples
            if route_progress.get("distance_to_next_waypoint_m") is not None
        ]
        cross_track_errors = [
            float(route_progress.get("cross_track_error_m"))
            for route_progress in route_progress_samples
            if route_progress.get("cross_track_error_m") is not None
        ]
        heading_errors = [
            abs(float(route_progress.get("heading_error_deg")))
            for route_progress in route_progress_samples
            if route_progress.get("heading_error_deg") is not None
        ]
        remaining_waypoints = [
            int(route_progress.get("remaining_waypoints"))
            for route_progress in route_progress_samples
            if route_progress.get("remaining_waypoints") is not None
        ]
        progress_to_goal = distance_samples[0] - distance_samples[-1] if len(distance_samples) >= 2 else None

        result.metadata["telemetry"] = {
            "status": self._status,
            "trace_count": len(self._trace),
            "sample_every_ticks": self.config.sample_every_ticks,
            "max_speed_mps": max(speed_samples) if speed_samples else None,
            "max_speed_kph": (max(speed_samples) * 3.6) if speed_samples else None,
            "final_speed_mps": speed_samples[-1] if speed_samples else None,
            "final_distance_to_goal_m": distance_samples[-1] if distance_samples else None,
            "min_distance_to_goal_m": min(distance_samples) if distance_samples else None,
            "progress_to_goal_m": progress_to_goal,
            "throttle_active_ticks": sum(1 for value in throttle_samples if value > 0.05),
            "full_brake_ticks": sum(1 for value in brake_samples if value >= 0.95),
            "conflicting_control_ticks": conflicting_control_ticks,
            "stalled_throttle_ticks": stalled_throttle_ticks,
            "max_abs_steer": max(steer_samples) if steer_samples else None,
            "sensor_wait_ticks": waiting_ticks,
            "route_progress": {
                "final_distance_to_next_waypoint_m": next_waypoint_distances[-1] if next_waypoint_distances else None,
                "min_distance_to_next_waypoint_m": min(next_waypoint_distances) if next_waypoint_distances else None,
                "final_cross_track_error_m": cross_track_errors[-1] if cross_track_errors else None,
                "max_cross_track_error_m": max(cross_track_errors) if cross_track_errors else None,
                "final_abs_heading_error_deg": heading_errors[-1] if heading_errors else None,
                "max_abs_heading_error_deg": max(heading_errors) if heading_errors else None,
                "final_remaining_waypoints": remaining_waypoints[-1] if remaining_waypoints else None,
                "min_remaining_waypoints": min(remaining_waypoints) if remaining_waypoints else None,
            },
            "control_summary": {
                "final_throttle": throttle_samples[-1] if throttle_samples else None,
                "final_brake": brake_samples[-1] if brake_samples else None,
                "mean_throttle": (sum(throttle_samples) / len(throttle_samples)) if throttle_samples else None,
                "mean_brake": (sum(brake_samples) / len(brake_samples)) if brake_samples else None,
            },
            "route_preview": self._route_preview,
            "first_deviation_alert": self._first_deviation_alert,
            "trace_dump_path": trace_path,
        }

    def _write_trace_dump(self, result: RunResult, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = output_dir / f"{result.scenario_id}-{result.agent_kind}-{timestamp}-telemetry.json"
        out_path.write_text(json.dumps(self._trace, indent=2), encoding="utf-8")
        return out_path