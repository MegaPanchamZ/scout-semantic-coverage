#!/usr/bin/env python3
"""Run controlled positive/negative/uncertainty validation cases for outcome detectors."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.outcome_config import (
    DEFAULT_PROFILE,
    implementation_digest,
    load_outcome_profile,
    profile_content_digest,
    profile_digest,
)
from research.harness.outcome_io import EvidenceWriter, atomic_json, file_digest, iter_evidence, load_capture_manifest
from research.harness.outcome_models import (
    ActorState, CaptureManifest, ChannelCoverage, FrameEvidence, RoadRuleContext,
    RouteProgress, ScenarioChecks, SensorEvidence, TaskContext, strict_json,
)
from research.harness.safety_outcomes import OutcomeEvaluator, classify_run


SUITE_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = SUITE_DIR / "cases.json"
ALL_CHANNELS = (
    "actors", "ego_motion", "collision", "acceleration", "controls",
    "lane_invasion", "road_context", "task_context", "route_progress",
)
VALID_CHECKS = ScenarioChecks(True, True, True, True, True, evidence_refs=("validation_control",))
COMPLIANT_ROAD = RoadRuleContext(
    lane_id="1:-1", lane_type="driving", in_junction=False, lane_change_permission="both",
    permitted_travel_yaw_rad=0.0, speed_limit_mps=13.89, red_light_state="green",
    stop_line_applicable=False, stop_line_crossed=False,
    stop_sign_applicable=False, stop_sign_approach=False, stop_sign_line_crossed=False,
    evidence_refs=("validation_map",),
)
FREE_TASK = TaskContext(True, False, False, False, False, False, ("validation_task",))


def load_cases(path: Path = DEFAULT_CASES) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("Validation cases file must define a nonempty 'cases' list")
    return data


def suite_content_digest(suite: dict[str, Any], suite_dir: Path = SUITE_DIR) -> str:
    """Content-address the full validation suite definition and every referenced spec file."""
    spec_hashes: dict[str, str | None] = {}
    for case in suite.get("cases") or ():
        if not isinstance(case, dict):
            continue
        rel = str(case.get("spec") or "")
        if not rel or rel in spec_hashes:
            continue
        spec_path = suite_dir / rel
        spec_hashes[rel] = file_digest(spec_path) if spec_path.is_file() else None
    payload = {"suite": suite, "specs": spec_hashes}
    return hashlib.sha256(strict_json(payload).encode("utf-8")).hexdigest()


def validate_case(case: dict[str, Any], suite_dir: Path = SUITE_DIR) -> list[str]:
    errors: list[str] = []
    for key in ("case_id", "family", "control_kind", "spec", "expected"):
        if not case.get(key):
            errors.append(f"missing_{key}")
    if case.get("control_kind") not in {"positive", "negative", "uncertainty"}:
        errors.append("invalid_control_kind")
    spec_path = suite_dir / str(case.get("spec", ""))
    if not spec_path.is_file():
        errors.append(f"missing_spec_file:{spec_path}")
    return errors


def _ego(x=0.0, y=0.0, yaw=0.0, vx=10.0, vy=0.0):
    return ActorState("ego", "vehicle", x, y, yaw, 2.0, 1.0, vx, vy, 0.0, 2.0, role="ego")


def _synthetic_evidence(case_id: str) -> tuple[list[FrameEvidence], list[SensorEvidence], CaptureManifest]:
    """Deterministic independent synthetic ground truth for each declared case."""
    run_id = f"val-{case_id}"
    dt = 0.1

    def mk_frames(n=21, *, speed=10.0, yaw=0.0, road=COMPLIANT_ROAD, task=FREE_TASK,
                  actors_fn=None, accel_fn=None, brake_fn=None, reverse=False,
                  channels=ALL_CHANNELS, advance_per_step=1.0):
        vx, vy = speed * math.cos(yaw), speed * math.sin(yaw)
        out = []
        for i in range(n):
            t = round(i * dt, 8)
            s_val = round(i * advance_per_step, 6)
            out.append(
                FrameEvidence(
                    run_id=run_id, sim_frame=i, sim_time_s=t,
                    ego=_ego(x=i * vx * dt, y=i * vy * dt, yaw=yaw, vx=vx, vy=vy),
                    actors=tuple(actors_fn(i) if actors_fn else ()),
                    available_channels=channels,
                    longitudinal_acceleration_mps2=accel_fn(i) if accel_fn else 0.0,
                    brake=brake_fn(i) if brake_fn else 0.0,
                    task_context=task if "task_context" in channels else None,
                    route_progress=RouteProgress("val_route", "available", 1000.0, s_val, s_val, 0.0) if "route_progress" in channels else None,
                    road_context=road if "road_context" in channels else None,
                    reverse=reverse,
                )
            )
        return out

    def mk_manifest(end_s=2.0, *, channels=ALL_CHANNELS, checks=VALID_CHECKS):
        return CaptureManifest(
            run_id=run_id, start_s=0.0, end_s=end_s,
            execution_status="completed", termination_reason="horizon_exhausted",
            channels={c: ChannelCoverage("complete", 0.0, end_s) for c in channels},
            reached_goal=False, scan_radius_m=50.0, projection_horizon_s=3.0, max_sample_interval_s=0.1,
            scenario_checks=checks, provenance={"mode": "offline_synthetic_control"},
        )

    sensors: list[SensorEvidence] = []
    if case_id == "collision_positive_contact":
        frames = mk_frames(21, actors_fn=lambda i: (ActorState("ped", "pedestrian", 10.0, 0.0, 0.0, 0.4, 0.4, 0.0, 0.0, 0.0, 1.8, role="pedestrian_subject"),))
        sensors = [SensorEvidence(run_id, 10, 1.0, "collision", "ped", impulse_ns=(5.0, 0.0, 0.0), callback_id="c1")]
        return frames, sensors, mk_manifest(2.0)

    if case_id == "collision_negative_close_pass":
        frames = mk_frames(21, actors_fn=lambda i: (ActorState("ped", "pedestrian", i * 1.0, 2.8, 0.0, 0.4, 0.4, 10.0, 0.0, 0.0, 1.8, role="pedestrian_subject"),))
        return frames, [], mk_manifest(2.0)

    if case_id == "collision_uncertainty_disabled_sensor":
        ch = tuple(c for c in ALL_CHANNELS if c != "collision")
        return mk_frames(21, channels=ch), [], mk_manifest(2.0, channels=ch)

    if case_id == "near_miss_positive_resolved_conflict":
        frames = mk_frames(21, actors_fn=lambda i: (ActorState("lead", "vehicle", 14.0 + 2.0 * i, 0.0, 0.0, 2.0, 1.0, 0.0 if i == 0 else 20.0, 0.0, 0.0, 2.0, role="lead_vehicle"),))
        return frames, [], mk_manifest(2.0)

    if case_id == "near_miss_negative_parallel_adjacent":
        frames = mk_frames(21, actors_fn=lambda i: (ActorState("adj", "vehicle", i * 1.0, 3.0, 0.0, 2.0, 1.0, 10.0, 0.0, 0.0, 2.0, role="adjacent_vehicle"),))
        return frames, [], mk_manifest(2.0)

    if case_id == "lane_positive_prohibited_solid_crossing":
        frames = mk_frames(21)
        sensors = [SensorEvidence(run_id, 5, 0.5, "lane_invasion", "ego", lane_markings=("Solid",), callback_id="l1")]
        return frames, sensors, mk_manifest(2.0)

    if case_id == "lane_negative_permitted_broken_change":
        frames = mk_frames(21)
        sensors = [SensorEvidence(run_id, 5, 0.5, "lane_invasion", "ego", lane_markings=("Broken",), callback_id="l1")]
        return frames, sensors, mk_manifest(2.0)

    if case_id == "braking_positive_hazard_associated":
        frames = mk_frames(
            21,
            actors_fn=lambda i: (ActorState("lead", "vehicle", 10.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 2.0, role="lead_vehicle"),) if i == 1 else (),
            accel_fn=lambda i: -10.0 if 2 <= i <= 4 else 0.0,
            brake_fn=lambda i: 1.0 if 2 <= i <= 4 else 0.0,
        )
        return frames, [], mk_manifest(2.0)

    if case_id == "braking_negative_low_speed_and_no_hazard":
        frames = mk_frames(21, speed=4.0, accel_fn=lambda i: -10.0 if 2 <= i <= 4 else 0.0, brake_fn=lambda i: 1.0 if 2 <= i <= 4 else 0.0)
        return frames, [], mk_manifest(2.0)

    if case_id == "red_light_positive_crossing":
        frames = mk_frames(21)
        frames[10] = replace(frames[10], road_context=replace(COMPLIANT_ROAD, stop_line_applicable=True, stop_line_crossed=True, red_light_state="red"))
        return frames, [], mk_manifest(2.0)

    if case_id == "red_light_negative_approach_stop":
        frames = mk_frames(21)
        frames[10] = replace(frames[10], road_context=replace(COMPLIANT_ROAD, stop_line_applicable=True, stop_line_crossed=False, red_light_state="red"))
        return frames, [], mk_manifest(2.0)

    if case_id == "other_rules_positive_stop_sign_speeding_wrong_way":
        frames = mk_frames(21, speed=16.0, yaw=math.pi, reverse=False)
        for i in range(3, 8):
            frames[i] = replace(frames[i], road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=False))
        frames[8] = replace(frames[8], road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=True))
        return frames, [], mk_manifest(2.0)

    if case_id == "other_rules_negative_compliant_and_reverse":
        frames = mk_frames(21, speed=3.0, yaw=math.pi, reverse=True)
        return frames, [], mk_manifest(2.0)

    if case_id == "task_positive_free_route_stuck":
        frames = mk_frames(101, speed=0.0, advance_per_step=0.0)
        return frames, [], mk_manifest(10.0)

    if case_id == "task_negative_legitimate_red_light_wait":
        frames = mk_frames(101, speed=0.0, advance_per_step=0.0, task=replace(FREE_TASK, red_light_stop=True))
        return frames, [], mk_manifest(10.0)

    if case_id == "quality_positive_low_interaction":
        frames = mk_frames(101, speed=5.0, advance_per_step=0.5)
        return frames, [], mk_manifest(10.0)

    if case_id == "quality_positive_spawn_failure":
        frames = mk_frames(21, speed=5.0, advance_per_step=0.5)
        return frames, [], mk_manifest(2.0, checks=replace(VALID_CHECKS, required_actors_spawned=False))

    if case_id == "quality_negative_safe_meaningful_interaction":
        frames = mk_frames(101, speed=5.0, advance_per_step=0.5, actors_fn=lambda i: (ActorState("adj", "vehicle", i * 0.5, 8.0, 0.0, 2.0, 1.0, 5.0, 0.0, 0.0, 2.0, role="adjacent_vehicle"),))
        return frames, [], mk_manifest(10.0)

    raise KeyError(f"No synthetic control builder for case {case_id}")


def compare_expected_outcomes(expected: dict[str, str], outcome: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    observed = {}
    for key, exp_status in expected.items():
        grp, name = key.split(".", 1)
        got = outcome["categories"][grp][name]["status"]
        observed[key] = got
        if got != exp_status:
            mismatches.append({"metric": key, "expected": exp_status, "observed": got})
    return {"passed": not mismatches, "observed": observed, "mismatches": mismatches}


def run_case(
    case: dict[str, Any],
    output_dir: Path,
    *,
    profile_path: Path = DEFAULT_PROFILE,
    mode: str = "offline-synthetic",
    host: str = "127.0.0.1",
    port: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    profile = load_outcome_profile(profile_path)
    case_dir = output_dir / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)

    if mode == "live":
        from research.harness.config import AppConfig
        from research.harness.observers.semantic import SemanticObserver, SemanticObserverConfig
        from research.harness.runner import HarnessRunner
        from research.harness.scenarios import load_scenario_spec

        try:
            spec_path = SUITE_DIR / str(case["spec"])
            scenario = load_scenario_spec(spec_path)
            cfg = AppConfig()
            cfg.harness.host = host
            cfg.harness.port = int(port)
            cfg.harness.seed = int(seed)
            cfg.harness.reload_world = True
            cfg.harness.output_dir = case_dir / "runs"
            cfg.agent.kind = "scripted"
            cfg.agent.init_kwargs["case_id"] = case["case_id"]
            cfg.run.stop_on_first_collision = case.get("family") == "collision"
            cfg.outcomes.enabled = True
            cfg.outcomes.profile_path = profile_path
            cfg.outcomes.evidence_output_dir = case_dir / "evidence"
            if case["case_id"] == "collision_uncertainty_disabled_sensor":
                cfg.sensors.attach_collision_sensor = False

            sem_observer = SemanticObserver(
                SemanticObserverConfig(
                    stream_output_dir=case_dir / "semantic",
                    trace_output_dir=case_dir / "semantic",
                    stream_every_ticks=1,
                )
            )
            result = HarnessRunner(cfg, observers=[sem_observer]).run(scenario)
        except Exception as exc:  # noqa: BLE001 - capture runner errors into failed validation record
            return {
                "case_id": case["case_id"],
                "family": case["family"],
                "control_kind": case["control_kind"],
                "mode": mode,
                "run_purpose": "detector_validation",
                "passed": False,
                "replay_matches_live": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        live_outcome = result.metadata.get("outcome_v2")
        manifest_path = Path(result.capture_manifest_path) if result.capture_manifest_path else None
        if not isinstance(live_outcome, dict) or manifest_path is None:
            return {
                "case_id": case["case_id"],
                "family": case["family"],
                "control_kind": case["control_kind"],
                "mode": mode,
                "run_purpose": "detector_validation",
                "passed": False,
                "replay_matches_live": False,
                "error": "Live run did not emit outcome_v2 or capture_manifest_path",
            }
        loaded_manifest, resolved_trace = load_capture_manifest(manifest_path)
        evidence_records = list(iter_evidence(resolved_trace))
        replayed = classify_run(
            {"capture_manifest": asdict(loaded_manifest)},
            profile=profile,
            evidence=evidence_records,
        )
        replay_matches = live_outcome == replayed
        comparison = compare_expected_outcomes(case["expected"], live_outcome)

        # Verify observed duration and intended physical actor roles when declared.
        extra_failures: list[str] = []
        req_dur = float(case.get("required_duration_s") or 0.0)
        obs_dur = float(result.observed_duration_s or 0.0)
        expects_early_stop = (
            case.get("family") == "collision"
            or case.get("expected", {}).get("quality.invalid_scenario") == "present"
        )
        if req_dur > 0.0 and not expects_early_stop and obs_dur + 1e-6 < req_dur:
            extra_failures.append(f"insufficient_duration:{obs_dur:.3f}<{req_dur:.3f}")
        observed_roles = {
            a.role
            for rec in evidence_records
            if isinstance(rec, FrameEvidence) and rec.phase == "evaluation"
            for a in (rec.ego, *rec.actors)
            if a.role
        }
        missing_roles = [r for r in (case.get("intended_roles") or ()) if r not in observed_roles]
        if missing_roles:
            extra_failures.append(f"missing_intended_roles:{','.join(missing_roles)}")
        if case["case_id"] == "red_light_negative_approach_stop":
            saw_red_approach = any(
                isinstance(rec, FrameEvidence)
                and rec.phase == "evaluation"
                and rec.road_context is not None
                and rec.road_context.stop_line_applicable is True
                and rec.road_context.red_light_state == "red"
                for rec in evidence_records
            )
            if not saw_red_approach:
                extra_failures.append("red_light_exposure_not_observed")

        return {
            "case_id": case["case_id"],
            "family": case["family"],
            "control_kind": case["control_kind"],
            "mode": mode,
            "run_purpose": "detector_validation",
            "passed": comparison["passed"] and replay_matches and not extra_failures,
            "replay_matches_live": replay_matches,
            "observed_duration_s": obs_dur,
            "observed_roles": sorted(observed_roles),
            "extra_failures": extra_failures,
            "expected": case["expected"],
            "observed": comparison["observed"],
            "mismatches": comparison["mismatches"],
            "manifest_path": str(manifest_path),
        }

    frames, sensors, cap_manifest = _synthetic_evidence(case["case_id"])
    trace_path = case_dir / "trace.jsonl"
    writer = EvidenceWriter(trace_path, cap_manifest.run_id)
    evaluator = OutcomeEvaluator(profile)
    for f in frames:
        evaluator.observe(f)
        writer.append(f)
    for s in sensors:
        evaluator.record_sensor_event(s)
        writer.append(s)
    manifest_path = writer.finish(cap_manifest)
    live_outcome = evaluator.finish(cap_manifest).to_dict()

    loaded_manifest, resolved_trace = load_capture_manifest(manifest_path)
    replayed = classify_run(
        {"capture_manifest": asdict(loaded_manifest)},
        profile=profile,
        evidence=list(iter_evidence(resolved_trace)),
    )
    replay_matches = live_outcome == replayed
    comparison = compare_expected_outcomes(case["expected"], live_outcome)
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "control_kind": case["control_kind"],
        "mode": mode,
        "passed": comparison["passed"] and replay_matches,
        "replay_matches_live": replay_matches,
        "expected": case["expected"],
        "observed": comparison["observed"],
        "mismatches": comparison["mismatches"],
        "manifest_path": str(manifest_path),
    }


def write_validation_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "validation_report.json"
    atomic_json(target, report)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--mode", choices=("offline-synthetic", "live"), default="offline-synthetic")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats per selected case.")
    parser.add_argument("--case-id", action="append", default=[], help="Optional subset of case IDs to execute.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    suite = load_cases(args.cases)
    profile = load_outcome_profile(args.profile)
    available_ids = {str(c.get("case_id")) for c in suite["cases"] if c.get("case_id")}
    selected = set(args.case_id)
    unknown_ids = sorted(selected - available_ids)
    if unknown_ids:
        parser.error(f"Unknown --case-id selection(s): {', '.join(unknown_ids)}")

    selected_cases = [c for c in suite["cases"] if not selected or c.get("case_id") in selected]
    if not selected_cases:
        parser.error("No validation cases selected")

    results = []
    for case in selected_cases:
        errs = validate_case(case, args.cases.parent)
        if errs:
            results.append({"case_id": case.get("case_id"), "passed": False, "validation_errors": errs})
            continue
        for repeat_idx in range(args.repeats):
            rep_dir = args.output_dir / "cases" if args.repeats == 1 else args.output_dir / f"repeat_{repeat_idx}" / "cases"
            rec = run_case(
                case,
                rep_dir,
                profile_path=args.profile,
                mode=args.mode,
                host=args.host,
                port=args.port,
                seed=args.seed + repeat_idx,
            )
            if args.repeats > 1:
                rec["repeat_index"] = repeat_idx
            results.append(rec)

    passed = sum(1 for r in results if r["passed"])
    canonical_suite = load_cases(DEFAULT_CASES)
    canonical_ids = [str(c.get("case_id")) for c in canonical_suite["cases"] if c.get("case_id")]
    expected_ids = [str(c.get("case_id")) for c in suite["cases"] if c.get("case_id")]
    executed_ids = sorted({str(r.get("case_id")) for r in results if r.get("case_id")})
    passed_ids = sorted({str(r.get("case_id")) for r in results if r.get("case_id") and r.get("passed")})
    actual_cases_digest = suite_content_digest(suite, args.cases.parent)
    canonical_cases_digest = suite_content_digest(canonical_suite, DEFAULT_CASES.parent)
    suite_complete = (
        suite.get("suite_id") == canonical_suite.get("suite_id")
        and actual_cases_digest == canonical_cases_digest
        and set(canonical_ids).issubset(set(executed_ids))
    )
    report = {
        "suite_id": suite["suite_id"],
        "cases_digest": actual_cases_digest,
        "mode": args.mode,
        "repeats": args.repeats,
        "suite_complete": suite_complete,
        "expected_case_ids": expected_ids,
        "passed_case_ids": passed_ids,
        "profile_id": profile.profile_id,
        "profile_status": profile.status,
        "profile_digest": profile_digest(profile),
        "profile_content_digest": profile_content_digest(profile),
        "implementation_digest": implementation_digest(),
        "n_cases": len(results),
        "n_passed": passed,
        "all_passed": len(results) > 0 and passed == len(results),
        "note": (
            "Offline synthetic controls verify software calculations and live/replay equivalence; "
            "live CARLA execution is still required to graduate a draft profile to validated."
            if args.mode == "offline-synthetic"
            else "Live CARLA detector validation run."
        ),
        "cases": results,
    }
    write_validation_report(report, args.output_dir)
    print(f"validation ({args.mode}): {passed}/{len(results)} passed")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
