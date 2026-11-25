"""Tests for the matched five-evaluation diagnostic pilot and controlled validation suite."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import pytest

from research.harness.outcome_config import load_outcome_profile
from research.harness.outcome_models import SensorEvidence
from research.harness.safety_outcomes import OutcomeEvaluator
from research.tests.outcome_helpers import frame, manifest
from research.tests.test_scenario_quality import VALID

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_SCRIPT = REPO_ROOT / "research/experiments/EXP-020-policy-comparison/outcome-pilot/run_pilot.py"
VAL_SCRIPT = REPO_ROOT / "research/experiments/outcome-validation/run_validation.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pilot = _load_module("outcome_pilot_under_test", PILOT_SCRIPT)
val_runner = _load_module("outcome_validation_under_test", VAL_SCRIPT)


def _validated_profile():
    p = load_outcome_profile()
    return replace(p, status="validated", validation_refs=("live-control-run-001",))


def _baseline_outcome(*, collision=False, valid=True):
    ev = OutcomeEvaluator(_validated_profile())
    for i in range(11):
        ev.observe(frame(i))
    if collision:
        ev.record_sensor_event(SensorEvidence("run", 5, 0.5, "collision", "a"))
    m = manifest(1.0, scenario_checks=VALID if valid else None)
    return ev.finish(m).to_dict()


def _write_verified_live_report(base_dir: Path, profile, *, mutate_case=None) -> Path:
    """Create a checksum-verified 18-case live validation report whose on-disk captures replay cleanly."""
    import json as _json
    from research.harness.outcome_config import implementation_digest, profile_content_digest, profile_digest
    from research.harness.outcome_io import EvidenceWriter

    canonical = val_runner.load_cases()
    caps_dir = base_dir / "verified_live_caps"
    caps_dir.mkdir(parents=True, exist_ok=True)
    case_entries = []
    for c in canonical["cases"]:
        cid = c["case_id"]
        spec_payload = _json.loads((val_runner.SUITE_DIR / c["spec"]).read_text())
        frames, sensors, cap_m = val_runner._synthetic_evidence(cid)
        live_prov = {
            "capture_source": "carla_live_runner",
            "scenario_id": spec_payload["scenario_id"],
            "town": spec_payload["town"],
            "agent_kind": "scripted",
            "case_id": cid,
        }
        live_cap = replace(cap_m, provenance=live_prov)
        if mutate_case is not None:
            frames, sensors, live_cap = mutate_case(cid, frames, sensors, live_cap)
        c_dir = caps_dir / cid
        c_dir.mkdir(parents=True, exist_ok=True)
        w = EvidenceWriter(c_dir / "trace.jsonl", live_cap.run_id)
        for f in frames:
            w.append(f)
        for s in sensors:
            w.append(s)
        m_path = w.finish(live_cap)
        case_entries.append({
            "case_id": cid,
            "family": c["family"],
            "control_kind": c["control_kind"],
            "mode": "live",
            "run_purpose": "detector_validation",
            "passed": True,
            "replay_matches_live": True,
            "expected": c["expected"],
            "manifest_path": str(m_path),
        })
    report_path = base_dir / "verified_live_val.json"
    report_path.write_text(_json.dumps({
        "suite_id": canonical["suite_id"],
        "cases_digest": val_runner.suite_content_digest(canonical, val_runner.DEFAULT_CASES.parent),
        "mode": "live",
        "suite_complete": True,
        "all_passed": True,
        "passed_case_ids": [c["case_id"] for c in canonical["cases"]],
        "cases": case_entries,
        "profile_digest": profile_digest(profile),
        "profile_content_digest": profile_content_digest(profile),
        "implementation_digest": implementation_digest(),
    }))
    return report_path


def test_pilot_admission_rejects_draft_profile_collision_and_missing_validity():
    protocol = pilot.load_protocol()
    draft = load_outcome_profile()
    validated = _validated_profile()

    # Draft profile refused
    res_draft = pilot.validate_initial_configuration(_baseline_outcome(), protocol, profile=draft)
    assert res_draft["admitted"] is False
    assert any("profile_not_ready" in r for r in res_draft["reasons"])

    # Baseline with collision refused
    res_col = pilot.validate_initial_configuration(_baseline_outcome(collision=True), protocol, profile=validated)
    assert res_col["admitted"] is False
    assert any("collision_free" in r for r in res_col["reasons"])

    # Missing scenario checks refused
    res_invalid = pilot.validate_initial_configuration(_baseline_outcome(valid=False), protocol, profile=validated)
    assert res_invalid["admitted"] is False

    # Valid collision-free baseline admitted
    res_ok = pilot.validate_initial_configuration(_baseline_outcome(), protocol, profile=validated, realized_state_digest="abc")
    assert res_ok["admitted"] is True


def test_pilot_matched_arms_shared_eval0_failed_budget_and_incompatible_resume(tmp_path):
    protocol = pilot.load_protocol()
    validated = _validated_profile()
    matched = pilot.build_matched_arms(
        protocol, validated, "pedestrian_crossing", seed=13, shared_eval_zero={"evidence_sha256": "deadbeef"}
    )
    assert len(matched["arms"]) == 4

    arm_plan = matched["arms"][0]
    shared_row = {"outcome_v2": _baseline_outcome(), "evidence_sha256": "deadbeef"}
    calls = []

    def fake_eval(arm, idx, cand):
        calls.append((arm, idx, cand))
        if idx == 2:
            return {"execution_status": "failed", "run_error": "timeout"}
        if idx == 3:
            return {"execution_status": "completed", "outcome_v2": _baseline_outcome(collision=True)}
        return {"execution_status": "completed", "outcome_v2": _baseline_outcome()}

    arm_dir = tmp_path / "arm0"
    res = pilot.run_arm(arm_plan, fake_eval, shared_eval_zero_row=shared_row, output_dir=arm_dir)
    assert res["budget"]["logical_evaluations"] == 5
    assert res["budget"]["physical_simulation_calls"] == 4  # shared eval 0 not re-simulated
    assert res["budget"]["failed_attempts_charged"] == 1
    assert res["budget"]["discoveries_credited"] == 1
    assert res["attempts"][0]["eligible_for_discovery"] is False
    assert res["attempts"][0]["credited_as_discovery"] is False
    assert res["attempts"][1]["credited_as_discovery"] is False  # clean run is not a discovery
    assert res["attempts"][2]["credited_as_discovery"] is False  # failed run is not a discovery
    assert res["attempts"][3]["credited_as_discovery"] is True   # newly confirmed collision!
    assert res["attempts"][3]["new_outcomes"] == ["safety.collision"]
    assert res["attempts"][4]["credited_as_discovery"] is False

    # Compatible resume into completed directory must NOT re-simulate completed indices!
    calls_before_resume = len(calls)
    resumed = pilot.run_arm(arm_plan, fake_eval, shared_eval_zero_row=shared_row, output_dir=arm_dir)
    assert len(calls) == calls_before_resume
    assert len(resumed["attempts"]) == 5

    # Incompatible resume into same directory must fail
    other_matched = pilot.build_matched_arms(protocol, validated, "pedestrian_crossing", seed=99)
    with pytest.raises(ValueError, match="Incompatible pilot checkpoint"):
        pilot.run_arm(other_matched["arms"][0], fake_eval, output_dir=arm_dir)


def test_pilot_admission_requires_realized_digest_and_intended_witness():
    protocol = pilot.load_protocol()
    validated = _validated_profile()
    fam = protocol["interaction_families"][0]  # pedestrian_crossing declares intended_role + expected_predicates

    no_digest = pilot.validate_initial_configuration(_baseline_outcome(), protocol, profile=validated)
    assert no_digest["admitted"] is False
    assert "missing_realized_state_digest" in no_digest["reasons"]

    no_witness = pilot.validate_initial_configuration(
        _baseline_outcome(), protocol, profile=validated, realized_state_digest="abc", intended_witnessed=None, family=fam,
    )
    assert no_witness["admitted"] is False
    assert "intended_interaction_not_witnessed" in no_witness["reasons"]

    ok = pilot.validate_initial_configuration(
        _baseline_outcome(), protocol, profile=validated, realized_state_digest="abc", intended_witnessed=True, family=fam,
    )
    assert ok["admitted"] is True


def test_pilot_arms_differentiate_proposed_candidates_by_policy(tmp_path):
    protocol = pilot.load_protocol()
    validated = _validated_profile()
    matched = pilot.build_matched_arms(protocol, validated, "pedestrian_crossing", seed=7)
    base_spec = {"controller_params": {"trigger_radius_m": 20.0}}

    def eval_with_gradient(arm, idx, cand):
        # Reward smaller trigger_radius_m with higher criticality so critonly diverges from random
        rad = float(cand.get("trigger_radius_m", 20.0))
        return {
            "execution_status": "completed",
            "outcome_v2": _baseline_outcome(),
            "criticality": round(max(0.0, 1.0 - rad / 40.0), 4),
        }

    rand_plan = next(a for a in matched["arms"] if a["arm"] == "random")
    crit_plan = next(a for a in matched["arms"] if a["arm"] == "critonly")
    res_rand = pilot.run_arm(rand_plan, eval_with_gradient, base_spec=base_spec)
    res_crit = pilot.run_arm(crit_plan, eval_with_gradient, base_spec=base_spec)
    rand_cands = [a["candidate"] for a in res_rand["attempts"]]
    crit_cands = [a["candidate"] for a in res_crit["attempts"]]
    assert rand_cands != crit_cands


def test_pilot_rejects_auto_extend_protocol(tmp_path):
    bad = tmp_path / "bad_proto.json"
    proto = pilot.load_protocol()
    proto["budget_per_arm"]["auto_extend"] = True
    bad.write_text(__import__("json").dumps(proto))
    with pytest.raises(ValueError, match="automatic budget extension"):
        pilot.load_protocol(bad)


def test_offline_controlled_validation_suite_passes_all_18_controls(tmp_path):
    rc = val_runner.main(["--mode", "offline-synthetic", "--output-dir", str(tmp_path / "val")])
    assert rc == 0
    report = __import__("json").loads((tmp_path / "val" / "validation_report.json").read_text())
    assert report["all_passed"] is True
    assert report["n_cases"] == 18
    assert all(c["replay_matches_live"] for c in report["cases"])

    # Unknown --case-id must fail instead of returning all_passed=True on 0 cases
    with pytest.raises(SystemExit):
        val_runner.main(["--mode", "offline-synthetic", "--case-id", "nonexistent_case", "--output-dir", str(tmp_path / "val_bad")])


def test_all_validation_specs_build_valid_scenario_controllers():
    from research.harness.scenario_runtime import build_scenario_controller
    from research.harness.scenarios import load_scenario_spec
    specs_dir = VAL_SCRIPT.parent / "specs"
    specs = sorted(specs_dir.glob("*.json"))
    assert len(specs) == 9
    for sp in specs:
        scenario = load_scenario_spec(sp)
        controller = build_scenario_controller(scenario)
        assert controller is not None


def test_validation_report_verification_rejects_subset_and_accepts_promoted_profile(tmp_path):
    from research.harness.outcome_config import implementation_digest, profile_content_digest, profile_digest
    import json as _json

    draft = load_outcome_profile()
    promoted = _validated_profile()
    canonical = val_runner.load_cases()
    all_case_ids = [c["case_id"] for c in canonical["cases"]]

    # 1. Partial subset report (suite_complete=False) must be rejected
    subset_path = tmp_path / "subset_val.json"
    subset_path.write_text(_json.dumps({
        "suite_id": canonical["suite_id"],
        "mode": "live", "suite_complete": False, "all_passed": True,
        "passed_case_ids": all_case_ids[:1],
        "profile_digest": profile_digest(draft),
        "profile_content_digest": profile_content_digest(draft),
        "implementation_digest": implementation_digest(),
    }))
    v_sub = pilot.verify_validation_report(subset_path, promoted)
    assert v_sub["verified"] is False
    assert "validation_report_incomplete_suite" in v_sub["reasons"]

    # 1b. Forged 1-case custom --cases file claiming suite_complete=True must ALSO be rejected!
    forged_path = tmp_path / "forged_val.json"
    forged_path.write_text(_json.dumps({
        "suite_id": canonical["suite_id"],
        "mode": "live", "suite_complete": True, "all_passed": True,
        "passed_case_ids": all_case_ids[:1],
        "profile_digest": profile_digest(draft),
        "profile_content_digest": profile_content_digest(draft),
        "implementation_digest": implementation_digest(),
    }))
    v_forged = pilot.verify_validation_report(forged_path, promoted)
    assert v_forged["verified"] is False
    assert any("validation_missing_canonical_cases" in r for r in v_forged["reasons"])

    # 2. Full live report produced against draft passes verification after promoting profile to validated
    full_path = _write_verified_live_report(tmp_path / "full_live_ok", draft)
    v_ok = pilot.verify_validation_report(full_path, promoted)
    assert v_ok["verified"] is True

    # 2a. One empty not_executed capture reused across all 18 cases must be rejected!
    from research.harness.outcome_io import EvidenceWriter
    empty_cap_dir = tmp_path / "empty_cap"
    empty_cap_dir.mkdir()
    ew = EvidenceWriter(empty_cap_dir / "empty.jsonl", "empty-not-exec")
    empty_manifest = ew.finish(manifest(0.0, run_id="empty-not-exec", execution_status="not_executed", provenance={"mode": "live_carla_test"}))
    fake_unrelated_path = tmp_path / "fake_unrelated_val.json"
    fake_unrelated_path.write_text(_json.dumps({
        "suite_id": canonical["suite_id"],
        "cases_digest": val_runner.suite_content_digest(canonical, val_runner.DEFAULT_CASES.parent),
        "mode": "live", "suite_complete": True, "all_passed": True,
        "passed_case_ids": all_case_ids,
        "cases": [
            {
                "case_id": c["case_id"], "family": c["family"], "control_kind": c["control_kind"],
                "mode": "live", "run_purpose": "detector_validation",
                "passed": True, "replay_matches_live": True,
                "expected": c["expected"], "manifest_path": str(empty_manifest),
            }
            for c in canonical["cases"]
        ],
        "profile_digest": profile_digest(draft),
        "profile_content_digest": profile_content_digest(draft),
        "implementation_digest": implementation_digest(),
    }))
    v_unrel = pilot.verify_validation_report(fake_unrelated_path, promoted)
    assert v_unrel["verified"] is False
    assert any("validation_case_synthetic_capture" in r for r in v_unrel["reasons"])
    assert any("validation_case_reused_capture" in r for r in v_unrel["reasons"])

    # 2a2. Missing live runner provenance, short 0.2s lane trace, and red-light negative with no red exposure must all fail!
    def _bad_controls(cid, frames, sensors, cap_m):
        if cid == "lane_positive_prohibited_solid_crossing":
            # Only 0.2s of trace frames even though manifest claims end_s=2.0 AND outcomes match!
            short = list(frames[:3])
            short_sensors = [replace(sensors[0], sim_frame=1, sim_time_s=0.1)] if sensors else []
            return short, short_sensors, cap_m
        if cid == "red_light_negative_approach_stop":
            # Remove red-light exposure frame -> no red signal ever observed
            no_red = [replace(f, road_context=replace(f.road_context, stop_line_applicable=False, red_light_state="green")) if f.road_context else f for f in frames]
            return no_red, sensors, cap_m
        if cid == "collision_positive_contact":
            return frames, sensors, replace(cap_m, provenance={"mode": "custom_synthetic"})
        if cid == "braking_negative_low_speed_and_no_hazard":
            # Substitute another case_id sharing the same scenario/town/agent_kind!
            return frames, sensors, replace(cap_m, provenance={**cap_m.provenance, "case_id": "red_light_negative_approach_stop"})
        return frames, sensors, cap_m

    bad_ctrl_path = _write_verified_live_report(tmp_path / "bad_live_controls", draft, mutate_case=_bad_controls)
    v_bad_ctrl = pilot.verify_validation_report(bad_ctrl_path, promoted)
    assert v_bad_ctrl["verified"] is False
    assert any("validation_case_synthetic_capture:collision_positive_contact" in r for r in v_bad_ctrl["reasons"])
    assert any("validation_case_insufficient_duration:lane_positive_prohibited_solid_crossing" in r for r in v_bad_ctrl["reasons"])
    assert any("validation_case_missing_red_light_exposure:red_light_negative_approach_stop" in r for r in v_bad_ctrl["reasons"])
    assert any("validation_case_id_mismatch:braking_negative_low_speed_and_no_hazard" in r for r in v_bad_ctrl["reasons"])

    # 2b. Flipping only top-level mode="live" on an offline-synthetic canonical report must be rejected!
    syn_out = tmp_path / "syn_val_out"
    assert val_runner.main(["--mode", "offline-synthetic", "--output-dir", str(syn_out)]) == 0
    syn_report = _json.loads((syn_out / "validation_report.json").read_text())
    syn_report["mode"] = "live"
    syn_forged_path = tmp_path / "syn_top_live.json"
    syn_forged_path.write_text(_json.dumps(syn_report))
    v_syn = pilot.verify_validation_report(syn_forged_path, promoted)
    assert v_syn["verified"] is False
    assert any("validation_case_mode_not_live" in r for r in v_syn["reasons"])
    assert any("validation_case_synthetic_capture" in r for r in v_syn["reasons"])

    # 2c. Relabeling BOTH top-level and case modes to "live" AND stripping manifest_path must ALSO be rejected!
    stripped_report = _json.loads(_json.dumps(syn_report))
    for entry in stripped_report["cases"]:
        entry["mode"] = "live"
        entry.pop("manifest_path", None)
    stripped_path = tmp_path / "syn_stripped_live.json"
    stripped_path.write_text(_json.dumps(stripped_report))
    v_stripped = pilot.verify_validation_report(stripped_path, promoted)
    assert v_stripped["verified"] is False
    assert any("validation_case_missing_manifest" in r for r in v_stripped["reasons"])

    # 3. Altered canonical controls (--cases with modified expected assertion) must fail suite_complete AND pilot verifier
    tampered = _json.loads(_json.dumps(canonical))
    tampered["cases"][0]["expected"] = {"quality.invalid_scenario": "absent"}
    tampered_dir = tmp_path / "tampered_suite"
    tampered_dir.mkdir()
    tampered_cases = tampered_dir / "cases.json"
    tampered_cases.write_text(_json.dumps(tampered))
    # Point spec symlink or copy specs so validate_case succeeds
    (tampered_dir / "specs").symlink_to(val_runner.SUITE_DIR / "specs")
    val_out = tmp_path / "tampered_val_out"
    rc_tamp = val_runner.main(["--cases", str(tampered_cases), "--mode", "offline-synthetic", "--output-dir", str(val_out)])
    assert rc_tamp == 0
    tamp_report = _json.loads((val_out / "validation_report.json").read_text())
    assert tamp_report["suite_complete"] is False
    # Even if someone forces mode="live" and suite_complete=True on the tampered report, pilot verifier rejects it!
    tamp_report["mode"] = "live"
    tamp_report["suite_complete"] = True
    tamp_path = tmp_path / "tampered_live.json"
    tamp_path.write_text(_json.dumps(tamp_report))
    v_tamp = pilot.verify_validation_report(tamp_path, promoted)
    assert v_tamp["verified"] is False
    assert "validation_cases_digest_mismatch" in v_tamp["reasons"]
    assert any("validation_case_expectations_mismatch" in r for r in v_tamp["reasons"])


def test_pilot_all_protocol_families_apply_default_search_space_and_reject_changed_protocol_resume(tmp_path):
    import json as _json
    protocol = pilot.load_protocol()
    for fam in protocol["interaction_families"]:
        spec_path = (pilot.DEFAULT_PROTOCOL.parent / fam["spec"]).resolve()
        payload = _json.loads(spec_path.read_text())
        space = pilot._default_pilot_space(payload, seed=13)
        cand = space.from_payload(payload, fill_missing=True)
        assert cand is not None
        mutated = space.apply_to_payload(payload, cand)
        assert mutated["controller"] == payload["controller"]

    # Verify complete JSON line WITHOUT trailing newline does not corrupt attempts.jsonl on resume
    validated = _validated_profile()
    matched = pilot.build_matched_arms(protocol, validated, "safe_interaction_control", seed=13)
    arm_plan = matched["arms"][0]
    arm_dir = tmp_path / "no_nl_arm"
    base_spec_nl = {"controller_params": {"trigger_radius_m": 20.0}}
    with pytest.raises(RuntimeError, match="stop_after_eval0"):
        pilot.run_arm(
            arm_plan,
            lambda arm, idx, cand: (
                {"execution_status": "completed", "outcome_v2": _baseline_outcome()}
                if idx == 0
                else (_ for _ in ()).throw(RuntimeError("stop_after_eval0"))
            ),
            output_dir=arm_dir,
            base_spec=base_spec_nl,
        )
    (arm_dir / "inflight_attempt.json").unlink()
    # Strip trailing newline so eval-0 line is complete JSON WITHOUT trailing newline!
    (arm_dir / "attempts.jsonl").write_text((arm_dir / "attempts.jsonl").read_text().rstrip("\n"))
    res = pilot.run_arm(
        arm_plan,
        lambda arm, idx, cand: {"execution_status": "completed", "outcome_v2": _baseline_outcome()},
        output_dir=arm_dir,
        base_spec=base_spec_nl,
    )
    assert len(res["attempts"]) == 5
    # Second restart must parse every line cleanly without JSONDecodeError
    res2 = pilot.run_arm(
        arm_plan,
        lambda arm, idx, cand: (_ for _ in ()).throw(AssertionError("should not re-run")),
        output_dir=arm_dir,
        base_spec=base_spec_nl,
    )
    assert len(res2["attempts"]) == 5


def test_pilot_atomic_newline_repair_preserves_prior_rows_on_write_failure(tmp_path, monkeypatch):
    import os as _os

    protocol = pilot.load_protocol()
    validated = _validated_profile()
    matched = pilot.build_matched_arms(protocol, validated, "safe_interaction_control", seed=13)
    arm_plan = matched["arms"][0]
    arm_dir = tmp_path / "atomic_repair_arm"
    base_spec_ar = {"controller": "adjacent_vehicle", "controller_params": {"throttle": 0.4}}
    with pytest.raises(RuntimeError, match="stop_after_eval0"):
        pilot.run_arm(
            arm_plan,
            lambda arm, idx, cand: (
                {"execution_status": "completed", "outcome_v2": _baseline_outcome()}
                if idx == 0
                else (_ for _ in ()).throw(RuntimeError("stop_after_eval0"))
            ),
            output_dir=arm_dir,
            base_spec=base_spec_ar,
        )
    (arm_dir / "inflight_attempt.json").unlink()
    attempts_file = arm_dir / "attempts.jsonl"
    raw_initial = attempts_file.read_text().rstrip("\n")  # no trailing newline -> triggers repair
    attempts_file.write_text(raw_initial)

    def _boom_replace(src, dst):
        raise OSError("simulated crash during atomic replace")

    monkeypatch.setattr(_os, "replace", _boom_replace)
    with pytest.raises(OSError, match="simulated crash"):
        pilot.run_arm(
            arm_plan,
            lambda arm, idx, cand: {"execution_status": "completed", "outcome_v2": _baseline_outcome()},
            output_dir=arm_dir,
            base_spec=base_spec_ar,
        )
    # Original valid eval-0 row must remain completely intact!
    assert attempts_file.read_text() == raw_initial


def test_live_validation_and_pilot_evaluator_wire_semantic_observer_and_realized_digest(tmp_path, monkeypatch):
    """Mocked HarnessRunner integration check exercising real SemanticObserverConfig & realized_initial_state_digest."""
    from dataclasses import asdict
    import json as _json
    from research.harness.models import RunResult
    from research.harness.outcome_io import EvidenceWriter
    from research.tests.outcome_helpers import actor

    def _fake_runner_run(self, scenario):
        out_dir = Path(self.config.outcomes.evidence_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"mock-{scenario.scenario_id}"
        frames = [
            frame(i, run_id=run_id, ego=actor("1", role="ego"), actors=(actor("2", actor_type="pedestrian", x=10.0, y=5.0, role="pedestrian_subject"),))
            for i in range(25)
        ]
        writer = EvidenceWriter(out_dir / f"{run_id}.jsonl", run_id)
        active_profile = load_outcome_profile(self.config.outcomes.profile_path) if self.config.outcomes.profile_path else load_outcome_profile()
        ev = OutcomeEvaluator(active_profile)
        for f in frames:
            writer.append(f)
            ev.observe(f)
        cap_m = manifest(2.4, run_id=run_id, scenario_checks=VALID)
        m_path = writer.finish(cap_m)
        outcome_dict = ev.finish(cap_m).to_dict()
        # Inject semantic trace matching intended interaction into attached SemanticObserver
        for obs in self.observers:
            if hasattr(obs, "_trace"):
                obs._trace.append({
                    "sim_time_s": 0.5,
                    "predicates": [
                        {"predicate": "in_front_of", "signature": "in_front_of(pedestrian,ego)", "actor_ids": ["2", "1"], "actor_roles": ["pedestrian_subject", "ego"]},
                        {"predicate": "crossing_path", "signature": "crossing_path(pedestrian,ego)", "actor_ids": ["2", "1"], "actor_roles": ["pedestrian_subject", "ego"]},
                    ],
                })
        return RunResult(
            scenario_id=scenario.scenario_id, town=scenario.town, agent_kind=self.config.agent.kind,
            succeeded=True, dry_run=False, ticks_executed=25, reached_goal=False,
            terminated_by_collision=False, collision_count=0,
            run_id=run_id, execution_status="completed", termination_reason="horizon_exhausted",
            observed_duration_s=2.4, capture_manifest_path=str(m_path),
            metadata={"outcome_v2": outcome_dict, "semantic": {"obligation_credit": {"fulfilled_obligations": ["in_front_of(pedestrian,ego)"]}}},
        )

    monkeypatch.setattr("research.harness.runner.HarnessRunner.run", _fake_runner_run)

    # 1. Exercise live validation runner (verifies SemanticObserverConfig & role/duration checks)
    val_case = {
        "case_id": "collision_negative_close_pass",
        "family": "collision",
        "control_kind": "negative",
        "spec": "specs/town01_pedestrian_crossing.json",
        "intended_roles": ["ego", "pedestrian_subject"],
        "expected": {"safety.collision": "absent"},
        "required_duration_s": 2.0,
    }
    val_res = val_runner.run_case(val_case, tmp_path / "live_val", mode="live")
    assert val_res["passed"] is True
    assert val_res["replay_matches_live"] is True
    assert "pedestrian_subject" in val_res["observed_roles"]

    # 2. Exercise pilot CLI end-to-end AND interrupted restart (baseline_checkpoint.json + torn attempts.jsonl)
    from research.harness.outcome_config import implementation_digest, profile_content_digest, profile_digest
    prof_path = tmp_path / "validated_profile.json"
    validated_prof = _validated_profile()
    prof_data = asdict(validated_prof)
    prof_data["settings"] = [dict(s) for s in prof_data["settings"]]
    prof_data["validation_refs"] = list(prof_data["validation_refs"])
    prof_path.write_text(_json.dumps(prof_data))

    # Placeholder validation_refs with NO --validation-report must fail BEFORE launching evaluator!
    with pytest.raises(SystemExit, match="verified live validation_report"):
        pilot.main([
            "--profile", str(prof_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(tmp_path / "pilot_no_report"),
        ])

    # Create a verified live validation report whose 18 captures replay cleanly on disk
    live_report_path = _write_verified_live_report(tmp_path / "cli_live_val", validated_prof)

    # 2b. Interruption during shared baseline eval 0 must NOT silently relaunch baseline on resume!
    interrupted_dir = tmp_path / "pilot_interrupted_baseline"
    baseline_calls = []

    def _interrupt_baseline(self, scenario):
        baseline_calls.append(scenario.scenario_id)
        raise KeyboardInterrupt("simulated abrupt exit during baseline eval 0")

    monkeypatch.setattr("research.harness.runner.HarnessRunner.run", _interrupt_baseline)
    with pytest.raises(KeyboardInterrupt):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(interrupted_dir),
        ])
    assert len(baseline_calls) == 1
    assert (interrupted_dir / "baseline_inflight.json").is_file()

    # Exclusive reservation helper prevents concurrent callers from both reserving the same slot!
    assert pilot._create_exclusive_json(interrupted_dir / "baseline_inflight.json", {"dup": True}) is False

    # Resume into interrupted_dir: must charge interrupted baseline and NOT relaunch it!
    monkeypatch.setattr("research.harness.runner.HarnessRunner.run", _fake_runner_run)
    rc_int = pilot.main([
        "--profile", str(prof_path),
        "--validation-report", str(live_report_path),
        "--family", "pedestrian_crossing",
        "--seed", "13",
        "--output-dir", str(interrupted_dir),
    ])
    assert rc_int == 1
    assert len(baseline_calls) == 1
    int_rep = _json.loads((interrupted_dir / "pilot_report.json").read_text())
    assert int_rep["status"] == "admission_rejected"
    assert "baseline_interrupted_before_completion" in int_rep["admission"]["reasons"]

    # 2c. If pilot_report.json was missing after a rejected baseline_checkpoint.json, resume must recreate it WITH physical_calls_charged!
    #     AND --plan-only must refuse to overwrite an output directory that already has pilot execution state!
    with pytest.raises(RuntimeError, match="Refusing --plan-only"):
        pilot.main([
            "--profile", str(prof_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(interrupted_dir),
            "--plan-only",
        ])
    (interrupted_dir / "pilot_report.json").unlink()
    rc_recreate = pilot.main([
        "--profile", str(prof_path),
        "--validation-report", str(live_report_path),
        "--family", "pedestrian_crossing",
        "--seed", "13",
        "--output-dir", str(interrupted_dir),
    ])
    assert rc_recreate == 1
    recreated_rep = _json.loads((interrupted_dir / "pilot_report.json").read_text())
    assert recreated_rep["physical_calls_charged"] == 1

    # 2d. Concurrent run_arm invocation targeting the same arm directory must fail immediately on flock without launching!
    import threading
    conc_arm_dir = tmp_path / "concurrent_arm_dir"
    conc_proto = pilot.load_protocol()
    conc_plan = pilot.build_matched_arms(conc_proto, validated_prof, "pedestrian_crossing", seed=13)["arms"][0]
    conc_plan["total_budget"] = 2
    started = threading.Event()
    release = threading.Event()
    conc_launches = []

    def _blocking_eval(arm, idx, cand):
        conc_launches.append(idx)
        started.set()
        release.wait(timeout=5.0)
        return {"execution_status": "completed", "outcome_v2": _baseline_outcome()}

    second_err = []

    def _second_caller():
        started.wait(timeout=5.0)
        try:
            pilot.run_arm(conc_plan, _blocking_eval, output_dir=conc_arm_dir)
        except RuntimeError as exc:
            second_err.append(str(exc))
        finally:
            release.set()

    t2 = threading.Thread(target=_second_caller)
    t2.start()
    res_first = pilot.run_arm(conc_plan, _blocking_eval, output_dir=conc_arm_dir)
    t2.join(timeout=5.0)
    assert len(second_err) == 1 and "lock" in second_err[0].lower()
    assert conc_launches == [0, 1]  # Exactly 2 physical calls (NEVER 3!)
    assert len(res_first["attempts"]) == 2

    pilot_out = tmp_path / "pilot_cli"
    rc1 = pilot.main([
        "--profile", str(prof_path),
        "--validation-report", str(live_report_path),
        "--family", "pedestrian_crossing",
        "--seed", "13",
        "--output-dir", str(pilot_out),
    ])
    assert rc1 == 0
    assert (pilot_out / "baseline_checkpoint.json").is_file()

    # Corrupt trailing line of one arm's attempts.jsonl and re-run CLI: must recover & NOT rerun baseline!
    arm_attempts = pilot_out / "arms" / "random" / "attempts.jsonl"
    lines = arm_attempts.read_text().splitlines()
    # Keep first 3 valid attempts + append torn 4th line
    arm_attempts.write_text("\n".join(lines[:3]) + '\n{"truncated":')

    calls_on_resume = []
    orig_run = _fake_runner_run

    def _counting_run(self, scenario):
        calls_on_resume.append(scenario.scenario_id)
        return orig_run(self, scenario)

    monkeypatch.setattr("research.harness.runner.HarnessRunner.run", _counting_run)
    rc2 = pilot.main([
        "--profile", str(prof_path),
        "--validation-report", str(live_report_path),
        "--family", "pedestrian_crossing",
        "--seed", "13",
        "--output-dir", str(pilot_out),
    ])
    assert rc2 == 0
    # Only the 2 remaining attempts of the 'random' arm (evals 3 & 4) were executed — baseline & other 3 arms skipped!
    assert len(calls_on_resume) == 2
    assert all("random" in sid for sid in calls_on_resume)

    # main() must hand the evaluator a private family copy (evaluator mutations cannot alter baseline/arm
    # admission rules) and must digest the SAME protocol bytes it parsed (no second file read).
    orig_make_eval = pilot._make_live_evaluator
    orig_file_digest = pilot.file_digest

    def _mutating_make_eval(**kw):
        fam_ref = kw["family"]
        inner = orig_make_eval(**kw)

        def _eval(*a, **kwa):
            res = inner(*a, **kwa)
            fam_ref.pop("intended_role", None)
            fam_ref["expected_predicates"] = []
            return {**res, "intended_witnessed": None}

        return _eval

    def _no_protocol_redigest(path):
        assert Path(path).resolve() != Path(pilot.DEFAULT_PROTOCOL).resolve(), "protocol must not be re-read for digest"
        return orig_file_digest(path)

    monkeypatch.setattr(pilot, "_make_live_evaluator", _mutating_make_eval)
    monkeypatch.setattr(pilot, "file_digest", _no_protocol_redigest)
    mut_out = tmp_path / "pilot_cli_mutating_family"
    cli_args = [
        "--profile", str(prof_path),
        "--validation-report", str(live_report_path),
        "--family", "pedestrian_crossing",
        "--seed", "13",
        "--output-dir", str(mut_out),
    ]
    # An unwitnessed baseline must be rejected under the ORIGINAL family rules, even though the
    # evaluator stripped intended_role/expected_predicates from the family object it was given.
    assert pilot.main(cli_args) == 1
    mut_rep = _json.loads((mut_out / "pilot_report.json").read_text())
    assert mut_rep["status"] == "admission_rejected"
    monkeypatch.setattr(pilot, "_make_live_evaluator", orig_make_eval)
    monkeypatch.setattr(pilot, "file_digest", orig_file_digest)
    proto_data, proto_dig = pilot._load_protocol_with_digest(pilot.DEFAULT_PROTOCOL)
    assert proto_dig == orig_file_digest(pilot.DEFAULT_PROTOCOL)
    assert proto_data == pilot.load_protocol()

    # Malformed LF-terminated final record (U+2028 prefix or bare CR between two objects) must NOT be treated
    # as a torn append: resume must reject before rewriting attempts.jsonl or launching another evaluation!
    full_lines = [ln for ln in arm_attempts.read_bytes().decode("utf-8").split("\n") if ln.strip(" \t\r")]
    assert len(full_lines) == 5
    for bad_final_bytes in (
        ("\n".join(full_lines[:4]) + "\n\u2028" + full_lines[4] + "\n").encode("utf-8"),
        ("\n".join(full_lines[:4]) + "\n" + full_lines[4] + "\r" + full_lines[4] + "\n").encode("utf-8"),
        ("\n".join(full_lines[:3]) + "\n" + full_lines[3].replace('"fitness":', '"bad_nan": NaN, "fitness":', 1) + "\n").encode("utf-8"),
        ("\n".join(full_lines[:3]) + "\n" + full_lines[3].replace('"fitness":', '"bad_inf": 1e999, "fitness":', 1) + "\n").encode("utf-8"),
    ):
        arm_attempts.write_bytes(bad_final_bytes)
        calls_on_corrupt = []
        monkeypatch.setattr("research.harness.runner.HarnessRunner.run", lambda self, s: calls_on_corrupt.append(s))
        with pytest.raises(ValueError, match="Corrupt checkpoint row"):
            pilot.main([
                "--profile", str(prof_path),
                "--validation-report", str(live_report_path),
                "--family", "pedestrian_crossing",
                "--seed", "13",
                "--output-dir", str(pilot_out),
            ])
        assert calls_on_corrupt == []
        assert arm_attempts.read_bytes() == bad_final_bytes
    # Cross-arm upfront validation: torn row in 'random' + (a) malformed LF-terminated row in 'critonly',
    # (b) invalid rng_state in 'critonly' attempt row, (c) invalid rng_state in 'critonly' inflight reservation,
    # (d) missing critonly/pilot_manifest.json, or (e) missing baseline_checkpoint.json with existing arm state
    # must ALL reject BEFORE rewriting 'random' or launching any evaluation!
    torn_random_bytes = ("\n".join(full_lines[:4]) + '\n{"truncated":').encode("utf-8")
    arm_attempts.write_bytes(torn_random_bytes)
    crit_dir = pilot_out / "arms" / "critonly"
    crit_attempts = crit_dir / "attempts.jsonl"
    orig_crit_bytes = crit_attempts.read_bytes()
    crit_lines = [ln for ln in orig_crit_bytes.decode("utf-8").split("\n") if ln.strip(" \t\r")]
    bad_crit_bytes = ("\n".join(crit_lines[:4]) + "\n\u2028" + crit_lines[4] + "\n").encode("utf-8")
    crit_attempts.write_bytes(bad_crit_bytes)
    calls_cross_arm = []
    monkeypatch.setattr("research.harness.runner.HarnessRunner.run", lambda self, s: calls_cross_arm.append(s))
    with pytest.raises(ValueError, match="Corrupt checkpoint row"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    assert calls_cross_arm == []
    assert arm_attempts.read_bytes() == torn_random_bytes
    assert crit_attempts.read_bytes() == bad_crit_bytes

    # (b) Invalid rng_state (bad length, negative/overflowing internal int, corrupt string gauss cache,
    #     AND complete corrupt row WITHOUT trailing newline) must all reject upfront without rewriting random!
    bad_rng_row = _json.loads(crit_lines[4])
    for bad_state in (
        [3, [0], None],
        [3, [0] * 624 + [624], None],
        [3, [0x7FFFFFFF] + [0] * 623 + [624], None],
        [3, [-1] + [0] * 624, None],
        [3, [(1 << 32) + 1] + [0] * 624, None],
        [3, [10**100] + [0] * 624, None],
        [3, [1] + [0] * 623 + [624], "corrupt_gauss"],
    ):
        bad_rng_row["rng_state"] = bad_state
        for trailing in ("\n", ""):
            crit_attempts.write_bytes(("\n".join(crit_lines[:4]) + "\n" + _json.dumps(bad_rng_row) + trailing).encode("utf-8"))
            with pytest.raises(ValueError, match="Corrupt rng_state"):
                pilot.main([
                    "--profile", str(prof_path),
                    "--validation-report", str(live_report_path),
                    "--family", "pedestrian_crossing",
                    "--seed", "13",
                    "--output-dir", str(pilot_out),
                ])
            assert calls_cross_arm == []
            assert arm_attempts.read_bytes() == torn_random_bytes
    # Complete overflowing float literal (1e999), trailing junk or trailing 0xC2 after completed row without LF,
    # unescaped CR at EOF inside string (b'{"name":"abc\r', b'{"name":"caf\xc3\r'), malformed mid-row syntax without LF,
    # missing required fields (rng_state, reused_shared_eval_zero, discovered_new), and out-of-sequence eval_index must ALL reject!
    _crit_row0 = _json.loads(crit_lines[0])
    _big_int_str = str(10**309)
    for bad_crit_payload in (
        ("\n".join(crit_lines[:4]) + "\n" + crit_lines[4].replace('"fitness":', '"bad_inf": 1e999, "fitness":', 1)).encode("utf-8"),
        ("\n".join(crit_lines[:4]) + "\n" + crit_lines[4] + " junk").encode("utf-8"),
        ("\n".join(crit_lines[:4]) + "\n" + crit_lines[4]).encode("utf-8") + b"\xc2",
        f'{{"n": {_big_int_str}e+'.encode("utf-8"),
        b'{"name":"abc\r',
        b'{"name":"caf\xc3\r',
        b'{"eval_index":0,"fitness":1 2',
        b'{"eval_index": 00',
        b'{"eval_index": 0, 2',
        b'{"eval_index": 0, tru',
        b'{"eval_index" 0',
        b"junk\xc3",
        b"{}\n",
        b'{"eval_index": 0}\n',
        (_json.dumps({**_crit_row0, "eval_index": 1}) + "\n").encode("utf-8"),
        (_json.dumps({**_crit_row0, "reused_shared_eval_zero": not _crit_row0["reused_shared_eval_zero"]}) + "\n").encode("utf-8"),
        (_json.dumps({k: v for k, v in _crit_row0.items() if k != "evidence_sha256"}) + "\n").encode("utf-8"),
        (_json.dumps({**_crit_row0, "evidence_sha256": "0" * 64}) + "\n").encode("utf-8"),
        (_json.dumps({**_crit_row0, "outcome_v2": _baseline_outcome(collision=True)}) + "\n").encode("utf-8"),
        (_json.dumps({**_crit_row0, "fitness": "bad"}) + "\n").encode("utf-8"),
        (_json.dumps({**_crit_row0, "fitness": 999.0}) + "\n").encode("utf-8"),
        (_json.dumps({**_crit_row0, "fitness": 10**400}) + "\n").encode("utf-8"),
        (_json.dumps({**_crit_row0, "candidate": {"trigger_radius_m": "bad"}}) + "\n").encode("utf-8"),
        (_json.dumps({**_crit_row0, "candidate": {"trigger_radius_m": 30.0}}) + "\n").encode("utf-8"),
        (_json.dumps({**_crit_row0, "candidate": {}}) + "\n").encode("utf-8"),
        (_json.dumps({**_crit_row0, "candidate": {"trigger_radius_m": 20.0, "extra_dim": 1.0}}) + "\n").encode("utf-8"),
        (_json.dumps({**_crit_row0, "candidate": {"trigger_radius_m": 9999.0}}) + "\n").encode("utf-8"),
        (_json.dumps({**_crit_row0, "intended_witnessed": 1}) + "\n").encode("utf-8"),
        (_json.dumps({**_crit_row0, "engine_run_obligations": ["fake_ob"]}) + "\n").encode("utf-8"),
        (_json.dumps({**_crit_row0, "run_error": "fake_error"}) + "\n").encode("utf-8"),
        (_json.dumps({**_crit_row0, "feedback_version": "legacy_v1"}) + "\n").encode("utf-8"),
        (crit_lines[0] + "\n" + _json.dumps({**_json.loads(crit_lines[1]), "outcome_v2": {**_baseline_outcome(), "categories": None}}) + "\n").encode("utf-8"),
        (crit_lines[0] + "\n" + _json.dumps({**_json.loads(crit_lines[1]), "outcome_v2": {**_baseline_outcome(), "categories": {**_baseline_outcome()["categories"], "safety": None}}}) + "\n").encode("utf-8"),
        (crit_lines[0] + "\n" + _json.dumps({**_json.loads(crit_lines[1]), "outcome_v2": {**_baseline_outcome(), "categories": {**_baseline_outcome()["categories"], "rules": None}}}) + "\n").encode("utf-8"),
        (crit_lines[0] + "\n" + _json.dumps({**_json.loads(crit_lines[1]), "outcome_v2": {**_baseline_outcome(), "schema_version": 2.0}}) + "\n").encode("utf-8"),
        (crit_lines[0] + "\n" + _json.dumps({**_json.loads(crit_lines[1]), "discovered_new": True, "credited_as_discovery": True, "new_outcomes": ["safety.collision"]}) + "\n").encode("utf-8"),
        (_json.dumps({k: v for k, v in _crit_row0.items() if k != "rng_state"}) + "\n").encode("utf-8"),
        (_json.dumps({k: v for k, v in _crit_row0.items() if k != "reused_shared_eval_zero"}) + "\n").encode("utf-8"),
        (_json.dumps({k: v for k, v in _crit_row0.items() if k != "discovered_new"}) + "\n").encode("utf-8"),
    ):
        crit_attempts.write_bytes(bad_crit_payload)
        with pytest.raises(ValueError, match="Corrupt (checkpoint row|candidate dimension)"):
            pilot.main([
                "--profile", str(prof_path),
                "--validation-report", str(live_report_path),
                "--family", "pedestrian_crossing",
                "--seed", "13",
                "--output-dir", str(pilot_out),
            ])
        assert calls_cross_arm == []
        assert arm_attempts.read_bytes() == torn_random_bytes
    # Genuine mid-float, negative-sign, \uXXXX escape, inside-string UTF-8, huge JSON int prefix (10**309 ),
    # negative-exponent prefix (<10**309>e-0), and 4301-digit mantissa + 'e' EOF truncations MUST recover!
    _ped_base_spec = {"controller_params": {"trigger_radius_m": 20.0}}
    for valid_torn in (
        b'{"fitness": 0.',
        b'{"fitness": 1e-',
        b'{"fitness": -',
        b'{"a": "\\u12',
        b'{"a": "caf\xc3',
        f'{{"n": {_big_int_str} '.encode("utf-8"),
        f'{{"n": {_big_int_str}e-0'.encode("utf-8"),
        f'{{"n": {"1" * 4301}e'.encode("utf-8"),
    ):
        crit_attempts.write_bytes(("\n".join(crit_lines[:4]) + "\n").encode("utf-8") + valid_torn)
        parsed_t, _vlines_t, is_torn_t, _rb_t = pilot._validate_arm_checkpoint(
            _json.loads((crit_dir / "pilot_manifest.json").read_text()),
            crit_dir,
            base_spec=_ped_base_spec,
        )
        assert is_torn_t is True and len(parsed_t) == 4
    # Budget reporting must count failures recorded only in outcome_v2.execution_status (without top-level status or run_error)!
    assert pilot.record_budget(
        [{"eval_index": 0, "reused_shared_eval_zero": False, "outcome_v2": {"execution_status": "failed"}}],
        shared_eval_zero=False,
    )["failed_attempts_charged"] == 1
    crit_attempts.write_bytes(orig_crit_bytes)

    # (c) Invalid inflight reservation ({}, bare {"eval_index": 0}, missing rng_state, reused/discovered flags, bad rng_state) must reject upfront!
    crit_inflight = crit_dir / "inflight_attempt.json"
    _valid_rng = list(_json.loads(crit_lines[4])["rng_state"])
    _valid_inf_base = {
        "arm": "critonly", "policy": "critonly", "eval_index": 4,
        "candidate": {"trigger_radius_m": 20.0},
        "execution_status": "interrupted", "run_error": "interrupted_before_completion",
        "reused_shared_eval_zero": False, "eligible_for_discovery": True,
        "discovered_new": False, "credited_as_discovery": False, "new_outcomes": [],
        "outcome_v2": None, "feedback_version": "v2_normalized",
        "criticality": None, "fitness": None, "tiebreak": None,
        "fallback_selection": True,
        "rng_state": _valid_rng,
    }
    for bad_inf, err_pat in (
        ({}, "Corrupt inflight reservation"),
        ({"eval_index": 0}, "Corrupt inflight reservation"),
        ({**_valid_inf_base, "eval_index": 0}, "Corrupt inflight reservation"),
        ({**_valid_inf_base, "rng_state": None}, "Corrupt inflight reservation"),
        ({**_valid_inf_base, "reused_shared_eval_zero": True}, "Corrupt inflight reservation"),
        ({**_valid_inf_base, "discovered_new": True}, "Corrupt inflight reservation"),
        ({**_valid_inf_base, "fitness": 17.0}, "Corrupt inflight reservation"),
        ({**_valid_inf_base, "criticality": 0.75, "fitness": 0.75, "fallback_selection": False}, "Corrupt inflight reservation"),
        ({**_valid_inf_base, "eligible_for_discovery": False}, "Corrupt inflight reservation"),
        ({**_valid_inf_base, "fallback_selection": False}, "Corrupt inflight reservation"),
        ({**_valid_inf_base, "feedback_version": "legacy_v1"}, "Corrupt inflight reservation"),
        ({k: v for k, v in _valid_inf_base.items() if k != "fitness"}, "Corrupt inflight reservation"),
        ({k: v for k, v in _valid_inf_base.items() if k != "outcome_v2"}, "Corrupt inflight reservation"),
        ({**_valid_inf_base, "candidate": {"trigger_radius_m": 9999.0}}, "Corrupt inflight reservation"),
        ({**_valid_inf_base, "rng_state": [3, [0], None]}, "Corrupt rng_state"),
        ({**_valid_inf_base, "rng_state": [3, [(1 << 32) + 1] + [0] * 624, None]}, "Corrupt rng_state"),
        ({**_valid_inf_base, "rng_state": [3, [0] * 624 + [624], "bad"]}, "Corrupt rng_state"),
    ):
        crit_inflight.write_text(_json.dumps(bad_inf))
        with pytest.raises(ValueError, match=err_pat):
            pilot.main([
                "--profile", str(prof_path),
                "--validation-report", str(live_report_path),
                "--family", "pedestrian_crossing",
                "--seed", "13",
                "--output-dir", str(pilot_out),
            ])
        assert calls_cross_arm == []
        assert arm_attempts.read_bytes() == torn_random_bytes
    crit_inflight.unlink()

    # (d) Missing/mismatched critonly/pilot_manifest.json, cross-arm manifest, direct run_arm budget inflation,
    #     invalid shared_eval_zero_hash (even with 0 rows), build_matched_arms empty shared hash, and NaN evaluator output!
    crit_manifest = crit_dir / "pilot_manifest.json"
    orig_crit_manifest = crit_manifest.read_bytes()
    crit_plan_dict = _json.loads(orig_crit_manifest.decode("utf-8"))
    orig_base_row = _json.loads((pilot_out / "baseline_checkpoint.json").read_text())["baseline_row"]
    for bad_shared in ({}, {"evidence_sha256": "   "}):
        with pytest.raises(ValueError, match="non-empty"):
            pilot.build_matched_arms(pilot.load_protocol(), _validated_profile(), "pedestrian_crossing", seed=13, shared_eval_zero=bad_shared)
    with pytest.raises(ValueError, match="invalid shared_eval_zero_hash"):
        pilot._validate_arm_checkpoint({**crit_plan_dict, "shared_eval_zero_hash": "  "}, tmp_path / "empty_bad_hash_arm")
    with pytest.raises(ValueError, match="requires shared_eval_zero_row"):
        pilot.run_arm(crit_plan_dict, lambda *a, **kw: calls_cross_arm.append("missing_shared"), output_dir=tmp_path / "missing_shared_arm")
    with pytest.raises(ValueError, match="does not match arm_plan shared_eval_zero_hash"):
        pilot.run_arm(
            crit_plan_dict,
            lambda *a, **kw: calls_cross_arm.append("wrong_shared"),
            shared_eval_zero_row={"evidence_sha256": "wrong"},
            output_dir=tmp_path / "wrong_shared_arm",
        )
    # Caller inflating total_budget while keeping manifest_digest must NOT launch eval 5!
    with pytest.raises(ValueError, match="Incompatible pilot checkpoint manifest"):
        pilot.run_arm(
            {**crit_plan_dict, "total_budget": int(crit_plan_dict["total_budget"]) + 1},
            lambda *a, **kw: calls_cross_arm.append("budget_inflate"),
            shared_eval_zero_row=orig_base_row,
            output_dir=crit_dir,
            base_spec=_ped_base_spec,
        )
    # Cross-arm manifest (random manifest inside critonly directory) and JSON float/int type change (5 -> 5.0) must reject!
    cross_arm_dir = tmp_path / "cross_arm_manifest"
    cross_arm_dir.mkdir()
    (cross_arm_dir / "pilot_manifest.json").write_text(_json.dumps({**crit_plan_dict, "arm": "random"}))
    with pytest.raises(ValueError, match="Incompatible pilot checkpoint manifest"):
        pilot._validate_arm_checkpoint(crit_plan_dict, cross_arm_dir, base_spec=_ped_base_spec)
    (cross_arm_dir / "pilot_manifest.json").write_text(_json.dumps({**crit_plan_dict, "total_budget": float(crit_plan_dict["total_budget"])}))
    with pytest.raises(ValueError, match="Incompatible pilot checkpoint manifest"):
        pilot._validate_arm_checkpoint(crit_plan_dict, cross_arm_dir, base_spec=_ped_base_spec)
    # Direct run_arm() resume with changed effective SearchSpace bounds (high=30 -> high=100) must reject!
    from research.harness.search_space import make_legacy_space
    space_dir = tmp_path / "space_bound_arm"
    space_plan = {k: v for k, v in crit_plan_dict.items() if k != "search_space_signature"}
    space_plan["shared_eval_zero_hash"] = None
    space_plan["realized_state_digest"] = None
    space_plan["total_budget"] = 2
    space_30 = make_legacy_space(low=10.0, high=30.0, seed=13)
    space_100 = make_legacy_space(low=10.0, high=100.0, seed=13)
    pilot.run_arm(
        space_plan,
        lambda *a, **kw: {"execution_status": "completed", "outcome_v2": _baseline_outcome()},
        output_dir=space_dir,
        search_space=space_30,
    )
    with pytest.raises(ValueError, match="Incompatible pilot checkpoint manifest"):
        pilot.run_arm(
            space_plan,
            lambda *a, **kw: calls_cross_arm.append("space_drift"),
            output_dir=space_dir,
            search_space=space_100,
        )
    # Custom SearchSpace edge cases: [0.0, 0.96] decimals=1, [0.0, 0.06] step=0.01 decimals=1,
    # [0.0, 0.98] step=0.01 decimals=1, [0.91, 0.94] decimals=1, int stepped bounds, OverflowError on 10**400,
    # keyword-only candidate evaluator, and semantic tuple obligations normalized before scoring!
    from research.harness.search_space import SearchDimension, SearchSpace, SearchSpaceError
    import random as _random
    for sp_idx, (sp_obj, sp_seed, sp_base) in enumerate((
        (make_legacy_space(low=0.0, high=0.96, decimals=1, seed=153), 153, {"controller_params": {"trigger_radius_m": 0.96}}),
        (make_legacy_space(low=0.0, high=0.06, step=0.01, decimals=1, seed=153), 153, None),
        (make_legacy_space(low=0.0, high=0.98, step=0.01, decimals=1, seed=52), 52, None),
        (make_legacy_space(low=0.91, high=0.94, decimals=1, seed=0), 0, None),
    )):
        b_plan = {**space_plan, "seed": sp_seed, "total_budget": 3}
        b_dir = tmp_path / f"bounded_decimals_arm_{sp_idx}"
        res_b = pilot.run_arm(
            b_plan,
            lambda *a, **kw: {"execution_status": "completed", "outcome_v2": _baseline_outcome()},
            output_dir=b_dir,
            search_space=sp_obj,
            base_spec=sp_base,
        )
        dim0 = sp_obj.dimensions()[0]
        assert all(dim0.in_bounds(row["candidate"]["trigger_radius_m"]) for row in res_b["attempts"])
        res_b2 = pilot.run_arm(
            b_plan,
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not re-run")),
            output_dir=b_dir,
            search_space=sp_obj,
            base_spec=sp_base,
        )
        assert len(res_b2["attempts"]) == 3
    for int_dim in (
        SearchDimension("k", "controller_params.k", low=0.0, high=1.8, kind="int", step=1.6),
        SearchDimension("k", "controller_params.k", low=1.2, high=3.2, kind="int", step=2.0),
    ):
        rng_int = _random.Random(1)
        for _ in range(20):
            assert int_dim.in_bounds(int_dim.sample_value(rng_int))
        assert int_dim.in_bounds(10**400) is False
        assert isinstance(int_dim.coerce(10**400), int)
        with pytest.raises(SearchSpaceError):
            int_dim.validate(10**400)
        with pytest.raises(SearchSpaceError):
            int_dim.coerce(True)
    with pytest.raises(SearchSpaceError):
        SearchDimension("x", "controller_params.x", low=0.0, high=1.0).coerce(10**400)
    # Idempotency of SearchSpace.validate() on stepped float midpoints and fine grids (step<=1e-9),
    # fast O(log max_k) fallback on step=1e-12, key() consistency, overflow/empty-int construction guards,
    # bool/OverflowError rejection in clamp(), and staging offset round-trip!
    for idemp_space, test_val in (
        (make_legacy_space(low=0.15, high=0.35, step=0.1, decimals=1, seed=13), 0.31),
        (make_legacy_space(low=3.95, high=7.95, step=0.1, decimals=1, seed=13), 5.64),
        (make_legacy_space(low=0.91, high=0.94, step=1e-12, decimals=1, seed=13), 0.92),
        (make_legacy_space(low=0.0, high=1.0, step=1e-10, decimals=12, seed=13), 0.5),
        (make_legacy_space(low=0.0, high=1.0, step=1e-9, decimals=9, seed=13), 0.5),
    ):
        v1 = idemp_space.validate({"trigger_radius_m": test_val})
        v2 = idemp_space.validate(v1.to_dict())
        assert v1.to_dict() == v2.to_dict()
        if test_val == 0.5:
            assert v1.to_dict()["trigger_radius_m"] == 0.5
        with pytest.raises(SearchSpaceError):
            idemp_space.clamp({"trigger_radius_m": True})
        with pytest.raises(SearchSpaceError):
            idemp_space.clamp({"trigger_radius_m": 10**400})
    # SearchSpace.key() must distinguish distinct canonical grid points (no dedup collapse)!
    sp_key_check = make_legacy_space(low=0.91, high=0.94, decimals=1, seed=13)
    assert sp_key_check.key({"trigger_radius_m": 0.92}) != sp_key_check.key({"trigger_radius_m": 0.93})
    sp_key_step = make_legacy_space(low=3.95, high=7.95, step=0.1, decimals=1, seed=13)
    assert sp_key_step.key(sp_key_step.validate({"trigger_radius_m": 4.4})) != sp_key_step.key(sp_key_step.validate({"trigger_radius_m": 4.5}))
    # Dimension construction must reject empty integer range (even within 1e-9 of an int!),
    # float-overflowing span, and sub-ULP step overflow; tiny stepped upper bound [0, 1.99e-10] must never emit 2e-10!
    for bad_dim_kw in (
        dict(name="k", spec_path="p", low=0.2, high=0.3, kind="int"),
        dict(name="k", spec_path="p", low=0.9999999995, high=0.9999999999, kind="int"),
        dict(name="x", spec_path="p", low=0.0, high=1.0, step=1e-310),
        dict(name="x", spec_path="p", low=-1e308, high=1e308, step=1.0),
    ):
        with pytest.raises(SearchSpaceError):
            SearchDimension(**bad_dim_kw)
    tiny_sp = make_legacy_space(low=0.0, high=1.99e-10, step=1e-10, decimals=12, seed=13)
    assert tiny_sp.validate({"trigger_radius_m": 1.99e-10}).to_dict()["trigger_radius_m"] == 1e-10
    # Stepped endpoint rounded into bounds (0..1 step=0.51 decimals=1 -> 1.0), fixed float (1e-13),
    # exact integer > 2**53 containment, and decimals=309 fallback without OverflowError!
    ep_sp = make_legacy_space(low=0.0, high=1.0, step=0.51, decimals=1, seed=13)
    assert ep_sp.validate({"trigger_radius_m": 1.0}).to_dict()["trigger_radius_m"] == 1.0
    fixed_sp = make_legacy_space(low=1e-13, high=1e-13, decimals=12, seed=13)
    assert fixed_sp.validate({"trigger_radius_m": 1e-13}).to_dict()["trigger_radius_m"] == 1e-13
    assert fixed_sp.sample(_random.Random(13)).to_dict()["trigger_radius_m"] == 1e-13
    big_int_dim = SearchDimension("k", "p", low=2**53, high=2**53, kind="int")
    assert big_int_dim.in_bounds(2**53) is True
    assert big_int_dim.in_bounds(2**53 + 1) is False
    import sys as _sys
    big_int_odd = make_legacy_space(low=2**53 + 1, high=2**53 + 1, kind="int", step=0.5)
    assert big_int_odd.dimensions()[0].in_bounds(float(2**53)) is False
    assert big_int_odd.validate({"trigger_radius_m": 2**53 + 1}).to_dict()["trigger_radius_m"] == 2**53 + 1
    from decimal import Decimal as _Decimal
    from fractions import Fraction as _Fraction
    for float_step in (0.5, 1.0):
        wide_big_int = SearchSpace([SearchDimension("k", "p", low=0, high=2**53 + 2, kind="int", step=float_step)])
        assert wide_big_int.dimensions()[0].coerce(2**53 + 1) == 2**53 + 1
        assert wide_big_int.validate({"k": 2**53 + 1}).to_dict()["k"] == 2**53 + 1
    exact_int_sp = SearchSpace([SearchDimension("k", "p", low="0", high=str(2**53 + 2), kind="int", step="1")])
    for exact_raw in (_Fraction(2**53 + 1), _Decimal(str(2**53 + 1)), str(2**53 + 1)):
        assert exact_int_sp.clamp({"k": exact_raw}).to_dict()["k"] == 2**53 + 1
        assert exact_int_sp.dimensions()[0].in_bounds(exact_raw) is True
        assert exact_int_sp.validate({"k": exact_raw}).to_dict()["k"] == 2**53 + 1
    assert exact_int_sp.clamp({"k": _Decimal("NaN")}).to_dict()["k"] == exact_int_sp.dimensions()[0].missing_value()
    # Non-integral Fraction/Decimal/str bounds and defaults on kind="int" must preserve exact rational precision!
    frac_lo_dim = SearchDimension("k", "p", low=_Decimal("9007199254740992.25"), high=_Decimal("9007199254740994"), kind="int")
    assert frac_lo_dim.in_bounds(9007199254740992) is False
    assert frac_lo_dim.in_bounds(9007199254740993) is True
    frac_sp = SearchSpace([frac_lo_dim])
    assert SearchSpace.from_dict(frac_sp.to_dict()).signature() == frac_sp.signature()
    with pytest.raises(SearchSpaceError):
        SearchDimension("k", "p", low=_Decimal("9007199254740992.5"), high=_Decimal("9007199254740992.8"), kind="int")
    assert SearchDimension("k", "p", low=0, high=10, kind="int", default=_Decimal("0.5000000000000000001")).missing_value() == 1
    assert SearchDimension("k", "p", low=2**53, high=2**53 + 2, kind="int", default=_Decimal("9007199254740993.2")).missing_value() == 2**53 + 1
    # SearchSpace.from_dict() and SearchSpace.__init__() must not silently cast invalid decimals or seed,
    # float dimensions must agree on numeric strings ("0.1" accepted, "1/2" rejected), and huge-denominator
    # exact rationals like Decimal("1e-5000") must raise SearchSpaceError rather than failing later in describe()!
    for bad_dec in (1.5, True, -1):
        with pytest.raises(SearchSpaceError):
            SearchSpace.from_dict({"dimensions": [{"name": "x", "spec_path": "p", "low": 0.0, "high": 1.0, "decimals": bad_dec}]})
    for bad_seed in (True, 1.5):
        with pytest.raises(SearchSpaceError):
            SearchSpace.from_dict({"seed": bad_seed, "dimensions": [{"name": "x", "spec_path": "p", "low": 0.0, "high": 1.0}]})
        with pytest.raises(SearchSpaceError):
            SearchSpace([SearchDimension("x", "p", low=0.0, high=1.0)], seed=bad_seed)
    fixed_pt1 = SearchSpace([SearchDimension("x", "p", low=0.1, high=0.1)])
    assert fixed_pt1.dimensions()[0].in_bounds("0.1") is True
    assert fixed_pt1.dimensions()[0].coerce("0.1") == 0.1
    assert fixed_pt1.clamp({"x": "0.1"}).to_dict()["x"] == 0.1
    assert fixed_pt1.validate({"x": "0.1"}).to_dict()["x"] == 0.1
    unit_fl = SearchSpace([SearchDimension("x", "p", low=0.0, high=1.0)])
    assert unit_fl.dimensions()[0].in_bounds("1/2") is False
    with pytest.raises(SearchSpaceError):
        unit_fl.dimensions()[0].coerce("1/2")
    with pytest.raises(SearchSpaceError):
        unit_fl.clamp({"x": "1/2"})
    with pytest.raises(SearchSpaceError):
        unit_fl.validate({"x": "1/2"})
    with pytest.raises(SearchSpaceError):
        SearchDimension("k", "p", low=0, high=10, kind="int", default=_Decimal("1e-5000"))
    with pytest.raises(SearchSpaceError):
        SearchDimension("k", "p", low=10**5000, high=10**5000 + 1, kind="int")
    with pytest.raises(SearchSpaceError):
        SearchDimension("x", "p", low=10**5000, high=10**5000 + 1)
    with pytest.raises(SearchSpaceError):
        SearchDimension("x", "p", low=0.0, high=1.0, decimals=10**5000)
    with pytest.raises(SearchSpaceError):
        SearchDimension("x", "p", low=0.0, high=1.0, decimals=-(10**5000))
    with pytest.raises(SearchSpaceError):
        SearchDimension("x", "p", low=0.0, high=1.0, kind=10**5000)  # type: ignore[arg-type]
    with pytest.raises(SearchSpaceError):
        SearchDimension(10**5000, "p", low=0.0, high=1.0)  # type: ignore[arg-type]
    with pytest.raises(SearchSpaceError):
        SearchDimension("x", 10**5000, low=0.0, high=1.0)  # type: ignore[arg-type]
    with pytest.raises(SearchSpaceError):
        SearchSpace([SearchDimension("x", "p", low=0.0, high=1.0)], name=10**5000)  # type: ignore[arg-type]
    with pytest.raises(SearchSpaceError):
        make_legacy_space(spec_path=10**5000)  # type: ignore[arg-type]
    for bad_from_dict in (
        {"dimensions": [{"name": None, "spec_path": "p", "low": 0.0, "high": 1.0}]},
        {"dimensions": [{"name": 10**5000, "spec_path": "p", "low": 0.0, "high": 1.0}]},
        {"dimensions": [{"name": "x", "spec_path": "p", "low": 0.0, "high": 1.0, "kind": 10**5000}]},
    ):
        with pytest.raises(SearchSpaceError):
            SearchSpace.from_dict(bad_from_dict)
    with pytest.raises(SearchSpaceError):
        unit_fl.validate({1: 0.5, "x": 0.5})  # type: ignore[dict-item]
    with pytest.raises(SearchSpaceError):
        unit_fl.clamp({1: 0.5, "x": 0.5})  # type: ignore[dict-item]
    with pytest.raises(SearchSpaceError):
        unit_fl.dimension(10**5000)  # type: ignore[arg-type]
    with pytest.raises(SearchSpaceError):
        SearchSpace([SearchDimension("x", "p", low=0.0, high=1.0)], seed=10**5000)
    assert SearchDimension("x", "p", low=0.0, high=1.0, step=0.1, decimals=10**309)._rounding_pad() == 0.0
    # load_protocol() must reject float total_evaluations=2.9, duplicate/typo arms, and malformed admission fields!
    base_proto = pilot.load_protocol()
    for bad_proto in (
        {**base_proto, "budget_per_arm": {**base_proto["budget_per_arm"], "total_evaluations": 2.9}},
        {**base_proto, "arms": ["random", "random"]},
        {**base_proto, "arms": ["random", "random "]},
        {**base_proto, "require_collision_free_baseline": []},
        {**base_proto, "required_channels": False},
        {**base_proto, "shared_eval_zero_policy": "bogus"},
        {**base_proto, "shared_eval_zero_policy": []},
        {**base_proto, "shared_eval_zero_policy": {}},
    ):
        bp_path = tmp_path / "bad_proto.json"
        bp_path.write_text(_json.dumps(bad_proto))
        with pytest.raises(ValueError):
            pilot.load_protocol(bp_path)
    with pytest.raises(ValueError):
        pilot.build_matched_arms({**base_proto, "arms": None}, _validated_profile(), "pedestrian_crossing", 13)
    with pytest.raises(ValueError):
        pilot.build_matched_arms(
            {**base_proto, "budget_per_arm": {**base_proto["budget_per_arm"], "total_evaluations": 2.9}},
            _validated_profile(),
            "pedestrian_crossing",
            13,
        )
    with pytest.raises(ValueError, match="Unsupported pilot policy arm"):
        pilot._validate_arm_checkpoint({**space_plan, "arm": "random "}, tmp_path / "empty_bad_arm_dir")
    # Upfront arm_plan validation must reject invalid total_budget, seed (incl. 1100-deep list), missing/empty manifest_digest, and empty digests!
    deep_seed: object = 1
    for _ in range(1100):
        deep_seed = [deep_seed]
    for bad_arm_plan in (
        {**space_plan, "total_budget": 2.9},
        {**space_plan, "total_budget": 1},
        {**space_plan, "total_budget": True},
        {**space_plan, "total_budget": -1},
        {**space_plan, "seed": True},
        {**space_plan, "seed": 13.7},
        {**space_plan, "seed": deep_seed},
        {k: v for k, v in space_plan.items() if k != "manifest_digest"},
        {**space_plan, "manifest_digest": ""},
        {**space_plan, "realized_state_digest": ""},
        {**space_plan, "profile_digest": ""},
    ):
        with pytest.raises(ValueError):
            pilot._validate_arm_checkpoint(bad_arm_plan, tmp_path / "empty_bad_plan_dir")
        with pytest.raises(ValueError):
            pilot.run_arm(bad_arm_plan, lambda *a, **kw: {"execution_status": "completed", "outcome_v2": _baseline_outcome()})
    # _has_existing_arms_state must be iterative (no RecursionError on 1100-deep nested dirs)
    # and validate_initial_configuration must not hit RecursionError when formatting 1100-deep status lists!
    deep_dirs_root = tmp_path / "deep_arms_tree"
    cur_d = deep_dirs_root / "arms"
    cur_d.mkdir(parents=True)
    created_deep_dirs = [cur_d]
    try:
        for _ in range(1100):
            cur_d = cur_d / "a"
            cur_d.mkdir()
            created_deep_dirs.append(cur_d)
        assert pilot._has_existing_arms_state(deep_dirs_root) is False
    finally:
        for d_path in reversed(created_deep_dirs):
            d_path.rmdir()
        deep_dirs_root.rmdir()
    deep_status: object = "x"
    for _ in range(1100):
        deep_status = [deep_status]
    adm_deep_st = pilot.validate_initial_configuration(
        {**_baseline_outcome(), "execution_status": deep_status},
        base_proto,
        profile=_validated_profile(),
        realized_state_digest="abc",
        intended_witnessed=True,
    )
    assert adm_deep_st["admitted"] is False
    # _strict_json_equal must be iterative (no RecursionError on 600-nested arrays)!
    deep_a: object = 1
    deep_b: object = 1
    for _ in range(600):
        deep_a = [deep_a]
        deep_b = [deep_b]
    assert pilot._strict_json_equal(deep_a, deep_b) is True
    # validate_initial_configuration() must fail closed on malformed require_collision_free_baseline / required_channels (incl. tuple)!
    col_out = _baseline_outcome()
    col_out["categories"]["safety"]["collision"] = {"status": "present", "event_ids": ["e1"], "reasons": []}
    adm_bad_proto = pilot.validate_initial_configuration(
        col_out,
        {**base_proto, "require_collision_free_baseline": [], "required_channels": ()},
        profile=_validated_profile(),
        realized_state_digest="abc",
        intended_witnessed=True,
    )
    assert adm_bad_proto["admitted"] is False
    assert "invalid_protocol_require_collision_free_baseline" in adm_bad_proto["reasons"]
    assert "invalid_protocol_required_channels" in adm_bad_proto["reasons"]
    # Deeply nested corrupt JSON in pilot_report.json / attempts.jsonl must not raise RecursionError!
    deep_rep_dir = tmp_path / "deep_report_dir"
    deep_rep_dir.mkdir()
    (deep_rep_dir / "pilot_report.json").write_text("[" * 1100)
    assert pilot._has_existing_execution_state(deep_rep_dir) is True
    deep_arm_dir = tmp_path / "deep_arm_dir"
    deep_plan = {**space_plan, "arm": "random", "total_budget": 2}
    pilot.run_arm(deep_plan, lambda *a, **kw: {"execution_status": "completed", "outcome_v2": _baseline_outcome()}, output_dir=deep_arm_dir, base_spec=_ped_base_spec)
    (deep_arm_dir / "attempts.jsonl").write_text('{"a":' * 1100)
    with pytest.raises(ValueError, match="Corrupt checkpoint row"):
        pilot._validate_arm_checkpoint(deep_plan, deep_arm_dir, base_spec=_ped_base_spec)
    # Independent eval-0 (share_eval_zero=False) must reject mismatched realized_state_digest, colliding,
    # partial, or failed eval-0 AND resuming into that directory must NOT advance past eval 0!
    indep_plan = {
        **space_plan,
        "arm": "critonly",
        "total_budget": 2,
        "shared_eval_zero_hash": None,
        "realized_state_digest": "admitted-digest-123",
    }
    indep_ctx = (base_proto, _validated_profile(), base_proto["interaction_families"][0])
    with pytest.raises(ValueError, match="independent eval-0 realized_state_digest"):
        pilot.run_arm(
            indep_plan,
            lambda *a, **kw: {
                "execution_status": "completed",
                "outcome_v2": _baseline_outcome(),
                "realized_state_digest": "different-digest-999",
                "intended_witnessed": True,
            },
            output_dir=tmp_path / "indep_bad_digest",
            base_spec=_ped_base_spec,
            baseline_admission_context=indep_ctx,
        )
    indep_col_dir = tmp_path / "indep_bad_col"
    with pytest.raises(ValueError, match="independent eval-0 failed baseline admission"):
        pilot.run_arm(
            indep_plan,
            lambda *a, **kw: {
                "execution_status": "completed",
                "outcome_v2": col_out,
                "realized_state_digest": "admitted-digest-123",
                "intended_witnessed": True,
            },
            output_dir=indep_col_dir,
            base_spec=_ped_base_spec,
            baseline_admission_context=indep_ctx,
        )
    # Resuming after a rejected independent eval-0 must refuse to advance to eval 1 or relaunch eval 0
    # BOTH with AND without baseline_admission_context!
    for ctx_arg, err_pat in (
        (indep_ctx, "refusing to advance arm past unadmitted baseline"),
        (None, "Incompatible pilot checkpoint manifest"),
    ):
        with pytest.raises(ValueError, match=err_pat):
            pilot.run_arm(
                indep_plan,
                lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not call evaluator on rejected eval-0 resume")),
                output_dir=indep_col_dir,
                base_spec=_ped_base_spec,
                baseline_admission_context=ctx_arg,
            )
    # Tampered extra key (e.g. realized_state_digest) in inflight_attempt.json must reject as corrupt!
    tampered_inf_path = indep_col_dir / "inflight_attempt.json"
    tampered_inf = _json.loads(tampered_inf_path.read_text())
    tampered_inf["realized_state_digest"] = "wrong-digest-999"
    tampered_inf_path.write_text(_json.dumps(tampered_inf))
    with pytest.raises(ValueError, match="Corrupt inflight reservation"):
        pilot._validate_arm_checkpoint(
            indep_plan,
            indep_col_dir,
            base_spec=_ped_base_spec,
            baseline_admission_context=indep_ctx,
        )
    # Passing baseline_admission_context with an unpinned independent arm plan must reject!
    with pytest.raises(ValueError, match="requires a pinned realized_state_digest"):
        pilot.run_arm(
            {**indep_plan, "realized_state_digest": None},
            lambda *a, **kw: {"execution_status": "completed", "outcome_v2": _baseline_outcome()},
            output_dir=tmp_path / "indep_unpinned",
            base_spec=_ped_base_spec,
            baseline_admission_context=indep_ctx,
        )
    # Pinned independent eval-0 with baseline_admission_context=None must STILL reject failed/interrupted eval-0!
    with pytest.raises(ValueError, match="independent eval-0 realized_state_digest"):
        pilot.run_arm(
            indep_plan,
            lambda *a, **kw: {"execution_status": "failed", "outcome_v2": None, "realized_state_digest": None, "run_error": "boom"},
            output_dir=tmp_path / "indep_pinned_failed_no_ctx",
            base_spec=_ped_base_spec,
        )
    # Manifest binds baseline_admission_context: crashing after writing pilot_manifest.json (before eval-0 reservation)
    # and resuming WITHOUT baseline_admission_context must reject as incompatible manifest!
    pre_res_dir = tmp_path / "indep_pre_res_manifest"
    pre_res_dir.mkdir()
    pre_space = pilot._default_pilot_space(_ped_base_spec, int(indep_plan["seed"]))
    pre_manifest = pilot._expected_arm_manifest(indep_plan, pre_space, baseline_admission_context=indep_ctx)
    (pre_res_dir / "pilot_manifest.json").write_text(_json.dumps(pre_manifest))
    with pytest.raises(ValueError, match="Incompatible pilot checkpoint manifest"):
        pilot.run_arm(
            indep_plan,
            lambda *a, **kw: {"execution_status": "completed", "outcome_v2": col_out, "realized_state_digest": "admitted-digest-123"},
            output_dir=pre_res_dir,
            base_spec=_ped_base_spec,
        )
    # Shared eval-0 must ALSO reject a shared_eval_zero_row whose realized_state_digest contradicts arm_plan!
    with pytest.raises(ValueError, match="realized_state_digest"):
        pilot.run_arm(
            crit_plan_dict,
            lambda *a, **kw: {"execution_status": "completed", "outcome_v2": _baseline_outcome()},
            shared_eval_zero_row={**orig_base_row, "realized_state_digest": "wrong-shared-digest"},
            output_dir=tmp_path / "shared_bad_realized_digest",
            base_spec=_ped_base_spec,
        )
    # Shared eval-0 with an explicit realized_state_digest=None pin must reject a row carrying a digest
    # in live execution AND checkpoint validation (exact equality, no `is not None` skip).
    none_pin_plan = {**crit_plan_dict, "realized_state_digest": None}
    digest_shared_row = {**orig_base_row, "realized_state_digest": "unexpected-shared-digest"}
    with pytest.raises(ValueError, match="realized_state_digest"):
        pilot.run_arm(
            none_pin_plan,
            lambda *a, **kw: {"execution_status": "completed", "outcome_v2": _baseline_outcome()},
            shared_eval_zero_row=digest_shared_row,
            output_dir=tmp_path / "shared_none_pin",
            base_spec=_ped_base_spec,
        )
    with pytest.raises(ValueError, match="realized_state_digest"):
        pilot._validate_arm_checkpoint(
            none_pin_plan,
            tmp_path / "shared_none_pin_ckpt",
            shared_eval_zero_row=digest_shared_row,
            base_spec=_ped_base_spec,
        )
    # Admission rules are snapshotted once: an evaluator that mutates the caller's protocol dict
    # (flipping require_collision_free_baseline) must NOT let a colliding eval-0 commit.
    mut_proto = _json.loads(_json.dumps(base_proto))
    mut_ctx = (mut_proto, _validated_profile(), mut_proto["interaction_families"][0])

    def _mutating_eval(*a, **kw):
        mut_proto["require_collision_free_baseline"] = False
        return {
            "execution_status": "completed",
            "outcome_v2": col_out,
            "realized_state_digest": "admitted-digest-123",
            "intended_witnessed": True,
        }

    mut_dir = tmp_path / "indep_mutating_ctx"
    with pytest.raises(ValueError, match="independent eval-0 failed baseline admission"):
        pilot.run_arm(
            indep_plan,
            _mutating_eval,
            output_dir=mut_dir,
            base_spec=_ped_base_spec,
            baseline_admission_context=mut_ctx,
        )
    assert not (mut_dir / "attempts.jsonl").exists() or not (mut_dir / "attempts.jsonl").read_text().strip()
    # The snapshot must preserve types: an invalid tuple required_channels must stay invalid (not become a list).
    with pytest.raises(ValueError, match="independent eval-0 failed baseline admission"):
        pilot.run_arm(
            indep_plan,
            lambda *a, **kw: {
                "execution_status": "completed",
                "outcome_v2": _baseline_outcome(),
                "realized_state_digest": "admitted-digest-123",
                "intended_witnessed": True,
            },
            output_dir=tmp_path / "indep_tuple_channels",
            base_spec=_ped_base_spec,
            baseline_admission_context=({**base_proto, "required_channels": ()}, _validated_profile(), base_proto["interaction_families"][0]),
        )
    # The snapshot must also freeze mutable profile metadata.
    list_prof = replace(_validated_profile(), profile_id=["original"])
    list_ctx = (base_proto, list_prof, base_proto["interaction_families"][0])
    snap = pilot._snapshot_admission_context(list_ctx)
    snap_digest = pilot._baseline_admission_context_digest(snap)
    list_prof.profile_id.append("changed")
    assert snap[1].profile_id == ["original"]
    assert pilot._baseline_admission_context_digest(snap) == snap_digest
    for bad_status in ("partial", "failed"):
        with pytest.raises(ValueError, match="independent eval-0 failed baseline admission"):
            pilot.run_arm(
                indep_plan,
                lambda *a, _st=bad_status, **kw: {
                    "execution_status": _st,
                    "outcome_v2": col_out if _st == "partial" else None,
                    "realized_state_digest": "admitted-digest-123",
                    "intended_witnessed": True,
                },
                output_dir=tmp_path / f"indep_bad_{bad_status}",
                base_spec=_ped_base_spec,
                baseline_admission_context=indep_ctx,
            )
    with pytest.raises(SearchSpaceError):
        SearchDimension("x", "p", low=0.0, high=1.0, step=0.1, decimals=-309)
    with pytest.raises(SearchSpaceError):
        SearchDimension("x", "p", low=0.0, high=1.0, decimals=1.5)
    with pytest.raises(SearchSpaceError):
        make_legacy_space(sigma=10**400)
    assert SearchDimension("x", "p", low=0.0, high=_sys.float_info.max, step=1e307).in_bounds(1e307) is True
    big_int_254 = SearchDimension("k", "p", low=2**54 + 1, high=2**54 + 20, kind="int", step=2.5)
    assert big_int_254._coerce_in_bounds(float(2**54 + 8)) == 2**54 + 8
    big_int_step = SearchSpace([SearchDimension("k", "p", low=0, high=2**60, kind="int", step=1)])
    assert big_int_step.validate({"k": 2**53 + 1}).to_dict()["k"] == 2**53 + 1
    huge_int_space = SearchSpace([SearchDimension("k", "p", low=10**400, high=10**400 + 10, kind="int", step=1, default=1.0)])
    assert huge_int_space.dimensions()[0].in_bounds(1.0) is False
    assert huge_int_space.clamp({"k": 1.0}).to_dict()["k"] == 10**400
    assert huge_int_space.clamp({"k": 10**400 + 2}).to_dict()["k"] == 10**400 + 2
    assert huge_int_space.dimensions()[0].missing_value() == 10**400
    huge_int_zero = SearchDimension("k", "p", low=0, high=10**400, kind="int", step=1, default=1.0)
    assert SearchSpace([huge_int_zero]).clamp({"k": 2}).to_dict()["k"] == 2
    assert huge_int_zero.missing_value() == 1
    with pytest.raises(SearchSpaceError):
        _ = huge_int_zero.sigma_or_default
    assert SearchDimension("k", "p", low=0.5, high=10**400, kind="int").in_bounds(10**400) is True
    assert SearchDimension("k", "p", low=10**400, high=10**400 + 10, kind="int", step=0.5).coerce(1.0) == 1
    with pytest.raises(SearchSpaceError):
        SearchDimension("x", "p", low=0.0, high=1.0, step=0.5).coerce(1e308)
    assert SearchDimension("x", "p", low=0.0, high=_sys.float_info.max).in_bounds(10**400) is False
    # No double-rounding on stepped integer dimensions or stepped midpoints!
    tick_dim = SearchDimension("trigger_tick", "p", low=10, high=400, kind="int", step=10)
    assert tick_dim.coerce(15.4) == 20 and tick_dim._coerce_in_bounds(15.4) == 20
    assert tick_dim.coerce(24.6) == 20 and tick_dim._coerce_in_bounds(24.6) == 20
    step2_dim = SearchDimension("k", "p", low=0, high=10, kind="int", step=2)
    assert step2_dim._coerce_in_bounds(1.4) == 2 and step2_dim._coerce_in_bounds(2.6) == 2
    assert SearchDimension("k", "p", low=1, high=2, kind="int").missing_value() == 2
    assert SearchDimension("k", "p", low=0, high=9, kind="int", step=3).missing_value() == 6
    assert SearchDimension("k", "p", low=0, high=5, kind="int", step=4).missing_value() == 4
    assert SearchDimension("k", "p", low=1, high=4, kind="int", step=2).missing_value() == 3
    # No hang under float absorption in _step_range()!
    for absorb_dim in (
        SearchDimension("x", "p", low=1.0, high=1.0, step=1e-30, decimals=16),
        SearchDimension("x", "p", low=1e30, high=1e30, step=1.0),
        SearchDimension("x", "p", low=1e30, high=1e30 + 1e15, step=1.0),
        SearchDimension("k", "p", low=1e30, high=1e30, kind="int", step=1),
    ):
        k_lo, k_hi = absorb_dim._step_range(absorb_dim._bound_tol())
        assert k_lo <= k_hi
        sv_ab = absorb_dim.sample_value(_random.Random(13))
        assert absorb_dim.in_bounds(sv_ab)
    assert SearchDimension("k", "p", low=0, high=8, kind="int", step=1.7)._coerce_in_bounds(8) == 8
    assert SearchDimension("x", "p", low=0.0, high=8.0, kind="float", step=1.7000000000000002, decimals=0)._coerce_in_bounds(8.0) == 8.0
    int_step_frac = SearchSpace([SearchDimension("k", "p", low=0.5, high=3.5, kind="int", step=1)])
    for _ in range(10):
        sv_i = int_step_frac.sample(_random.Random(1)).to_dict()
        assert int_step_frac.validate(sv_i).to_dict() == sv_i
    int_step_16 = SearchDimension("k", "p", low=0, high=3, kind="int", step=1.6)
    sampled_16 = {int_step_16.sample_value(_random.Random(s)) for s in range(50)}
    assert 3 in sampled_16
    fine_sp = make_legacy_space(low=1.23e-310, high=1.24e-310, decimals=309, seed=13)
    assert fine_sp.validate({"trigger_radius_m": 1.24e-310}).to_dict()["trigger_radius_m"] == 1.24e-310
    sub_ep = make_legacy_space(low=0.0, high=2e-309, step=2.2e-309, decimals=309, seed=13)
    assert sub_ep.validate({"trigger_radius_m": 2e-309}).to_dict()["trigger_radius_m"] == 2e-309
    sub_tie = make_legacy_space(low=0.0, high=2e-308, step=6e-309, decimals=308, seed=13)
    assert sub_tie.validate({"trigger_radius_m": 4e-309}).to_dict()["trigger_radius_m"] == 0.0
    lat_space = SearchSpace([SearchDimension("lat", "controller_params.staging_lateral_offset_m", low=0.91, high=0.94, kind="lateral_offset_m", decimals=1)])
    lat_spec = {
        "controller_params": {
            "spawn_transform": {"location": {"x": 10.0, "y": 5.0, "z": 0.5}, "rotation": {"yaw": 0.0}},
            "destination_location": {"x": 10.0, "y": -5.0, "z": 0.5},
        }
    }
    applied_lat = lat_space.apply_to_payload(lat_spec, {"lat": 0.92})
    read_lat = lat_space.from_payload(applied_lat)
    assert read_lat is not None and lat_space.validate(read_lat.to_dict()).to_dict() == {"lat": 0.92}
    # Mixed evaluator signatures (*args+kwonly candidate, opt+kwonly candidate, 2-arg named candidate) + tuple obligations!
    kw_seen_cands = []
    def _kw_only_eval(arm, idx, *, candidate=None):
        kw_seen_cands.append(candidate)
        return {
            "execution_status": "completed",
            "outcome_v2": _baseline_outcome(),
            "semantic_fulfilled_obligations": ("one",),
        }
    kw_dir = tmp_path / "kw_only_semantic_arm"
    kw_plan = {**space_plan, "arm": "semantic", "total_budget": 2}
    res_kw = pilot.run_arm(kw_plan, _kw_only_eval, output_dir=kw_dir, base_spec=_ped_base_spec)
    assert len(kw_seen_cands) == 2 and all(isinstance(c, dict) for c in kw_seen_cands)
    assert res_kw["attempts"][0]["fitness"] == 1.0
    assert not (kw_dir / "inflight_attempt.json").exists()
    res_kw2 = pilot.run_arm(kw_plan, lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no rerun")), output_dir=kw_dir, base_spec=_ped_base_spec)
    assert len(res_kw2["attempts"]) == 2
    # Non-shared eval-0 checkpoint must ALSO reject tampered eval-0 candidate (20.0 -> 30.0)!
    kw_lines = (kw_dir / "attempts.jsonl").read_text().splitlines()
    kw_tampered0 = _json.loads(kw_lines[0])
    kw_tampered0["candidate"] = {"trigger_radius_m": 30.0}
    (kw_dir / "attempts.jsonl").write_text("\n".join([_json.dumps(kw_tampered0), *kw_lines[1:]]) + "\n")
    with pytest.raises(ValueError, match="eval-0 candidate"):
        pilot.run_arm(kw_plan, lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no rerun")), output_dir=kw_dir, base_spec=_ped_base_spec)
    assert pilot._candidate_dispatch_mode(lambda arm, idx, *args, candidate=None: None) == "keyword"
    def _opt_kw_eval(arm, idx, opt=None, *, candidate=None):
        return {"execution_status": "completed", "outcome_v2": _baseline_outcome()}
    def _two_arg_cand(arm, candidate):
        return {"execution_status": "completed", "outcome_v2": _baseline_outcome()}
    def _two_arg_cand_kw(arm, candidate, **kw):
        return {"execution_status": "completed", "outcome_v2": _baseline_outcome()}
    assert pilot._candidate_dispatch_mode(_opt_kw_eval) == "keyword"
    assert pilot._candidate_dispatch_mode(_two_arg_cand) == "none"
    assert pilot._candidate_dispatch_mode(_two_arg_cand_kw) == "none"
    # Callback mutating its received candidate dict must NOT alter the persisted candidate,
    # AND tampering an eval-1 candidate (to off-grid 23.4567 OR another on-grid value) must reject on resume!
    def _mutating_eval(arm, idx, cand):
        cand["trigger_radius_m"] = 30.0
        return {"execution_status": "completed", "outcome_v2": _baseline_outcome()}
    mut_dir = tmp_path / "mutating_callback_arm"
    mut_plan = {**space_plan, "arm": "random", "total_budget": 2}
    res_mut = pilot.run_arm(mut_plan, _mutating_eval, output_dir=mut_dir, base_spec=_ped_base_spec)
    assert res_mut["attempts"][0]["candidate"]["trigger_radius_m"] == 20.0
    assert res_mut["attempts"][1]["candidate"]["trigger_radius_m"] != 30.0
    mut_lines = (mut_dir / "attempts.jsonl").read_text().splitlines()
    for bad_c1 in (23.4567, 30.0):
        tampered_c1 = _json.loads(mut_lines[1])
        tampered_c1["candidate"] = {"trigger_radius_m": bad_c1}
        (mut_dir / "attempts.jsonl").write_text("\n".join([mut_lines[0], _json.dumps(tampered_c1)]) + "\n")
        with pytest.raises(ValueError, match="Corrupt checkpoint row"):
            pilot.run_arm(mut_plan, lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no rerun")), output_dir=mut_dir, base_spec=_ped_base_spec)
    # Tampering ONLY rng_state on row 0 or on a pending inflight reservation must reject on resume!
    tampered_rng_row0 = _json.loads(mut_lines[0])
    tampered_rng_row0["rng_state"] = list(_random.Random(999).getstate())
    (mut_dir / "attempts.jsonl").write_text("\n".join([_json.dumps(tampered_rng_row0), mut_lines[1]]) + "\n")
    with pytest.raises(ValueError, match="rng_state contradicts deterministic replay"):
        pilot.run_arm(mut_plan, lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no rerun")), output_dir=mut_dir, base_spec=_ped_base_spec)
    # Stateful policy (avfuzzer) interrupted at eval 2 must resume cleanly AND pass re-validation on next resume!
    avf_dir = tmp_path / "avfuzzer_resume_arm"
    avf_plan = {**space_plan, "arm": "avfuzzer", "seed": 13, "total_budget": 5}
    with pytest.raises(RuntimeError, match="avf_crash_eval2"):
        pilot.run_arm(
            avf_plan,
            lambda arm, idx, cand: (
                (_ for _ in ()).throw(RuntimeError("avf_crash_eval2"))
                if idx == 2
                else {"execution_status": "completed", "outcome_v2": _baseline_outcome()}
            ),
            output_dir=avf_dir,
            base_spec=_ped_base_spec,
        )
    # Tampering ONLY the pending inflight reservation's rng_state (keeping candidate intact) must reject!
    avf_inf_path = avf_dir / "inflight_attempt.json"
    orig_avf_inf = avf_inf_path.read_bytes()
    tampered_avf_inf = _json.loads(orig_avf_inf.decode("utf-8"))
    tampered_avf_inf["rng_state"] = list(_random.Random(999).getstate())
    avf_inf_path.write_text(_json.dumps(tampered_avf_inf))
    with pytest.raises(ValueError, match="rng_state contradicts deterministic replay"):
        pilot.run_arm(
            avf_plan,
            lambda *a, **kw: {"execution_status": "completed", "outcome_v2": _baseline_outcome()},
            output_dir=avf_dir,
            base_spec=_ped_base_spec,
        )
    avf_inf_path.write_bytes(orig_avf_inf)
    res_avf1 = pilot.run_arm(
        avf_plan,
        lambda *a, **kw: {"execution_status": "completed", "outcome_v2": _baseline_outcome()},
        output_dir=avf_dir,
        base_spec=_ped_base_spec,
    )
    assert len(res_avf1["attempts"]) == 5
    res_avf2 = pilot.run_arm(
        avf_plan,
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no rerun")),
        output_dir=avf_dir,
        base_spec=_ped_base_spec,
    )
    assert len(res_avf2["attempts"]) == 5
    # Stale inflight reservation contradicting committed candidate OR rng_state must reject!
    for stale_field, stale_val, stale_pat in (
        ("rng_state", list(_random.Random(999).getstate()), "stale reservation rng_state contradicts committed row"),
        ("candidate", {"trigger_radius_m": 30.0}, "stale reservation candidate"),
    ):
        stale_inf_dict = _json.loads(orig_avf_inf.decode("utf-8"))
        stale_inf_dict[stale_field] = stale_val
        avf_inf_path.write_text(_json.dumps(stale_inf_dict))
        with pytest.raises(ValueError, match=stale_pat):
            pilot.run_arm(
                avf_plan,
                lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no rerun")),
                output_dir=avf_dir,
                base_spec=_ped_base_spec,
            )
    avf_inf_path.unlink()
    # Direct run_arm() with tampered shared_eval_zero_row criticality=99.0 must reject!
    with pytest.raises(ValueError, match="criticality"):
        pilot.run_arm(
            crit_plan_dict,
            lambda *a, **kw: {"execution_status": "completed", "outcome_v2": _baseline_outcome()},
            shared_eval_zero_row={**orig_base_row, "criticality": 99.0},
            output_dir=tmp_path / "tampered_direct_shared_crit",
            base_spec=_ped_base_spec,
        )
    # Evaluator returning NaN, outcome_v2=[], criticality=True, wrong schema_version="2", mismatched profile_digest,
    # or unhashable-after-JSON tuple must fail BEFORE appending to attempts.jsonl and leave inflight_attempt.json intact!
    for bad_idx, (arm_override, bad_eval_ret, bad_pat) in enumerate((
        ("critonly", {"execution_status": "completed", "outcome_v2": _baseline_outcome(), "bad_nan": float("nan")}, "Out of range float"),
        ("critonly", {"execution_status": "completed", "outcome_v2": []}, "outcome_v2 must be an object or null"),
        ("critonly", {"execution_status": "completed", "outcome_v2": _baseline_outcome(), "criticality": True}, "invalid criticality"),
        ("critonly", {"execution_status": "completed", "outcome_v2": {**_baseline_outcome(), "schema_version": "2", "profile_digest": "f" * 64}}, "schema_version must be integer 2"),
        ("critonly", {"execution_status": "completed", "outcome_v2": {**_baseline_outcome(), "profile_digest": "f" * 64}}, "profile_digest"),
        ("semantic_crit", {"execution_status": "completed", "outcome_v2": _baseline_outcome(collision=True), "semantic_covered_signatures": [("sig", 1)]}, "policy observation failed"),
    )):
        bad_arm_dir = tmp_path / f"bad_writer_arm_{bad_idx}"
        bad_plan = {
            **crit_plan_dict,
            "arm": arm_override,
            "shared_eval_zero_hash": None,
            "realized_state_digest": None,
            "total_budget": 2,
        }
        with pytest.raises(ValueError, match=bad_pat):
            pilot.run_arm(
                bad_plan,
                lambda *a, _ret=bad_eval_ret, **kw: _ret,
                output_dir=bad_arm_dir,
                base_spec=_ped_base_spec,
            )
        assert not (bad_arm_dir / "attempts.jsonl").exists()
        assert (bad_arm_dir / "inflight_attempt.json").is_file()
    crit_manifest.unlink()
    with pytest.raises(ValueError, match="without pilot_manifest.json"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    with pytest.raises(ValueError, match="without pilot_manifest.json"):
        pilot.run_arm(
            crit_plan_dict,
            lambda *a, **kw: calls_cross_arm.append("direct"),
            shared_eval_zero_row=orig_base_row,
            output_dir=crit_dir,
            base_spec=_ped_base_spec,
        )
    assert calls_cross_arm == []
    assert arm_attempts.read_bytes() == torn_random_bytes
    crit_manifest.write_bytes(orig_crit_manifest)

    # (e) Missing baseline_checkpoint.json, tampered admission/baseline_row/share_eval_zero/plan, or non-dict baseline_inflight.json rejected!
    base_ckpt = pilot_out / "baseline_checkpoint.json"
    base_inflight = pilot_out / "baseline_inflight.json"
    orig_base_ckpt = base_ckpt.read_bytes()
    tampered_ckpt = _json.loads(orig_base_ckpt.decode("utf-8"))
    tampered_ckpt["share_eval_zero"] = False
    base_ckpt.write_text(_json.dumps(tampered_ckpt))
    with pytest.raises(ValueError, match="contradicts"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    # Tampering baseline_row to a collision + intended_witnessed=False while leaving admitted=True must reject!
    tampered_base_row_ckpt = _json.loads(orig_base_ckpt.decode("utf-8"))
    tampered_base_row_ckpt["baseline_row"]["outcome_v2"] = _baseline_outcome(collision=True)
    tampered_base_row_ckpt["baseline_row"]["intended_witnessed"] = False
    base_ckpt.write_text(_json.dumps(tampered_base_row_ckpt))
    with pytest.raises(ValueError, match="admission contradicts baseline_row"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    # Tampering baseline_row top-level execution_status='failed'+run_error OR explicit execution_status=None must reject!
    tampered_failed_base = _json.loads(orig_base_ckpt.decode("utf-8"))
    tampered_failed_base["baseline_row"]["execution_status"] = "failed"
    tampered_failed_base["baseline_row"]["run_error"] = "simulated_crash"
    base_ckpt.write_text(_json.dumps(tampered_failed_base))
    with pytest.raises(ValueError, match="admission contradicts baseline_row"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    tampered_null_status = _json.loads(orig_base_ckpt.decode("utf-8"))
    tampered_null_status["baseline_row"]["execution_status"] = None
    base_ckpt.write_text(_json.dumps(tampered_null_status))
    with pytest.raises(ValueError, match="admission contradicts baseline_row"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    # validate_initial_configuration must handle non-dict nested fields, float schema_version=2.0,
    # mismatched profile_digest, quality.low_interaction=[1], and blank/non-str digest without AttributeError!
    proto_check = pilot.load_protocol()
    val_prof_check = _validated_profile()
    for bad_out, bad_dig in (
        ({**_baseline_outcome(), "categories": [1]}, "abc"),
        ({**_baseline_outcome(), "provenance": [1]}, "abc"),
        ({**_baseline_outcome(), "schema_version": 2.0}, "abc"),
        ({**_baseline_outcome(), "profile_digest": "0" * 64}, "abc"),
        ({**_baseline_outcome(), "categories": {**_baseline_outcome()["categories"], "rules": []}}, "abc"),
        ({**_baseline_outcome(), "categories": {**_baseline_outcome()["categories"], "quality": {**_baseline_outcome()["categories"]["quality"], "low_interaction": [1]}}}, "abc"),
        ({**_baseline_outcome(), "categories": {**_baseline_outcome()["categories"], "behaviour": [1]}}, "abc"),
        ({**_baseline_outcome(), "categories": {**_baseline_outcome()["categories"], "behaviour": {"close_proximity": [1]}}}, "abc"),
        (_baseline_outcome(), "   "),
        (_baseline_outcome(), 123),
    ):
        adm_bad = pilot.validate_initial_configuration(bad_out, proto_check, profile=val_prof_check, realized_state_digest=bad_dig)
        assert adm_bad["admitted"] is False
    # Resumed baseline_checkpoint.json with shared baseline_row["criticality"]=True OR injected 99.0 must reject BEFORE unlinking baseline_inflight.json!
    for bad_crit_val, err_pat in ((True, "invalid criticality"), (99.0, "criticality")):
        base_inflight.write_text('{"marker": true}')
        tampered_base_crit = _json.loads(orig_base_ckpt.decode("utf-8"))
        tampered_base_crit["baseline_row"]["criticality"] = bad_crit_val
        base_ckpt.write_text(_json.dumps(tampered_base_crit))
        with pytest.raises(ValueError, match=err_pat):
            pilot.main([
                "--profile", str(prof_path),
                "--validation-report", str(live_report_path),
                "--family", "pedestrian_crossing",
                "--seed", "13",
                "--output-dir", str(pilot_out),
            ])
        assert base_inflight.is_file()
        base_inflight.unlink()
    # Rejected baseline_checkpoint.json with float schema_version=2.0 must reject on resume (never re-emit report)!
    tampered_rej_sv = _json.loads(orig_base_ckpt.decode("utf-8"))
    tampered_rej_sv["baseline_row"]["outcome_v2"]["schema_version"] = 2.0
    tampered_rej_sv["admission"] = pilot.validate_initial_configuration(
        tampered_rej_sv["baseline_row"]["outcome_v2"],
        proto_check,
        profile=val_prof_check,
        realized_state_digest=tampered_rej_sv["baseline_row"].get("realized_state_digest"),
        intended_witnessed=tampered_rej_sv["baseline_row"].get("intended_witnessed"),
        family=proto_check["interaction_families"][0],
        row_execution_status=tampered_rej_sv["baseline_row"].get("execution_status"),
        run_error=tampered_rej_sv["baseline_row"].get("run_error"),
    )
    tampered_rej_sv["plan"] = None
    tampered_rej_sv["share_eval_zero"] = False
    base_ckpt.write_text(_json.dumps(tampered_rej_sv))
    with pytest.raises(ValueError, match="Malformed outcome_v2: schema_version must be integer 2"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    # Missing or inflated (2 != 1) physical_calls_charged on baseline_checkpoint.json must reject!
    for bad_calls_ckpt, calls_pat in (
        ({k: v for k, v in _json.loads(orig_base_ckpt.decode("utf-8")).items() if k != "physical_calls_charged"}, "missing physical_calls_charged"),
        ({**_json.loads(orig_base_ckpt.decode("utf-8")), "physical_calls_charged": 2}, "invalid physical_calls_charged"),
    ):
        base_ckpt.write_text(_json.dumps(bad_calls_ckpt))
        with pytest.raises(ValueError, match=calls_pat):
            pilot.main([
                "--profile", str(prof_path),
                "--validation-report", str(live_report_path),
                "--family", "pedestrian_crossing",
                "--seed", "13",
                "--output-dir", str(pilot_out),
            ])
    # Flipping only admission.admitted=False must reject BEFORE unlinking baseline_inflight.json!
    base_inflight.write_text('{"marker": true}')
    tampered_admit_false = _json.loads(orig_base_ckpt.decode("utf-8"))
    tampered_admit_false["admission"]["admitted"] = False
    base_ckpt.write_text(_json.dumps(tampered_admit_false))
    with pytest.raises(ValueError, match="admission contradicts baseline_row"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    assert base_inflight.is_file()
    base_inflight.unlink()
    tampered_plan_ckpt = _json.loads(orig_base_ckpt.decode("utf-8"))
    tampered_plan_ckpt["plan"]["arms"][0]["total_budget"] += 1
    base_ckpt.write_text(_json.dumps(tampered_plan_ckpt))
    with pytest.raises(ValueError, match="does not match protocol/baseline_row"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    assert calls_cross_arm == []
    # Stale baseline_inflight.json with mismatched identity must NOT be silently deleted on resume,
    # AND standalone baseline_inflight.json with status='completed' must reject as corrupt!
    orig_id_obj = _json.loads(orig_base_ckpt.decode("utf-8"))["identity"]
    base_ckpt.write_bytes(orig_base_ckpt)
    base_inflight.write_text(_json.dumps({
        "identity": {**orig_id_obj, "seed": 999},
        "status": "interrupted",
        "run_error": "baseline_interrupted_before_completion",
    }))
    with pytest.raises(ValueError, match="Incompatible pilot baseline checkpoint"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    assert base_inflight.is_file()
    # Duplicate keys in baseline_inflight.json must reject via strict JSON parser!
    base_inflight.write_text(
        '{"identity": '
        + _json.dumps(orig_id_obj)
        + ', "status": "completed", "status": "interrupted", "run_error": "baseline_interrupted_before_completion"}'
    )
    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    assert base_inflight.is_file()
    # Valid stale baseline_inflight.json must NOT be deleted when an arm checkpoint fails upfront validation!
    valid_inflight_json = _json.dumps({
        "identity": orig_id_obj,
        "status": "interrupted",
        "run_error": "baseline_interrupted_before_completion",
    })
    base_inflight.write_text(valid_inflight_json)
    saved_crit_bytes = crit_attempts.read_bytes()
    crit_attempts.write_text("{corrupt_arm_line\n")
    try:
        with pytest.raises(ValueError):
            pilot.main([
                "--profile", str(prof_path),
                "--validation-report", str(live_report_path),
                "--family", "pedestrian_crossing",
                "--seed", "13",
                "--output-dir", str(pilot_out),
            ])
        assert base_inflight.is_file()
    finally:
        crit_attempts.write_bytes(saved_crit_bytes)
    # Standalone valid baseline_inflight.json alongside orphaned arms/ state must reject as corrupt!
    base_ckpt.unlink()
    with pytest.raises(ValueError, match="standalone baseline_inflight.json"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    assert base_inflight.is_file() and not base_ckpt.exists()
    # Rejected baseline_checkpoint.json coexisting with orphaned arms/ state must reject BEFORE unlinking or writing report!
    rej_orphan_ckpt = _json.loads(orig_base_ckpt.decode("utf-8"))
    rej_orphan_ckpt["baseline_row"] = {
        "arm": "baseline_admission",
        "policy": "baseline_admission",
        "eval_index": 0,
        "execution_status": "interrupted",
        "run_error": "baseline_interrupted_before_completion",
        "outcome_v2": None,
    }
    rej_orphan_ckpt["admission"] = {
        "admitted": False,
        "reasons": ["baseline_interrupted_before_completion"],
        "profile_digest": orig_id_obj["profile_digest"],
        "realized_state_digest": None,
        "intended_witnessed": None,
    }
    rej_orphan_ckpt["plan"] = None
    rej_orphan_ckpt["share_eval_zero"] = False
    base_ckpt.write_text(_json.dumps(rej_orphan_ckpt))
    with pytest.raises(ValueError, match="contradicts rejected baseline_checkpoint.json"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    assert base_inflight.is_file()
    base_ckpt.unlink()
    # Malformed planned_only report ({"status":"planned_only","plan":{"arms":[null]}}) must NOT bypass standalone reservation check!
    bad_plan_dir = tmp_path / "bad_planned_only_dir"
    bad_plan_dir.mkdir()
    (bad_plan_dir / "pilot_report.json").write_text('{"status":"planned_only","plan":{"arms":[null]}}')
    (bad_plan_dir / "baseline_inflight.json").write_text(valid_inflight_json)
    assert pilot._has_existing_execution_state(bad_plan_dir, ignore_baseline_inflight=True) is True
    with pytest.raises(ValueError, match="standalone baseline_inflight.json"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(bad_plan_dir),
        ])
    assert (bad_plan_dir / "baseline_inflight.json").is_file() and not (bad_plan_dir / "baseline_checkpoint.json").exists()
    base_inflight.write_text(_json.dumps({
        "identity": orig_id_obj,
        "status": "completed",
        "run_error": "baseline_interrupted_before_completion",
    }))
    with pytest.raises(ValueError, match="Corrupt baseline inflight reservation"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    # Non-dict baseline_inflight.json must raise ValueError (not AttributeError)!
    base_inflight.write_text("[]")
    with pytest.raises(ValueError, match="Incompatible pilot baseline checkpoint"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    base_inflight.unlink()
    with pytest.raises(ValueError, match="without baseline_checkpoint.json"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
        ])
    with pytest.raises(RuntimeError, match="already contains pilot execution state"):
        pilot.main([
            "--profile", str(prof_path),
            "--validation-report", str(live_report_path),
            "--family", "pedestrian_crossing",
            "--seed", "13",
            "--output-dir", str(pilot_out),
            "--plan-only",
        ])
    assert calls_cross_arm == []
    assert not base_ckpt.exists()
    base_ckpt.write_bytes(orig_base_ckpt)

    # (f) Valid --plan-only (both WITH and WITHOUT --validation-report) MUST allow repeat --plan-only and first live run!
    plan_no_val = tmp_path / "plan_no_val_then_live"
    monkeypatch.setattr("research.harness.runner.HarnessRunner.run", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no live call during plan-only")))
    assert pilot.main([
        "--profile", str(prof_path),
        "--family", "pedestrian_crossing",
        "--seed", "13",
        "--output-dir", str(plan_no_val),
        "--plan-only",
    ]) == 0
    no_val_rep = _json.loads((plan_no_val / "pilot_report.json").read_text())
    assert no_val_rep["status"] == "planned_only" and no_val_rep["validation_check"] is None
    assert pilot._has_existing_execution_state(plan_no_val) is False
    # Re-running --plan-only into the same directory must succeed!
    assert pilot.main([
        "--profile", str(prof_path),
        "--family", "pedestrian_crossing",
        "--seed", "13",
        "--output-dir", str(plan_no_val),
        "--plan-only",
    ]) == 0
    # Malformed plan metadata (forged spec_digest={}, agent_config_digest=False, search_space_signature=[], budget=1) must fail exemption!
    for bad_plan_mut in (
        lambda p: (p.update({"spec_digest": {}}), [a.update({"spec_digest": {}}) for a in p["arms"]]),
        lambda p: (p.update({"agent_config_digest": False}), [a.update({"agent_config_digest": False}) for a in p["arms"]]),
        lambda p: (p.update({"search_space_signature": []}), [a.update({"search_space_signature": []}) for a in p["arms"]]),
        lambda p: ([a.update({"total_budget": 1}) for a in p["arms"]]),
        lambda p: (p["arms"][0].update({"total_budget": 99})),
    ):
        forged_rep = _json.loads(_json.dumps(no_val_rep))
        bad_plan_mut(forged_rep["plan"])
        assert pilot._is_valid_planned_only_report(forged_rep) is False
    monkeypatch.setattr("research.harness.runner.HarnessRunner.run", orig_run)
    assert pilot.main([
        "--profile", str(prof_path),
        "--validation-report", str(live_report_path),
        "--family", "pedestrian_crossing",
        "--seed", "13",
        "--output-dir", str(plan_no_val),
    ]) == 0
    assert _json.loads((plan_no_val / "pilot_report.json").read_text())["status"] == "completed"
    # Directory occupying an arm checkpoint name (arms/random/attempts.jsonl/) must be detected as existing arm state!
    dir_ckpt_out = tmp_path / "dir_ckpt_name_out"
    (dir_ckpt_out / "arms" / "random" / "attempts.jsonl").mkdir(parents=True)
    assert pilot._has_existing_arms_state(dir_ckpt_out) is True
    assert pilot._has_existing_execution_state(dir_ckpt_out) is True

    plan_then_live = tmp_path / "plan_then_live"
    assert pilot.main([
        "--profile", str(prof_path),
        "--validation-report", str(live_report_path),
        "--family", "pedestrian_crossing",
        "--seed", "13",
        "--output-dir", str(plan_then_live),
        "--plan-only",
    ]) == 0
    assert _json.loads((plan_then_live / "pilot_report.json").read_text())["status"] == "planned_only"
    monkeypatch.setattr("research.harness.runner.HarnessRunner.run", orig_run)
    assert pilot.main([
        "--profile", str(prof_path),
        "--validation-report", str(live_report_path),
        "--family", "pedestrian_crossing",
        "--seed", "13",
        "--output-dir", str(plan_then_live),
    ]) == 0
    assert _json.loads((plan_then_live / "pilot_report.json").read_text())["status"] == "completed"

    # (g) Symlinked arm directory, symlink cycles in arms/, and bare {"status": "planned_only"} (missing plan payload)!
    sym_target = tmp_path / "sym_arm_target"
    sym_target.mkdir()
    (sym_target / "attempts.jsonl").write_text("{}\n")
    sym_dir = tmp_path / "sym_pilot_dir"
    (sym_dir / "arms").mkdir(parents=True)
    (sym_dir / "arms" / "critonly").symlink_to(sym_target)
    assert pilot._has_existing_execution_state(sym_dir) is True
    # Symlink cycle inside arms/ must terminate cleanly without hanging!
    cycle_dir = tmp_path / "sym_cycle_dir"
    (cycle_dir / "arms").mkdir(parents=True)
    (cycle_dir / "arms" / "loop_a").symlink_to(cycle_dir / "arms")
    (cycle_dir / "arms" / "loop_b").symlink_to(cycle_dir / "arms")
    assert pilot._has_existing_execution_state(cycle_dir) is False
    bare_plan_dir = tmp_path / "bare_plan_dir"
    bare_plan_dir.mkdir()
    (bare_plan_dir / "pilot_report.json").write_text(_json.dumps({"status": "planned_only"}))
    assert pilot._has_existing_execution_state(bare_plan_dir) is True

    # Duplicate keys in an unterminated trailing attempts.jsonl line must reject as corrupt (never torn-truncate)!
    assert pilot._is_strict_json_object_prefix('{"candidate":{"x":1,"\\u0078":2}') is False
    assert pilot._is_strict_json_object_prefix('{"a":1,"a":2,"b":') is False
    dup_line_arm = tmp_path / "dup_trailing_arm"
    dup_plan = {**space_plan, "arm": "random", "total_budget": 2}
    pilot.run_arm(
        dup_plan,
        lambda *a, **kw: {"execution_status": "completed", "outcome_v2": _baseline_outcome()},
        output_dir=dup_line_arm,
        base_spec=_ped_base_spec,
    )
    dup_lines = (dup_line_arm / "attempts.jsonl").read_text().splitlines()
    (dup_line_arm / "attempts.jsonl").write_bytes((dup_lines[0] + '\n{"candidate":{"x":1,"\\u0078":2}').encode("utf-8"))
    with pytest.raises(ValueError, match="Corrupt checkpoint row"):
        pilot.run_arm(
            dup_plan,
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no rerun on duplicate-key corrupt row")),
            output_dir=dup_line_arm,
            base_spec=_ped_base_spec,
        )
    # Restore valid attempts.jsonl files so any downstream assertions in this test remain clean
    arm_attempts.write_bytes(("\n".join(full_lines) + "\n").encode("utf-8"))
    crit_attempts.write_bytes(orig_crit_bytes)


def test_pilot_resume_after_inflight_interruption_never_exceeds_physical_budget(tmp_path):
    protocol = pilot.load_protocol()
    validated = _validated_profile()
    matched = pilot.build_matched_arms(protocol, validated, "pedestrian_crossing", seed=13)
    arm_plan = matched["arms"][0]
    arm_dir = tmp_path / "inflight_arm"

    physical_launches = []

    def crashing_eval(arm, idx, cand):
        physical_launches.append(idx)
        if idx == 1:
            raise RuntimeError("simulated crash after physical launch before row append")
        return {"execution_status": "completed", "outcome_v2": _baseline_outcome()}

    with pytest.raises(RuntimeError, match="simulated crash"):
        pilot.run_arm(arm_plan, crashing_eval, output_dir=arm_dir)

    assert physical_launches == [0, 1]
    assert (arm_dir / "inflight_attempt.json").is_file()

    def healthy_eval(arm, idx, cand):
        physical_launches.append(idx)
        return {"execution_status": "completed", "outcome_v2": _baseline_outcome()}

    resumed = pilot.run_arm(arm_plan, healthy_eval, output_dir=arm_dir)
    # Total physical launches across crashed run + resumed run MUST equal budget (5), NEVER 6!
    assert physical_launches == [0, 1, 2, 3, 4]
    assert resumed["budget"]["logical_evaluations"] == 5
    assert resumed["budget"]["physical_simulation_calls"] == 5
    assert resumed["budget"]["failed_attempts_charged"] == 1
    assert resumed["attempts"][1]["execution_status"] == "interrupted"
    assert not (arm_dir / "inflight_attempt.json").exists()


def test_search_space_rejects_null_bounds():
    from research.harness.search_space import SearchSpace, SearchSpaceError, make_legacy_space

    for bad in ({"low": None, "high": 1.0}, {"low": 0.0, "high": None}):
        payload = {"dimensions": [{"name": "r", "spec_path": "controller_params.trigger_radius_m", **bad}]}
        with pytest.raises(SearchSpaceError, match="cannot be null"):
            SearchSpace.from_dict(payload)
    with pytest.raises(SearchSpaceError, match="cannot be null"):
        make_legacy_space(high=None)
    with pytest.raises(SearchSpaceError, match="cannot be null"):
        make_legacy_space(low=None)
