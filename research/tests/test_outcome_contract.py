from dataclasses import replace
import json

import pytest

from research.harness.outcome_config import DEFAULT_PROFILE, load_outcome_profile, profile_digest, validate_capture_support
from research.harness.outcome_models import (
    AssessmentStatus as S, MetricAssessment, aggregate_assessments, assessment_is_present, strict_json,
    CaptureManifest, ChannelCoverage,
)
from research.harness.safety_outcomes import classify_run
from research.harness.legacy_outcomes import classify_run as legacy
from research.tests.outcome_helpers import actor, manifest


def test_four_state_aggregation_and_no_truthiness():
    absent = MetricAssessment(S.ABSENT, reasons=("complete",))
    unknown = MetricAssessment()
    present = MetricAssessment(S.PRESENT, ("event",))
    na = MetricAssessment(S.NOT_APPLICABLE)
    assert aggregate_assessments([absent, unknown]).status == S.UNKNOWN
    assert aggregate_assessments([present, unknown]).status == S.PRESENT
    assert aggregate_assessments([absent, na]).status == S.ABSENT
    assert aggregate_assessments([na]).status == S.NOT_APPLICABLE
    assert aggregate_assessments([]).status == S.NOT_APPLICABLE
    assert assessment_is_present(present)
    assert not assessment_is_present({"status": "unknown"})
    with pytest.raises(TypeError):
        bool(unknown)
    with pytest.raises(ValueError):
        MetricAssessment(S.ABSENT)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_invalid_geometry_rejected(value):
    with pytest.raises(ValueError):
        actor(x=value)


def test_strict_json_rejects_nonfinite():
    with pytest.raises(ValueError):
        strict_json({"ttc": float("inf")})


@pytest.mark.parametrize("change", ["missing", "duplicate", "unit", "negative", "nan", "horizon", "unknown"])
def test_invalid_profiles_rejected(tmp_path, change):
    data = json.loads(DEFAULT_PROFILE.read_text())
    if change == "missing":
        data["settings"] = [s for s in data["settings"] if s["name"] != "minimum_observation_s"]
    elif change == "duplicate":
        data["settings"].append(data["settings"][0])
    elif change == "unit":
        data["settings"][0]["unit"] = "m"
    elif change == "negative":
        data["settings"][0]["value"] = -1
    elif change == "nan":
        data["settings"][0]["value"] = float("nan")
    elif change == "horizon":
        data["settings"][0]["value"] = 100
    else:
        data["settings"][0]["name"] = "typo"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_outcome_profile(path)


def test_profile_identity_and_draft_gate():
    p = load_outcome_profile()
    assert profile_digest(p) == profile_digest(replace(p, settings=tuple(reversed(p.settings))))
    changed = replace(p, settings=(replace(p.settings[0], value=1.0), *p.settings[1:]))
    assert profile_digest(changed) != profile_digest(p)
    with pytest.raises(ValueError, match="validation"):
        p.require_pilot_ready()
    with pytest.raises(ValueError):
        replace(p, status="validated")
    assert validate_capture_support(p, manifest()) == []
    assert "unsupported_scan_radius_m" in validate_capture_support(p, manifest(scan_radius_m=10))
    assert "unsupported_max_sample_interval_s" in validate_capture_support(p, manifest(max_sample_interval_s=.2))


def test_coverage_cannot_claim_complete_without_interval():
    with pytest.raises(ValueError):
        ChannelCoverage("complete")
    with pytest.raises(ValueError):
        manifest(end_s=-1)
    with pytest.raises(ValueError):
        manifest(reached_goal="false")
    original = manifest()
    from dataclasses import asdict
    assert CaptureManifest.from_dict(asdict(original)) == original


def test_legacy_proxy_counts_do_not_become_new_outcomes():
    payload = {"ticks_executed": 100, "reached_goal": False, "collision_count": 0,
               "safety_metrics": {"min_ttc": .1, "red_light_violations": 5, "max_lane_offset_m": 8,
                                  "max_stationary_streak_ticks": 500, "max_deceleration_mps2": 50}}
    result = classify_run(payload)
    assert result["schema_version"] == 2
    assert result["categories"]["safety"]["near_miss"]["status"] == "unknown"
    assert result["categories"]["rules"]["red_light_violation"]["status"] == "unknown"
    assert result["categories"]["task"]["stuck"]["status"] == "unknown"
    assert result["categories"]["task"]["route_incomplete"]["status"] == "present"
    assert result["aggregates"]["safety_event"]["status"] == "unknown"
    assert classify_run(payload, schema_version=1) == legacy(payload)


def test_partial_legacy_contact_survives_error_but_missing_data_is_not_safe():
    payload = {"ticks_executed": 10, "run_error": "server lost", "collisions": [
        {"frame": 12, "actor_id": 9, "actor_type": "vehicle.test"}]}
    result = classify_run(payload)
    assert result["execution_status"] == "partial"
    assert result["categories"]["safety"]["collision"]["status"] == "present"
    assert classify_run({})["execution_status"] == "unknown"
    assert classify_run({})["categories"]["safety"]["collision"]["status"] == "unknown"
    assert classify_run({"collision_count": 10})["categories"]["safety"]["collision"]["status"] == "unknown"
    assert classify_run({**payload, "dry_run": True})["categories"]["safety"]["collision"]["status"] == "unknown"


def test_safety_schema_dispatches_v1_and_v2():
    from research.harness.safety_outcomes import safety_schema
    assert safety_schema(schema_version=2)["schema_version"] == 2
    assert safety_schema(schema_version=1)["schema_version"] == 1
    with pytest.raises(ValueError):
        safety_schema(schema_version=99)
