from dataclasses import replace

import pytest

from research.harness.outcome_config import load_outcome_profile
from research.harness.outcome_io import EvidenceWriter
from research.scripts.outcome_sensitivity_report import replay_profiles
from research.tests.outcome_helpers import manifest
from research.tests.test_outcome_detectors import interaction_frames


def test_profile_replay_recomputes_events_and_rejects_unsupported_capture(tmp_path):
    writer = EvidenceWriter(tmp_path / "trace.jsonl", "run")
    for f in interaction_frames():
        writer.append(f)
    path = writer.finish(manifest())
    normal = load_outcome_profile()
    stricter = replace(normal, settings=tuple(replace(s, value=.8) if s.name == "ttc_s" else s for s in normal.settings))
    report = replay_profiles(path, [normal, stricter])
    first, second = report["profiles"]
    assert first["categories"]["safety"]["near_miss"]["status"] == "present"
    assert second["categories"]["safety"]["near_miss"]["status"] == "absent"
    assert first["episode_counts"]["near_miss"] == 1
    assert second["episode_counts"].get("near_miss", 0) == 0
    unsupported = replace(normal, settings=tuple(replace(s, value=100) if s.name == "scan_radius_m" else s for s in normal.settings))
    with pytest.raises(ValueError, match="unsupported_scan_radius"):
        replay_profiles(path, [unsupported])


def test_frozen_sensitivity_grid_builds_expected_variants_and_comparisons(tmp_path):
    from research.scripts.outcome_sensitivity_report import build_profiles, compare_assessments
    writer = EvidenceWriter(tmp_path / "grid.jsonl", "run")
    for f in interaction_frames():
        writer.append(f)
    path = writer.finish(manifest())
    profiles = build_profiles(load_outcome_profile())
    # Baseline + 2 variants each for 5 OFAT params + 2 joint (strict/lenient) = 13 unique profiles
    assert len(profiles) == 13
    report = replay_profiles(path, profiles)
    comp = report["comparison"]
    assert comp["n_variants"] == 12
    assert any(c["n_status_changes"] >= 1 for c in comp["changes"])
    assert compare_assessments(report) == comp
