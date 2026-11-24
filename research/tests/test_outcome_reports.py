"""Tests for schema-v2 outcome and utility reports (defensible denominators, missingness strata, legacy mode)."""
from __future__ import annotations

import json
from pathlib import Path

from research.harness.outcome_models import SensorEvidence
from research.harness.safety_outcomes import OutcomeEvaluator
from research.scripts.safety_outcome_report import build_markdown, build_report, write_report
from research.scripts.safety_utility_report import build_utility_report
from research.tests.outcome_helpers import actor, frame, manifest
from research.tests.test_scenario_quality import VALID


def _v2_outcome(*, collision=False, near_miss=False, partial=False, run_id="r"):
    if near_miss:
        frames = [frame(i, actors=(actor("a", x=14 + 2 * i, vx=0 if i == 0 else 20),)) for i in range(11)]
    else:
        frames = [frame(i) for i in range(11)]
    sensors = [SensorEvidence(run_id, 5, 0.5, "collision", "b")] if collision else []
    ev = OutcomeEvaluator()
    for f in frames:
        ev.observe(frame(f.sim_frame, run_id=run_id, actors=f.actors))
    for s in sensors:
        ev.record_sensor_event(s)
    m = manifest(1.0, run_id=run_id, execution_status="partial" if partial else "completed", scenario_checks=VALID)
    return ev.finish(m).to_dict()


def test_v2_outcome_report_reconciles_counts_missingness_strata_and_overlapping_labels(tmp_path):
    sem_dir = tmp_path / "routeA" / "semantic"
    sem_dir.mkdir(parents=True)
    # Row 0: both near_miss AND collision present (overlapping labels), completed
    # Row 1: partial execution with collision present
    # Row 2: failed attempt without run log (missing/failed retained as unknown)
    # Row 3: duplicate attempt_id copy (must be counted as duplicate and skipped)
    rows = [
        {"policy": "semantic", "route": "routeA", "eval_index": 0, "attempt_id": "a0",
         "outcome_v2": _v2_outcome(collision=True, near_miss=True, run_id="r0"),
         "engine_run_obligations": ["crossing_path(pedestrian,ego)"]},
        {"policy": "semantic", "route": "routeA", "eval_index": 1, "attempt_id": "a1",
         "outcome_v2": _v2_outcome(collision=True, partial=True, run_id="r1"),
         "engine_run_obligations": ["crossing_path(pedestrian,ego)"]},
        {"policy": "semantic", "route": "routeA", "eval_index": 2, "attempt_id": "a2",
         "run_error": "simulator crash", "run_json_path": str(tmp_path / "missing.json")},
        {"policy": "semantic", "route": "routeA", "eval_index": 0, "attempt_id": "a0",
         "outcome_v2": _v2_outcome(collision=True, near_miss=True, run_id="r0")},
        "not-valid-json",
    ]
    (sem_dir / "rows.jsonl").write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows))

    report = build_report(tmp_path, schema_version=2)
    assert report["schema_version"] == 2
    assert report["data_state"]["n_attempted"] == 3
    assert report["data_state"]["n_duplicate_rows"] == 1
    assert report["data_state"]["n_malformed_rows"] == 1

    bucket = report["policies"]["semantic"]
    assert bucket["n_attempted"] == 3
    assert bucket["n_completed"] == 1
    assert bucket["n_partial"] == 1
    assert bucket["n_failed_or_missing"] == 1
    # Collision: 2 present (r0, r1), 0 absent, 1 unknown (a2)
    col = bucket["metrics"]["safety.collision"]
    assert (col["present"], col["absent"], col["unknown"]) == (2, 0, 1)
    assert col["assessable_rate"] == 1.0
    assert col["discovery_per_attempt"] == 2 / 3
    # Near miss: 1 present (r0), 1 absent (r1), 1 unknown (a2)
    nm = bucket["metrics"]["safety.near_miss"]
    assert (nm["present"], nm["absent"], nm["unknown"]) == (1, 1, 1)
    assert nm["assessable_rate"] == 0.5
    # Complete execution stratum only counts r0
    assert bucket["complete_execution_stratum"]["n_completed"] == 1
    assert bucket["complete_execution_stratum"]["metrics"]["safety.collision"]["present"] == 1

    md = build_markdown(report)
    assert "No global `unsafe_rate`" in md
    paths = write_report(report, tmp_path / "out")
    assert paths["json"].is_file() and paths["markdown"].is_file()


def test_v2_utility_report_handles_empty_denominators_and_unknown_priors(tmp_path):
    arm_dir = tmp_path / "seed-1" / "routeA" / "semantic"
    arm_dir.mkdir(parents=True)
    rows = [
        {"policy": "semantic", "route": "routeA", "eval_index": 0, "attempt_id": "s0",
         "outcome_v2": _v2_outcome(collision=True, run_id="s0"),
         "engine_run_obligations": []},  # baseline eval 0 excluded from discovery, observed absent ob_a
        {"policy": "semantic", "route": "routeA", "eval_index": 1, "attempt_id": "s1",
         "run_error": "crash"},  # unknown prior before discovery, semantic unobserved -> n_obligation_unknown
        {"policy": "semantic", "route": "routeA", "eval_index": 2, "attempt_id": "s2",
         "outcome_v2": _v2_outcome(collision=True, run_id="s2"),
         "engine_run_obligations": ["ob_a"]},
        {"policy": "semantic", "route": "routeA", "eval_index": 3, "attempt_id": "s3",
         "outcome_v2": {
             "schema_version": 2, "run_id": "s3", "execution_status": "partial", "termination_reason": "timeout",
             "categories": {"safety": {"collision": {"status": "unknown"}}},
             "aggregates": {"safety_event": {"status": "unknown"}},
         },
         "engine_run_obligations": []},  # semantic observed absent ob_a, outcome unknown -> absent_unknown == 1
    ]
    (arm_dir / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    utility = build_utility_report([tmp_path], schema_version=2, metric="safety.collision", exclude_eval_zero=True)
    arms = utility["time_to_first"]["arms"]
    assert len(arms) == 1
    assert arms[0]["first_eval"] == 2
    assert arms[0]["n_unknown_prior"] == 1
    assert arms[0]["censoring_reason"] == "found_with_unknown_prior"
    # Obligation association excludes unobserved failed runs from absent denominator and counts absent_unknown explicitly
    assoc = utility["association"]["witnessed"]["obligations"]["ob_a"]
    assert assoc["present_hits"] == 1 and assoc["present_assessable"] == 1
    assert assoc["absent_unknown"] == 1
    assert assoc["n_obligation_unknown"] == 1

    # Verify v2 Markdown and CLI path work end-to-end without KeyErrors
    from research.scripts.safety_utility_report import build_markdown as build_util_md, main as util_main
    md = build_util_md(utility)
    assert "Safety-utility evaluation (schema v2)" in md
    assert "found_with_unknown_prior" in md
    out_dir = tmp_path / "util_cli_out"
    assert util_main(["--search-root", str(tmp_path), "--out-dir", str(out_dir), "--outcome-schema", "2", "--metric", "safety.collision", "--exclude-eval-zero"]) == 0
    assert (out_dir / "safety_utility.json").is_file() and (out_dir / "safety_utility.md").is_file()
