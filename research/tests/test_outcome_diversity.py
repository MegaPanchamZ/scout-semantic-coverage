from copy import deepcopy
from dataclasses import replace
import json

import pytest

from research.harness.outcome_diversity import build_event_signature, build_run_signature, cluster_outcomes, summarize_diversity
from research.harness.outcome_io import EvidenceWriter
from research.harness.outcome_models import SensorEvidence
from research.harness.safety_outcomes import OutcomeEvaluator
from research.scripts.outcome_diversity_report import main, report_captures
from research.tests.outcome_helpers import actor, frame, manifest


def collision_run(*, actor_id="a", role="subject", time=.2, run_id="run"):
    evidence = [replace(frame(i, actors=(actor(actor_id, x=50, role=role),), ego=actor(x=i, role="ego")), run_id=run_id) for i in range(11)]
    contact = SensorEvidence(run_id, round(time*10), time, "collision", actor_id)
    evaluator = OutcomeEvaluator()
    for f in evidence:
        evaluator.observe(f)
    evaluator.record_sensor_event(contact)
    return evaluator.finish(replace(manifest(), run_id=run_id)), [*evidence, contact]


def test_event_pattern_ignores_ephemeral_ids_run_ids_and_absolute_time():
    first, evidence = collision_run()
    second, other = collision_run(actor_id="999", time=.4, run_id="second")
    a, b = build_run_signature(first, evidence), build_run_signature(second, other)
    assert a["status"] == b["status"] == "available"
    assert a["digest"] == b["digest"]
    assert a["event_sources"] != b["event_sources"]


def test_role_and_observation_coverage_change_pattern():
    first, evidence = collision_run()
    second, other = collision_run(role="other_subject")
    a, b = build_run_signature(first, evidence), build_run_signature(second, other)
    assert a["digest"] != b["digest"]
    changed = first.to_dict()
    changed["categories"]["task"]["stuck"]["status"] = "absent"
    assert build_run_signature(changed, evidence)["digest"] != a["digest"]


def test_missing_ambiguous_or_reassigned_roles_are_not_equated():
    outcome, evidence = collision_run(role=None)
    assert build_run_signature(outcome, evidence)["status"] == "unknown"
    outcome, evidence = collision_run(role="ego")
    assert build_run_signature(outcome, evidence)["status"] == "unknown"
    outcome, evidence = collision_run()
    evidence[5] = replace(evidence[5], actors=(replace(evidence[5].actors[0], role="reassigned"),))
    assert build_run_signature(outcome, evidence)["status"] == "unknown"


def test_event_interface_uses_only_explicit_stable_context_fields():
    outcome, _ = collision_run()
    event = outcome.events[0]
    assert build_event_signature(event, {"a": {"role": None, "actor_type": "vehicle"}})["status"] == "unknown"
    context = {"a": {"role": "subject", "actor_type": "vehicle", "actor_id": "ephemeral"}}
    pattern = build_event_signature(event, context)["pattern"]
    assert pattern["actors"] == [{"role": "subject", "actor_type": "vehicle"}]


def test_event_order_preserved_and_simultaneous_events_order_independent():
    outcome, evidence = collision_run()
    data = outcome.to_dict()
    first = data["events"][0]
    other = {**first, "event_id": "other", "label": "near_miss", "start_s": .4}
    data["events"] = [first, other]
    original = build_run_signature(data, evidence)["digest"]
    data["events"] = [{**first, "start_s": .4}, {**other, "start_s": .2}]
    assert build_run_signature(data, evidence)["digest"] != original
    data["events"] = [first, {**other, "start_s": first["start_s"]}]
    simultaneous = build_run_signature(data, evidence)["digest"]
    data["events"].reverse()
    assert build_run_signature(data, evidence)["digest"] == simultaneous


def test_empty_events_are_not_a_safe_behaviour_cluster():
    outcome, evidence = collision_run()
    data = outcome.to_dict()
    data["events"] = []
    signature = build_run_signature(data, evidence)
    result = cluster_outcomes([{"record_id": "empty", "signature": signature}])
    assert signature["status"] == "not_applicable"
    assert result["clusters"] == []
    assert len(result["unclustered"]) == 1


def test_clusters_are_deterministic_scoped_and_validate_record_identity():
    outcome, evidence = collision_run()
    signature = build_run_signature(outcome, evidence)
    records = [{"record_id": "b", "signature": signature}, {"record_id": "a", "signature": signature}]
    report = cluster_outcomes(records)
    assert report == cluster_outcomes(reversed(records))
    assert report["clusters"][0]["members"] == ["a", "b"]
    assert report["clusters"][0]["representative"] == "a"
    assert next(iter(report["scopes"].values()))["repeat_fraction_of_assessable_records"] == .5
    with pytest.raises(ValueError):
        cluster_outcomes([records[0], records[0]])
    corrupt = deepcopy(signature)
    corrupt["pattern"]["execution_status"] = "failed"
    with pytest.raises(ValueError):
        cluster_outcomes([{"record_id": "x", "signature": corrupt}])
    data = outcome.to_dict()
    data["profile_digest"] = "different"
    records.append({"record_id": "c", "signature": build_run_signature(data, evidence)})
    assert len(cluster_outcomes(records)["scopes"]) == 2


def test_summary_counts_episodes_and_unknowns_separately():
    outcome, _ = collision_run()
    summary = summarize_diversity([outcome, outcome])
    assert summary["event_episode_counts"]["collision"] == 2
    assert summary["present_record_counts"]["safety.collision"] == 2
    assert summary["unknown_record_counts"]["rules.red_light_violation"] == 2
    assert summary["assessment_state_counts"]["safety.collision"] == {"present": 2, "absent": 0, "unknown": 0, "not_applicable": 0}


def test_cli_replays_checksums_and_refuses_duplicate_traces_or_overwrite(tmp_path):
    paths = []
    for run_id in ("one", "two"):
        outcome, evidence = collision_run(run_id=run_id)
        writer = EvidenceWriter(tmp_path / f"{run_id}.jsonl", run_id)
        for record in evidence:
            writer.append(record)
        paths.append(writer.finish(replace(manifest(), run_id=run_id)))
    output = tmp_path / "report"
    argv = ["--capture-manifest", str(paths[0]), "--capture-manifest", str(paths[1]), "--output-dir", str(output)]
    assert main(argv) == 0
    result = json.loads((output / "diversity.json").read_text())
    assert len(result["clusters"]) == 1
    assert len(result["sources"]) == 2
    with pytest.raises(ValueError, match="more than once"):
        report_captures([paths[0], paths[0]])
    with pytest.raises(SystemExit):
        main(argv)
    (tmp_path / "one.jsonl").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        report_captures(paths)


def test_cumulative_discovery_tracks_unique_and_repeated_patterns_by_attempt():
    from research.harness.outcome_diversity import cumulative_discovery
    first, ev1 = collision_run(run_id="r1", role="ped")
    second, ev2 = collision_run(run_id="r2", role="ped")
    third, ev3 = collision_run(run_id="r3", role="veh")
    records = [
        {"record_id": "r1", "eval_index": 0, "signature": build_run_signature(first, ev1)},
        {"record_id": "r2", "eval_index": 1, "signature": build_run_signature(second, ev2)},
        {"record_id": "r3", "eval_index": 2, "signature": build_run_signature(third, ev3)},
    ]
    disc = cumulative_discovery(records)
    assert disc["n_attempted"] == 3
    assert disc["n_unique_patterns"] == 2
    assert disc["n_repeated_patterns"] == 1
    assert [step["cumulative_unique_patterns"] for step in disc["curve"]] == [1, 1, 2]


def test_enriched_event_signature_distinguishes_severity_and_manoeuvre():
    from research.harness.outcome_models import RoadRuleContext
    r_low, ev_low = collision_run(run_id="r_low", role="ped")
    r_hi, ev_hi = collision_run(run_id="r_hi", role="ped")
    d_low, d_hi = r_low.to_dict(), r_hi.to_dict()
    d_low["events"][0]["measurements"]["peak_impulse_ns"] = 100.0
    d_hi["events"][0]["measurements"]["peak_impulse_ns"] = 9000.0
    ev_low = [replace(x, steer=0.0, road_context=RoadRuleContext(lane_type="driving", in_junction=False)) if hasattr(x, "steer") else x for x in ev_low]
    ev_hi = [replace(x, steer=0.5, road_context=RoadRuleContext(lane_type="driving", in_junction=True)) if hasattr(x, "steer") else x for x in ev_hi]
    # Coarse pattern equates them; enriched pattern distinguishes severity & manoeuvre
    assert build_run_signature(d_low, ev_low)["digest"] == build_run_signature(d_hi, ev_hi)["digest"]
    sig_low = build_run_signature(d_low, ev_low, include_rich_context=True)
    sig_hi = build_run_signature(d_hi, ev_hi, include_rich_context=True)
    assert sig_low["scope"]["criterion"] == "enriched_event_context_v1"
    assert sig_low["digest"] != sig_hi["digest"]
