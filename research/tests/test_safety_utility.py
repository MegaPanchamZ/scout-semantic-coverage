"""Tests for safety-utility analysis (time-to-first and obligation linkage)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.harness.safety_utility import (  # noqa: E402
    ArmFirstIssue,
    EvalRecord,
    load_records,
    obligation_association,
    paired_time_to_first,
    rank_associations,
    summarize_time_to_first,
    summarize_time_to_first_by_replicate,
    time_to_first,
)
from research.scripts.safety_utility_report import (  # noqa: E402
    build_utility_report,
    expand_seed_roots,
)


def _outcome(unsafe: bool = False, **names: bool) -> dict:
    outcome = {name: False for name in ("collision", "near_collision", "unsafe_proximity", "stuck")}
    outcome.update(names)
    outcome["unsafe"] = unsafe
    outcome["reasons"] = [name for name, value in names.items() if value]
    return outcome


def _record(
    policy: str,
    replicate: str,
    route: str,
    eval_index: int,
    unsafe: bool,
    obligations: tuple[str, ...] = (),
    target: str | None = None,
    **names: bool,
) -> EvalRecord:
    return EvalRecord(
        policy=policy,
        route=route,
        replicate=replicate,
        eval_index=eval_index,
        outcome=_outcome(unsafe, **names),
        obligations=obligations,
        target=target,
        run_json_path=f"/tmp/{policy}-{replicate}-{route}-{eval_index}.json",
    )


# --------------------------------------------------------------------------
# Time to first
# --------------------------------------------------------------------------


def test_time_to_first_finds_first_unsafe_and_censors():
    records = [
        _record("semantic", "seed-13", "routeA", 0, False),
        _record("semantic", "seed-13", "routeA", 1, True, near_collision=True),
        _record("semantic", "seed-13", "routeA", 2, True),
        _record("semantic", "seed-13", "routeB", 0, False),
        _record("semantic", "seed-13", "routeB", 1, False),
        _record("random", "seed-13", "routeA", 0, True),
    ]
    arms = time_to_first(records)
    by_key = {(arm.policy, arm.route): arm for arm in arms}
    assert by_key[("semantic", "routeA")].first_eval == 1
    assert by_key[("semantic", "routeA")].censored is False
    assert by_key[("semantic", "routeA")].outcomes == ("near_collision",)
    assert by_key[("semantic", "routeB")].first_eval is None
    assert by_key[("semantic", "routeB")].censored is True
    assert by_key[("semantic", "routeB")].budget == 1


def test_summarize_time_to_first_mean_std_top1():
    records = [
        _record("semantic", "seed-13", "routeA", 0, True),
        _record("semantic", "seed-13", "routeB", 0, False),
        _record("semantic", "seed-13", "routeB", 3, True),
        _record("semantic", "seed-13", "routeC", 0, False),
        _record("semantic", "seed-13", "routeC", 1, False),
    ]
    summary = summarize_time_to_first(time_to_first(records))
    entry = summary["semantic"]
    assert entry["n_arms"] == 3
    assert entry["n_found"] == 2
    assert entry["found_rate"] == 2 / 3
    assert entry["top1_count"] == 1
    assert entry["mean_first_found"] == 1.5  # (0 + 3) / 2
    assert entry["median_first_found"] == 1.5
    # Censored arm (routeC) scores budget + 1 = 2.
    assert abs(entry["mean_first_censored"] - (0 + 3 + 2) / 3) < 1e-9


def test_summarize_by_replicate_and_paired():
    records = []
    for replicate, semantic_first, random_first in (("seed-13", 1, 3), ("seed-23", 2, 4)):
        records.append(_record("semantic", replicate, "routeA", semantic_first, True))
        records.append(_record("random", replicate, "routeA", random_first, True))
    arms = time_to_first(records)
    by_rep = summarize_time_to_first_by_replicate(arms)
    assert by_rep["semantic"]["seed-13"]["mean_first_found"] == 1.0
    assert by_rep["semantic"]["seed-23"]["mean_first_found"] == 2.0
    paired = paired_time_to_first(arms, "semantic", "random")
    assert paired["n_pairs"] == 2
    assert paired["mean_difference"] == -2.0
    assert paired["wins_primary_lower"] == 2


def test_time_to_first_respects_fixed_budget_for_censoring():
    records = [
        _record("semantic", "seed-13", "routeA", 0, False),
        _record("semantic", "seed-13", "routeA", 1, False),
    ]
    arms = time_to_first(records, budget=49)
    assert arms[0].budget == 49
    summary = summarize_time_to_first(arms)
    assert summary["semantic"]["mean_first_censored"] == 50.0


# --------------------------------------------------------------------------
# Obligation -> outcome association
# --------------------------------------------------------------------------


def test_obligation_association_rates_and_risk_ratio():
    crossing = "crossing_path(pedestrian,ego)"
    braking = "braking(vehicle)"
    records = [
        _record("semantic", "seed-13", "routeA", 0, True, (crossing,), near_collision=True),
        _record("semantic", "seed-13", "routeA", 1, False, (crossing,)),
        _record("semantic", "seed-13", "routeA", 2, False, (braking,)),
        _record("semantic", "seed-13", "routeA", 3, False, ()),
    ]
    result = obligation_association(records)
    entry = result["obligations"][crossing]
    assert entry["support"] == 2
    assert entry["unsafe_when_present"] == 1
    assert entry["unsafe_rate_present"] == 0.5
    assert entry["unsafe_rate_absent"] == 0.0
    # absent rate is zero, so the risk ratio is undefined rather than inf
    assert entry["risk_ratio_unsafe"] is None
    assert entry["lift_unsafe"] == 0.5
    assert entry["outcomes"]["near_collision"]["present_hits"] == 1
    assert result["obligations"][braking]["support"] == 1


def test_obligation_association_target_source():
    records = [
        _record("semantic", "seed-13", "routeA", 0, True, (), target="braking(vehicle)"),
        _record("semantic", "seed-13", "routeA", 1, False, (), target="braking(vehicle)"),
    ]
    result = obligation_association(records, source="target")
    assert result["obligations"]["braking(vehicle)"]["support"] == 2
    assert result["obligations"]["braking(vehicle)"]["unsafe_rate_present"] == 0.5


def test_rank_associations_orders_by_lift():
    records = [
        _record("semantic", "seed-13", "r", 0, True, ("a",), near_collision=True),
        _record("semantic", "seed-13", "r", 1, True, ("a",), near_collision=True),
        _record("semantic", "seed-13", "r", 2, False, ("b",)),
        _record("semantic", "seed-13", "r", 3, False, ("b",)),
    ]
    ranked = rank_associations(obligation_association(records), "unsafe")
    assert [item["obligation"] for item in ranked][0] == "a"


# --------------------------------------------------------------------------
# Log loading and replicate inference
# --------------------------------------------------------------------------


def _write_run(path: Path, unsafe: bool) -> None:
    payload = {
        "reached_goal": True,
        "ticks_executed": 100,
        "collision_count": 0,
        "terminated_by_collision": False,
        "safety_metrics": {
            "min_pedestrian_distance_m": 1.0 if unsafe else 10.0,
            "min_vehicle_distance_m": 10.0,
        },
    }
    path.write_text(json.dumps(payload))


def test_load_records_infers_seed_replicate_and_obligations(tmp_path):
    route_dir = "town01_spawn0_goal82_benign_seed0"  # must NOT be read as a replicate
    semantic = tmp_path / "seed-13" / route_dir / "semantic"
    semantic.mkdir(parents=True)
    run = semantic / "run.json"
    _write_run(run, unsafe=True)
    row = {
        "policy": "semantic",
        "route": route_dir,
        "eval_index": 4,
        "run_json_path": str(run),
        "engine_run_obligations": ["braking(vehicle)", "same_lane(vehicle,ego)"],
        "hazard_target": "braking(vehicle)",
    }
    (semantic / "rows.jsonl").write_text(json.dumps(row) + "\n")

    records, counters = load_records(tmp_path)
    assert counters["n_rows"] == 1
    assert len(records) == 1
    record = records[0]
    assert record.replicate == "seed-13"
    assert record.eval_index == 4
    assert record.obligations == ("braking(vehicle)", "same_lane(vehicle,ego)")
    assert record.target == "braking(vehicle)"
    assert record.outcome["unsafe_proximity"] is True


def test_load_records_counts_missing_and_falls_back_to_semantic_obligations(tmp_path):
    semantic = tmp_path / "seed-23" / "routeA" / "random"
    semantic.mkdir(parents=True)
    good = semantic / "run.json"
    _write_run(good, unsafe=False)
    rows = [
        {"policy": "random", "route": "routeA", "eval_index": 0, "run_json_path": str(good),
         "semantic_fulfilled_obligations": ["braking(vehicle)"]},
        {"policy": "random", "route": "routeA", "eval_index": 1, "run_json_path": str(tmp_path / "nope.json")},
    ]
    (semantic / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    records, counters = load_records(tmp_path)
    assert counters["n_rows"] == 2
    assert counters["n_missing_run_path"] == 1
    assert len(records) == 1
    assert records[0].obligations == ("braking(vehicle)",)


def test_expand_seed_roots(tmp_path):
    (tmp_path / "seed-13").mkdir()
    (tmp_path / "seed-23").mkdir()
    expanded = expand_seed_roots([tmp_path])
    assert {path.name for path in expanded} == {tmp_path.name, "seed-13", "seed-23"}


def test_build_utility_report_end_to_end(tmp_path):
    for policy, first in (("semantic", 1), ("random", 3)):
        arm = tmp_path / "seed-13" / "routeA" / policy
        arm.mkdir(parents=True)
        rows = []
        for index in range(first + 1):
            run = arm / f"run-{index}.json"
            _write_run(run, unsafe=(index == first))
            rows.append(
                {
                    "policy": policy,
                    "route": "routeA",
                    "eval_index": index,
                    "run_json_path": str(run),
                    "engine_run_obligations": ["braking(vehicle)"] if index == first else [],
                }
            )
        (arm / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    result = build_utility_report([tmp_path])
    summary = result["time_to_first"]["summary"]
    assert summary["semantic"]["mean_first_found"] == 1.0
    assert summary["random"]["mean_first_found"] == 3.0
    paired = result["paired_time_to_first"][0]
    assert paired["mean_difference"] == -2.0
    assert result["association"]["top_witnessed"]


def test_v2_ledger_retains_missing_unreadable_malformed_and_enforces_budget_cutoff(tmp_path):
    arm = tmp_path / "seed-7" / "routeZ" / "semantic"
    arm.mkdir(parents=True)
    bad_json = arm / "broken.json"
    bad_json.write_text("{not-json")
    rows = [
        json.dumps({"policy": "semantic", "route": "routeZ", "eval_index": 0, "attempt_id": "id0", "run_error": "setup fail"}),
        json.dumps({"policy": "semantic", "route": "routeZ", "eval_index": 1, "attempt_id": "id1", "run_json_path": str(bad_json)}),
        json.dumps({"policy": "semantic", "route": "routeZ", "eval_index": 1, "attempt_id": "id1", "run_json_path": str(bad_json)}),  # duplicate
        "{corrupt line",
        json.dumps({
            "policy": "semantic", "route": "routeZ", "eval_index": 5, "attempt_id": "id5",
            "outcome_v2": {
                "schema_version": 2, "run_id": "r5", "execution_status": "completed", "termination_reason": "collision_stop",
                "categories": {"safety": {"collision": {"status": "present"}}},
                "aggregates": {"safety_event": {"status": "present"}},
            },
        }),
    ]
    (arm / "rows.jsonl").write_text("\n".join(rows))

    records, counters = load_records(tmp_path, schema_version=2)
    assert len(records) == 3  # id0 (missing/failed), id1 (unreadable), id5 (embedded v2)
    assert counters["n_malformed_rows"] == 1
    assert counters["n_duplicate_rows"] == 1
    assert counters["n_unreadable_run"] == 1
    assert counters["n_failed_attempts"] == 1

    # Budget cutoff at 5 (attempts 0..4) must exclude 0-indexed eval_index=5 (the 6th attempt)!
    arms_five = time_to_first(records, budget=5, metric="safety.collision", exclude_eval_zero=True)
    assert arms_five[0].first_eval is None
    assert arms_five[0].censored is True
    assert arms_five[0].censoring_reason == "right_censored_with_unknown"

    # With budget=6 (attempts 0..5), eval_index=5 is found, with 4 unknown priors (eval 1 unreadable + evals 2..4 missing)
    arms_full = time_to_first(records, budget=6, metric="safety.collision", exclude_eval_zero=True)
    assert arms_full[0].first_eval == 5
    assert arms_full[0].n_unknown_prior == 4
    assert arms_full[0].censoring_reason == "found_with_unknown_prior"


def test_v2_dedup_preserves_multi_route_and_shared_eval0_and_rank_associations_primary(tmp_path):
    # Same attempt_id across two different routes/policies (e.g. shared eval 0) must NOT be dropped
    for route, policy, hit in (("routeA", "semantic", True), ("routeB", "semantic", False), ("routeA", "random", False)):
        arm_dir = tmp_path / "seed-1" / route / policy
        arm_dir.mkdir(parents=True)
        row = {
            "policy": policy,
            "route": route,
            "eval_index": 0,
            "attempt_id": "shared-eval-0",
            "engine_run_obligations": ["ob_x"] if hit else [],
            "outcome_v2": {
                "schema_version": 2,
                "run_id": "shared-eval-0",
                "execution_status": "completed",
                "categories": {"safety": {"collision": {"status": "present" if hit else "absent"}}},
                "aggregates": {"safety_event": {"status": "present" if hit else "absent"}},
            },
        }
        (arm_dir / "rows.jsonl").write_text(json.dumps(row) + "\n")

    records, counters = load_records(tmp_path, schema_version=2)
    assert len(records) == 3
    assert counters["n_duplicate_rows"] == 0

    assoc = obligation_association(records, metric="safety_event")
    ranked = rank_associations(assoc, "safety_event")
    assert len(ranked) == 1
    assert ranked[0]["obligation"] == "ob_x"
    assert ranked[0]["lift"] == 1.0

    # Paired v2 comparison refuses unmatched_pooled replicates and mismatched budgets
    pooled_arms = [
        ArmFirstIssue("semantic", "all", "r1", 1, 5, False, metric="safety_event", replicate_provenance="unmatched_pooled"),
        ArmFirstIssue("random", "all", "r1", 3, 5, False, metric="safety_event", replicate_provenance="unmatched_pooled"),
        ArmFirstIssue("semantic", "seed-1", "r2", 1, 5, False, metric="safety_event", replicate_provenance="stored:replicate_id"),
        ArmFirstIssue("random", "seed-1", "r2", 3, 4, False, metric="safety_event", replicate_provenance="stored:replicate_id"),
    ]
    paired = paired_time_to_first(pooled_arms, "semantic", "random")
    assert paired["n_pairs"] == 0
    assert paired["n_excluded_unmatched_pooled"] == 1
    assert paired["n_excluded_budget_mismatch"] == 1


def test_v2_time_to_first_incomplete_budget_unindexed_unknown_top1_and_reclassify_embedded(tmp_path):
    from research.harness.outcome_config import load_outcome_profile, profile_digest
    from research.scripts.reclassify_outcomes import audit_archive

    def _v2_rec(eval_idx, status_val, attempt_id):
        return EvalRecord(
            policy="semantic", route="r1", replicate="seed-1", eval_index=eval_idx,
            outcome={
                "schema_version": 2, "run_id": attempt_id, "execution_status": "completed",
                "categories": {"safety": {"collision": {"status": status_val}}},
                "aggregates": {"safety_event": {"status": status_val}},
            },
            obligations=(), target=None, run_json_path=None, attempt_id=attempt_id,
            schema_version=2, replicate_provenance="stored:replicate_id",
        )

    # 1. Budget=5 with only evals 0 and 1 absent (3 missing slots) must NOT be right_censored_complete!
    partial_recs = [_v2_rec(0, "absent", "a0"), _v2_rec(1, "absent", "a1")]
    arms_inc = time_to_first(partial_recs, budget=5, metric="safety.collision", exclude_eval_zero=True)
    assert arms_inc[0].censoring_reason == "right_censored_with_unknown"

    # 2. Unindexed unknown attempt AND missing prior slots count as prior uncertainty for an indexed hit
    unidx_recs = [_v2_rec(0, "absent", "a0"), _v2_rec(3, "present", "a3"), _v2_rec(None, "unknown", "a_none")]
    arms_unidx = time_to_first(unidx_recs, budget=5, metric="safety.collision", exclude_eval_zero=True)
    assert arms_unidx[0].first_eval == 3
    assert arms_unidx[0].n_unknown_prior == 3  # 1 unindexed unknown + 2 missing prior indices (1, 2)
    assert arms_unidx[0].censoring_reason == "found_with_unknown_prior"
    # Hit at eval 3 when eligible evals start at 3 (no earlier unindexed row) earns Top-1 credit
    arms_first_3 = time_to_first([_v2_rec(0, "absent", "a0"), _v2_rec(3, "present", "a3")], budget=5, metric="safety.collision", exclude_eval_zero=True)
    summary = summarize_time_to_first(arms_first_3)
    assert summary["semantic"]["top1_count"] == 1

    # 2b. Missing evals 1..3 before hit at eval 4 -> n_unknown_prior == 3, found_with_unknown_prior
    gap_recs = [_v2_rec(0, "absent", "g0"), _v2_rec(4, "present", "g4")]
    arms_gap = time_to_first(gap_recs, budget=5, metric="safety.collision", exclude_eval_zero=True)
    assert arms_gap[0].n_unknown_prior == 3
    assert arms_gap[0].censoring_reason == "found_with_unknown_prior"

    # 3. reclassify_outcomes preserves embedded_v2 (default profile & promoted row-only) and survives corrupt row-only manifest
    prof = load_outcome_profile()
    arm_dir = tmp_path / "sem"
    arm_dir.mkdir()
    run_file = arm_dir / "run.json"
    emb = {
        "schema_version": 2, "run_id": "r_emb", "execution_status": "completed",
        "profile_digest": profile_digest(prof),
        "categories": {"safety": {"near_miss": {"status": "present"}}},
    }
    run_file.write_text(json.dumps({"metadata": {"outcome_v2": emb}}))
    bad_manifest = arm_dir / "bad.manifest.json"
    bad_manifest.write_text("{corrupt")
    (arm_dir / "rows.jsonl").write_text(
        "\n".join([
            json.dumps({"policy": "sem", "run_json_path": str(run_file)}),
            json.dumps({"policy": "sem", "capture_manifest_path": str(bad_manifest)}),
            json.dumps({"policy": "sem", "outcome_v2": {**emb, "profile_digest": "promoted-digest"}}),
        ]) + "\n"
    )
    audited = audit_archive(tmp_path, profile=prof)
    assert audited[0]["status"] == "embedded_v2"
    assert audited[0]["outcome"]["categories"]["safety"]["near_miss"]["status"] == "present"
    assert audited[1]["status"] == "unreadable_run"  # corrupt row-only manifest did not abort audit
    # With profile=None (CLI without --profile), row-only promoted-profile embedded_v2 is preserved!
    audited_cli_default = audit_archive(tmp_path, profile=None)
    assert audited_cli_default[2]["status"] == "embedded_v2"


def test_v2_budget_excludes_out_of_budget_unindexed_and_bounds_unindexed_first_eval_and_reclassify_provenance(tmp_path):
    import pytest
    from research.harness.outcome_config import load_outcome_profile, profile_digest
    from research.harness.outcome_io import EvidenceReadError, load_capture_manifest
    from research.scripts import reclassify_outcomes as ro

    def _v2_rec(eval_idx, status_val, attempt_id):
        return EvalRecord(
            policy="semantic", route="r1", replicate="seed-1", eval_index=eval_idx,
            outcome={
                "schema_version": 2, "run_id": attempt_id, "execution_status": "completed",
                "categories": {"safety": {"collision": {"status": status_val}}},
                "aggregates": {"safety_event": {"status": status_val}},
            },
            obligations=(), target=None, run_json_path=None, attempt_id=attempt_id,
            schema_version=2, replicate_provenance="stored:replicate_id",
        )

    # 1. Under budget=3 with indexed evals 0, 1, 2 all absent, an extra unindexed unknown row is outside budget
    #    and must NOT turn clean right-censoring into right_censored_with_unknown.
    full_three_plus_extra = [
        _v2_rec(0, "absent", "a0"),
        _v2_rec(1, "absent", "a1"),
        _v2_rec(2, "absent", "a2"),
        _v2_rec(None, "unknown", "extra_unindexed"),
    ]
    arm_clean = time_to_first(full_three_plus_extra, budget=3, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_clean.censored is True
    assert arm_clean.n_unknown_prior == 0
    assert arm_clean.censoring_reason == "right_censored_complete"

    # 2. Under budget=3 (valid 0-indexed slots 0..2), when indexed evals 1 and 2 are both absent,
    #    every eligible slot (1..2) is already a known negative -> unindexed hit must NOT be assigned to slot 2!
    missing_zero_unidx_hit = [
        _v2_rec(1, "absent", "b1"),
        _v2_rec(2, "absent", "b2"),
        _v2_rec(None, "present", "b_unidx"),
    ]
    arm_hit = time_to_first(missing_zero_unidx_hit, budget=3, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_hit.first_eval is None

    # 2b. Under budget=3 with indexed evals 0 and 2 absent, an unindexed hit takes the sole free eligible slot (1) AND earns Top-1!
    free_slot_one = [
        _v2_rec(0, "absent", "c0"),
        _v2_rec(2, "absent", "c2"),
        _v2_rec(None, "present", "c_unidx"),
    ]
    arms_free1 = time_to_first(free_slot_one, budget=3, metric="safety.collision", exclude_eval_zero=True)
    assert arms_free1[0].first_eval == 1
    assert arms_free1[0].first_eligible_eval == 1
    assert summarize_time_to_first(arms_free1)["semantic"]["top1_count"] == 1

    # 2b2. When BOTH unindexed (uniquely slot 1) AND indexed slot 2 are present, first_eval must be 1 (NOT 2)!
    both_pos = [
        _v2_rec(0, "absent", "c0"),
        _v2_rec(2, "present", "c2"),
        _v2_rec(None, "present", "c_unidx"),
    ]
    arm_both = time_to_first(both_pos, budget=3, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_both.first_eval == 1

    # 2c. Without explicit budget, a single indexed hit at 3 earns Top-1 credit (first_eligible_eval not clamped to 0)
    single_unbounded = [_v2_rec(3, "present", "d3")]
    arm_single = time_to_first(single_unbounded, budget=None, metric="safety.collision", exclude_eval_zero=True)
    assert arm_single[0].first_eval == 3
    assert arm_single[0].first_eligible_eval == 3
    assert summarize_time_to_first(arm_single)["semantic"]["top1_count"] == 1

    # 2d. Sparse unbounded arm with missing slots (0 and 3 absent, 1 & 2 missing) must NOT be right_censored_complete!
    sparse_unbounded = [_v2_rec(0, "absent", "s0"), _v2_rec(3, "absent", "s3")]
    arm_sparse = time_to_first(sparse_unbounded, budget=None, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_sparse.censoring_reason == "right_censored_with_unknown"

    # 2e. Distinct attempts sharing an index in an unbounded arm must NOT hide missing slot 1!
    dup_idx_unbounded = [_v2_rec(0, "absent", "u0"), _v2_rec(2, "absent", "u2a"), _v2_rec(2, "absent", "u2b")]
    arm_dup = time_to_first(dup_idx_unbounded, budget=None, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_dup.censoring_reason == "right_censored_with_unknown"

    # 2f. Uniquely assigned negative before a hit must NOT count as missing prior slot!
    uniq_neg_before = [_v2_rec(0, "absent", "n0"), _v2_rec(2, "present", "n2"), _v2_rec(None, "absent", "n_slot1")]
    arm_uneg = time_to_first(uniq_neg_before, budget=3, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_uneg.first_eval == 2
    assert arm_uneg.n_unknown_prior == 0
    assert arm_uneg.censoring_reason == "found"

    # 2g. Uniquely assigned unknown AFTER a hit must NOT count as prior uncertainty!
    uniq_unk_after = [_v2_rec(0, "absent", "k0"), _v2_rec(1, "present", "k1"), _v2_rec(None, "unknown", "k_slot2")]
    arm_uunk = time_to_first(uniq_unk_after, budget=3, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_uunk.first_eval == 1
    assert arm_uunk.n_unknown_prior == 0
    assert arm_uunk.censoring_reason == "found"

    # 2h. Multiple unindexed rows when EVERY possible free slot follows the hit must NOT count as prior uncertainty!
    multi_after_hit = [
        _v2_rec(0, "absent", "m0"),
        _v2_rec(1, "present", "m1"),
        _v2_rec(None, "unknown", "m_u1"),
        _v2_rec(None, "absent", "m_u2"),
    ]
    arm_multi = time_to_first(multi_after_hit, budget=5, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_multi.first_eval == 1
    assert arm_multi.n_unknown_prior == 0
    assert arm_multi.censoring_reason == "found"

    # 2i. Negative eval_index must NOT hide a missing eligible slot!
    neg_idx = [_v2_rec(0, "absent", "neg0"), _v2_rec(-1, "absent", "neg_bad"), _v2_rec(2, "absent", "neg2")]
    arm_neg = time_to_first(neg_idx, budget=3, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_neg.censoring_reason == "right_censored_with_unknown"

    # 2j. Arm whose rows ALL have negative indices must NOT vanish from summary denominator!
    all_neg = [_v2_rec(-1, "absent", "only_neg")]
    arms_all_neg = time_to_first(all_neg, budget=None, metric="safety.collision", exclude_eval_zero=True)
    assert len(arms_all_neg) == 1
    assert arms_all_neg[0].censored is True
    assert arms_all_neg[0].censoring_reason == "right_censored_with_unknown"
    assert summarize_time_to_first(arms_all_neg)["semantic"]["n_arms"] == 1

    # 2k. Swapping attempt_ids of ambiguous unindexed hit + unindexed unknown must NEVER flip result to clean Top-1!
    order_ab = [_v2_rec(0, "absent", "z0"), _v2_rec(None, "present", "a_id"), _v2_rec(None, "unknown", "b_id")]
    order_ba = [_v2_rec(0, "absent", "z0"), _v2_rec(None, "unknown", "a_id"), _v2_rec(None, "present", "b_id")]
    arm_ab = time_to_first(order_ab, budget=3, metric="safety.collision", exclude_eval_zero=True)[0]
    arm_ba = time_to_first(order_ba, budget=3, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_ab.first_eval == arm_ba.first_eval == 2
    assert arm_ab.censoring_reason == arm_ba.censoring_reason == "found_with_unknown_prior"
    assert summarize_time_to_first([arm_ab])["semantic"]["top1_count"] == 0

    # 2l. Unindexed hit when eval 0 is unobserved (budget=3, eval 2 absent, None present) could be excluded eval 0 -> NO discovery / NO Top-1!
    unidx_possible_eval0 = [_v2_rec(2, "absent", "p2"), _v2_rec(None, "present", "p_none")]
    arms_e0 = time_to_first(unidx_possible_eval0, budget=3, metric="safety.collision", exclude_eval_zero=True)
    assert arms_e0[0].first_eval is None
    assert arms_e0[0].censoring_reason == "right_censored_with_unknown"
    assert summarize_time_to_first(arms_e0)["semantic"]["top1_count"] == 0

    # 2m. Swapping attempt_ids between unindexed absent and not_applicable must NOT change n_assessable!
    na_ab = [_v2_rec(2, "absent", "q2"), _v2_rec(None, "absent", "id_a"), _v2_rec(None, "not_applicable", "id_b")]
    na_ba = [_v2_rec(2, "absent", "q2"), _v2_rec(None, "not_applicable", "id_a"), _v2_rec(None, "absent", "id_b")]
    arm_na_ab = time_to_first(na_ab, budget=3, metric="safety.collision", exclude_eval_zero=True)[0]
    arm_na_ba = time_to_first(na_ba, budget=3, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_na_ab.n_assessable == arm_na_ba.n_assessable

    # 2n. Partially observed arm with unindexed hits must NOT claim unsupported Top-1 credit!
    #     With budget=4, eval 2 absent, and two unindexed hits, the hits could occupy evals 0 and 3.
    partial_two_hits = [_v2_rec(2, "absent", "r2"), _v2_rec(None, "present", "r_u1"), _v2_rec(None, "present", "r_u2")]
    arms_p2h = time_to_first(partial_two_hits, budget=4, metric="safety.collision", exclude_eval_zero=True)
    assert arms_p2h[0].first_eval == 3
    assert arms_p2h[0].censoring_reason == "found_with_unknown_prior"
    assert summarize_time_to_first(arms_p2h)["semantic"]["top1_count"] == 0
    # Same arm without explicit budget must also place the remaining unindexed hit after eval 2 (no Top-1 credit)
    arms_p2h_unb = time_to_first(partial_two_hits, budget=None, metric="safety.collision", exclude_eval_zero=True)
    assert arms_p2h_unb[0].first_eval == 3
    assert arms_p2h_unb[0].censoring_reason == "found_with_unknown_prior"
    assert summarize_time_to_first(arms_p2h_unb)["semantic"]["top1_count"] == 0
    # Single unindexed hit with budget=4 and eval 0 absent could occupy eval 3 (not guaranteed eval 1 -> no Top-1 credit)
    sparse_one_hit = [_v2_rec(0, "absent", "s0"), _v2_rec(None, "present", "s_u1")]
    arms_s1h = time_to_first(sparse_one_hit, budget=4, metric="safety.collision", exclude_eval_zero=True)
    assert arms_s1h[0].first_eval == 3
    assert arms_s1h[0].censoring_reason == "found_with_unknown_prior"
    assert summarize_time_to_first(arms_s1h)["semantic"]["top1_count"] == 0
    # Unindexed row with unobserved eval 0 and indexed hit at eval 2 could occupy eval 1 -> no Top-1 credit
    unidx_before_indexed = [_v2_rec(2, "present", "t2"), _v2_rec(None, "not_applicable", "t_u1")]
    arms_ubi = time_to_first(unidx_before_indexed, budget=3, metric="safety.collision", exclude_eval_zero=True)
    assert arms_ubi[0].first_eval == 2
    assert arms_ubi[0].first_eligible_eval == 1
    assert summarize_time_to_first(arms_ubi)["semantic"]["top1_count"] == 0

    # 2o. Budget cutoff over competing unindexed rows must NOT hide prior uncertainty when a free slot exists!
    overflow_found = [_v2_rec(0, "absent", "o0"), _v2_rec(2, "present", "o2"), _v2_rec(None, "absent", "ou1"), _v2_rec(None, "unknown", "ou2")]
    arm_ovf = time_to_first(overflow_found, budget=3, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_ovf.first_eval == 2
    assert arm_ovf.n_unknown_prior > 0
    assert arm_ovf.censoring_reason == "found_with_unknown_prior"
    overflow_cens = [_v2_rec(0, "absent", "o0"), _v2_rec(2, "absent", "o2"), _v2_rec(None, "absent", "ou1"), _v2_rec(None, "unknown", "ou2")]
    arm_ovc = time_to_first(overflow_cens, budget=3, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_ovc.first_eval is None
    assert arm_ovc.censoring_reason == "right_censored_with_unknown"

    # 2p. Unindexed hit preceded ONLY by known indexed absent slot(s) must NOT acquire impossible prior uncertainty!
    known_prior_two_hits = [_v2_rec(1, "absent", "kp1"), _v2_rec(None, "present", "kpu1"), _v2_rec(None, "present", "kpu2")]
    for b_val in (3, None):
        arm_kp = time_to_first(known_prior_two_hits, budget=b_val, metric="safety.collision", exclude_eval_zero=True)[0]
        assert arm_kp.first_eval == 2
        assert arm_kp.n_unknown_prior == 0
        assert arm_kp.censoring_reason == "found"

    # 2q. Swapping attempt_ids on duplicate indexed attempts must NOT flip discovery, Top-1 credit, or outcomes tuple!
    dup_ab = [_v2_rec(0, "absent", "d0"), _v2_rec(1, "absent", "id_a"), _v2_rec(1, "present", "id_b")]
    dup_ba = [_v2_rec(0, "absent", "d0"), _v2_rec(1, "present", "id_a"), _v2_rec(1, "absent", "id_b")]
    arms_dab = time_to_first(dup_ab, budget=2, metric="safety.collision", exclude_eval_zero=True)
    arms_dba = time_to_first(dup_ba, budget=2, metric="safety.collision", exclude_eval_zero=True)
    assert arms_dab[0].first_eval == arms_dba[0].first_eval is None
    assert arms_dab[0].censoring_reason == arms_dba[0].censoring_reason == "right_censored_with_unknown"
    assert summarize_time_to_first(arms_dab)["semantic"]["top1_count"] == summarize_time_to_first(arms_dba)["semantic"]["top1_count"] == 0

    def _v2_custom(eval_idx, cat_key, status_val, agg_status, attempt_id):
        return EvalRecord(
            policy="semantic", route="r1", replicate="seed-1", eval_index=eval_idx,
            outcome={
                "schema_version": 2, "run_id": attempt_id, "execution_status": "completed",
                "categories": {"safety": {cat_key: {"status": status_val}}},
                "aggregates": {"safety_event": {"status": agg_status}},
            },
            obligations=(), target=None, run_json_path=None, attempt_id=attempt_id,
            schema_version=2, replicate_provenance="stored:replicate_id",
        )

    dup_out_ab = [
        _v2_custom(0, "collision", "absent", "absent", "o0"),
        _v2_custom(1, "collision", "present", "present", "id_a"),
        _v2_custom(1, "near_miss", "present", "present", "id_b"),
    ]
    dup_out_ba = [
        _v2_custom(0, "collision", "absent", "absent", "o0"),
        _v2_custom(1, "near_miss", "present", "present", "id_a"),
        _v2_custom(1, "collision", "present", "present", "id_b"),
    ]
    arm_do_ab = time_to_first(dup_out_ab, budget=2, metric="safety_event", exclude_eval_zero=True)[0]
    arm_do_ba = time_to_first(dup_out_ba, budget=2, metric="safety_event", exclude_eval_zero=True)[0]
    assert arm_do_ab.outcomes == arm_do_ba.outcomes

    # 2r. Truncated sole unindexed row must NOT mark a free slot as uniquely known and erase prior uncertainty!
    trunc_unidx = [
        _v2_rec(0, "absent", "tu0"),
        _v2_rec(2, "absent", "tu2a"),
        _v2_rec(2, "present", "tu2b"),
        _v2_rec(None, "unknown", "tu_none"),
    ]
    arm_tu = time_to_first(trunc_unidx, budget=3, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_tu.first_eval == 2
    assert arm_tu.n_unknown_prior > 0
    assert arm_tu.censoring_reason == "found_with_unknown_prior"

    # 2s. Malformed non-string status (e.g. dict) normalizes to unknown and never yields right_censored_complete!
    bad_status_recs = [_v2_custom(0, "collision", "absent", "absent", "bs0"), _v2_custom(1, "collision", {"bad": 1}, {"bad": 1}, "bs1")]
    arm_bs = time_to_first(bad_status_recs, budget=2, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_bs.n_assessable == 0
    assert arm_bs.n_unknown_prior == 1
    assert arm_bs.censoring_reason == "right_censored_with_unknown"

    # 2t. Multiple unindexed absent rows filling ALL free eligible slots before an indexed hit -> n_unknown_prior == 0!
    all_neg_fill = [
        _v2_rec(0, "absent", "an0"),
        _v2_rec(3, "present", "an3"),
        _v2_rec(None, "absent", "anu1"),
        _v2_rec(None, "absent", "anu2"),
    ]
    arm_an = time_to_first(all_neg_fill, budget=4, metric="safety.collision", exclude_eval_zero=True)[0]
    assert arm_an.first_eval == 3
    assert arm_an.n_unknown_prior == 0
    assert arm_an.censoring_reason == "found"

    # 2u. Distinct attempts sharing eval 1 (one absent, one present) under budget=3 must NOT earn Top-1 credit!
    dup_eval1_mixed = [_v2_rec(0, "absent", "m0"), _v2_rec(1, "absent", "m1a"), _v2_rec(1, "present", "m1b")]
    arms_d1m = time_to_first(dup_eval1_mixed, budget=3, metric="safety.collision", exclude_eval_zero=True)
    assert arms_d1m[0].first_eval == 1
    assert arms_d1m[0].top1_eligible is False
    assert summarize_time_to_first(arms_d1m)["semantic"]["top1_count"] == 0

    # 2v. Schema-v2 safety_outcome_report must NOT crash on malformed non-dict categories, metrics, aggregates, or events,
    #     and must ignore invalid actor_ids entries (None, dicts, lists, empty strings)!
    from research.scripts import safety_outcome_report as sor
    mal_root = tmp_path / "malformed_v2_report"
    mal_arm = mal_root / "seed-1" / "routeA" / "semantic"
    mal_arm.mkdir(parents=True)
    mal_rows = [
        {"policy": "semantic", "route": "routeA", "eval_index": 0, "attempt_id": "m1",
         "outcome_v2": {"schema_version": 2, "execution_status": "completed", "categories": ["invalid"], "aggregates": {"safety_event": 42}, "events": 99}},
        {"policy": "semantic", "route": "routeA", "eval_index": 1, "attempt_id": "m2",
         "outcome_v2": {
             "schema_version": 2, "execution_status": "completed",
             "categories": {"safety": {"collision": "invalid"}, "quality": "bad"},
             "aggregates": ["bad"],
             "events": [{"label": "collision", "actor_ids": [None, {}, [], "", "real-actor"]}],
         }},
    ]
    (mal_arm / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in mal_rows) + "\n")
    mal_rep = sor.build_report(mal_root, schema_version=2)
    sem_bucket = mal_rep["policies"]["semantic"]
    assert sem_bucket["n_attempted"] == 2
    assert sem_bucket["metrics"]["safety.collision"]["unknown"] == 2
    assert sem_bucket["metrics"]["safety.collision"]["distinct_actors"] == 1
    assert sem_bucket["aggregates"]["safety_event"]["unknown"] == 2

    # 2w. Truncated unindexed row must STILL prevent Top-1 credit when eval 1 is free before indexed eval 2 hits!
    dup_trunc_top1 = [
        _v2_rec(0, "absent", "dt0"),
        _v2_rec(2, "present", "dt2a"),
        _v2_rec(2, "present", "dt2b"),
        _v2_rec(None, "unknown", "dt_none"),
    ]
    arms_dtt = time_to_first(dup_trunc_top1, budget=3, metric="safety.collision", exclude_eval_zero=True)
    assert arms_dtt[0].first_eval == 2
    assert arms_dtt[0].first_eligible_eval == 1
    assert arms_dtt[0].top1_eligible is False
    assert summarize_time_to_first(arms_dtt)["semantic"]["top1_count"] == 0

    # 2x. Criticality, baselines gap_fitness, and guidance diagnostics must handle corrupt v2 rows safely without raising or fabricating!
    import importlib.util as _ilu
    from research.harness.criticality import criticality_score
    from research.harness.outcome_linkage import build_guidance_diagnostics
    assert criticality_score({"outcome_v2": {"schema_version": 2, "execution_status": []}}, feedback_version="v2_normalized") is None
    assert criticality_score({"outcome_v2": {"schema_version": 2, "execution_status": {}}}, feedback_version="v2_normalized") is None
    assert criticality_score({
        "outcome_v2": {"schema_version": 2, "execution_status": "garbage", "categories": {"safety": {"collision": {"status": "present"}}}}
    }, feedback_version="v2_normalized") is None
    assert criticality_score({
        "outcome_v2": {
            "schema_version": 2, "execution_status": "completed",
            "categories": {"safety": {"collision": 42, "near_miss": {"status": "absent"}}},
        }
    }, feedback_version="v2_normalized") is None
    assert criticality_score({
        "outcome_v2": {
            "schema_version": 2, "execution_status": "completed",
            "categories": {
                "safety": {"collision": {"status": "absent"}, "near_miss": {"status": "present"}},
                "behaviour": {"close_proximity": {"status": "absent"}},
            },
        }
    }, feedback_version="v2_normalized") is None
    assert criticality_score({
        "outcome_v2": {
            "schema_version": 2, "execution_status": "completed",
            "categories": {"safety": {"collision": {"status": "absent"}}},
            "events": [{"measurements": {"min_gap_m": 10**309}}],
        }
    }, feedback_version="v2_normalized") is None
    # Confirmed near_miss with unrelated 100000m gap rounding to 0.0 must NOT return 0.0!
    assert criticality_score({
        "outcome_v2": {
            "schema_version": 2, "execution_status": "completed",
            "categories": {
                "safety": {"collision": {"status": "absent"}, "near_miss": {"status": "present"}},
                "behaviour": {"close_proximity": {"status": "absent"}},
            },
            "events": [{"measurements": {"min_gap_m": 100000.0}}],
        }
    }, feedback_version="v2_normalized") is None
    diag_corrupt = build_guidance_diagnostics([
        {
            "policy": "semantic", "target_obligation": "ob1", "target_witnessed": True,
            "engine_new_obligations": 1, "fitness": 10**309, "outcome_v2": [], "terminated_by_collision": False,
        },
        {
            "policy": "semantic", "target_obligation": "ob1", "target_witnessed": True,
            "engine_new_obligations": 1, "outcome_schema_version": 2, "outcome_v2": None, "terminated_by_collision": True,
        },
        {
            "policy": "semantic", "target_obligation": "ob1", "target_witnessed": True,
            "engine_new_obligations": 1,
            "outcome_v2": {"schema_version": 1, "execution_status": "completed", "aggregates": {"safety_event": {"status": "present"}}},
        },
    ])
    assert diag_corrupt["denominators"]["n_outcome_unknown"] == 3
    assert diag_corrupt["denominators"]["eligible_for_outcome"] == 0
    assert diag_corrupt["failure_stage_counts"] == {"outcome_unknown": 3}
    # Policy search _row_from_run under outcome_schema=2 stamps outcome_schema_version=2 even without embedded outcome_v2
    _ps_path = REPO_ROOT / "research" / "experiments" / "EXP-020-policy-comparison" / "proof-of-concept" / "policy_search.py"
    _ps_spec = _ilu.spec_from_file_location("_test_policy_search_r18", _ps_path)
    _ps_mod = _ilu.module_from_spec(_ps_spec)
    _ps_spec.loader.exec_module(_ps_mod)
    legacy_run_under_v2 = _ps_mod._row_from_run(
        {"reached_goal": True, "ticks_executed": 50, "collision_count": 1, "terminated_by_collision": True},
        1.0,
        attempt_id="att-legacy-v2",
        outcome_schema=2,
    )
    assert legacy_run_under_v2["outcome_schema_version"] == 2
    legacy_run_under_v2.update({"target_obligation": "ob1", "target_witnessed": True, "engine_new_obligations": 1})
    diag_ps_v2 = build_guidance_diagnostics([legacy_run_under_v2])
    assert diag_ps_v2["denominators"]["n_outcome_unknown"] == 1
    assert diag_ps_v2["denominators"]["eligible_for_outcome"] == 0

    # 2y. Guidance diagnostics report honors sibling campaign_protocol.json (outcome_schema=2) for unmarked rows,
    #     and outcome_linkage recognizes v2 markers (backend_capabilities / feedback_version) without legacy fallback!
    from research.scripts import guidance_diagnostics_report as gdr
    gdr_dir = tmp_path / "resumed_v2_campaign"
    gdr_dir.mkdir()
    (gdr_dir / "campaign_protocol.json").write_text(json.dumps({"outcome_schema": 2}))
    (gdr_dir / "rows.jsonl").write_text(
        "\n".join([
            json.dumps({
                "policy": "semantic", "target_obligation": "ob1", "target_witnessed": True,
                "engine_new_obligations": 1, "terminated_by_collision": True,
            }),
            json.dumps({
                "policy": "semantic", "target_obligation": "ob1", "target_witnessed": True,
                "engine_new_obligations": 1, "outcome_schema_version": None, "terminated_by_collision": True,
            }),
            json.dumps({
                "policy": "semantic", "target_obligation": "ob1", "target_witnessed": True,
                "engine_new_obligations": 1, "outcome_schema_version": 1, "terminated_by_collision": True,
            }),
            json.dumps({
                "policy": "semantic", "target_obligation": "ob1", "target_witnessed": True,
                "engine_new_obligations": 1, "outcome_schema_version": 2, "terminated_by_collision": True,
            }),
        ]) + "\n"
    )
    gdr_rep = gdr.build_report(gdr_dir)
    assert gdr_rep["overall"]["denominators"]["n_outcome_unknown"] == 4
    assert gdr_rep["overall"]["denominators"]["eligible_for_outcome"] == 0
    assert gdr_rep["overall"]["failure_stage_counts"] == {"outcome_unknown": 4}
    # Non-UTF-8 campaign_protocol.json must not abort guidance reporting
    bad_utf8_dir = tmp_path / "non_utf8_proto"
    bad_utf8_dir.mkdir()
    (bad_utf8_dir / "campaign_protocol.json").write_bytes(b"\xff\xfe\x00invalid")
    (bad_utf8_dir / "rows.jsonl").write_text(json.dumps({"policy": "semantic", "target_obligation": "ob1"}) + "\n")
    assert gdr.build_report(bad_utf8_dir)["overall"]["n_attempts"] == 1
    # Direct build_guidance_diagnostics also treats backend_capabilities/feedback_version v2 rows as v2
    diag_direct_v2 = build_guidance_diagnostics([
        {
            "policy": "semantic", "target_obligation": "ob1", "target_witnessed": True,
            "engine_new_obligations": 1, "terminated_by_collision": True,
            "backend_capabilities": {"backend": "diagnostics", "native_v2_capture": False},
        },
        {
            "policy": "semantic", "target_obligation": "ob1", "target_witnessed": True,
            "engine_new_obligations": 1, "terminated_by_collision": True,
            "feedback_version": "v2_normalized",
        },
    ])
    assert diag_direct_v2["denominators"]["n_outcome_unknown"] == 2
    assert diag_direct_v2["denominators"]["eligible_for_outcome"] == 0
    # And policy_search checkpoint validation refuses resuming unmarked rows (including null/[]/schema-1 outcome_v2) under outcome_schema=2!
    import argparse as _ap
    _chk_dir = tmp_path / "unmarked_v2_resume"
    _chk_dir.mkdir()
    _chk_args = _ap.Namespace(
        policy="semantic", route_label="r1", search_space="campaign", mode="diagnostics",
        seed=0, agent_kind="scripted", pcla_agent=None, agent_repo_path=None, agent_config=None,
        max_ticks=100, control=False, paired_controls=False, frozen_base=False,
        engine_metrics=False, hazard_search=False, hazard_stall_limit=0, shared_suite=False,
        base_spec=None, oracle=None, coverage_profile=None,
        outcome_schema=2, feedback_version="legacy_v1", outcome_profile=None, replicate_id=None,
        output_dir=_chk_dir,
    )
    _ps_mod._validate_protocol(_chk_args, _chk_dir / "rows.jsonl", {})
    for bad_existing in (
        {"policy": "semantic", "terminated_by_collision": True},
        {"policy": "semantic", "outcome_v2": None},
        {"policy": "semantic", "outcome_v2": []},
        {"policy": "semantic", "outcome_v2": {"schema_version": 1}},
    ):
        (_chk_dir / "rows.jsonl").write_text(json.dumps(bad_existing) + "\n")
        with pytest.raises(ValueError, match="outcome_schema_version=2"):
            _ps_mod._validate_protocol(_chk_args, _chk_dir / "rows.jsonl", {})
    # Bare non-object JSON line (e.g. null after a valid v2 row) must also be rejected during validation
    valid_v2_line = json.dumps({"policy": "semantic", "outcome_schema_version": 2})
    (_chk_dir / "rows.jsonl").write_text(f"{valid_v2_line}\nnull\n")
    with pytest.raises(ValueError, match="non-object JSON"):
        _ps_mod._validate_protocol(_chk_args, _chk_dir / "rows.jsonl", {})
    assert list(_ps_mod._iter_rows(_chk_dir / "rows.jsonl")) == [{"policy": "semantic", "outcome_schema_version": 2}]
    # Truncated/malformed JSON row and non-UTF-8 bytes in schema-2 checkpoint must be rejected!
    (_chk_dir / "rows.jsonl").write_text(f"{valid_v2_line}\n{{\"policy\": \"semantic\"")
    with pytest.raises(ValueError, match="malformed/truncated JSON"):
        _ps_mod._validate_protocol(_chk_args, _chk_dir / "rows.jsonl", {})
    (_chk_dir / "rows.jsonl").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        _ps_mod._validate_protocol(_chk_args, _chk_dir / "rows.jsonl", {})
    # Valid schema-2 checkpoint missing a trailing newline is normalized and preserves all rows on append!
    (_chk_dir / "rows.jsonl").write_text(valid_v2_line)  # no trailing newline
    _ps_mod._validate_protocol(_chk_args, _chk_dir / "rows.jsonl", {})
    assert (_chk_dir / "rows.jsonl").read_bytes().endswith(b"\n")
    _ps_mod._append_jsonl_row(_chk_dir / "rows.jsonl", {"policy": "semantic", "outcome_schema_version": 2, "eval_index": 1})
    assert len(list(_ps_mod._iter_rows(_chk_dir / "rows.jsonl"))) == 2
    # Direct _append_jsonl_row also inserts a newline boundary if the file lacks a trailing newline
    no_nl_path = tmp_path / "no_nl_direct.jsonl"
    no_nl_path.write_text(valid_v2_line)
    _ps_mod._append_jsonl_row(no_nl_path, {"policy": "semantic", "outcome_schema_version": 2, "eval_index": 2})
    assert len(list(_ps_mod._iter_rows(no_nl_path))) == 2
    # U+2028 inside a valid JSON string must pass validation and replay as 1 row,
    # while U+2028 separating two objects on one LF-line must be rejected (NOT split into 2 rows)!
    u2028_inside = json.dumps({"policy": "semantic", "outcome_schema_version": 2, "note": "a\u2028b"}, ensure_ascii=False)
    (_chk_dir / "rows.jsonl").write_text(u2028_inside + "\n", encoding="utf-8")
    _ps_mod._validate_protocol(_chk_args, _chk_dir / "rows.jsonl", {})
    replayed_u2028 = list(_ps_mod._iter_rows(_chk_dir / "rows.jsonl"))
    assert len(replayed_u2028) == 1 and replayed_u2028[0]["note"] == "a\u2028b"
    (_chk_dir / "rows.jsonl").write_text(f"{valid_v2_line}\u2028{valid_v2_line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed/truncated JSON"):
        _ps_mod._validate_protocol(_chk_args, _chk_dir / "rows.jsonl", {})
    assert list(_ps_mod._iter_rows(_chk_dir / "rows.jsonl")) == []
    # Bare CR inside valid JSON whitespace must NOT be split by universal newlines, while bare CR
    # separating two objects on one LF-line must be rejected, and leading U+2028 must NOT be masked by strip()!
    cr_inside = '{"policy": "semantic",\r "outcome_schema_version": 2}\n'.encode("utf-8")
    (_chk_dir / "rows.jsonl").write_bytes(cr_inside)
    _ps_mod._validate_protocol(_chk_args, _chk_dir / "rows.jsonl", {})
    assert len(list(_ps_mod._iter_rows(_chk_dir / "rows.jsonl"))) == 1
    cr_between = f"{valid_v2_line}\r{valid_v2_line}\n".encode("utf-8")
    (_chk_dir / "rows.jsonl").write_bytes(cr_between)
    with pytest.raises(ValueError, match="malformed/truncated JSON"):
        _ps_mod._validate_protocol(_chk_args, _chk_dir / "rows.jsonl", {})
    assert list(_ps_mod._iter_rows(_chk_dir / "rows.jsonl")) == []
    (_chk_dir / "rows.jsonl").write_bytes(f"\u2028{valid_v2_line}\n".encode("utf-8"))
    with pytest.raises(ValueError, match="malformed/truncated JSON"):
        _ps_mod._validate_protocol(_chk_args, _chk_dir / "rows.jsonl", {})
    assert list(_ps_mod._iter_rows(_chk_dir / "rows.jsonl")) == []
    assert gdr.load_decisions(_chk_dir / "rows.jsonl") == []
    recs_u2028, ctrs_u2028 = load_records(_chk_dir, schema_version=2)
    assert recs_u2028 == [] and ctrs_u2028["n_malformed_rows"] == 1
    # Overflowing float literal (1e999 -> inf) must be rejected by schema-2 checkpoint validation!
    (_chk_dir / "rows.jsonl").write_bytes(b'{"policy": "semantic", "outcome_schema_version": 2, "fitness": 1e999}\n')
    with pytest.raises(ValueError, match="malformed/truncated JSON"):
        _ps_mod._validate_protocol(_chk_args, _chk_dir / "rows.jsonl", {})
    _b_path = REPO_ROOT / "research" / "experiments" / "EXP-020-policy-comparison" / "proof-of-concept" / "baselines.py"
    _b_spec = _ilu.spec_from_file_location("_test_baselines_r16", _b_path)
    _b_mod = _ilu.module_from_spec(_b_spec)
    _b_spec.loader.exec_module(_b_mod)
    for bad_v2 in (
        {"schema_version": 2, "execution_status": "completed", "categories": ["bad"]},
        {"schema_version": 2, "execution_status": "completed", "categories": {"safety": {"collision": 42}}, "events": 99},
        {"schema_version": 1, "execution_status": "completed", "categories": {"safety": {"collision": {"status": "present"}}}},
        {"schema_version": 2, "execution_status": "not_executed", "categories": {"safety": {"collision": {"status": "present"}}}},
        {"schema_version": 2, "execution_status": "completed", "events": [{"measurements": {"min_gap_m": 10**309}}]},
    ):
        assert _b_mod.gap_fitness({"outcome_v2": bad_v2}, feedback_version="v2_normalized") is None

    # 3. Malformed {} capture manifest raises EvidenceReadError in load_capture_manifest and does NOT abort audit_archive
    empty_manifest = tmp_path / "empty.manifest.json"
    empty_manifest.write_text("{}")
    with pytest.raises(EvidenceReadError):
        load_capture_manifest(empty_manifest)

    arch = tmp_path / "archive"
    pol_dir = arch / "sem"
    pol_dir.mkdir(parents=True)
    prof = load_outcome_profile()
    base_cats = {
        grp: {m: {"status": "absent"} for m in names}
        for grp, names in ro.CATEGORIES.items()
    }
    emb_a = {"schema_version": 2, "run_id": "ra", "execution_status": "completed", "profile_digest": "digest-prof-A", "profile_status": "validated", "categories": base_cats}
    emb_b = {"schema_version": 2, "run_id": "rb", "execution_status": "completed", "profile_digest": "digest-prof-B", "profile_status": "draft", "categories": base_cats}
    # Partial embedded v2 outcome (only safety.collision present) + unhashable dict/list status must NOT crash summarize_records/main!
    emb_partial = {
        "schema_version": 2, "run_id": "rp", "execution_status": "completed", "profile_digest": "digest-prof-A",
        "categories": {"safety": {"collision": {"status": "present"}, "near_miss": {"status": {"bad": 1}}, "unsafe_proximity": {"status": ["bad"]}}},
    }
    (pol_dir / "rows.jsonl").write_text(
        "\n".join([
            json.dumps({"policy": "sem", "capture_manifest_path": str(empty_manifest)}),
            json.dumps({"policy": "sem", "outcome_v2": emb_a}),
            json.dumps({"policy": "sem", "outcome_v2": emb_b}),
            json.dumps({"policy": "sem", "outcome_v2": emb_partial}),
        ]) + "\n"
    )
    out_dir = tmp_path / "audit_out"
    assert ro.main(["--search-root", str(arch), "--output-dir", str(out_dir)]) == 0
    ledger = json.loads((out_dir / "ledger.json").read_text())
    report = json.loads((out_dir / "report.json").read_text())
    assert ledger[0]["status"] == "unreadable_run"
    assert ledger[1]["status"] == "embedded_v2"
    assert ledger[2]["status"] == "embedded_v2"
    assert ledger[3]["status"] == "embedded_v2"
    # Mixed profiles must be reported honestly rather than stamped with the default draft profile digest!
    assert report["mixed_profiles"] is True
    assert report["profile_mode"] == "mixed_embedded_profiles"
    assert report["profile_digest"] is None
    assert report["profile_status"] == "mixed"
    assert set(report["observed_profile_digests"]) == {"digest-prof-A", "digest-prof-B"}
    assert report["fallback_profile_digest"] == profile_digest(prof)

    # 4. Uniform validated embedded archive (nested provenance.profile_status) + one unreadable row must stay uniform & validated!
    #    AND explicit-profile audit must match embedded outcomes whose digest lives under provenance.profile_digest!
    from dataclasses import replace as _replace
    from research.harness.safety_outcomes import OutcomeEvaluator
    from research.tests.outcome_helpers import frame, manifest
    validated_prof = _replace(prof, status="validated", validation_refs=("live-val-1",))
    ev = OutcomeEvaluator(validated_prof)
    for i in range(5):
        ev.observe(frame(i))
    genuine_val_outcome = ev.finish(manifest(0.4)).to_dict()
    nested_only_outcome = dict(genuine_val_outcome)
    nested_only_outcome.pop("profile_digest", None)
    nested_only_outcome["provenance"] = {**dict(genuine_val_outcome.get("provenance") or {}), "profile_digest": profile_digest(validated_prof)}

    uniform_arch = tmp_path / "uniform_val_archive"
    u_pol = uniform_arch / "sem"
    u_pol.mkdir(parents=True)
    (u_pol / "rows.jsonl").write_text(
        "\n".join([
            json.dumps({"policy": "sem", "outcome_v2": genuine_val_outcome}),
            json.dumps({"policy": "sem", "outcome_v2": nested_only_outcome}),
            json.dumps({"policy": "sem", "capture_manifest_path": str(empty_manifest)}),
        ]) + "\n"
    )
    u_out = tmp_path / "uniform_audit_out"
    assert ro.main(["--search-root", str(uniform_arch), "--output-dir", str(u_out)]) == 0
    u_report = json.loads((u_out / "report.json").read_text())
    assert u_report["mixed_profiles"] is False
    assert u_report["profile_mode"] == "preserved_embedded_uniform"
    assert u_report["profile_digest"] == profile_digest(validated_prof)
    assert u_report["profile_status"] == "validated"
    assert u_report["observed_profile_digests"] == [profile_digest(validated_prof)]
    # Explicit profile audit must recognize nested provenance.profile_digest too!
    explicit_audited = ro.audit_archive(uniform_arch, profile=validated_prof)
    assert explicit_audited[0]["status"] == "embedded_v2"
    assert explicit_audited[1]["status"] == "embedded_v2"
