"""Tests for safety-utility analysis (time-to-first and obligation linkage)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.harness.safety_utility import (  # noqa: E402
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
