"""Tests for the EXP-020 gap-driven policy selection path.

These tests drive ``proof-of-concept/policy_search.py`` with synthetic rows and
synthetic engine outputs. They cover:

- gap-closure fitness and tie-breaking (new obligations first, then obligations
  witnessed in the run; collisions are not rewarded);
- suite accumulation and first-uncover indexing in ``EngineCoverageState``;
- missing/failed stream handling (nulls, no exception);
- engine-off equivalence: engine-off runs are deterministic and carry no
  engine columns, and baseline policies select the same elites with the engine
  metrics path enabled;
- a dry end-to-end ``main`` run against a stubbed evaluator with real engine
  scoring of synthetic semantic streams.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = REPO_ROOT / "research" / "experiments" / "EXP-020-policy-comparison" / "proof-of-concept"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_policy_search():
    spec = importlib.util.spec_from_file_location("policy_search", POLICY_DIR / "policy_search.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["policy_search"] = module
    spec.loader.exec_module(module)
    return module


policy_search = _load_policy_search()


# ---------------------------------------------------------------------------
# Synthetic engine outputs
# ---------------------------------------------------------------------------

_ENGINE_OBLIGATIONS = {
    "V": (("node(pedestrian)", True), ("node(vehicle)", True), ("node(animal)", False)),
    "A": (("stationary(ego)", True), ("moving(ego)", True)),
    "E": (("in_front_of(pedestrian,ego)", True),),
    "H": (("hazard(other: pedestrian)", True),),
}


def _engine_report(covered: set[str]) -> dict:
    dimensions = {}
    for axis, obligations in _ENGINE_OBLIGATIONS.items():
        rows = [
            {"signature": signature, "mapped": mapped, "covered": mapped and signature in covered}
            for signature, mapped in obligations
        ]
        dimensions[axis] = {
            "mapped_obligations": sum(1 for _, mapped in obligations if mapped),
            "obligations": rows,
        }
    return {"dimensions": dimensions}


def _write_stream(semantic_dir: Path, payload: str = "") -> Path:
    semantic_dir.mkdir(parents=True, exist_ok=True)
    path = semantic_dir / "run-pcla-20260101T000000Z-semantic-stream.jsonl"
    path.write_text(payload, encoding="utf-8")
    return path


@pytest.fixture()
def space():
    return policy_search.make_legacy_space(seed=7)


# ---------------------------------------------------------------------------
# EngineCoverageState: suite accumulation, first-uncover, failure handling
# ---------------------------------------------------------------------------


def test_engine_state_accumulates_suite_and_first_uncover(tmp_path, monkeypatch):
    state = policy_search.EngineCoverageState(oracle=None)
    stream_dir = tmp_path / "semantic"
    _write_stream(stream_dir)
    reports = iter(
        [
            _engine_report({"node(pedestrian)", "stationary(ego)"}),
            _engine_report({"node(pedestrian)", "moving(ego)"}),
        ]
    )
    monkeypatch.setattr(policy_search, "load_semantic_trace", lambda path: [])
    monkeypatch.setattr(policy_search, "compute_cov", lambda *args, **kwargs: next(reports))

    first = state.observe_stream(stream_dir, 0)
    assert first["engine_new_obligations"] == 2
    assert first["engine_run_covered_count"] == 2
    assert first["engine_uncovered_count"] == 4
    assert first["engine_first_uncover"] == {"node(pedestrian)": 0, "stationary(ego)": 0}
    assert first["engine_cov_v"] == pytest.approx(0.5)
    assert first["engine_cov_a"] == pytest.approx(0.5)
    assert first["engine_cov_e"] == 0.0
    assert first["engine_cov_h"] == 0.0
    assert first["engine_error"] is None

    second = state.observe_stream(stream_dir, 1)
    assert second["engine_new_obligations"] == 1
    assert second["engine_run_covered_count"] == 2
    assert second["engine_uncovered_count"] == 3
    assert second["engine_first_uncover"] == {
        "node(pedestrian)": 0,
        "stationary(ego)": 0,
        "moving(ego)": 1,
    }
    assert second["engine_cov_a"] == pytest.approx(1.0)
    assert second["engine_cov_v"] == pytest.approx(0.5)

    assert state.suite_covered == {"node(pedestrian)", "stationary(ego)", "moving(ego)"}


def test_engine_state_repeated_obligations_do_not_recount(tmp_path, monkeypatch):
    state = policy_search.EngineCoverageState(oracle=None)
    stream_dir = tmp_path / "semantic"
    _write_stream(stream_dir)
    reports = iter(
        [
            _engine_report({"node(pedestrian)"}),
            _engine_report({"node(pedestrian)"}),
        ]
    )
    monkeypatch.setattr(policy_search, "load_semantic_trace", lambda path: [])
    monkeypatch.setattr(policy_search, "compute_cov", lambda *args, **kwargs: next(reports))

    assert state.observe_stream(stream_dir, 0)["engine_new_obligations"] == 1
    second = state.observe_stream(stream_dir, 1)
    assert second["engine_new_obligations"] == 0
    assert second["engine_first_uncover"] == {"node(pedestrian)": 0}
    assert state.mapped_total == 6
    assert second["engine_uncovered_count"] == 5


def test_engine_state_forwards_stream_ego_route_to_engine(tmp_path, monkeypatch):
    state = policy_search.EngineCoverageState(oracle=None)
    stream_dir = tmp_path / "semantic"
    _write_stream(stream_dir)
    route = [{"x": 1.0, "y": 2.0}, {"x": 1.0, "y": 3.0}]
    seen: dict = {}
    monkeypatch.setattr(policy_search, "load_semantic_trace", lambda path: [])
    monkeypatch.setattr(policy_search, "load_ego_route", lambda path: route)

    def fake_compute(oracle, ticks, **kwargs):
        seen.update(kwargs)
        return _engine_report(set())

    monkeypatch.setattr(policy_search, "compute_cov", fake_compute)

    state.observe_stream(stream_dir, 0)

    assert seen["ego_route"] == route


def test_engine_state_missing_stream_logs_nulls(tmp_path):
    state = policy_search.EngineCoverageState(oracle=None)
    metrics = state.observe_stream(tmp_path / "absent", 0)
    assert metrics["engine_new_obligations"] is None
    assert metrics["engine_cov_v"] is None
    assert metrics["engine_first_uncover"] is None
    assert "no semantic stream" in metrics["engine_error"]


def test_engine_state_failed_stream_logs_nulls_and_keeps_suite(tmp_path, monkeypatch):
    state = policy_search.EngineCoverageState(oracle=None)
    stream_dir = tmp_path / "semantic"
    _write_stream(stream_dir)
    monkeypatch.setattr(policy_search, "load_semantic_trace", lambda path: [])
    monkeypatch.setattr(
        policy_search,
        "compute_cov",
        lambda *args, **kwargs: _engine_report({"node(pedestrian)"}),
    )
    assert state.observe_stream(stream_dir, 0)["engine_new_obligations"] == 1

    def boom(path):
        raise ValueError("truncated JSONL row")

    monkeypatch.setattr(policy_search, "load_semantic_trace", boom)
    metrics = state.observe_stream(stream_dir, 1)
    assert metrics["engine_new_obligations"] is None
    assert metrics["engine_cov_a"] is None
    assert "coverage engine failed" in metrics["engine_error"]
    assert state.suite_covered == {"node(pedestrian)"}


def test_engine_state_restore_from_row_rebuilds_suite():
    state = policy_search.EngineCoverageState(oracle=None)
    state.restore_from_row({"engine_first_uncover": {"node(pedestrian)": 0, "moving(ego)": 2}})
    assert state.suite_covered == {"node(pedestrian)", "moving(ego)"}
    assert state.first_uncover == {"node(pedestrian)": 0, "moving(ego)": 2}
    state.restore_from_row({})
    state.restore_from_row({"engine_first_uncover": None})
    assert state.first_uncover == {"node(pedestrian)": 0, "moving(ego)": 2}


# ---------------------------------------------------------------------------
# Gap-driven fitness and elite selection
# ---------------------------------------------------------------------------


def _engine_row(**overrides) -> dict:
    row = {
        "semantic_fulfilled_obligations": ["a", "b"],
        "terminated_by_collision": False,
        "engine_new_obligations": 1,
        "engine_run_covered_count": 2,
    }
    row.update(overrides)
    return row


def test_engine_fitness_is_gap_closure_not_fulfilled_count():
    saturated = _engine_row(
        semantic_fulfilled_obligations=["a", "b", "c", "d"],
        engine_new_obligations=0,
        engine_run_covered_count=4,
    )
    gap_closing = _engine_row(
        semantic_fulfilled_obligations=["a"],
        engine_new_obligations=2,
        engine_run_covered_count=1,
    )
    assert policy_search._semantic_fitness(saturated, "semantic") == 4.0
    assert policy_search._semantic_fitness(gap_closing, "semantic") == 1.0
    assert policy_search._semantic_fitness(saturated, "semantic", engine_active=True) == 0.0
    assert policy_search._semantic_fitness(gap_closing, "semantic", engine_active=True) == 2.0


def test_engine_fitness_does_not_reward_collisions():
    clean = _engine_row(engine_new_obligations=2, engine_run_covered_count=2)
    collided = _engine_row(
        engine_new_obligations=2,
        engine_run_covered_count=2,
        terminated_by_collision=True,
    )
    assert policy_search._semantic_fitness(clean, "semantic", engine_active=True) == (
        policy_search._semantic_fitness(collided, "semantic", engine_active=True)
    )


def test_semantic_engine_off_keeps_archived_formula():
    row = _engine_row(
        semantic_fulfilled_obligations=["a", "b"],
        terminated_by_collision=True,
        engine_new_obligations=0,
        engine_run_covered_count=10,
    )
    assert policy_search._semantic_fitness(row, "semantic") == 3.0
    assert policy_search._semantic_tiebreak(row, "semantic") is None


def test_gap_driven_semantic_elite_switches_to_gap_closer(space):
    state = policy_search.PolicyState("semantic", space)
    saturated_values = {"trigger_radius_m": 30.0}
    gap_values = {"trigger_radius_m": 8.0}
    saturated = _engine_row(
        semantic_fulfilled_obligations=["a", "b", "c", "d"],
        engine_new_obligations=0,
        engine_run_covered_count=4,
    )
    gap = _engine_row(
        semantic_fulfilled_obligations=["a"],
        engine_new_obligations=2,
        engine_run_covered_count=1,
    )
    for values, row in ((saturated_values, saturated), (gap_values, gap)):
        state.observe(
            values,
            policy_search._semantic_fitness(row, "semantic", engine_active=True),
            policy_search._semantic_tiebreak(row, "semantic", engine_active=True),
        )
    assert state.best_values == gap_values
    assert state.best_fitness == 2.0


def test_gap_driven_semantic_tiebreak_prefers_more_run_witnesses(space):
    state = policy_search.PolicyState("semantic", space)
    few_values = {"trigger_radius_m": 12.0}
    many_values = {"trigger_radius_m": 20.0}
    few = _engine_row(engine_new_obligations=2, engine_run_covered_count=2)
    many = _engine_row(engine_new_obligations=2, engine_run_covered_count=5)
    for values, row in ((few_values, few), (many_values, many)):
        state.observe(
            values,
            policy_search._semantic_fitness(row, "semantic", engine_active=True),
            policy_search._semantic_tiebreak(row, "semantic", engine_active=True),
        )
    assert state.best_values == many_values

    reverse = policy_search.PolicyState("semantic", space)
    for values, row in ((many_values, many), (few_values, few)):
        reverse.observe(
            values,
            policy_search._semantic_fitness(row, "semantic", engine_active=True),
            policy_search._semantic_tiebreak(row, "semantic", engine_active=True),
        )
    assert reverse.best_values == many_values


def test_baseline_selection_unchanged_when_engine_metrics_enabled(space):
    samples = [
        ({"trigger_radius_m": 30.0}, 0.2),
        ({"trigger_radius_m": 8.0}, 0.9),
        ({"trigger_radius_m": 16.0}, 0.5),
    ]
    for policy in ("lsa", "kmnc"):
        engine_state = policy_search.PolicyState(policy, space)
        plain_state = policy_search.PolicyState(policy, space)
        for values, coverage in samples:
            row = {
                "coverage_lsa_max": coverage,
                "coverage_kmnc": coverage,
                "semantic_fulfilled_obligations": ["x"],
                "terminated_by_collision": True,
                "engine_new_obligations": 3,
                "engine_run_covered_count": 9,
            }
            assert policy_search._semantic_tiebreak(row, policy, engine_active=True) is None
            engine_state.observe(
                values,
                policy_search._semantic_fitness(row, policy, engine_active=True),
                policy_search._semantic_tiebreak(row, policy, engine_active=True),
            )
            plain_state.observe(
                values, policy_search._semantic_fitness(row, policy, engine_active=False)
            )
        assert engine_state.best_values == plain_state.best_values


# ---------------------------------------------------------------------------
# Dry end-to-end runs against a stubbed evaluator
# ---------------------------------------------------------------------------

DRY_ORACLE = {
    "metadata": {"split": "synthetic"},
    "counts": {"total": {"obligation_count": 2, "defined_predicates": 2}},
    "dimensions": {
        "node": {
            "vocabulary": ["pedestrian"],
            "obligations": [
                {
                    "dimension": "node",
                    "predicate": "node:pedestrian",
                    "signature": "node(pedestrian)",
                    "grounding": "direct",
                    "node_types": ["pedestrian"],
                }
            ],
        },
        "attribute": {
            "vocabulary": ["stationary"],
            "obligations": [
                {
                    "dimension": "attribute",
                    "predicate": "stationary",
                    "signature": "stationary(ego)",
                    "grounding": "derived",
                    "node_types": ["ego"],
                }
            ],
        },
        "relation": {"vocabulary": [], "obligations": []},
        "hazard_class": {"vocabulary": [], "obligations": []},
    },
}


def _stationary_tick(index: int, *, with_pedestrian: bool) -> dict:
    actors = []
    if with_pedestrian:
        actors.append(
            {
                "id": 7,
                "type_id": "walker.pedestrian.0001",
                "distance_to_ego_m": 8.0,
                "location": {"x": 0.0, "y": 8.0, "z": 0.0},
                "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                "velocity_mps": 1.0,
                "same_road_as_ego": True,
                "same_lane_as_ego": False,
                "is_in_front_of_ego": True,
            }
        )
    return {
        "tick": index,
        "scenario_id": "dry",
        "town": "Town01",
        "ego": {
            "id": 1,
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "velocity_mps": 0.0,
            "waypoint": {"road_id": 10, "lane_id": -1, "s": 0.0, "is_junction": False},
        },
        "telemetry": {
            "speed_mps": 0.0,
            "control": {"throttle": 0.0, "steer": 0.0, "brake": 0.0, "gear": 1},
        },
        "nearby_actors": actors,
    }


def _write_base_spec(tmp_path: Path) -> Path:
    payload = {
        "scenario_id": "dry",
        "description": "dry run",
        "controller_params": {"trigger_radius_m": 20.0},
    }
    path = tmp_path / "base_spec.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_oracle(tmp_path: Path) -> Path:
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(DRY_ORACLE), encoding="utf-8")
    return path


def _install_stub_evaluator(monkeypatch, stream_factory):
    def stub(args, spec_payload, candidate_id, work_dir, port):
        if stream_factory is not None:
            semantic_dir = work_dir / "semantic"
            semantic_dir.mkdir(parents=True, exist_ok=True)
            frames = stream_factory(spec_payload)
            stream_path = (
                semantic_dir / f"{candidate_id}-pcla-20260101T000000Z-semantic-stream.jsonl"
            )
            stream_path.write_text(
                "".join(json.dumps(frame) + "\n" for frame in frames), encoding="utf-8"
            )
        radius = float(spec_payload.get("controller_params", {}).get("trigger_radius_m", 20.0))
        return {
            "ticks_executed": 50,
            "reached_goal": False,
            "collision_count": 0,
            "terminated_by_collision": False,
            "coverage_status": "ok",
            "coverage_kmnc": radius / 40.0,
            "coverage_lsa_max": radius / 40.0,
            "coverage_lsa_mean": radius / 40.0,
            "semantic_fulfilled_obligations": ["x"],
            "semantic_missing_obligations": [],
            "semantic_covered_predicates": [],
            "semantic_covered_signatures": [],
            "run_error": None,
        }

    monkeypatch.setattr(policy_search, "_run_evaluation_server", stub)


def _run_main(
    monkeypatch,
    output_dir: Path,
    base_spec: Path,
    *,
    policy: str,
    engine: bool,
    oracle: Path | None = None,
    evals: int = 3,
    seed: int = 11,
) -> None:
    argv = [
        "policy_search.py",
        "--policy",
        policy,
        "--base-spec",
        str(base_spec),
        "--route-label",
        "dry-route",
        "--output-dir",
        str(output_dir),
        "--evals",
        str(evals),
        "--seed",
        str(seed),
    ]
    if engine:
        argv += ["--engine-metrics", "--oracle", str(oracle)]
    monkeypatch.setattr(sys, "argv", argv)
    policy_search.main()


def _read_rows(output_dir: Path) -> list[dict]:
    text = (output_dir / "rows.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_engine_off_rows_have_no_engine_columns_and_are_deterministic(tmp_path, monkeypatch):
    base_spec = _write_base_spec(tmp_path)
    _install_stub_evaluator(monkeypatch, stream_factory=None)

    output_a = tmp_path / "a"
    output_b = tmp_path / "b"
    _run_main(monkeypatch, output_a, base_spec, policy="lsa", engine=False)
    _run_main(monkeypatch, output_b, base_spec, policy="lsa", engine=False)

    rows_a = _read_rows(output_a)
    rows_b = _read_rows(output_b)
    assert rows_a == rows_b
    assert len(rows_a) == 3
    assert not any(key.startswith("engine_") for row in rows_a for key in row)


def test_engine_metrics_requires_oracle(tmp_path, monkeypatch):
    base_spec = _write_base_spec(tmp_path)
    argv = [
        "policy_search.py",
        "--policy",
        "lsa",
        "--base-spec",
        str(base_spec),
        "--route-label",
        "dry-route",
        "--output-dir",
        str(tmp_path / "out"),
        "--engine-metrics",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        policy_search.main()


def test_baseline_rows_match_engine_off_apart_from_engine_columns(tmp_path, monkeypatch):
    base_spec = _write_base_spec(tmp_path)
    oracle_path = _write_oracle(tmp_path)
    _install_stub_evaluator(
        monkeypatch,
        stream_factory=lambda payload: [_stationary_tick(0, with_pedestrian=True)],
    )

    output_off = tmp_path / "off"
    output_on = tmp_path / "on"
    _run_main(monkeypatch, output_off, base_spec, policy="lsa", engine=False)
    _run_main(
        monkeypatch, output_on, base_spec, policy="lsa", engine=True, oracle=oracle_path
    )

    rows_off = _read_rows(output_off)
    rows_on = _read_rows(output_on)
    assert len(rows_on) == len(rows_off)
    stripped = [
        {key: value for key, value in row.items() if not key.startswith("engine_")}
        for row in rows_on
    ]
    assert stripped == rows_off

    assert rows_on[0]["engine_new_obligations"] == 2
    assert rows_on[0]["engine_run_covered_count"] == 2
    assert rows_on[0]["engine_first_uncover"] == {
        "node(pedestrian)": 0,
        "stationary(ego)": 0,
    }
    assert rows_on[0]["engine_cov_v"] == pytest.approx(1.0)
    assert rows_on[0]["engine_cov_a"] == pytest.approx(1.0)
    assert rows_on[0]["engine_cov_e"] is None
    assert rows_on[0]["engine_cov_h"] is None
    assert rows_on[0]["engine_error"] is None
    assert rows_on[1]["engine_new_obligations"] == 0
    assert rows_on[1]["engine_uncovered_count"] == 0


def test_semantic_engine_run_accumulates_suite_gap_closure(tmp_path, monkeypatch):
    base_spec = _write_base_spec(tmp_path)
    oracle_path = _write_oracle(tmp_path)
    _install_stub_evaluator(
        monkeypatch,
        stream_factory=lambda payload: [_stationary_tick(0, with_pedestrian=True)],
    )

    output = tmp_path / "semantic"
    _run_main(
        monkeypatch,
        output,
        base_spec,
        policy="semantic",
        engine=True,
        oracle=oracle_path,
        evals=4,
    )
    rows = _read_rows(output)
    assert len(rows) == 4
    assert rows[0]["engine_new_obligations"] == 2
    assert all(row["engine_new_obligations"] == 0 for row in rows[1:])
    assert sum(row["engine_new_obligations"] for row in rows) == 2
    assert rows[-1]["engine_first_uncover"] == {
        "node(pedestrian)": 0,
        "stationary(ego)": 0,
    }
    assert rows[-1]["engine_cov_v"] == pytest.approx(1.0)
