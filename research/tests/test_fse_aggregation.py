"""Tests for the FSE coverage-engine aggregation and figures.

Synthetic rows cover the three arms of the task:

- engine rows with well-defined AUC and discovery schedules, on 6 routes so the
  paired Wilcoxon wiring is exercised at n >= 6;
- legacy (pilot) rows without ``engine_*`` keys, plus engine arms with only
  null metrics, so the skip accounting is pinned;
- flat ``<root>/<policy>/rows.jsonl`` and nested ``<root>/<route>/<policy>/``
  layouts.

File outputs (JSON/MD/CSV and the three figure pairs) are written into
``tmp_path`` and asserted to exist and parse.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AGGREGATE_PATH = (
    REPO_ROOT
    / "research"
    / "experiments"
    / "EXP-020-policy-comparison"
    / "proof-of-concept"
    / "aggregate_fse_search.py"
)
FIGURES_PATH = REPO_ROOT / "research" / "scripts" / "make_fse_figures.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


aggregate = _load_module("aggregate_fse_search", AGGREGATE_PATH)

ROUTES = [f"route{index}" for index in range(6)]
POLICIES = ("random", "lsa", "kmnc", "semantic")

# Per-policy coverage schedules: values at eval indices 0..4 (node dimension).
COV_V = {
    "semantic": [0.2, 0.5, 0.7, 0.9, 1.0],
    "random": [0.1, 0.15, 0.2, 0.25, 0.3],
    "lsa": [0.15, 0.3, 0.45, 0.6, 0.7],
    "kmnc": [0.1, 0.2, 0.3, 0.4, 0.5],
}
# New obligations closed at each eval index (front-loaded for semantic).
NEW_PER_EVAL = {
    "semantic": [4, 3, 2, 1, 0],
    "random": [1, 1, 1, 1, 1],
    "lsa": [2, 1, 2, 1, 1],
    "kmnc": [1, 2, 1, 2, 1],
}
TOTAL_OBLIGATIONS = 20


def _rows_for(policy: str, route: str, route_index: int) -> list[dict]:
    offset = 0.01 * route_index
    closed: dict[str, int] = {}
    rows = []
    counter = 0
    for eval_index, schedule in enumerate(NEW_PER_EVAL[policy]):
        for _ in range(schedule):
            closed[f"ob{counter:02d}"] = eval_index
            counter += 1
        base = min(1.0, COV_V[policy][eval_index] + offset)
        rows.append(
            {
                "policy": policy,
                "route": route,
                "eval_index": eval_index,
                "ticks_executed": 100,
                "engine_cov_v": base,
                "engine_cov_a": round(base * 0.5, 6),
                "engine_cov_e": round(base * 0.25, 6),
                "engine_cov_h": 0.0,
                "engine_new_obligations": schedule,
                "engine_run_covered_count": counter,
                "engine_uncovered_count": TOTAL_OBLIGATIONS - counter,
                "engine_first_uncover": dict(closed),
                "engine_error": None,
            }
        )
    return rows


def _write_rows(root: Path, route: str, policy: str, rows: list[dict]) -> None:
    path = root / route / policy / "rows.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _build_root(tmp_path: Path) -> Path:
    root = tmp_path / "policy_search"
    for route_index, route in enumerate(ROUTES):
        for policy in POLICIES:
            _write_rows(root, route, policy, _rows_for(policy, route, route_index))
    # Engine arm with no valid rows: every engine metric is null.
    _write_rows(
        root,
        "route_dead",
        "kmnc",
        [
            {
                "policy": "kmnc",
                "route": "route_dead",
                "eval_index": 0,
                "ticks_executed": 10,
                "engine_cov_v": None,
                "engine_cov_a": None,
                "engine_cov_e": None,
                "engine_cov_h": None,
                "engine_new_obligations": None,
                "engine_uncovered_count": None,
                "engine_first_uncover": None,
                "engine_error": "no semantic stream",
            }
        ],
    )
    # Null engine row inside a valid arm: counted, excluded.
    _write_rows(
        root,
        "route0",
        "random",
        _rows_for("random", "route0", 0)
        + [
            {
                "policy": "random",
                "route": "route0",
                "eval_index": 9,
                "ticks_executed": None,
                "engine_cov_v": None,
                "engine_cov_a": None,
                "engine_cov_e": None,
                "engine_cov_h": None,
                "engine_new_obligations": None,
                "engine_first_uncover": None,
                "engine_error": "failed stream",
            }
        ],
    )
    # Legacy pilot arm: no engine keys at all.
    _write_rows(
        root,
        "route0",
        "control",
        [
            {
                "policy": "control",
                "route": "route0",
                "eval_index": index,
                "ticks_executed": 50,
                "collision_count": 0,
            }
            for index in range(3)
        ],
    )
    return root


def test_trajectory_auc_known_values() -> None:
    assert aggregate.trajectory_auc([0, 1, 2], [0.0, 0.5, 1.0]) == pytest.approx(0.5)
    assert aggregate.trajectory_auc([0], [0.75]) == pytest.approx(0.75)
    # Constant curve over a span: AUC equals the value.
    assert aggregate.trajectory_auc([0, 2], [0.4, 0.4]) == pytest.approx(0.4)
    # Duplicate eval indices keep the last value.
    assert aggregate.trajectory_auc([0, 1, 1], [0.0, 0.5, 1.0]) == pytest.approx(0.5)
    assert aggregate.trajectory_auc([], []) is None


def test_discovery_metrics_known_values() -> None:
    times = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert aggregate.close_index(times, 50.0) == pytest.approx(2.0)
    assert aggregate.close_index(times, 25.0) == pytest.approx(1.0)
    assert aggregate.close_index(times, 75.0) == pytest.approx(3.0)
    assert aggregate.close_index([], 50.0) is None
    xs, ys = aggregate.discovery_curve(times, 4)
    assert xs == [0, 1, 2, 3, 4]
    assert ys == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])


def test_load_skips_legacy_and_null_rows(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    loaded = aggregate.load_engine_rows(root)
    state = {
        "skipped": loaded["skipped_arms"],
        "empty": loaded["empty_engine_arms"],
        "legacy": loaded["legacy_rows"],
        "failed": loaded["failed_engine_rows"],
        "layouts": loaded["layouts"],
    }
    assert state["legacy"] == 3
    assert [arm["policy"] for arm in state["skipped"]] == ["control"]
    assert state["failed"] == 2
    assert [arm["route"] for arm in state["empty"]] == ["route_dead"]
    assert state["layouts"] == ["nested"]
    assert ("route0", "random") in loaded["arms"]
    assert loaded["arms"][("route0", "random")][-1].valid is False


def test_compute_statistics_auc_and_discovery(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    result = aggregate.compute_fse_statistics(root)
    semantic_route0 = next(
        record
        for record in result["arms"]
        if record["route"] == "route0" and record["policy"] == "semantic"
    )
    assert semantic_route0["auc"]["auc_v"] == pytest.approx(0.675)
    assert semantic_route0["final_cov"]["V"] == pytest.approx(1.0)
    assert semantic_route0["discovery"]["n_closed"] == 10
    assert semantic_route0["discovery"]["median_first_uncover"] == pytest.approx(1.0)
    assert semantic_route0["discovery"]["q75_close_index"] == pytest.approx(1.75)
    assert semantic_route0["mapped_total"] == pytest.approx(20)
    random_route0 = next(
        record
        for record in result["arms"]
        if record["route"] == "route0" and record["policy"] == "random"
    )
    assert random_route0["n_skipped_null_rows"] == 1

    semantic_summary = result["cross_route"]["semantic"]
    assert semantic_summary["n_routes"] == 6
    assert semantic_summary["auc_mean"]["auc_v"] == pytest.approx(0.696875)
    assert semantic_summary["auc_ci"]["auc_v"] is not None
    pooled = semantic_summary["discovery"]["pooled"]
    assert pooled["n_obligations"] == 60
    assert pooled["median_first_uncover"] == pytest.approx(1.0)


def test_pairwise_wiring_wilcoxon_holm_and_effect(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    result = aggregate.compute_fse_statistics(root, min_test_routes=6)
    records = result["pairwise"]["auc_v"]
    assert [record["policy_b"] for record in records] == ["random", "lsa", "kmnc"]
    for record in records:
        assert record["n_pairs"] == 6
        assert record["testable"] is True
        assert record["p_wilcoxon"] is not None
        assert record["p_wilcoxon_holm"] is not None
        assert record["p_mannwhitney"] is not None
        assert record["mean_difference"] is not None
        low, high = record["mean_difference_ci"]
        assert low <= record["mean_difference"] <= high
        assert record["a12_paired"] == pytest.approx(1.0)
        assert record["effect"] == "large"
    discovery_records = result["pairwise"]["median_first_uncover"]
    semantic_vs_random = next(
        record for record in discovery_records if record["policy_b"] == "random"
    )
    assert semantic_vs_random["testable"] is True
    assert semantic_vs_random["mean_difference"] < 0.0
    assert semantic_vs_random["p_wilcoxon"] < 0.05
    # Non-testable when only fewer routes pair up.
    small = aggregate.compute_fse_statistics(
        root, comparison_policies=("random",), min_test_routes=100
    )
    assert small["pairwise"]["auc_v"][0]["testable"] is False
    assert small["pairwise"]["auc_v"][0]["p_wilcoxon"] is None


def test_legacy_only_root_is_explicitly_skipped(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    _write_rows(
        root,
        "routeA",
        "semantic",
        [{"policy": "semantic", "route": "routeA", "eval_index": 0, "ticks_executed": 10}],
    )
    result = aggregate.compute_fse_statistics(root)
    state = result["data_state"]
    assert result["arms"] == []
    assert state["n_rows_total"] == 1
    assert state["legacy_rows"] == 1
    assert state["n_arms_skipped_no_engine"] == 1
    assert state["skipped_arms"][0]["policy"] == "semantic"


def test_flat_layout_uses_row_route(tmp_path: Path) -> None:
    root = tmp_path / "flat"
    for policy in ("random", "semantic"):
        rows = _rows_for(policy, f"flat-{policy}", 0)
        path = root / policy / "rows.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    loaded = aggregate.load_engine_rows(root)
    assert loaded["layouts"] == ["flat"]
    assert sorted(route for route, _ in loaded["arms"]) == ["flat-random", "flat-semantic"]


def test_output_files_written(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    result = aggregate.compute_fse_statistics(root)
    out_dir = tmp_path / "out"
    paths = aggregate.write_fse_outputs(result, out_dir)
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["data_state"]["legacy_rows"] == 3
    assert payload["data_state"]["n_arms_skipped_no_engine"] == 1
    assert "auc_v" in payload["arms"][0]["auc"]
    header = paths["coverage_growth_csv"].read_text(encoding="utf-8").splitlines()[0]
    assert header == "route,policy,dimension,eval_index,cov,new_obligations"
    discovery_header = paths["discovery_csv"].read_text(encoding="utf-8").splitlines()[0]
    assert discovery_header == "route,policy,obligation,first_uncover"
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "Arms skipped (no engine metrics)" in markdown
    assert "route0" in markdown


def test_figures_written(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    result = aggregate.compute_fse_statistics(root)
    figures_module = _load_module("make_fse_figures", FIGURES_PATH)
    output_dir = tmp_path / "figures"
    figures = figures_module.generate_figures(result, output_dir)
    assert set(figures) == {"coverage_growth", "discovery", "auc"}
    for pdf_path, png_path in figures.values():
        assert pdf_path.suffix == ".pdf" and pdf_path.stat().st_size > 0
        assert png_path.suffix == ".png" and png_path.stat().st_size > 0
    assert (output_dir / "fig-fse-coverage-growth.pdf").exists()
    assert (output_dir / "fig-fse-discovery.png").exists()
    assert (output_dir / "fig-fse-auc.pdf").exists()
