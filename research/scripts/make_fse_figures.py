#!/usr/bin/env python
"""FSE-submission figures from coverage-engine policy-search metrics.

Usage:
    research/.venv/bin/python research/scripts/make_fse_figures.py \
        --search-root research/logs/policy_search \
        --output-dir research/logs/fse_figures

Figures (PDF + 150 dpi PNG, deterministic):

- ``fig-fse-coverage-growth``: 2x2 panels (node/attribute/relation/hazard
  coverage vs eval index), four policies, mean and bootstrap 95% CI band over
  routes.
- ``fig-fse-discovery``: (a) per-policy median eval index to first-uncover with
  CI, (b) cumulative fraction of discovered obligations vs eval index.
- ``fig-fse-auc``: per-dimension AUC bars per policy with CIs, plus the
  paired-difference panel (semantic minus each baseline) with a zero line.

Style follows ``research/scripts/make_paper_figures.py``: Okabe-Ito palette,
DejaVu Serif, no titles, tight bbox, deterministic resampling (seed 0, 2000
resamples). Metrics are computed by the sibling aggregator
``aggregate_fse_search.py`` so summary and figures cannot diverge.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_ROOT = REPO_ROOT / "research" / "logs" / "policy_search"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research" / "logs" / "fse_figures"
AGGREGATE_PATH = (
    REPO_ROOT
    / "research"
    / "experiments"
    / "EXP-020-policy-comparison"
    / "proof-of-concept"
    / "aggregate_fse_search.py"
)

N_BOOT = 2000
BOOT_SEED = 0

POLICY_ORDER = ("random", "lsa", "kmnc", "semantic")
POLICY_COLORS = {
    "random": "#999999",
    "lsa": "#0072B2",
    "kmnc": "#009E73",
    "semantic": "#D55E00",
    "control": "#666666",
}
DIMENSIONS = ("V", "A", "E", "H")
DIM_LABELS = {
    "V": "node",
    "A": "attribute",
    "E": "relation",
    "H": "hazard",
}
BASELINE_MARKERS = {"random": "o", "lsa": "s", "kmnc": "^"}


def _load_aggregate() -> Any:
    spec = importlib.util.spec_from_file_location("aggregate_fse_search", AGGREGATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_fse_search"] = module
    spec.loader.exec_module(module)
    return module


aggregate = _load_aggregate()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.linewidth": 1.0,
        }
    )


def finalise_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)


def save_figure(fig: plt.Figure, name: str, output_dir: Path) -> tuple[Path, Path]:
    pdf_path = output_dir / f"{name}.pdf"
    png_path = output_dir / f"{name}.png"
    fig.add_artist(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor="none",
            linewidth=0.0,
        )
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.0,
        metadata={"CreationDate": None},
    )
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.0, dpi=150)
    plt.close(fig)
    return pdf_path, png_path


def bootstrap_band(
    matrix: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(matrix, axis=0)
        if matrix.shape[0] < 2:
            return mean, mean.copy(), mean.copy()
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, matrix.shape[0], size=(n_boot, matrix.shape[0]))
        resampled = np.nanmean(matrix[draws], axis=1)
        lo = np.nanpercentile(resampled, 2.5, axis=0)
        hi = np.nanpercentile(resampled, 97.5, axis=0)
    return mean, lo, hi


def bootstrap_ci(
    values: Sequence[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> tuple[float, float]:
    array = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(n_boot, array.size))
    means = array[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def present_policies(result: dict[str, Any]) -> list[str]:
    policies = set(result["policies"])
    ordered = [policy for policy in POLICY_ORDER if policy in policies]
    ordered.extend(sorted(policy for policy in policies if policy not in POLICY_ORDER))
    return ordered


def records_for(result: dict[str, Any], policy: str) -> list[dict[str, Any]]:
    return sorted(
        (record for record in result["arms"] if record["policy"] == policy),
        key=lambda record: record["route"],
    )


def max_eval(result: dict[str, Any]) -> int:
    indices = [index for record in result["arms"] for index in record["eval_indices"]]
    return max(indices) if indices else 0


def coverage_matrix(records: Sequence[dict[str, Any]], dim: str, grid: np.ndarray) -> np.ndarray:
    """Per-route coverage trajectories on a shared eval grid (carry last value)."""
    rows: list[np.ndarray] = []
    for record in records:
        observed = {
            int(eval_index): value
            for eval_index, value in zip(record["eval_indices"], record["cov_series"][dim])
            if value is not None
        }
        if not observed:
            continue
        start = min(observed)
        row = np.full(grid.size, np.nan)
        current: float | None = None
        for position, eval_index in enumerate(grid):
            if eval_index in observed:
                current = observed[eval_index]
            if eval_index >= start and current is not None:
                row[position] = current
        rows.append(row)
    if not rows:
        return np.empty((0, grid.size))
    return np.vstack(rows)


def discovery_matrix(records: Sequence[dict[str, Any]], grid: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    for record in records:
        curve = record["discovery_curve"]
        if not curve["eval_indices"]:
            continue
        observed = dict(zip(curve["eval_indices"], curve["fraction"]))
        row = np.full(grid.size, np.nan)
        current: float | None = None
        for position, eval_index in enumerate(grid):
            if eval_index in observed:
                current = observed[eval_index]
            if current is not None:
                row[position] = current
        rows.append(row)
    if not rows:
        return np.empty((0, grid.size))
    return np.vstack(rows)


def figure_coverage_growth(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    policies = present_policies(result)
    grid = np.arange(0, max_eval(result) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    for ax, dim in zip(axes.flat, DIMENSIONS):
        for policy in policies:
            matrix = coverage_matrix(records_for(result, policy), dim, grid)
            if matrix.size == 0:
                continue
            mean, lo, hi = bootstrap_band(matrix)
            color = POLICY_COLORS.get(policy, "#444444")
            ax.plot(grid, mean, color=color, linewidth=1.2, label=policy)
            ax.fill_between(grid, lo, hi, color=color, alpha=0.16, linewidth=0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(-0.5, max(grid[-1], 1) + 0.5)
        ax.set_ylabel(f"{DIM_LABELS[dim]} coverage")
        finalise_axis(ax)
    for ax in axes[1]:
        ax.set_xlabel("evaluation index")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, -0.01),
        )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    return save_figure(fig, "fig-fse-coverage-growth", output_dir)


def figure_discovery(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    policies = present_policies(result)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    values = []
    for position, policy in enumerate(policies):
        medians = [
            float(record["discovery"]["median_first_uncover"])
            for record in records_for(result, policy)
            if record["discovery"]["median_first_uncover"] is not None
        ]
        if not medians:
            continue
        lo, hi = bootstrap_ci(medians)
        mean = float(np.mean(medians))
        color = POLICY_COLORS.get(policy, "#444444")
        ax.bar(position, mean, width=0.62, color=color, edgecolor="black", linewidth=0.4)
        ax.errorbar(
            [position], [mean], yerr=[[mean - lo], [hi - mean]],
            fmt="none", ecolor="black", elinewidth=0.8, capsize=3.0,
        )
        offsets = np.linspace(-0.14, 0.14, len(medians)) if len(medians) > 1 else np.asarray([0.0])
        ax.scatter(
            position + offsets, medians, s=10, facecolor="white",
            edgecolor="black", linewidth=0.5, zorder=3,
        )
        values.append((policy, mean, lo, hi))
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies)
    ax.set_ylabel("eval index to median first-uncover")
    finalise_axis(ax)

    ax = axes[1]
    grid = np.arange(0, max_eval(result) + 1)
    for policy in policies:
        matrix = discovery_matrix(records_for(result, policy), grid)
        if matrix.size == 0:
            continue
        mean, lo, hi = bootstrap_band(matrix)
        color = POLICY_COLORS.get(policy, "#444444")
        ax.plot(grid, mean, color=color, linewidth=1.2, label=policy)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.16, linewidth=0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(-0.5, max(grid[-1], 1) + 0.5)
    ax.set_xlabel("evaluation index")
    ax.set_ylabel("fraction of discovered obligations")
    ax.legend(loc="lower right")
    finalise_axis(ax)

    fig.tight_layout()
    return save_figure(fig, "fig-fse-discovery", output_dir)


def figure_auc(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    policies = present_policies(result)
    metrics = [f"auc_{dim.lower()}" for dim in DIMENSIONS]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    width = 0.8 / max(len(policies), 1)
    for index, policy in enumerate(policies):
        entry = result["cross_route"].get(policy)
        if entry is None:
            continue
        xs, means, lower, upper = [], [], [], []
        for metric_index, metric in enumerate(metrics):
            mean = entry["auc_mean"].get(metric)
            ci = entry["auc_ci"].get(metric)
            if mean is None:
                continue
            xs.append(metric_index + (index - (len(policies) - 1) / 2.0) * width)
            means.append(mean)
            lower.append(mean - (ci[0] if ci else mean))
            upper.append((ci[1] if ci else mean) - mean)
        ax.bar(
            xs, means, width=width, yerr=[lower, upper], capsize=2.5,
            color=POLICY_COLORS.get(policy, "#444444"), edgecolor="black",
            linewidth=0.4, label=policy,
        )
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([DIM_LABELS[dim] for dim in DIMENSIONS])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("coverage AUC (normalised)")
    ax.legend(loc="upper right", ncol=2)
    finalise_axis(ax)

    ax = axes[1]
    baselines = [policy for policy in result["config"]["comparison_policies"] if policy in policies]
    for comparison_index, baseline in enumerate(baselines):
        xs, means, lower, upper = [], [], [], []
        for metric_index, metric in enumerate(metrics):
            record = next(
                (
                    item
                    for item in result["pairwise"].get(metric, [])
                    if item["policy_b"] == baseline
                ),
                None,
            )
            if record is None or record["mean_difference"] is None:
                continue
            ci = record["mean_difference_ci"]
            xs.append(metric_index + (comparison_index - (len(baselines) - 1) / 2.0) * 0.22)
            means.append(record["mean_difference"])
            lower.append(record["mean_difference"] - (ci[0] if ci else record["mean_difference"]))
            upper.append((ci[1] if ci else record["mean_difference"]) - record["mean_difference"])
        if not xs:
            continue
        ax.errorbar(
            xs, means, yerr=[lower, upper], fmt=BASELINE_MARKERS.get(baseline, "o"),
            color=POLICY_COLORS.get(baseline, "#444444"), markersize=4.0,
            elinewidth=0.8, capsize=2.5, label=f"vs {baseline}",
        )
    ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([DIM_LABELS[dim] for dim in DIMENSIONS])
    ax.set_ylabel("semantic − baseline AUC")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
    finalise_axis(ax)

    fig.tight_layout()
    return save_figure(fig, "fig-fse-auc", output_dir)


def generate_figures(result: dict[str, Any], output_dir: Path) -> dict[str, tuple[Path, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    return {
        "coverage_growth": figure_coverage_growth(result, output_dir),
        "discovery": figure_discovery(result, output_dir),
        "auc": figure_auc(result, output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the FSE submission figures.")
    parser.add_argument(
        "--search-root",
        type=Path,
        default=DEFAULT_SEARCH_ROOT,
        help="Root containing <route>/<policy>/rows.jsonl or <policy>/rows.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the PDF/PNG figure pairs.",
    )
    parser.add_argument("--seed", type=int, default=BOOT_SEED, help="Bootstrap RNG seed.")
    parser.add_argument("--bootstrap", type=int, default=N_BOOT, help="Bootstrap resamples.")
    parser.add_argument(
        "--min-test-routes", type=int, default=6, help="Minimum paired routes for Wilcoxon."
    )
    return parser


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    search_root = _resolve(args.search_root)
    output_dir = _resolve(args.output_dir)
    result = aggregate.compute_fse_statistics(
        search_root,
        seed=args.seed,
        n_boot=args.bootstrap,
        min_test_routes=args.min_test_routes,
    )
    if not result["arms"]:
        state = result["data_state"]
        print(
            f"no engine rows under {search_root}: {state['n_arms_skipped_no_engine']} arm(s) "
            f"skipped for lacking engine_* fields, {state['n_arms_engine_no_valid_rows']} engine "
            "arm(s) with no valid rows; nothing to plot"
        )
        return 2
    figures = generate_figures(result, output_dir)
    for name, (pdf_path, png_path) in figures.items():
        print(f"wrote {pdf_path} [{name}]")
        print(f"wrote {png_path} [{name}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
