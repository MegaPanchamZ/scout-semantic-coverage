from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib  # type: ignore
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.observers.coverage import score_traces_against_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score saved activation traces against a KMNC/LSA coverage profile.")
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Path to a .npz trace dump containing 'traces'. Repeat to combine multiple runs.",
    )
    parser.add_argument(
        "--input-glob",
        action="append",
        default=[],
        help="Workspace-relative glob for .npz trace dumps, for example 'research/logs/coverage/*.npz'.",
    )
    parser.add_argument("--profile", required=True, type=Path, help="Path to a .joblib coverage profile.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON path for the score summary.")
    parser.add_argument("--start-index", type=int, default=0, help="Inclusive start index applied after concatenating input traces.")
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive end index applied after concatenating input traces.")
    return parser


def _resolve_input_paths(inputs: list[Path], input_globs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            resolved.extend(sorted(input_path.glob("*.npz")))
            continue
        resolved.append(input_path)

    for pattern in input_globs:
        resolved.extend(sorted(WORKSPACE_ROOT.glob(pattern)))

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        normalized = path.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_paths.append(normalized)
    return unique_paths


def _load_trace_batches(paths: list[Path]) -> np.ndarray:
    if not paths:
        raise ValueError("At least one input trace dump is required.")

    batches: list[np.ndarray] = []
    expected_dim: int | None = None
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Trace dump not found: {path}")
        traces = np.load(path)["traces"]
        if traces.ndim != 2:
            raise ValueError(f"Expected 2D traces in {path}, got shape {traces.shape!r}")
        if expected_dim is None:
            expected_dim = int(traces.shape[1])
        elif int(traces.shape[1]) != expected_dim:
            raise ValueError(
                f"Trace dimension mismatch between inputs: expected {expected_dim}, got {traces.shape[1]} in {path}"
            )
        batches.append(traces)
    return np.concatenate(batches, axis=0)


def _slice_traces(traces: np.ndarray, start_index: int, end_index: int | None) -> np.ndarray:
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    resolved_end = traces.shape[0] if end_index is None else end_index
    if resolved_end <= start_index:
        raise ValueError(f"Invalid trace slice [{start_index}:{resolved_end}] for {traces.shape[0]} traces.")
    if resolved_end > traces.shape[0]:
        raise ValueError(f"--end-index {resolved_end} exceeds trace count {traces.shape[0]}.")
    return traces[start_index:resolved_end]


def main() -> None:
    args = build_parser().parse_args()
    input_paths = _resolve_input_paths(args.input, args.input_glob)
    raw_traces = _load_trace_batches(input_paths)
    traces = _slice_traces(raw_traces, args.start_index, args.end_index)
    profile = joblib.load(args.profile)
    summary = score_traces_against_profile(traces, profile)
    summary["profile_path"] = str(args.profile)
    summary["input_paths"] = [str(path) for path in input_paths]
    summary["source_trace_count"] = int(raw_traces.shape[0])
    summary["start_index"] = int(args.start_index)
    summary["end_index"] = int(args.end_index) if args.end_index is not None else None

    rendered = json.dumps(summary, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()