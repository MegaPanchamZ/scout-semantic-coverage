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

from research.harness.observers.coverage import build_coverage_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a KMNC/LSA coverage profile from saved traces.")
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Path to a .npz trace dump containing 'traces'. Repeat to combine multiple nominal runs.",
    )
    parser.add_argument(
        "--input-glob",
        action="append",
        default=[],
        help="Workspace-relative glob for .npz trace dumps, for example 'research/logs/coverage/*.npz'.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to the output .joblib profile file.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional JSON summary written next to the binary profile.")
    parser.add_argument("--layer-name", default="auto", help="Logical layer name stored in the profile metadata.")
    parser.add_argument("--k-sections", type=int, default=1000)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--bandwidth", type=float, default=None, help="Optional KDE bandwidth override for LSA.")
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
    profile = build_coverage_profile(
        traces,
        k_sections=args.k_sections,
        pca_components=args.pca_components,
        layer_name=args.layer_name,
        bandwidth=args.bandwidth,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(profile, args.output)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "profile_path": str(args.output),
            "input_paths": [str(path) for path in input_paths],
            "layer_name": profile.get("layer_name"),
            "trace_count": int(profile.get("trace_count", traces.shape[0])),
            "source_trace_count": int(raw_traces.shape[0]),
            "start_index": int(args.start_index),
            "end_index": int(args.end_index) if args.end_index is not None else None,
            "raw_trace_dim": int(profile.get("raw_trace_dim", traces.shape[1])),
            "effective_trace_dim": int(profile.get("effective_trace_dim", traces.shape[1])),
            "k_sections": int(profile.get("k_sections", args.k_sections)),
            "pca_components": int(getattr(profile["pca"], "n_components_", args.pca_components)),
            "bandwidth": float(profile.get("bandwidth")) if profile.get("bandwidth") is not None else None,
        }
        args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote coverage profile to {args.output}")


if __name__ == "__main__":
    main()