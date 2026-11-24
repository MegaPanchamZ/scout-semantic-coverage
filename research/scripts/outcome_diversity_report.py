#!/usr/bin/env python3
"""Replay normalized captures into an offline event-pattern diversity sidecar."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.outcome_config import load_outcome_profile
from research.harness.outcome_diversity import build_run_signature, cluster_outcomes, summarize_diversity
from research.harness.outcome_io import atomic_json, file_digest, iter_evidence, load_capture_manifest
from research.harness.outcome_models import FrameEvidence
from research.harness.safety_outcomes import OutcomeEvaluator


def report_captures(paths, profile=None):
    records, outcomes, sources = [], [], []
    seen_traces = set()
    for path in paths:
        path = Path(path).resolve()
        manifest, evidence_path = load_capture_manifest(path)
        trace_key = (manifest.run_id, file_digest(evidence_path))
        if trace_key in seen_traces:
            raise ValueError("The same run/evidence was supplied more than once")
        seen_traces.add(trace_key)
        evidence = tuple(iter_evidence(evidence_path))
        evaluator = OutcomeEvaluator(profile)
        for record in evidence:
            if isinstance(record, FrameEvidence):
                evaluator.observe(record)
            else:
                evaluator.record_sensor_event(record)
        outcome = evaluator.finish(manifest)
        outcomes.append(outcome)
        record_id = str(path)
        records.append({"record_id": record_id, "signature": build_run_signature(outcome, evidence)})
        sources.append({"record_id": record_id, "manifest_sha256": file_digest(path), "evidence_sha256": trace_key[1],
                        "run_id": manifest.run_id})
    report = cluster_outcomes(records)
    report.update(sources=sorted(sources, key=lambda r: r["record_id"]), counts=summarize_diversity(outcomes))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", type=Path, action="append", required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Use a fresh output directory")
    result = report_captures(args.capture_manifest, load_outcome_profile(args.profile) if args.profile else None)
    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir / "diversity.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
