#!/usr/bin/env python3
"""Replay a normalized capture under explicit profiles; never reuse old counters."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys
from typing import Sequence

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.outcome_config import OutcomeProfile, load_outcome_profile, profile_digest, validate_capture_support
from research.harness.outcome_io import atomic_json, file_digest, iter_evidence, load_capture_manifest
from research.harness.outcome_models import FrameEvidence
from research.harness.safety_outcomes import OutcomeEvaluator


# Pre-pilot experimental sensitivity grid (Plan 3 Section 3):
# one-factor-at-a-time sweeps plus declared joint strict/lenient settings.
FROZEN_SENSITIVITY_GRID: dict[str, tuple[float, ...]] = {
    "ttc_s": (1.0, 1.5, 2.0),
    "proximity_m": (1.5, 2.5, 3.5),
    "deceleration_mps2": (6.0, 8.0, 10.0),
    "stationary_speed_mps": (0.1, 0.2, 0.3),
    "immobilization_s": (5.0, 10.0, 15.0),
}


def _with_settings(base: OutcomeProfile, profile_id: str, overrides: dict[str, float]) -> OutcomeProfile:
    settings = tuple(
        replace(s, value=float(overrides[s.name])) if s.name in overrides else s
        for s in base.settings
    )
    return replace(base, profile_id=profile_id, status="draft", validation_refs=(), settings=settings)


def build_profiles(base_profile: OutcomeProfile | None = None) -> list[OutcomeProfile]:
    """Generate the frozen one-factor-at-a-time + joint sensitivity profile set."""
    base = base_profile or load_outcome_profile()
    profiles: list[OutcomeProfile] = [base]
    seen_digests = {profile_digest(base)}

    for name, values in FROZEN_SENSITIVITY_GRID.items():
        for val in values:
            if abs(base.value(name) - val) < 1e-12:
                continue
            cand = _with_settings(base, f"{base.profile_id}__{name}_{val}", {name: val})
            dig = profile_digest(cand)
            if dig not in seen_digests:
                seen_digests.add(dig)
                profiles.append(cand)

    joint_strict = _with_settings(
        base,
        f"{base.profile_id}__joint_strict",
        {"ttc_s": 1.0, "proximity_m": 1.5, "deceleration_mps2": 10.0, "stationary_speed_mps": 0.1, "immobilization_s": 15.0},
    )
    joint_lenient = _with_settings(
        base,
        f"{base.profile_id}__joint_lenient",
        {"ttc_s": 2.0, "proximity_m": 3.5, "deceleration_mps2": 6.0, "stationary_speed_mps": 0.3, "immobilization_s": 5.0},
    )
    for cand in (joint_strict, joint_lenient):
        dig = profile_digest(cand)
        if dig not in seen_digests:
            seen_digests.add(dig)
            profiles.append(cand)
    return profiles


def _event_signature_set(outcome: dict) -> set[tuple]:
    return {
        (
            str(e.get("label")),
            tuple(sorted(str(a) for a in (e.get("actor_ids") or ()))),
            e.get("start_s"),
            e.get("end_s"),
            str(e.get("state")),
            tuple(e.get("evidence_frames") or ()),
        )
        for e in (outcome.get("events") or ())
        if isinstance(e, dict)
    }


def compare_assessments(report: dict) -> dict:
    """Summarize status, aggregate, episode-count and event-membership changes relative to the baseline profile."""
    profiles = report.get("profiles") or []
    if not profiles:
        return {"baseline_profile_id": None, "changes": []}
    baseline = profiles[0]
    base_cats = baseline["categories"]
    base_aggs = baseline.get("outcome", {}).get("aggregates") or {}
    base_counts = baseline["episode_counts"]
    base_events = _event_signature_set(baseline.get("outcome") or {})
    changes = []
    for entry in profiles[1:]:
        status_diffs = {}
        for group, metrics in entry["categories"].items():
            for name, assessment in metrics.items():
                b_status = base_cats[group][name]["status"]
                e_status = assessment["status"]
                if b_status != e_status:
                    status_diffs[f"{group}.{name}"] = {"baseline": b_status, "variant": e_status}
        entry_aggs = entry.get("outcome", {}).get("aggregates") or {}
        for agg_name in sorted(set(base_aggs) | set(entry_aggs)):
            b_st = (base_aggs.get(agg_name) or {}).get("status")
            e_st = (entry_aggs.get(agg_name) or {}).get("status")
            if b_st != e_st:
                status_diffs[f"aggregates.{agg_name}"] = {"baseline": b_st, "variant": e_st}
        all_labels = sorted(set(base_counts) | set(entry["episode_counts"]))
        count_diffs = {
            label: {"baseline": base_counts.get(label, 0), "variant": entry["episode_counts"].get(label, 0)}
            for label in all_labels
            if base_counts.get(label, 0) != entry["episode_counts"].get(label, 0)
        }
        entry_events = _event_signature_set(entry.get("outcome") or {})
        added = sorted(entry_events - base_events)
        removed = sorted(base_events - entry_events)
        changes.append({
            "profile_id": entry["profile_id"],
            "profile_digest": entry["profile_digest"],
            "status_changes": status_diffs,
            "episode_count_changes": count_diffs,
            "event_membership_changed": bool(added or removed),
            "n_added_events": len(added),
            "n_removed_events": len(removed),
            "n_status_changes": len(status_diffs),
        })
    return {
        "baseline_profile_id": baseline["profile_id"],
        "baseline_profile_digest": baseline["profile_digest"],
        "n_variants": len(changes),
        "changes": changes,
    }


def replay_profiles(capture_path: Path, profiles: Sequence[OutcomeProfile]) -> dict:
    manifest, evidence_path = load_capture_manifest(capture_path)
    evidence_records = tuple(iter_evidence(evidence_path))
    results = []
    for profile in profiles:
        unsupported = validate_capture_support(profile, manifest)
        if unsupported:
            raise ValueError(f"Profile {profile.profile_id} cannot be assessed: {', '.join(unsupported)}")
        evaluator = OutcomeEvaluator(profile)
        for rec in evidence_records:
            if isinstance(rec, FrameEvidence):
                evaluator.observe(rec)
            else:
                evaluator.record_sensor_event(rec)
        result = evaluator.finish(manifest).to_dict()
        results.append({
            "profile_digest": profile_digest(profile),
            "profile_id": profile.profile_id,
            "profile_status": profile.status,
            "categories": result["categories"],
            "episode_counts": dict(Counter(e["label"] for e in result["events"])),
            "outcome": result,
        })
    report = {
        "run_id": manifest.run_id,
        "evidence_sha256": file_digest(evidence_path),
        "profiles": results,
        "scope": "offline_evaluation_sensitivity_not_adaptive_search_performance",
    }
    report["comparison"] = compare_assessments(report)
    return report


def write_report(report: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "sensitivity.json"
    atomic_json(target, report)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, action="append", default=[])
    parser.add_argument("--use-default-grid", action="store_true",
                        help="Replay the frozen pre-pilot one-factor-at-a-time + joint sensitivity grid.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Use a fresh output directory")
    if args.use_default_grid:
        base = load_outcome_profile(args.profile[0]) if args.profile else load_outcome_profile()
        profiles = build_profiles(base)
    elif args.profile:
        profiles = [load_outcome_profile(p) for p in args.profile]
    else:
        parser.error("Specify at least one --profile or --use-default-grid")
    result = replay_profiles(args.capture_manifest, profiles)
    write_report(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
