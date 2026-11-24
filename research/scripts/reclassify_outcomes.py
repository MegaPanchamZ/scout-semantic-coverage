#!/usr/bin/env python3
"""Offline schema-2 replay or conservative archive audit; writes fresh sidecars only."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.outcome_config import load_outcome_profile, profile_digest
from research.harness.outcome_io import atomic_json, file_digest, iter_evidence, load_capture_manifest
from research.harness.outcome_models import CATEGORIES
from research.harness.safety_outcomes import OutcomeEvaluator, classify_run


_CLASSIFIED_STATUSES = {"replayed_v2_capture", "embedded_v2", "classified_legacy"}


def _outcome_profile_digest(out: Any) -> str | None:
    if not isinstance(out, dict):
        return None
    dig = out.get("profile_digest")
    if not dig and isinstance(out.get("provenance"), dict):
        dig = out["provenance"].get("profile_digest")
    return str(dig) if dig else None


def _outcome_profile_status(out: Any) -> str | None:
    if not isinstance(out, dict):
        return None
    st = out.get("profile_status")
    if not st and isinstance(out.get("provenance"), dict):
        st = out["provenance"].get("profile_status")
    return str(st) if st else None


def audit_archive(root: Path, profile=None, *, workspace_root: Path = WORKSPACE_ROOT) -> list[dict]:
    """Retain every nonempty row, including corrupt rows and missing run artifacts.

    Counts describe source rows. Without persisted attempt IDs they are not
    asserted to be unique executions. This deliberately does not infer identity.
    """
    root = Path(root).resolve()
    records = []
    for row_file in sorted(root.rglob("rows.jsonl")):
        try:
            lines = [ln.rstrip("\r") for ln in row_file.read_bytes().decode("utf-8").split("\n")]
        except (OSError, UnicodeError) as exc:
            records.append({"source": str(row_file), "status": "unreadable_rows_file", "error": str(exc)})
            continue
        for number, line in enumerate(lines, 1):
            if not line.strip(" \t\r"):
                continue
            record = {"source": str(row_file), "line": number,
                      "source_row_sha256": hashlib.sha256(line.encode()).hexdigest()}
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Row must be an object")
            except ValueError as exc:
                record.update(status="corrupt_row", error=str(exc))
                records.append(record)
                continue
            record.update(policy=str(row.get("policy") or row_file.parent.name),
                          attempt_id=row.get("attempt_id"), eval_index=row.get("eval_index"),
                          outcome=classify_run({}, profile))
            raw = row.get("run_json_path")
            row_manifest_raw = row.get("capture_manifest_path")
            row_embedded_v2 = row.get("outcome_v2")
            want_digest = profile_digest(profile) if profile is not None else None

            def _embedded_matches(emb) -> bool:
                return (
                    isinstance(emb, dict)
                    and emb.get("schema_version") == 2
                    and (want_digest is None or _outcome_profile_digest(emb) == want_digest)
                )

            if not isinstance(raw, str) or not raw:
                if isinstance(row_manifest_raw, str) and row_manifest_raw:
                    mpath = Path(row_manifest_raw)
                    if not mpath.is_absolute():
                        mpath = workspace_root / mpath
                    if mpath.is_file():
                        record["capture_manifest_path"] = str(mpath)
                        try:
                            record["outcome"] = replay_capture(mpath, profile).to_dict()
                            record["status"] = "replayed_v2_capture"
                        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
                            record.update(status="unreadable_run", error=str(exc))
                    elif _embedded_matches(row_embedded_v2):
                        record["outcome"] = row_embedded_v2
                        record["status"] = "embedded_v2"
                    else:
                        record.update(status="missing_run", error="No run_json_path")
                elif _embedded_matches(row_embedded_v2):
                    record["outcome"] = row_embedded_v2
                    record["status"] = "embedded_v2"
                else:
                    record.update(status="missing_run", error="No run_json_path")
            else:
                path = Path(raw)
                if not path.is_absolute():
                    # Existing campaign rows use workspace-relative paths.
                    path = workspace_root / path
                record["run_path"] = str(path)
                try:
                    payload = json.loads(path.read_text())  # legacy NaN/Infinity permitted only here
                    if not isinstance(payload, dict):
                        raise ValueError("Run must be an object")
                    record["run_sha256"] = file_digest(path)
                    manifest_raw = payload.get("capture_manifest_path") or row_manifest_raw
                    embedded_v2 = (payload.get("metadata") or {}).get("outcome_v2") if isinstance(payload.get("metadata"), dict) else None
                    if not _embedded_matches(embedded_v2) and _embedded_matches(row_embedded_v2):
                        embedded_v2 = row_embedded_v2
                    if isinstance(manifest_raw, str) and manifest_raw:
                        mpath = Path(manifest_raw)
                        if not mpath.is_absolute():
                            mpath = workspace_root / mpath
                        if mpath.is_file():
                            record["capture_manifest_path"] = str(mpath)
                            record["outcome"] = replay_capture(mpath, profile).to_dict()
                            record["status"] = "replayed_v2_capture"
                        elif _embedded_matches(embedded_v2):
                            record["outcome"] = embedded_v2
                            record["status"] = "embedded_v2"
                        else:
                            record["outcome"] = classify_run(payload, profile)
                            record["status"] = "classified_legacy"
                    elif _embedded_matches(embedded_v2):
                        record["outcome"] = embedded_v2
                        record["status"] = "embedded_v2"
                    else:
                        record["outcome"] = classify_run(payload, profile)
                        record["status"] = "classified_legacy"
                except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
                    record.update(status="unreadable_run", error=str(exc))
            records.append(record)
    return records


def summarize_records(records: list[dict]) -> dict:
    policies = {}
    profile_counts: Counter[str] = Counter()
    for record in records:
        if "outcome" not in record:
            continue
        out = record["outcome"]
        if record.get("status") in _CLASSIFIED_STATUSES:
            dig = _outcome_profile_digest(out)
            if dig:
                profile_counts[dig] += 1
        bucket = policies.setdefault(record["policy"], {"n_source_rows": 0, "metrics": {}})
        bucket["n_source_rows"] += 1
        cats = out.get("categories") if isinstance(out, dict) else None
        cats = cats if isinstance(cats, dict) else {}
        for group, names in CATEGORIES.items():
            grp_dict = cats.get(group) if isinstance(cats.get(group), dict) else {}
            for name in names:
                counts = bucket["metrics"].setdefault(name, dict.fromkeys(("present", "absent", "unknown", "not_applicable"), 0))
                metric_obj = grp_dict.get(name) if isinstance(grp_dict.get(name), dict) else {}
                st = metric_obj.get("status")
                counts[st if isinstance(st, str) and st in counts else "unknown"] += 1
    for bucket in policies.values():
        for counts in bucket["metrics"].values():
            n = counts["present"] + counts["absent"]
            counts.update(n_assessable=n, assessable_rate=counts["present"] / n if n else None)
    observed_digests = sorted(profile_counts)
    return {"n_source_records": len(records), "record_status_counts": dict(Counter(r["status"] for r in records)),
            "observed_profile_digests": observed_digests,
            "profile_digest_counts": dict(sorted(profile_counts.items())),
            "mixed_profiles": len(observed_digests) > 1,
            "counting_unit": "source_rows_not_deduplicated_attempts",
            "interpretation": "Rates condition on assessable source rows. Legacy contact positives can be assessable while negatives are unknown; these rates are not comparative safety estimates.",
            "policies": policies}


def replay_capture(path: Path, profile=None):
    manifest, evidence_path = load_capture_manifest(path)
    evaluator = OutcomeEvaluator(profile)
    from research.harness.outcome_models import FrameEvidence
    for record in iter_evidence(evidence_path):
        if isinstance(record, FrameEvidence):
            evaluator.observe(record)
        else:
            evaluator.record_sensor_event(record)
    return evaluator.finish(manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--search-root", type=Path)
    mode.add_argument("--capture-manifest", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    explicit_profile = load_outcome_profile(args.profile) if args.profile else None
    profile = explicit_profile or load_outcome_profile()
    if args.output_dir.exists():
        parser.error("Use a fresh output directory; source artifacts are never overwritten")
    if args.search_root:
        if not args.search_root.is_dir():
            parser.error("Search root must be an existing directory")
        # Pass explicit_profile (None when --profile omitted) so archived embedded_v2 runs
        # from promoted profiles are preserved faithfully while legacy runs still use default profile.
        records = audit_archive(args.search_root, explicit_profile)
        report = summarize_records(records)
        observed = report["observed_profile_digests"]
        if explicit_profile is not None:
            report.update(
                profile_mode="explicit",
                profile_digest=profile_digest(explicit_profile),
                profile_status=explicit_profile.status,
            )
        elif len(observed) == 1:
            only_digest = observed[0]
            only_status = next(
                (
                    _outcome_profile_status(r.get("outcome"))
                    for r in records
                    if r.get("status") in _CLASSIFIED_STATUSES
                    and _outcome_profile_digest(r.get("outcome")) == only_digest
                    and _outcome_profile_status(r.get("outcome"))
                ),
                profile.status if only_digest == profile_digest(profile) else "unknown",
            )
            report.update(
                profile_mode="preserved_embedded_uniform",
                profile_digest=only_digest,
                profile_status=only_status,
                fallback_profile_digest=profile_digest(profile),
            )
        elif len(observed) > 1:
            report.update(
                profile_mode="mixed_embedded_profiles",
                profile_digest=None,
                profile_status="mixed",
                fallback_profile_digest=profile_digest(profile),
            )
        else:
            report.update(
                profile_mode="default_fallback",
                profile_digest=profile_digest(profile),
                profile_status=profile.status,
            )
        args.output_dir.mkdir(parents=True)
        atomic_json(args.output_dir / "ledger.json", records)
        atomic_json(args.output_dir / "report.json", report)
    else:
        result = replay_capture(args.capture_manifest, profile)
        args.output_dir.mkdir(parents=True)
        atomic_json(args.output_dir / "outcome.json", result.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
