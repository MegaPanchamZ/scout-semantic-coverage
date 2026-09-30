from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.dota_symbolic import DEFAULT_CHECKLIST_PATH  # noqa: E402


DEFAULT_FALSE_ROOT = WORKSPACE_ROOT / "research/logs/falsification"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "research/logs/checklists"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join the Symbolic Checklist against saved DoTA-seeded CARLA runs and emit symbolic coverage summaries."
    )
    parser.add_argument("--checklist", type=Path, default=DEFAULT_CHECKLIST_PATH)
    parser.add_argument("--falsification-root", type=Path, default=DEFAULT_FALSE_ROOT)
    parser.add_argument(
        "--min-match-ratio",
        type=float,
        default=0.5,
        help="Minimum required predicate overlap ratio before a run counts as symbolic coverage for an archetype.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_ROOT / "symbolic_coverage_report.json")
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_ROOT / "symbolic_coverage_report.md")
    parser.add_argument("--output-text", type=Path, default=DEFAULT_OUTPUT_ROOT / "symbolic_coverage_report.txt")
    return parser


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _collect_session_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for session_summary in sorted(root.glob("*/session-summary.json")):
        payload = _load_json(session_summary)
        if not isinstance(payload, dict):
            continue
        scenario_source = payload.get("scenario_source") or {}
        if not isinstance(scenario_source, dict):
            continue
        if scenario_source.get("mode") != "dota-class":
            continue
        generator_metadata = scenario_source.get("generator_metadata") or {}
        source = generator_metadata.get("source") or {}
        canonical_class = str(source.get("anomaly_class") or scenario_source.get("dota_class") or "").strip()
        clip_id = str(source.get("clip_id") or scenario_source.get("dota_clip_id") or "").strip()
        for result in payload.get("results", []) or []:
            if not isinstance(result, dict):
                continue
            records.append(
                {
                    "session_id": payload.get("session_id"),
                    "canonical_class": canonical_class,
                    "clip_id": clip_id,
                    "candidate_id": result.get("candidate_id"),
                    "critical_edge_case": bool(result.get("critical_edge_case")),
                    "terminated_by_collision": bool(result.get("terminated_by_collision")),
                    "collision_count": int(result.get("collision_count") or 0),
                    "semantic_covered_predicates": list(result.get("semantic_covered_predicates") or []),
                    "semantic_covered_signatures": list(result.get("semantic_covered_signatures") or []),
                    "semantic_fulfilled_obligations": list(result.get("semantic_fulfilled_obligations") or []),
                    "semantic_missing_obligations": list(result.get("semantic_missing_obligations") or []),
                    "coverage_kmnc": _safe_float(result.get("coverage_kmnc")),
                    "coverage_lsa_max": _safe_float(result.get("coverage_lsa_max")),
                    "coverage_lsa_mean": _safe_float(result.get("coverage_lsa_mean")),
                    "score": _safe_float(result.get("score")),
                    "scenario_spec_path": result.get("scenario_spec_path"),
                    "run_json_path": result.get("run_json_path"),
                }
            )
    return records


def _best_record(records: list[dict[str, Any]], required_predicates: set[str]) -> dict[str, Any] | None:
    if not records:
        return None

    def _key(record: dict[str, Any]) -> tuple[Any, ...]:
        covered = set(record.get("semantic_covered_predicates") or [])
        fulfilled = set(record.get("semantic_fulfilled_obligations") or [])
        verified = required_predicates.issubset(covered) if required_predicates else bool(covered)
        return (
            int(verified),
            len(fulfilled),
            len(covered.intersection(required_predicates)) if required_predicates else len(covered),
            int(not bool(record.get("terminated_by_collision"))),
            int(bool(record.get("critical_edge_case"))),
            float(record.get("score") or 0.0),
        )

    return max(records, key=_key)


def _match_ratio(record: dict[str, Any], required_predicates: set[str]) -> float:
    covered = set(record.get("semantic_covered_predicates") or [])
    if not covered:
        return 0.0
    if not required_predicates:
        return 1.0
    return float(len(covered.intersection(required_predicates)) / len(required_predicates))


def _render_paragraph(aggregate: dict[str, Any]) -> str:
    archetype_count = int(aggregate["archetype_count"])
    covered_count = int(aggregate["covered_archetype_count"])
    coverage_pct = float(aggregate["symbolic_coverage_pct"])
    verified_runs = int(aggregate["verified_run_count"])
    pass_runs = int(aggregate["verified_pass_run_count"])
    fail_runs = int(aggregate["verified_fail_run_count"])
    kmnc_count = int(aggregate["kmnc_measured_run_count"])
    lsa_count = int(aggregate["lsa_measured_run_count"])
    return (
        f"We extracted {archetype_count} unique traffic anomaly archetypes from the DoTA dataset. "
        f"We achieved {coverage_pct:.1f}% Symbolic Coverage by replaying and semantically verifying {covered_count} of those archetypes in CARLA. "
        f"Across the verified executions, the LASER-oracle branch confirmed {verified_runs} edge-case runs while KMNC was measured on {kmnc_count} runs and LSA on {lsa_count} runs. "
        f"The ADS successfully navigated {pass_runs} of the verified edge-case executions and failed {fail_runs}."
    )


def _render_markdown(archetypes: list[dict[str, Any]], aggregate: dict[str, Any], paragraph: str) -> str:
    lines = [
        "# Symbolic Coverage Report",
        "",
        paragraph,
        "",
        "| Archetype | Clips | Covered | Verified Runs | Pass | Fail | Required Predicates | Representative Clip |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for archetype in archetypes:
        lines.append(
            "| {canonical_class} | {clip_count} | {covered} | {verified_runs} | {pass_runs} | {fail_runs} | {required} | {representative} |".format(
                canonical_class=archetype["canonical_class"],
                clip_count=archetype["clip_count"],
                covered="yes" if archetype["covered"] else "no",
                verified_runs=archetype["verified_run_count"],
                pass_runs=archetype["verified_pass_run_count"],
                fail_runs=archetype["verified_fail_run_count"],
                required=", ".join(archetype["verification_predicates"]) or "n/a",
                representative=archetype.get("representative_clip_id") or "n/a",
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Archetypes: {aggregate['archetype_count']}",
            f"- Covered archetypes: {aggregate['covered_archetype_count']}",
            f"- Symbolic coverage: {aggregate['symbolic_coverage_pct']:.1f}%",
            f"- Verified runs: {aggregate['verified_run_count']}",
            f"- Verified pass runs: {aggregate['verified_pass_run_count']}",
            f"- Verified fail runs: {aggregate['verified_fail_run_count']}",
            f"- KMNC-measured runs: {aggregate['kmnc_measured_run_count']}",
            f"- LSA-measured runs: {aggregate['lsa_measured_run_count']}",
            f"- Obligation-credited runs: {aggregate['obligation_credited_run_count']}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    checklist_payload = _load_json(args.checklist)
    if not isinstance(checklist_payload, dict):
        raise SystemExit(f"Checklist payload at {args.checklist} is not a JSON object.")

    session_records = _collect_session_records(args.falsification_root)
    archetype_summaries: list[dict[str, Any]] = []
    for entry in checklist_payload.get("archetypes", []) or []:
        if not isinstance(entry, dict):
            continue
        canonical_class = str(entry.get("canonical_class") or "")
        matching = [record for record in session_records if record.get("canonical_class") == canonical_class]
        required_predicates = set(entry.get("verification_predicates") or [])
        for record in matching:
            record["match_ratio"] = _match_ratio(record, required_predicates)
        verified = [
            record
            for record in matching
            if float(record.get("match_ratio") or 0.0) >= args.min_match_ratio
        ]
        fully_verified = [record for record in matching if float(record.get("match_ratio") or 0.0) >= 1.0]
        pass_runs = [record for record in verified if not bool(record.get("terminated_by_collision"))]
        fail_runs = [record for record in verified if bool(record.get("terminated_by_collision"))]
        best = _best_record(matching, required_predicates)
        archetype_summaries.append(
            {
                "archetype_id": entry.get("archetype_id"),
                "canonical_class": canonical_class,
                "clip_count": int(entry.get("clip_count") or 0),
                "representative_clip_id": entry.get("representative_clip_id"),
                "verification_predicates": list(entry.get("verification_predicates") or []),
                "matching_run_count": len(matching),
                "verified_run_count": len(verified),
                "fully_verified_run_count": len(fully_verified),
                "verified_pass_run_count": len(pass_runs),
                "verified_fail_run_count": len(fail_runs),
                "covered": bool(verified),
                "best_run": best,
            }
        )

    aggregate = {
        "archetype_count": len(archetype_summaries),
        "covered_archetype_count": sum(1 for item in archetype_summaries if item["covered"]),
        "symbolic_coverage_pct": (
            100.0 * sum(1 for item in archetype_summaries if item["covered"]) / len(archetype_summaries)
            if archetype_summaries
            else 0.0
        ),
        "session_record_count": len(session_records),
        "verified_run_count": sum(item["verified_run_count"] for item in archetype_summaries),
        "verified_pass_run_count": sum(item["verified_pass_run_count"] for item in archetype_summaries),
        "verified_fail_run_count": sum(item["verified_fail_run_count"] for item in archetype_summaries),
        "kmnc_measured_run_count": sum(1 for record in session_records if record.get("coverage_kmnc") is not None),
        "lsa_measured_run_count": sum(1 for record in session_records if record.get("coverage_lsa_max") is not None),
        "obligation_credited_run_count": sum(1 for record in session_records if record.get("semantic_fulfilled_obligations")),
        "mean_kmnc": _mean([record.get("coverage_kmnc") for record in session_records]),
        "mean_lsa_max": _mean([record.get("coverage_lsa_max") for record in session_records]),
    }
    paragraph = _render_paragraph(aggregate)
    output = {
        "checklist_path": str(args.checklist),
        "falsification_root": str(args.falsification_root),
        "aggregate": aggregate,
        "value_proposition_paragraph": paragraph,
        "archetypes": archetype_summaries,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_text.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    args.output_markdown.write_text(_render_markdown(archetype_summaries, aggregate, paragraph), encoding="utf-8")
    args.output_text.write_text(paragraph + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()