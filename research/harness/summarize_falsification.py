from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib  # type: ignore
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.observers.coverage import score_traces_against_profile


DEFAULT_ACTIONABLE_PREDICATES = (
    "jaywalking",
    "crossing_path",
    "obstructing",
    "occluded",
    "colliding",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize falsification sessions into thesis-ready Markdown and LaTeX tables."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("research/logs/falsification"),
        help="Root directory containing falsification session folders.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("research/logs/coverage/if-if-safe-prefix-profile.joblib"),
        help="Coverage profile used to rescore candidate trace dumps.",
    )
    parser.add_argument(
        "--nominal-score",
        type=Path,
        default=Path("research/logs/coverage/if-if-safe-prefix-self-score.json"),
        help="Nominal score JSON used as the LSA exceedance threshold reference.",
    )
    parser.add_argument("--threshold", choices=["max", "p95"], default="max")
    parser.add_argument(
        "--actionable-predicate",
        action="append",
        default=[],
        help="Repeat to override the default actionable STSG predicates.",
    )
    parser.add_argument(
        "--include-noncritical",
        action="store_true",
        help="Include evaluations that were not marked as critical edge cases.",
    )
    parser.add_argument(
        "--include-running-sessions",
        action="store_true",
        help="Include completed evaluations from sessions that do not yet have a session-summary.json.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["trigger_radius", "stsg_lead", "lsa_lead", "score", "session", "candidate"],
        default="trigger_radius",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("research/logs/falsification/falsification-summary.md"),
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=Path("research/logs/falsification/falsification-summary.tex"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("research/logs/falsification/falsification-summary.json"),
    )
    parser.add_argument(
        "--caption",
        default="Procedural falsification summary comparing actionable STSG onset against LSA exceedance across completed edge cases.",
    )
    parser.add_argument("--label", default="tab:falsification-procedural-summary")
    return parser


def _workspace_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return (WORKSPACE_ROOT / path).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_lsa_exceedance(lsa_scores: list[float], threshold_value: float) -> int | None:
    for index, value in enumerate(lsa_scores, start=1):
        if float(value) > threshold_value:
            return index
    return None


def _mean_or_none(values: list[int | float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _session_short_name(session_id: str) -> str:
    tokens = [token for token in session_id.split("-") if token]
    if len(tokens) >= 2 and tokens[-1].isdigit():
        return tokens[-2]
    if len(tokens) >= 2 and tokens[-1].startswith("20"):
        return tokens[-2]
    if tokens:
        return tokens[-1]
    return session_id


def _case_label(row: dict[str, Any]) -> str:
    session_id = str(row.get("session_id") or "session")
    candidate_id = str(row.get("candidate_id") or "candidate")
    return f"{_session_short_name(session_id)}:{candidate_id}"


def _find_session_directories(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def _find_evaluation_summaries(session_dir: Path) -> list[Path]:
    return sorted(session_dir.glob("evaluations/*/evaluation-summary.json"))


def _load_semantic_trace(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list semantic trace in {path}.")
    return [frame for frame in payload if isinstance(frame, dict)]


def _first_actionable_semantic_event(
    semantic_trace: list[dict[str, Any]],
    actionable_predicates: set[str],
) -> tuple[int | None, str | None, str | None]:
    for frame in semantic_trace:
        tick = _safe_int(frame.get("tick"))
        predicates = frame.get("predicates", [])
        if not isinstance(predicates, list):
            continue
        for predicate_entry in predicates:
            if not isinstance(predicate_entry, dict):
                continue
            predicate = str(predicate_entry.get("predicate"))
            if predicate not in actionable_predicates:
                continue
            signature = predicate_entry.get("signature")
            return tick, predicate, str(signature) if signature is not None else None
    return None, None, None


def _first_predicate_tick(semantic_trace: list[dict[str, Any]], predicate_name: str) -> int | None:
    for frame in semantic_trace:
        tick = _safe_int(frame.get("tick"))
        predicates = frame.get("predicates", [])
        if not isinstance(predicates, list):
            continue
        for predicate_entry in predicates:
            if isinstance(predicate_entry, dict) and predicate_entry.get("predicate") == predicate_name:
                return tick
    return None


def _derive_impact_tick(deviation_tick: int | None, collision_tick: int | None, ticks_executed: int | None, has_collision: bool) -> tuple[int | None, str]:
    candidates: list[tuple[int, str]] = []
    if deviation_tick is not None:
        candidates.append((deviation_tick, "deviation"))
    if collision_tick is not None:
        candidates.append((collision_tick, "collision"))
    if candidates:
        return min(candidates, key=lambda item: item[0])
    if has_collision and ticks_executed is not None:
        return ticks_executed, "terminal-collision-fallback"
    return None, "none"


def _classify_lsa_actionability(lsa_tick: int | None, semantic_tick: int | None, impact_tick: int | None) -> str:
    if lsa_tick is None:
        return "no-alert"
    if impact_tick is None:
        return "false-positive"
    if semantic_tick is None:
        return "unexplained"
    if lsa_tick < semantic_tick:
        return "early-non-actionable"
    if lsa_tick > impact_tick:
        return "late"
    return "actionable"


def _summarize_evaluation(
    evaluation_path: Path,
    *,
    profile: dict[str, Any],
    threshold_value: float,
    actionable_predicates: set[str],
) -> dict[str, Any] | None:
    evaluation = _load_json(evaluation_path)
    if not isinstance(evaluation, dict):
        return None
    if _safe_int(evaluation.get("subprocess_returncode")) not in (0, None):
        return None
    if evaluation.get("run_error") is not None:
        return None

    run_json_path = Path(str(evaluation.get("run_json_path"))) if evaluation.get("run_json_path") else None
    if run_json_path is None or not run_json_path.exists():
        return None
    run_payload = _load_json(run_json_path)
    if not isinstance(run_payload, dict):
        return None

    semantic_trace_path_raw = run_payload.get("metadata", {}).get("semantic", {}).get("trace_dump_path")
    coverage_trace_path_raw = run_payload.get("metadata", {}).get("coverage", {}).get("trace_dump_path")
    if semantic_trace_path_raw is None or coverage_trace_path_raw is None:
        return None

    semantic_trace_path = Path(str(semantic_trace_path_raw))
    coverage_trace_path = Path(str(coverage_trace_path_raw))
    if not semantic_trace_path.exists() or not coverage_trace_path.exists():
        return None

    semantic_trace = _load_semantic_trace(semantic_trace_path)
    actionable_tick, actionable_predicate, actionable_signature = _first_actionable_semantic_event(
        semantic_trace,
        actionable_predicates,
    )
    collision_tick = _first_predicate_tick(semantic_trace, "colliding")

    telemetry = run_payload.get("metadata", {}).get("telemetry", {})
    if not isinstance(telemetry, dict):
        telemetry = {}
    first_deviation_alert = telemetry.get("first_deviation_alert") or {}
    if not isinstance(first_deviation_alert, dict):
        first_deviation_alert = {}
    deviation_tick = _safe_int(first_deviation_alert.get("tick"))
    ticks_executed = _safe_int(run_payload.get("ticks_executed"))
    has_collision = bool(run_payload.get("terminated_by_collision")) or _safe_int(run_payload.get("collision_count")) not in (None, 0)
    impact_tick, impact_source = _derive_impact_tick(deviation_tick, collision_tick, ticks_executed, has_collision)

    traces = np.load(coverage_trace_path)["traces"]
    score_summary = score_traces_against_profile(traces, profile)
    lsa_scores = [float(value) for value in score_summary.get("lsa_scores", [])]
    lsa_tick = _first_lsa_exceedance(lsa_scores, threshold_value)

    stsg_lead = impact_tick - actionable_tick if impact_tick is not None and actionable_tick is not None else None
    lsa_lead = impact_tick - lsa_tick if impact_tick is not None and lsa_tick is not None else None
    lsa_actionability = _classify_lsa_actionability(lsa_tick, actionable_tick, impact_tick)

    session_dir = evaluation_path.parents[2]
    summary = {
        "session_id": session_dir.name,
        "candidate_id": evaluation.get("candidate_id"),
        "scenario_id": evaluation.get("scenario_id"),
        "mutation_path": evaluation.get("mutation_path"),
        "trigger_radius_m": _safe_float(evaluation.get("mutation_value")),
        "collision": bool(evaluation.get("critical_edge_case")) if _safe_int(evaluation.get("collision_count")) not in (None, 0) or evaluation.get("terminated_by_collision") else False,
        "critical_edge_case": bool(evaluation.get("critical_edge_case")),
        "impact_tick": impact_tick,
        "impact_source": impact_source,
        "deviation_tick": deviation_tick,
        "collision_tick": collision_tick,
        "stsg_tick": actionable_tick,
        "stsg_predicate": actionable_predicate,
        "stsg_signature": actionable_signature,
        "stsg_lead_ticks": stsg_lead,
        "lsa_tick": lsa_tick,
        "lsa_lead_ticks": lsa_lead,
        "lsa_threshold_value": threshold_value,
        "lsa_actionability": lsa_actionability,
        "lsa_false_positive": lsa_actionability == "false-positive",
        "fitness_score": _safe_float(evaluation.get("score")),
        "coverage_lsa_max": _safe_float(evaluation.get("coverage_lsa_max")),
        "coverage_kmnc": _safe_float(evaluation.get("coverage_kmnc")),
        "evaluation_summary_path": str(evaluation_path),
        "run_json_path": str(run_json_path),
        "semantic_trace_path": str(semantic_trace_path),
        "coverage_trace_path": str(coverage_trace_path),
    }
    return summary


def _sort_rows(rows: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    def _key(row: dict[str, Any]) -> tuple[Any, ...]:
        if sort_by == "stsg_lead":
            value = row.get("stsg_lead_ticks")
        elif sort_by == "lsa_lead":
            value = row.get("lsa_lead_ticks")
        elif sort_by == "score":
            value = row.get("fitness_score")
        elif sort_by == "session":
            return (str(row.get("session_id")), str(row.get("candidate_id")))
        elif sort_by == "candidate":
            return (str(row.get("candidate_id")),)
        else:
            value = row.get("trigger_radius_m")
        return (value is None, value, str(row.get("session_id")), str(row.get("candidate_id")))

    reverse = sort_by in {"stsg_lead", "lsa_lead", "score"}
    return sorted(rows, key=_key, reverse=reverse)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actionability_counts: dict[str, int] = {}
    for row in rows:
        actionability = str(row.get("lsa_actionability"))
        actionability_counts[actionability] = actionability_counts.get(actionability, 0) + 1
    return {
        "case_count": len(rows),
        "collision_case_count": sum(1 for row in rows if bool(row.get("collision"))),
        "critical_case_count": sum(1 for row in rows if bool(row.get("critical_edge_case"))),
        "mean_trigger_radius_m": _mean_or_none([_safe_float(row.get("trigger_radius_m")) for row in rows]),
        "mean_impact_tick": _mean_or_none([_safe_int(row.get("impact_tick")) for row in rows]),
        "mean_stsg_lead_ticks": _mean_or_none([_safe_int(row.get("stsg_lead_ticks")) for row in rows]),
        "mean_lsa_lead_ticks": _mean_or_none([_safe_int(row.get("lsa_lead_ticks")) for row in rows]),
        "mean_lsa_tick": _mean_or_none([_safe_int(row.get("lsa_tick")) for row in rows]),
        "mean_stsg_tick": _mean_or_none([_safe_int(row.get("stsg_tick")) for row in rows]),
        "lsa_actionability_counts": actionability_counts,
    }


def _render_markdown(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    lines = [
        "| Case | Trigger Radius (m) | Collision | Impact Tick | STSG Onset / Lead | LSA Onset / Lead | LSA Actionability | Notes |",
        "|------|--------------------|-----------|-------------|-------------------|------------------|-------------------|-------|",
    ]
    for row in rows:
        stsg_cell = f"{_format_int(_safe_int(row.get('stsg_tick')))} / {_format_int(_safe_int(row.get('stsg_lead_ticks')))}"
        lsa_cell = f"{_format_int(_safe_int(row.get('lsa_tick')))} / {_format_int(_safe_int(row.get('lsa_lead_ticks')))}"
        notes = row.get("stsg_signature") or row.get("stsg_predicate") or "n/a"
        case_label = _case_label(row)
        lines.append(
            f"| {case_label} | {_format_float(_safe_float(row.get('trigger_radius_m')))} | {bool(row.get('collision'))} | {_format_int(_safe_int(row.get('impact_tick')))} ({row.get('impact_source')}) | {stsg_cell} | {lsa_cell} | {row.get('lsa_actionability')} | {notes} |"
        )

    counts = ", ".join(
        f"{key}={value}" for key, value in sorted((aggregate.get("lsa_actionability_counts") or {}).items())
    ) or "n/a"
    lines.append(
        f"| Mean over {aggregate.get('case_count', 0)} cases | {_format_float(_safe_float(aggregate.get('mean_trigger_radius_m')))} | {aggregate.get('collision_case_count', 0)}/{aggregate.get('case_count', 0)} | {_format_float(_safe_float(aggregate.get('mean_impact_tick')), 1)} | n/a / {_format_float(_safe_float(aggregate.get('mean_stsg_lead_ticks')), 1)} | n/a / {_format_float(_safe_float(aggregate.get('mean_lsa_lead_ticks')), 1)} | {counts} | Aggregate lead-time summary |"
    )
    return "\n".join(lines)


def _render_latex(rows: list[dict[str, Any]], aggregate: dict[str, Any], caption: str, label: str) -> str:
    body: list[str] = []
    for row in rows:
        case_label = _case_label(row)
        stsg_cell = f"{_format_int(_safe_int(row.get('stsg_tick')))} / {_format_int(_safe_int(row.get('stsg_lead_ticks')))}"
        lsa_cell = f"{_format_int(_safe_int(row.get('lsa_tick')))} / {_format_int(_safe_int(row.get('lsa_lead_ticks')))}"
        notes = str(row.get("stsg_signature") or row.get("stsg_predicate") or "n/a")
        body.append(
            "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                _latex_escape(case_label),
                _latex_escape(_format_float(_safe_float(row.get("trigger_radius_m")))),
                _latex_escape(str(bool(row.get("collision")))),
                _latex_escape(f"{_format_int(_safe_int(row.get('impact_tick')))} ({row.get('impact_source')})"),
                _latex_escape(stsg_cell),
                _latex_escape(lsa_cell),
                _latex_escape(str(row.get("lsa_actionability"))),
                _latex_escape(notes),
            )
        )

    counts = ", ".join(
        f"{key}={value}" for key, value in sorted((aggregate.get("lsa_actionability_counts") or {}).items())
    ) or "n/a"
    body.append(
        "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
            _latex_escape(f"Mean over {aggregate.get('case_count', 0)} cases"),
            _latex_escape(_format_float(_safe_float(aggregate.get("mean_trigger_radius_m")))),
            _latex_escape(f"{aggregate.get('collision_case_count', 0)}/{aggregate.get('case_count', 0)}"),
            _latex_escape(_format_float(_safe_float(aggregate.get("mean_impact_tick")), 1)),
            _latex_escape(f"n/a / {_format_float(_safe_float(aggregate.get('mean_stsg_lead_ticks')), 1)}"),
            _latex_escape(f"n/a / {_format_float(_safe_float(aggregate.get('mean_lsa_lead_ticks')), 1)}"),
            _latex_escape(counts),
            _latex_escape("Aggregate lead-time summary"),
        )
    )

    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\caption{" + _latex_escape(caption) + r"}",
            r"\label{" + _latex_escape(label) + r"}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{0.92}",
            r"\begin{tabular}{@{}p{1.8cm} c c p{1.8cm} p{1.6cm} p{1.6cm} p{2.1cm} p{3.0cm}@{}}",
            r"\toprule",
            r"Case & Trigger Radius (m) & Collision & Impact Tick & STSG Onset / Lead & LSA Onset / Lead & LSA Actionability & Notes \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    args.root = _workspace_path(args.root) or args.root
    args.profile = _workspace_path(args.profile) or args.profile
    args.nominal_score = _workspace_path(args.nominal_score) or args.nominal_score
    args.markdown_output = _workspace_path(args.markdown_output) or args.markdown_output
    args.latex_output = _workspace_path(args.latex_output) or args.latex_output
    args.json_output = _workspace_path(args.json_output) or args.json_output

    actionable_predicates = set(args.actionable_predicate or DEFAULT_ACTIONABLE_PREDICATES)
    profile = joblib.load(args.profile)
    nominal_score = _load_json(args.nominal_score)
    threshold_key = "lsa_max" if args.threshold == "max" else "lsa_p95"
    threshold_value = float(nominal_score[threshold_key])

    rows: list[dict[str, Any]] = []
    for session_dir in _find_session_directories(args.root):
        session_summary_path = session_dir / "session-summary.json"
        session_complete = session_summary_path.exists()
        if not session_complete and not args.include_running_sessions:
            continue
        for evaluation_path in _find_evaluation_summaries(session_dir):
            summary = _summarize_evaluation(
                evaluation_path,
                profile=profile,
                threshold_value=threshold_value,
                actionable_predicates=actionable_predicates,
            )
            if summary is None:
                continue
            if not args.include_noncritical and not bool(summary.get("critical_edge_case")):
                continue
            rows.append(summary)

    rows = _sort_rows(rows, args.sort_by)
    aggregate = _aggregate(rows)
    summary_payload = {
        "root": str(args.root),
        "profile_path": str(args.profile),
        "nominal_score_path": str(args.nominal_score),
        "threshold_mode": args.threshold,
        "threshold_value": threshold_value,
        "actionable_predicates": sorted(actionable_predicates),
        "include_noncritical": args.include_noncritical,
        "include_running_sessions": args.include_running_sessions,
        "sort_by": args.sort_by,
        "aggregate": aggregate,
        "rows": rows,
    }

    markdown = _render_markdown(rows, aggregate)
    latex = _render_latex(rows, aggregate, args.caption, args.label)
    _write_text(args.markdown_output, markdown)
    _write_text(args.latex_output, latex)
    _write_text(args.json_output, json.dumps(summary_payload, indent=2))

    print(markdown)


if __name__ == "__main__":
    main()