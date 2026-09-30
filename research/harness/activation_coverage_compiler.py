from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.activation_graphs import (  # noqa: E402
    active_attribute_signatures,
    active_edge_signatures,
    default_base_graph,
    default_crash_graph,
    load_graph,
    render_activation_summary_markdown,
    runtime_semantic_obligations,
    summarize_activation,
    unmatched_attribute_signatures,
    unmatched_edge_signatures,
)


EXPERIMENT_ROOT = WORKSPACE_ROOT / "research/experiments/EXP-017-scene-graph-activation-mapping"
ARTIFACT_DIR = EXPERIMENT_ROOT / "artifacts"
DATE_TAG = datetime.now(timezone.utc).strftime("%Y%m%d")
DEFAULT_OUTPUT_JSON = ARTIFACT_DIR / f"activation_coverage_plan_{DATE_TAG}.json"
DEFAULT_OUTPUT_MD = ARTIFACT_DIR / f"activation_coverage_plan_{DATE_TAG}.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile an EXP-017 activation summary into runtime semantic obligations and a scenario-spec patch."
        )
    )
    parser.add_argument("--base-graph", type=Path, default=None)
    parser.add_argument("--activated-graph", type=Path, default=None)
    parser.add_argument(
        "--existing-coverage-json",
        type=Path,
        default=None,
        help="Optional JSON file carrying covered_obligations or existing_coverage.covered_obligations.",
    )
    parser.add_argument(
        "--covered-obligation",
        action="append",
        default=[],
        help="Repeat to mark already-covered runtime obligations.",
    )
    parser.add_argument(
        "--scenario-spec",
        type=Path,
        default=None,
        help="Optional ScenarioSpec JSON file to patch with activation coverage obligations.",
    )
    parser.add_argument(
        "--scenario-spec-output",
        type=Path,
        default=None,
        help="Output path for the patched ScenarioSpec JSON. Required when --scenario-spec is provided.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_covered_obligations(existing_coverage_json: Path | None, explicit: list[str]) -> set[str]:
    covered = {str(item).strip() for item in explicit if str(item).strip()}
    if existing_coverage_json is None:
        return covered

    payload = _load_json(existing_coverage_json)
    discovered: list[Any] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("covered_obligations"), list):
            discovered = list(payload.get("covered_obligations") or [])
        else:
            existing = payload.get("existing_coverage") or {}
            if isinstance(existing, dict):
                discovered = list(existing.get("covered_obligations") or [])

    covered.update(str(item).strip() for item in discovered if str(item).strip())
    return covered


def _merge_string_list(existing: Any, incoming: list[str]) -> list[str]:
    merged = {str(item).strip() for item in incoming if str(item).strip()}
    if isinstance(existing, list):
        merged.update(str(item).strip() for item in existing if str(item).strip())
    return sorted(merged)


def _merge_stsg_targets(existing: Any, incoming: dict[str, list[str]]) -> dict[str, list[str]]:
    existing_dict = existing if isinstance(existing, dict) else {}
    return {
        "nodes": _merge_string_list(existing_dict.get("nodes"), incoming.get("nodes", [])),
        "attributes": _merge_string_list(existing_dict.get("attributes"), incoming.get("attributes", [])),
        "relations": _merge_string_list(existing_dict.get("relations"), incoming.get("relations", [])),
        "checklist_items": _merge_string_list(existing_dict.get("checklist_items"), incoming.get("checklist_items", [])),
    }


def build_activation_coverage_plan(summary: dict[str, Any], covered_obligations: set[str]) -> dict[str, Any]:
    attribute_obligations = active_attribute_signatures(summary)
    relation_obligations = active_edge_signatures(summary)
    required_obligations = runtime_semantic_obligations(summary)
    covered_required = sorted(item for item in required_obligations if item in covered_obligations)
    missing_obligations = sorted(item for item in required_obligations if item not in covered_obligations)
    schema_gap_attributes = unmatched_attribute_signatures(summary)
    schema_gap_relations = unmatched_edge_signatures(summary)

    next_test_requests: list[dict[str, Any]] = []
    if missing_obligations:
        next_test_requests.append(
            {
                "axis": "semantic_obligation",
                "uncovered_values": missing_obligations,
                "reason": "Activated crash semantics have not yet been exercised by a retained run.",
            }
        )
    if schema_gap_attributes or schema_gap_relations:
        next_test_requests.append(
            {
                "axis": "schema_alignment",
                "uncovered_values": schema_gap_attributes + schema_gap_relations,
                "reason": "These crash semantics were not represented in the current base prior and need external graph/schema alignment.",
            }
        )

    stsg_targets = {
        "nodes": list(summary.get("active_subgraph", {}).get("nodes", [])),
        "attributes": attribute_obligations,
        "relations": relation_obligations,
        "checklist_items": [
            "activation-coverage",
            str(summary.get("crash_graph") or "activated-graph"),
        ],
    }

    return {
        "schema_version": "2026-05-17",
        "compiler": "activation_coverage_compiler",
        "base_graph": summary.get("base_graph"),
        "activated_graph": summary.get("crash_graph"),
        "mapping_policy": summary.get("mapping_policy"),
        "counts": dict(summary.get("counts", {})),
        "coverage": dict(summary.get("coverage", {})),
        "diagnostics": dict(summary.get("diagnostics", {})),
        "limitations": list(summary.get("limitations", []))
        + [
            "Only active attribute and relation signatures are emitted as runtime obligations because node presence is not directly scored by the current semantic observer.",
        ],
        "required_semantic_obligations": required_obligations,
        "covered_obligations": covered_required,
        "missing_obligations": missing_obligations,
        "active_subgraph": dict(summary.get("active_subgraph", {})),
        "unmatched_crash_semantics": {
            "attribute_signatures": schema_gap_attributes,
            "relation_signatures": schema_gap_relations,
            "raw": dict(summary.get("unmatched_crash_structures", {})),
        },
        "next_test_requests": next_test_requests,
        "scenario_patch": {
            "controller_params": {
                "coverage_obligations": required_obligations,
                "stsg_targets": stsg_targets,
                "activation_coverage": {
                    "base_graph": summary.get("base_graph"),
                    "activated_graph": summary.get("crash_graph"),
                    "mapping_policy": summary.get("mapping_policy"),
                    "coverage": dict(summary.get("coverage", {})),
                    "required_semantic_obligations": required_obligations,
                    "unmatched_crash_semantics": {
                        "attribute_signatures": schema_gap_attributes,
                        "relation_signatures": schema_gap_relations,
                    },
                },
            }
        },
    }


def patch_scenario_spec(scenario_payload: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    patched = dict(scenario_payload)
    controller_params = dict(patched.get("controller_params") or {})
    patch = dict((plan.get("scenario_patch") or {}).get("controller_params") or {})

    controller_params["coverage_obligations"] = _merge_string_list(
        controller_params.get("coverage_obligations"),
        list(patch.get("coverage_obligations") or []),
    )
    controller_params["stsg_targets"] = _merge_stsg_targets(
        controller_params.get("stsg_targets"),
        dict(patch.get("stsg_targets") or {}),
    )
    controller_params["activation_coverage"] = dict(patch.get("activation_coverage") or {})
    patched["controller_params"] = controller_params
    return patched


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# EXP-017 Activation Coverage Plan",
        "",
        f"- Base graph: `{plan['base_graph']}`",
        f"- Activated graph: `{plan['activated_graph']}`",
        f"- Mapping policy: `{plan['mapping_policy']}`",
        f"- Required semantic obligations: `{len(plan['required_semantic_obligations'])}`",
        f"- Covered obligations: `{len(plan['covered_obligations'])}`",
        f"- Missing obligations: `{len(plan['missing_obligations'])}`",
        "",
        "## Counts",
        "",
        f"- Base nodes: `{plan['counts'].get('base_nodes', 0)}`",
        f"- Base attributes: `{plan['counts'].get('base_attributes', 0)}`",
        f"- Base edges: `{plan['counts'].get('base_edges', 0)}`",
        f"- Active nodes: `{plan['counts'].get('active_nodes', 0)}`",
        f"- Active attributes: `{plan['counts'].get('active_attributes', 0)}`",
        f"- Active edges: `{plan['counts'].get('active_edges', 0)}`",
        f"- Unmatched crash nodes: `{plan['counts'].get('unmatched_crash_nodes', 0)}`",
        f"- Unmatched crash attributes: `{plan['counts'].get('unmatched_crash_attributes', 0)}`",
        f"- Unmatched crash edges: `{plan['counts'].get('unmatched_crash_edges', 0)}`",
        "",
        "## Coverage",
        "",
        f"- Node activation ratio: `{plan['coverage'].get('node_activation_ratio', 0.0):.3f}`",
        f"- Attribute activation ratio: `{plan['coverage'].get('attribute_activation_ratio', 0.0):.3f}`",
        f"- Edge activation ratio: `{plan['coverage'].get('edge_activation_ratio', 0.0):.3f}`",
        "",
        "## Required Semantic Obligations",
        "",
    ]
    for obligation in plan["required_semantic_obligations"]:
        marker = "covered" if obligation in set(plan["covered_obligations"]) else "missing"
        lines.append(f"- `{obligation}` [{marker}]")

    lines.extend(["", "## Next Test Requests", ""])
    if plan["next_test_requests"]:
        for request in plan["next_test_requests"]:
            values = ", ".join(request.get("uncovered_values", [])) or "none"
            lines.append(f"- `{request.get('axis', 'unknown')}`: {values}")
            lines.append(f"  Reason: {request.get('reason', 'n/a')}")
    else:
        lines.append("- none")

    lines.extend(["", "## Limitations", ""])
    for limitation in plan["limitations"]:
        lines.append(f"- {limitation}")

    lines.extend(["", "## Activation Summary", ""])
    lines.append(render_activation_summary_markdown(
        {
            "base_graph": plan["base_graph"],
            "crash_graph": plan["activated_graph"],
            "mapping_policy": plan["mapping_policy"],
            "coverage": plan["coverage"],
            "counts": plan["counts"],
            "limitations": plan["limitations"],
        }
    ).strip())
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    if args.scenario_spec is not None and args.scenario_spec_output is None:
        raise SystemExit("--scenario-spec-output is required when --scenario-spec is provided.")

    base_graph = load_graph(args.base_graph, default_base_graph())
    activated_graph = load_graph(args.activated_graph, default_crash_graph())
    summary = summarize_activation(base_graph, activated_graph)
    covered_obligations = _load_covered_obligations(args.existing_coverage_json, list(args.covered_obligation))
    plan = build_activation_coverage_plan(summary, covered_obligations)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(plan), encoding="utf-8")

    if args.scenario_spec is not None and args.scenario_spec_output is not None:
        scenario_payload = _load_json(args.scenario_spec)
        if not isinstance(scenario_payload, dict):
            raise SystemExit(f"ScenarioSpec payload at {args.scenario_spec} is not a JSON object.")
        patched = patch_scenario_spec(scenario_payload, plan)
        args.scenario_spec_output.parent.mkdir(parents=True, exist_ok=True)
        args.scenario_spec_output.write_text(json.dumps(patched, indent=2), encoding="utf-8")

    print(json.dumps(plan, indent=2))
    print(f"Saved JSON plan to {args.output_json}")
    print(f"Saved Markdown plan to {args.output_md}")
    if args.scenario_spec_output is not None:
        print(f"Saved patched ScenarioSpec to {args.scenario_spec_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())