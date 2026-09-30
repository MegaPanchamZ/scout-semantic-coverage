from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_base_graph() -> dict[str, Any]:
    return {
        "graph_name": "nuscenes_complete_prior_poc",
        "mapping_policy": "exact_id_intersection_phase0_probe",
        "schema_inventory": {
            "node_types": ["vehicle", "pedestrian", "traffic_light"],
            "attribute_vocab": [
                "ego_vehicle",
                "car",
                "pedestrian",
                "traffic_light",
                "moving",
                "braking",
                "stationary",
                "jaywalking",
                "crossing",
                "waiting",
                "red",
                "green",
            ],
            "edge_vocab": [
                "approaching",
                "in_front_of",
                "crossing_path",
                "obeying_signal",
                "near_lane",
                "colliding",
            ],
        },
        "scene_graph": {
            "nodes": [
                {"id": "ego", "type": "vehicle", "attributes": ["ego_vehicle", "moving"]},
                {"id": "lead", "type": "vehicle", "attributes": ["car", "braking"]},
                {"id": "ped", "type": "pedestrian", "attributes": ["pedestrian", "jaywalking", "crossing"]},
                {"id": "light", "type": "traffic_light", "attributes": ["traffic_light", "red"]},
                {"id": "parked", "type": "vehicle", "attributes": ["car", "stationary"]},
            ],
            "edges": [
                {"source": "ego", "relation": "approaching", "target": "lead"},
                {"source": "lead", "relation": "in_front_of", "target": "ego"},
                {"source": "ped", "relation": "crossing_path", "target": "ego"},
                {"source": "ego", "relation": "approaching", "target": "ped"},
                {"source": "ego", "relation": "obeying_signal", "target": "light"},
                {"source": "parked", "relation": "near_lane", "target": "ego"},
            ],
        },
    }


def default_crash_graph() -> dict[str, Any]:
    return {
        "graph_name": "dota_activated_crash_poc",
        "nodes": [
            {"id": "ego", "type": "vehicle", "attributes": ["ego_vehicle", "moving"]},
            {"id": "ped", "type": "pedestrian", "attributes": ["pedestrian", "jaywalking", "crossing"]},
            {"id": "lead", "type": "vehicle", "attributes": ["car", "braking"]},
        ],
        "edges": [
            {"source": "ped", "relation": "crossing_path", "target": "ego"},
            {"source": "ego", "relation": "approaching", "target": "ped"},
            {"source": "ego", "relation": "colliding", "target": "ped"},
            {"source": "ego", "relation": "approaching", "target": "lead"},
        ],
    }


def load_graph(path: Path | None, default_graph: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return default_graph
    return json.loads(path.read_text(encoding="utf-8"))


def graph_payload(graph: dict[str, Any]) -> dict[str, Any]:
    return graph.get("scene_graph") or graph


def node_index(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payload = graph_payload(graph)
    return {str(node["id"]): node for node in payload.get("nodes", [])}


def edge_set(graph: dict[str, Any]) -> set[tuple[str, str, str]]:
    payload = graph_payload(graph)
    return {
        (str(edge["source"]), str(edge["relation"]), str(edge["target"]))
        for edge in payload.get("edges", [])
    }


def attribute_set(graph: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    payload = graph_payload(graph)
    for node in payload.get("nodes", []):
        node_id = str(node["id"])
        for attribute in node.get("attributes", []):
            pairs.add((node_id, str(attribute)))
    return pairs


def _ratio(active: int, total: int) -> float:
    return float(active / total) if total else 0.0


def summarize_activation(base_graph: dict[str, Any], crash_graph: dict[str, Any]) -> dict[str, Any]:
    base_nodes = node_index(base_graph)
    crash_nodes = node_index(crash_graph)

    base_node_ids = set(base_nodes)
    crash_node_ids = set(crash_nodes)
    active_node_ids = sorted(base_node_ids & crash_node_ids)
    unmatched_crash_nodes = sorted(crash_node_ids - base_node_ids)

    base_attributes = attribute_set(base_graph)
    crash_attributes = attribute_set(crash_graph)
    active_attributes = sorted(base_attributes & crash_attributes)
    unmatched_crash_attributes = sorted(crash_attributes - base_attributes)

    base_edges = edge_set(base_graph)
    crash_edges = edge_set(crash_graph)
    active_edges = sorted(base_edges & crash_edges)
    unmatched_crash_edges = sorted(crash_edges - base_edges)

    return {
        "base_graph": base_graph.get("graph_name", "base_graph"),
        "crash_graph": crash_graph.get("graph_name", "crash_graph"),
        "mapping_policy": str(base_graph.get("mapping_policy") or "exact_id_intersection_phase0_probe"),
        "limitations": [
            "This Phase 0 probe assumes a shared schema already exists.",
            "It uses exact node-id overlap, not role alignment or schema matching across datasets.",
            "Per-type activation ratios are the primary signal; any combined scalar is exploratory only.",
        ],
        "counts": {
            "base_nodes": len(base_node_ids),
            "base_attributes": len(base_attributes),
            "base_edges": len(base_edges),
            "active_nodes": len(active_node_ids),
            "active_attributes": len(active_attributes),
            "active_edges": len(active_edges),
            "unmatched_crash_nodes": len(unmatched_crash_nodes),
            "unmatched_crash_attributes": len(unmatched_crash_attributes),
            "unmatched_crash_edges": len(unmatched_crash_edges),
        },
        "coverage": {
            "node_activation_ratio": _ratio(len(active_node_ids), len(base_node_ids)),
            "attribute_activation_ratio": _ratio(len(active_attributes), len(base_attributes)),
            "edge_activation_ratio": _ratio(len(active_edges), len(base_edges)),
        },
        "diagnostics": {
            "exploratory_union_activation_ratio": _ratio(
                len(active_node_ids) + len(active_attributes) + len(active_edges),
                len(base_node_ids) + len(base_attributes) + len(base_edges),
            ),
        },
        "active_subgraph": {
            "nodes": active_node_ids,
            "attributes": [{"node": node_id, "attribute": attribute} for node_id, attribute in active_attributes],
            "edges": [
                {"source": source, "relation": relation, "target": target}
                for source, relation, target in active_edges
            ],
        },
        "unmatched_crash_structures": {
            "nodes": unmatched_crash_nodes,
            "attributes": [{"node": node_id, "attribute": attribute} for node_id, attribute in unmatched_crash_attributes],
            "edges": [
                {"source": source, "relation": relation, "target": target}
                for source, relation, target in unmatched_crash_edges
            ],
        },
    }


def attribute_signature(node_id: str, attribute: str) -> str:
    return f"{attribute}({node_id})"


def edge_signature(source: str, relation: str, target: str) -> str:
    return f"{relation}({source},{target})"


def active_attribute_signatures(summary: dict[str, Any]) -> list[str]:
    return sorted(
        {
            attribute_signature(str(item["node"]), str(item["attribute"]))
            for item in summary.get("active_subgraph", {}).get("attributes", [])
        }
    )


def active_edge_signatures(summary: dict[str, Any]) -> list[str]:
    return sorted(
        {
            edge_signature(str(item["source"]), str(item["relation"]), str(item["target"]))
            for item in summary.get("active_subgraph", {}).get("edges", [])
        }
    )


def unmatched_attribute_signatures(summary: dict[str, Any]) -> list[str]:
    return sorted(
        {
            attribute_signature(str(item["node"]), str(item["attribute"]))
            for item in summary.get("unmatched_crash_structures", {}).get("attributes", [])
        }
    )


def unmatched_edge_signatures(summary: dict[str, Any]) -> list[str]:
    return sorted(
        {
            edge_signature(str(item["source"]), str(item["relation"]), str(item["target"]))
            for item in summary.get("unmatched_crash_structures", {}).get("edges", [])
        }
    )


def runtime_semantic_obligations(summary: dict[str, Any]) -> list[str]:
    return sorted(set(active_attribute_signatures(summary) + active_edge_signatures(summary)))


def render_activation_summary_markdown(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    counts = summary["counts"]
    lines = []
    lines.append("# EXP-017 PoC Result - Scene Graph Activation Mapping")
    lines.append("")
    lines.append(f"- Base graph: `{summary['base_graph']}`")
    lines.append(f"- Crash graph: `{summary['crash_graph']}`")
    lines.append(f"- Mapping policy: `{summary['mapping_policy']}`")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Base nodes: `{counts['base_nodes']}`")
    lines.append(f"- Base attributes: `{counts['base_attributes']}`")
    lines.append(f"- Base edges: `{counts['base_edges']}`")
    lines.append(f"- Active nodes: `{counts['active_nodes']}`")
    lines.append(f"- Active attributes: `{counts['active_attributes']}`")
    lines.append(f"- Active edges: `{counts['active_edges']}`")
    lines.append(f"- Unmatched crash nodes: `{counts['unmatched_crash_nodes']}`")
    lines.append(f"- Unmatched crash attributes: `{counts['unmatched_crash_attributes']}`")
    lines.append(f"- Unmatched crash edges: `{counts['unmatched_crash_edges']}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Node activation ratio: `{coverage['node_activation_ratio']:.3f}`")
    lines.append(f"- Attribute activation ratio: `{coverage['attribute_activation_ratio']:.3f}`")
    lines.append(f"- Edge activation ratio: `{coverage['edge_activation_ratio']:.3f}`")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in summary["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Active structures are crash-relevant pieces that can be treated as a semantic coverage signal.")
    lines.append("- Unmatched crash structures identify where the base prior or shared schema is incomplete.")
    lines.append("- The mixed union score is retained only as an exploratory diagnostic and should not be treated as the primary coverage metric.")
    return "\n".join(lines) + "\n"