from __future__ import annotations

from typing import Any

from research.harness.models import ScenarioSpec


def _normalize_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value).strip()]


def _binding_obligations(scenario: ScenarioSpec) -> set[str]:
    bindings = scenario.controller_params.get("coverage_bindings") or {}
    if not isinstance(bindings, dict):
        return set()
    return {f"{key}={value}" for key, value in bindings.items()}


def _classify_obligation(obligation: str) -> str:
    if "=" in obligation and "(" not in obligation:
        return "attribute"
    if obligation.endswith(")") and "(" in obligation:
        return "relation"
    return "predicate"


def _group_missing_by_axis(missing: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for obligation in missing:
        if "=" in obligation and "(" not in obligation:
            axis, value = obligation.split("=", 1)
            grouped.setdefault(axis, []).append(value)
        else:
            grouped.setdefault("semantic_relation", []).append(obligation)
    return [
        {
            "axis": axis,
            "uncovered_values": sorted(values),
            "reason": "Retained semantic evidence did not satisfy this obligation yet.",
        }
        for axis, values in sorted(grouped.items())
    ]


def compute_obligation_credit(scenario: ScenarioSpec, semantic_metadata: dict[str, Any]) -> dict[str, Any]:
    obligations = set(_normalize_list(scenario.controller_params.get("coverage_obligations")))
    covered_predicates = set(_normalize_list(semantic_metadata.get("covered_predicates")))
    covered_signatures = set(_normalize_list(semantic_metadata.get("covered_signatures")))
    binding_hits = _binding_obligations(scenario)

    fulfilled: set[str] = set()
    relation_hits: set[str] = set()
    attribute_hits: set[str] = set()
    predicate_hits: set[str] = set()

    for obligation in sorted(obligations):
        obligation_type = _classify_obligation(obligation)
        if obligation_type == "attribute":
            if obligation in binding_hits:
                fulfilled.add(obligation)
                attribute_hits.add(obligation)
            continue

        if obligation in covered_signatures:
            fulfilled.add(obligation)
            if obligation_type == "relation":
                relation_hits.add(obligation)
            else:
                predicate_hits.add(obligation)
            continue

        predicate_name = obligation.split("(", 1)[0] if "(" in obligation else obligation
        if predicate_name in covered_predicates and obligation_type == "predicate":
            fulfilled.add(obligation)
            predicate_hits.add(obligation)

    missing = sorted(obligations - fulfilled)
    return {
        "obligations": sorted(obligations),
        "fulfilled_obligations": sorted(fulfilled),
        "missing_obligations": missing,
        "relation_obligation_hits": sorted(relation_hits),
        "attribute_obligation_hits": sorted(attribute_hits),
        "predicate_obligation_hits": sorted(predicate_hits),
        "binding_hits": sorted(binding_hits.intersection(obligations)),
        "next_test_requests": _group_missing_by_axis(missing),
    }