"""Regression tests for ObligationScheduler.select (the selection contract).

Pins the review item: selection must be driven by the real covered set, so a
credited target advances instead of persisting.
"""
from __future__ import annotations

from research.harness.hazard_search import (
    DEFAULT_TEMPLATE,
    OBLIGATION_TEMPLATE_MAP,
    ObligationScheduler,
)


def test_select_returns_initial_target_and_template():
    scheduler = ObligationScheduler(uncovered=set())
    universe = {"in_front_of(vehicle,ego)", "crossing_path(pedestrian,ego)"}
    target, template = scheduler.select(universe, set())
    # both are relation-priority; alphabetical tie-break picks crossing_path
    assert target == "crossing_path(pedestrian,ego)"
    assert template == "pedestrian_crossing"


def test_select_advances_after_credit():
    scheduler = ObligationScheduler(uncovered=set())
    universe = {"in_front_of(vehicle,ego)", "crossing_path(pedestrian,ego)"}
    first, _ = scheduler.select(universe, set())
    second, template = scheduler.select(universe - {first}, {first})
    assert second == "in_front_of(vehicle,ego)"
    assert template == "lead_vehicle_braking"
    assert first in scheduler.history


def test_select_does_not_advance_without_credit():
    scheduler = ObligationScheduler(uncovered=set())
    universe = {"crossing_path(pedestrian,ego)", "in_front_of(vehicle,ego)"}
    first, _ = scheduler.select(universe, set())
    # nothing credited and the target is still uncovered: stay on it
    again, _ = scheduler.select(universe, set())
    assert again == first


def test_select_exhaustion_returns_none_and_default_template():
    scheduler = ObligationScheduler(uncovered=set())
    universe = {"braking(vehicle)"}
    scheduler.select(universe, set())
    target, template = scheduler.select(set(), universe)
    assert target is None
    assert template == DEFAULT_TEMPLATE


def test_obligation_template_map_routes_predicates_to_templates():
    assert OBLIGATION_TEMPLATE_MAP["braking(vehicle)"] == "lead_vehicle_braking"
    assert OBLIGATION_TEMPLATE_MAP["in_front_of(vehicle,ego)"] == "lead_vehicle_braking"
    assert OBLIGATION_TEMPLATE_MAP["same_lane(vehicle,ego)"] == "lead_vehicle_braking"
    assert OBLIGATION_TEMPLATE_MAP["crossing_path(pedestrian,ego)"] == "pedestrian_crossing"
    assert OBLIGATION_TEMPLATE_MAP["jaywalking(pedestrian)"] == "pedestrian_crossing"
