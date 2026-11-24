from dataclasses import asdict, replace

import pytest

from research.harness.outcome_models import CaptureManifest, ScenarioChecks, SensorEvidence
from research.harness.scenario_quality import (
    assess_validity, canonical_scenario_digest, parameter_similarity, realized_initial_state_digest,
)
from research.harness.search_space import make_legacy_space
from research.tests.outcome_helpers import actor, frame, manifest
from research.tests.test_outcome_detectors import evaluate, status


VALID = ScenarioChecks(True, True, True, True, True, evidence_refs=("constructed_checks",))
EXECUTION = {"map_digest": "map", "ads_digest": "ads", "route_digest": "route", "seed": 7}


def test_invalid_setup_is_distinct_from_execution_failure():
    assert assess_validity(replace(VALID, required_actors_spawned=False)).status == "present"
    assert assess_validity(ScenarioChecks(execution_error="server lost")).status == "unknown"
    assert assess_validity(replace(VALID, controller_ready=False)).status == "unknown"
    assert assess_validity(replace(VALID, execution_error="controller crashed after start")).status == "absent"
    assert CaptureManifest.from_dict(asdict(manifest(scenario_checks=VALID))).scenario_checks == VALID


def test_complete_valid_empty_scene_is_low_interaction_with_explicit_scope():
    result = evaluate([frame(i) for i in range(101)], capture=manifest(10, scenario_checks=VALID))
    assert status(result, "quality", "invalid_scenario") == "absent"
    assert status(result, "quality", "low_interaction") == "present"


def test_safe_parallel_interaction_is_not_low_interaction_or_near_miss():
    frames = [frame(i, actors=(actor("a", x=i, y=8, vx=10, role="neighbor"),)) for i in range(101)]
    result = evaluate(frames, capture=manifest(10, scenario_checks=VALID))
    assert status(result, "quality", "low_interaction") == "absent"
    assert status(result, "safety", "near_miss") == "absent"
    opportunities = result["provenance"]["interaction_opportunities"]
    assert len(opportunities) == 1
    assert opportunities[0]["role"] == "neighbor"
    assert opportunities[0]["min_gap_m"] == 6


def test_fast_actor_beyond_radius_can_supply_projected_interaction():
    frames = [frame(0, actors=(actor("a", x=30, vx=-10),))]
    result = evaluate(frames, capture=manifest(scenario_checks=VALID))
    assert status(result, "quality", "low_interaction") == "absent"


@pytest.mark.parametrize("mode", ["short", "gap", "invalid", "missing_checks", "no_sensor_coverage", "small_radius"])
def test_absence_of_collision_does_not_establish_low_interaction(mode):
    frames = [frame(i) for i in range(101)]
    capture = manifest(10, scenario_checks=VALID)
    if mode == "short":
        frames, capture = frames[:11], manifest(scenario_checks=VALID)
    elif mode == "gap":
        del frames[50]
    elif mode == "invalid":
        capture = replace(capture, scenario_checks=replace(VALID, route_feasible=False))
    elif mode == "missing_checks":
        capture = replace(capture, scenario_checks=None)
    elif mode == "no_sensor_coverage":
        capture = replace(capture, channels={k: v for k, v in capture.channels.items() if k != "collision"})
    else:
        capture = replace(capture, scan_radius_m=5)
    assert status(evaluate(frames, capture=capture), "quality", "low_interaction") == "unknown"


def test_sensor_contact_cannot_be_misclassified_as_low_interaction():
    contact = SensorEvidence("run", 50, 5, "collision", "not_in_actor_capture")
    result = evaluate([frame(i) for i in range(101)], [contact], manifest(10, scenario_checks=VALID))
    assert status(result, "quality", "low_interaction") == "absent"


def test_requested_identity_excludes_only_named_top_level_cosmetics():
    spec = {"scenario_id": "one", "description": "text", "weather": {"rain": 0}, "controller": {"speed": 1}}
    digest = canonical_scenario_digest(spec, EXECUTION)
    assert digest == canonical_scenario_digest({**spec, "scenario_id": "two", "controller": {"speed": 1.0}}, EXECUTION)
    assert digest != canonical_scenario_digest({**spec, "controller": {"speed": 2}}, EXECUTION)
    assert digest != canonical_scenario_digest({**spec, "controller": {"speed": 1, "description": "operative"}}, EXECUTION)
    assert digest != canonical_scenario_digest(spec, {**EXECUTION, "seed": 8})
    with pytest.raises(ValueError):
        canonical_scenario_digest(spec, {"seed": 7})
    with pytest.raises(ValueError):
        canonical_scenario_digest({"x": float("inf")}, EXECUTION)


def test_realized_identity_ignores_ephemeral_ids_but_requires_roles():
    actors = [actor("1", role="ego"), actor("2", role="subject", x=10)]
    digest = realized_initial_state_digest(actors, EXECUTION)
    assert digest == realized_initial_state_digest([replace(actors[1], actor_id="x"), replace(actors[0], actor_id="y")], EXECUTION)
    assert digest != realized_initial_state_digest([actors[0], replace(actors[1], x=11)], EXECUTION)
    with pytest.raises(ValueError):
        realized_initial_state_digest([actor()], EXECUTION)
    with pytest.raises(ValueError):
        realized_initial_state_digest([actor(role="same"), actor("2", role="same")], EXECUTION)


def test_parameter_distance_does_not_round_clamp_or_assert_behaviour_identity():
    space = make_legacy_space(low=5, high=35)
    assert parameter_similarity({"trigger_radius_m": 5}, {"trigger_radius_m": 35}, space)["distance"] == 1
    assert parameter_similarity({"trigger_radius_m": 5}, {"trigger_radius_m": 5.001}, space)["distance"] > 0
    with pytest.raises(ValueError):
        parameter_similarity({"trigger_radius_m": True}, {"trigger_radius_m": 35}, space)
    with pytest.raises(ValueError):
        parameter_similarity({"trigger_radius_m": 4}, {"trigger_radius_m": 35}, space)
