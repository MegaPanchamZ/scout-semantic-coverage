"""Tests for the declarative multi-dimensional search space (EXP-020 campaign).

The campaign requires at least four candidate dimensions so that a coverage gap
can imply a distinguishable action; the pilot sampled only ``trigger_radius_m``.
These tests pin the declarative contract, the legacy reproduction path, the
ScenarioSpec round trip, the runtime trigger-tick deadline, and the shared-space
wiring of all four selection policies and the optimiser sampler.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.harness import optimiser  # noqa: E402
from research.harness.models import ScenarioSpec  # noqa: E402
from research.harness.scenario_runtime import ThresholdCrossingAdversaryController  # noqa: E402
from research.harness.search_space import (  # noqa: E402
    LATERAL_OFFSET_ANNOTATION,
    LONGITUDINAL_OFFSET_ANNOTATION,
    SearchDimension,
    SearchSpace,
    SearchSpaceError,
    make_campaign_space,
    make_legacy_space,
)

POLICY_SEARCH_PATH = (
    REPO_ROOT / "research" / "experiments" / "EXP-020-policy-comparison" / "proof-of-concept" / "policy_search.py"
)
EXAMPLE_SPEC_PATH = REPO_ROOT / "research" / "harness" / "examples" / "town01_threshold_crossing.example.json"
CAMPAIGN_DIMENSION_NAMES = (
    "trigger_radius_m",
    "subject_speed_mps",
    "staging_lateral_offset_m",
    "staging_longitudinal_offset_m",
    "trigger_tick",
)


def _load_policy_search():
    spec = importlib.util.spec_from_file_location("exp020_policy_search", POLICY_SEARCH_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _example_payload() -> dict:
    return json.loads(EXAMPLE_SPEC_PATH.read_text(encoding="utf-8"))


def _example_spec() -> ScenarioSpec:
    return ScenarioSpec.from_dict(_example_payload())


def test_campaign_space_declares_the_required_dimensions() -> None:
    space = make_campaign_space(seed=13)
    assert len(space) >= 4
    assert space.names() == CAMPAIGN_DIMENSION_NAMES
    assert space.dimension("trigger_radius_m").spec_path == "controller_params.trigger_radius_m"
    assert space.dimension("subject_speed_mps").spec_path == "controller_params.speed"
    assert space.dimension("staging_lateral_offset_m").spec_path == "controller_params.spawn_transform.location"
    assert space.dimension("staging_longitudinal_offset_m").spec_path == "controller_params.spawn_transform.location"
    assert space.dimension("trigger_tick").spec_path == "controller_params.trigger_tick"
    assert space.dimension("trigger_radius_m").low == 5.0
    assert space.dimension("trigger_radius_m").high == 35.0
    longitudinal = space.dimension("staging_longitudinal_offset_m")
    assert (longitudinal.low, longitudinal.high) == (-10.0, 30.0)
    assert longitudinal.kind == "longitudinal_offset_m"
    assert longitudinal.runtime == "spec-baked"
    assert longitudinal.default == 0.0
    assert all(dimension.runtime in {"live", "spec-baked"} for dimension in space.dimensions())


def test_sampling_respects_bounds_grid_and_types() -> None:
    space = make_campaign_space(seed=1)
    for seed in (1, 2, 3):
        for candidate in space.sample_many(250, random.Random(seed)):
            space.validate(candidate.values)
            for dimension in space.dimensions():
                value = candidate.values[dimension.name]
                assert dimension.low <= float(value) <= dimension.high
                if dimension.kind == "int":
                    assert isinstance(value, int)
                    assert (value - int(dimension.low)) % int(dimension.step) == 0
                else:
                    assert isinstance(value, float)
                    assert round(value, dimension.decimals) == value


def test_sampling_is_deterministic_given_a_seed() -> None:
    first = make_campaign_space(seed=42)
    second = make_campaign_space(seed=42)
    other_seed = make_campaign_space(seed=43)
    first_draws = [candidate.to_dict() for candidate in first.sample_many(50)]
    second_draws = [candidate.to_dict() for candidate in second.sample_many(50)]
    other_draws = [candidate.to_dict() for candidate in other_seed.sample_many(50)]
    assert first_draws == second_draws
    assert first_draws != other_draws


def test_space_serialisation_round_trips() -> None:
    space = make_campaign_space(seed=7)
    restored = SearchSpace.from_dict(space.to_dict())
    assert restored.names() == space.names()
    assert restored.signature() == space.signature()
    assert restored.to_dict() == space.to_dict()


def test_round_trip_through_scenario_spec() -> None:
    space = make_campaign_space(seed=11)
    spec = _example_spec()
    candidate = space.sample_many(1)[0]
    applied = space.apply(spec, candidate)
    controller_params = applied.controller_params

    assert controller_params["trigger_radius_m"] == candidate["trigger_radius_m"]
    assert controller_params["speed"] == candidate["subject_speed_mps"]
    assert controller_params["trigger_tick"] == candidate["trigger_tick"]
    assert controller_params[LATERAL_OFFSET_ANNOTATION] == candidate["staging_lateral_offset_m"]
    assert controller_params[LONGITUDINAL_OFFSET_ANNOTATION] == candidate["staging_longitudinal_offset_m"]

    anchor = controller_params["route_anchor_location"]
    spawn = controller_params["spawn_transform"]["location"]
    destination = controller_params["destination_location"]
    crossing = (spawn["x"] - destination["x"], spawn["y"] - destination["y"])
    along = (
        spawn["x"] + destination["x"] - 2.0 * anchor["x"],
        spawn["y"] + destination["y"] - 2.0 * anchor["y"],
    )
    assert math.hypot(*crossing) == pytest.approx(2.0 * candidate["staging_lateral_offset_m"], abs=1e-6)
    assert math.hypot(*along) == pytest.approx(2.0 * abs(candidate["staging_longitudinal_offset_m"]), abs=1e-6)
    assert crossing[0] * along[0] + crossing[1] * along[1] == pytest.approx(0.0, abs=1e-6)

    assert space.from_payload(applied.to_dict()).values == candidate.values


def test_read_from_legacy_spec_uses_geometry_and_defaults() -> None:
    space = make_campaign_space(seed=11)
    values = space.from_payload(_example_payload())
    assert values is not None
    assert values["trigger_radius_m"] == 8.0
    assert values["subject_speed_mps"] == 1.8
    assert values["staging_lateral_offset_m"] == pytest.approx(6.4, abs=0.05)
    assert values["staging_longitudinal_offset_m"] == pytest.approx(0.0, abs=1e-6)
    assert values["trigger_tick"] == space.dimension("trigger_tick").default


def test_lateral_offset_application_preserves_crossing_axis() -> None:
    space = make_campaign_space(seed=3)
    payload = _example_payload()
    original = payload["controller_params"]
    anchor = original["route_anchor_location"]
    original_axis = (
        original["spawn_transform"]["location"]["x"] - anchor["x"],
        original["spawn_transform"]["location"]["y"] - anchor["y"],
    )
    candidate = space.clamp(
        {
            "trigger_radius_m": 20.0,
            "subject_speed_mps": 1.8,
            "staging_lateral_offset_m": 12.0,
            "staging_longitudinal_offset_m": 0.0,
            "trigger_tick": 100,
        }
    )
    applied = space.apply_to_payload(payload, candidate)
    params = applied["controller_params"]
    axis = (
        params["spawn_transform"]["location"]["x"] - anchor["x"],
        params["spawn_transform"]["location"]["y"] - anchor["y"],
    )
    cross = original_axis[0] * axis[1] - original_axis[1] * axis[0]
    assert cross == pytest.approx(0.0, abs=1e-6)


def test_longitudinal_offset_slides_crossing_along_route_heading() -> None:
    space = make_campaign_space(seed=3)
    payload = _example_payload()
    original = payload["controller_params"]
    anchor = original["route_anchor_location"]
    original_axis = (
        original["spawn_transform"]["location"]["x"] - anchor["x"],
        original["spawn_transform"]["location"]["y"] - anchor["y"],
    )
    yaw = math.radians(original["spawn_transform"]["rotation"]["yaw"])
    heading = (math.cos(yaw), math.sin(yaw))
    base = {
        "trigger_radius_m": 20.0,
        "subject_speed_mps": 1.8,
        "staging_lateral_offset_m": 12.0,
        "staging_longitudinal_offset_m": 8.0,
        "trigger_tick": 100,
    }

    def along_vector(longitudinal: float) -> tuple[float, float]:
        candidate = space.clamp({**base, "staging_longitudinal_offset_m": longitudinal})
        params = space.apply_to_payload(payload, candidate)["controller_params"]
        spawn = params["spawn_transform"]["location"]
        destination = params["destination_location"]
        crossing = (spawn["x"] - destination["x"], spawn["y"] - destination["y"])
        assert crossing[0] * original_axis[1] - crossing[1] * original_axis[0] == pytest.approx(0.0, abs=1e-6)
        return (
            spawn["x"] + destination["x"] - 2.0 * anchor["x"],
            spawn["y"] + destination["y"] - 2.0 * anchor["y"],
        )

    ahead = along_vector(8.0)
    behind = along_vector(-6.0)
    assert math.hypot(*ahead) == pytest.approx(2.0 * 8.0, abs=1e-6)
    assert math.hypot(*behind) == pytest.approx(2.0 * 6.0, abs=1e-6)
    assert ahead[0] * heading[0] + ahead[1] * heading[1] > 0.0
    assert behind[0] * heading[0] + behind[1] * heading[1] < 0.0


def test_validation_rejects_bad_candidates() -> None:
    space = make_campaign_space(seed=5)
    good = space.sample_many(1)[0].to_dict()
    with pytest.raises(SearchSpaceError):
        space.validate({**good, "trigger_radius_m": 1000.0})
    with pytest.raises(SearchSpaceError):
        space.validate({**good, "staging_longitudinal_offset_m": 100.0})
    with pytest.raises(SearchSpaceError):
        space.validate({**good, "not_a_dimension": 1.0})
    missing = dict(good)
    missing.pop("trigger_tick")
    with pytest.raises(SearchSpaceError):
        space.validate(missing)
    with pytest.raises(SearchSpaceError):
        space.validate({**good, "trigger_radius_m": float("nan")})

    clamped = space.clamp({**good, "trigger_radius_m": 1000.0, "trigger_tick": -50})
    space.validate(clamped.values)
    assert clamped["trigger_radius_m"] == 35.0
    assert clamped["trigger_tick"] == 10


def test_legacy_space_reproduces_pilot_candidates_and_writes() -> None:
    space = make_legacy_space(seed=13)
    assert space.names() == ("trigger_radius_m",)
    reference_rng = random.Random(13)
    for _ in range(100):
        candidate = space.sample()
        expected = round(reference_rng.uniform(5.0, 35.0), 2)
        assert candidate.values["trigger_radius_m"] == expected

    spec = _example_spec()
    applied = space.apply(spec, space.sample())
    original = spec.to_dict()
    original_params = dict(original["controller_params"])
    applied_params = dict(applied.controller_params)
    changed = {key for key in original_params if original_params[key] != applied_params.get(key)}
    assert changed == {"trigger_radius_m"}


def test_policy_search_builds_both_spaces_from_flags() -> None:
    module = _load_policy_search()
    campaign_args = argparse.Namespace(
        search_space="campaign",
        mutation_path="controller_params.trigger_radius_m",
        mutation_min=5.0,
        mutation_max=35.0,
        mutation_sigma=4.0,
        mutation_decimals=2,
        seed=13,
    )
    legacy_args = argparse.Namespace(**{**vars(campaign_args), "search_space": "legacy"})
    campaign = module._build_search_space(campaign_args)
    legacy = module._build_search_space(legacy_args)
    assert campaign.names() == CAMPAIGN_DIMENSION_NAMES
    assert legacy.names() == ("trigger_radius_m",)
    assert legacy.sample().values["trigger_radius_m"] == round(random.Random(13).uniform(5.0, 35.0), 2)


def test_policy_search_row_resume_reads_old_and_new_rows() -> None:
    module = _load_policy_search()
    campaign = make_campaign_space(seed=13)
    legacy = make_legacy_space(seed=13)
    new_candidate = campaign.sample_many(1)[0]
    new_row = {"candidate": new_candidate.to_dict(), "space": campaign.name}
    old_row = {"radius": 12.5}
    assert module._values_from_row(campaign, new_row) == new_candidate
    campaign_old = module._values_from_row(campaign, old_row)
    assert campaign_old is not None
    assert campaign_old["trigger_radius_m"] == 12.5
    assert campaign_old["subject_speed_mps"] == campaign.dimension("subject_speed_mps").default
    assert campaign_old["staging_longitudinal_offset_m"] == campaign.dimension(
        "staging_longitudinal_offset_m"
    ).default
    old_candidate = module._values_from_row(legacy, old_row)
    assert old_candidate is not None
    assert old_candidate.values["trigger_radius_m"] == 12.5


def test_all_four_policies_share_one_space_object() -> None:
    module = _load_policy_search()
    assert set(module.POLICIES) == {"random", "lsa", "kmnc", "semantic", "critonly"}
    space = make_campaign_space(name="campaign-shared", seed=23)
    states = [module.PolicyState(policy, space) for policy in module.POLICIES]
    assert all(state.space is space for state in states)
    assert {state.space.signature() for state in states} == {space.signature()}
    assert {state.space.dimensions() for state in states} == {space.dimensions()}

    for state in states:
        rng = random.Random(13)
        candidate = state.next_candidate(rng, set())
        space.validate(candidate.values)
        state.observe(candidate.to_dict(), 1.0)
        followed_up = state.next_candidate(rng, set())
        space.validate(followed_up.values)


def test_optimiser_initial_population_uses_campaign_space() -> None:
    space = make_campaign_space(seed=29)
    base_payload = _example_payload()
    base_candidate = space.from_payload(base_payload, fill_missing=True)
    assert base_candidate is not None

    plans = optimiser._build_initial_population(
        space=space,
        rng=random.Random(13),
        base_candidate=base_candidate,
        include_base=True,
        population_size=8,
    )
    assert plans[0].source == "base"
    assert plans[0].values == space.clamp(base_candidate.values).to_dict()
    assert len({space.key(plan.values) for plan in plans}) == len(plans)
    for plan in plans:
        space.validate(plan.values)

    elites = [
        {"candidate_id": plan.candidate_id, "candidate": plan.values}
        for plan in plans[:2]
    ]
    generation = optimiser._build_next_generation(
        space=space,
        rng=random.Random(14),
        generation=1,
        population_size=6,
        elites=elites,
    )
    assert generation[0].source == "elite-copy"
    assert len({space.key(plan.values) for plan in generation}) == len(generation)
    for plan in generation:
        space.validate(plan.values)


class _FakeLocation:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def distance(self, other: "_FakeLocation") -> float:
        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))


class _FakeCarla:
    Location = _FakeLocation


class _FakeEgo:
    def __init__(self, location: _FakeLocation) -> None:
        self._location = location

    def get_location(self) -> _FakeLocation:
        return self._location


class _FakeActor:
    def __init__(self) -> None:
        self.controls: list[object] = []

    def apply_control(self, control: object) -> None:
        self.controls.append(control)


class _FakeWalkerControl:
    def __init__(self) -> None:
        self.speed = 0.0


def _tick_context(ego_x: float) -> dict:
    return {
        "ego_vehicle": _FakeEgo(_FakeLocation(ego_x, 0.0)),
        "carla": _FakeCarla,
        "scenario_notes": [],
        "threshold_adversary_active": False,
        "threshold_adversary_triggered": False,
    }


def test_trigger_tick_releases_adversary_without_proximity() -> None:
    controller = ThresholdCrossingAdversaryController(
        {
            "trigger_location": {"x": 100.0, "y": 0.0, "z": 0.0},
            "trigger_radius_m": 5.0,
            "trigger_tick": 30,
            "speed": 2.5,
            "adversary_kind": "walker",
        }
    )
    controller._actor = _FakeActor()
    controller._walker_control = _FakeWalkerControl()
    context = _tick_context(ego_x=0.0)

    for tick_index in range(29):
        controller.on_tick(tick_index, context)
    assert context["threshold_adversary_triggered"] is False
    assert len(controller._actor.controls) == 0

    controller.on_tick(29, context)
    assert context["threshold_adversary_triggered"] is True
    assert controller._walker_control.speed == 2.5
    assert len(controller._actor.controls) == 1
    assert "tick deadline 30" in context["scenario_notes"][-1]


def test_distance_trigger_still_fires_first() -> None:
    controller = ThresholdCrossingAdversaryController(
        {
            "trigger_location": {"x": 3.0, "y": 0.0, "z": 0.0},
            "trigger_radius_m": 5.0,
            "trigger_tick": 400,
            "speed": 1.8,
            "adversary_kind": "walker",
        }
    )
    controller._actor = _FakeActor()
    controller._walker_control = _FakeWalkerControl()
    context = _tick_context(ego_x=0.0)
    controller.on_tick(0, context)
    assert context["threshold_adversary_triggered"] is True
    assert "distance 3.00 m" in context["scenario_notes"][-1]


def test_declarative_dimension_rejects_inconsistent_bounds() -> None:
    with pytest.raises(SearchSpaceError):
        SearchDimension(name="bad", spec_path="controller_params.x", low=10.0, high=5.0)
    with pytest.raises(SearchSpaceError):
        SearchDimension(name="bad", spec_path="controller_params.x", low=0.0, high=1.0, kind="categorical")
    with pytest.raises(SearchSpaceError):
        SearchSpace((), name="empty")
    space = SearchSpace((SearchDimension(name="x", spec_path="controller_params.x", low=0.0, high=1.0),))
    with pytest.raises(SearchSpaceError):
        space.dimension("missing")
