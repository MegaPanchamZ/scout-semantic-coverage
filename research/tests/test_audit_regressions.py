"""Regressions for the implementation audit, without CARLA or model weights."""
from types import SimpleNamespace
import random
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from research.harness.autovla_bridge import AutoVlaAdapter
from research.harness.hazard_search import ObligationScheduler
from research.harness.coverage_engine import (
    DerivedTrace, DerivationConfig, Fact, load_oracle,
    _hazard_mapping_status, _hazard_witness,
)

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "audit_policy", ROOT / "research/experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py"
)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def test_scheduler_advances_after_credit_stall_and_exhaustion():
    lead = "hazard(other: ahead_or_waiting)"
    crossing = "hazard(pedestrian_in_path)"
    universe = {lead, crossing}
    scheduler = ObligationScheduler(universe.copy())
    assert policy._pick_target(scheduler, set(), universe)[0] == lead
    assert policy._pick_target(scheduler, {lead}, universe)[0] == crossing
    assert policy._pick_target(scheduler, universe, universe)[0] is None
    scheduler = ObligationScheduler(universe.copy())
    policy._pick_target(scheduler, set(), universe)
    assert policy._pick_target(scheduler, set(), universe, {lead: 12})[0] == crossing


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setitem(sys.modules, "carla", SimpleNamespace(VehicleControl=lambda **kw: SimpleNamespace(**kw)))
    agent = AutoVlaAdapter.__new__(AutoVlaAdapter)
    agent._ego = SimpleNamespace(
        get_velocity=lambda: SimpleNamespace(x=0., y=0., z=0.),
        get_transform=lambda: SimpleNamespace(location=SimpleNamespace(x=0., y=0.), rotation=SimpleNamespace(yaw=0.)),
    )
    agent._target_speed = 5.5
    agent._tick = 1
    agent._last_inference_tick = 1
    agent._trajectory_origin = (0., 0., 0.)
    agent._fixed_delta_seconds = .1
    return agent


def test_autovla_stop_prediction_holds_brake(adapter):
    control = adapter._control_from_poses(np.zeros((10, 3)))
    assert control.throttle == 0.
    assert control.brake > 0.


def test_autovla_target_advances_in_world_frame(adapter):
    poses = np.array([[.5 * i, 0., 0.] for i in range(1, 11)])
    adapter._tick = 11
    adapter._ego.get_transform = lambda: SimpleNamespace(location=SimpleNamespace(x=1., y=0.), rotation=SimpleNamespace(yaw=90.))
    control = adapter._control_from_poses(poses)
    assert control.steer < 0., "old forward target is now left of the rotated ego"


def test_cut_in_requires_oncoming_before_same_lane():
    oracle = load_oracle(ROOT / "research/experiments/EXP-018-nuscenes-oracle-inventory/artifacts/oracle_inventory_v1.0-trainval.json")
    obligation = next(o for o in oracle.obligations if o.signature == "hazard(oncoming_cut_in)")
    derived = DerivedTrace(ticks=[])
    derived.actor_facts = {"vehicle#1": {
        "same_lane": [Fact("same_lane", "vehicle", "ego", 0, "vehicle#1", "fixture")],
        "oncoming": [Fact("oncoming", "vehicle", None, 5, "vehicle#1", "fixture")],
    }}
    _, mapping, _ = _hazard_mapping_status(obligation)
    assert _hazard_witness(derived, obligation, mapping, DerivationConfig()) is None
    derived.actor_facts["vehicle#1"]["same_lane"] = [Fact("same_lane", "vehicle", "ego", 15, "vehicle#1", "fixture")]
    assert _hazard_witness(derived, obligation, mapping, DerivationConfig()) is not None


def test_semantic_elite_is_rescored_against_remaining_gaps():
    state = policy.PolicyState("semantic", policy.make_legacy_space())
    old = {f"old-{i}" for i in range(11)}
    new = {f"new-{i}" for i in range(4)}
    state.observe({"trigger_radius_m": 8.}, 11., 11., run_obligations=old, covered_before=set())
    state.observe({"trigger_radius_m": 20.}, 4., 4., run_obligations=new, covered_before=old)
    assert state.best_values == {"trigger_radius_m": 20.}


def test_hazard_elite_is_reset_when_target_changes():
    state = policy._HazardPolicyState("semantic")
    state.next_candidate(random.Random(1), "pedestrian_crossing", "first")
    state.observe("pedestrian_crossing", {"trigger_radius_m": 8.}, 1., 2.)
    state.next_candidate(random.Random(2), "pedestrian_crossing", "second")
    assert "pedestrian_crossing" not in state.elites


def test_single_stationary_tick_does_not_count_as_waiting():
    oracle = load_oracle(ROOT / "research/experiments/EXP-018-nuscenes-oracle-inventory/artifacts/oracle_inventory_v1.0-trainval.json")
    obligation = next(o for o in oracle.obligations if o.signature == "hazard(other: ahead_or_waiting)")
    derived = DerivedTrace(ticks=[])
    derived.actor_facts = {"vehicle#1": {
        "in_front_of": [Fact("in_front_of", "vehicle", "ego", 5, "vehicle#1", "fixture")],
        "stationary": [Fact("stationary", "vehicle", None, 5, "vehicle#1", "fixture")],
    }}
    _, mapping, _ = _hazard_mapping_status(obligation)
    assert _hazard_witness(derived, obligation, mapping, DerivationConfig()) is None
    derived.actor_facts["vehicle#1"]["stationary"] = [Fact("stationary", "vehicle", None, t, "vehicle#1", "fixture") for t in range(6)]
    assert _hazard_witness(derived, obligation, mapping, DerivationConfig()) is not None


def test_autovla_slow_prediction_brakes_a_faster_ego(adapter):
    adapter._ego.get_velocity = lambda: SimpleNamespace(x=3., y=0., z=0.)
    poses = np.array([[.5 * i, 0., 0.] for i in range(1, 11)])
    control = adapter._control_from_poses(poses)
    assert control.throttle == 0.
    assert control.brake > 0.
    adapter._tick = 100
    assert adapter._control_from_poses(poses).brake > 0.


def test_autovla_uses_simulator_acceleration(adapter):
    adapter._route = []
    adapter._tick = 20
    adapter._last_inference_tick = 0
    adapter._last_poses = None
    adapter._last_speed = 0.
    adapter._frames = {name: ["frame.png"] * 4 for name in ("front_camera", "front_left_camera", "front_right_camera")}
    adapter._ego.get_acceleration = lambda: SimpleNamespace(x=2.5, y=0.)
    adapter._ego.get_velocity = lambda: SimpleNamespace(x=5., y=0., z=0.)
    seen = []
    adapter._backend = SimpleNamespace(
        name="fake",
        plan=lambda features: (seen.append(features) or np.zeros((10, 3)), ""),
    )
    adapter.run_step()
    assert seen[0]["vehicle_acceleration"] == [2.5, 0.]


def test_carla_pythonapi_candidates_prefers_carla_root(tmp_path, monkeypatch):
    from research.harness.leaderboard_bridge import _carla_pythonapi_candidates

    monkeypatch.setenv("CARLA_ROOT", str(tmp_path / "carla-0.9.16"))
    candidates = _carla_pythonapi_candidates()
    assert candidates[0] == tmp_path / "carla-0.9.16" / "PythonAPI" / "carla"


def test_compat_route_planner_discovers_carla_root(tmp_path, monkeypatch):
    module_spec = importlib.util.spec_from_file_location(
        "compat_route_planner",
        ROOT / "research/harness/compat/agents/navigation/global_route_planner.py",
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    agents_root = tmp_path / "CARLA" / "PythonAPI" / "carla"
    module_file = agents_root / "agents" / "navigation" / "global_route_planner.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("# fake modern planner\n")
    monkeypatch.setenv("CARLA_ROOT", str(tmp_path / "CARLA"))
    assert module._find_modern_global_route_planner() == module_file


def test_coverage_observer_survives_missing_profile(tmp_path):
    pytest.importorskip("torch")
    import torch

    from research.harness.observers.coverage import CoverageObserver, CoverageObserverConfig

    class _Agent:
        def __init__(self):
            self.torch_model = torch.nn.Linear(3, 3)

    observer = CoverageObserver(CoverageObserverConfig(profile_path=tmp_path / "missing.joblib"))
    context = {"agent": _Agent()}
    observer.on_run_start(SimpleNamespace(), context)
    assert observer._profile is None
    assert observer._status == "profile-load-failed"

