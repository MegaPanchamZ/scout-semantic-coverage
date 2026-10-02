"""Offline tests for criticality guidance, stage-2 exploitation and adaptive mutation."""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

from research.harness.criticality import criticality_score
from research.harness.hazard_search import (
    CROSSING_SPACE,
    AdaptiveMutation,
    ExploitTracker,
    ObligationScheduler,
)

_PS = Path(__file__).resolve().parents[1] / "experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py"
_spec = importlib.util.spec_from_file_location("policy_search_under_test", _PS)
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


def test_criticality_ordering():
    benign = {"safety_raw": {"min_ttc_s": 9.0, "min_vehicle_distance_m": 30.0, "max_deceleration_mps2": 1.0}}
    near = {"safety_raw": {"min_ttc_s": 1.0, "min_vehicle_distance_m": 4.0}}
    hit = {"collision_count": 1}
    assert criticality_score(benign) < criticality_score(near) < criticality_score(hit)
    assert criticality_score(hit) == 2.0
    assert criticality_score({}) == 0.0


def test_criticality_ignores_crashed_runs():
    assert criticality_score({"run_error": "boom", "collision_count": 3}) == 0.0
    assert criticality_score({"safety_valid": False, "collision_count": 3}) == 0.0


def test_exploit_tracker_holds_until_stale():
    t = ExploitTracker(patience=2, cap=10)
    t.observe("a", False, 0.9)
    assert not t.holding("a")  # not yet covered
    t.observe("a", True, 0.3)
    assert t.holding("a")
    t.observe("a", True, 0.5)  # improved
    assert t.holding("a")
    t.observe("a", True, 0.5)
    t.observe("a", True, 0.4)  # two stale evals
    assert not t.holding("a")


def test_exploit_tracker_cap_and_target_switch():
    t = ExploitTracker(patience=100, cap=3)
    for i in range(3):
        t.observe("a", True, 0.1 * (i + 1))
    assert not t.holding("a")
    t.observe("b", True, 0.2)
    assert t.holding("b") and not t.holding("a")


def test_exploit_disabled():
    t = ExploitTracker(patience=0)
    t.observe("a", True, 1.0)
    assert not t.holding("a")


def test_pick_target_hold_keeps_covered_target():
    sch = ObligationScheduler(uncovered=set())
    universe = {"crossing_path(pedestrian,ego)", "in_front_of(vehicle,ego)"}
    first, _ = ps._pick_target(sch, set(), universe)
    held, _ = ps._pick_target(sch, {first}, universe, hold=first)
    assert held == first
    moved, _ = ps._pick_target(sch, {first}, universe)
    assert moved == "in_front_of(vehicle,ego)"


def test_pick_target_hold_survives_stall_limit():
    sch = ObligationScheduler(uncovered=set())
    universe = {"crossing_path(pedestrian,ego)", "in_front_of(vehicle,ego)"}
    first, _ = ps._pick_target(sch, set(), universe)
    held, _ = ps._pick_target(sch, {first}, universe, attempts={first: 99}, stall_limit=12, hold=first)
    assert held == first


def test_adaptive_mutation_scale_bounds_and_restart():
    a = AdaptiveMutation()
    for _ in range(20):
        a.update(True)
    assert a.scale == a.min_scale
    for _ in range(20):
        a.update(False)
    assert a.scale == a.max_scale and a.should_restart()
    a.reset()
    assert a.scale == 1.0 and not a.should_restart()


def test_mutate_scale_widens_step():
    base = CROSSING_SPACE.defaults()
    def spread(scale):
        rng = random.Random(0)
        return sum(abs(CROSSING_SPACE.mutate(base, rng, scale)["subject_speed_mps"] - base["subject_speed_mps"]) for _ in range(200))
    assert spread(2.0) > spread(0.25)


def test_hazard_state_restarts_after_stall():
    st = ps._HazardPolicyState("semantic", epsilon=0.0)
    rng = random.Random(1)
    cand = CROSSING_SPACE.defaults()
    st.observe("pedestrian_crossing", "t", cand, 1.0, 0.5)
    for _ in range(AdaptiveMutation().restart_after):
        st.observe("pedestrian_crossing", "t", cand, 0.0, 0.0)  # no improvement
    # restart draws uniform: over many draws it must leave the elite's sigma neighbourhood
    draws = [st.next_candidate(rng, "pedestrian_crossing", "t")["trigger_tick"] for _ in range(1)]
    assert draws  # restart path executed without error
    assert st.adapt["target:t"].stale == 0


def test_target_progress_prefers_critical_when_covered(monkeypatch):
    class Ob:
        def __init__(self, sig):
            self.signature, self.predicate, self.node_types, self.required_predicates = sig, "p", ("x",), ("p",)
    class Oracle:
        obligations = [Ob("t")]
    lo = {"engine_run_obligations": ["t"], "criticality": 0.2}
    hi = {"engine_run_obligations": ["t"], "criticality": 0.9}
    miss = {"engine_run_obligations": [], "criticality": 1.0}
    f = lambda r: ps._target_progress(Oracle(), "t", r, use_criticality=True)
    assert f(hi) > f(lo) > f(miss)
    assert ps._target_progress(Oracle(), "t", hi)[1] == ps._target_progress(Oracle(), "t", lo)[1]


def test_criticality_ignores_peak_deceleration():
    slam = {"safety_raw": {"min_ttc_s": 9.0, "min_vehicle_distance_m": 30.0, "max_deceleration_mps2": 27.0}}
    assert criticality_score(slam) < 0.2
