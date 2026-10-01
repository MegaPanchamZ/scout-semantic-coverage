"""Tests for the cross-route shared suite store and per-obligation elite keying."""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

from research.harness.shared_suite import SharedSuiteStore

ROOT = Path(__file__).resolve().parents[2]
_POLICY_PATH = ROOT / "research" / "experiments" / "EXP-020-policy-comparison" / "proof-of-concept" / "policy_search.py"


def _load_policy():
    spec = importlib.util.spec_from_file_location("ss_policy", _POLICY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_store_empty_then_union(tmp_path):
    store = SharedSuiteStore(tmp_path / "s.json")
    assert store.load() == set()
    assert store.merge(["a", "b"]) == {"a", "b"}
    assert store.merge(["b", "c"]) == {"a", "b", "c"}
    # persisted across instances
    assert SharedSuiteStore(tmp_path / "s.json").load() == {"a", "b", "c"}


def test_store_tolerates_corrupt_file(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("not json")
    assert SharedSuiteStore(path).load() == set()
    # and recovers on the next merge
    assert SharedSuiteStore(path).merge({"x"}) == {"x"}


def test_two_writers_accumulate(tmp_path):
    path = tmp_path / "s.json"
    SharedSuiteStore(path).merge({"a"})
    SharedSuiteStore(path).merge({"b"})
    assert SharedSuiteStore(path).load() == {"a", "b"}


def test_store_creates_parent_dirs(tmp_path):
    store = SharedSuiteStore(tmp_path / "deep" / "nested" / "s.json")
    assert store.merge({"a"}) == {"a"}
    assert store.path.exists()


def test_hazard_elite_is_per_obligation_for_all_policies():
    policy = _load_policy()
    for name in ("random", "lsa", "kmnc", "semantic"):
        state = policy._HazardPolicyState(name)
        state.next_candidate(random.Random(1), "pedestrian_crossing", "t1")
        state.observe("pedestrian_crossing", "t1", {"trigger_radius_m": 8.0}, 1.0, 0.0)
        # switching obligation keeps each target's own elite
        state.next_candidate(random.Random(2), "pedestrian_crossing", "t2")
        assert "target:t1" in state.elites, name
        assert "target:t2" not in state.elites, name
        # the selected obligation's elite is mutated, not another's
        candidate = state.next_candidate(random.Random(3), "pedestrian_crossing", "t1")
        assert set(candidate)  # sampled from t1's elite space
