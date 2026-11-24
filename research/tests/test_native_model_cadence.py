from types import SimpleNamespace

import pytest
from research.harness.config import AppConfig
from research.harness.models import ScenarioSpec
from research.harness.oracles import CollisionOracle
from research.harness.outcome_io import iter_evidence, load_capture_manifest
from research.harness.outcome_models import FrameEvidence
from research.harness.runner import HarnessRunner, _control_cadence
from research.tests.test_outcome_capture import _Control, _FakeWorld, _Transform, _Vec


@pytest.mark.parametrize('agent',['lav_lav','lav_fast','simlingo_simlingo'])
def test_documented_twenty_hz_agents_keep_native_control_cadence(agent):
    c=AppConfig();c.agent.kind='pcla';c.agent.pcla_agent_name=agent
    assert _control_cadence(c)==(.05,2)


@pytest.mark.parametrize('module', ['submission_agent', 'transfuser_agent'])
def test_transfuser_releases_keep_native_control_cadence(module):
    c=AppConfig();c.agent.kind='leaderboard-module';c.agent.module_name=module
    assert _control_cadence(c)==(.05,2)


def test_physics_step_capture_observes_every_substep_including_truncated_collision_loop(tmp_path, monkeypatch):
    world = _FakeWorld(dt=0.05)

    class _NativeAgent:
        def __init__(self):
            self.step_calls = 0

        def set_destination(self, loc):
            pass

        def get_route_preview(self):
            return [{"x": 0.0, "y": 0.0}, {"x": 50.0, "y": 0.0}]

        def run_step(self):
            self.step_calls += 1
            # Trigger collision on the first substep of logical tick 1 (3rd physics step)
            if self.step_calls == 3:
                world.collision_sensor.fire(
                    SimpleNamespace(
                        frame=world._frame + 1,
                        timestamp=round(world._time + 0.05, 8),
                        normal_impulse=_Vec(10, 0, 0),
                        other_actor=SimpleNamespace(id=77, type_id="vehicle.other"),
                    )
                )
            return _Control(throttle=0.5)

        def done(self):
            return False

    agent = _NativeAgent()
    monkeypatch.setattr("research.harness.runner.connect_client", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr("research.harness.runner.load_world", lambda c, t: world)
    monkeypatch.setattr("research.harness.runner.apply_world_settings", lambda w, s, d: {})
    monkeypatch.setattr("research.harness.runner.restore_world_settings", lambda w, s: None)
    monkeypatch.setattr("research.harness.runner.resolve_weather", lambda w, p: None)
    monkeypatch.setattr("research.harness.runner.import_carla", lambda: SimpleNamespace(VehicleControl=_Control))
    monkeypatch.setattr("research.harness.runner.make_agent", lambda *a, **k: agent)
    monkeypatch.setattr(CollisionOracle, "_identity_transform", staticmethod(lambda w: _Transform()))
    monkeypatch.setattr("torch.manual_seed", lambda seed: None)

    cfg = AppConfig()
    cfg.agent.kind = "pcla"
    cfg.agent.pcla_agent_name = "lav_lav"  # 20 Hz -> 2 substeps per 0.1s logical tick
    cfg.run.warmup_ticks = 0
    cfg.harness.output_dir = tmp_path / "runs"
    cfg.outcomes.enabled = True
    cfg.outcomes.evidence_output_dir = tmp_path / "evidence"
    scenario = ScenarioSpec("cadence-test", "Town01", "ClearNoon", 0, 1, "cadence", max_ticks=5)

    res = HarnessRunner(cfg).run(scenario)
    assert res.terminated_by_collision is True
    assert res.termination_reason == "collision_stop"
    # Logical tick 0 executed 2 substeps; tick 1 stopped after substep 0 -> 3 physics steps total.
    assert agent.step_calls == 3
    m, trace_path = load_capture_manifest(res.capture_manifest_path)
    frames = [r for r in iter_evidence(trace_path) if isinstance(r, FrameEvidence) and r.phase == "evaluation"]
    assert len(frames) == 3
    assert [(f.logical_tick, f.substep) for f in frames] == [(0, 0), (0, 1), (1, 0)]
    assert res.metadata["outcome_v2"]["categories"]["safety"]["collision"]["status"] == "present"
