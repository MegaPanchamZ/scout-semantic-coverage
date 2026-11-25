"""Tests for live evidence capture, directed stop-line geometry, sensor draining and partial failure envelopes."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from research.harness.config import AppConfig
from research.harness.models import ScenarioSpec
from research.harness.oracles import CollisionOracle, LaneInvasionOracle, SafetyOracle
from research.harness.outcome_io import EvidenceReadError, iter_evidence, load_capture_manifest
from research.harness.outcome_models import SensorEvidence
from research.harness.road_context import crosses_directed_line
from research.harness.runner import HarnessRunner, _LiveOutcomeCapture
from research.harness.safety_outcomes import classify_run


class _Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5


class _Rot:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = float(pitch), float(yaw), float(roll)


class _Transform:
    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        self.location = _Vec(x, y, z)
        self.rotation = _Rot(0.0, yaw, 0.0)


class _BBox:
    def __init__(self, ex=2.0, ey=1.0, ez=0.75):
        self.extent = _Vec(ex, ey, ez)
        self.location = _Vec(0.0, 0.0, 0.0)
        self.rotation = _Rot()


class _Control:
    def __init__(self, throttle=0.0, brake=0.0, steer=0.0, reverse=False):
        self.throttle, self.brake, self.steer, self.reverse = throttle, brake, steer, reverse
        self.hand_brake = False
        self.manual_gear_shift = False
        self.gear = 1


class _FakeActor:
    def __init__(self, aid, type_id, x=0.0, y=0.0, vx=10.0, vy=0.0, ax=0.0):
        self.id = aid
        self.type_id = type_id
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.ax = float(ax)
        self.bounding_box = _BBox()
        self.control = _Control(throttle=0.4)
        self.destroyed = False

    def get_transform(self):
        return _Transform(self.x, self.y, 0.0, 0.0)

    def get_location(self):
        return _Vec(self.x, self.y, 0.0)

    def get_velocity(self):
        return _Vec(self.vx, self.vy, 0.0)

    def get_acceleration(self):
        return _Vec(self.ax, 0.0, 0.0)

    def get_control(self):
        return self.control

    def apply_control(self, control):
        self.control = control

    def get_speed_limit(self):
        return 50.0

    def is_at_traffic_light(self):
        return False

    def get_traffic_light(self):
        return None

    def destroy(self):
        self.destroyed = True


class _FakeSensor:
    def __init__(self):
        self.cb = None
        self.stopped = False
        self.destroyed = False

    def listen(self, cb):
        self.cb = cb

    def fire(self, ev):
        if self.cb is not None:
            self.cb(ev)

    def stop(self):
        self.stopped = True

    def destroy(self):
        self.destroyed = True


class _FakeWaypoint:
    def __init__(self, x=0.0, y=0.0):
        self.road_id = 1
        self.lane_id = -1
        self.lane_type = "Driving"
        self.is_junction = False
        self.lane_change = "Both"
        self.transform = _Transform(x, y, 0.0, 0.0)
        self.lane_width = 3.5


class _FakeMap:
    def get_spawn_points(self):
        return [_Transform(0.0, 0.0), _Transform(50.0, 0.0)]

    def get_waypoint(self, loc, project_to_road=True):
        return _FakeWaypoint(loc.x, loc.y)


class _FakeWorld:
    def __init__(self, actors=None, *, fail_actors_at=None, dt=0.1):
        self._actors = list(actors or [])
        self._frame = 0
        self._time = 0.0
        self._dt = dt
        self._fail_actors_at = fail_actors_at
        self.collision_sensor = _FakeSensor()
        self.lane_sensor = _FakeSensor()

    def get_map(self):
        return _FakeMap()

    def get_blueprint_library(self):
        return SimpleNamespace(
            find=lambda name: name,
            filter=lambda pattern: [pattern],
        )

    def spawn_actor(self, bp, transform, attach_to=None):
        if "collision" in str(bp):
            return self.collision_sensor
        if "lane_invasion" in str(bp):
            return self.lane_sensor
        raise RuntimeError(f"Unknown sensor {bp}")

    def try_spawn_actor(self, bp, transform):
        ego = _FakeActor(1, "vehicle.tesla.model3", x=transform.location.x, y=transform.location.y, vx=5.0)
        self._actors.append(ego)
        return ego

    def get_actors(self):
        if self._fail_actors_at is not None and self._frame >= self._fail_actors_at:
            raise RuntimeError("world actor scan lost")
        return list(self._actors)

    def tick(self):
        self._frame += 1
        self._time = round(self._time + self._dt, 8)
        for a in self._actors:
            a.x += a.vx * self._dt
            a.y += a.vy * self._dt
        return self._frame

    def get_snapshot(self):
        return SimpleNamespace(frame=self._frame, timestamp=SimpleNamespace(elapsed_seconds=self._time))


def test_crosses_directed_line_rejects_parallel_and_unrelated_lanes():
    # Stop line across x = 10, y in [-2, 2], forward normal = (+1, 0)
    prev = [(7.0, -1.0), (9.5, -1.0), (9.5, 1.0), (7.0, 1.0)]
    crossed = [(8.0, -1.0), (10.5, -1.0), (10.5, 1.0), (8.0, 1.0)]
    assert crosses_directed_line(prev, crossed, (10.0, -2.0), (10.0, 2.0), (1.0, 0.0)) is True
    # Reversed normal rejects backward crossing
    assert crosses_directed_line(prev, crossed, (10.0, -2.0), (10.0, 2.0), (-1.0, 0.0)) is False
    # Unrelated lane at y = 10 rejected
    other_lane_prev = [(7.0, 9.0), (9.5, 9.0), (9.5, 11.0), (7.0, 11.0)]
    other_lane_curr = [(8.0, 9.0), (10.5, 9.0), (10.5, 11.0), (8.0, 11.0)]
    assert crosses_directed_line(other_lane_prev, other_lane_curr, (10.0, -2.0), (10.0, 2.0), (1.0, 0.0)) is False
    # Motion parallel to stop line without crossing rejected
    parallel_curr = [(7.0, 0.0), (9.5, 0.0), (9.5, 2.0), (7.0, 2.0)]
    assert crosses_directed_line(prev, parallel_curr, (10.0, -2.0), (10.0, 2.0), (1.0, 0.0)) is False


def test_collision_and_lane_oracles_drain_late_and_setup_callbacks(monkeypatch):
    monkeypatch.setattr(CollisionOracle, "_identity_transform", staticmethod(lambda world: _Transform()))
    world = _FakeWorld()
    ego = _FakeActor(1, "vehicle.tesla.model3")

    col = CollisionOracle()
    col.set_phase("warmup")
    col.attach(world, ego)
    world.collision_sensor.fire(
        SimpleNamespace(frame=2, timestamp=0.2, normal_impulse=_Vec(3, 4, 0), other_actor=SimpleNamespace(id=9, type_id="vehicle.other"))
    )
    col.set_phase("evaluation")
    # Late callback from frame 5 arriving without timestamp uses frame_times map
    world.collision_sensor.fire(
        SimpleNamespace(frame=5, timestamp=None, normal_impulse=_Vec(0, 6, 8), other_actor=SimpleNamespace(id=9, type_id="vehicle.other"))
    )

    lane = LaneInvasionOracle()
    lane.set_phase("evaluation")
    lane.attach(world, ego)
    world.lane_sensor.fire(
        SimpleNamespace(frame=6, timestamp=0.6, crossed_lane_markings=[SimpleNamespace(type="LaneMarkingType.Solid")])
    )

    col_events = col.drain_events("run-1", frame_times={5: 0.5})
    assert [e.phase for e in col_events] == ["warmup", "evaluation"]
    assert col_events[1].sim_time_s == 0.5
    assert col_events[1].impulse_ns == (0.0, 6.0, 8.0)

    lane_events = lane.drain_events("run-1")
    assert len(lane_events) == 1
    assert lane_events[0].lane_markings == ("Solid",)
    assert col.drain_events("run-1") == []  # drained

    col.destroy()
    lane.destroy()
    assert world.collision_sensor.stopped and world.collision_sensor.destroyed
    assert world.lane_sensor.stopped and world.lane_sensor.destroyed


def test_capture_handles_channel_errors_disabled_sensors_and_replay_equivalence(tmp_path, monkeypatch):
    monkeypatch.setattr(CollisionOracle, "_identity_transform", staticmethod(lambda world: _Transform()))
    world = _FakeWorld(fail_actors_at=8)
    ego = _FakeActor(1, "vehicle.tesla.model3", vx=5.0)
    world._actors.append(ego)

    cfg = AppConfig()
    cfg.outcomes.enabled = True
    cfg.outcomes.evidence_output_dir = tmp_path / "evidence"
    scenario = ScenarioSpec("cap-test", "Town01", "ClearNoon", 0, 1, "test", max_ticks=12)

    capture = _LiveOutcomeCapture(cfg, scenario, "run-cap", 0.1)
    capture.init_world_context(world, [{"x": 0.0, "y": 0.0}, {"x": 50.0, "y": 0.0}])
    safety = SafetyOracle(tick_seconds=0.1)
    col = CollisionOracle()
    lane = LaneInvasionOracle()
    # Leave collision and lane sensors unattached to test unavailable sensor handling
    for i in range(12):
        world.tick()
        capture.observe_physics_step(
            world, ego, safety, col, lane, phase="evaluation", logical_tick=i, substep=0, actor_roles={1: "ego"}
        )
    m_path, o_path, live_dict = capture.finalize(
        execution_status="completed",
        termination_reason="horizon_exhausted",
        reached_goal=False,
        collision_oracle=col,
        lane_oracle=lane,
    )
    assert m_path and o_path
    loaded_manifest, trace_path = load_capture_manifest(m_path)
    records = list(iter_evidence(trace_path))
    replayed = classify_run({"capture_manifest": json.loads(m_path.read_text())["capture_manifest"]}, evidence=records)
    assert replayed == live_dict
    # Unattached collision/lane sensors and failed actor channel must stay unknown (never absent)
    assert live_dict["categories"]["safety"]["collision"]["status"] == "unknown"
    assert live_dict["categories"]["behaviour"]["lane_crossing"]["status"] == "unknown"
    assert loaded_manifest.channels["actors"].status == "partial"


def test_live_runner_writes_partial_envelope_on_midrun_failure(tmp_path, monkeypatch):
    world = _FakeWorld()

    class _BoomAgent:
        def __init__(self):
            self.steps = 0

        def set_destination(self, loc):
            pass

        def get_route_preview(self):
            return [{"x": 0.0, "y": 0.0}, {"x": 50.0, "y": 0.0}]

        def run_step(self):
            self.steps += 1
            if self.steps == 4:
                raise RuntimeError("simulated agent crash")
            return _Control(throttle=0.5)

        def done(self):
            return False

    monkeypatch.setattr("research.harness.runner.connect_client", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr("research.harness.runner.load_world", lambda c, t: world)
    monkeypatch.setattr("research.harness.runner.apply_world_settings", lambda w, s, d: {})
    monkeypatch.setattr("research.harness.runner.restore_world_settings", lambda w, s: None)
    monkeypatch.setattr("research.harness.runner.resolve_weather", lambda w, p: None)
    monkeypatch.setattr("research.harness.runner.import_carla", lambda: SimpleNamespace(VehicleControl=_Control))
    monkeypatch.setattr("research.harness.runner.make_agent", lambda *a, **k: _BoomAgent())
    monkeypatch.setattr(CollisionOracle, "_identity_transform", staticmethod(lambda w: _Transform()))

    cfg = AppConfig()
    cfg.harness.output_dir = tmp_path / "runs"
    cfg.run.warmup_ticks = 1
    cfg.outcomes.enabled = True
    cfg.outcomes.evidence_output_dir = tmp_path / "evidence"
    scenario = ScenarioSpec("partial-boom", "Town01", "ClearNoon", 0, 1, "boom test", max_ticks=10)

    runner = HarnessRunner(cfg)
    with pytest.raises(RuntimeError, match="simulated agent crash"):
        runner.run(scenario)

    manifests = list((tmp_path / "evidence").glob("*.manifest.json"))
    assert len(manifests) == 1
    m, trace_path = load_capture_manifest(manifests[0])
    assert m.execution_status == "partial"
    assert m.termination_reason == "agent_error"
    frames = [r for r in iter_evidence(trace_path) if not isinstance(r, SensorEvidence)]
    # 1 warmup + 3 evaluation frames before step 4 crashed
    assert [f.phase for f in frames] == ["warmup", "evaluation", "evaluation", "evaluation"]
    run_jsons = list((tmp_path / "runs").glob("*.json"))
    assert len(run_jsons) == 1
    saved = json.loads(run_jsons[0].read_text())
    assert saved["execution_status"] == "partial"
    assert saved["succeeded"] is False


def test_warmup_callback_arriving_during_evaluation_keeps_warmup_phase_and_does_not_stop_eval(monkeypatch):
    world = _FakeWorld()
    ego = _FakeActor(1, "vehicle.tesla.model3")
    monkeypatch.setattr(CollisionOracle, "_identity_transform", staticmethod(lambda w: _Transform()))
    col = CollisionOracle()
    sensor = col.attach(world, ego)
    col.set_phase("warmup")
    # Evaluation begins at frame 2
    col.set_phase("evaluation")
    frame_phases = {1: "warmup", 2: "evaluation"}
    frame_times = {1: 0.1, 2: 0.2}
    # Late callback from warmup frame 1 arrives after phase switched to evaluation
    late_ev = SimpleNamespace(
        frame=1,
        timestamp=0.1,
        other_actor=SimpleNamespace(id=77, type_id="static.wall"),
        normal_impulse=_Vec(50.0, 0.0, 0.0),
    )
    sensor.fire(late_ev)
    assert col.has_evaluation_collision(frame_phases=frame_phases) is False
    drained = col.drain_events("r1", frame_times=frame_times, frame_phases=frame_phases)
    assert len(drained) == 1 and drained[0].phase == "warmup"


def test_crosses_directed_line_rejects_diagonal_miss_outside_finite_segment():
    from research.harness.road_context import crosses_directed_line

    def _box(cx, cy):
        return [(cx - 2.0, cy - 1.0), (cx + 2.0, cy - 1.0), (cx + 2.0, cy + 1.0), (cx - 2.0, cy + 1.0)]

    # Stop line at x=10, y in [-2, 2], forward along +x; ego passes diagonally outside lateral segment (y=8 -> y=10)
    assert crosses_directed_line(
        _box(7.0, 8.0), _box(11.0, 10.0),
        (10.0, -2.0), (10.0, 2.0), (1.0, 0.0),
    ) is False
    # Ego crossing directly through the finite segment at y=0
    assert crosses_directed_line(
        _box(7.0, 0.0), _box(9.0, 0.0),
        (10.0, -2.0), (10.0, 2.0), (1.0, 0.0),
    ) is True


def test_safety_oracle_missing_bbox_reports_error_instead_of_fabricating_box():
    world = _FakeWorld()
    ego = _FakeActor(1, "vehicle.tesla.model3")
    bad_actor = SimpleNamespace(
        id=99,
        type_id="vehicle.mystery.nobbox",
        is_alive=True,
        attributes={"role_name": "npc"},
        get_transform=lambda: _Transform(x=10.0, y=0.0, yaw=0.0),
        get_velocity=lambda: _Vec(5.0, 0.0, 0.0),
        bounding_box=None,
    )
    world._actors.extend([ego, bad_actor])
    oracle = SafetyOracle()
    frame_ev, errors = oracle.capture_frame_evidence(
        world, ego, run_id="r", sim_frame=1, sim_time_s=0.1, phase="evaluation",
    )
    assert frame_ev is not None
    assert all(a.actor_id != "99" for a in frame_ev.actors)
    assert "actors" in errors


def test_make_agent_supports_behavior_and_scripted_without_unbound_import(monkeypatch):
    import sys
    import types
    from research.harness.agents import make_agent
    from research.harness.config import AgentConfig

    class _FakeBehaviorAgent:
        def __init__(self, vehicle, behavior="normal"):
            self.vehicle = vehicle
            self.behavior = behavior
            self.speed_kph = None

        def get_local_planner(self):
            return SimpleNamespace(set_speed=lambda s: setattr(self, "speed_kph", s))

    # Inject minimal fake agents.navigation.behavior_agent module
    agents_pkg = types.ModuleType("agents")
    nav_pkg = types.ModuleType("agents.navigation")
    ba_mod = types.ModuleType("agents.navigation.behavior_agent")
    ba_mod.BehaviorAgent = _FakeBehaviorAgent
    monkeypatch.setitem(sys.modules, "agents", agents_pkg)
    monkeypatch.setitem(sys.modules, "agents.navigation", nav_pkg)
    monkeypatch.setitem(sys.modules, "agents.navigation.behavior_agent", ba_mod)

    world = _FakeWorld()
    ego = _FakeActor(1, "vehicle.tesla.model3")
    ba = make_agent(world, ego, AgentConfig(kind="behavior", behavior="cautious"))
    assert isinstance(ba, _FakeBehaviorAgent) and ba.behavior == "cautious"

    scripted = make_agent(
        world,
        ego,
        AgentConfig(kind="scripted", init_kwargs={"case_id": "collision_positive_contact"}),
    )
    assert scripted.run_purpose == "detector_validation"
    ctrl = scripted.run_step()
    assert ctrl.throttle == 0.75


def test_live_capture_producer_confirms_hard_braking_and_hazard_braking_without_frame_collision_channel(tmp_path, monkeypatch):
    """Producer-to-detector check: SafetyOracle never emits 'collision' on frames, yet live braking must be PRESENT."""
    from research.harness.runner import _LiveOutcomeCapture

    world = _FakeWorld()
    ego = _FakeActor(1, "vehicle.tesla.model3", x=0.0, y=0.0, vx=10.0, ax=0.0)
    lead = _FakeActor(2, "vehicle.audi.tt", x=10.0, y=0.0, vx=0.0, ax=0.0)
    world._actors.extend([ego, lead])

    monkeypatch.setattr(CollisionOracle, "_identity_transform", staticmethod(lambda w: _Transform()))
    col = CollisionOracle()
    col.attach(world, ego)
    lane = LaneInvasionOracle()
    safety = SafetyOracle()

    cfg = AppConfig()
    cfg.outcomes.enabled = True
    cfg.outcomes.evidence_output_dir = tmp_path / "evidence"
    cfg.outcomes.callback_drain_timeout_s = 0.01
    scenario = ScenarioSpec("brake-live", "Town01", "ClearNoon", 0, 1, "live brake", max_ticks=15)
    cap = _LiveOutcomeCapture(cfg, scenario, "run-brake-live", 0.1)

    for i in range(12):
        world._frame = i + 1
        world._time = round(i * 0.1, 4)
        ego.x = float(i)
        if 2 <= i <= 4:
            ego.ax = -10.0
            ego.control = _Control(brake=1.0)
        else:
            ego.ax = 0.0
            ego.control = _Control(throttle=0.4)
        # Keep lead actor only on frame 1 (low-TTC conflict right before braking at frame 2..4)
        world._actors = [ego, lead] if i == 1 else [ego]
        cap.observe_physics_step(
            world, ego, safety, col, lane,
            phase="evaluation", tick_frame=world._frame, control=ego.control,
            actor_roles={1: "ego", 2: "lead_vehicle"},
        )
    _, _, outcome = cap.finalize(
        execution_status="completed",
        termination_reason="horizon_exhausted",
        reached_goal=False,
        collision_oracle=col,
        lane_oracle=lane,
    )
    assert outcome is not None
    assert outcome["categories"]["behaviour"]["hard_braking"]["status"] == "present"
    assert outcome["categories"]["safety"]["hazard_associated_braking"]["status"] == "present"


def test_live_capture_quiescent_drain_catches_delayed_callback_and_unindexed_terminal_timestamp(tmp_path, monkeypatch):
    import threading
    import time as _time
    from research.harness.runner import _LiveOutcomeCapture

    world = _FakeWorld()
    ego = _FakeActor(1, "vehicle.tesla.model3", x=0.0, y=0.0, vx=5.0)
    world._actors.append(ego)
    monkeypatch.setattr(CollisionOracle, "_identity_transform", staticmethod(lambda w: _Transform()))
    col = CollisionOracle()
    sensor = col.attach(world, ego)
    lane = LaneInvasionOracle()
    safety = SafetyOracle()

    cfg = AppConfig()
    cfg.outcomes.enabled = True
    cfg.outcomes.evidence_output_dir = tmp_path / "evidence"
    cfg.outcomes.callback_drain_timeout_s = 0.05
    scenario = ScenarioSpec("late-cb", "Town01", "ClearNoon", 0, 1, "late cb", max_ticks=5)
    cap = _LiveOutcomeCapture(cfg, scenario, "run-late-cb", 0.1)

    for i in range(3):
        world._frame = i + 1
        world._time = round(i * 0.1, 4)
        cap.observe_physics_step(world, ego, safety, col, lane, phase="evaluation", tick_frame=world._frame)

    # Fire a delayed unindexed callback at 8ms (within 50ms timeout) AFTER finalize begins draining
    def _fire_delayed():
        _time.sleep(0.008)
        sensor.fire(
            SimpleNamespace(
                frame=99,  # unindexed terminal frame!
                timestamp=0.35,  # > last frame 0.2s!
                other_actor=SimpleNamespace(id=55, type_id="vehicle.audi.tt"),
                normal_impulse=_Vec(20.0, 0.0, 0.0),
            )
        )

    t = threading.Thread(target=_fire_delayed)
    t.start()
    m_path, _, outcome = cap.finalize(
        execution_status="completed",
        termination_reason="horizon_exhausted",
        reached_goal=False,
        collision_oracle=col,
        lane_oracle=lane,
    )
    t.join()
    assert outcome is not None and m_path is not None
    # Delayed callback was caught AND unindexed timestamp > frame_end did NOT crash replay!
    assert outcome["categories"]["safety"]["collision"]["status"] == "present"
    manifest_obj, _ = load_capture_manifest(m_path)
    assert manifest_obj.channels["collision"].status == "partial"
    assert manifest_obj.end_s >= 0.35


def test_distant_unreadable_actor_outside_scan_radius_does_not_poison_actors_channel():
    world = _FakeWorld()
    ego = _FakeActor(1, "vehicle.tesla.model3", x=0.0, y=0.0)
    far_bad = SimpleNamespace(
        id=999,
        type_id="vehicle.far.nobbox",
        is_alive=True,
        get_location=lambda: _Vec(300.0, 0.0, 0.0),
        bounding_box=None,
    )
    world._actors.extend([ego, far_bad])
    oracle = SafetyOracle()
    frame_ev, errors = oracle.capture_frame_evidence(
        world, ego, run_id="r", sim_frame=1, sim_time_s=0.1, phase="evaluation", scan_radius_m=50.0,
    )
    assert frame_ev is not None
    assert "actors" in frame_ev.available_channels
    assert "actors" not in errors


def test_road_context_junction_boundary_crossing_and_rotated_line_negative():
    import math
    from research.harness.interaction_geometry import world_footprint
    from research.harness.outcome_models import ActorState
    from research.harness.road_context import RoadContextIndex, crosses_directed_line

    # 1. Exact rotated-rectangle negative control: swept box passes outside finite segment [y=-2..2]
    prev_box = world_footprint(ActorState("ego", "vehicle", 9.0, 4.0, math.radians(35), 2.0, 1.0, 5.0, 0.0, 0.0, 1.5))
    curr_box = world_footprint(ActorState("ego", "vehicle", 11.0, 4.5, math.radians(35), 2.0, 1.0, 5.0, 0.0, 0.0, 1.5))
    assert crosses_directed_line(prev_box, curr_box, (10.0, -2.0), (10.0, 2.0), (1.0, 0.0)) is False

    # 2. Junction-boundary stop-line crossing: ego approaches on road 10 lane -1, exits into junction road 99
    stop_wp = SimpleNamespace(
        road_id=10, lane_id=-1, lane_width=3.5, is_junction=False, lane_type="Driving", lane_change="Both",
        transform=_Transform(x=10.0, y=0.0, yaw=0.0),
    )
    tl = SimpleNamespace(id=7, get_state=lambda: "Red", get_stop_waypoints=lambda: [stop_wp])

    class _JuncMap:
        def __init__(self):
            self.in_junc = False

        def get_waypoint(self, loc, project_to_road=True, lane_type=None):
            if self.in_junc:
                return SimpleNamespace(
                    road_id=99, lane_id=-1, lane_width=3.5, is_junction=True, lane_type="Driving", lane_change="None",
                    transform=_Transform(x=loc.x, y=loc.y, yaw=0.0),
                )
            return SimpleNamespace(
                road_id=10, lane_id=-1, lane_width=3.5, is_junction=False, lane_type="Driving", lane_change="Both",
                transform=_Transform(x=loc.x, y=loc.y, yaw=0.0),
            )

    jmap = _JuncMap()
    jworld = SimpleNamespace(get_map=lambda: jmap, get_actors=lambda: [])
    idx = RoadContextIndex(jworld)

    ego_v = _FakeActor(1, "vehicle.tesla.model3", x=7.0, y=0.0)
    ego_v.is_at_traffic_light = lambda: not jmap.in_junc
    ego_v.get_traffic_light = lambda: tl if not jmap.in_junc else None

    s0 = ActorState("1", "vehicle", 7.0, 0.0, 0.0, 2.0, 1.0, 10.0, 0.0, 0.0, 1.5, role="ego")
    rc0, _, _ = idx.sample(ego_v, s0, None, ())
    assert rc0.stop_line_applicable is True and rc0.red_light_state == "red"

    # Now ego crosses x=10 into junction road 99 (and exits trigger box!)
    jmap.in_junc = True
    ego_v.x = 9.0
    s1 = ActorState("1", "vehicle", 9.0, 0.0, 0.0, 2.0, 1.0, 10.0, 0.0, 0.0, 1.5, role="ego")
    rc1, _, _ = idx.sample(ego_v, s1, s0, ())
    assert rc1.stop_line_applicable is True
    assert rc1.stop_line_crossed is True
    assert rc1.red_light_state == "red"


def test_road_context_detects_approaching_same_lane_red_light_before_trigger_box():
    from research.harness.outcome_models import ActorState
    from research.harness.road_context import RoadContextIndex

    same_lane_stop = SimpleNamespace(
        road_id=19, lane_id=-1, lane_width=3.5, is_junction=False, lane_type="Driving", lane_change="None",
        transform=_Transform(x=0.0, y=10.0, yaw=90.0),
    )
    other_lane_stop = SimpleNamespace(
        road_id=19, lane_id=1, lane_width=3.5, is_junction=False, lane_type="Driving", lane_change="None",
        transform=_Transform(x=3.5, y=10.0, yaw=-90.0),
    )
    tl_same = SimpleNamespace(
        id=41, type_id="traffic.traffic_light",
        get_state=lambda: "Red", get_stop_waypoints=lambda: [same_lane_stop],
    )
    tl_other = SimpleNamespace(
        id=42, type_id="traffic.traffic_light",
        get_state=lambda: "Green", get_stop_waypoints=lambda: [other_lane_stop],
    )

    class _ApproachMap:
        def get_waypoint(self, loc, project_to_road=True, lane_type=None):
            return SimpleNamespace(
                road_id=19, lane_id=-1, lane_width=3.5, is_junction=False, lane_type="Driving", lane_change="None",
                transform=_Transform(x=loc.x, y=loc.y, yaw=90.0),
            )

    world = SimpleNamespace(get_map=lambda: _ApproachMap(), get_actors=lambda: [tl_other, tl_same])
    idx = RoadContextIndex(world)

    # Farther than 10m behind stop line -> not applicable yet
    ego_far = _FakeActor(1, "vehicle.tesla.model3", x=0.0, y=-5.0)
    ego_far.is_at_traffic_light = lambda: False
    ego_far.get_traffic_light = lambda: None
    s_far = ActorState("1", "vehicle", 0.0, -5.0, 1.5707963267948966, 2.0, 1.0, 0.0, 0.0, 0.0, 1.5, role="ego")
    rc_far, tc_far, _ = idx.sample(ego_far, s_far, None, ())
    assert rc_far.stop_line_applicable is False
    assert tc_far.red_light_stop is False

    # Stopped 3.5m before stop line (front bumper 1.5m before line), outside CARLA's narrow trigger box
    ego_v = _FakeActor(1, "vehicle.tesla.model3", x=0.0, y=6.5)
    ego_v.is_at_traffic_light = lambda: False
    ego_v.get_traffic_light = lambda: None
    s0 = ActorState("1", "vehicle", 0.0, 6.5, 1.5707963267948966, 2.0, 1.0, 0.0, 0.0, 0.0, 1.5, role="ego")
    rc0, tc0, _ = idx.sample(ego_v, s0, s_far, ())
    assert rc0.stop_line_applicable is True
    assert rc0.red_light_state == "red"
    assert rc0.stop_line_crossed is False
    assert tc0.red_light_stop is True


def test_agent_route_waypoints_extracts_from_carla_local_planner():
    from research.harness.runner import _agent_route_waypoints, _extract_route_points

    plan = [
        (SimpleNamespace(transform=_Transform(x=1.0, y=2.0)), "LANEFOLLOW"),
        (SimpleNamespace(transform=_Transform(x=6.0, y=2.0)), "LANEFOLLOW"),
        (SimpleNamespace(transform=_Transform(x=11.0, y=2.0)), "STRAIGHT"),
    ]
    planner = SimpleNamespace(get_plan=lambda: plan)
    behavior_like_agent = SimpleNamespace(get_local_planner=lambda: planner)
    raw = _agent_route_waypoints(behavior_like_agent)
    pts = _extract_route_points(raw)
    assert pts == [(1.0, 2.0), (6.0, 2.0), (11.0, 2.0)]
