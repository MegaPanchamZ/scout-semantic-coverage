from dataclasses import asdict, replace

import pytest

from research.harness.outcome_config import load_outcome_profile, REQUIRED_SETTING_UNITS
from research.harness.outcome_models import ChannelCoverage, FrameEvidence, RouteProgress, TaskContext
from research.harness.safety_outcomes import OutcomeEvaluator
from research.tests.outcome_helpers import CHANNELS, actor, frame, manifest


FREE = TaskContext(True, False, False, False, False, False, ("constructed_context",))
TASK_CHANNELS = (*CHANNELS, "task_context", "route_progress")


def trace(n=101, *, context=FREE, speed=0, advance=0, dt=.1):
    return [frame(i, sim_time_s=round(i*dt, 8), ego=actor(x=i*advance, vx=speed),
                  available_channels=TASK_CHANNELS, task_context=context,
                  route_progress=RouteProgress("route", "available", 1000, i*advance, i*advance, 0)) for i in range(n)]


def run(frames, *, profile=None, channels=TASK_CHANNELS):
    end = frames[-1].sim_time_s
    evaluator = OutcomeEvaluator(profile)
    for f in frames:
        evaluator.observe(f)
    return evaluator.finish(manifest(end, channels={c: ChannelCoverage("complete", 0, end) for c in channels})).to_dict()


def status(result, name):
    return result["categories"]["task"][name]["status"]


def test_unjustified_stationary_window_and_event_grouping():
    result = run(trace(151))
    for label in ("stuck", "no_progress", "prolonged_immobilization"):
        assert status(result, label) == "present"
        episodes = [e for e in result["events"] if e["label"] == label]
        assert len(episodes) == 1
        assert (episodes[0]["start_s"], episodes[0]["end_s"]) == (0, 15)
        assert episodes[0]["state"] == "censored"


@pytest.mark.parametrize("reason", ["red_light_stop", "stop_sign_stop", "blocked_route", "yielding"])
def test_legitimate_wait_is_not_stuck_or_no_progress(reason):
    result = run(trace(context=replace(FREE, **{reason: True})))
    assert status(result, "prolonged_immobilization") == "present"
    assert status(result, "stuck") == status(result, "no_progress") == "absent"


@pytest.mark.parametrize("context", [None, TaskContext(objective_active=True), replace(FREE, sensor_wait=True)])
def test_missing_or_execution_wait_context_is_unknown(context):
    result = run(trace(context=context))
    assert status(result, "stuck") == status(result, "no_progress") == "unknown"


def test_inactive_objective_is_explicitly_not_applicable():
    result = run(trace(context=replace(FREE, objective_active=False)))
    assert status(result, "stuck") == status(result, "no_progress") == "not_applicable"


def test_creeping_and_forward_progress_are_distinguished():
    creeping = run(trace(speed=.21, advance=.005))
    assert status(creeping, "stuck") == "absent"
    assert status(creeping, "no_progress") == "present"
    advancing = run(trace(speed=1, advance=.1))
    assert status(advancing, "stuck") == status(advancing, "no_progress") == "absent"


def test_reversing_or_retracing_does_not_increase_route_high_water():
    frames = trace(speed=1)
    frames = [replace(f, route_progress=RouteProgress("route", "available", 1000, 100-f.sim_time_s, 100, 0)) for f in frames]
    assert status(run(frames), "no_progress") == "present"


@pytest.mark.parametrize("break_kind", ["gap", "phase", "context", "discontinuity", "route", "anchor"])
def test_windows_do_not_bridge_missing_evidence_or_route_reacquisition(break_kind):
    frames = trace()
    if break_kind == "gap":
        del frames[50]
    elif break_kind == "phase":
        frames[50] = replace(frames[50], phase="startup_hold")
    elif break_kind == "context":
        frames[50] = replace(frames[50], task_context=None)
    elif break_kind == "discontinuity":
        frames[50] = replace(frames[50], discontinuity=True)
    else:
        frames[50:] = [replace(f, route_progress=replace(f.route_progress, **(
            {"route_id": "different"} if break_kind == "route" else {"continuity_id": 1}))) for f in frames[50:]]
    result = run(frames)
    assert status(result, "no_progress") == "unknown"
    assert status(result, "stuck") == ("present" if break_kind in {"route", "anchor"} else "unknown")


def test_positive_window_survives_later_missing_context():
    frames = trace(121)
    frames[-1] = replace(frames[-1], task_context=None)
    result = run(frames)
    assert status(result, "stuck") == status(result, "no_progress") == "present"


def test_legitimate_wait_breaks_eligible_window_without_creating_unknown():
    frames = trace()
    frames[50] = replace(frames[50], task_context=replace(FREE, yielding=True))
    result = run(frames)
    assert status(result, "stuck") == status(result, "no_progress") == "absent"


def test_threshold_boundaries_use_speed_strictly_and_progress_inclusively():
    result = run(trace(speed=.2, advance=.01))
    assert status(result, "stuck") == "absent"
    assert status(result, "no_progress") == "present"
    result = run(trace(speed=.2, advance=.0101))
    assert status(result, "no_progress") == "absent"


def test_time_windows_use_seconds_and_require_motion_only_for_stuck():
    frames = trace(101, dt=.05)
    assert status(run(frames), "no_progress") == "unknown"
    frames = [replace(f, route_progress=None) for f in trace()]
    assert status(run(frames), "stuck") == "present"
    assert status(run(frames), "no_progress") == "unknown"


def test_unknown_frame_route_channel_prevents_negative_claim():
    frames = trace(speed=1, advance=.1)
    frames[50] = replace(frames[50], available_channels=CHANNELS + ("task_context",))
    assert status(run(frames), "no_progress") == "unknown"


def test_contract_round_trip_and_old_profile_does_not_acquire_defaults():
    first = trace()[0]
    assert FrameEvidence.from_dict(asdict(first)) == first
    profile = load_outcome_profile()
    old = replace(profile, settings=tuple(s for s in profile.settings if s.name in REQUIRED_SETTING_UNITS))
    assert status(run(trace(), profile=old), "no_progress") == "unknown"
    with pytest.raises(ValueError):
        replace(profile, settings=tuple(s for s in profile.settings if s.name != "progress_epsilon_m"))
    with pytest.raises(ValueError):
        TaskContext(objective_active="true")
