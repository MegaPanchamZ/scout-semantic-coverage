from dataclasses import replace

import pytest

from research.harness.outcome_models import SensorEvidence, ChannelCoverage
from research.harness.safety_outcomes import OutcomeEvaluator
from research.tests.outcome_helpers import CHANNELS, actor, frame, manifest


def evaluate(frames, sensors=(), capture=None):
    e = OutcomeEvaluator()
    for f in frames:
        e.observe(f)
    for s in sensors:
        e.record_sensor_event(s)
    return e.finish(capture or manifest()).to_dict()


def status(result, category, name):
    return result["categories"][category][name]["status"]


def interaction_frames(n=11):
    # At t=0, the ego closes on a stopped actor. At t=.1, that actor
    # accelerates away; subsequent observed separation closes the encounter.
    return [frame(i, actors=(actor("a", x=14 + 2*i, vx=0 if i == 0 else 20),)) for i in range(n)]


def test_resolved_near_miss_and_later_other_collision_can_coexist():
    contact = SensorEvidence("run", 8, .8, "collision", "b", impulse_ns=(3, 4, 0))
    result = evaluate(interaction_frames(), [contact])
    assert status(result, "safety", "near_miss") == "present"
    assert status(result, "safety", "collision") == "present"
    assert next(e for e in result["events"] if e["label"] == "collision")["measurements"]["peak_impulse_ns"] == 5


def test_delayed_callback_suppresses_same_encounter_near_miss_and_groups_contact():
    callbacks = [SensorEvidence("run", i, i / 10, "collision", "a", callback_id=str(i)) for i in (3, 2, 4)]
    result = evaluate(interaction_frames(), callbacks + [callbacks[0]])
    assert status(result, "safety", "near_miss") == "absent"
    contacts = [e for e in result["events"] if e["label"] == "collision"]
    assert len(contacts) == 1
    assert contacts[0]["measurements"]["callback_count"] == 3


@pytest.mark.parametrize("mode", ["horizon", "missing_actor", "gap", "sensor_missing"])
def test_unfinished_or_unobserved_encounters_are_not_completed_near_misses(mode):
    frames = interaction_frames()
    capture = manifest()
    if mode == "horizon":
        frames = frames[:4]
        capture = manifest(.3)
    elif mode == "missing_actor":
        frames = [frames[0], *(replace(f, actors=()) for f in frames[1:])]
    elif mode == "gap":
        frames = [frames[0], *frames[3:]]
    else:
        capture = manifest(channels={})
    result = evaluate(frames, capture=capture)
    assert status(result, "safety", "near_miss") == "unknown"


def test_partial_contact_survives_and_no_evidence_means_unknown():
    contact = SensorEvidence("run", 3, .3, "collision", "a")
    result = evaluate([], [contact], manifest(execution_status="partial", channels={}))
    assert status(result, "safety", "collision") == "present"
    assert status(result, "behaviour", "hard_braking") == "unknown"
    assert status(evaluate([], capture=manifest(channels={})), "safety", "collision") == "unknown"


def test_sustained_proximity_is_one_episode_and_is_not_a_near_miss():
    frames = [frame(i, actors=(actor("a", x=i, y=3, vx=10),)) for i in range(11)]
    result = evaluate(frames)
    assert status(result, "behaviour", "close_proximity") == "present"
    assert status(result, "safety", "near_miss") == "absent"
    assert len([e for e in result["events"] if e["label"] == "close_proximity"]) == 1


def test_stationary_is_not_stuck_and_short_horizon_is_not_timeout():
    frames = [frame(i, ego=actor(vx=0)) for i in range(101)]
    result = evaluate(frames, capture=manifest(10))
    assert status(result, "task", "prolonged_immobilization") == "present"
    assert next(e for e in result["events"] if e["label"] == "prolonged_immobilization")["state"] == "censored"
    assert status(result, "task", "stuck") == "unknown"
    assert status(result, "task", "timeout") == "unknown"
    assert status(result, "rules", "red_light_violation") == "unknown"
    assert result["aggregates"]["safety_event"]["status"] == "unknown"
    result = evaluate(frames, capture=manifest(10, completion_deadline_s=10))
    assert status(result, "task", "timeout") == "present"
    result = evaluate(frames, capture=manifest(10, completion_deadline_s=10, termination_reason="collision_stop"))
    assert status(result, "task", "timeout") == "unknown"


def braking_frames():
    return [frame(i, longitudinal_acceleration_mps2=-10 if 2 <= i <= 4 else 0,
                  brake=1 if 2 <= i <= 4 else 0, ego=actor(vx=10 - min(max(i-1, 0), 3))) for i in range(11)]


def test_braking_requires_speed_controls_duration_and_impact_exclusion():
    frames = braking_frames()
    assert status(evaluate(frames), "behaviour", "hard_braking") == "present"
    assert status(evaluate([replace(f, brake=0) for f in frames]), "behaviour", "hard_braking") == "absent"
    assert status(evaluate([replace(f, ego=actor(vx=4)) for f in frames]), "behaviour", "hard_braking") == "absent"
    assert status(evaluate([replace(f, longitudinal_acceleration_mps2=-8) for f in frames]), "behaviour", "hard_braking") == "absent"
    contact = SensorEvidence("run", 3, .3, "collision", "a")
    assert status(evaluate(frames, [contact]), "behaviour", "hard_braking") == "absent"
    assert status(evaluate(frames, capture=manifest(channels={})), "behaviour", "hard_braking") == "unknown"


def test_missing_sample_fields_cannot_prove_negative_braking():
    frames = [replace(f, brake=None) for f in braking_frames()]
    assert status(evaluate(frames), "behaviour", "hard_braking") == "unknown"


def test_startup_hold_excluded_and_timestamp_gap_breaks_streak():
    frames = [frame(i, ego=actor(vx=0)) for i in range(101)]
    frames[50] = replace(frames[50], phase="startup_hold")
    result = evaluate(frames, capture=manifest(10))
    assert status(result, "task", "prolonged_immobilization") == "unknown"


def test_repeat_finish_is_deterministic_and_identity_order_is_checked():
    e = OutcomeEvaluator()
    e.observe(frame(0))
    with pytest.raises(ValueError):
        e.observe(frame(0))
    with pytest.raises(ValueError):
        e.observe(replace(frame(1), run_id="other"))
    e.observe(frame(1))
    assert e.finish(manifest()).to_dict() == e.finish(manifest()).to_dict()
    with pytest.raises(ValueError):
        e.finish(manifest(.05))
    with pytest.raises(ValueError):
        e.finish(manifest(execution_status="dry_run"))


def test_short_phase_break_censors_encounter_even_inside_sampling_tolerance():
    frames = interaction_frames()
    # Insert a non-evaluation phase only .05 seconds long; it must not be
    # bridged merely because the remaining frame interval is within .11 s.
    frames = [replace(f, sim_frame=f.sim_frame * 2) for f in frames]
    hold = replace(frame(1), sim_frame=1, sim_time_s=.05, phase="startup_hold")
    frames.insert(1, hold)
    result = evaluate(frames)
    assert status(result, "safety", "near_miss") == "unknown"


def test_conflicting_callback_identity_is_rejected():
    e = OutcomeEvaluator()
    s = SensorEvidence("run", 1, .1, "collision", "a", callback_id="one")
    e.record_sensor_event(s)
    with pytest.raises(ValueError):
        e.record_sensor_event(replace(s, actor_id="b"))


def test_velocity_difference_braking_fallback_emits_tagged_positive_without_claiming_absent():
    # Acceleration channel absent; ego decelerates from 10 to 7 m/s over 0.3s (10 m/s^2) with brake=1
    no_accel_channels = ("actors", "ego_motion", "collision", "controls")
    frames = [
        frame(
            i,
            ego=actor(vx=10.0 - min(max(i - 1, 0), 3)),
            available_channels=no_accel_channels,
            longitudinal_acceleration_mps2=None,
            brake=1.0 if 2 <= i <= 4 else 0.0,
        )
        for i in range(11)
    ]
    cap = manifest(channels={c: ChannelCoverage("complete", 0, 1.0) for c in no_accel_channels})
    res = evaluate(frames, capture=cap)
    assert status(res, "behaviour", "hard_braking") == "present"
    ev = next(e for e in res["events"] if e["label"] == "hard_braking")
    assert ev["measurements"]["source"] == "velocity_difference_fallback"
    # Without braking events and without acceleration channel, absence stays unknown (never fabricated)
    calm = [replace(f, ego=actor(vx=10.0), brake=0.0) for f in frames]
    assert status(evaluate(calm, capture=cap), "behaviour", "hard_braking") == "unknown"


def test_float_boundary_duration_0_4_to_1_4_satisfies_1_0s_requirement():
    # 1.4 - 0.4 == 0.9999999999999999 in IEEE 754; must not be rejected as < 1.0s
    frames = [frame(i, sim_time_s=0.4 + 0.1 * i) for i in range(11)]
    cap = manifest(1.4, start_s=0.4)
    res = evaluate(frames, capture=cap)
    assert status(res, "safety", "collision") == "absent"
    assert status(res, "safety", "near_miss") == "absent"


def test_partial_run_preserves_hard_braking_positive_when_collision_covers_streak():
    # Collision channel only covers 0.0..0.6 (partial), braking occurs at 0.2..0.4 (inside collision coverage)
    frames = [
        frame(
            i,
            longitudinal_acceleration_mps2=-10.0 if 2 <= i <= 4 else 0.0,
            brake=1.0 if 2 <= i <= 4 else 0.0,
        )
        for i in range(11)
    ]
    ch = {c: ChannelCoverage("complete", 0.0, 1.0) for c in ("actors", "ego_motion", "acceleration", "controls")}
    ch["collision"] = ChannelCoverage("partial", 0.0, 0.6)
    cap = manifest(1.0, execution_status="partial", channels=ch)
    res = evaluate(frames, capture=cap)
    assert status(res, "behaviour", "hard_braking") == "present"


def test_velocity_fallback_closed_braking_not_falsely_censored_and_already_overlapping_not_near_miss():
    # Clean closed velocity-difference braking streak (decel at frames 2..4, stops braking at 5..10) -> state == 'closed'
    no_accel = ("actors", "ego_motion", "collision", "controls")
    frames = [
        frame(
            i,
            ego=actor(vx=10.0 - min(max(i - 1, 0), 3)),
            available_channels=no_accel,
            longitudinal_acceleration_mps2=None,
            brake=1.0 if 2 <= i <= 4 else 0.0,
        )
        for i in range(11)
    ]
    cap = manifest(channels={c: ChannelCoverage("complete", 0, 1.0) for c in no_accel})
    res = evaluate(frames, capture=cap)
    ev = next(e for e in res["events"] if e["label"] == "hard_braking")
    assert ev["state"] == "closed"

    # Initial overlap without predicted closing conflict must never be confirmed as near_miss
    overlap_frames = [
        frame(i, actors=(actor("ov", x=0.5 if i == 0 else 20.0 + i, vx=10.0),))
        for i in range(11)
    ]
    res_ov = evaluate(overlap_frames)
    assert status(res_ov, "safety", "near_miss") == "unknown"
    ce = next(e for e in res_ov["events"] if e["label"] == "conflict_exposure")
    assert "geometric_overlap_without_sensor_confirmation" in ce["reasons"]
