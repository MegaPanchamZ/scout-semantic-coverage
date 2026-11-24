"""Positive, negative and uncertainty controls for traffic rules, lane legality and hazard braking."""
from dataclasses import asdict, replace
import math

import pytest

from research.harness.outcome_config import REQUIRED_SETTING_UNITS, load_outcome_profile
from research.harness.outcome_models import (
    ChannelCoverage, FrameEvidence, RoadRuleContext, SensorEvidence,
)
from research.harness.safety_outcomes import OutcomeEvaluator
from research.tests.outcome_helpers import CHANNELS, actor, frame, manifest


RULE_CHANNELS = (*CHANNELS, "lane_invasion", "road_context")
COMPLIANT_ROAD = RoadRuleContext(
    lane_id="lane_1",
    lane_type="driving",
    in_junction=False,
    lane_change_permission="both",
    permitted_travel_yaw_rad=0.0,
    speed_limit_mps=13.89,  # 50 km/h
    red_light_state="green",
    stop_line_applicable=False,
    stop_line_crossed=False,
    stop_sign_applicable=False,
    stop_sign_approach=False,
    stop_sign_line_crossed=False,
    evidence_refs=("constructed_map",),
)


def rule_frames(n=21, *, road=COMPLIANT_ROAD, speed=10.0, yaw=0.0, reverse=False, **kwargs):
    vx = speed * math.cos(yaw)
    vy = speed * math.sin(yaw)
    return [
        frame(
            i,
            ego=actor(x=i * vx * 0.1, y=i * vy * 0.1, yaw_rad=yaw, vx=vx, vy=vy),
            available_channels=RULE_CHANNELS,
            road_context=road,
            reverse=reverse,
            **kwargs,
        )
        for i in range(n)
    ]


def evaluate(frames, sensors=(), *, profile=None, channels=RULE_CHANNELS):
    end = frames[-1].sim_time_s
    evaluator = OutcomeEvaluator(profile)
    for f in frames:
        evaluator.observe(f)
    for s in sensors:
        evaluator.record_sensor_event(s)
    capture = manifest(end, channels={c: ChannelCoverage("complete", 0, end) for c in channels})
    return evaluator.finish(capture).to_dict()


def status(result, group, name):
    return result["categories"][group][name]["status"]


def test_compliant_driving_gives_absent_rules_and_round_trips():
    frames = rule_frames()
    assert FrameEvidence.from_dict(asdict(frames[0])) == frames[0]
    result = evaluate(frames)
    for name in ("red_light_violation", "stop_sign_violation", "speeding", "wrong_way", "illegal_lane_change"):
        assert status(result, "rules", name) == "absent"
    assert status(result, "behaviour", "lane_crossing") == "absent"
    assert status(result, "safety", "unsafe_lane_departure") == "absent"
    assert status(result, "safety", "hazard_associated_braking") == "absent"
    assert result["aggregates"]["traffic_rule_violation"]["status"] == "absent"
    assert result["aggregates"]["safety_event"]["status"] == "absent"


def test_red_light_crossing_vs_approach_and_unrelated_signal():
    # Crossing applicable stop line on red -> present
    frames = rule_frames()
    frames[10] = replace(
        frames[10],
        road_context=replace(COMPLIANT_ROAD, stop_line_applicable=True, stop_line_crossed=True, red_light_state="red"),
    )
    assert status(evaluate(frames), "rules", "red_light_violation") == "present"

    # Approaching red light without crossing -> absent
    approach = rule_frames()
    approach[10] = replace(
        approach[10],
        road_context=replace(COMPLIANT_ROAD, stop_line_applicable=True, stop_line_crossed=False, red_light_state="red"),
    )
    assert status(evaluate(approach), "rules", "red_light_violation") == "absent"

    # Unrelated lane's red light -> absent
    unrelated = rule_frames()
    unrelated[10] = replace(
        unrelated[10],
        road_context=replace(COMPLIANT_ROAD, stop_line_applicable=False, stop_line_crossed=True, red_light_state="red"),
    )
    assert status(evaluate(unrelated), "rules", "red_light_violation") == "absent"

    # Unknown signal state during applicable crossing -> unknown
    unknown_sig = rule_frames()
    unknown_sig[10] = replace(
        unknown_sig[10],
        road_context=replace(COMPLIANT_ROAD, stop_line_applicable=True, stop_line_crossed=True, red_light_state="unknown"),
    )
    assert status(evaluate(unknown_sig), "rules", "red_light_violation") == "unknown"


def test_stop_sign_roll_through_vs_compliant_dwell_and_outside_approach_stop():
    # Compliant: 1.0 s dwell inside approach zone before crossing
    compliant = rule_frames(31, speed=5.0)
    for i in range(5, 20):
        compliant[i] = replace(
            compliant[i],
            ego=replace(compliant[i].ego, vx=0.0),
            road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=False),
        )
    compliant[20] = replace(
        compliant[20],
        road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=True),
    )
    assert status(evaluate(compliant), "rules", "stop_sign_violation") == "absent"

    # Roll-through: never dwelled long enough inside approach
    roll = rule_frames(21, speed=5.0)
    for i in range(5, 10):
        roll[i] = replace(
            roll[i],
            road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=False),
        )
    roll[10] = replace(
        roll[10],
        road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=True),
    )
    assert status(evaluate(roll), "rules", "stop_sign_violation") == "present"

    # Stopping *outside* the approach region does not satisfy stop-sign dwell
    early_stop = rule_frames(31, speed=5.0)
    for i in range(0, 15):
        early_stop[i] = replace(early_stop[i], ego=replace(early_stop[i].ego, vx=0.0))
    for i in range(16, 20):
        early_stop[i] = replace(
            early_stop[i],
            road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=False),
        )
    early_stop[20] = replace(
        early_stop[20],
        road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=True),
    )
    assert status(evaluate(early_stop), "rules", "stop_sign_violation") == "present"


def test_speeding_respects_tolerance_duration_limit_transitions_and_missing_limit():
    # Within tolerance (13.89 + 0.9 <= 14.89) -> absent
    assert status(evaluate(rule_frames(21, speed=14.5)), "rules", "speeding") == "absent"

    # Above limit + tolerance for >= 1.0 s -> present
    assert status(evaluate(rule_frames(21, speed=16.0)), "rules", "speeding") == "present"

    # Limit transition drop mid-run triggers speeding
    frames = rule_frames(25, speed=12.0)
    for i in range(10, 25):
        frames[i] = replace(frames[i], road_context=replace(COMPLIANT_ROAD, speed_limit_mps=8.0))
    assert status(evaluate(frames), "rules", "speeding") == "present"

    # Unknown map speed limit -> unknown
    no_limit = rule_frames(21, road=replace(COMPLIANT_ROAD, speed_limit_mps=None))
    assert status(evaluate(no_limit), "rules", "speeding") == "unknown"


def test_wrong_way_excludes_permitted_reverse_and_junctions():
    # Opposing motion (yaw=pi, permitted=0) for > 1 s -> present
    opposing = rule_frames(21, speed=5.0, yaw=math.pi, reverse=False)
    assert status(evaluate(opposing), "rules", "wrong_way") == "present"

    # Legal reverse manoeuvre -> absent
    reversing = rule_frames(21, speed=3.0, yaw=math.pi, reverse=True)
    assert status(evaluate(reversing), "rules", "wrong_way") == "absent"

    # Ambiguous junction topology excluded -> absent
    junction = rule_frames(21, speed=5.0, yaw=math.pi, reverse=False, road=replace(COMPLIANT_ROAD, in_junction=True))
    assert status(evaluate(junction), "rules", "wrong_way") == "absent"

    # Missing reverse gear state -> unknown
    unknown_rev = rule_frames(21, speed=5.0, yaw=0.0, reverse=None)
    assert status(evaluate(unknown_rev), "rules", "wrong_way") == "unknown"


def test_lane_crossing_legality_and_unsafe_lane_departure():
    frames = rule_frames(21)

    # Permitted broken line crossing -> lane_crossing present, illegal_lane_change absent
    broken = SensorEvidence("run", 5, 0.5, "lane_invasion", "ego", lane_markings=("Broken",))
    res_ok = evaluate(frames, [broken])
    assert status(res_ok, "behaviour", "lane_crossing") == "present"
    assert status(res_ok, "rules", "illegal_lane_change") == "absent"
    assert status(res_ok, "safety", "unsafe_lane_departure") == "absent"

    # Prohibited solid line crossing -> illegal_lane_change present
    solid = SensorEvidence("run", 5, 0.5, "lane_invasion", "ego", lane_markings=("Solid",))
    res_bad = evaluate(frames, [solid])
    assert status(res_bad, "behaviour", "lane_crossing") == "present"
    assert status(res_bad, "rules", "illegal_lane_change") == "present"

    # Junction crossing with solid marking excluded from illegal_lane_change
    junc_frames = rule_frames(21, road=replace(COMPLIANT_ROAD, in_junction=True))
    res_junc = evaluate(junc_frames, [solid])
    assert status(res_junc, "rules", "illegal_lane_change") == "absent"

    # Prohibited road surface (sidewalk) -> unsafe_lane_departure present
    sidewalk = rule_frames(21)
    sidewalk[10] = replace(sidewalk[10], road_context=replace(COMPLIANT_ROAD, lane_type="sidewalk"))
    assert status(evaluate(sidewalk), "safety", "unsafe_lane_departure") == "present"

    # Disabled lane_invasion sensor -> unknown
    res_no_sensor = evaluate(frames, channels=CHANNELS)
    assert status(res_no_sensor, "behaviour", "lane_crossing") == "unknown"
    assert status(res_no_sensor, "rules", "illegal_lane_change") == "unknown"


def test_hazard_associated_braking_distinguishes_hazard_from_legitimate_stop():
    # Hard braking with no nearby/conflicting actor -> hard_braking present, hazard_associated_braking absent
    no_hazard = [
        frame(
            i,
            ego=actor(x=i, vx=10.0 - min(max(i - 1, 0), 3)),
            available_channels=RULE_CHANNELS,
            road_context=COMPLIANT_ROAD,
            reverse=False,
            longitudinal_acceleration_mps2=-10.0 if 2 <= i <= 4 else 0.0,
            brake=1.0 if 2 <= i <= 4 else 0.0,
        )
        for i in range(15)
    ]
    res_clean = evaluate(no_hazard)
    assert status(res_clean, "behaviour", "hard_braking") == "present"
    assert status(res_clean, "safety", "hazard_associated_braking") == "absent"

    # Hard braking shortly after a low-TTC actor conflict -> hazard_associated_braking present
    with_hazard = [
        replace(f, actors=(actor("ped", actor_type="pedestrian", x=10.0, vx=0.0),)) if i == 1 else f
        for i, f in enumerate(no_hazard)
    ]
    res_haz = evaluate(with_hazard)
    assert status(res_haz, "behaviour", "hard_braking") == "present"
    assert status(res_haz, "safety", "hazard_associated_braking") == "present"
    ev = next(e for e in res_haz["events"] if e["label"] == "hazard_associated_braking")
    assert "ped" in ev["actor_ids"]
    assert ev["measurements"]["association"] == "antecedent_or_concurrent_actor_hazard"


def test_old_profile_without_rule_settings_stays_unknown():
    profile = load_outcome_profile()
    old = replace(profile, settings=tuple(s for s in profile.settings if s.name in REQUIRED_SETTING_UNITS))
    result = evaluate(rule_frames(21, speed=20.0), profile=old)
    assert status(result, "rules", "speeding") == "unknown"
    assert status(result, "safety", "hazard_associated_braking") == "unknown"


def test_rule_boundary_controls_junction_lane_red_light_first_frame_and_stop_sign_crossing():
    # 1. Frame-aligned solid lane callback where preceding frame is in_junction=True -> excluded from illegal_lane_change
    frames = rule_frames(21)
    frames[4] = replace(frames[4], road_context=replace(COMPLIANT_ROAD, in_junction=True))
    solid = SensorEvidence("run", 5, 0.5, "lane_invasion", "ego", lane_markings=("Solid",))
    res_j = evaluate(frames, [solid])
    assert status(res_j, "rules", "illegal_lane_change") == "absent"

    # 2. One-sided in_junction=False (pre) + None (post) -> unknown, never fabricated illegal_lane_change
    frames_one_sided = rule_frames(21)
    frames_one_sided[5] = replace(frames_one_sided[5], road_context=replace(COMPLIANT_ROAD, in_junction=None))
    res_one = evaluate(frames_one_sided, [solid])
    assert status(res_one, "rules", "illegal_lane_change") == "unknown"

    # 3. Red-light crossing on first evaluation frame or after discontinuity -> unknown (not fabricated present)
    first_red = rule_frames(21)
    first_red[0] = replace(
        first_red[0],
        road_context=replace(COMPLIANT_ROAD, stop_line_applicable=True, stop_line_crossed=True, red_light_state="red"),
    )
    assert status(evaluate(first_red), "rules", "red_light_violation") == "unknown"

    post_disc = rule_frames(21)
    post_disc[9] = replace(post_disc[9], discontinuity=True)
    post_disc[10] = replace(
        post_disc[10],
        road_context=replace(COMPLIANT_ROAD, stop_line_applicable=True, stop_line_crossed=True, red_light_state="red"),
    )
    assert status(evaluate(post_disc), "rules", "red_light_violation") == "unknown"

    # 4. Stop-sign crossing_active across 2 consecutive frames emits exactly 1 violation (no dead-code double fire)
    ss_two = rule_frames(21, speed=5.0)
    for i in range(5, 10):
        ss_two[i] = replace(ss_two[i], road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=False))
    for i in (10, 11):
        ss_two[i] = replace(ss_two[i], road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=True))
    res_ss = evaluate(ss_two)
    ss_events = [e for e in res_ss["events"] if e["label"] == "stop_sign_violation"]
    assert len(ss_events) == 1

    # 5. Dropped collision callback (partial with sensor_append_failed reason) must NOT confirm hard_braking
    braking_frames = [
        frame(
            i,
            ego=actor(x=i, vx=10.0),
            available_channels=RULE_CHANNELS,
            road_context=COMPLIANT_ROAD,
            reverse=False,
            longitudinal_acceleration_mps2=-10.0 if 2 <= i <= 4 else 0.0,
            brake=1.0 if 2 <= i <= 4 else 0.0,
        )
        for i in range(15)
    ]
    ch_loss = {c: ChannelCoverage("complete", 0.0, 1.4) for c in RULE_CHANNELS}
    ch_loss["collision"] = ChannelCoverage("partial", 0.0, 1.4, reasons=("sensor_append_failed:OSError:disk",))
    ev_loss = OutcomeEvaluator()
    for f in braking_frames:
        ev_loss.observe(f)
    res_loss = ev_loss.finish(manifest(1.4, channels=ch_loss)).to_dict()
    assert status(res_loss, "behaviour", "hard_braking") == "unknown"

    # 6. Overlap-start encounter with predicted conflict MUCH later must NOT retroactively qualify earlier braking
    retro_frames = [
        frame(
            i,
            ego=actor(x=float(i), vx=10.0),
            # t=0.0: overlap; t=0.1..1.9: close parallel proximity keeping encounter alive; t=2.2: predicted conflict!
            actors=(
                actor("a", x=float(i) + 0.5, y=0.0, vx=10.0) if i == 0
                else (actor("a", x=float(i), y=2.2, vx=10.0) if i < 22 else actor("a", x=float(i) + 10.0, y=0.0, vx=0.0)),
            ),
            available_channels=RULE_CHANNELS,
            road_context=COMPLIANT_ROAD,
            reverse=False,
            longitudinal_acceleration_mps2=-10.0 if 2 <= i <= 4 else 0.0,
            brake=1.0 if 2 <= i <= 4 else 0.0,
        )
        for i in range(25)
    ]
    res_retro = evaluate(retro_frames)
    assert status(res_retro, "behaviour", "hard_braking") == "present"
    assert status(res_retro, "safety", "hazard_associated_braking") == "absent"

    # 7. Red-light prior applicable sample with None state -> unknown (not confirmed violation)
    prev_none_red = rule_frames(21)
    prev_none_red[9] = replace(
        prev_none_red[9],
        road_context=replace(COMPLIANT_ROAD, stop_line_applicable=True, stop_line_crossed=False, red_light_state=None, evidence_refs=("tl:7:unknown",)),
    )
    prev_none_red[10] = replace(
        prev_none_red[10],
        road_context=replace(COMPLIANT_ROAD, stop_line_applicable=True, stop_line_crossed=True, red_light_state="red", evidence_refs=("tl:7:red",)),
    )
    assert status(evaluate(prev_none_red), "rules", "red_light_violation") == "unknown"

    # 8. Stop-sign dwell at sign A must NOT survive switching to sign B and clear a later crossing at sign A
    ss_switch = rule_frames(35, speed=5.0)
    for i in range(2, 16):  # dwell at sign A
        ss_switch[i] = replace(
            ss_switch[i], ego=replace(ss_switch[i].ego, vx=0.0),
            road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=False, evidence_refs=("stop_sign:A",)),
        )
    ss_switch[16] = replace(
        ss_switch[16],
        road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=False, evidence_refs=("stop_sign:B",)),
    )
    for i in range(17, 20):  # re-enter sign A without dwelling
        ss_switch[i] = replace(
            ss_switch[i],
            road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=False, evidence_refs=("stop_sign:A",)),
        )
    ss_switch[20] = replace(
        ss_switch[20],
        road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=True, evidence_refs=("stop_sign:A",)),
    )
    assert status(evaluate(ss_switch), "rules", "stop_sign_violation") == "present"

    # 9. Future predicted conflict at t=0.6s must NOT retroactively qualify a frame-aligned lane crossing at t=0.5s
    future_lane_frames = [
        frame(
            i,
            ego=actor(x=float(i), vx=10.0),
            actors=(
                # t=0.0..0.5: overlap/parallel (no predicted conflict); t=0.6: first predicted conflict!
                actor("a", x=float(i) + 0.5, y=0.0, vx=10.0) if i <= 5
                else actor("a", x=float(i) + 10.0, y=0.0, vx=0.0),
            ),
            available_channels=RULE_CHANNELS,
            road_context=COMPLIANT_ROAD,
            reverse=False,
        )
        for i in range(15)
    ]
    crossing_at_05 = SensorEvidence("run", 5, 0.5, "lane_invasion", "ego", lane_markings=("Broken",))
    res_future_lane = evaluate(future_lane_frames, [crossing_at_05])
    assert status(res_future_lane, "behaviour", "lane_crossing") == "present"
    assert status(res_future_lane, "safety", "unsafe_lane_departure") == "absent"

    # 10. Single-frame road_context dropout immediately before red-light stop-line crossing -> unknown (not present)
    dropout_red = rule_frames(21)
    dropout_red[9] = replace(
        dropout_red[9],
        available_channels=tuple(c for c in RULE_CHANNELS if c != "road_context"),
        road_context=None,
    )
    dropout_red[10] = replace(
        dropout_red[10],
        road_context=replace(COMPLIANT_ROAD, stop_line_applicable=True, stop_line_crossed=True, red_light_state="red", evidence_refs=("tl:9:red",)),
    )
    assert status(evaluate(dropout_red), "rules", "red_light_violation") == "unknown"

    # 11. Single-frame road_context dropout AFTER completed stop-sign dwell must yield unknown (NEVER false present)
    dropout_stop = rule_frames(31, speed=5.0)
    for i in range(5, 18):
        dropout_stop[i] = replace(
            dropout_stop[i],
            ego=replace(dropout_stop[i].ego, vx=0.0),
            road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=False, evidence_refs=("stop_sign:S1",)),
        )
    # Single-frame road_context dropout at i=18 after dwell already completed
    dropout_stop[18] = replace(
        dropout_stop[18],
        available_channels=tuple(c for c in RULE_CHANNELS if c != "road_context"),
        road_context=None,
    )
    # Still in approach at i=19, then crosses at i=20
    dropout_stop[19] = replace(
        dropout_stop[19],
        road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=False, evidence_refs=("stop_sign:S1",)),
    )
    dropout_stop[20] = replace(
        dropout_stop[20],
        road_context=replace(COMPLIANT_ROAD, stop_sign_applicable=True, stop_sign_approach=True, stop_sign_line_crossed=True, evidence_refs=("stop_sign:S1",)),
    )
    assert status(evaluate(dropout_stop), "rules", "stop_sign_violation") == "unknown"
