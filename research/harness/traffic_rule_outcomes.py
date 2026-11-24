"""Pure offline detectors for traffic rules, lane legality and hazard-associated braking.

Every detector operates solely on normalized FrameEvidence / SensorEvidence
records without importing CARLA. Missing or ambiguous map/rule context leaves
contextual assessments explicitly unknown rather than fabricating compliance.
"""
from __future__ import annotations

import math

from research.harness.outcome_models import AssessmentStatus as Status, MetricAssessment


PROHIBITED_MARKINGS = frozenset({
    "solid", "doublesolid", "solidsolid", "curb", "grass", "bottsdots", "solidsingle",
})
PERMITTED_MARKINGS = frozenset({"broken", "brokenbroken", "none"})
AMBIGUOUS_MARKINGS = frozenset({"solidbroken", "brokensolid"})
PROHIBITED_LANE_TYPES = frozenset({
    "sidewalk", "shoulder", "median", "offroad", "restricted", "biking", "parking",
})
NON_UNIDIRECTIONAL_LANE_TYPES = frozenset({"bidirectional", "parking", "offroad", "unknown"})
RULE_SETTINGS = (
    "hazard_braking_antecedent_s",
    "speeding_tolerance_mps",
    "speeding_duration_s",
    "stop_sign_speed_mps",
    "stop_sign_dwell_s",
    "wrong_way_speed_mps",
    "wrong_way_duration_s",
)


def _norm_marking(marking: str) -> str:
    return "".join(ch for ch in marking.lower() if ch.isalnum())


def _prev_frame(frames, target_s: float, max_dt: float):
    """Return the nearest strictly earlier frame within max_dt (or exact match only if no earlier frame exists)."""
    strict_prev = None
    exact = None
    for f in frames:
        if f.sim_time_s < target_s - 1e-9 and target_s - f.sim_time_s <= max_dt + 1e-9:
            strict_prev = f
        elif abs(f.sim_time_s - target_s) <= 1e-9:
            exact = f
    return strict_prev if strict_prev is not None else exact


def _next_frame(frames, target_s: float, max_dt: float):
    for f in frames:
        if f.sim_time_s >= target_s - 1e-9 and f.sim_time_s - target_s <= max_dt + 1e-9:
            return f
    return None


def _traffic_light_id(rc) -> str | None:
    if rc is None:
        return None
    for ref in rc.evidence_refs or ():
        if isinstance(ref, str) and ref.startswith("tl:"):
            parts = ref.split(":")
            if len(parts) >= 2 and parts[1]:
                return parts[1]
    return None


def assess_hazard_associated_braking(
    braking_events,
    encounters,
    proximity_events,
    manifest,
    profile,
    emit_event,
    coverage_complete,
    braking_assessment: MetricAssessment,
) -> MetricAssessment:
    """Link commanded hard braking to concurrent or antecedent actor conflict encounters.

    Requires an active TTC conflict encounter (using ``last_active_s`` rather than
    post-conflict rearm tail; passive parallel proximity alone is insufficient).
    Records observational association (actor IDs and lag), never causal fault.
    """
    del proximity_events  # passive proximity alone cannot promote legitimate braking to a safety event
    duration = manifest.end_s - manifest.start_s
    if not profile.supports("hazard_braking_antecedent_s"):
        return MetricAssessment(reasons=("profile_missing_rule_settings",), expected_duration_s=duration)
    antecedent = profile.value("hazard_braking_antecedent_s")
    associated = []
    for brk in braking_events:
        window_start = brk.start_s - antecedent
        matched_actors: set[str] = set()
        lags: list[float] = []
        for enc in encounters:
            if not enc.get("had_predicted_conflict", True):
                continue
            pred_times = enc.get("predicted_conflict_times")
            if pred_times:
                matching_times = [
                    ct for ct in pred_times
                    if window_start - 1e-9 <= ct <= brk.end_s + 1e-9
                ]
                if matching_times:
                    matched_actors.add(enc["actor"])
                    lags.append(max(0.0, brk.start_s - matching_times[0]))
            else:
                first_c = enc.get("first_predicted_conflict_s")
                last_c = enc.get("last_predicted_conflict_s")
                if first_c is not None and last_c is not None:
                    if last_c + 1e-9 >= window_start and first_c <= brk.end_s + 1e-9:
                        matched_actors.add(enc["actor"])
                        lags.append(max(0.0, brk.start_s - first_c))
        if matched_actors:
            ego_id = brk.actor_ids[0] if brk.actor_ids else "ego"
            actors = [ego_id, *sorted(matched_actors)]
            associated.append(
                emit_event(
                    "hazard_associated_braking",
                    actors,
                    brk.start_s,
                    brk.end_s,
                    brk.state,
                    list(brk.evidence_frames),
                    {
                        "braking_event_id": brk.event_id,
                        "association": "antecedent_or_concurrent_actor_hazard",
                        "hazard_kinds": ["conflict_ttc"],
                        "min_lag_s": min(lags) if lags else 0.0,
                        "antecedent_window_s": antecedent,
                    },
                    ("observational_association_not_causal_attribution",),
                )
            )
    if associated:
        return MetricAssessment(Status.PRESENT, tuple(e.event_id for e in associated))
    if braking_assessment.status == Status.ABSENT:
        return MetricAssessment(
            Status.ABSENT,
            reasons=("no_hard_braking_in_complete_domain",),
            assessed_duration_s=duration,
            expected_duration_s=duration,
        )
    if braking_assessment.status == Status.PRESENT and coverage_complete("actors", "ego_motion", "collision", geometry=True):
        return MetricAssessment(
            Status.ABSENT,
            reasons=("hard_braking_without_observed_actor_hazard",),
            assessed_duration_s=duration,
            expected_duration_s=duration,
        )
    return MetricAssessment(reasons=("incomplete_braking_or_actor_hazard_evidence",), expected_duration_s=duration)


def assess_lane_outcomes(
    frames,
    sensors,
    encounters,
    manifest,
    profile,
    emit_event,
    coverage_complete,
) -> dict[str, MetricAssessment]:
    """Separate descriptive lane crossings from illegal changes and unsafe departures."""
    duration = manifest.end_s - manifest.start_s
    max_dt = profile.value("max_sample_interval_s")
    invasions = [s for s in sensors if s.channel == "lane_invasion"]

    groups = []
    for s in invasions:
        if not groups or s.sim_time_s - groups[-1][-1].sim_time_s > max_dt + 1e-9:
            groups.append([])
        groups[-1].append(s)

    crossings = []
    illegal_events = []
    unsafe_events = []
    illegal_uncertain = False
    unsafe_uncertain = False

    for group in groups:
        start_s, end_s = group[0].sim_time_s, group[-1].sim_time_s
        markings = sorted({_norm_marking(m) for s in group for m in s.lane_markings if m})
        ego_id = group[0].actor_id
        crossing_ev = emit_event(
            "lane_crossing",
            [ego_id],
            start_s,
            end_s,
            "closed",
            [s.sim_frame for s in group],
            {"callback_count": len(group), "markings": markings},
        )
        crossings.append(crossing_ev)

        pre_frame = _prev_frame(frames, start_s, max_dt)
        post_frame = _next_frame(frames, end_s, max_dt)
        pre_rc = pre_frame.road_context if (pre_frame and "road_context" in pre_frame.available_channels) else None
        post_rc = post_frame.road_context if (post_frame and "road_context" in post_frame.available_channels) else None

        pre_j = pre_rc.in_junction if pre_rc is not None else None
        post_j = post_rc.in_junction if post_rc is not None else None
        in_junction = (pre_j is True) or (post_j is True)
        # Either side inside a junction excludes illegal lane change; establishing non-junction
        # requires both pre- and post-crossing context to confirm in_junction is False.
        junction_known = in_junction or (pre_j is False and post_j is False)

        has_prohibited = any(m in PROHIBITED_MARKINGS for m in markings)
        has_ambiguous = any(m in AMBIGUOUS_MARKINGS for m in markings)
        has_unrecognized = any(m not in (PROHIBITED_MARKINGS | PERMITTED_MARKINGS | AMBIGUOUS_MARKINGS) for m in markings)
        all_permitted = bool(markings) and all(m in PERMITTED_MARKINGS for m in markings)
        perm = pre_rc.lane_change_permission if pre_rc is not None else None

        if not junction_known:
            illegal_uncertain = True
            unsafe_uncertain = True
        elif in_junction:
            pass  # Junction transitions excluded from illegal lane changes.
        elif has_prohibited or perm == "none":
            illegal_events.append(
                emit_event(
                    "illegal_lane_change",
                    [ego_id],
                    start_s,
                    end_s,
                    "closed",
                    [s.sim_frame for s in group],
                    {"markings": markings, "lane_change_permission": perm},
                    ("prohibited_marking_or_permission",),
                )
            )
        elif not markings or has_ambiguous or has_unrecognized or (not all_permitted and perm in (None, "unknown")):
            illegal_uncertain = True

        # Unsafe lane departure via non-junction crossing overlapping an active actor conflict
        if junction_known and not in_junction:
            overlapping_actors = sorted({
                enc["actor"]
                for enc in encounters
                if enc.get("had_predicted_conflict", True)
                and (
                    any(
                        start_s - max_dt - 1e-9 <= ct <= end_s + 1e-9
                        for ct in (enc.get("predicted_conflict_times") or ())
                    )
                    if enc.get("predicted_conflict_times")
                    else (
                        enc.get("first_predicted_conflict_s") is not None
                        and enc.get("last_predicted_conflict_s") is not None
                        and enc["last_predicted_conflict_s"] + max_dt + 1e-9 >= start_s
                        and enc["first_predicted_conflict_s"] <= end_s + 1e-9
                    )
                )
            })
            if overlapping_actors:
                unsafe_events.append(
                    emit_event(
                        "unsafe_lane_departure",
                        [ego_id, *overlapping_actors],
                        start_s,
                        end_s,
                        "closed",
                        [s.sim_frame for s in group],
                        {"trigger": "crossing_with_conflict", "markings": markings},
                        ("lane_crossing_overlapping_actor_conflict",),
                    )
                )

    # Unsafe lane departure via entering a prohibited road surface
    prohibited_streak = []
    prev = None

    def flush_prohibited(censored=False):
        if not prohibited_streak:
            return
        lane_types = sorted({f.road_context.lane_type for f in prohibited_streak if f.road_context and f.road_context.lane_type})
        unsafe_events.append(
            emit_event(
                "unsafe_lane_departure",
                [prohibited_streak[0].ego.actor_id],
                prohibited_streak[0].sim_time_s,
                prohibited_streak[-1].sim_time_s,
                "censored" if censored else "closed",
                [f.sim_frame for f in prohibited_streak],
                {"trigger": "prohibited_surface", "lane_types": lane_types},
                ("prohibited_road_surface_departure",),
            )
        )
        prohibited_streak.clear()

    road_context_complete = coverage_complete("road_context")
    for f in frames:
        continuous = prev is not None and f.sim_time_s - prev.sim_time_s <= max_dt + 1e-9
        if not continuous or f.discontinuity:
            flush_prohibited(censored=True)
        rc = f.road_context if ("road_context" in f.available_channels and not f.discontinuity) else None
        if rc is None or rc.lane_type in (None, "unknown") or rc.in_junction is None:
            flush_prohibited(censored=True)
            unsafe_uncertain = True
        elif rc.in_junction is False and rc.lane_type in PROHIBITED_LANE_TYPES:
            prohibited_streak.append(f)
        else:
            flush_prohibited(censored=False)
        prev = f
    flush_prohibited(censored=True)

    lane_crossing_assessment = (
        MetricAssessment(Status.PRESENT, tuple(e.event_id for e in crossings))
        if crossings
        else (
            MetricAssessment(Status.ABSENT, reasons=("complete_declared_observation_domain",),
                             assessed_duration_s=duration, expected_duration_s=duration)
            if coverage_complete("lane_invasion")
            else MetricAssessment(reasons=("lane_invasion_channel_unavailable",), expected_duration_s=duration)
        )
    )

    all_frames_have_lane_ctx = bool(frames) and all(
        f.road_context is not None and f.road_context.in_junction is not None
        for f in frames
    )
    if illegal_events:
        illegal_assessment = MetricAssessment(Status.PRESENT, tuple(e.event_id for e in illegal_events))
    elif not illegal_uncertain and coverage_complete("lane_invasion", "road_context") and all_frames_have_lane_ctx:
        illegal_assessment = MetricAssessment(
            Status.ABSENT,
            reasons=("complete_lane_rule_domain",),
            assessed_duration_s=duration,
            expected_duration_s=duration,
        )
    else:
        illegal_assessment = MetricAssessment(
            reasons=("lane_rule_context_incomplete_or_ambiguous",),
            expected_duration_s=duration,
        )

    if unsafe_events:
        unsafe_assessment = MetricAssessment(Status.PRESENT, tuple(e.event_id for e in unsafe_events))
    elif (
        not unsafe_uncertain
        and road_context_complete
        and coverage_complete("lane_invasion", "actors", "ego_motion", geometry=True)
    ):
        unsafe_assessment = MetricAssessment(
            Status.ABSENT,
            reasons=("complete_lane_departure_domain",),
            assessed_duration_s=duration,
            expected_duration_s=duration,
        )
    else:
        unsafe_assessment = MetricAssessment(
            reasons=("lane_departure_evidence_incomplete",),
            expected_duration_s=duration,
        )

    return {
        "lane_crossing": lane_crossing_assessment,
        "illegal_lane_change": illegal_assessment,
        "unsafe_lane_departure": unsafe_assessment,
    }


def _stop_sign_key(rc) -> str:
    for ref in rc.evidence_refs:
        if ref.startswith("stop_sign:"):
            return ref
    return "default_stop_sign"


def assess_traffic_rules(
    frames,
    manifest,
    profile,
    emit_event,
    coverage_complete,
) -> dict[str, MetricAssessment]:
    """Assess red-light, stop-sign, speeding and wrong-way rules from explicit RoadRuleContext."""
    duration = manifest.end_s - manifest.start_s
    if not profile.supports(*RULE_SETTINGS):
        missing = MetricAssessment(reasons=("profile_missing_rule_settings",), expected_duration_s=duration)
        return {
            "red_light_violation": missing,
            "stop_sign_violation": missing,
            "speeding": missing,
            "wrong_way": missing,
        }

    max_dt = profile.value("max_sample_interval_s")
    speed_tol = profile.value("speeding_tolerance_mps")
    speed_dur = profile.value("speeding_duration_s")
    stop_speed = profile.value("stop_sign_speed_mps")
    stop_dwell = profile.value("stop_sign_dwell_s")
    ww_speed = profile.value("wrong_way_speed_mps")
    ww_dur = profile.value("wrong_way_duration_s")

    # 1. Red-light stop-line crossing (rejects phase transitions, first-sample, and post-gap/discontinuity ambiguity)
    red_events = []
    red_uncertain = False
    red_streak = []
    prev = None
    prev_state: str | None = None
    prev_applicable: bool | None = None
    prev_tl_id: str | None = None

    def flush_red():
        if not red_streak:
            return
        red_events.append(
            emit_event(
                "red_light_violation",
                [red_streak[0][0].ego.actor_id],
                red_streak[0][0].sim_time_s,
                red_streak[-1][0].sim_time_s,
                "closed",
                [item[0].sim_frame for item in red_streak],
                {
                    "red_light_state": "red",
                    "prev_frame": red_streak[0][1],
                    "prev_red_light_state": red_streak[0][2],
                },
                ("applicable_stop_line_crossed_on_red",),
            )
        )
        red_streak.clear()

    for f in frames:
        has_continuous_prev = (
            prev is not None
            and not prev.discontinuity
            and not f.discontinuity
            and f.sim_time_s - prev.sim_time_s <= max_dt + 1e-9
        )
        if prev is not None and not has_continuous_prev:
            flush_red()
            red_uncertain = True
            prev_state = None
            prev_applicable = None
            prev_tl_id = None
        rc = f.road_context if ("road_context" in f.available_channels and not f.discontinuity) else None
        if rc is None or rc.stop_line_applicable is None:
            flush_red()
            red_uncertain = True
            prev_state = None
            prev_applicable = None
            prev_tl_id = None
        elif rc.stop_line_applicable is False:
            flush_red()
            prev_state = rc.red_light_state
            prev_applicable = False
            prev_tl_id = _traffic_light_id(rc)
        else:
            curr_tl_id = _traffic_light_id(rc)
            if rc.stop_line_crossed is None or rc.red_light_state in (None, "unknown"):
                flush_red()
                red_uncertain = True
            elif rc.stop_line_crossed is True:
                same_signal = (prev_tl_id is None or curr_tl_id is None or prev_tl_id == curr_tl_id)
                prior_applicable_uncertain = (
                    has_continuous_prev
                    and (
                        prev_applicable is None
                        or (prev_applicable is True and same_signal and prev_state in (None, "unknown"))
                    )
                )
                phase_transition = (
                    has_continuous_prev
                    and prev_applicable is True
                    and same_signal
                    and prev_state is not None
                    and prev_state != rc.red_light_state
                )
                if not has_continuous_prev or phase_transition or prior_applicable_uncertain:
                    flush_red()
                    red_uncertain = True
                elif rc.red_light_state == "red":
                    red_streak.append((f, prev.sim_frame if prev is not None else None, prev_state))
                else:
                    flush_red()
            else:
                flush_red()
            prev_state = rc.red_light_state
            prev_applicable = True
            prev_tl_id = curr_tl_id
        prev = f
    flush_red()

    if red_events:
        red_assessment = MetricAssessment(Status.PRESENT, tuple(e.event_id for e in red_events))
    elif not red_uncertain and coverage_complete("road_context"):
        red_assessment = MetricAssessment(
            Status.ABSENT, reasons=("complete_red_light_rule_domain",),
            assessed_duration_s=duration, expected_duration_s=duration,
        )
    else:
        red_assessment = MetricAssessment(reasons=("red_light_context_incomplete",), expected_duration_s=duration)

    # 2. Stop-sign compliance keyed per sign; tracks per-sign approach entry and preserves crossing_active
    stop_events = []
    stop_uncertain = False
    sign_state: dict[str, dict] = {}
    prev = None
    prev_sign_key: str | None = None
    prev_was_in_approach = False
    prev_stop_context_valid = False

    for f in frames:
        has_continuous_prev = (
            prev is not None
            and not prev.discontinuity
            and not f.discontinuity
            and f.sim_time_s - prev.sim_time_s <= max_dt + 1e-9
        )
        rc = f.road_context if ("road_context" in f.available_channels and not f.discontinuity) else None
        has_motion = "ego_motion" in f.available_channels and not f.discontinuity
        if prev is not None and not has_continuous_prev:
            sign_state.clear()
            prev_sign_key = None
            prev_was_in_approach = False
            prev_stop_context_valid = False
            stop_uncertain = True
        if rc is None or rc.stop_sign_applicable is None or not has_motion:
            sign_state.clear()
            prev_sign_key = None
            prev_was_in_approach = False
            prev_stop_context_valid = False
            stop_uncertain = True
        elif rc.stop_sign_applicable is False:
            sign_state.clear()
            prev_sign_key = None
            prev_was_in_approach = False
            prev_stop_context_valid = True
        else:
            if rc.stop_sign_approach is None or rc.stop_sign_line_crossed is None:
                stop_uncertain = True
                sign_state.clear()
                prev_sign_key = None
                prev_was_in_approach = False
                prev_stop_context_valid = False
            else:
                key = _stop_sign_key(rc)
                # Switching to a different stop sign clears prior sign dwell so dwell at sign A
                # cannot survive a trip through sign B and clear a later crossing at sign A.
                for other_key in list(sign_state):
                    if other_key != key:
                        sign_state.pop(other_key, None)
                st = sign_state.get(key)
                speed = math.hypot(f.ego.vx, f.ego.vy)
                # Entry into THIS sign's approach is observed only if the preceding continuous frame
                # had valid stop-sign context and was not already inside this sign's approach zone.
                entered_cleanly = (
                    has_continuous_prev
                    and prev_stop_context_valid
                    and not (prev_sign_key == key and prev_was_in_approach)
                )
                if rc.stop_sign_approach and not rc.stop_sign_line_crossed:
                    if st is None:
                        st = {
                            "entry_observed": entered_cleanly,
                            "entered_s": f.sim_time_s,
                            "dwell_start_s": None,
                            "complied": False,
                            "crossing_active": False,
                        }
                        sign_state[key] = st
                    elif st["crossing_active"]:
                        # Still in approach after completing a crossing; reset only when leaving approach
                        pass
                    if not st["crossing_active"]:
                        if speed <= stop_speed:
                            if st["dwell_start_s"] is None:
                                st["dwell_start_s"] = f.sim_time_s
                            if f.sim_time_s - st["dwell_start_s"] + 1e-9 >= stop_dwell:
                                st["complied"] = True
                        else:
                            st["dwell_start_s"] = None
                elif rc.stop_sign_line_crossed:
                    # Do NOT accumulate additional dwell on the crossing sample before checking compliance.
                    if st is None:
                        stop_uncertain = True
                        sign_state[key] = {
                            "entry_observed": False,
                            "entered_s": f.sim_time_s,
                            "dwell_start_s": None,
                            "complied": False,
                            "crossing_active": True,
                        }
                    elif not st["crossing_active"]:
                        st["crossing_active"] = True
                        if not st["entry_observed"]:
                            stop_uncertain = True
                        elif not st["complied"]:
                            stop_events.append(
                                emit_event(
                                    "stop_sign_violation",
                                    [f.ego.actor_id],
                                    f.sim_time_s,
                                    f.sim_time_s,
                                    "closed",
                                    [f.sim_frame],
                                    {"crossing_speed_mps": speed, "required_dwell_s": stop_dwell, "stop_sign_id": key},
                                    ("crossed_stop_sign_without_required_approach_dwell",),
                                )
                            )
                else:
                    # Neither approach nor crossing for this sign: clear state cleanly.
                    sign_state.pop(key, None)
                prev_sign_key = key
                prev_was_in_approach = bool(rc.stop_sign_approach)
                prev_stop_context_valid = True
        prev = f

    if stop_events:
        stop_assessment = MetricAssessment(Status.PRESENT, tuple(e.event_id for e in stop_events))
    elif not stop_uncertain and coverage_complete("road_context", "ego_motion"):
        stop_assessment = MetricAssessment(
            Status.ABSENT, reasons=("complete_stop_sign_rule_domain",),
            assessed_duration_s=duration, expected_duration_s=duration,
        )
    else:
        stop_assessment = MetricAssessment(reasons=("stop_sign_context_incomplete",), expected_duration_s=duration)

    # 3. Speeding
    speed_events = []
    speed_uncertain = False
    streak = []
    prev = None

    def flush_speed(censored=False):
        if not streak:
            return
        if streak[-1].sim_time_s - streak[0].sim_time_s + 1e-9 >= speed_dur:
            excesses = [
                math.hypot(x.ego.vx, x.ego.vy) - x.road_context.speed_limit_mps
                for x in streak
            ]
            speed_events.append(
                emit_event(
                    "speeding",
                    [streak[0].ego.actor_id],
                    streak[0].sim_time_s,
                    streak[-1].sim_time_s,
                    "censored" if censored else "closed",
                    [x.sim_frame for x in streak],
                    {"max_excess_mps": max(excesses), "tolerance_mps": speed_tol},
                    ("exceeded_map_speed_limit",),
                )
            )
        streak.clear()

    for f in frames:
        continuous = prev is None or (not f.discontinuity and f.sim_time_s - prev.sim_time_s <= max_dt + 1e-9)
        if not continuous:
            flush_speed(censored=True)
            speed_uncertain = True
        rc = f.road_context if ("road_context" in f.available_channels and not f.discontinuity) else None
        has_motion = "ego_motion" in f.available_channels and not f.discontinuity
        if rc is None or rc.speed_limit_mps is None or not has_motion:
            flush_speed(censored=True)
            speed_uncertain = True
        else:
            speed = math.hypot(f.ego.vx, f.ego.vy)
            if speed > rc.speed_limit_mps + speed_tol:
                streak.append(f)
            else:
                flush_speed(censored=False)
        prev = f
    flush_speed(censored=True)

    if speed_events:
        speed_assessment = MetricAssessment(Status.PRESENT, tuple(e.event_id for e in speed_events))
    elif not speed_uncertain and duration + 1e-9 >= speed_dur and coverage_complete("road_context", "ego_motion"):
        speed_assessment = MetricAssessment(
            Status.ABSENT, reasons=("complete_speed_rule_domain",),
            assessed_duration_s=duration, expected_duration_s=duration,
        )
    else:
        speed_assessment = MetricAssessment(reasons=("speed_limit_context_incomplete",), expected_duration_s=duration)

    # 4. Wrong-way travel (excludes legal reverse, junctions, and non-unidirectional lane types)
    ww_events = []
    ww_uncertain = False
    ww_streak = []
    prev = None

    def flush_ww(censored=False):
        if not ww_streak:
            return
        if ww_streak[-1].sim_time_s - ww_streak[0].sim_time_s + 1e-9 >= ww_dur:
            opposing = [
                -(x.ego.vx * math.cos(x.road_context.permitted_travel_yaw_rad)
                  + x.ego.vy * math.sin(x.road_context.permitted_travel_yaw_rad))
                for x in ww_streak
            ]
            ww_events.append(
                emit_event(
                    "wrong_way",
                    [ww_streak[0].ego.actor_id],
                    ww_streak[0].sim_time_s,
                    ww_streak[-1].sim_time_s,
                    "censored" if censored else "closed",
                    [x.sim_frame for x in ww_streak],
                    {"max_opposing_speed_mps": max(opposing)},
                    ("sustained_opposing_lane_travel",),
                )
            )
        ww_streak.clear()

    for f in frames:
        continuous = prev is None or (not f.discontinuity and f.sim_time_s - prev.sim_time_s <= max_dt + 1e-9)
        if not continuous:
            flush_ww(censored=True)
            ww_uncertain = True
        rc = f.road_context if ("road_context" in f.available_channels and not f.discontinuity) else None
        has_motion = "ego_motion" in f.available_channels and not f.discontinuity
        if rc is None or rc.in_junction is None or f.reverse is None or not has_motion:
            flush_ww(censored=True)
            ww_uncertain = True
        elif rc.in_junction is True or f.reverse is True or rc.lane_type in NON_UNIDIRECTIONAL_LANE_TYPES:
            flush_ww(censored=False)
        elif rc.permitted_travel_yaw_rad is None or rc.lane_type is None:
            flush_ww(censored=True)
            ww_uncertain = True
        else:
            along = (
                f.ego.vx * math.cos(rc.permitted_travel_yaw_rad)
                + f.ego.vy * math.sin(rc.permitted_travel_yaw_rad)
            )
            if -along > ww_speed:
                ww_streak.append(f)
            else:
                flush_ww(censored=False)
        prev = f
    flush_ww(censored=True)

    if ww_events:
        ww_assessment = MetricAssessment(Status.PRESENT, tuple(e.event_id for e in ww_events))
    elif not ww_uncertain and duration + 1e-9 >= ww_dur and coverage_complete("road_context", "ego_motion"):
        ww_assessment = MetricAssessment(
            Status.ABSENT, reasons=("complete_wrong_way_rule_domain",),
            assessed_duration_s=duration, expected_duration_s=duration,
        )
    else:
        ww_assessment = MetricAssessment(reasons=("wrong_way_context_incomplete",), expected_duration_s=duration)

    return {
        "red_light_violation": red_assessment,
        "stop_sign_violation": stop_assessment,
        "speeding": speed_assessment,
        "wrong_way": ww_assessment,
    }
