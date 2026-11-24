"""Private offline detectors shared by live capture and offline replay."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math

from research.harness.interaction_geometry import constant_velocity_ttc, footprint_clearance
from research.harness.outcome_config import implementation_digest, profile_digest, validate_capture_support
from research.harness.task_outcomes import assess_task_outcomes
from research.harness.scenario_quality import assess_scenario_quality
from research.harness.traffic_rule_outcomes import (
    assess_hazard_associated_braking, assess_lane_outcomes, assess_traffic_rules,
)
from research.harness.outcome_models import (
    AssessmentStatus as Status, MetricAssessment, OutcomeEvent, RunOutcome,
    strict_json, unknown_categories,
)


def evaluate_evidence(frames, sensors, manifest, profile):
    p = profile.value
    duration = manifest.end_s - manifest.start_s
    geom_support = validate_capture_support(profile, manifest)
    result = RunOutcome(manifest.run_id, manifest.execution_status, manifest.termination_reason,
                        profile_digest(profile), unknown_categories("insufficient_evidence"),
                        provenance={"profile_status": profile.status, "implementation_digest": implementation_digest(), "capture": asdict(manifest),
                                    "evaluation_mode": "offline", "geometry_support": geom_support})

    def event(label, actors, start, end, state, frame_ids, measurements=None, reasons=()):
        identity = [manifest.run_id, result.profile_digest, label, actors, start, end, frame_ids]
        eid = hashlib.sha256(strict_json(identity).encode()).hexdigest()[:24]
        item = OutcomeEvent(eid, label, tuple(actors), start, end, state, tuple(frame_ids), measurements or {}, tuple(reasons))
        result.events.append(item)
        return item

    def complete(*channels, geometry=False):
        if manifest.execution_status not in {"completed", "partial"} or duration + 1e-9 < p("minimum_observation_s"):
            return False
        if not all(name in manifest.channels and manifest.channels[name].covers(manifest.start_s, manifest.end_s)
                   for name in channels):
            return False
        if geometry and geom_support:
            return False
        sampled = set(channels) - {"collision", "lane_invasion"}
        if sampled:
            if not frames or frames[0].sim_time_s > manifest.start_s + 1e-9 or frames[-1].sim_time_s < manifest.end_s - 1e-9:
                return False
            if any(not sampled.issubset(f.available_channels) or f.discontinuity for f in frames):
                return False
            if any(b.sim_time_s - a.sim_time_s > p("max_sample_interval_s") + 1e-9 for a, b in zip(frames, frames[1:])):
                return False
        return True

    def assess(group, name, events, covered):
        if events:
            assessment = MetricAssessment(Status.PRESENT, tuple(e.event_id for e in events))
        elif covered:
            assessment = MetricAssessment(Status.ABSENT, reasons=("complete_declared_observation_domain",),
                                          assessed_duration_s=duration, expected_duration_s=duration)
        else:
            assessment = MetricAssessment(reasons=("incomplete_or_unavailable_evidence",), expected_duration_s=duration)
        result.categories[group][name] = assessment

    contacts = [s for s in sensors if s.channel == "collision"]
    collisions = []
    for actor in sorted({s.actor_id for s in contacts}):
        groups = []
        for s in (s for s in contacts if s.actor_id == actor):
            if not groups or s.sim_time_s - groups[-1][-1].sim_time_s > p("collision_gap_s"):
                groups.append([])
            groups[-1].append(s)
        for group in groups:
            impulses = [math.sqrt(sum(v * v for v in s.impulse_ns)) for s in group if s.impulse_ns is not None]
            channel = manifest.channels.get("collision")
            closed = (channel is not None and channel.covers(group[-1].sim_time_s, manifest.end_s)
                      and manifest.end_s - group[-1].sim_time_s > p("collision_gap_s"))
            collisions.append(event("collision", [actor], group[0].sim_time_s, group[-1].sim_time_s,
                                    "closed" if closed else "censored", [s.sim_frame for s in group],
                                    {"callback_count": len(group), "peak_impulse_ns": max(impulses) if impulses else None}))
    assess("safety", "collision", collisions, complete("collision"))

    active, encounters, proximity = {}, [], []
    previous = None

    def close(actor, end, closed):
        entry = active.pop(actor)
        entry.update(actor=actor, end=end, closed=closed)
        encounters.append(entry)

    for f in frames:
        valid = {"actors", "ego_motion"}.issubset(f.available_channels) and not f.discontinuity
        gap = previous is not None and f.sim_time_s - previous.sim_time_s > p("max_sample_interval_s") + 1e-9
        if not valid or gap:
            for actor in list(active):
                close(actor, previous.sim_time_s if previous else f.sim_time_s, False)
        if valid:
            visible = {a.actor_id for a in f.actors}
            for actor in list(active):
                if actor not in visible:
                    close(actor, previous.sim_time_s if previous else f.sim_time_s, False)
            for actor in f.actors:
                distance = footprint_clearance(f.ego, actor)
                prediction = constant_velocity_ttc(f.ego, actor, p("projection_horizon_s"))
                already_overlapping = prediction.status == "already_overlapping"
                conflict = (prediction.status == "predicted_contact"
                            and prediction.ttc_s is not None and prediction.ttc_s < p("ttc_s"))
                close_gap = distance is not None and distance < p("proximity_m")
                if close_gap:
                    proximity.append((actor.actor_id, f, distance))
                if (conflict or already_overlapping) and actor.actor_id not in active:
                    active[actor.actor_id] = {
                        "start": f.sim_time_s,
                        "last_active_s": f.sim_time_s,
                        "first_predicted_conflict_s": f.sim_time_s if conflict else None,
                        "last_predicted_conflict_s": f.sim_time_s if conflict else None,
                        "predicted_conflict_times": [f.sim_time_s] if conflict else [],
                        "frames": [],
                        "clear_since": None,
                        "min_ttc_s": prediction.ttc_s if prediction.ttc_s is not None else 0.0,
                        "had_overlap": already_overlapping,
                        "had_predicted_conflict": conflict,
                    }
                if actor.actor_id in active:
                    entry = active[actor.actor_id]
                    entry["frames"].append(f.sim_frame)
                    if already_overlapping:
                        entry["had_overlap"] = True
                    if conflict:
                        entry["had_predicted_conflict"] = True
                        if entry["first_predicted_conflict_s"] is None:
                            entry["first_predicted_conflict_s"] = f.sim_time_s
                        entry["last_predicted_conflict_s"] = f.sim_time_s
                        if not entry["predicted_conflict_times"] or entry["predicted_conflict_times"][-1] != f.sim_time_s:
                            entry["predicted_conflict_times"].append(f.sim_time_s)
                    if prediction.ttc_s is not None:
                        entry["min_ttc_s"] = min(entry["min_ttc_s"], prediction.ttc_s)
                    if conflict or already_overlapping:
                        entry["last_active_s"] = f.sim_time_s
                    if conflict or already_overlapping or close_gap:
                        entry["clear_since"] = None
                    else:
                        if entry["clear_since"] is None:
                            entry["clear_since"] = f.sim_time_s
                        if f.sim_time_s - entry["clear_since"] + 1e-9 >= p("encounter_rearm_s"):
                            close(actor.actor_id, f.sim_time_s, True)
        previous = f
    for actor in list(active):
        close(actor, manifest.end_s, False)

    near_misses, unresolved = [], False
    max_dt = p("max_sample_interval_s")
    for encounter in encounters:
        # Include +max_dt slack on censored encounters so an impact-frame callback
        # arriving right as the contacted actor disappears still suppresses near_miss.
        contact_end = encounter["end"] + (0.0 if encounter["closed"] else max_dt)
        touched = any(
            s.actor_id == encounter["actor"]
            and encounter["start"] - 1e-9 <= s.sim_time_s <= contact_end + 1e-9
            for s in contacts
        )
        coverage = manifest.channels.get("collision")
        # Initial/geometric overlap without sensor confirmation is never confirmed as a clean near miss.
        resolved = (
            encounter["closed"]
            and encounter["had_predicted_conflict"]
            and not encounter["had_overlap"]
            and coverage is not None
            and coverage.covers(encounter["start"], encounter["end"])
        )
        if not touched and resolved:
            near_misses.append(event("near_miss", [encounter["actor"]], encounter["start"], encounter["end"], "closed",
                                     encounter["frames"], {"min_ttc_s": encounter["min_ttc_s"]}))
        else:
            unresolved |= not touched
            reason = (
                "contact_in_encounter" if touched
                else ("geometric_overlap_without_sensor_confirmation" if encounter["had_overlap"] else "unresolved_encounter")
            )
            event("conflict_exposure", [encounter["actor"]], encounter["start"], encounter["end"],
                  "closed" if touched else "censored", encounter["frames"], {"min_ttc_s": encounter["min_ttc_s"]},
                  (reason,))
    assess("safety", "near_miss", near_misses, not unresolved and complete("actors", "ego_motion", "collision", geometry=True))

    # Proximity samples are grouped into consecutive-frame episodes, separately from conflicts.
    prox_events = []
    frame_positions = {f.sim_frame: i for i, f in enumerate(frames)}
    last_frame_idx = len(frames) - 1
    for actor in sorted({actor for actor, _, _ in proximity}):
        groups = []
        for _, f, distance in (x for x in proximity if x[0] == actor):
            if (not groups or frame_positions[f.sim_frame] != frame_positions[groups[-1][-1][0].sim_frame] + 1
                    or f.sim_time_s - groups[-1][-1][0].sim_time_s > max_dt + 1e-9):
                groups.append([])
            groups[-1].append((f, distance))
        for group in groups:
            end_pos = frame_positions[group[-1][0].sim_frame]
            if end_pos == last_frame_idx:
                ep_state = "censored"
            else:
                nxt = frames[end_pos + 1]
                has_gap = nxt.discontinuity or (nxt.sim_time_s - group[-1][0].sim_time_s > max_dt + 1e-9) or not {"actors", "ego_motion"}.issubset(nxt.available_channels)
                ep_state = "censored" if has_gap else "closed"
            prox_events.append(event("close_proximity", [actor], group[0][0].sim_time_s, group[-1][0].sim_time_s,
                                     ep_state, [f.sim_frame for f, _ in group], {"min_gap_m": min(d for _, d in group)}))
    assess("behaviour", "close_proximity", prox_events, complete("actors", "ego_motion", geometry=True))

    braking, stationary = [], []
    col_channel = manifest.channels.get("collision")
    for label, destination in (("hard_braking", braking), ("prolonged_immobilization", stationary)):
        streak, prev = [], None
        used_vel_fallback = False

        def flush(censored=False):
            nonlocal used_vel_fallback
            if not streak:
                return
            threshold = p("braking_duration_s" if label == "hard_braking" else "immobilization_s")
            if streak[-1].sim_time_s - streak[0].sim_time_s + 1e-9 >= threshold:
                if label == "hard_braking":
                    # Per-episode collision coverage is required to exclude impact deceleration
                    # without erasing valid braking episodes on partial or short runs.
                    cov_start = max(manifest.start_s, streak[0].sim_time_s - max_dt)
                    cov_end = streak[-1].sim_time_s
                    # Collision is an event-driven sensor channel tracked in manifest.channels.
                    # A partial collision channel with callback loss/error cannot certify an impact-free streak.
                    loss_prefixes = (
                        "sensor_append_failed",
                        "late_or_unsettled",
                        "unindexed_sensor",
                        "sensor_timestamp_outside",
                        "stop:",
                        "destroy:",
                        "attach:",
                    )
                    col_has_loss = col_channel is not None and any(
                        str(r).startswith(loss_prefixes) for r in (col_channel.reasons or ())
                    )
                    has_col_cov = (
                        col_channel is not None
                        and col_channel.status in {"complete", "partial"}
                        and not col_has_loss
                        and col_channel.start_s is not None
                        and col_channel.end_s is not None
                        and col_channel.start_s <= cov_start + 1e-9
                        and col_channel.end_s >= cov_end - 1e-9
                    )
                    impact = any(
                        streak[0].sim_time_s - max_dt - 1e-9 <= s.sim_time_s <= streak[-1].sim_time_s + 1e-9
                        for s in contacts
                    )
                    if has_col_cov and not impact:
                        meas = {"source": "velocity_difference_fallback" if used_vel_fallback else "longitudinal_acceleration"}
                        reasons = ("velocity_difference_fallback",) if used_vel_fallback else ()
                        destination.append(event(label, [streak[0].ego.actor_id], streak[0].sim_time_s,
                                                 streak[-1].sim_time_s, "censored" if censored else "closed",
                                                 [f.sim_frame for f in streak], meas, reasons))
                else:
                    destination.append(event(label, [streak[0].ego.actor_id], streak[0].sim_time_s,
                                             streak[-1].sim_time_s, "censored" if censored else "closed",
                                             [f.sim_frame for f in streak]))
            streak.clear()
            used_vel_fallback = False

        for f in frames:
            continuous = prev is not None and f.sim_time_s - prev.sim_time_s <= max_dt + 1e-9
            if not continuous or f.discontinuity:
                flush(censored=True)
            valid = "ego_motion" in f.available_channels and not f.discontinuity
            frame_fallback = False
            if label == "hard_braking":
                cos_y, sin_y = math.cos(f.ego.yaw_rad), math.sin(f.ego.yaw_rad)
                forward = f.ego.vx * cos_y + f.ego.vy * sin_y
                long_acc = f.longitudinal_acceleration_mps2
                if (long_acc is None and "acceleration" not in f.available_channels
                        and continuous and prev is not None and not prev.discontinuity
                        and "ego_motion" in prev.available_channels and f.sim_time_s > prev.sim_time_s):
                    # Project world-frame velocity difference onto ONE declared heading (current ego forward)
                    # so yaw rotation alone cannot masquerade as deceleration.
                    prev_fwd_in_prev = prev.ego.vx * math.cos(prev.ego.yaw_rad) + prev.ego.vy * math.sin(prev.ego.yaw_rad)
                    if prev_fwd_in_prev >= 0 and f.reverse is not True and prev.reverse is not True:
                        dt_s = f.sim_time_s - prev.sim_time_s
                        long_acc = ((f.ego.vx - prev.ego.vx) * cos_y + (f.ego.vy - prev.ego.vy) * sin_y) / dt_s
                        frame_fallback = True
                has_accel = ("acceleration" in f.available_channels and f.longitudinal_acceleration_mps2 is not None) or frame_fallback
                valid = (valid and continuous and prev is not None and not prev.discontinuity
                         and "ego_motion" in prev.available_channels and forward >= 0
                         and f.reverse is not True
                         and "controls" in f.available_channels and has_accel
                         and long_acc is not None
                         and -long_acc > p("deceleration_mps2")
                         and f.brake is not None and f.brake > 0
                         and (bool(streak) or math.hypot(prev.ego.vx, prev.ego.vy) > p("braking_speed_mps")))
                if valid and frame_fallback:
                    used_vel_fallback = True
            else:
                valid = valid and math.hypot(f.ego.vx, f.ego.vy) < p("stationary_speed_mps")
            if valid:
                streak.append(f)
            else:
                if label == "hard_braking":
                    # Clean end of fallback braking (brake == 0 with valid motion/controls) is NOT a censor.
                    can_eval_accel = ("acceleration" in f.available_channels and f.longitudinal_acceleration_mps2 is not None) or frame_fallback
                    missing = (
                        "ego_motion" not in f.available_channels
                        or f.discontinuity
                        or "controls" not in f.available_channels
                        or f.brake is None
                        or not can_eval_accel
                    )
                else:
                    missing = "ego_motion" not in f.available_channels or f.discontinuity
                flush(censored=missing)
            prev = f
        flush(censored=True)

    braking_covered = complete("ego_motion", "acceleration", "controls", "collision") and all(
        f.longitudinal_acceleration_mps2 is not None and f.brake is not None for f in frames)
    assess("behaviour", "hard_braking", braking, braking_covered)
    result.categories["safety"]["hazard_associated_braking"] = assess_hazard_associated_braking(
        braking, encounters, prox_events, manifest, profile, event, complete,
        result.categories["behaviour"]["hard_braking"],
    )
    lane_results = assess_lane_outcomes(frames, sensors, encounters, manifest, profile, event, complete)
    result.categories["behaviour"]["lane_crossing"] = lane_results["lane_crossing"]
    result.categories["rules"]["illegal_lane_change"] = lane_results["illegal_lane_change"]
    result.categories["safety"]["unsafe_lane_departure"] = lane_results["unsafe_lane_departure"]
    result.categories["rules"].update(assess_traffic_rules(frames, manifest, profile, event, complete))
    assess("task", "prolonged_immobilization", stationary, duration + 1e-9 >= p("immobilization_s") and complete("ego_motion"))
    if manifest.execution_status in {"completed", "partial"} and manifest.reached_goal is not None:
        result.categories["task"]["route_incomplete"] = MetricAssessment(
            Status.ABSENT if manifest.reached_goal else Status.PRESENT, reasons=("explicit_goal_completion_state",))
    if manifest.completion_deadline_s is None:
        result.categories["task"]["timeout"] = MetricAssessment(
            Status.UNKNOWN, reasons=("no_completion_deadline_declared",), expected_duration_s=duration)
    elif manifest.execution_status in {"completed", "partial"} and manifest.reached_goal is True:
        result.categories["task"]["timeout"] = MetricAssessment(
            Status.ABSENT, reasons=("goal_reached_within_deadline",),
            assessed_duration_s=duration, expected_duration_s=manifest.completion_deadline_s)
    elif (manifest.reached_goal is False and manifest.execution_status in {"completed", "partial"}
          and manifest.termination_reason == "horizon_exhausted" and duration + 1e-9 >= manifest.completion_deadline_s):
        result.categories["task"]["timeout"] = MetricAssessment(
            Status.PRESENT, reasons=("declared_deadline_exhausted",),
            assessed_duration_s=duration, expected_duration_s=manifest.completion_deadline_s)
    else:
        result.categories["task"]["timeout"] = MetricAssessment(
            Status.UNKNOWN, reasons=("terminated_before_deadline_without_goal",),
            expected_duration_s=manifest.completion_deadline_s)
    result.categories["task"].update(assess_task_outcomes(frames, manifest, profile, event, complete))
    quality, opportunities = assess_scenario_quality(frames, manifest, profile, complete, observed_contact=bool(contacts))
    result.categories["quality"].update(quality)
    result.provenance["interaction_opportunities"] = opportunities
    result.events.sort(key=lambda e: (e.start_s, e.label, e.actor_ids, e.event_id))
    return result
