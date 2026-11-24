"""Contextual task failures from explicit waiting evidence and route arc length."""
from collections import deque
import math

from research.harness.outcome_models import AssessmentStatus as Status, MetricAssessment


def assess_task_outcomes(frames, manifest, profile, emit_event, coverage_complete):
    """Return independent stuck/no_progress assessments; preserve positive windows.

    Windows use actual sample timestamps and never bridge unknown context,
    legitimate waits, phase breaks, capture gaps, or route reacquisition.
    """
    assessments = {}
    duration = manifest.end_s - manifest.start_s
    for label in ("stuck", "no_progress"):
        if label == "no_progress" and not profile.supports("task_window_s", "progress_epsilon_m"):
            assessments[label] = MetricAssessment(reasons=("profile_missing_task_settings",))
            continue
        window_s = profile.value("immobilization_s" if label == "stuck" else "task_window_s")
        channels = ("task_context", "ego_motion") if label == "stuck" else ("task_context",)
        covered = coverage_complete(*channels) and duration + 1e-9 >= window_s
        all_inactive = bool(frames)
        reasons, events, window = set(), [], deque()
        previous, previous_route, episode = None, None, None

        def flush(censored=False):
            nonlocal episode
            if episode is not None:
                events.append(emit_event(label, [episode["ego"]], episode["start"], episode["end"],
                                         "censored" if censored else "closed", episode["frames"],
                                         {"window_s": window_s}, ("explicit_wait_context",)))
                episode = None

        for frame in frames:
            context = frame.task_context
            eligible, reason = context.eligibility() if context is not None and "task_context" in frame.available_channels else ("unknown", "waiting_context_unavailable")
            all_inactive &= context is not None and "task_context" in frame.available_channels and context.objective_active is False
            continuous = not frame.discontinuity and (previous is None or
                frame.sim_time_s - previous.sim_time_s <= profile.value("max_sample_interval_s") + 1e-9)
            route = frame.route_progress
            if label == "no_progress" and eligible == "eligible":
                if route is None or route.status != "available" or "route_progress" not in frame.available_channels:
                    eligible, reason = "unknown", "route_progress_unavailable"
                elif previous_route is not None and (
                        (route.route_id, route.total_m, route.continuity_id) !=
                        (previous_route.route_id, previous_route.total_m, previous_route.continuity_id)
                        or route.high_water_m < previous_route.high_water_m):
                    continuous = False
                    reasons.add("route_continuity_lost")
                channel = manifest.channels.get("route_progress")
                if channel is None or not channel.covers(frame.sim_time_s, frame.sim_time_s):
                    covered = False
                    reasons.add("route_coverage_incomplete")
            if label == "stuck" and eligible == "eligible" and "ego_motion" not in frame.available_channels:
                eligible, reason = "unknown", "motion_unavailable"
            if eligible == "unknown" or not continuous:
                covered = False
                reasons.add(reason if eligible == "unknown" else "capture_or_route_discontinuity")
            if eligible != "eligible" or not continuous:
                flush(censored=eligible == "unknown" or not continuous)
                window.clear()
                previous_route = None
            if eligible == "eligible" and not frame.discontinuity:
                window.append(frame)
                # Keep the last sample at/before the window boundary. The span
                # can exceed the requested window by at most one sample period.
                while len(window) > 1 and frame.sim_time_s - window[1].sim_time_s + 1e-9 >= window_s:
                    window.popleft()
                enough = frame.sim_time_s - window[0].sim_time_s + 1e-9 >= window_s
                qualifies = enough and (
                    all(math.hypot(f.ego.vx, f.ego.vy) < profile.value("stationary_speed_mps") for f in window)
                    if label == "stuck" else
                    frame.route_progress.high_water_m - window[0].route_progress.high_water_m <= profile.value("progress_epsilon_m"))
                if qualifies:
                    if episode is None:
                        episode = {"ego": frame.ego.actor_id, "start": window[0].sim_time_s,
                                   "end": frame.sim_time_s, "frames": [f.sim_frame for f in window]}
                    else:
                        episode["end"] = frame.sim_time_s
                        episode["frames"].append(frame.sim_frame)
                else:
                    flush()
                previous_route = route if label == "no_progress" else None
            previous = frame
        flush(censored=True)
        if events:
            assessments[label] = MetricAssessment(Status.PRESENT, tuple(e.event_id for e in events))
        elif all_inactive and coverage_complete("task_context"):
            assessments[label] = MetricAssessment(Status.NOT_APPLICABLE, reasons=("objective_inactive_throughout",))
        elif covered:
            assessments[label] = MetricAssessment(Status.ABSENT, reasons=("complete_contextual_observation_domain",),
                                                 assessed_duration_s=duration, expected_duration_s=duration)
        else:
            assessments[label] = MetricAssessment(reasons=tuple(sorted(reasons)) or ("insufficient_contextual_observation",),
                                                 expected_duration_s=duration)
    return assessments
