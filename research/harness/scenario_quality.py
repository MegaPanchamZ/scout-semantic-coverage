"""Offline test quality and input identity; none of these changes search deduplication."""
from dataclasses import asdict
import hashlib
import math

from research.harness.interaction_geometry import constant_velocity_ttc, footprint_clearance
from research.harness.outcome_models import AssessmentStatus as Status, MetricAssessment, finite, strict_json


def assess_validity(checks):
    """Invalidity requires an explicit failed feasibility/setup check."""
    if checks is None:
        return MetricAssessment(reasons=("scenario_checks_unavailable",))
    names = ("configuration_feasible", "initial_geometry_feasible", "required_actors_spawned", "route_feasible")
    failures = tuple(name for name in names if getattr(checks, name) is False)
    if failures:
        return MetricAssessment(Status.PRESENT, reasons=failures)
    if all(getattr(checks, name) is True for name in names) and checks.controller_ready is True:
        return MetricAssessment(Status.ABSENT, reasons=("all_declared_feasibility_checks_passed",))
    return MetricAssessment(reasons=("setup_or_controller_evidence_incomplete",))


def interaction_opportunities(frames, profile):
    """Conservative nearby-actor/projection proxy, including safe interactions.

    Separate vertical layers do not form an opportunity. Episodes end at the
    last observed qualifying sample; a gap/disappearance is not resolution.
    """
    if not profile.supports("interaction_radius_m"):
        return []
    active, result = {}, []
    previous = None
    for frame in frames:
        continuous = not frame.discontinuity and (previous is None or
            frame.sim_time_s - previous.sim_time_s <= profile.value("max_sample_interval_s") + 1e-9)
        valid = {"actors", "ego_motion"}.issubset(frame.available_channels) and not frame.discontinuity
        if not continuous or not valid:
            for episode in active.values():
                episode["state"] = "censored"
            active.clear()
        seen = set()
        if valid:
            for actor in frame.actors:
                gap = footprint_clearance(frame.ego, actor)
                prediction = constant_velocity_ttc(frame.ego, actor, profile.value("projection_horizon_s"))
                if gap is None or (gap > profile.value("interaction_radius_m") and prediction.ttc_s is None):
                    continue
                seen.add(actor.actor_id)
                if actor.actor_id not in active:
                    episode = {"actor_id": actor.actor_id, "actor_type": actor.actor_type, "role": actor.role,
                               "start_s": frame.sim_time_s, "end_s": frame.sim_time_s,
                               "evidence_frames": [], "min_gap_m": gap, "min_ttc_s": None,
                               "state": "censored", "criterion": "nearby_or_projected_contact_v1"}
                    active[actor.actor_id] = episode
                    result.append(episode)
                episode = active[actor.actor_id]
                episode["end_s"] = frame.sim_time_s
                episode["evidence_frames"].append(frame.sim_frame)
                episode["min_gap_m"] = min(episode["min_gap_m"], gap)
                if prediction.ttc_s is not None:
                    episode["min_ttc_s"] = min(episode["min_ttc_s"], prediction.ttc_s) if episode["min_ttc_s"] is not None else prediction.ttc_s
        for actor_id in set(active) - seen:
            active[actor_id]["state"] = "closed" if valid and actor_id in {a.actor_id for a in frame.actors} else "censored"
            del active[actor_id]
        previous = frame
    return result


def assess_scenario_quality(frames, manifest, profile, coverage_complete, *, observed_contact=False):
    validity = assess_validity(manifest.scenario_checks)
    opportunities = interaction_opportunities(frames, profile)
    if validity.status != Status.ABSENT:
        low = MetricAssessment(reasons=("scenario_validity_not_established",))
    elif not profile.supports("interaction_radius_m", "quality_min_observation_s"):
        low = MetricAssessment(reasons=("profile_missing_quality_settings",))
    elif opportunities or observed_contact:
        low = MetricAssessment(Status.ABSENT, reasons=("observed_interaction_opportunity",))
    elif (manifest.end_s - manifest.start_s + 1e-9 >= profile.value("quality_min_observation_s")
          and coverage_complete("actors", "ego_motion", "collision", geometry=True)):
        low = MetricAssessment(Status.PRESENT, reasons=("complete_domain_without_nearby_or_projected_interaction",))
    else:
        low = MetricAssessment(reasons=("insufficient_interaction_coverage_or_duration",))
    return {"invalid_scenario": validity, "low_interaction": low}, opportunities


def _canonical(value):
    """Normalize JSON numbers without coercing bools, strings, or nested keys."""
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Identity object keys must be strings")
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        finite(value, "identity value")
        return int(value) if value.is_integer() else value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise ValueError(f"Identity must contain JSON values, got {type(value).__name__}")


def _execution_identity(execution_config):
    if not isinstance(execution_config, dict):
        raise ValueError("Execution identity must be a mapping")
    for name in ("map_digest", "ads_digest", "route_digest"):
        if not isinstance(execution_config.get(name), str) or not execution_config[name]:
            raise ValueError(f"Explicit {name} is required")
    seed = execution_config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Explicit integer seed is required")
    return _canonical(execution_config)


def canonical_scenario_digest(spec, execution_config):
    """Identity of the effective *requested* spec, never proof of realization.

    Callers must resolve defaults and supply effective map/ADS/route identities,
    weather, controller configuration and other execution settings. Only these
    top-level cosmetic fields are removed; nested names remain operational.
    """
    if hasattr(spec, "to_dict"):
        spec = spec.to_dict()
    if not isinstance(spec, dict) or not spec:
        raise ValueError("A nonempty effective scenario mapping is required")
    cosmetic = {"scenario_id", "description", "output_dir", "output_path"}
    data = {"criterion": "effective_requested_input_v1",
            "spec": _canonical({k: v for k, v in spec.items() if k not in cosmetic}),
            "execution": _execution_identity(execution_config)}
    return hashlib.sha256(strict_json(data).encode()).hexdigest()


def realized_initial_state_digest(actors, execution_config):
    """Separate identity of observed initial actors, indexed by stable roles."""
    states, roles = [], set()
    for actor in actors:
        if not isinstance(actor.role, str) or not actor.role or actor.role in roles:
            raise ValueError("Realized identity requires unique nonempty actor roles")
        roles.add(actor.role)
        state = asdict(actor)
        del state["actor_id"]
        states.append(state)
    if not states:
        raise ValueError("No realized initial states")
    data = {"criterion": "realized_actor_state_v1", "actors": sorted(states, key=lambda x: x["role"]),
            "execution": _execution_identity(execution_config)}
    return hashlib.sha256(strict_json(_canonical(data)).encode()).hexdigest()


def parameter_similarity(candidate_a, candidate_b, space):
    """Exact normalized RMS over numeric dimensions; no grid rounding or clipping."""
    names = space.names()
    if set(candidate_a) != set(names) or set(candidate_b) != set(names):
        raise ValueError("Both candidates must specify exactly the declared dimensions")
    squared, dimensions = [], []
    for name in names:
        dimension = space.dimension(name)
        a, b = candidate_a[name], candidate_b[name]
        finite(a, name)
        finite(b, name)
        if not dimension.low <= a <= dimension.high or not dimension.low <= b <= dimension.high:
            raise ValueError(f"Candidate outside bounds for {name}")
        width = dimension.high - dimension.low
        delta = (a - b) / width if width else 0.0
        squared.append(delta**2)
        dimensions.append({"name": name, "kind": dimension.kind, "low": dimension.low, "high": dimension.high,
                           "spec_path": dimension.spec_path})
    if not squared:
        raise ValueError("No dimensions to compare")
    return {"criterion": "numeric_range_normalized_rms_v1", "distance": math.sqrt(sum(squared) / len(squared)),
            "dimensions": dimensions, "interpretation": "input_distance_not_behavioural_equivalence"}
