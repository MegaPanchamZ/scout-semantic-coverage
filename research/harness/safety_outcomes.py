"""Schema-2 assessments. Historical boolean/proxy semantics live in legacy_outcomes."""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Iterable

from research.harness.outcome_config import OutcomeProfile, implementation_digest, load_outcome_profile, profile_digest
from research.harness.outcome_models import (
    CATEGORIES, DEFINITION_VERSION, AssessmentStatus as Status, CaptureManifest,
    FrameEvidence, MetricAssessment, RunOutcome, SensorEvidence,
    assessment_is_present, unknown_categories,
)

SAFETY_OUTCOME_SCHEMA_VERSION = 2
OUTCOME_NAMES = tuple(name for names in CATEGORIES.values() for name in names)


class OutcomeEvaluator:
    """Normalized-evidence evaluator shared by live capture and offline replay.

    Frames arrive in strictly increasing frame/time order. Sensor callbacks may
    arrive out of order before finish() and are reconciled deterministically.
    """

    def __init__(self, profile: OutcomeProfile | None = None):
        self.profile = profile or load_outcome_profile()
        self.frames: list[FrameEvidence] = []
        self.sensors: list[SensorEvidence] = []
        self.run_id: str | None = None

    def _identity(self, run_id):
        if self.run_id is not None and self.run_id != run_id:
            raise ValueError("Cannot mix runs in one evaluator")
        self.run_id = run_id

    def observe(self, frame: FrameEvidence) -> None:
        self._identity(frame.run_id)
        if self.frames and (frame.sim_frame <= self.frames[-1].sim_frame or
                            frame.sim_time_s <= self.frames[-1].sim_time_s):
            raise ValueError("Frame/time order must be strictly increasing")
        self.frames.append(frame)

    def record_sensor_event(self, event: SensorEvidence) -> None:
        self._identity(event.run_id)
        if event.callback_id is not None:
            for old in self.sensors:
                if old.callback_id == event.callback_id and old.channel == event.channel:
                    if old != event:
                        raise ValueError("Conflicting duplicate callback ID")
                    return
        self.sensors.append(event)

    def finish(self, manifest: CaptureManifest) -> RunOutcome:
        from research.harness.outcome_detectors import evaluate_evidence
        self._identity(manifest.run_id)
        frames = []
        phase_break = False
        for f in self.frames:
            if f.phase != "evaluation":
                phase_break = bool(frames)
            else:
                frames.append(replace(f, discontinuity=True) if phase_break else f)
                phase_break = False
        sensors = sorted((s for s in self.sensors if s.phase == "evaluation"),
                         key=lambda s: (s.sim_time_s, s.sim_frame, s.actor_id, s.callback_id or ""))
        for record in (*frames, *sensors):
            if not manifest.start_s <= record.sim_time_s <= manifest.end_s:
                raise ValueError("Evaluation evidence lies outside capture interval")
        if (frames or sensors) and manifest.execution_status in {"dry_run", "not_executed"}:
            raise ValueError("Non-executed run cannot contain evaluation evidence")
        return evaluate_evidence(frames, sensors, manifest, self.profile)


def adapt_legacy_evidence(payload: dict, profile: OutcomeProfile | None = None) -> RunOutcome:
    """Keep old proxy counters as unavailable for v2; preserve explicit contact evidence."""
    profile = profile or load_outcome_profile()
    ticks = payload.get("ticks_executed")
    executed = isinstance(ticks, int) and not isinstance(ticks, bool) and ticks > 0
    error = payload.get("run_error") or payload.get("succeeded") is False
    status = "dry_run" if payload.get("dry_run") else ("partial" if executed and error else
             "failed" if error else "completed" if executed else
             "not_executed" if isinstance(ticks, int) and not isinstance(ticks, bool) and ticks == 0 else "unknown")
    result = RunOutcome(str(payload.get("run_id") or payload.get("scenario_id") or "legacy-unidentified"),
                        status, "legacy_unspecified", profile_digest(profile),
                        unknown_categories("legacy_evidence_insufficient"),
                        provenance={"source_schema": "legacy", "profile_status": profile.status,
                                    "implementation_digest": implementation_digest()})
    records = payload.get("collisions")
    valid_contacts = isinstance(records, list) and any(
        isinstance(c, dict) and isinstance(c.get("frame"), int) and not isinstance(c.get("frame"), bool)
        and c["frame"] >= 0 and c.get("actor_id") is not None and isinstance(c.get("actor_type"), str)
        and (c.get("phase") is None or c.get("phase") == "evaluation")
        for c in records)
    if status != "dry_run" and valid_contacts:
        all_explicit_eval = all(
            c.get("phase") == "evaluation"
            for c in records
            if isinstance(c, dict) and isinstance(c.get("frame"), int) and not isinstance(c.get("frame"), bool)
            and c["frame"] >= 0 and c.get("actor_id") is not None and isinstance(c.get("actor_type"), str)
            and (c.get("phase") is None or c.get("phase") == "evaluation")
        )
        reason = "explicit_legacy_evaluation_collision_record" if all_explicit_eval else "explicit_legacy_collision_record_phase_unresolved"
        result.categories["safety"]["collision"] = MetricAssessment(Status.PRESENT, reasons=(reason,))
    goal = payload.get("reached_goal")
    if executed and status != "dry_run" and isinstance(goal, bool):
        result.categories["task"]["route_incomplete"] = MetricAssessment(
            Status.ABSENT if goal else Status.PRESENT, reasons=("explicit_legacy_goal_state",))
    metrics = payload.get("safety_metrics")
    result.provenance["ignored_proxy_fields"] = sorted(metrics) if isinstance(metrics, dict) else []
    return result


def classify_run(run_payload: dict, profile: OutcomeProfile | None = None, *,
                 evidence: Iterable[FrameEvidence | SensorEvidence] | None = None, schema_version: int = 2) -> dict:
    if schema_version == 1:
        if evidence is not None or profile is not None:
            raise ValueError("Schema 1 uses legacy_outcomes.classify_run for custom thresholds")
        from research.harness.legacy_outcomes import classify_run as legacy
        return legacy(run_payload)
    if schema_version != 2:
        raise ValueError("Unsupported outcome schema")
    if evidence is None:
        if "capture_manifest" in run_payload:
            raise ValueError("Normalized capture requires an explicit evidence iterable (possibly empty)")
        return adapt_legacy_evidence(run_payload, profile).to_dict()
    evaluator = OutcomeEvaluator(profile)
    for record in evidence:
        if isinstance(record, FrameEvidence):
            evaluator.observe(record)
        elif isinstance(record, SensorEvidence):
            evaluator.record_sensor_event(record)
        else:
            raise TypeError("Expected normalized FrameEvidence or SensorEvidence")
    return evaluator.finish(CaptureManifest.from_dict(run_payload["capture_manifest"])).to_dict()


def safety_schema(profile: OutcomeProfile | None = None, *, schema_version: int = 2) -> dict:
    if schema_version == 1:
        if profile is not None:
            raise ValueError("Schema 1 uses legacy_outcomes.safety_schema for custom thresholds")
        from research.harness.legacy_outcomes import safety_schema as legacy_schema
        return legacy_schema()
    if schema_version != 2:
        raise ValueError("Unsupported outcome schema")
    profile = profile or load_outcome_profile()
    return {"schema_version": 2, "definition_version": DEFINITION_VERSION,
            "categories": CATEGORIES, "assessment_states": [s.value for s in Status],
            "profile": asdict(profile), "profile_digest": profile_digest(profile)}
