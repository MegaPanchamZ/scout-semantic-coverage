#!/usr/bin/env python3
"""Matched five-evaluation diagnostic pilot runner with strict admission, policy search, and budget accounting."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Callable

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.criticality import criticality_score
from research.harness.outcome_config import (
    OutcomeProfile,
    implementation_digest,
    load_outcome_profile,
    profile_content_digest,
    profile_digest,
)
from research.harness.outcome_io import atomic_json, file_digest, iter_evidence, load_capture_manifest, loads_strict_json, read_strict_json
from research.harness.outcome_linkage import evaluate_intended_interaction
from research.harness.outcome_models import FrameEvidence, strict_json
from research.harness.safety_outcomes import OutcomeEvaluator
from research.harness.scenario_quality import realized_initial_state_digest
from research.harness.search_space import SearchSpace, make_campaign_space, make_legacy_space


PILOT_DIR = Path(__file__).resolve().parent
POC_DIR = PILOT_DIR.parent / "proof-of-concept"
DEFAULT_PROTOCOL = PILOT_DIR / "protocol.json"
CANONICAL_VALIDATION_CASES = WORKSPACE_ROOT / "research" / "experiments" / "outcome-validation" / "cases.json"

_POLICY_MODULES_CACHE: tuple[Any, Any] | None = None


def _load_policy_modules():
    """Load policy_search and baselines modules from proof-of-concept."""
    global _POLICY_MODULES_CACHE
    if _POLICY_MODULES_CACHE is not None:
        return _POLICY_MODULES_CACHE
    if str(POC_DIR) not in sys.path:
        sys.path.insert(0, str(POC_DIR))
    ps_mod = sys.modules.get("scout_pilot_policy_search")
    if ps_mod is None:
        spec = importlib.util.spec_from_file_location("scout_pilot_policy_search", POC_DIR / "policy_search.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load policy_search.py from {POC_DIR}")
        ps_mod = importlib.util.module_from_spec(spec)
        sys.modules["scout_pilot_policy_search"] = ps_mod
        spec.loader.exec_module(ps_mod)
    import baselines as bl_mod  # type: ignore[import-not-found]
    _POLICY_MODULES_CACHE = (ps_mod, bl_mod)
    return _POLICY_MODULES_CACHE


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    return _load_protocol_with_digest(path)[0]


def _load_protocol_with_digest(path: Path = DEFAULT_PROTOCOL) -> tuple[dict[str, Any], str]:
    """Read the protocol file once; parse and digest the SAME bytes so they cannot diverge."""
    raw_protocol = Path(path).read_bytes()
    protocol_digest = hashlib.sha256(raw_protocol).hexdigest()
    data = loads_strict_json(raw_protocol.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Pilot protocol must be a JSON object")
    budget = data.get("budget_per_arm")
    if not isinstance(budget, dict):
        raise ValueError("Pilot protocol requires a budget_per_arm object")
    if budget.get("auto_extend") is not False:
        raise ValueError("Pilot protocol must disable automatic budget extension")
    if budget.get("failed_attempts_consume_budget") is not True:
        raise ValueError("Pilot protocol must charge failed attempts to the declared budget")
    total_evals = budget.get("total_evaluations")
    if type(total_evals) is not int or total_evals < 2:
        raise ValueError("Pilot budget must be an integer >= 2 (eval 0 plus at least one search evaluation)")
    if "require_collision_free_baseline" in data and not isinstance(data["require_collision_free_baseline"], bool):
        raise ValueError("Pilot protocol require_collision_free_baseline must be a boolean")
    if "required_channels" in data:
        req_channels = data["required_channels"]
        if not isinstance(req_channels, list) or any(not isinstance(c, str) or not c.strip() for c in req_channels):
            raise ValueError("Pilot protocol required_channels must be a list of non-empty channel names")
    if "shared_eval_zero_policy" in data and data["shared_eval_zero_policy"] is not None:
        szp = data["shared_eval_zero_policy"]
        if not isinstance(szp, str) or szp not in {"allow_if_identical_hashes", "never"}:
            raise ValueError(
                f"Unsupported shared_eval_zero_policy {szp!r}"
            )
    ps_mod, _ = _load_policy_modules()
    valid_policies = set(ps_mod.POLICIES)
    arms = data.get("arms")
    if (
        not isinstance(arms, list)
        or not arms
        or any(not isinstance(a, str) or a not in valid_policies for a in arms)
        or len(set(arms)) != len(arms)
    ):
        raise ValueError(
            f"Pilot protocol requires a non-empty list of unique supported policy arms ({sorted(valid_policies)})"
        )
    return data, protocol_digest


@contextmanager
def _exclusive_dir_lock(lock_path: Path):
    """Hold a non-blocking OS flock (per open file description) so concurrent threads or processes cannot race."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another pilot invocation already holds lock {lock_path}") from exc
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _create_exclusive_json(path: Path, payload: dict[str, Any]) -> bool:
    """Atomically create a reservation file only if it does not already exist (POSIX O_CREAT|O_EXCL)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = strict_json(payload).encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(path), flags, 0o600)
    except FileExistsError:
        return False
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def _canonical_cases_digest(canonical_cases_path: Path) -> tuple[dict[str, Any], str]:
    canonical = read_strict_json(Path(canonical_cases_path))
    suite_dir = Path(canonical_cases_path).parent
    spec_hashes: dict[str, str | None] = {}
    for case in canonical.get("cases") or ():
        if not isinstance(case, dict):
            continue
        rel = str(case.get("spec") or "")
        if not rel or rel in spec_hashes:
            continue
        spec_path = suite_dir / rel
        spec_hashes[rel] = file_digest(spec_path) if spec_path.is_file() else None
    digest = hashlib.sha256(strict_json({"suite": canonical, "specs": spec_hashes}).encode("utf-8")).hexdigest()
    return canonical, digest


def verify_validation_report(
    report_path: Path,
    profile: OutcomeProfile,
    *,
    canonical_cases_path: Path = CANONICAL_VALIDATION_CASES,
) -> dict[str, Any]:
    """Verify a full live CARLA validation report against the pinned canonical suite, profile thresholds, and implementation."""
    report = read_strict_json(Path(report_path))
    canonical, expected_cases_digest = _canonical_cases_digest(canonical_cases_path)
    canonical_suite_id = canonical.get("suite_id")
    canonical_cases = [c for c in (canonical.get("cases") or ()) if isinstance(c, dict) and c.get("case_id")]
    canonical_ids = {str(c["case_id"]) for c in canonical_cases}
    canonical_by_id = {str(c["case_id"]): c for c in canonical_cases}
    canonical_expected = {cid: c.get("expected") for cid, c in canonical_by_id.items()}
    canonical_specs: dict[str, dict[str, Any]] = {}
    for cid, c in canonical_by_id.items():
        rel_spec = str(c.get("spec") or "")
        sp = Path(canonical_cases_path).parent / rel_spec if rel_spec else None
        if sp is not None and sp.is_file():
            canonical_specs[cid] = read_strict_json(sp)
    reasons: list[str] = []
    if report.get("mode") != "live":
        reasons.append(f"validation_mode_not_live:{report.get('mode')}")
    if report.get("suite_id") != canonical_suite_id:
        reasons.append(f"validation_suite_id_mismatch:{report.get('suite_id')}")
    if report.get("cases_digest") != expected_cases_digest:
        reasons.append("validation_cases_digest_mismatch")
    if report.get("suite_complete") is not True:
        reasons.append("validation_report_incomplete_suite")
    if report.get("all_passed") is not True:
        reasons.append("validation_report_not_all_passed")
    passed_ids = set(str(cid) for cid in (report.get("passed_case_ids") or ()) if cid)
    if not passed_ids and isinstance(report.get("cases"), list):
        passed_ids = {str(c["case_id"]) for c in report["cases"] if isinstance(c, dict) and c.get("passed") and c.get("case_id")}
    missing_cases = sorted(canonical_ids - passed_ids)
    if missing_cases:
        reasons.append(f"validation_missing_canonical_cases:{','.join(missing_cases[:5])}")
    cases_list = report.get("cases")
    if not isinstance(cases_list, list):
        reasons.append("validation_missing_cases_list")
    else:
        by_id: dict[str, list[dict[str, Any]]] = {}
        for c in cases_list:
            if isinstance(c, dict) and c.get("case_id"):
                by_id.setdefault(str(c["case_id"]), []).append(c)
        seen_manifest_paths: set[Path] = set()
        seen_run_ids: set[str] = set()
        for cid, want_exp in sorted(canonical_expected.items()):
            canonical_case = canonical_by_id[cid]
            entries = by_id.get(cid) or []
            if not entries:
                reasons.append(f"validation_missing_case_record:{cid}")
                continue
            if not all(e.get("mode") == "live" for e in entries):
                reasons.append(f"validation_case_mode_not_live:{cid}")
            if not all(e.get("run_purpose") == "detector_validation" for e in entries):
                reasons.append(f"validation_case_invalid_run_purpose:{cid}")
            if not all(e.get("replay_matches_live") is True for e in entries):
                reasons.append(f"validation_case_replay_mismatch:{cid}")
            if not all(e.get("passed") is True and e.get("expected") == want_exp for e in entries):
                reasons.append(f"validation_case_expectations_mismatch:{cid}")
            for e in entries:
                mp = e.get("manifest_path")
                if not isinstance(mp, str) or not mp.strip():
                    reasons.append(f"validation_case_missing_manifest:{cid}")
                    break
                raw_mp = Path(mp)
                candidates = (
                    [raw_mp]
                    if raw_mp.is_absolute()
                    else [Path(report_path).parent / raw_mp, WORKSPACE_ROOT / raw_mp]
                )
                mpath = next((c for c in candidates if c.is_file()), None)
                if mpath is None:
                    reasons.append(f"validation_case_missing_manifest:{cid}")
                    break
                resolved_mp = mpath.resolve()
                if resolved_mp in seen_manifest_paths:
                    reasons.append(f"validation_case_reused_capture:{cid}")
                    break
                seen_manifest_paths.add(resolved_mp)
                try:
                    cap_m, trace_path = load_capture_manifest(mpath)
                    prov = cap_m.provenance or {}
                    if prov.get("mode") == "offline_synthetic_control" or prov.get("capture_source") != "carla_live_runner":
                        reasons.append(f"validation_case_synthetic_capture:{cid}")
                        break
                    spec_meta = canonical_specs.get(cid) or {}
                    if spec_meta.get("scenario_id") and prov.get("scenario_id") != spec_meta.get("scenario_id"):
                        reasons.append(f"validation_case_scenario_mismatch:{cid}")
                        break
                    if spec_meta.get("town") and prov.get("town") != spec_meta.get("town"):
                        reasons.append(f"validation_case_town_mismatch:{cid}")
                        break
                    if prov.get("agent_kind") != "scripted":
                        reasons.append(f"validation_case_agent_mismatch:{cid}")
                        break
                    if prov.get("case_id") != cid:
                        reasons.append(f"validation_case_id_mismatch:{cid}")
                        break
                    if cap_m.execution_status in {"not_executed", "dry_run"}:
                        reasons.append(f"validation_case_not_executed:{cid}")
                        break
                    if cap_m.run_id in seen_run_ids:
                        reasons.append(f"validation_case_reused_run_id:{cid}")
                        break
                    seen_run_ids.add(cap_m.run_id)
                    ev_records = list(iter_evidence(trace_path))
                    evaluator = OutcomeEvaluator(profile)
                    for rec in ev_records:
                        if isinstance(rec, FrameEvidence):
                            evaluator.observe(rec)
                        else:
                            evaluator.record_sensor_event(rec)
                    replayed = evaluator.finish(cap_m).to_dict()
                except Exception:
                    reasons.append(f"validation_case_unreadable_manifest:{cid}")
                    break
                # Independently verify replayed outcomes, actual trace duration, roles, and exposure from disk.
                cats = replayed.get("categories") or {}
                outcome_bad = False
                for dotted, want_status in (want_exp or {}).items():
                    grp, metric_name = dotted.split(".", 1)
                    got_status = ((cats.get(grp) or {}).get(metric_name) or {}).get("status")
                    if got_status != want_status:
                        reasons.append(f"validation_case_replayed_outcome_mismatch:{cid}:{dotted}:{got_status}!={want_status}")
                        outcome_bad = True
                        break
                if outcome_bad:
                    break
                req_dur = float(canonical_case.get("required_duration_s") or 0.0)
                eval_times = [
                    float(rec.sim_time_s)
                    for rec in ev_records
                    if isinstance(rec, FrameEvidence) and rec.phase == "evaluation"
                ]
                manifest_dur = max(0.0, float(cap_m.end_s) - float(cap_m.start_s))
                trace_dur = (max(eval_times) - min(eval_times)) if len(eval_times) >= 2 else 0.0
                obs_dur = min(manifest_dur, trace_dur)
                expects_early_stop = (
                    canonical_case.get("family") == "collision"
                    or (want_exp or {}).get("quality.invalid_scenario") == "present"
                )
                if req_dur > 0.0 and not expects_early_stop and obs_dur + 1e-6 < req_dur:
                    reasons.append(f"validation_case_insufficient_duration:{cid}:{obs_dur:.3f}<{req_dur:.3f}")
                    break
                observed_roles = {
                    a.role
                    for rec in ev_records
                    if isinstance(rec, FrameEvidence) and rec.phase == "evaluation"
                    for a in (rec.ego, *rec.actors)
                    if a.role
                }
                missing_roles = [r for r in (canonical_case.get("intended_roles") or ()) if r not in observed_roles]
                if missing_roles:
                    reasons.append(f"validation_case_missing_roles:{cid}:{','.join(missing_roles)}")
                    break
                if cid == "red_light_negative_approach_stop":
                    saw_red_approach = any(
                        isinstance(rec, FrameEvidence)
                        and rec.phase == "evaluation"
                        and rec.road_context is not None
                        and rec.road_context.stop_line_applicable is True
                        and rec.road_context.red_light_state == "red"
                        for rec in ev_records
                    )
                    if not saw_red_approach:
                        reasons.append(f"validation_case_missing_red_light_exposure:{cid}")
                        break
    rep_content_dig = report.get("profile_content_digest")
    if rep_content_dig is not None:
        if rep_content_dig != profile_content_digest(profile):
            reasons.append("validation_profile_content_digest_mismatch")
    elif report.get("profile_digest") != profile_digest(profile):
        reasons.append("validation_profile_digest_mismatch")
    if report.get("implementation_digest") != implementation_digest():
        reasons.append("validation_implementation_digest_mismatch")
    return {"verified": not reasons, "reasons": reasons, "report_sha256": file_digest(Path(report_path))}


_UNSET = object()


def validate_initial_configuration(
    baseline_outcome: dict[str, Any],
    protocol: dict[str, Any],
    *,
    profile: OutcomeProfile,
    realized_state_digest: str | None = None,
    intended_witnessed: bool | None = None,
    family: dict[str, Any] | None = None,
    row_execution_status: Any = _UNSET,
    run_error: Any = None,
) -> dict[str, Any]:
    """Verify baseline feasibility, required channels, profile readiness, realized identity, and collision-free admission."""
    reasons: list[str] = []
    try:
        profile.require_pilot_ready()
    except ValueError as exc:
        reasons.append(f"profile_not_ready:{exc}")

    def _fmt_reason_val(val: Any) -> str:
        if isinstance(val, str):
            return val
        if val is None:
            return "None"
        return type(val).__name__

    if row_execution_status is not _UNSET and row_execution_status != "completed":
        reasons.append(f"baseline_row_not_completed:{_fmt_reason_val(row_execution_status)}")
    if run_error:
        reasons.append(f"baseline_row_error:{_fmt_reason_val(run_error)}")

    if not isinstance(baseline_outcome, dict):
        reasons.append("invalid_baseline_outcome_type")
        outcome_map: dict[str, Any] = {}
    else:
        outcome_map = baseline_outcome

    sv = outcome_map.get("schema_version")
    if type(sv) is not int or sv != 2:
        reasons.append("non_v2_baseline_outcome")
    expected_prof_digest = profile_digest(profile)
    if outcome_map.get("profile_digest") != expected_prof_digest:
        reasons.append("baseline_profile_digest_mismatch")
    exec_status = outcome_map.get("execution_status")
    if exec_status != "completed":
        reasons.append(f"baseline_not_completed:{_fmt_reason_val(exec_status)}")

    raw_categories = outcome_map.get("categories")
    if isinstance(raw_categories, dict):
        categories = raw_categories
    else:
        categories = {}
        reasons.append("invalid_baseline_categories")

    quality_grp = categories.get("quality")
    if not isinstance(quality_grp, dict):
        quality_grp = {}
        reasons.append("invalid_baseline_quality_category")
    elif any(not isinstance(v, dict) for v in quality_grp.values()):
        reasons.append("invalid_baseline_quality_assessment")
    inv_entry = quality_grp.get("invalid_scenario")
    invalid_status = inv_entry.get("status") if isinstance(inv_entry, dict) else None
    if invalid_status != "absent":
        reasons.append(f"scenario_validity_not_established:{_fmt_reason_val(invalid_status)}")

    if "task" in categories:
        task_grp = categories.get("task")
        if not isinstance(task_grp, dict) or any(not isinstance(v, dict) for v in task_grp.values()):
            reasons.append("invalid_baseline_task_category")
    if "behaviour" in categories:
        beh_grp = categories.get("behaviour")
        if not isinstance(beh_grp, dict) or any(not isinstance(v, dict) for v in beh_grp.values()):
            reasons.append("invalid_baseline_behaviour_category")

    safety_grp = categories.get("safety")
    if not isinstance(safety_grp, dict):
        safety_grp = {}
        reasons.append("invalid_baseline_safety_category")
    elif any(not isinstance(v, dict) for v in safety_grp.values()):
        reasons.append("invalid_baseline_safety_assessment")

    rules_grp = categories.get("rules")
    if not isinstance(rules_grp, dict):
        reasons.append("invalid_baseline_rules_category")
    elif any(not isinstance(v, dict) for v in rules_grp.values()):
        reasons.append("invalid_baseline_rules_assessment")

    req_collision_free = protocol.get("require_collision_free_baseline", True)
    if not isinstance(req_collision_free, bool):
        reasons.append("invalid_protocol_require_collision_free_baseline")
    elif req_collision_free:
        col_entry = safety_grp.get("collision")
        col_status = col_entry.get("status") if isinstance(col_entry, dict) else None
        if col_status != "absent":
            reasons.append(f"baseline_not_verified_collision_free:{_fmt_reason_val(col_status)}")

    prov = outcome_map.get("provenance")
    capture = prov.get("capture") if isinstance(prov, dict) else None
    channels = capture.get("channels") if isinstance(capture, dict) else None
    channels_map = channels if isinstance(channels, dict) else {}
    if "required_channels" in protocol:
        req_channels_val = protocol["required_channels"]
        if not isinstance(req_channels_val, list) or any(
            not isinstance(c, str) or not c.strip() for c in req_channels_val
        ):
            reasons.append("invalid_protocol_required_channels")
            req_channels_iter: tuple[str, ...] = ()
        else:
            req_channels_iter = tuple(req_channels_val)
    else:
        req_channels_iter = ()
    for ch in req_channels_iter:
        ch_info = channels_map.get(ch)
        ch_status = ch_info.get("status") if isinstance(ch_info, dict) else None
        if ch_status != "complete":
            reasons.append(f"required_channel_incomplete:{ch}")

    if not isinstance(realized_state_digest, str) or not realized_state_digest.strip():
        reasons.append("missing_realized_state_digest")

    if intended_witnessed is not None and not isinstance(intended_witnessed, bool):
        reasons.append(f"invalid_intended_witnessed:{type(intended_witnessed).__name__}")
    requires_witness = bool(
        family and (family.get("intended_role") or family.get("expected_predicates"))
    )
    if requires_witness:
        if intended_witnessed is not True:
            reasons.append("intended_interaction_not_witnessed")
    elif intended_witnessed is False:
        reasons.append("intended_interaction_not_witnessed")

    return {
        "admitted": not reasons,
        "reasons": reasons,
        "profile_digest": profile_digest(profile),
        "realized_state_digest": realized_state_digest,
        "intended_witnessed": intended_witnessed,
    }


def _protocol_manifest_digest(
    protocol: dict[str, Any],
    profile: OutcomeProfile,
    family_id: str,
    seed: int,
    *,
    spec_digest: str | None = None,
    agent_config_digest: str | None = None,
    realized_state_digest: str | None = None,
    shared_eval_zero_hash: str | None = None,
    search_space_signature: str | None = None,
) -> str:
    payload = {
        "protocol_id": protocol.get("protocol_id"),
        "family_id": family_id,
        "seed": int(seed),
        "arms": list(protocol["arms"]),
        "budget_per_arm": protocol["budget_per_arm"],
        "shared_eval_zero_policy": protocol.get("shared_eval_zero_policy"),
        "profile_digest": profile_digest(profile),
        "implementation_digest": implementation_digest(),
        "spec_digest": spec_digest,
        "agent_config_digest": agent_config_digest,
        "realized_state_digest": realized_state_digest,
        "shared_eval_zero_hash": shared_eval_zero_hash,
    }
    if search_space_signature is not None:
        payload["search_space_signature"] = search_space_signature
    return hashlib.sha256(strict_json(payload).encode()).hexdigest()


def build_matched_arms(
    protocol: dict[str, Any],
    profile: OutcomeProfile,
    family_id: str,
    seed: int,
    *,
    shared_eval_zero: dict[str, Any] | None = None,
    spec_digest: str | None = None,
    agent_config_digest: str | None = None,
    realized_state_digest: str | None = None,
    search_space: SearchSpace | None = None,
) -> dict[str, Any]:
    """Construct immutable matched arm plans for one (family, seed) group."""
    profile.require_pilot_ready()
    budget = protocol.get("budget_per_arm") if isinstance(protocol, dict) else None
    if not isinstance(budget, dict):
        raise ValueError("Protocol requires a budget_per_arm object")
    total_budget = budget.get("total_evaluations")
    if type(total_budget) is not int or total_budget < 2:
        raise ValueError("Protocol budget_per_arm.total_evaluations must be an integer >= 2")
    ps_mod, _ = _load_policy_modules()
    valid_policies = set(ps_mod.POLICIES)
    raw_arms = protocol.get("arms")
    if (
        not isinstance(raw_arms, list)
        or not raw_arms
        or any(not isinstance(a, str) or a not in valid_policies for a in raw_arms)
        or len(set(raw_arms)) != len(raw_arms)
    ):
        raise ValueError(
            f"Protocol arms must be a non-empty list of unique supported policies ({sorted(valid_policies)})"
        )
    shared_hash: str | None = None
    if shared_eval_zero is not None:
        if not isinstance(shared_eval_zero, dict):
            raise ValueError("shared_eval_zero must be a dict with a non-empty evidence_sha256")
        raw_hash = shared_eval_zero.get("evidence_sha256")
        if not isinstance(raw_hash, str) or not raw_hash.strip():
            raise ValueError("shared_eval_zero must provide a non-empty string evidence_sha256")
        shared_hash = raw_hash
    space_sig = search_space.signature() if search_space is not None else None
    manifest_digest = _protocol_manifest_digest(
        protocol,
        profile,
        family_id,
        seed,
        spec_digest=spec_digest,
        agent_config_digest=agent_config_digest,
        realized_state_digest=realized_state_digest,
        shared_eval_zero_hash=shared_hash,
        search_space_signature=space_sig,
    )
    prof_dig = profile_digest(profile)
    arms = []
    for arm_name in raw_arms:
        arm_entry: dict[str, Any] = {
            "arm": arm_name,
            "family_id": family_id,
            "seed": int(seed),
            "total_budget": total_budget,
            "profile_digest": prof_dig,
            "shared_eval_zero_hash": shared_hash,
            "spec_digest": spec_digest,
            "agent_config_digest": agent_config_digest,
            "realized_state_digest": realized_state_digest,
            "manifest_digest": manifest_digest,
        }
        if space_sig is not None:
            arm_entry["search_space_signature"] = space_sig
        arms.append(arm_entry)
    plan_out: dict[str, Any] = {
        "manifest_digest": manifest_digest,
        "family_id": family_id,
        "seed": int(seed),
        "profile_digest": prof_dig,
        "spec_digest": spec_digest,
        "agent_config_digest": agent_config_digest,
        "realized_state_digest": realized_state_digest,
        "shared_eval_zero": shared_eval_zero is not None,
        "arms": arms,
    }
    if space_sig is not None:
        plan_out["search_space_signature"] = space_sig
    return plan_out


def _confirmed_v2_outcomes(outcome_v2: Any) -> set[str]:
    """Return confirmed schema-2 safety/rule outcome labels from an evaluation outcome."""
    if outcome_v2 is None:
        return set()
    if not isinstance(outcome_v2, dict):
        raise ValueError("Malformed outcome_v2: must be an object or null")
    sv = outcome_v2.get("schema_version")
    if type(sv) is not int or sv != 2:
        raise ValueError(f"Malformed outcome_v2: schema_version must be integer 2, got {sv!r}")
    categories = outcome_v2.get("categories")
    if not isinstance(categories, dict):
        raise ValueError("Malformed outcome_v2: categories must be an object")
    found: set[str] = set()
    for grp in ("safety", "rules"):
        grp_map = categories.get(grp)
        if not isinstance(grp_map, dict):
            raise ValueError(f"Malformed outcome_v2: categories.{grp} must be an object")
        for name, assess in grp_map.items():
            if not isinstance(assess, dict):
                raise ValueError(f"Malformed outcome_v2: categories.{grp}.{name} must be an object")
            if assess.get("status") == "present":
                found.add(f"{grp}.{name}")
    return found


def _attempt_execution_status(attempt: dict[str, Any]) -> str | None:
    st = attempt.get("execution_status")
    if isinstance(st, str) and st.strip():
        return st
    ov2 = attempt.get("outcome_v2")
    if isinstance(ov2, dict):
        st2 = ov2.get("execution_status")
        if isinstance(st2, str) and st2.strip():
            return st2
    return None


def record_budget(attempts: list[dict[str, Any]], *, shared_eval_zero: bool) -> dict[str, Any]:
    """Distinguish logical arm evaluations from physical simulation calls and charge failed attempts."""
    n_logical = len(attempts)
    n_physical = sum(1 for a in attempts if not a.get("reused_shared_eval_zero"))
    n_failed = sum(1 for a in attempts if _attempt_execution_status(a) == "failed" or a.get("run_error"))
    n_search_attempts = sum(1 for a in attempts if (a.get("eval_index") or 0) > 0)
    n_discoveries = sum(1 for a in attempts if a.get("discovered_new"))
    return {
        "logical_evaluations": n_logical,
        "physical_simulation_calls": n_physical,
        "failed_attempts_charged": n_failed,
        "search_evaluations": n_search_attempts,
        "discoveries_credited": n_discoveries,
        "shared_eval_zero_reused": shared_eval_zero,
    }


def _candidate_dispatch_mode(evaluator_fn: Callable) -> str:
    """Return how ``evaluator_fn`` accepts a candidate argument: 'positional', 'keyword', or 'none'."""
    try:
        sig = inspect.signature(evaluator_fn)
    except (TypeError, ValueError):
        return "none"
    params = list(sig.parameters.values())
    positional = [
        p for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(p.kind == inspect.Parameter.KEYWORD_ONLY and p.name == "candidate" for p in params):
        return "keyword"
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params) or len(positional) >= 3:
        return "positional"
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params) and not any(p.name == "candidate" for p in positional):
        return "keyword"
    return "none"


def _accepts_candidate_arg(evaluator_fn: Callable) -> bool:
    return _candidate_dispatch_mode(evaluator_fn) != "none"


def _default_pilot_space(base_spec: dict[str, Any] | None, seed: int) -> SearchSpace:
    """Build a compatible SearchSpace mapped to parameters the family controller actually consumes."""
    ctrl_params = (base_spec or {}).get("controller_params") or {}
    controller = str((base_spec or {}).get("controller") or "")
    if isinstance(ctrl_params, dict) and "trigger_radius_m" in ctrl_params:
        return make_legacy_space(seed=seed)
    if controller == "adjacent_vehicle" or (isinstance(ctrl_params, dict) and "throttle" in ctrl_params):
        return make_legacy_space(
            spec_path="controller_params.throttle",
            low=0.1,
            high=0.7,
            decimals=2,
            sigma=0.1,
            seed=seed,
        )
    return make_campaign_space(seed=seed)


def run_arm(
    arm_plan: dict[str, Any],
    evaluator_fn: Callable,
    *,
    shared_eval_zero_row: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    search_space: SearchSpace | None = None,
    base_spec: dict[str, Any] | None = None,
    baseline_admission_context: tuple[dict[str, Any], OutcomeProfile, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Execute one matched arm under its fixed budget with named policy guidance and durable resume."""
    if output_dir is not None:
        with _exclusive_dir_lock(output_dir / ".pilot.lock"):
            return _run_arm_locked(
                arm_plan,
                evaluator_fn,
                shared_eval_zero_row=shared_eval_zero_row,
                output_dir=output_dir,
                search_space=search_space,
                base_spec=base_spec,
                baseline_admission_context=baseline_admission_context,
            )
    return _run_arm_locked(
        arm_plan,
        evaluator_fn,
        shared_eval_zero_row=shared_eval_zero_row,
        output_dir=output_dir,
        search_space=search_space,
        base_spec=base_spec,
        baseline_admission_context=baseline_admission_context,
    )


def _validate_rng_state(rng_state: Any, source: str) -> None:
    if rng_state is None:
        return
    try:
        if not isinstance(rng_state, (list, tuple)) or len(rng_state) != 3:
            raise ValueError("rng_state must be a 3-element sequence")
        ver, internal, gauss = rng_state
        if isinstance(ver, bool) or ver != 3:
            raise ValueError(f"Unsupported rng_state version: {ver!r}")
        if not isinstance(internal, (list, tuple)) or len(internal) != 625:
            raise ValueError("rng_state internal state must be a 625-element sequence of ints")
        if any(
            isinstance(x, bool) or not isinstance(x, int) or x < 0 or x >= (1 << 32)
            for x in internal[:624]
        ):
            raise ValueError("rng_state MT19937 words must be uint32 ints in [0, 2**32)")
        if (internal[0] & 0x80000000) == 0 and not any(internal[1:624]):
            raise ValueError("rng_state MT19937 19937-bit state vector cannot be all zero")
        mt_index = internal[624]
        if isinstance(mt_index, bool) or not isinstance(mt_index, int) or not (0 <= mt_index <= 624):
            raise ValueError(f"rng_state MT19937 index out of range: {mt_index!r}")
        if gauss is not None:
            if isinstance(gauss, bool) or not isinstance(gauss, (int, float)):
                raise ValueError(f"Invalid gauss_next cache value: {gauss!r}")
            gauss_val = float(gauss)
            if not math.isfinite(gauss_val):
                raise ValueError(f"Non-finite gauss_next cache value: {gauss!r}")
        else:
            gauss_val = None
        probe = random.Random(0)
        probe.setstate((ver, tuple(internal), gauss_val))
        probe.gauss(0.0, 1.0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Corrupt rng_state in {source}: {exc}") from exc


def _is_strict_json_object_prefix(text: str, *, require_open_string: bool = False) -> bool:
    """Exact RFC 8259 recognizer for strict prefixes of a valid finite JSON object (``{...}``).

    When ``require_open_string=True`` (used for EOF-truncated multi-byte UTF-8 sequences),
    returns True only when ``text`` ends inside an unescaped open JSON string literal.
    """
    n = len(text)
    i = 0

    def skip_ws() -> None:
        nonlocal i
        while i < n and text[i] in " \t\r":
            i += 1

    def parse_string() -> str:
        nonlocal i
        i += 1  # consume opening '"'
        while i < n:
            ch = text[i]
            if ch == '"':
                i += 1
                return "closed"
            if ch == "\\":
                i += 1
                if i == n:
                    return "eof_escape"
                esc = text[i]
                if esc in '"\\/bfnrt':
                    i += 1
                elif esc == "u":
                    i += 1
                    for _ in range(4):
                        if i == n:
                            return "eof_escape"
                        if text[i] not in "0123456789abcdefABCDEF":
                            return "invalid"
                        i += 1
                else:
                    return "invalid"
            elif ord(ch) < 0x20:
                return "invalid"
            else:
                i += 1
        return "eof_open_string"

    def parse_number() -> str:
        nonlocal i
        start = i
        is_float = False
        neg_exp = False
        if text[i] == "-":
            i += 1
            if i == n:
                return "eof"
        if text[i] == "0":
            i += 1
            if i == n:
                return "eof"
        elif "1" <= text[i] <= "9":
            while i < n and "0" <= text[i] <= "9":
                i += 1
            if i == n:
                return "eof"
        else:
            return "invalid"
        if i < n and text[i] == ".":
            is_float = True
            i += 1
            if i == n:
                return "eof"
            if not ("0" <= text[i] <= "9"):
                return "invalid"
            while i < n and "0" <= text[i] <= "9":
                i += 1
            if i == n:
                return "eof"
        if i < n and text[i] in "eE":
            is_float = True
            mantissa_end = i
            i += 1
            if i == n:
                return "eof"
            if text[i] in "+-":
                neg_exp = text[i] == "-"
                if not neg_exp:
                    # With an explicit '+' exponent, no suffix can reduce an already-infinite mantissa.
                    try:
                        mval = float(text[start:mantissa_end])
                    except OverflowError:
                        return "invalid"
                    if not math.isfinite(mval):
                        return "invalid"
                i += 1
                if i == n:
                    return "eof"
            if not ("0" <= text[i] <= "9"):
                return "invalid"
            while i < n and "0" <= text[i] <= "9":
                i += 1
            if i == n:
                # A negative exponent at EOF (e.g., <10**309>e-0) can still become finite with more digits;
                # a non-negative exponent that is already infinite can never become finite.
                if not neg_exp:
                    try:
                        val = float(text[start:i])
                    except OverflowError:
                        return "invalid"
                    if not math.isfinite(val):
                        return "invalid"
                return "eof"
        if is_float:
            try:
                val = float(text[start:i])
            except OverflowError:
                return "invalid"
            if not math.isfinite(val):
                return "invalid"
        else:
            max_int_digits = sys.get_int_max_str_digits() if hasattr(sys, "get_int_max_str_digits") else 0
            digit_count = i - start - (1 if text[start] == "-" else 0)
            if max_int_digits > 0 and digit_count > max_int_digits:
                return "invalid"
        return "closed"

    skip_ws()
    if i == n or text[i] != "{":
        return False
    i += 1
    stack = ["{"]
    key_stack: list[set[str]] = [set()]
    state = "OBJ_KEY_OR_END"

    while True:
        skip_ws()
        if i == n:
            return not require_open_string
        if state in ("OBJ_KEY_OR_END", "OBJ_KEY"):
            if state == "OBJ_KEY_OR_END" and text[i] == "}":
                i += 1
                stack.pop()
                key_stack.pop()
                if not stack:
                    return False  # complete root object is not a strict prefix
                state = "AFTER_VALUE"
                continue
            if text[i] != '"':
                return False
            key_start = i
            res = parse_string()
            if res == "eof_open_string":
                return True
            if res == "eof_escape":
                return not require_open_string
            if res != "closed":
                return False
            try:
                decoded_key = json.loads(text[key_start:i])
            except (json.JSONDecodeError, ValueError):
                return False
            if not isinstance(decoded_key, str) or decoded_key in key_stack[-1]:
                return False
            key_stack[-1].add(decoded_key)
            state = "OBJ_COLON"
            continue
        if state == "OBJ_COLON":
            if text[i] != ":":
                return False
            i += 1
            state = "VALUE"
            continue
        if state == "ARR_VAL_OR_END":
            if text[i] == "]":
                i += 1
                stack.pop()
                state = "AFTER_VALUE"
                continue
            state = "VALUE"
            continue
        if state == "VALUE":
            ch = text[i]
            if ch == "{":
                if len(stack) >= 256:
                    return False
                i += 1
                stack.append("{")
                key_stack.append(set())
                state = "OBJ_KEY_OR_END"
            elif ch == "[":
                if len(stack) >= 256:
                    return False
                i += 1
                stack.append("[")
                state = "ARR_VAL_OR_END"
            elif ch == '"':
                res = parse_string()
                if res == "eof_open_string":
                    return True
                if res == "eof_escape":
                    return not require_open_string
                if res != "closed":
                    return False
                state = "AFTER_VALUE"
            elif ch in "tfn":
                kw = {"t": "true", "f": "false", "n": "null"}[ch]
                rem = text[i:]
                if len(rem) < len(kw):
                    return (not require_open_string) and kw.startswith(rem)
                if text.startswith(kw, i):
                    i += len(kw)
                    state = "AFTER_VALUE"
                else:
                    return False
            elif ch == "-" or ("0" <= ch <= "9"):
                res = parse_number()
                if res == "eof":
                    return not require_open_string
                if res != "closed":
                    return False
                state = "AFTER_VALUE"
            else:
                return False
            continue
        if state == "AFTER_VALUE":
            if not stack:
                return False
            ch = text[i]
            if stack[-1] == "{":
                if ch == ",":
                    i += 1
                    state = "OBJ_KEY"
                elif ch == "}":
                    i += 1
                    stack.pop()
                    key_stack.pop()
                    if not stack:
                        return False  # root object closed -> not a strict prefix
                else:
                    return False
            else:
                if ch == ",":
                    i += 1
                    state = "VALUE"
                elif ch == "]":
                    i += 1
                    stack.pop()
                else:
                    return False


def _has_existing_arms_state(output_dir: Path) -> bool:
    """Iterative check for arm execution artifacts under ``output_dir / 'arms'`` (cycle- and depth-safe)."""
    arms_root = output_dir / "arms"
    if not arms_root.exists():
        return False
    checkpoint_names = {"pilot_manifest.json", "attempts.jsonl", "inflight_attempt.json"}
    visited: set[tuple[int, int]] = set()
    try:
        root_st = os.stat(arms_root)
        visited.add((root_st.st_dev, root_st.st_ino))
    except OSError:
        return False
    stack: list[str] = [str(arms_root)]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as entries:
                for entry in entries:
                    if entry.name in checkpoint_names:
                        return True
                    try:
                        if not entry.is_dir(follow_symlinks=True):
                            continue
                        st = entry.stat(follow_symlinks=True)
                    except OSError:
                        continue
                    key = (st.st_dev, st.st_ino)
                    if key not in visited:
                        visited.add(key)
                        stack.append(entry.path)
        except OSError:
            continue
    return False


def _is_valid_planned_only_report(rep: Any) -> bool:
    """Return True iff ``rep`` matches the exact schema written by ``--plan-only``."""
    if not isinstance(rep, dict) or set(rep.keys()) != {"status", "validation_check", "plan"}:
        return False
    if rep.get("status") != "planned_only":
        return False
    vc = rep.get("validation_check")
    if vc is not None and (
        not isinstance(vc, dict)
        or set(vc.keys()) != {"verified", "reasons", "report_sha256"}
        or vc.get("verified") is not True
        or vc.get("reasons") != []
        or not isinstance(vc.get("report_sha256"), str)
        or not vc["report_sha256"].strip()
    ):
        return False
    plan = rep.get("plan")
    base_plan_keys = {
        "manifest_digest",
        "family_id",
        "seed",
        "profile_digest",
        "spec_digest",
        "agent_config_digest",
        "realized_state_digest",
        "shared_eval_zero",
        "arms",
    }
    if not isinstance(plan, dict) or set(plan.keys()) not in (base_plan_keys, base_plan_keys | {"search_space_signature"}):
        return False
    if (
        not isinstance(plan.get("manifest_digest"), str)
        or not plan["manifest_digest"].strip()
        or not isinstance(plan.get("family_id"), str)
        or not plan["family_id"].strip()
        or not isinstance(plan.get("profile_digest"), str)
        or not plan["profile_digest"].strip()
        or type(plan.get("seed")) is not int
        or plan.get("shared_eval_zero") is not False
        or plan.get("realized_state_digest") is not None
    ):
        return False
    spec_dig = plan.get("spec_digest")
    if spec_dig is not None and (not isinstance(spec_dig, str) or not spec_dig.strip()):
        return False
    agent_dig = plan.get("agent_config_digest")
    if agent_dig is not None and (not isinstance(agent_dig, str) or not agent_dig.strip()):
        return False
    has_sig = "search_space_signature" in plan
    if has_sig:
        sig = plan.get("search_space_signature")
        if not isinstance(sig, str) or not sig.strip():
            return False
    arms = plan.get("arms")
    if not isinstance(arms, list) or not arms:
        return False
    base_arm_keys = {
        "arm",
        "family_id",
        "seed",
        "total_budget",
        "profile_digest",
        "shared_eval_zero_hash",
        "spec_digest",
        "agent_config_digest",
        "realized_state_digest",
        "manifest_digest",
    }
    expected_arm_keys = base_arm_keys | ({"search_space_signature"} if has_sig else set())
    seen_arms: set[str] = set()
    common_budget: int | None = None
    for arm_entry in arms:
        if not isinstance(arm_entry, dict) or set(arm_entry.keys()) != expected_arm_keys:
            return False
        arm_name = arm_entry.get("arm")
        if not isinstance(arm_name, str) or not arm_name.strip() or arm_name in seen_arms:
            return False
        seen_arms.add(arm_name)
        arm_budget = arm_entry.get("total_budget")
        if type(arm_budget) is not int or arm_budget < 2:
            return False
        if common_budget is None:
            common_budget = arm_budget
        elif arm_budget != common_budget:
            return False
        if (
            not _strict_json_equal(arm_entry.get("family_id"), plan["family_id"])
            or not _strict_json_equal(arm_entry.get("seed"), plan["seed"])
            or not _strict_json_equal(arm_entry.get("profile_digest"), plan["profile_digest"])
            or not _strict_json_equal(arm_entry.get("manifest_digest"), plan["manifest_digest"])
            or arm_entry.get("shared_eval_zero_hash") is not None
            or arm_entry.get("realized_state_digest") is not None
            or not _strict_json_equal(arm_entry.get("spec_digest"), spec_dig)
            or not _strict_json_equal(arm_entry.get("agent_config_digest"), agent_dig)
            or (has_sig and not _strict_json_equal(arm_entry.get("search_space_signature"), plan.get("search_space_signature")))
        ):
            return False
    return True


def _has_existing_execution_state(output_dir: Path, *, ignore_baseline_inflight: bool = False) -> bool:
    """True when ``output_dir`` already contains pilot execution artifacts (ignoring valid ``planned_only`` reports)."""
    if (output_dir / "baseline_checkpoint.json").exists() or (
        not ignore_baseline_inflight and (output_dir / "baseline_inflight.json").exists()
    ):
        return True
    if _has_existing_arms_state(output_dir):
        return True
    report_path = output_dir / "pilot_report.json"
    if report_path.exists():
        try:
            rep = read_strict_json(report_path)
            if _is_valid_planned_only_report(rep):
                return False
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
            pass
        return True
    return False


_ARM_ROW_OVERRIDE_KEYS = frozenset({
    "arm",
    "policy",
    "eval_index",
    "candidate",
    "reused_shared_eval_zero",
    "eligible_for_discovery",
    "discovered_new",
    "new_outcomes",
    "credited_as_discovery",
    "feedback_version",
    "fitness",
    "tiebreak",
    "fallback_selection",
    "rng_state",
})


def _strict_json_equal(a: Any, b: Any) -> bool:
    """Iterative type-sensitive deep equality for JSON values (distinguishes int/float/bool)."""
    stack: list[tuple[Any, Any]] = [(a, b)]
    while stack:
        left, right = stack.pop()
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            if left.keys() != right.keys():
                return False
            for k in left:
                stack.append((left[k], right[k]))
        elif isinstance(left, list):
            if len(left) != len(right):
                return False
            stack.extend(zip(left, right))
        elif isinstance(left, float):
            if not (math.isfinite(left) and math.isfinite(right) and left == right):
                return False
        elif left != right:
            return False
    return True


def _is_optional_finite_number(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _expected_eval_zero_candidate(
    space: SearchSpace,
    base_spec: dict[str, Any] | None,
    arm_seed: int,
) -> dict[str, Any]:
    cand_obj = space.from_payload(base_spec, fill_missing=True) if base_spec is not None else None
    if cand_obj is None:
        cand_obj = space.sample(random.Random(arm_seed))
    return loads_strict_json(strict_json(space.validate(cand_obj.to_dict()).to_dict()))


def _safe_repr(value: Any) -> str:
    """Format arbitrary values for error messages without tripping int_max_str_digits or RecursionError."""
    try:
        return repr(value)
    except (ValueError, RecursionError):
        return f"<{type(value).__name__}>"


def _snapshot_admission_context(
    baseline_admission_context: Any,
) -> tuple[dict[str, Any], OutcomeProfile, dict[str, Any] | None] | None:
    """Freeze the admission context once so hashing and validation see identical rules.

    Protocol, profile, and family are deep-copied so later mutation of the
    caller's objects (e.g. by an evaluator) cannot change admission rules mid-run.
    """
    if baseline_admission_context is None:
        return None
    if not isinstance(baseline_admission_context, tuple) or len(baseline_admission_context) != 3:
        raise ValueError("baseline_admission_context must be a (protocol, profile, family) tuple")
    proto_ctx, prof_ctx, fam_ctx = baseline_admission_context
    if not isinstance(proto_ctx, dict):
        raise ValueError("baseline_admission_context protocol must be a dict")
    if fam_ctx is not None and not isinstance(fam_ctx, dict):
        raise ValueError("baseline_admission_context family must be a dict or None")
    if not isinstance(prof_ctx, OutcomeProfile):
        raise ValueError("baseline_admission_context profile must be an OutcomeProfile")
    # copy.deepcopy (not a JSON round-trip) preserves types, so e.g. an invalid tuple
    # required_channels stays invalid; the profile is copied too so mutable metadata is frozen.
    return copy.deepcopy((proto_ctx, prof_ctx, fam_ctx))


def _baseline_admission_context_digest(
    baseline_admission_context: tuple[dict[str, Any], OutcomeProfile, dict[str, Any] | None],
) -> str:
    proto_ctx, prof_ctx, fam_ctx = baseline_admission_context
    payload = {
        "require_collision_free_baseline": proto_ctx.get("require_collision_free_baseline", True),
        "required_channels": proto_ctx.get("required_channels", []),
        "profile_digest": profile_digest(prof_ctx),
        "family": fam_ctx,
    }
    return hashlib.sha256(strict_json(payload).encode("utf-8")).hexdigest()


def _expected_arm_manifest(
    arm_plan: dict[str, Any],
    space: SearchSpace,
    *,
    baseline_admission_context: tuple[dict[str, Any], OutcomeProfile, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    space_sig = space.signature()
    if "search_space_signature" in arm_plan:
        if not isinstance(arm_plan["search_space_signature"], str) or arm_plan["search_space_signature"] != space_sig:
            raise ValueError(
                "Incompatible pilot checkpoint manifest: search_space_signature does not match effective SearchSpace"
            )
        manifest = dict(arm_plan)
    else:
        manifest = {**arm_plan, "search_space_signature": space_sig}
    if baseline_admission_context is not None and not arm_plan.get("shared_eval_zero_hash"):
        adm_digest = _baseline_admission_context_digest(baseline_admission_context)
        if "baseline_admission_context_digest" in manifest and manifest["baseline_admission_context_digest"] != adm_digest:
            raise ValueError(
                "Incompatible pilot checkpoint manifest: baseline_admission_context_digest does not match effective admission context"
            )
        manifest["baseline_admission_context_digest"] = adm_digest
    return manifest


def _validate_attempt_row(
    prev: Any,
    *,
    idx_ln: int,
    arm_name: str,
    total_budget: int,
    expected_shared_hash: str | None,
    norm_shared_row: dict[str, Any] | None,
    space: SearchSpace,
    seen_outcomes: set[str],
    ps_mod: Any,
    source: str,
    expected_eval0_candidate: dict[str, Any] | None = None,
    expected_profile_digest: str | None = None,
    expected_realized_state_digest: str | None = None,
    baseline_admission_context: tuple[dict[str, Any], OutcomeProfile, dict[str, Any] | None] | None = None,
    allow_empty_interrupted_digest: bool = False,
) -> tuple[Any, set[str]]:
    """Validate one attempt row's schema, candidate bounds, shared eval-0 match, and derived fields.

    Returns ``(validated_candidate, run_outcomes)``. Mutates nothing in caller state.
    """
    if not isinstance(prev, dict):
        raise ValueError(f"Corrupt checkpoint row at {source}: Attempt row must be a JSON object")
    ev_idx = prev.get("eval_index")
    if isinstance(ev_idx, bool) or not isinstance(ev_idx, int) or ev_idx != idx_ln or ev_idx >= total_budget:
        raise ValueError(
            f"Corrupt checkpoint row at {source}: expected eval_index={idx_ln} (< {total_budget}), got {ev_idx!r}"
        )
    if prev.get("arm") != arm_name or prev.get("policy") != arm_name:
        raise ValueError(f"Corrupt checkpoint row at {source}: arm/policy mismatch for {arm_name!r}")
    cand_dict = prev.get("candidate")
    if not isinstance(cand_dict, dict):
        raise ValueError(f"Corrupt checkpoint row at {source}: missing candidate dict")
    try:
        cand_obj = space.validate(cand_dict)
    except Exception as exc:
        raise ValueError(f"Corrupt checkpoint row at {source}: invalid candidate: {exc}") from exc
    canonical_cand = loads_strict_json(strict_json(cand_obj.to_dict()))
    if not _strict_json_equal(cand_dict, canonical_cand):
        raise ValueError(
            f"Corrupt checkpoint row at {source}: candidate {cand_dict!r} is not canonical (expected {canonical_cand!r})"
        )
    if _attempt_execution_status(prev) is None:
        raise ValueError(f"Corrupt checkpoint row at {source}: missing execution_status")
    expected_reused = (idx_ln == 0) and bool(expected_shared_hash)
    if not isinstance(prev.get("reused_shared_eval_zero"), bool) or prev["reused_shared_eval_zero"] != expected_reused:
        raise ValueError(
            f"Corrupt checkpoint row at {source}: expected reused_shared_eval_zero={expected_reused!r}, "
            f"got {prev.get('reused_shared_eval_zero')!r}"
        )
    if expected_reused and prev.get("evidence_sha256") != expected_shared_hash:
        raise ValueError(
            f"Corrupt checkpoint row at {source}: shared eval-0 evidence_sha256 "
            f"{prev.get('evidence_sha256')!r} != {expected_shared_hash!r}"
        )
    if idx_ln == 0 and expected_eval0_candidate is not None and not _strict_json_equal(prev["candidate"], expected_eval0_candidate):
        raise ValueError(
            f"Corrupt checkpoint row at {source}: eval-0 candidate "
            f"{prev['candidate']!r} != expected {expected_eval0_candidate!r}"
        )
    if expected_reused and norm_shared_row is not None:
        if "candidate" in norm_shared_row and not _strict_json_equal(prev["candidate"], norm_shared_row["candidate"]):
            raise ValueError(
                f"Corrupt checkpoint row at {source}: shared eval-0 candidate "
                f"{prev['candidate']!r} contradicts shared_eval_zero_row candidate {norm_shared_row['candidate']!r}"
            )
        if not _strict_json_equal(prev.get("outcome_v2"), norm_shared_row.get("outcome_v2")):
            raise ValueError(
                f"Corrupt checkpoint row at {source}: shared eval-0 outcome_v2 contradicts shared_eval_zero_row"
            )
        for k, v in norm_shared_row.items():
            if k not in _ARM_ROW_OVERRIDE_KEYS and (k not in prev or not _strict_json_equal(prev[k], v)):
                raise ValueError(
                    f"Corrupt checkpoint row at {source}: shared eval-0 field {k!r} contradicts shared_eval_zero_row"
                )
        for k in prev:
            if k not in norm_shared_row and k not in _ARM_ROW_OVERRIDE_KEYS and k != "criticality":
                raise ValueError(
                    f"Corrupt checkpoint row at {source}: unexpected shared eval-0 field {k!r} not in shared_eval_zero_row"
                )
    if expected_reused:
        if prev.get("realized_state_digest") != expected_realized_state_digest:
            raise ValueError(
                f"Corrupt checkpoint row at {source}: shared eval-0 realized_state_digest "
                f"{_safe_repr(prev.get('realized_state_digest'))} != admitted {expected_realized_state_digest!r}"
            )
        expected_shared_crit = criticality_score(prev, feedback_version="v2_normalized")
        if not _strict_json_equal(prev.get("criticality"), expected_shared_crit):
            raise ValueError(
                f"Corrupt checkpoint row at {source}: shared eval-0 criticality "
                f"{prev.get('criticality')!r} != {expected_shared_crit!r}"
            )
    elif idx_ln == 0:
        row_exec_status = _attempt_execution_status(prev)
        if baseline_admission_context is not None:
            if expected_realized_state_digest is not None and prev.get("realized_state_digest") != expected_realized_state_digest:
                raise ValueError(
                    f"Corrupt checkpoint row at {source}: independent eval-0 realized_state_digest "
                    f"{_safe_repr(prev.get('realized_state_digest'))} != admitted {expected_realized_state_digest!r}"
                )
            proto_ctx, prof_ctx, fam_ctx = baseline_admission_context
            indep_adm = validate_initial_configuration(
                prev.get("outcome_v2") or {},
                proto_ctx,
                profile=prof_ctx,
                realized_state_digest=prev.get("realized_state_digest"),
                intended_witnessed=prev.get("intended_witnessed"),
                family=fam_ctx,
                row_execution_status=(prev["execution_status"] if "execution_status" in prev else _UNSET),
                run_error=prev.get("run_error"),
            )
            if not indep_adm["admitted"]:
                raise ValueError(
                    f"Corrupt checkpoint row at {source}: independent eval-0 failed baseline admission: {indep_adm['reasons']}"
                )
        elif expected_realized_state_digest is not None:
            is_valid_inflight_empty = (
                allow_empty_interrupted_digest
                and row_exec_status == "interrupted"
                and prev.get("outcome_v2") is None
                and prev.get("realized_state_digest") is None
            )
            if not is_valid_inflight_empty and prev.get("realized_state_digest") != expected_realized_state_digest:
                raise ValueError(
                    f"Corrupt checkpoint row at {source}: independent eval-0 realized_state_digest "
                    f"{_safe_repr(prev.get('realized_state_digest'))} != admitted {expected_realized_state_digest!r}"
                )
    if prev.get("feedback_version") != "v2_normalized":
        raise ValueError(
            f"Corrupt checkpoint row at {source}: expected feedback_version='v2_normalized', "
            f"got {prev.get('feedback_version')!r}"
        )
    ov2 = prev.get("outcome_v2")
    if ov2 is not None:
        if not isinstance(ov2, dict):
            raise ValueError(f"Corrupt checkpoint row at {source}: outcome_v2 must be an object or null")
        sv = ov2.get("schema_version")
        if type(sv) is not int or sv != 2:
            raise ValueError(
                f"Corrupt checkpoint row at {source}: outcome_v2 schema_version must be integer 2, got {sv!r}"
            )
        if expected_profile_digest is not None and ov2.get("profile_digest") != expected_profile_digest:
            raise ValueError(
                f"Corrupt checkpoint row at {source}: outcome_v2 profile_digest "
                f"{ov2.get('profile_digest')!r} != {expected_profile_digest!r}"
            )
    if not _is_optional_finite_number(prev.get("fitness")):
        raise ValueError(f"Corrupt checkpoint row at {source}: invalid fitness {prev.get('fitness')!r}")
    if not _is_optional_finite_number(prev.get("tiebreak")):
        raise ValueError(f"Corrupt checkpoint row at {source}: invalid tiebreak {prev.get('tiebreak')!r}")
    if not _is_optional_finite_number(prev.get("criticality")):
        raise ValueError(f"Corrupt checkpoint row at {source}: invalid criticality {prev.get('criticality')!r}")

    try:
        run_outcomes = _confirmed_v2_outcomes(prev.get("outcome_v2"))
    except Exception as exc:
        raise ValueError(f"Corrupt checkpoint row at {source}: malformed outcome_v2: {exc}") from exc
    expected_new_outcomes = [] if idx_ln == 0 else sorted(run_outcomes - seen_outcomes)
    expected_discovered_new = bool(expected_new_outcomes)
    if prev.get("eligible_for_discovery") is not (idx_ln > 0):
        raise ValueError(
            f"Corrupt checkpoint row at {source}: invalid eligible_for_discovery {prev.get('eligible_for_discovery')!r}"
        )
    if not _strict_json_equal(prev.get("new_outcomes"), expected_new_outcomes):
        raise ValueError(
            f"Corrupt checkpoint row at {source}: new_outcomes {prev.get('new_outcomes')!r} != {expected_new_outcomes!r}"
        )
    if prev.get("discovered_new") is not expected_discovered_new:
        raise ValueError(
            f"Corrupt checkpoint row at {source}: discovered_new {prev.get('discovered_new')!r} != {expected_discovered_new!r}"
        )
    if prev.get("credited_as_discovery") is not expected_discovered_new:
        raise ValueError(
            f"Corrupt checkpoint row at {source}: credited_as_discovery "
            f"{prev.get('credited_as_discovery')!r} != {expected_discovered_new!r}"
        )

    try:
        expected_fitness = ps_mod._semantic_fitness(prev, arm_name, engine_active=False, feedback_version="v2_normalized")
        expected_tiebreak = ps_mod._semantic_tiebreak(prev, arm_name, engine_active=False, feedback_version="v2_normalized")
    except Exception as exc:
        raise ValueError(f"Corrupt checkpoint row at {source}: guidance score computation failed: {exc}") from exc
    expected_fallback = bool(expected_fitness is None and arm_name != "random")
    if not _strict_json_equal(prev.get("fitness"), expected_fitness):
        raise ValueError(
            f"Corrupt checkpoint row at {source}: fitness {prev.get('fitness')!r} != expected {expected_fitness!r}"
        )
    if not _strict_json_equal(prev.get("tiebreak"), expected_tiebreak):
        raise ValueError(
            f"Corrupt checkpoint row at {source}: tiebreak {prev.get('tiebreak')!r} != expected {expected_tiebreak!r}"
        )
    if prev.get("fallback_selection") is not expected_fallback:
        raise ValueError(
            f"Corrupt checkpoint row at {source}: fallback_selection {prev.get('fallback_selection')!r} != {expected_fallback!r}"
        )
    if prev.get("rng_state") is None:
        raise ValueError(f"Corrupt checkpoint row at {source}: missing rng_state")
    _validate_rng_state(prev["rng_state"], source)
    return cand_obj, run_outcomes


_INFLIGHT_ATTEMPT_KEYS = frozenset({
    "arm",
    "policy",
    "eval_index",
    "candidate",
    "reused_shared_eval_zero",
    "eligible_for_discovery",
    "execution_status",
    "run_error",
    "outcome_v2",
    "discovered_new",
    "new_outcomes",
    "credited_as_discovery",
    "feedback_version",
    "criticality",
    "fitness",
    "tiebreak",
    "fallback_selection",
    "rng_state",
})


def _validate_arm_plan_inputs(
    arm_plan: Any,
    *,
    baseline_admission_context: tuple[dict[str, Any], OutcomeProfile, dict[str, Any] | None] | None,
) -> tuple[str, int, int, str, str | None, str | None, str | None, Any, Any]:
    """Validate arm plan types, policy name, budget, seed, and digest pins upfront."""
    if not isinstance(arm_plan, dict):
        raise ValueError("Arm plan must be a JSON object")
    ps_mod, bl_mod = _load_policy_modules()
    arm_name = arm_plan.get("arm")
    if not isinstance(arm_name, str) or arm_name not in set(ps_mod.POLICIES):
        raise ValueError(f"Unsupported pilot policy arm {_safe_repr(arm_name)}")
    total_budget = arm_plan.get("total_budget")
    if type(total_budget) is not int or total_budget < 2:
        raise ValueError(
            f"Arm plan for {arm_name!r} has invalid total_budget {_safe_repr(total_budget)}; expected integer >= 2"
        )
    arm_seed = arm_plan.get("seed", 13)
    if type(arm_seed) is not int:
        raise ValueError(f"Arm plan for {arm_name!r} has invalid seed {_safe_repr(arm_seed)}; expected integer")
    manifest_digest = arm_plan.get("manifest_digest")
    if not isinstance(manifest_digest, str) or not manifest_digest.strip():
        raise ValueError(
            f"Arm plan for {arm_name!r} has invalid manifest_digest {_safe_repr(manifest_digest)}"
        )
    expected_shared_hash = arm_plan.get("shared_eval_zero_hash")
    if expected_shared_hash is not None and (
        not isinstance(expected_shared_hash, str) or not expected_shared_hash.strip()
    ):
        raise ValueError(
            f"Arm plan for {arm_name!r} has invalid shared_eval_zero_hash {_safe_repr(expected_shared_hash)}"
        )
    expected_profile_digest = arm_plan.get("profile_digest")
    if expected_profile_digest is not None and (
        not isinstance(expected_profile_digest, str) or not expected_profile_digest.strip()
    ):
        raise ValueError(
            f"Arm plan for {arm_name!r} has invalid profile_digest {_safe_repr(expected_profile_digest)}"
        )
    expected_realized_digest = arm_plan.get("realized_state_digest")
    if expected_realized_digest is not None and (
        not isinstance(expected_realized_digest, str) or not expected_realized_digest.strip()
    ):
        raise ValueError(
            f"Arm plan for {arm_name!r} has invalid realized_state_digest {_safe_repr(expected_realized_digest)}"
        )
    if "baseline_admission_context_digest" in arm_plan:
        raw_adm_dig = arm_plan["baseline_admission_context_digest"]
        if not isinstance(raw_adm_dig, str) or not raw_adm_dig.strip():
            raise ValueError(
                f"Arm plan for {arm_name!r} has invalid baseline_admission_context_digest {_safe_repr(raw_adm_dig)}"
            )
        if baseline_admission_context is None:
            raise ValueError(
                f"Arm plan for {arm_name!r} requires baseline_admission_context matching baseline_admission_context_digest"
            )
        if _baseline_admission_context_digest(baseline_admission_context) != raw_adm_dig:
            raise ValueError(
                f"Arm plan for {arm_name!r} baseline_admission_context_digest does not match effective admission context"
            )
    if baseline_admission_context is not None and not expected_shared_hash and expected_realized_digest is None:
        raise ValueError(
            f"Arm plan for {arm_name!r} with baseline_admission_context requires a pinned realized_state_digest "
            "when shared_eval_zero_hash is unset"
        )
    return (
        arm_name,
        total_budget,
        arm_seed,
        manifest_digest,
        expected_shared_hash,
        expected_profile_digest,
        expected_realized_digest,
        ps_mod,
        bl_mod,
    )


def _validate_arm_checkpoint(
    arm_plan: dict[str, Any],
    output_dir: Path,
    *,
    shared_eval_zero_row: dict[str, Any] | None = None,
    search_space: SearchSpace | None = None,
    base_spec: dict[str, Any] | None = None,
    baseline_admission_context: tuple[dict[str, Any], OutcomeProfile, dict[str, Any] | None] | None = None,
) -> tuple[list[dict[str, Any]], list[str], bool, bytes]:
    """Read-only validation of one arm's checkpoint files before any rewrite or evaluation.

    Returns ``(parsed_attempts, valid_lines, torn_trailing, raw_attempts_bytes)``.
    Raises ``ValueError`` if the manifest is missing alongside arm state or mismatches
    ``arm_plan`` (including effective ``search_space_signature``), any row is malformed or
    contains non-finite JSON constants/overflowing floats (``NaN``/``Infinity``/``1e999``),
    required attempt fields, candidate bounds, derived discovery/guidance fields, or
    sequential ``eval_index`` are invalid, ``rng_state`` is invalid, shared eval-0 diverges
    from ``shared_eval_zero_row``, policy/RNG replay fails, or an inflight reservation is
    unreadable/invalid. Only genuine premature-EOF UTF-8 or JSON-object prefix truncation
    on an unterminated final line is treated as a recoverable torn append.
    """
    baseline_admission_context = _snapshot_admission_context(baseline_admission_context)
    (
        arm_name,
        total_budget,
        arm_seed,
        _manifest_digest,
        expected_shared_hash,
        expected_profile_digest,
        expected_realized_digest,
        ps_mod,
        bl_mod,
    ) = _validate_arm_plan_inputs(arm_plan, baseline_admission_context=baseline_admission_context)
    norm_shared_row: dict[str, Any] | None = None
    if expected_shared_hash and shared_eval_zero_row is not None:
        if (
            not isinstance(shared_eval_zero_row, dict)
            or shared_eval_zero_row.get("evidence_sha256") != expected_shared_hash
        ):
            raise ValueError(
                f"shared_eval_zero_row evidence_sha256 mismatches arm_plan shared_eval_zero_hash {expected_shared_hash!r}"
            )
        if shared_eval_zero_row.get("realized_state_digest") != expected_realized_digest:
            raise ValueError(
                f"shared_eval_zero_row realized_state_digest {_safe_repr(shared_eval_zero_row.get('realized_state_digest'))} "
                f"mismatches arm_plan realized_state_digest {expected_realized_digest!r}"
            )
        norm_shared_row = loads_strict_json(strict_json(shared_eval_zero_row))
    elif not expected_shared_hash and shared_eval_zero_row is not None:
        raise ValueError(
            f"Arm plan for {arm_name!r} does not permit shared_eval_zero_row when shared_eval_zero_hash is unset"
        )

    probe_space = search_space or _default_pilot_space(base_spec, arm_seed)
    expected_manifest = _expected_arm_manifest(
        arm_plan,
        probe_space,
        baseline_admission_context=baseline_admission_context,
    )
    norm_expected_manifest = loads_strict_json(strict_json(expected_manifest))

    manifest_file = output_dir / "pilot_manifest.json"
    attempts_path = output_dir / "attempts.jsonl"
    inflight_path = output_dir / "inflight_attempt.json"
    parsed_attempts: list[dict[str, Any]] = []
    valid_lines: list[str] = []
    torn_trailing = False
    raw_bytes = b""
    if not manifest_file.exists():
        if attempts_path.exists() or inflight_path.exists():
            raise ValueError(
                f"Corrupt arm checkpoint in {output_dir}: found attempts/inflight state without pilot_manifest.json"
            )
        return parsed_attempts, valid_lines, torn_trailing, raw_bytes
    stored = read_strict_json(manifest_file)
    if not isinstance(stored, dict) or not _strict_json_equal(stored, norm_expected_manifest):
        raise ValueError("Incompatible pilot checkpoint manifest; refusing to resume into mismatched directory")
    probe_policy = bl_mod.make_policy_state(arm_name, probe_space, ps_mod.PolicyState, feedback_version="v2_normalized")
    probe_rng = random.Random(arm_seed)
    probe_seen: set[tuple] = set()
    probe_seen_outcomes: set[str] = set()
    expected_eval0_cand = _expected_eval_zero_candidate(probe_space, base_spec, arm_seed)
    if attempts_path.exists():
        raw_bytes = attempts_path.read_bytes()
        raw_segments = raw_bytes.split(b"\n")
        final_record_unterminated = (
            bool(raw_bytes)
            and not raw_bytes.endswith(b"\n")
            and bool(raw_segments[-1].strip(b" \t\r"))
        )
        # Only strip a single trailing CR when the segment was actually LF-terminated (CRLF);
        # never strip a bare CR at EOF of an unterminated final segment.
        raw_lines_bytes: list[bytes] = []
        for seg_idx, seg in enumerate(raw_segments):
            cleaned = seg[:-1] if (seg_idx < len(raw_segments) - 1 and seg.endswith(b"\r")) else seg
            if cleaned.strip(b" \t\r"):
                raw_lines_bytes.append(cleaned)
        for idx_ln, raw_seg in enumerate(raw_lines_bytes):
            is_last_unterminated = idx_ln == len(raw_lines_bytes) - 1 and final_record_unterminated
            try:
                line = raw_seg.decode("utf-8")
            except UnicodeDecodeError as exc:
                if (
                    is_last_unterminated
                    and exc.end == len(raw_seg)
                    and "unexpected end of data" in exc.reason
                    and _is_strict_json_object_prefix(
                        raw_seg[:exc.start].decode("utf-8"),
                        require_open_string=True,
                    )
                ):
                    torn_trailing = True
                    break
                raise ValueError(f"Corrupt checkpoint row at line {idx_ln + 1} in {attempts_path}") from exc
            try:
                prev = loads_strict_json(line)
            except json.JSONDecodeError as exc:
                if is_last_unterminated and _is_strict_json_object_prefix(line):
                    torn_trailing = True
                    break
                raise ValueError(f"Corrupt checkpoint row at line {idx_ln + 1} in {attempts_path}") from exc
            except ValueError as exc:
                # Truncated numeric prefixes (e.g. <10**309>e-0 or <4301 digits>e) can trip parse_float
                # or Python's int_max_str_digits before EOF; _is_strict_json_object_prefix returns True
                # ONLY for genuinely incomplete finite-continuable prefixes (never complete objects).
                if is_last_unterminated and _is_strict_json_object_prefix(line):
                    torn_trailing = True
                    break
                raise ValueError(f"Corrupt checkpoint row at line {idx_ln + 1} in {attempts_path}: {exc}") from exc
            except RecursionError as exc:
                raise ValueError(f"Corrupt checkpoint row at line {idx_ln + 1} in {attempts_path}: {exc}") from exc
            cand_obj, run_outcomes = _validate_attempt_row(
                prev,
                idx_ln=idx_ln,
                arm_name=arm_name,
                total_budget=total_budget,
                expected_shared_hash=expected_shared_hash,
                norm_shared_row=norm_shared_row,
                space=probe_space,
                seen_outcomes=probe_seen_outcomes,
                ps_mod=ps_mod,
                source=f"line {idx_ln + 1} in {attempts_path}",
                expected_eval0_candidate=expected_eval0_cand if idx_ln == 0 else None,
                expected_profile_digest=expected_profile_digest,
                expected_realized_state_digest=expected_realized_digest if idx_ln == 0 else None,
                baseline_admission_context=baseline_admission_context if idx_ln == 0 else None,
            )
            if idx_ln == 0:
                cand0_probe = probe_space.from_payload(base_spec, fill_missing=True) if base_spec is not None else None
                if cand0_probe is None:
                    probe_space.sample(probe_rng)
            else:
                proposed = probe_space.validate(probe_policy.next_candidate(probe_rng, probe_seen).to_dict())
                expected_cand = loads_strict_json(strict_json(proposed.to_dict()))
                if not _strict_json_equal(prev["candidate"], expected_cand):
                    raise ValueError(
                        f"Corrupt checkpoint row at line {idx_ln + 1} in {attempts_path}: "
                        f"candidate {prev['candidate']!r} != expected proposal {expected_cand!r}"
                    )
            expected_rng_state = loads_strict_json(strict_json(probe_rng.getstate()))
            if not _strict_json_equal(prev["rng_state"], expected_rng_state):
                raise ValueError(
                    f"Corrupt checkpoint row at line {idx_ln + 1} in {attempts_path}: "
                    "rng_state contradicts deterministic replay"
                )
            probe_seen_outcomes |= run_outcomes
            probe_seen.add(probe_space.key(cand_obj))
            try:
                probe_policy.observe(
                    prev["candidate"],
                    prev.get("fitness"),
                    prev.get("tiebreak"),
                    row=prev,
                    feedback_version="v2_normalized",
                )
            except Exception as exc:
                raise ValueError(
                    f"Corrupt checkpoint row at line {idx_ln + 1} in {attempts_path}: replay failed: {exc}"
                ) from exc
            valid_lines.append(line)
            parsed_attempts.append(prev)
    pending_inflight: dict[str, Any] | None = None
    if inflight_path.exists():
        inflight = read_strict_json(inflight_path)
        if not isinstance(inflight, dict) or set(inflight.keys()) != _INFLIGHT_ATTEMPT_KEYS:
            raise ValueError(f"Corrupt inflight reservation in {inflight_path}")
        inflight_idx = inflight.get("eval_index")
        if (
            isinstance(inflight_idx, bool)
            or not isinstance(inflight_idx, int)
            or inflight_idx < 0
            or inflight_idx > len(parsed_attempts)
            or inflight_idx >= total_budget
            or (inflight_idx == 0 and bool(expected_shared_hash))
        ):
            raise ValueError(
                f"Corrupt inflight reservation in {inflight_path}: invalid eval_index {inflight_idx!r} "
                f"for {len(parsed_attempts)} recorded attempts (budget={total_budget})"
            )
        if inflight.get("arm") != arm_name or inflight.get("policy") != arm_name:
            raise ValueError(
                f"Corrupt inflight reservation in {inflight_path}: arm/policy mismatch for {arm_name!r}"
            )
        if not isinstance(inflight.get("candidate"), dict):
            raise ValueError(f"Corrupt inflight reservation in {inflight_path}: missing candidate dict")
        try:
            inflight_cand_obj = probe_space.validate(inflight["candidate"])
        except Exception as exc:
            raise ValueError(
                f"Corrupt inflight reservation in {inflight_path}: invalid candidate: {exc}"
            ) from exc
        if inflight.get("execution_status") != "interrupted":
            raise ValueError(
                f"Corrupt inflight reservation in {inflight_path}: expected execution_status='interrupted', "
                f"got {inflight.get('execution_status')!r}"
            )
        run_error = inflight.get("run_error")
        if not isinstance(run_error, str) or not run_error.strip():
            raise ValueError(
                f"Corrupt inflight reservation in {inflight_path}: missing non-empty run_error"
            )
        if inflight.get("reused_shared_eval_zero") is not False:
            raise ValueError(
                f"Corrupt inflight reservation in {inflight_path}: reused_shared_eval_zero must be False"
            )
        if inflight.get("discovered_new") is not False or inflight.get("credited_as_discovery") is not False:
            raise ValueError(
                f"Corrupt inflight reservation in {inflight_path}: interrupted attempt cannot claim discovery"
            )
        for req_k in ("outcome_v2", "new_outcomes", "fitness", "tiebreak", "criticality"):
            if req_k not in inflight:
                raise ValueError(
                    f"Corrupt inflight reservation in {inflight_path}: missing required key {req_k!r}"
                )
        if inflight["new_outcomes"] != [] or inflight["outcome_v2"] is not None:
            raise ValueError(
                f"Corrupt inflight reservation in {inflight_path}: interrupted attempt must have empty new_outcomes and null outcome_v2"
            )
        if inflight.get("eligible_for_discovery") is not (inflight_idx > 0):
            raise ValueError(
                f"Corrupt inflight reservation in {inflight_path}: invalid eligible_for_discovery "
                f"{inflight.get('eligible_for_discovery')!r}"
            )
        if inflight.get("feedback_version") != "v2_normalized":
            raise ValueError(
                f"Corrupt inflight reservation in {inflight_path}: invalid feedback_version "
                f"{inflight.get('feedback_version')!r}"
            )
        if inflight.get("fallback_selection") is not (arm_name != "random"):
            raise ValueError(
                f"Corrupt inflight reservation in {inflight_path}: invalid fallback_selection "
                f"{inflight.get('fallback_selection')!r}"
            )
        if (
            inflight["fitness"] is not None
            or inflight["tiebreak"] is not None
            or inflight["criticality"] is not None
        ):
            raise ValueError(
                f"Corrupt inflight reservation in {inflight_path}: interrupted attempt must have null fitness, tiebreak, and criticality"
            )
        if inflight.get("rng_state") is None:
            raise ValueError(
                f"Corrupt inflight reservation in {inflight_path}: missing rng_state"
            )
        _validate_rng_state(inflight["rng_state"], str(inflight_path))
        try:
            _validate_attempt_row(
                inflight,
                idx_ln=inflight_idx,
                arm_name=arm_name,
                total_budget=total_budget,
                expected_shared_hash=expected_shared_hash,
                norm_shared_row=norm_shared_row,
                space=probe_space,
                seen_outcomes=set(),
                ps_mod=ps_mod,
                source=f"inflight reservation in {inflight_path}",
                expected_eval0_candidate=expected_eval0_cand if inflight_idx == 0 else None,
                expected_profile_digest=expected_profile_digest,
                expected_realized_state_digest=expected_realized_digest if inflight_idx == 0 else None,
                allow_empty_interrupted_digest=True,
            )
        except ValueError as exc:
            if "Corrupt rng_state" in str(exc):
                raise
            raise ValueError(f"Corrupt inflight reservation in {inflight_path}: {exc}") from exc
        if inflight_idx == len(parsed_attempts) and len(parsed_attempts) < total_budget:
            if inflight_idx == 0 and not expected_shared_hash and (
                baseline_admission_context is not None or expected_realized_digest is not None
            ):
                raise ValueError(
                    f"Independent eval-0 for arm {arm_name!r} was interrupted or rejected before completion "
                    f"({inflight_path}); refusing to advance arm past unadmitted baseline"
                )
            if inflight_idx == 0:
                cand0_inf = probe_space.from_payload(base_spec, fill_missing=True) if base_spec is not None else None
                if cand0_inf is None:
                    probe_space.sample(probe_rng)
            else:
                proposed_inf = probe_space.validate(probe_policy.next_candidate(probe_rng, probe_seen).to_dict())
                expected_inf_cand = loads_strict_json(strict_json(proposed_inf.to_dict()))
                if not _strict_json_equal(inflight["candidate"], expected_inf_cand):
                    raise ValueError(
                        f"Corrupt inflight reservation in {inflight_path}: "
                        f"candidate {inflight['candidate']!r} != expected proposal {expected_inf_cand!r}"
                    )
            expected_inf_rng = loads_strict_json(strict_json(probe_rng.getstate()))
            if not _strict_json_equal(inflight["rng_state"], expected_inf_rng):
                raise ValueError(
                    f"Corrupt inflight reservation in {inflight_path}: "
                    "rng_state contradicts deterministic replay"
                )
            pending_inflight = inflight
            probe_seen.add(probe_space.key(inflight_cand_obj))
            try:
                probe_policy.observe(
                    pending_inflight["candidate"],
                    None,
                    None,
                    row=pending_inflight,
                    feedback_version="v2_normalized",
                )
            except Exception as exc:
                raise ValueError(
                    f"Corrupt inflight reservation in {inflight_path}: replay failed: {exc}"
                ) from exc
        elif inflight_idx < len(parsed_attempts):
            committed = parsed_attempts[inflight_idx]
            if not _strict_json_equal(inflight["candidate"], committed["candidate"]):
                raise ValueError(
                    f"Corrupt inflight reservation in {inflight_path}: "
                    f"stale reservation candidate {inflight['candidate']!r} != committed {committed['candidate']!r}"
                )
            if not _strict_json_equal(inflight["rng_state"], committed["rng_state"]):
                raise ValueError(
                    f"Corrupt inflight reservation in {inflight_path}: "
                    "stale reservation rng_state contradicts committed row"
                )

    replayed_count = len(parsed_attempts) + (1 if pending_inflight is not None else 0)
    if 0 < replayed_count < total_budget:
        try:
            probe_policy.next_candidate(probe_rng, set(probe_seen))
        except Exception as exc:
            raise ValueError(
                f"Corrupt checkpoint replay state in {output_dir}: next_candidate failed: {exc}"
            ) from exc
    return parsed_attempts, valid_lines, torn_trailing, raw_bytes


def _preflight_shared_eval_zero(
    plan: dict[str, Any],
    baseline_row: dict[str, Any],
    *,
    search_space: SearchSpace,
    base_spec: dict[str, Any] | None,
) -> None:
    """Validate that a shared eval-0 baseline row passes schema + policy observation for every arm."""
    if not isinstance(baseline_row, dict):
        raise ValueError("Invalid shared baseline row: expected dict")
    norm_shared = loads_strict_json(strict_json(baseline_row))
    if "criticality" in norm_shared:
        if not _is_optional_finite_number(norm_shared["criticality"]):
            raise ValueError(
                f"Invalid shared baseline row: invalid criticality {norm_shared['criticality']!r}"
            )
        expected_base_crit = criticality_score(norm_shared, feedback_version="v2_normalized")
        if not _strict_json_equal(norm_shared["criticality"], expected_base_crit):
            raise ValueError(
                f"Invalid shared baseline row: criticality {norm_shared['criticality']!r} != {expected_base_crit!r}"
            )
    ps_mod, bl_mod = _load_policy_modules()
    for arm_plan in plan["arms"]:
        arm_name = str(arm_plan["arm"])
        arm_seed = int(arm_plan.get("seed", 13))
        total_budget = int(arm_plan["total_budget"])
        expected_shared_hash = arm_plan.get("shared_eval_zero_hash")
        expected_prof_dig = arm_plan.get("profile_digest")
        probe_rng = random.Random(arm_seed)
        cand_obj = search_space.from_payload(base_spec, fill_missing=True) if base_spec is not None else None
        if cand_obj is None:
            cand_obj = search_space.sample(probe_rng)
        cand_obj = search_space.validate(cand_obj.to_dict())
        cand_dict = cand_obj.to_dict()
        expected_cand0 = loads_strict_json(strict_json(cand_dict))
        row: dict[str, Any] = {
            **norm_shared,
            "arm": arm_name,
            "policy": arm_name,
            "eval_index": 0,
            "candidate": cand_dict,
            "reused_shared_eval_zero": True,
            "eligible_for_discovery": False,
            "discovered_new": False,
            "new_outcomes": [],
            "credited_as_discovery": False,
            "feedback_version": "v2_normalized",
        }
        if "criticality" not in row:
            row["criticality"] = criticality_score(row, feedback_version="v2_normalized")
        fitness_val = ps_mod._semantic_fitness(row, arm_name, engine_active=False, feedback_version="v2_normalized")
        tiebreak_val = ps_mod._semantic_tiebreak(row, arm_name, engine_active=False, feedback_version="v2_normalized")
        row["fitness"] = fitness_val
        row["tiebreak"] = tiebreak_val
        row["fallback_selection"] = bool(fitness_val is None and arm_name != "random")
        row["rng_state"] = probe_rng.getstate()
        norm_row = loads_strict_json(strict_json(row))
        _validate_attempt_row(
            norm_row,
            idx_ln=0,
            arm_name=arm_name,
            total_budget=total_budget,
            expected_shared_hash=expected_shared_hash,
            norm_shared_row=norm_shared,
            space=search_space,
            seen_outcomes=set(),
            ps_mod=ps_mod,
            source=f"shared baseline eval_0 ({arm_name})",
            expected_eval0_candidate=expected_cand0,
            expected_profile_digest=expected_prof_dig,
            expected_realized_state_digest=arm_plan.get("realized_state_digest"),
        )
        probe_policy = bl_mod.make_policy_state(arm_name, search_space, ps_mod.PolicyState, feedback_version="v2_normalized")
        try:
            probe_policy.observe(
                norm_row["candidate"],
                norm_row["fitness"],
                norm_row["tiebreak"],
                row=norm_row,
                feedback_version="v2_normalized",
            )
        except Exception as exc:
            raise ValueError(
                f"Invalid shared baseline row for arm {arm_name!r}: policy observation failed: {exc}"
            ) from exc


def _run_arm_locked(
    arm_plan: dict[str, Any],
    evaluator_fn: Callable,
    *,
    shared_eval_zero_row: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    search_space: SearchSpace | None = None,
    base_spec: dict[str, Any] | None = None,
    baseline_admission_context: tuple[dict[str, Any], OutcomeProfile, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    baseline_admission_context = _snapshot_admission_context(baseline_admission_context)
    (
        arm_name,
        total_budget,
        arm_seed,
        manifest_digest,
        expected_shared_hash,
        expected_profile_digest,
        expected_realized_digest,
        ps_mod,
        bl_mod,
    ) = _validate_arm_plan_inputs(arm_plan, baseline_admission_context=baseline_admission_context)
    if expected_shared_hash:
        if not isinstance(shared_eval_zero_row, dict):
            raise ValueError(
                f"Arm plan for {arm_name!r} requires shared_eval_zero_row matching shared_eval_zero_hash"
            )
        if shared_eval_zero_row.get("evidence_sha256") != expected_shared_hash:
            raise ValueError(
                f"shared_eval_zero_row evidence_sha256 {shared_eval_zero_row.get('evidence_sha256')!r} "
                f"does not match arm_plan shared_eval_zero_hash {expected_shared_hash!r}"
            )
        if shared_eval_zero_row.get("realized_state_digest") != expected_realized_digest:
            raise ValueError(
                f"shared_eval_zero_row realized_state_digest {_safe_repr(shared_eval_zero_row.get('realized_state_digest'))} "
                f"does not match arm_plan realized_state_digest {expected_realized_digest!r}"
            )
    elif shared_eval_zero_row is not None:
        raise ValueError(
            f"Arm plan for {arm_name!r} does not permit shared_eval_zero_row when shared_eval_zero_hash is unset"
        )

    space = search_space or _default_pilot_space(base_spec, arm_seed)
    expected_manifest = _expected_arm_manifest(
        arm_plan,
        space,
        baseline_admission_context=baseline_admission_context,
    )
    norm_shared_row = loads_strict_json(strict_json(shared_eval_zero_row)) if shared_eval_zero_row is not None else None
    if norm_shared_row is not None and "criticality" in norm_shared_row:
        if not _is_optional_finite_number(norm_shared_row["criticality"]):
            raise ValueError(
                f"Invalid shared_eval_zero_row for {arm_name!r}: invalid criticality {norm_shared_row['criticality']!r}"
            )
        expected_shared_crit = criticality_score(norm_shared_row, feedback_version="v2_normalized")
        if not _strict_json_equal(norm_shared_row["criticality"], expected_shared_crit):
            raise ValueError(
                f"Invalid shared_eval_zero_row for {arm_name!r}: criticality "
                f"{norm_shared_row['criticality']!r} != {expected_shared_crit!r}"
            )
    policy_state = bl_mod.make_policy_state(arm_name, space, ps_mod.PolicyState, feedback_version="v2_normalized")
    rng = random.Random(arm_seed)
    seen: set[tuple] = set()
    seen_outcomes: set[str] = set()
    dispatch_mode = _candidate_dispatch_mode(evaluator_fn)
    expected_eval0_cand = _expected_eval_zero_candidate(space, base_spec, arm_seed)

    attempts_path: Path | None = None
    inflight_path: Path | None = None
    attempts: list[dict[str, Any]] = []
    if output_dir is not None:
        manifest_file = output_dir / "pilot_manifest.json"
        attempts_path = output_dir / "attempts.jsonl"
        inflight_path = output_dir / "inflight_attempt.json"
        # Always validate checkpoint state even when manifest_file is absent so direct run_arm()
        # calls reject orphan attempts.jsonl / inflight_attempt.json without pilot_manifest.json.
        parsed_attempts, valid_lines, torn_trailing, raw_bytes = _validate_arm_checkpoint(
            arm_plan,
            output_dir,
            shared_eval_zero_row=shared_eval_zero_row,
            search_space=space,
            base_spec=base_spec,
            baseline_admission_context=baseline_admission_context,
        )
        if manifest_file.exists():
            for idx_prev, prev in enumerate(parsed_attempts):
                if idx_prev == 0:
                    cand0_rep = space.from_payload(base_spec, fill_missing=True) if base_spec is not None else None
                    if cand0_rep is None:
                        space.sample(rng)
                else:
                    # Advance stateful policy queues (e.g. AVFuzzerState.pending) and rng identically to fresh execution.
                    policy_state.next_candidate(rng, seen)
                attempts.append(prev)
                cand_dict = prev["candidate"]
                cand_obj = space.validate(cand_dict)
                seen.add(space.key(cand_obj))
                seen_outcomes |= _confirmed_v2_outcomes(prev.get("outcome_v2"))
                policy_state.observe(
                    cand_dict,
                    prev.get("fitness"),
                    prev.get("tiebreak"),
                    row=prev,
                    feedback_version="v2_normalized",
                )
            if attempts_path.exists() and (torn_trailing or (valid_lines and not raw_bytes.endswith(b"\n"))):
                tmp_attempts = attempts_path.with_name(f".{attempts_path.name}.tmp.{os.getpid()}")
                repaired = "".join(ln + "\n" for ln in valid_lines)
                try:
                    with tmp_attempts.open("w", encoding="utf-8") as fh:
                        fh.write(repaired)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp_attempts, attempts_path)
                except Exception:
                    try:
                        tmp_attempts.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise
            # Recover any pre-launch write-ahead reservation so an interrupted physical call
            # is charged to the budget rather than re-launched and exceeding physical_budget.
            if inflight_path.exists():
                inflight = read_strict_json(inflight_path)
                inflight_idx = inflight.get("eval_index")
                if inflight_idx == len(attempts) and len(attempts) < total_budget:
                    if inflight_idx == 0:
                        cand0_inf_rep = space.from_payload(base_spec, fill_missing=True) if base_spec is not None else None
                        if cand0_inf_rep is None:
                            space.sample(rng)
                    else:
                        policy_state.next_candidate(rng, seen)
                    cand_dict = inflight["candidate"]
                    cand_obj = space.validate(cand_dict)
                    seen.add(space.key(cand_obj))
                    policy_state.observe(
                        cand_dict,
                        None,
                        None,
                        row=inflight,
                        feedback_version="v2_normalized",
                    )
                    inflight_line = strict_json(inflight)
                    attempts.append(inflight)
                    with attempts_path.open("a", encoding="utf-8") as handle:
                        handle.write(inflight_line + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    inflight_path.unlink(missing_ok=True)
                elif isinstance(inflight_idx, int) and inflight_idx < len(attempts):
                    inflight_path.unlink(missing_ok=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(manifest_file, expected_manifest)

    for eval_idx in range(len(attempts), total_budget):
        cand_obj = None
        if eval_idx == 0:
            if base_spec is not None:
                cand_obj = space.from_payload(base_spec, fill_missing=True)
            if cand_obj is None:
                cand_obj = space.sample(rng)
        else:
            cand_obj = policy_state.next_candidate(rng, seen)
        # Validate candidate against SearchSpace BEFORE recording seen, reserving inflight, or launching!
        cand_obj = space.validate(cand_obj.to_dict())
        seen.add(space.key(cand_obj))
        cand_dict = cand_obj.to_dict()

        if eval_idx == 0 and shared_eval_zero_row is not None:
            raw = dict(shared_eval_zero_row)
            reused = True
        else:
            reused = False
            if inflight_path is not None:
                # Exclusive write-ahead reservation (O_CREAT|O_EXCL) BEFORE physical launch so a crash
                # charges this physical call on resume and concurrent runs cannot both launch the same slot.
                reserved = _create_exclusive_json(
                    inflight_path,
                    {
                        "arm": arm_name,
                        "policy": arm_name,
                        "eval_index": eval_idx,
                        "candidate": cand_dict,
                        "reused_shared_eval_zero": False,
                        "eligible_for_discovery": eval_idx > 0,
                        "execution_status": "interrupted",
                        "run_error": "interrupted_before_completion",
                        "outcome_v2": None,
                        "discovered_new": False,
                        "new_outcomes": [],
                        "credited_as_discovery": False,
                        "feedback_version": "v2_normalized",
                        "criticality": None,
                        "fitness": None,
                        "tiebreak": None,
                        "fallback_selection": arm_name != "random",
                        "rng_state": rng.getstate(),
                    },
                )
                if not reserved:
                    raise RuntimeError(f"Concurrent or uncommitted inflight reservation already exists at {inflight_path}")
            if dispatch_mode == "positional":
                raw = evaluator_fn(arm_name, eval_idx, dict(cand_dict))
            elif dispatch_mode == "keyword":
                raw = evaluator_fn(arm_name, eval_idx, candidate=dict(cand_dict))
            else:
                raw = evaluator_fn(arm_name, eval_idx)

        if not isinstance(raw, dict):
            raise ValueError(f"Invalid evaluator result for {arm_name!r} eval {eval_idx}: expected dict, got {type(raw).__name__}")
        # JSON-normalize evaluator output BEFORE computing outcomes/guidance scores so tuples->lists
        # and strict-JSON numeric checks are consistent between fresh execution and replay.
        raw = loads_strict_json(strict_json(raw))
        if raw.get("outcome_v2") is not None and not isinstance(raw.get("outcome_v2"), dict):
            raise ValueError(f"Invalid evaluator result for {arm_name!r} eval {eval_idx}: outcome_v2 must be an object or null")
        run_outcomes = _confirmed_v2_outcomes(raw.get("outcome_v2"))
        if eval_idx == 0:
            new_outcomes: list[str] = []
            discovered_new = False
        else:
            new_outcomes = sorted(run_outcomes - seen_outcomes)
            discovered_new = bool(new_outcomes)

        row = {
            **raw,
            "arm": arm_name,
            "policy": arm_name,
            "eval_index": eval_idx,
            "candidate": cand_dict,
            "reused_shared_eval_zero": reused,
            "eligible_for_discovery": eval_idx > 0,
            "discovered_new": discovered_new,
            "new_outcomes": new_outcomes,
            "credited_as_discovery": discovered_new,
            "feedback_version": "v2_normalized",
        }
        if "criticality" not in row:
            row["criticality"] = criticality_score(row, feedback_version="v2_normalized")
        elif not _is_optional_finite_number(row["criticality"]):
            raise ValueError(
                f"Invalid evaluator result for {arm_name!r} eval {eval_idx}: invalid criticality {row['criticality']!r}"
            )
        fitness_val = ps_mod._semantic_fitness(row, arm_name, engine_active=False, feedback_version="v2_normalized")
        tiebreak_val = ps_mod._semantic_tiebreak(row, arm_name, engine_active=False, feedback_version="v2_normalized")
        row["fitness"] = fitness_val
        row["tiebreak"] = tiebreak_val
        row["fallback_selection"] = bool(fitness_val is None and arm_name != "random")
        row["rng_state"] = rng.getstate()

        # Validate serialized row against the exact checkpoint reader contract and observe the
        # JSON-normalized row BEFORE mutating state, appending to attempts.jsonl, or clearing inflight.
        row_line = strict_json(row)
        norm_row = loads_strict_json(row_line)
        _validate_attempt_row(
            norm_row,
            idx_ln=eval_idx,
            arm_name=arm_name,
            total_budget=total_budget,
            expected_shared_hash=expected_shared_hash,
            norm_shared_row=norm_shared_row,
            space=space,
            seen_outcomes=seen_outcomes,
            ps_mod=ps_mod,
            source=f"{arm_name}:eval_{eval_idx}",
            expected_eval0_candidate=expected_eval0_cand if eval_idx == 0 else None,
            expected_profile_digest=expected_profile_digest,
            expected_realized_state_digest=expected_realized_digest if eval_idx == 0 else None,
            baseline_admission_context=baseline_admission_context if eval_idx == 0 else None,
        )
        try:
            policy_state.observe(
                norm_row["candidate"],
                norm_row["fitness"],
                norm_row["tiebreak"],
                row=norm_row,
                feedback_version="v2_normalized",
            )
        except Exception as exc:
            raise ValueError(
                f"Invalid evaluator result for {arm_name!r} eval {eval_idx}: policy observation failed on normalized row: {exc}"
            ) from exc
        seen_outcomes |= run_outcomes
        attempts.append(norm_row)
        if attempts_path is not None:
            with attempts_path.open("a", encoding="utf-8") as handle:
                handle.write(row_line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        if inflight_path is not None and not reused:
            inflight_path.unlink(missing_ok=True)

    # Strict budget enforcement: never exceed total_budget (no automatic extension).
    assert len(attempts) == total_budget
    budget_summary = record_budget(attempts, shared_eval_zero=shared_eval_zero_row is not None)
    return {
        "arm": arm_name,
        "manifest_digest": manifest_digest,
        "budget": budget_summary,
        "attempts": attempts,
    }


def write_pilot_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "pilot_report.json"
    atomic_json(target, report)
    return target


def _family_obligations(family: dict[str, Any]) -> list[str]:
    role = str(family.get("intended_role") or "")
    actor_alias = "pedestrian" if "pedestrian" in role else ("vehicle" if role else "ego")
    obs: list[str] = []
    for pred in family.get("expected_predicates") or ():
        text = str(pred)
        if "(" in text:
            obs.append(text)
        elif actor_alias == "ego":
            obs.append(f"{text}(ego)")
        else:
            obs.append(f"{text}({actor_alias},ego)")
    return obs


def _make_live_evaluator(
    *,
    spec_path: Path,
    base_spec_payload: dict[str, Any],
    search_space: SearchSpace,
    family: dict[str, Any],
    profile_path: Path,
    output_root: Path,
    host: str,
    port: int,
    agent_kind: str,
    pcla_agent: str | None,
    base_seed: int,
    agent_cfg_digest: str,
) -> Callable[[str, int, dict[str, float] | None], dict[str, Any]]:
    """Build a live HarnessRunner evaluator for one family across matched pilot arms."""
    from research.harness.config import AppConfig
    from research.harness.observers.semantic import SemanticObserver, SemanticObserverConfig
    from research.harness.runner import HarnessRunner
    from research.harness.scenarios import load_scenario_spec

    family_obs = _family_obligations(family)

    def _evaluate(arm: str, eval_idx: int, candidate: dict[str, float] | None = None) -> dict[str, Any]:
        eval_dir = output_root / arm / f"eval_{eval_idx:02d}"
        try:
            eval_dir.mkdir(parents=True, exist_ok=True)
            if candidate is not None:
                cand_obj = search_space.validate(candidate)
                mutated_payload = search_space.apply_to_payload(base_spec_payload, cand_obj)
            else:
                mutated_payload = json.loads(json.dumps(base_spec_payload))
            mutated_payload["scenario_id"] = f"{mutated_payload.get('scenario_id', 'pilot')}-{arm}-e{eval_idx:02d}"
            ctrl_params = mutated_payload.setdefault("controller_params", {})
            if isinstance(ctrl_params, dict) and family_obs and not ctrl_params.get("coverage_obligations"):
                ctrl_params["coverage_obligations"] = list(family_obs)
            mutated_spec_path = eval_dir / "scenario_spec.json"
            atomic_json(mutated_spec_path, mutated_payload)
            scenario = load_scenario_spec(mutated_spec_path)

            cfg = AppConfig()
            cfg.harness.host = host
            cfg.harness.port = int(port)
            cfg.harness.seed = int(base_seed + eval_idx)
            cfg.harness.reload_world = True
            cfg.harness.output_dir = eval_dir / "runs"
            cfg.agent.kind = agent_kind
            cfg.agent.pcla_agent_name = pcla_agent
            cfg.outcomes.enabled = True
            cfg.outcomes.profile_path = profile_path
            cfg.outcomes.evidence_output_dir = eval_dir / "evidence"

            sem_dir = eval_dir / "semantic"
            sem_observer = SemanticObserver(
                SemanticObserverConfig(
                    stream_output_dir=sem_dir,
                    trace_output_dir=sem_dir,
                    stream_every_ticks=1,
                )
            )
            res = HarnessRunner(cfg, observers=[sem_observer]).run(scenario)
            ev_sha = None
            realized_digest = None
            intended_result = None
            if res.capture_manifest_path:
                _, trace_path = load_capture_manifest(Path(res.capture_manifest_path))
                ev_sha = file_digest(trace_path)
                evidence_frames = [
                    rec for rec in iter_evidence(trace_path)
                    if isinstance(rec, FrameEvidence) and rec.phase == "evaluation"
                ]
                if evidence_frames:
                    first_frame = evidence_frames[0]
                    map_dig = hashlib.sha256(str(scenario.town).encode()).hexdigest()
                    route_id = (
                        first_frame.route_progress.route_id
                        if first_frame.route_progress is not None
                        else f"{scenario.town}:{scenario.ego_spawn_index}->{scenario.goal_spawn_index}"
                    )
                    route_dig = hashlib.sha256(str(route_id).encode()).hexdigest()
                    try:
                        realized_digest = realized_initial_state_digest(
                            (first_frame.ego, *first_frame.actors),
                            {
                                "seed": cfg.harness.seed,
                                "map_digest": map_dig,
                                "ads_digest": agent_cfg_digest,
                                "route_digest": route_dig,
                            },
                        )
                    except ValueError:
                        realized_digest = None
                    if family.get("intended_role") or family.get("expected_predicates"):
                        intended_result = evaluate_intended_interaction(
                            {
                                "intended_role": family.get("intended_role"),
                                "required_predicates": family.get("expected_predicates") or (),
                            },
                            sem_observer._trace,
                            evidence_frames,
                            ego_actor_id=first_frame.ego.actor_id,
                        )
            sem_meta = (res.metadata.get("semantic") or {}) if isinstance(res.metadata, dict) else {}
            ob_credit = sem_meta.get("obligation_credit") or {}
            return {
                "execution_status": res.execution_status or "completed",
                "run_id": res.run_id,
                "outcome_v2": res.metadata.get("outcome_v2"),
                "capture_manifest_path": res.capture_manifest_path,
                "evidence_sha256": ev_sha,
                "realized_state_digest": realized_digest,
                "intended_interaction": intended_result,
                "intended_witnessed": intended_result["witnessed"] if isinstance(intended_result, dict) else None,
                "semantic_fulfilled_obligations": list(ob_credit.get("fulfilled_obligations") or ()),
                "semantic_covered_signatures": list(sem_meta.get("covered_signatures") or ()),
                "semantic_covered_predicates": list(sem_meta.get("covered_predicates") or ()),
                "run_error": None,
            }
        except Exception as exc:  # noqa: BLE001 - failed attempts consume budget explicitly
            return {
                "execution_status": "failed",
                "run_id": None,
                "outcome_v2": None,
                "capture_manifest_path": None,
                "evidence_sha256": None,
                "realized_state_digest": None,
                "intended_witnessed": None,
                "run_error": f"{type(exc).__name__}:{exc}",
            }

    return _evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, default=None,
                        help="Optional live validation_report.json to verify against profile/implementation digests.")
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--agent-kind", default="behavior")
    parser.add_argument("--pcla-agent", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true", help="Validate profile/protocol and write immutable plan without launching CARLA.")
    args = parser.parse_args(argv)

    protocol, protocol_file_digest = _load_protocol_with_digest(args.protocol)
    profile = load_outcome_profile(args.profile)
    # Enforce profile validation gate BEFORE launching any simulation.
    profile.require_pilot_ready()

    validation_check = None
    if args.validation_report is not None:
        validation_check = verify_validation_report(args.validation_report, profile)
        if not validation_check["verified"]:
            raise SystemExit(f"Validation report rejected: {', '.join(validation_check['reasons'])}")
    elif not args.plan_only:
        # Live pilot execution cannot bypass live validation via a placeholder validation_refs string.
        for ref in profile.validation_refs:
            ref_path = Path(ref)
            candidates = (
                [ref_path]
                if ref_path.is_absolute()
                else [Path(".") / ref_path, args.profile.parent / ref_path, WORKSPACE_ROOT / ref_path]
            )
            for cand in candidates:
                if cand.is_file():
                    try:
                        check = verify_validation_report(cand, profile)
                    except Exception:
                        continue
                    if check["verified"]:
                        validation_check = check
                        break
            if validation_check is not None:
                break
        if validation_check is None or not validation_check["verified"]:
            raise SystemExit(
                "Live pilot execution requires a verified live validation_report.json "
                "(pass --validation-report or point profile.validation_refs to a verified live report)"
            )

    family = next((f for f in protocol.get("interaction_families", ()) if f.get("family_id") == args.family), None)
    if family is None:
        raise SystemExit(f"Unknown pilot family '{args.family}'")

    spec_path = (args.protocol.parent / family["spec"]).resolve()
    # Parse and digest the SAME spec bytes so the pinned spec_digest matches what is evaluated.
    raw_spec = spec_path.read_bytes()
    base_spec_payload = loads_strict_json(raw_spec.decode("utf-8"))
    spec_digest = hashlib.sha256(raw_spec).hexdigest()
    agent_cfg_digest = hashlib.sha256(
        strict_json({"agent_kind": args.agent_kind, "pcla_agent": args.pcla_agent}).encode()
    ).hexdigest()
    search_space = _default_pilot_space(base_spec_payload, args.seed)

    if args.plan_only:
        with _exclusive_dir_lock(args.output_dir / ".pilot.lock"):
            if _has_existing_execution_state(args.output_dir):
                raise RuntimeError(
                    f"Refusing --plan-only into {args.output_dir}: directory already contains pilot execution state"
                )
            plan = build_matched_arms(
                protocol,
                profile,
                args.family,
                args.seed,
                spec_digest=spec_digest,
                agent_config_digest=agent_cfg_digest,
                search_space=search_space,
            )
            write_pilot_report(
                {"status": "planned_only", "validation_check": validation_check, "plan": plan},
                args.output_dir,
            )
        return 0

    # The evaluator gets private deep copies of the family, base spec, and search space so it cannot
    # mutate the objects main() uses for baseline admission, arm admission context, checkpoint identity,
    # and the arms' candidate generation.
    evaluator = _make_live_evaluator(
        spec_path=spec_path,
        base_spec_payload=copy.deepcopy(base_spec_payload),
        search_space=copy.deepcopy(search_space),
        family=copy.deepcopy(family),
        profile_path=args.profile,
        output_root=args.output_dir / "runs",
        host=args.host,
        port=args.port,
        agent_kind=args.agent_kind,
        pcla_agent=args.pcla_agent,
        base_seed=args.seed,
        agent_cfg_digest=agent_cfg_digest,
    )
    baseline_ckpt_path = args.output_dir / "baseline_checkpoint.json"
    baseline_inflight_path = args.output_dir / "baseline_inflight.json"
    ckpt_identity = {
        "protocol_id": protocol.get("protocol_id"),
        "protocol_digest": protocol_file_digest,
        "arms": list(protocol["arms"]),
        "budget_per_arm": protocol["budget_per_arm"],
        "shared_eval_zero_policy": protocol.get("shared_eval_zero_policy"),
        "family_id": args.family,
        "seed": int(args.seed),
        "spec_digest": spec_digest,
        "agent_config_digest": agent_cfg_digest,
        "profile_digest": profile_digest(profile),
        "implementation_digest": implementation_digest(),
        "search_space_signature": search_space.signature(),
    }
    norm_ckpt_identity = loads_strict_json(strict_json(ckpt_identity))
    def _validate_baseline_inflight() -> None:
        stored_inflight = read_strict_json(baseline_inflight_path)
        if not isinstance(stored_inflight, dict) or not _strict_json_equal(stored_inflight.get("identity"), norm_ckpt_identity):
            raise ValueError("Incompatible pilot baseline checkpoint; refusing to resume into mismatched directory")
        if (
            stored_inflight.get("status") != "interrupted"
            or stored_inflight.get("run_error") != "baseline_interrupted_before_completion"
            or set(stored_inflight.keys()) != {"identity", "status", "run_error"}
        ):
            raise ValueError(f"Corrupt baseline inflight reservation in {baseline_inflight_path}")

    with _exclusive_dir_lock(args.output_dir / ".pilot.lock"):
        stale_baseline_inflight = False
        if baseline_ckpt_path.exists():
            stored_ckpt = read_strict_json(baseline_ckpt_path)
            if not isinstance(stored_ckpt, dict) or not _strict_json_equal(stored_ckpt.get("identity"), norm_ckpt_identity):
                raise ValueError("Incompatible pilot baseline checkpoint; refusing to resume into mismatched directory")
            admission = stored_ckpt.get("admission")
            baseline_row = stored_ckpt.get("baseline_row")
            if not isinstance(admission, dict) or not isinstance(admission.get("admitted"), bool):
                raise ValueError("Corrupt baseline_checkpoint.json: missing valid admission dict")
            if not isinstance(baseline_row, dict):
                raise ValueError("Corrupt baseline_checkpoint.json: missing baseline_row dict")
            if "physical_calls_charged" not in stored_ckpt:
                raise ValueError("Corrupt baseline_checkpoint.json: missing physical_calls_charged")
            physical_calls = stored_ckpt["physical_calls_charged"]
            if isinstance(physical_calls, bool) or not isinstance(physical_calls, int) or physical_calls != 1:
                raise ValueError("Corrupt baseline_checkpoint.json: invalid physical_calls_charged")
            is_interrupted_baseline = (
                baseline_row.get("execution_status") == "interrupted"
                and baseline_row.get("run_error") == "baseline_interrupted_before_completion"
                and baseline_row.get("outcome_v2") is None
            )
            if is_interrupted_baseline:
                expected_admission = {
                    "admitted": False,
                    "reasons": ["baseline_interrupted_before_completion"],
                    "profile_digest": profile_digest(profile),
                    "realized_state_digest": None,
                    "intended_witnessed": None,
                }
            else:
                if baseline_row.get("outcome_v2") is not None and not isinstance(baseline_row.get("outcome_v2"), dict):
                    raise ValueError("Corrupt baseline_checkpoint.json: baseline_row.outcome_v2 must be an object or null")
                _confirmed_v2_outcomes(baseline_row.get("outcome_v2"))
                expected_admission = validate_initial_configuration(
                    baseline_row.get("outcome_v2") or {},
                    protocol,
                    profile=profile,
                    realized_state_digest=baseline_row.get("realized_state_digest"),
                    intended_witnessed=baseline_row.get("intended_witnessed"),
                    family=family,
                    row_execution_status=(
                        baseline_row["execution_status"] if "execution_status" in baseline_row else _UNSET
                    ),
                    run_error=baseline_row.get("run_error"),
                )
            norm_expected_admission = loads_strict_json(strict_json(expected_admission))
            if not _strict_json_equal(admission, norm_expected_admission):
                raise ValueError(
                    "Corrupt baseline_checkpoint.json: admission contradicts baseline_row"
                )
            if not admission["admitted"]:
                if stored_ckpt.get("plan") is not None or stored_ckpt.get("share_eval_zero") is not False:
                    raise ValueError(
                        "Corrupt baseline_checkpoint.json: rejected baseline has invalid plan/sharing metadata"
                    )
                if _has_existing_arms_state(args.output_dir):
                    raise ValueError(
                        "Corrupt pilot checkpoint: existing arm execution state contradicts rejected baseline_checkpoint.json"
                    )
                if baseline_inflight_path.exists():
                    _validate_baseline_inflight()
                    baseline_inflight_path.unlink(missing_ok=True)
                write_pilot_report(
                    {
                        "status": "admission_rejected",
                        "validation_check": validation_check,
                        "admission": admission,
                        "baseline": baseline_row,
                        "physical_calls_charged": physical_calls,
                    },
                    args.output_dir,
                )
                return 1
            plan = stored_ckpt.get("plan")
            share_eval_zero = stored_ckpt.get("share_eval_zero")
            if not isinstance(plan, dict) or not isinstance(plan.get("arms"), list) or not isinstance(share_eval_zero, bool):
                raise ValueError("Corrupt baseline_checkpoint.json: missing valid plan or boolean share_eval_zero")
            expected_share_eval_zero = (
                protocol.get("shared_eval_zero_policy") == "allow_if_identical_hashes"
                and isinstance(baseline_row.get("evidence_sha256"), str)
                and bool(baseline_row.get("evidence_sha256").strip())
            )
            if share_eval_zero is not expected_share_eval_zero:
                raise ValueError(
                    "Corrupt baseline_checkpoint.json: share_eval_zero contradicts protocol/baseline_row"
                )
            expected_plan = build_matched_arms(
                protocol,
                profile,
                args.family,
                args.seed,
                shared_eval_zero=baseline_row if expected_share_eval_zero else None,
                spec_digest=spec_digest,
                agent_config_digest=agent_cfg_digest,
                realized_state_digest=baseline_row.get("realized_state_digest"),
                search_space=search_space,
            )
            norm_expected_plan = loads_strict_json(strict_json(expected_plan))
            if not _strict_json_equal(plan, norm_expected_plan):
                raise ValueError(
                    "Corrupt baseline_checkpoint.json: plan does not match protocol/baseline_row"
                )
            if share_eval_zero:
                _preflight_shared_eval_zero(
                    plan,
                    baseline_row,
                    search_space=search_space,
                    base_spec=base_spec_payload,
                )
            if baseline_inflight_path.exists():
                _validate_baseline_inflight()
                stale_baseline_inflight = True
        elif baseline_inflight_path.exists():
            _validate_baseline_inflight()
            if _has_existing_execution_state(args.output_dir, ignore_baseline_inflight=True):
                raise ValueError(
                    "Corrupt pilot checkpoint: existing arm/report state found alongside standalone baseline_inflight.json"
                )
            # Previous shared baseline launch was interrupted before completion: charge it without relaunching!
            interrupted_admission = {
                "admitted": False,
                "reasons": ["baseline_interrupted_before_completion"],
                "profile_digest": profile_digest(profile),
                "realized_state_digest": None,
                "intended_witnessed": None,
            }
            interrupted_row = {
                "arm": "baseline_admission",
                "policy": "baseline_admission",
                "eval_index": 0,
                "execution_status": "interrupted",
                "run_error": "baseline_interrupted_before_completion",
                "outcome_v2": None,
            }
            atomic_json(
                baseline_ckpt_path,
                {
                    "identity": ckpt_identity,
                    "baseline_row": interrupted_row,
                    "admission": interrupted_admission,
                    "plan": None,
                    "share_eval_zero": False,
                    "physical_calls_charged": 1,
                },
            )
            baseline_inflight_path.unlink(missing_ok=True)
            write_pilot_report(
                {
                    "status": "admission_rejected",
                    "validation_check": validation_check,
                    "admission": interrupted_admission,
                    "baseline": interrupted_row,
                    "physical_calls_charged": 1,
                },
                args.output_dir,
            )
            return 1
        else:
            if _has_existing_execution_state(args.output_dir):
                raise ValueError(
                    "Corrupt pilot checkpoint: existing arm/report state found without baseline_checkpoint.json"
                )
            base_candidate = search_space.from_payload(base_spec_payload, fill_missing=True)
            if base_candidate is not None:
                base_candidate = search_space.validate(base_candidate.to_dict())
            args.output_dir.mkdir(parents=True, exist_ok=True)
            # Exclusive write-ahead reservation (O_CREAT|O_EXCL) BEFORE launching shared baseline eval 0 so
            # an abrupt interruption is never silently relaunched beyond the planned physical budget.
            reserved = _create_exclusive_json(
                baseline_inflight_path,
                {
                    "identity": ckpt_identity,
                    "status": "interrupted",
                    "run_error": "baseline_interrupted_before_completion",
                },
            )
            if not reserved:
                raise RuntimeError(
                    f"Concurrent or uncommitted baseline reservation already exists at {baseline_inflight_path}"
                )
            baseline_row = evaluator(
                "baseline_admission",
                0,
                base_candidate.to_dict() if base_candidate is not None else None,
            )
            if not isinstance(baseline_row, dict):
                raise ValueError(
                    f"Invalid baseline evaluator result: expected dict, got {type(baseline_row).__name__}"
                )
            if baseline_row.get("outcome_v2") is not None and not isinstance(baseline_row.get("outcome_v2"), dict):
                raise ValueError("Invalid baseline evaluator result: outcome_v2 must be an object or null")
            # Ensure baseline_row is strict-JSON-serializable and structurally valid before committing checkpoint.
            baseline_row = loads_strict_json(strict_json(baseline_row))
            _confirmed_v2_outcomes(baseline_row.get("outcome_v2"))
            admission = validate_initial_configuration(
                baseline_row.get("outcome_v2") or {},
                protocol,
                profile=profile,
                realized_state_digest=baseline_row.get("realized_state_digest"),
                intended_witnessed=baseline_row.get("intended_witnessed"),
                family=family,
                row_execution_status=(
                    baseline_row["execution_status"] if "execution_status" in baseline_row else _UNSET
                ),
                run_error=baseline_row.get("run_error"),
            )
            if not admission["admitted"]:
                atomic_json(
                    baseline_ckpt_path,
                    {
                        "identity": ckpt_identity,
                        "baseline_row": baseline_row,
                        "admission": admission,
                        "plan": None,
                        "share_eval_zero": False,
                        "physical_calls_charged": 1,
                    },
                )
                baseline_inflight_path.unlink(missing_ok=True)
                write_pilot_report(
                    {
                        "status": "admission_rejected",
                        "validation_check": validation_check,
                        "admission": admission,
                        "baseline": baseline_row,
                        "physical_calls_charged": 1,
                    },
                    args.output_dir,
                )
                return 1

            share_eval_zero = (
                protocol.get("shared_eval_zero_policy") == "allow_if_identical_hashes"
                and isinstance(baseline_row.get("evidence_sha256"), str)
                and bool(baseline_row.get("evidence_sha256").strip())
            )
            plan = build_matched_arms(
                protocol,
                profile,
                args.family,
                args.seed,
                shared_eval_zero=baseline_row if share_eval_zero else None,
                spec_digest=spec_digest,
                agent_config_digest=agent_cfg_digest,
                realized_state_digest=baseline_row.get("realized_state_digest"),
                search_space=search_space,
            )
            if share_eval_zero:
                _preflight_shared_eval_zero(
                    plan,
                    baseline_row,
                    search_space=search_space,
                    base_spec=base_spec_payload,
                )
            atomic_json(
                baseline_ckpt_path,
                {
                    "identity": ckpt_identity,
                    "baseline_row": baseline_row,
                    "admission": admission,
                    "plan": plan,
                    "share_eval_zero": share_eval_zero,
                    "physical_calls_charged": 1,
                },
            )
            baseline_inflight_path.unlink(missing_ok=True)
        # Validate every existing arm checkpoint upfront before rewriting any checkpoint or launching any arm.
        admission_ctx = (protocol, profile, family)
        for arm_plan in plan["arms"]:
            _validate_arm_checkpoint(
                arm_plan,
                args.output_dir / "arms" / arm_plan["arm"],
                shared_eval_zero_row=baseline_row if share_eval_zero else None,
                search_space=search_space,
                base_spec=base_spec_payload,
                baseline_admission_context=admission_ctx,
            )
        if stale_baseline_inflight:
            baseline_inflight_path.unlink(missing_ok=True)
        arm_results = [
            run_arm(
                arm_plan,
                evaluator,
                shared_eval_zero_row=baseline_row if share_eval_zero else None,
                output_dir=args.output_dir / "arms" / arm_plan["arm"],
                search_space=search_space,
                base_spec=base_spec_payload,
                baseline_admission_context=admission_ctx,
            )
            for arm_plan in plan["arms"]
        ]
        write_pilot_report(
            {"status": "completed", "validation_check": validation_check, "admission": admission, "plan": plan, "arms": arm_results},
            args.output_dir,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
