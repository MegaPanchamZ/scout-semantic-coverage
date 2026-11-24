"""One immutable, content-addressed profile for offline outcome evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from research.harness.outcome_models import CaptureManifest, finite, strict_json


DEFAULT_PROFILE = Path(__file__).with_name("outcome_profiles") / "default-v2.json"
REQUIRED_SETTING_UNITS = {
    "ttc_s": "s", "proximity_m": "m", "deceleration_mps2": "m/s^2",
    "braking_speed_mps": "m/s", "stationary_speed_mps": "m/s",
    "immobilization_s": "s", "collision_gap_s": "s", "encounter_rearm_s": "s",
    "braking_duration_s": "s", "projection_horizon_s": "s", "scan_radius_m": "m",
    "max_sample_interval_s": "s", "minimum_observation_s": "s",
}
OPTIONAL_SETTING_GROUPS = (
    {"task_window_s": "s", "progress_epsilon_m": "m"},
    {"quality_min_observation_s": "s", "interaction_radius_m": "m"},
    {
        "hazard_braking_antecedent_s": "s",
        "speeding_tolerance_mps": "m/s",
        "speeding_duration_s": "s",
        "stop_sign_speed_mps": "m/s",
        "stop_sign_dwell_s": "s",
        "wrong_way_speed_mps": "m/s",
        "wrong_way_duration_s": "s",
    },
)
SETTING_UNITS = {**REQUIRED_SETTING_UNITS, **{k: v for group in OPTIONAL_SETTING_GROUPS for k, v in group.items()}}


@dataclass(frozen=True)
class Setting:
    name: str
    value: float
    unit: str
    provenance: str
    rationale: str

    def __post_init__(self):
        finite(self.value, self.name)
        if self.value <= 0 or self.unit != SETTING_UNITS.get(self.name):
            raise ValueError(f"Invalid value/unit for {self.name}")
        if self.provenance not in {"literature_informed", "supervisor_proposal", "experimental"}:
            raise ValueError(f"Invalid provenance for {self.name}")
        if not self.rationale.strip():
            raise ValueError(f"Missing rationale for {self.name}")


@dataclass(frozen=True)
class OutcomeProfile:
    profile_id: str
    version: int
    status: str
    settings: tuple[Setting, ...]
    validation_refs: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "settings", tuple(self.settings))
        object.__setattr__(self, "validation_refs", tuple(self.validation_refs))
        if self.version != 2 or not self.profile_id or self.status not in {"draft", "validated"}:
            raise ValueError("Unsupported profile identity/version/status")
        names = [s.name for s in self.settings]
        if (len(names) != len(set(names)) or not set(REQUIRED_SETTING_UNITS).issubset(names)
                or not set(names).issubset(SETTING_UNITS)):
            raise ValueError("Profile must specify required settings exactly once; no unknown settings")
        for group in OPTIONAL_SETTING_GROUPS:
            if set(names).intersection(group) and not set(group).issubset(names):
                raise ValueError("Optional setting groups must be supplied in full")
        if self.value("ttc_s") > self.value("projection_horizon_s"):
            raise ValueError("TTC threshold exceeds projection horizon")
        if self.value("proximity_m") > self.value("scan_radius_m"):
            raise ValueError("Proximity threshold exceeds capture radius")
        if self.supports("interaction_radius_m") and not self.value("proximity_m") <= self.value("interaction_radius_m") <= self.value("scan_radius_m"):
            raise ValueError("Interaction radius must span proximity and fit within capture radius")
        if self.status == "validated" and not self.validation_refs:
            raise ValueError("Validated profile requires validation evidence references")

    def value(self, name: str) -> float:
        for setting in self.settings:
            if setting.name == name:
                return setting.value
        raise KeyError(name)

    def supports(self, *names: str) -> bool:
        return set(names).issubset(s.name for s in self.settings)

    def require_pilot_ready(self):
        if self.status != "validated":
            raise ValueError("Draft outcome profile: live validation is required before a pilot")


def load_outcome_profile(path: Path = DEFAULT_PROFILE) -> OutcomeProfile:
    data = json.loads(Path(path).read_text())
    data["settings"] = tuple(Setting(**s) for s in data["settings"])
    data["validation_refs"] = tuple(data.get("validation_refs", ()))
    return OutcomeProfile(**data)


def profile_digest(profile: OutcomeProfile) -> str:
    data = asdict(profile)
    data["settings"] = sorted(data["settings"], key=lambda x: x["name"])
    return hashlib.sha256(strict_json(data).encode()).hexdigest()


def profile_content_digest(profile: OutcomeProfile) -> str:
    """Hash profile thresholds/definition independent of promotion metadata (status/validation_refs)."""
    data = asdict(profile)
    data.pop("status", None)
    data.pop("validation_refs", None)
    data["settings"] = sorted(data["settings"], key=lambda x: x["name"])
    return hashlib.sha256(strict_json(data).encode()).hexdigest()


def implementation_digest() -> str:
    """Identify uncommitted detector and live-capture code as well as released code."""
    names = ("outcome_config.py", "outcome_models.py", "outcome_detectors.py", "interaction_geometry.py",
             "safety_outcomes.py", "route_progress.py", "task_outcomes.py", "scenario_quality.py",
             "traffic_rule_outcomes.py", "road_context.py", "oracles.py", "runner.py")
    hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    return hashlib.sha256(strict_json(hashes).encode()).hexdigest()


def validate_capture_support(profile: OutcomeProfile, manifest: CaptureManifest) -> list[str]:
    reasons = []
    for name in ("scan_radius_m", "projection_horizon_s", "max_sample_interval_s"):
        observed = getattr(manifest, name)
        required = profile.value(name)
        if observed is None:
            reasons.append(f"unrecorded_{name}")
        elif (observed > required if name == "max_sample_interval_s" else observed < required):
            reasons.append(f"unsupported_{name}")
    return reasons
