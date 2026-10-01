from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = WORKSPACE_ROOT / "datasets/Detection-of-Traffic-Anomaly/dataset/metadata_val.json"
DEFAULT_CHECKLIST_PATH = WORKSPACE_ROOT / "research/logs/checklists/symbolic_checklist.json"


@dataclass(slots=True)
class DotaClip:
    clip_id: str
    anomaly_class: str
    anomaly_start: int
    anomaly_end: int
    num_frames: int
    subset: str
    video_start: int
    video_end: int

    @property
    def anomaly_progress_ratio(self) -> float:
        if self.num_frames <= 1:
            return 0.5
        return min(max(float(self.anomaly_start) / float(self.num_frames - 1), 0.0), 1.0)

    @property
    def anomaly_duration_ratio(self) -> float:
        if self.num_frames <= 0:
            return 0.0
        duration = max(int(self.anomaly_end) - int(self.anomaly_start), 1)
        return min(max(float(duration) / float(self.num_frames), 0.0), 1.0)


ALIASES = {
    "other:lateral": "other: lateral",
    "other: lateral": "other: lateral",
    "other:moving_ahead_or_waiting": "other: ahead_or_waiting",
    "other: moving_ahead_or_waiting": "other: ahead_or_waiting",
    "other:ahead_or_waiting": "other: ahead_or_waiting",
    "other: ahead_or_waiting": "other: ahead_or_waiting",
    "ego:leave_to_left": "ego: leave_to_left",
    "ego: leave_to_left": "ego: leave_to_left",
    "ego:leave_to_right": "ego: leave_to_right",
    "ego: leave_to_right": "ego: leave_to_right",
}


ANOMALY_DESCRIPTIONS = {
    "other: lateral": {
        "description": "Another road user moves laterally into or across the ego trajectory, creating a crossing or cut-in conflict that forces the ego vehicle to react.",
        "stsl_formula": "Eventually(lane_changing(adversary) AND crossing_path(adversary, ego))",
        "verification_predicates": ["crossing_path", "in_front_of"],
        "scenario_intent": "ThresholdCrossingAdversary",
        "controller": "threshold_crossing_adversary",
        "controller_params": {
            "adversary_kind": "walker",
            "trigger_radius_m": 8.0,
            "walker_speed_mps": 1.8,
            "placement_hint": "spawn adversary beside the ego lane and trigger a crossing near a junction",
        },
        "scallop_rules": [
            "hazard(adversary, ego) :- lane_changing(adversary), crossing_path(adversary, ego).",
            "covered(other_lateral) :- hazard(adversary, ego).",
        ],
    },
    "other: ahead_or_waiting": {
        "description": "A vehicle ahead of the ego vehicle is stopped, waiting, or brakes sharply in-lane, creating a lead-vehicle conflict that tests headway handling.",
        "stsl_formula": "Eventually(in_front_of(adversary, ego) AND (stationary(adversary) OR braking(adversary)))",
        "verification_predicates": ["in_front_of", "waiting"],
        "scenario_intent": "LeadVehicleBraking",
        "controller": "lead_vehicle_braking",
        "controller_params": {
            "trigger_radius_m": 10.0,
            "pre_brake_throttle": 0.3,
            "placement_hint": "spawn a same-lane adversary ahead of the ego route and force a hard brake in the causal corridor",
        },
        "scallop_rules": [
            "hazard(adversary, ego) :- in_front_of(adversary, ego), stationary(adversary).",
            "covered(other_ahead_or_waiting) :- hazard(adversary, ego).",
        ],
    },
    "other: turning": {
        "description": "Another traffic participant turns across or into the ego route, creating an intersection conflict with crossing trajectories.",
        "stsl_formula": "Eventually(turning(adversary) AND crossing_path(adversary, ego))",
        "verification_predicates": ["crossing_path", "in_front_of"],
        "scenario_intent": "IntersectionTurningConflict",
        "controller": "intersection_turning_adversary",
        "controller_params": {
            "trigger_radius_m": 12.0,
            "placement_hint": "stage an NPC in a junction approach and trigger a crossing turn into the ego route",
        },
        "scallop_rules": [
            "hazard(adversary, ego) :- turning(adversary), crossing_path(adversary, ego).",
            "covered(other_turning) :- hazard(adversary, ego).",
        ],
    },
    "other: oncoming": {
        "description": "An oncoming vehicle encroaches into the ego corridor, creating a wrong-way or head-on approach hazard.",
        "stsl_formula": "Eventually(oncoming(adversary) AND approaching(adversary, ego))",
        "verification_predicates": ["in_front_of"],
        "scenario_intent": "OncomingEncroachment",
        "controller": "oncoming_encroachment",
        "controller_params": {
            "trigger_radius_m": 14.0,
            "placement_hint": "spawn an adversary in the opposing lane and drift it into the ego path",
        },
        "scallop_rules": [
            "hazard(adversary, ego) :- oncoming(adversary), approaching(adversary, ego).",
            "covered(other_oncoming) :- hazard(adversary, ego).",
        ],
    },
    "other: pedestrian": {
        "description": "A pedestrian enters the roadway ahead of the ego vehicle and forces a semantic crossing-path hazard.",
        "stsl_formula": "Eventually(jaywalking(adversary) AND crossing_path(adversary, ego))",
        "verification_predicates": ["jaywalking", "crossing_path"],
        "scenario_intent": "PedestrianCrossing",
        "controller": "threshold_crossing_adversary",
        "controller_params": {
            "adversary_kind": "walker",
            "trigger_radius_m": 8.0,
            "walker_speed_mps": 1.8,
            "placement_hint": "spawn a pedestrian at curbside and trigger a crossing into the ego lane",
        },
        "scallop_rules": [
            "hazard(adversary, ego) :- jaywalking(adversary), crossing_path(adversary, ego).",
            "covered(other_pedestrian) :- hazard(adversary, ego).",
        ],
    },
    "other: obstacle": {
        "description": "A static obstacle occupies or obstructs the drivable corridor ahead of the ego vehicle.",
        "stsl_formula": "Eventually(on_road(adversary) AND obstructing(adversary, ego))",
        "verification_predicates": ["on_road", "obstructing"],
        "scenario_intent": "StaticObstacleIntrusion",
        "controller": "static_obstacle_intrusion",
        "controller_params": {
            "placement_hint": "place a blocking prop or stopped vehicle in the ego lane within braking distance",
        },
        "scallop_rules": [
            "hazard(adversary, ego) :- on_road(adversary), obstructing(adversary, ego).",
            "covered(other_obstacle) :- hazard(adversary, ego).",
        ],
    },
    "other: leave_to_left": {
        "description": "Another vehicle departs leftward from its lane and intrudes into the ego path, testing lane-boundary conflict handling.",
        "stsl_formula": "Eventually(departing_left(adversary) AND crossing_path(adversary, ego))",
        "verification_predicates": ["crossing_path", "in_front_of"],
        "scenario_intent": "LaneIntrusionFromLeft",
        "controller": "lane_departure_adversary",
        "controller_params": {
            "placement_hint": "stage an adjacent-lane actor and activate a leftward intrusion into the ego lane",
        },
        "scallop_rules": [
            "hazard(adversary, ego) :- departing_left(adversary), crossing_path(adversary, ego).",
            "covered(other_leave_to_left) :- hazard(adversary, ego).",
        ],
    },
    "other: leave_to_right": {
        "description": "Another vehicle departs rightward from its lane and enters the ego corridor.",
        "stsl_formula": "Eventually(departing_right(adversary) AND crossing_path(adversary, ego))",
        "verification_predicates": ["crossing_path", "in_front_of"],
        "scenario_intent": "LaneIntrusionFromRight",
        "controller": "lane_departure_adversary",
        "controller_params": {
            "placement_hint": "stage an adjacent-lane actor and activate a rightward intrusion into the ego lane",
        },
        "scallop_rules": [
            "hazard(adversary, ego) :- departing_right(adversary), crossing_path(adversary, ego).",
            "covered(other_leave_to_right) :- hazard(adversary, ego).",
        ],
    },
    "other: start_stop_or_stationary": {
        "description": "Another road user transitions abruptly between moving and stationary states ahead of the ego vehicle, creating a stop-go hazard.",
        "stsl_formula": "Eventually(in_front_of(adversary, ego) AND stationary(adversary))",
        "verification_predicates": ["in_front_of", "waiting"],
        "scenario_intent": "StopGoLeadConflict",
        "controller": "lead_vehicle_braking",
        "controller_params": {
            "trigger_radius_m": 10.0,
            "placement_hint": "spawn a lead actor in-lane and force a stop or hesitation event",
        },
        "scallop_rules": [
            "hazard(adversary, ego) :- in_front_of(adversary, ego), stationary(adversary).",
            "covered(other_start_stop_or_stationary) :- hazard(adversary, ego).",
        ],
    },
    "ego: lateral": {
        "description": "The ego vehicle develops an unsafe lateral deviation relative to the lane centerline.",
        "stsl_formula": "Eventually(lane_changing(ego) AND out_of_control(ego))",
        "verification_predicates": ["crossing_path"],
        "scenario_intent": "EgoLateralDeviation",
        "controller": "lane_departure_adversary",
        "controller_params": {
            "placement_hint": "perturb the ego path with an intrusion or occlusion that induces lateral deviation",
        },
        "scallop_rules": [
            "hazard(ego) :- lane_changing(ego), out_of_control(ego).",
            "covered(ego_lateral) :- hazard(ego).",
        ],
    },
    "ego: leave_to_left": {
        "description": "The ego vehicle leaves its lane to the left and violates the expected route geometry.",
        "stsl_formula": "Eventually(departing_left(ego) AND out_of_control(ego))",
        "verification_predicates": ["colliding"],
        "scenario_intent": "LaneDepartureAdversary",
        "controller": "lane_departure_adversary",
        "controller_params": {
            "trigger_radius_m": 8.0,
            "placement_hint": "place an adversary or blockage that pushes the ego left of lane center",
        },
        "scallop_rules": [
            "hazard(ego) :- departing_left(ego), out_of_control(ego).",
            "covered(ego_leave_to_left) :- hazard(ego).",
        ],
    },
    "ego: leave_to_right": {
        "description": "The ego vehicle leaves its lane to the right and violates the expected route geometry.",
        "stsl_formula": "Eventually(departing_right(ego) AND out_of_control(ego))",
        "verification_predicates": ["colliding"],
        "scenario_intent": "RightLaneDepartureAdversary",
        "controller": "lane_departure_adversary",
        "controller_params": {
            "trigger_radius_m": 8.0,
            "placement_hint": "place an adversary or blockage that pushes the ego right of lane center",
        },
        "scallop_rules": [
            "hazard(ego) :- departing_right(ego), out_of_control(ego).",
            "covered(ego_leave_to_right) :- hazard(ego).",
        ],
    },
    "ego: moving_ahead_or_waiting": {
        "description": "The ego vehicle encounters a stop-go or hesitation condition while moving ahead, indicating a control or perception bottleneck.",
        "stsl_formula": "Eventually(moving(ego) AND braking(ego))",
        "verification_predicates": ["in_front_of"],
        "scenario_intent": "EgoStopGoStress",
        "controller": "lead_vehicle_braking",
        "controller_params": {
            "placement_hint": "stress the ego with a close lead actor or junction hesitation event",
        },
        "scallop_rules": [
            "hazard(ego) :- moving(ego), braking(ego).",
            "covered(ego_moving_ahead_or_waiting) :- hazard(ego).",
        ],
    },
    "ego: turning": {
        "description": "The ego vehicle turns into a hazardous geometric or semantic conflict region.",
        "stsl_formula": "Eventually(turning(ego) AND crossing_path(ego, adversary))",
        "verification_predicates": ["crossing_path"],
        "scenario_intent": "EgoTurningConflict",
        "controller": "intersection_turning_adversary",
        "controller_params": {
            "placement_hint": "stage a junction conflict that requires safe ego turning behavior",
        },
        "scallop_rules": [
            "hazard(ego, adversary) :- turning(ego), crossing_path(ego, adversary).",
            "covered(ego_turning) :- hazard(ego, adversary).",
        ],
    },
    "ego: pedestrian": {
        "description": "The ego vehicle encounters a pedestrian-centered anomaly where safe yielding or stopping is required.",
        "stsl_formula": "Eventually(jaywalking(adversary) AND approaching(ego, adversary))",
        "verification_predicates": ["jaywalking", "crossing_path"],
        "scenario_intent": "PedestrianYieldConflict",
        "controller": "threshold_crossing_adversary",
        "controller_params": {
            "adversary_kind": "walker",
            "placement_hint": "place a pedestrian so the ego must yield or brake correctly",
        },
        "scallop_rules": [
            "hazard(ego, adversary) :- jaywalking(adversary), approaching(ego, adversary).",
            "covered(ego_pedestrian) :- hazard(ego, adversary).",
        ],
    },
    "ego: obstacle": {
        "description": "The ego vehicle encounters an obstacle-centered anomaly requiring evasive or braking behavior.",
        "stsl_formula": "Eventually(on_road(adversary) AND obstructing(adversary, ego))",
        "verification_predicates": ["on_road", "obstructing"],
        "scenario_intent": "ObstacleAvoidance",
        "controller": "static_obstacle_intrusion",
        "controller_params": {
            "placement_hint": "place a blocking obstacle inside the ego lane or turning corridor",
        },
        "scallop_rules": [
            "hazard(adversary, ego) :- on_road(adversary), obstructing(adversary, ego).",
            "covered(ego_obstacle) :- hazard(adversary, ego).",
        ],
    },
    "ego: oncoming": {
        "description": "The ego vehicle faces an oncoming or wrong-way conflict that requires path correction or emergency braking.",
        "stsl_formula": "Eventually(oncoming(adversary) AND approaching(adversary, ego))",
        "verification_predicates": ["in_front_of"],
        "scenario_intent": "OncomingAvoidance",
        "controller": "oncoming_encroachment",
        "controller_params": {
            "placement_hint": "induce a wrong-way encounter on a narrow corridor or at a merge",
        },
        "scallop_rules": [
            "hazard(adversary, ego) :- oncoming(adversary), approaching(adversary, ego).",
            "covered(ego_oncoming) :- hazard(adversary, ego).",
        ],
    },
    "ego: start_stop_or_stationary": {
        "description": "The ego vehicle exhibits a start-stop anomaly, such as hesitation or unexpected standstill in a traffic flow context.",
        "stsl_formula": "Eventually(stationary(ego) AND in_front_of(adversary, ego))",
        "verification_predicates": ["in_front_of"],
        "scenario_intent": "EgoStationaryConflict",
        "controller": "lead_vehicle_braking",
        "controller_params": {
            "placement_hint": "stress the ego with close headway and stopping requirements",
        },
        "scallop_rules": [
            "hazard(ego, adversary) :- stationary(ego), in_front_of(adversary, ego).",
            "covered(ego_start_stop_or_stationary) :- hazard(ego, adversary).",
        ],
    },
    "ego: unknown": {
        "description": "The clip contains an ego-involved anomaly that is not cleanly categorized, so the symbolic abstraction should be treated as an open-world safety conflict.",
        "stsl_formula": "Eventually(out_of_control(ego) OR colliding(ego, adversary))",
        "verification_predicates": ["colliding"],
        "scenario_intent": "OpenWorldConflict",
        "controller": "route_only",
        "controller_params": {
            "placement_hint": "manually specialize this archetype after reviewing clip-level evidence",
        },
        "scallop_rules": [
            "hazard(ego, adversary) :- out_of_control(ego).",
            "hazard(ego, adversary) :- colliding(ego, adversary).",
            "covered(ego_unknown) :- hazard(ego, adversary).",
        ],
    },
    "other: unknown": {
        "description": "The clip contains a non-ego anomaly that is not cleanly categorized, so the symbolic abstraction should be treated as an open-world traffic conflict.",
        "stsl_formula": "Eventually(crossing_path(adversary, ego) OR colliding(adversary, ego))",
        "verification_predicates": ["crossing_path", "colliding"],
        "scenario_intent": "OpenWorldConflict",
        "controller": "route_only",
        "controller_params": {
            "placement_hint": "manually specialize this archetype after reviewing clip-level evidence",
        },
        "scallop_rules": [
            "hazard(adversary, ego) :- crossing_path(adversary, ego).",
            "hazard(adversary, ego) :- colliding(adversary, ego).",
            "covered(other_unknown) :- hazard(adversary, ego).",
        ],
    },
}


def slugify_label(value: str) -> str:
    return value.lower().replace(":", "_").replace(" ", "_")


def normalize_dota_class(raw_label: str) -> str:
    normalized = " ".join(str(raw_label).strip().lower().split())
    return ALIASES.get(normalized, normalized)


def load_dota_metadata(path: Path) -> dict[str, DotaClip]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clips: dict[str, DotaClip] = {}
    for clip_id, item in payload.items():
        if not isinstance(item, dict):
            continue
        clips[str(clip_id)] = DotaClip(
            clip_id=str(clip_id),
            anomaly_class=normalize_dota_class(str(item.get("anomaly_class", ""))),
            anomaly_start=int(item.get("anomaly_start") or 0),
            anomaly_end=int(item.get("anomaly_end") or 0),
            num_frames=int(item.get("num_frames") or 0),
            subset=str(item.get("subset") or "unknown"),
            video_start=int(item.get("video_start") or 0),
            video_end=int(item.get("video_end") or 0),
        )
    return clips


def select_representative_clip(clips: list[DotaClip]) -> DotaClip | None:
    if not clips:
        return None
    ordered = sorted(clips, key=lambda clip: (clip.anomaly_progress_ratio, clip.anomaly_duration_ratio, clip.clip_id))
    return ordered[len(ordered) // 2]


def build_archetype_seed(canonical_class: str) -> dict[str, Any]:
    canonical_class = normalize_dota_class(canonical_class)
    entry = ANOMALY_DESCRIPTIONS.get(canonical_class)
    if entry is not None:
        return {
            "canonical_class": canonical_class,
            **entry,
        }

    role, _, token = canonical_class.partition(":")
    role = role.strip() or "other"
    token = token.strip() or "unknown"
    subject = "ego" if role == "ego" else "adversary"
    description = f"A DoTA anomaly in which {role} exhibits the '{token}' motion pattern under safety-critical traffic conditions."
    return {
        "canonical_class": canonical_class,
        "description": description,
        "stsl_formula": f"Eventually({token.replace('-', '_')}({subject}))",
        "verification_predicates": ["colliding" if role == "ego" else "crossing_path"],
        "scenario_intent": "OpenWorldConflict",
        "controller": "route_only",
        "controller_params": {
            "placement_hint": "specialize this class with manual CARLA mapping before execution",
        },
        "scallop_rules": [
            f"covered({slugify_label(canonical_class)}) :- {token.replace('-', '_')}({subject}).",
        ],
    }


def build_prompt_context(canonical_class: str, clip: DotaClip | None = None, clip_count: int | None = None) -> str:
    seed = build_archetype_seed(canonical_class)
    lines = [
        f"DoTA anomaly class: {canonical_class}",
        f"Seed description: {seed['description']}",
        f"Seed STSL formula: {seed['stsl_formula']}",
        f"Seed scenario intent: {seed['scenario_intent']}",
    ]
    if clip_count is not None:
        lines.append(f"Archetype clip count: {clip_count}")
    if clip is not None:
        lines.extend(
            [
                f"Representative clip id: {clip.clip_id}",
                f"Subset: {clip.subset}",
                f"Anomaly window: frames {clip.anomaly_start}-{clip.anomaly_end} of {clip.num_frames}",
                f"Anomaly progress ratio: {clip.anomaly_progress_ratio:.4f}",
                f"Anomaly duration ratio: {clip.anomaly_duration_ratio:.4f}",
            ]
        )
    return "\n".join(lines)


def resolve_env_file_candidates(explicit_path: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.extend(
        [
            Path.cwd() / ".env",
            WORKSPACE_ROOT / ".env",
            WORKSPACE_ROOT / "research/.env",
            WORKSPACE_ROOT.parent / ".env",
        ]
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve() if candidate.exists() else candidate
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(candidate)
    return unique


def load_dotenv_values(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_openai_api_key(explicit_env_path: Path | None = None) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    for env_path in resolve_env_file_candidates(explicit_env_path):
        values = load_dotenv_values(env_path)
        api_key = values.get("OPENAI_API_KEY")
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            return api_key
    return None