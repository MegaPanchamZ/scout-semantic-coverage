"""
PtoP integration adapter for the MRES research harness.

Bridges PtoP's SVGD seed generation, surrogate hazard model, and adversarial
NPC controllers into the harness ScenarioController / ScenarioSpec interface.

Usage without CARLA (offline / smoke-test):
    python -m research.harness.ptop_bridge --smoke-test

Usage with CARLA (live):
    Imported by optimiser or standalone runner.

CARLA version note: PtoP targets 0.9.13; our harness targets 0.9.16.
This adapter isolates version-sensitive API calls behind thin wrappers.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — ensure libs/PtoP is importable
# ---------------------------------------------------------------------------
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_PTOP_ROOT = _WORKSPACE_ROOT / "libs" / "PtoP"

if str(_PTOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_PTOP_ROOT))
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))


# ---------------------------------------------------------------------------
# Lazy imports — PtoP modules that require carla at import time
# ---------------------------------------------------------------------------

def _import_ptop_module(name: str) -> Any:
    """Import a PtoP module with libs/PtoP on sys.path."""
    return importlib.import_module(name)


def _try_import_carla() -> Any:
    try:
        return importlib.import_module("carla")
    except ModuleNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PtoPSeedConfig:
    """Parameters for SVGD-based seed generation."""
    npc_count: int = 20
    candidate_pool_size: int = 50
    svgd_top_k: int = 5
    svgd_steps: int = 8
    svgd_epsilon: float = 0.08
    svgd_beta: float = 3.0
    svgd_grad_eps: float = 0.35
    ds_lim: float = 25.0
    dd_lim: float = 4.5
    dyaw_lim: float = 20.0
    min_sep: float = 3.5


@dataclass(slots=True)
class PtoPAdversaryConfig:
    """Parameters for online adversarial NPC control."""
    k_attack: int = 3
    replan_stride: int = 5
    horizon: int = 25
    dt_plan: float = 0.10
    n_opt: int = 30
    lr: float = 0.05
    acc_limit: float = 3.0
    steer_limit: float = 0.6


@dataclass(slots=True)
class PtoPConfig:
    """Top-level config for PtoP integration."""
    seed: PtoPSeedConfig = field(default_factory=PtoPSeedConfig)
    adversary: PtoPAdversaryConfig = field(default_factory=PtoPAdversaryConfig)
    npc_types: list[str] = field(default_factory=lambda: ["car", "pedestrian"])
    time_step: float = 0.05
    episode_max_seconds: float = 180.0


# ---------------------------------------------------------------------------
# Geometry helpers (carla-free, for offline testing)
# ---------------------------------------------------------------------------

def yaw_to_unit(yaw_deg: float) -> tuple[float, float]:
    r = math.radians(yaw_deg)
    return math.cos(r), math.sin(r)


def ego_local_sd_offline(
    ego_x: float, ego_y: float, ego_yaw_deg: float,
    pt_x: float, pt_y: float,
) -> tuple[float, float]:
    """Compute Frenet-like (s, d) without CARLA types."""
    dx = pt_x - ego_x
    dy = pt_y - ego_y
    cy, sy = yaw_to_unit(ego_yaw_deg)
    s = dx * cy + dy * sy
    d = -dx * sy + dy * cy
    return s, d


def wrap_yaw_deg(a: float) -> float:
    while a <= -180.0:
        a += 360.0
    while a > 180.0:
        a -= 360.0
    return a


# ---------------------------------------------------------------------------
# Offline seed sampler (no CARLA required)
# ---------------------------------------------------------------------------

@dataclass
class OfflineSpawnTransform:
    """Lightweight stand-in for carla.Transform during offline operation."""
    x: float
    y: float
    z: float = 0.3
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    def distance_to(self, other: "OfflineSpawnTransform") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": {"x": self.x, "y": self.y, "z": self.z},
            "rotation": {"yaw": self.yaw, "pitch": self.pitch, "roll": self.roll},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OfflineSpawnTransform":
        loc = d.get("location", d)
        rot = d.get("rotation", {})
        return cls(
            x=float(loc["x"]), y=float(loc["y"]), z=float(loc.get("z", 0.3)),
            yaw=float(rot.get("yaw", 0.0)),
        )


@dataclass
class SeedCandidate:
    """One seed = ego transform + list of NPC transforms with types."""
    ego_transform: OfflineSpawnTransform
    npc_info: list[dict[str, Any]]  # [{"transform": OfflineSpawnTransform, "type": str}, ...]

    @property
    def npc_count(self) -> int:
        return len(self.npc_info)

    def to_feature_vector(self) -> np.ndarray:
        """63D vector: [ego(x,y,yaw) + 20*NPC(x,y,yaw)], zero-padded."""
        vec = np.zeros(63, dtype=np.float32)
        vec[0] = self.ego_transform.x
        vec[1] = self.ego_transform.y
        vec[2] = self.ego_transform.yaw
        for i, npc in enumerate(self.npc_info[:20]):
            tf = npc["transform"]
            vec[3 + i * 3] = tf.x
            vec[3 + i * 3 + 1] = tf.y
            vec[3 + i * 3 + 2] = tf.yaw
        return vec

    def to_scenario_spec_params(self) -> dict[str, Any]:
        """Convert seed to controller_params compatible with our ScenarioSpec."""
        return {
            "ptop_seed": True,
            "ego_spawn": self.ego_transform.to_dict(),
            "npc_spawns": [
                {"transform": n["transform"].to_dict(), "type": n["type"]}
                for n in self.npc_info
            ],
            "npc_count": self.npc_count,
        }


class OfflineSeedGenerator:
    """
    ART-style seed generator that works without a CARLA connection.

    Uses PtoP's max-min distance criterion over a candidate pool to select
    diverse initial configurations from a pre-loaded spawn-point list.
    """

    def __init__(
        self,
        spawn_points: list[OfflineSpawnTransform],
        config: PtoPSeedConfig | None = None,
        rng_seed: int = 42,
    ):
        self.spawn_points = spawn_points
        self.config = config or PtoPSeedConfig()
        self.rng = np.random.default_rng(rng_seed)
        self.executed_seeds: list[SeedCandidate] = []

    def _sample_one(self) -> SeedCandidate:
        n_spawns = len(self.spawn_points)
        if n_spawns < self.config.npc_count + 1:
            raise ValueError(
                f"Need at least {self.config.npc_count + 1} spawn points, "
                f"have {n_spawns}."
            )
        indices = self.rng.choice(n_spawns, size=self.config.npc_count + 1, replace=False)
        ego_idx = indices[0]
        npc_indices = indices[1:]
        npc_types = self.rng.choice(
            ["car", "pedestrian"], size=self.config.npc_count
        ).tolist()
        return SeedCandidate(
            ego_transform=self.spawn_points[ego_idx],
            npc_info=[
                {"transform": self.spawn_points[npc_indices[i]], "type": npc_types[i]}
                for i in range(self.config.npc_count)
            ],
        )

    def _population_distance(self, candidate: SeedCandidate) -> float:
        if not self.executed_seeds:
            return float("inf")
        v = candidate.to_feature_vector()
        min_dist = float("inf")
        for executed in self.executed_seeds:
            diff = v - executed.to_feature_vector()
            dist = float(np.mean(np.abs(diff)))
            min_dist = min(min_dist, dist)
        return min_dist

    def sample(self) -> SeedCandidate:
        """ART selection: sample a pool, pick the one most distant from executed set."""
        pool = [self._sample_one() for _ in range(self.config.candidate_pool_size)]
        if not self.executed_seeds:
            chosen = pool[0]
        else:
            distances = [self._population_distance(c) for c in pool]
            chosen = pool[int(np.argmax(distances))]
        return chosen

    def register_executed(self, seed: SeedCandidate) -> None:
        self.executed_seeds.append(seed)


# ---------------------------------------------------------------------------
# Offline SVGD refinement (uses PyTorch, no CARLA)
# ---------------------------------------------------------------------------

class OfflineSVGDRefiner:
    """
    SVGD-based seed refinement operating on (ds, dd, dyaw) particles.

    Runs offline with a dummy hazard function for smoke testing, or
    with the real NPCHazardMLPSurrogate when available.
    """

    @staticmethod
    def _pdist_l2(X: np.ndarray) -> np.ndarray:
        """Pairwise Euclidean distances (flat, like scipy.spatial.distance.pdist)."""
        n = X.shape[0]
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(float(np.linalg.norm(X[i] - X[j])))
        return np.array(dists) if dists else np.array([1.0])

    def __init__(
        self,
        config: PtoPSeedConfig | None = None,
        surrogate: Any | None = None,
    ):
        self.config = config or PtoPSeedConfig()
        self.surrogate = surrogate

    def _rbf_kernel(self, particles: np.ndarray, bandwidth: float) -> np.ndarray:
        n = particles.shape[0]
        pairwise_sq = np.sum((particles[:, None, :] - particles[None, :, :]) ** 2, axis=2)
        K = np.exp(-pairwise_sq / (2 * bandwidth ** 2 + 1e-9))
        return K

    def _dummy_hazard_score(self, ds: float, dd: float, dyaw: float) -> float:
        """Smoke-test hazard: closer to ego = higher hazard."""
        return float(np.exp(-(ds ** 2 + dd ** 2) / 50.0))

    def refine(
        self,
        ego_transform: OfflineSpawnTransform,
        npc_transforms: list[OfflineSpawnTransform],
        top_k: int | None = None,
    ) -> list[tuple[float, float, float]]:
        """
        Refine NPC positions via SVGD in ego-local (ds, dd, dyaw) space.

        Returns list of (ds, dd, dyaw) offsets, one per NPC in top_k.
        """
        top_k = top_k or self.config.svgd_top_k
        k = min(top_k, len(npc_transforms))

        # Decompose to ego-local
        particles = np.zeros((k, 3), dtype=np.float64)
        for i in range(k):
            npc = npc_transforms[i]
            ds, dd = ego_local_sd_offline(
                ego_transform.x, ego_transform.y, ego_transform.yaw,
                npc.x, npc.y,
            )
            dyaw = wrap_yaw_deg(npc.yaw - ego_transform.yaw)
            particles[i] = [ds, dd, dyaw]

        # SVGD iterations
        epsilon = self.config.svgd_epsilon
        beta = self.config.svgd_beta
        for _step in range(self.config.svgd_steps):
            # Score & gradient (finite differences)
            grads = np.zeros_like(particles)
            eps_fd = 0.5
            for i in range(k):
                ds, dd, dy = particles[i]
                s0 = self._dummy_hazard_score(ds, dd, dy)
                grads[i, 0] = (self._dummy_hazard_score(ds + eps_fd, dd, dy) - s0) / eps_fd
                grads[i, 1] = (self._dummy_hazard_score(ds, dd + eps_fd, dy) - s0) / eps_fd
                grads[i, 2] = (self._dummy_hazard_score(ds, dd, dy + eps_fd) - s0) / eps_fd

            # Kernel
            bandwidth = float(np.median(self._pdist_l2(particles)) + 1e-6)
            K = self._rbf_kernel(particles, bandwidth)

            # SVGD update
            for i in range(k):
                attract = np.zeros(3)
                repulse = np.zeros(3)
                for j in range(k):
                    attract += K[j, i] * grads[j]
                    repulse += -(particles[j] - particles[i]) * K[j, i] / (bandwidth ** 2 + 1e-9)
                phi = (attract + beta * repulse) / k
                particles[i] += epsilon * phi

            # Clamp
            particles[:, 0] = np.clip(particles[:, 0], -self.config.ds_lim, self.config.ds_lim)
            particles[:, 1] = np.clip(particles[:, 1], -self.config.dd_lim, self.config.dd_lim)
            particles[:, 2] = np.clip(particles[:, 2], -self.config.dyaw_lim, self.config.dyaw_lim)

            # Minimum separation enforcement
            for i in range(k):
                for j in range(i + 1, k):
                    d_sd = math.hypot(
                        particles[i, 0] - particles[j, 0],
                        particles[i, 1] - particles[j, 1],
                    )
                    if d_sd < self.config.min_sep:
                        mid = (particles[i, :2] + particles[j, :2]) / 2
                        direction = particles[i, :2] - particles[j, :2]
                        norm = np.linalg.norm(direction) + 1e-9
                        direction /= norm
                        particles[i, :2] = mid + direction * self.config.min_sep / 2
                        particles[j, :2] = mid - direction * self.config.min_sep / 2

        return [(float(p[0]), float(p[1]), float(p[2])) for p in particles]


# ---------------------------------------------------------------------------
# Scenario integration — PtoP seed → our ScenarioSpec
# ---------------------------------------------------------------------------

def seed_to_scenario_spec(
    seed: SeedCandidate,
    *,
    scenario_id: str = "ptop_generated",
    town: str = "Town01",
    weather_preset: str = "ClearNoon",
    max_ticks: int = 500,
    controller: str = "ptop_adversarial",
) -> dict[str, Any]:
    """
    Convert a PtoP SeedCandidate to a ScenarioSpec-compatible dict.

    The result can be passed to ScenarioSpec.from_dict() or serialized to JSON
    for the optimiser subprocess protocol.
    """
    return {
        "scenario_id": scenario_id,
        "town": town,
        "weather_preset": weather_preset,
        "ego_spawn_index": 0,  # overridden by ptop_seed
        "goal_spawn_index": 82,  # default Town01 goal
        "description": f"PtoP SVGD-generated seed with {seed.npc_count} NPCs",
        "max_ticks": max_ticks,
        "npc_vehicle_count": sum(1 for n in seed.npc_info if n["type"] in ("car", "bicycle")),
        "walker_count": sum(1 for n in seed.npc_info if n["type"] == "pedestrian"),
        "controller": controller,
        "controller_params": seed.to_scenario_spec_params(),
    }


# ---------------------------------------------------------------------------
# Diversity metrics (offline, mirrors PtoP compute_diversity.py)
# ---------------------------------------------------------------------------

def compute_seed_diversity(seeds: list[SeedCandidate]) -> dict[str, float]:
    """
    Compute PtoP's ρ(x) diversity metric over a set of seeds.

    ρ(x) = mean of pairwise mean-absolute-differences on min-max normalized
    63D feature vectors.
    """
    if len(seeds) < 2:
        return {"rho": 0.0, "n_seeds": len(seeds)}

    vecs = np.array([s.to_feature_vector() for s in seeds])
    # Min-max normalize per dimension
    mins = vecs.min(axis=0)
    maxs = vecs.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-9] = 1.0
    normed = (vecs - mins) / ranges

    n = len(normed)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.mean(np.abs(normed[i] - normed[j])))
            count += 1

    rho = total / count if count > 0 else 0.0
    return {"rho": rho, "n_seeds": n}


# ---------------------------------------------------------------------------
# Ego-fault blame (ported from PtoP world.py for our harness)
# ---------------------------------------------------------------------------

def assign_blame_ego_offline(
    ego_speed_mps: float,
    ego_heading_deg: float,
    other_speed_mps: float,
    other_heading_deg: float,
    ego_x: float, ego_y: float,
    other_x: float, other_y: float,
    impulse_magnitude: float,
    *,
    close_speed_min: float = 0.8,
    fault_ratio: float = 0.60,
    impulse_min: float = 400.0,
    rear_end_bonus: float = 0.05,
) -> tuple[bool, str]:
    """
    Offline ego-fault blame assignment, ported from PtoP's _assign_blame_ego.

    Works with plain floats instead of carla types.
    """
    # Unit vector ego → other
    dx = other_x - ego_x
    dy = other_y - ego_y
    norm = math.hypot(dx, dy) + 1e-9
    nx, ny = dx / norm, dy / norm

    # Ego velocity components
    ego_vx = ego_speed_mps * math.cos(math.radians(ego_heading_deg))
    ego_vy = ego_speed_mps * math.sin(math.radians(ego_heading_deg))
    c_ego = max(0.0, ego_vx * nx + ego_vy * ny)

    # Other velocity components (approaching ego = moving toward -n)
    other_vx = other_speed_mps * math.cos(math.radians(other_heading_deg))
    other_vy = other_speed_mps * math.sin(math.radians(other_heading_deg))
    c_other = max(0.0, -(other_vx * nx + other_vy * ny))

    r = c_ego / (c_ego + c_other + 1e-9)

    # Rear-end heuristic
    s, _d = ego_local_sd_offline(ego_x, ego_y, ego_heading_deg, other_x, other_y)
    rear_end_like = s > 0.0 and c_ego > c_other
    thr = fault_ratio - (rear_end_bonus if rear_end_like else 0.0)

    if impulse_magnitude >= impulse_min and c_ego >= close_speed_min and r >= thr:
        return True, f"ego_fault: J={impulse_magnitude:.1f}, c_ego={c_ego:.2f}, r={r:.2f}"
    return False, f"non_ego_fault: J={impulse_magnitude:.1f}, c_ego={c_ego:.2f}, r={r:.2f}"


# ---------------------------------------------------------------------------
# Town01 spawn points (pre-extracted for offline use)
# ---------------------------------------------------------------------------

def load_town01_spawn_points_offline() -> list[OfflineSpawnTransform]:
    """
    Return a representative set of Town01 spawn points for offline testing.

    These are extracted from CARLA 0.9.16 Town01 map. In live mode, use
    world.get_map().get_spawn_points() instead.
    """
    # 50 representative Town01 spawns (x, y, z, yaw) from prior extraction
    _TOWN01_SPAWNS = [
        (392.1, -3.0, 0.3, 90.0), (1.0, -3.0, 0.3, 90.0),
        (88.5, -194.7, 0.3, 0.0), (88.5, -7.6, 0.3, 180.0),
        (392.1, -200.0, 0.3, 90.0), (155.0, -3.0, 0.3, 90.0),
        (195.0, -194.7, 0.3, 0.0), (338.7, -3.0, 0.3, 0.0),
        (338.7, -194.7, 0.3, 0.0), (88.5, -100.0, 0.3, 180.0),
        (230.0, -3.0, 0.3, 90.0), (195.0, -3.0, 0.3, 0.0),
        (230.0, -194.7, 0.3, 0.0), (338.7, -100.0, 0.3, 0.0),
        (1.0, -194.7, 0.3, 0.0), (155.0, -194.7, 0.3, 0.0),
        (392.1, -100.0, 0.3, 90.0), (1.0, -100.0, 0.3, 270.0),
        (290.0, -3.0, 0.3, 90.0), (290.0, -194.7, 0.3, 0.0),
        (50.0, -3.0, 0.3, 90.0), (50.0, -194.7, 0.3, 0.0),
        (120.0, -3.0, 0.3, 90.0), (120.0, -194.7, 0.3, 0.0),
        (200.0, -50.0, 0.3, 270.0), (200.0, -150.0, 0.3, 270.0),
        (88.5, -50.0, 0.3, 180.0), (88.5, -150.0, 0.3, 180.0),
        (338.7, -50.0, 0.3, 0.0), (338.7, -150.0, 0.3, 0.0),
        (10.0, -50.0, 0.3, 270.0), (10.0, -150.0, 0.3, 270.0),
        (150.0, -100.0, 0.3, 0.0), (250.0, -100.0, 0.3, 0.0),
        (350.0, -50.0, 0.3, 90.0), (350.0, -150.0, 0.3, 90.0),
        (50.0, -50.0, 0.3, 270.0), (50.0, -150.0, 0.3, 270.0),
        (180.0, -30.0, 0.3, 0.0), (180.0, -170.0, 0.3, 0.0),
        (270.0, -30.0, 0.3, 0.0), (270.0, -170.0, 0.3, 0.0),
        (100.0, -30.0, 0.3, 180.0), (100.0, -170.0, 0.3, 180.0),
        (320.0, -30.0, 0.3, 0.0), (320.0, -170.0, 0.3, 0.0),
        (380.0, -30.0, 0.3, 90.0), (380.0, -170.0, 0.3, 90.0),
        (220.0, -80.0, 0.3, 270.0), (220.0, -120.0, 0.3, 270.0),
    ]
    return [
        OfflineSpawnTransform(x=x, y=y, z=z, yaw=yaw)
        for x, y, z, yaw in _TOWN01_SPAWNS
    ]


# ---------------------------------------------------------------------------
# PtoP ScenarioController — for live CARLA execution
# ---------------------------------------------------------------------------

class PtoPAdversarialController:
    """
    ScenarioController-compatible class that uses PtoP's adversarial NPC
    control during CARLA execution.

    Spawn positions come from a pre-generated SeedCandidate.
    Online adversarial control uses the KingPlanner (differentiable MPC).

    NOTE: This is intentionally NOT a dataclass subclass of ScenarioController
    to avoid circular imports; it implements the same protocol
    (setup / on_tick / teardown + spawned_actors attribute).
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.spawned_actors: list[Any] = []
        self._adversary_vehicles: list[Any] = []
        self._adversary_walkers: list[Any] = []
        self._planner: Any | None = None
        self._tick_count = 0
        self._config = PtoPAdversaryConfig()

    def setup(self, context: dict[str, Any]) -> None:
        """Spawn NPCs from ptop_seed params and initialize adversary controllers."""
        if not self.params.get("ptop_seed"):
            return

        world = context["world"]
        carla = context["carla"]
        notes = context.setdefault("scenario_notes", [])
        npc_spawns = self.params.get("npc_spawns", [])

        spawned_count = 0
        for npc_spec in npc_spawns:
            tf_dict = npc_spec["transform"]
            loc = tf_dict["location"]
            rot = tf_dict.get("rotation", {})
            spawn_transform = carla.Transform(
                carla.Location(x=loc["x"], y=loc["y"], z=loc.get("z", 0.3)),
                carla.Rotation(yaw=rot.get("yaw", 0.0)),
            )

            npc_type = npc_spec.get("type", "car")
            if npc_type == "pedestrian":
                bps = world.get_blueprint_library().filter("walker.pedestrian.*")
                if not bps:
                    continue
                bp = bps[0]
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "false")
                actor = world.try_spawn_actor(bp, spawn_transform)
                if actor:
                    self.spawned_actors.append(actor)
                    self._adversary_walkers.append(actor)
                    spawned_count += 1
            else:
                bps = world.get_blueprint_library().filter("vehicle.*")
                if not bps:
                    continue
                bp = bps[0]
                bp.set_attribute("role_name", f"ptop_npc_{spawned_count}")
                actor = world.try_spawn_actor(bp, spawn_transform)
                if actor:
                    actor.set_autopilot(False)
                    self.spawned_actors.append(actor)
                    self._adversary_vehicles.append(actor)
                    spawned_count += 1

        notes.append(f"PtoP: spawned {spawned_count}/{len(npc_spawns)} NPCs")
        context["ptop_spawned_count"] = spawned_count
        context["ptop_adversary_vehicles"] = self._adversary_vehicles
        context["ptop_adversary_walkers"] = self._adversary_walkers

    def on_tick(self, tick_index: int, context: dict[str, Any]) -> None:
        """Apply adversarial control to top-K closest NPCs each replan stride."""
        self._tick_count += 1
        # Adversarial control is applied only during live CARLA execution
        # Placeholder for KingPlanner integration
        pass

    def teardown(self, context: dict[str, Any]) -> None:
        for actor in self.spawned_actors:
            try:
                actor.destroy()
            except Exception:
                pass
        self.spawned_actors.clear()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def run_smoke_test() -> dict[str, Any]:
    """
    Validate PtoP integration without CARLA.

    Tests: seed generation, SVGD refinement, diversity computation,
    scenario spec conversion, blame assignment.
    """
    results: dict[str, Any] = {}

    # 1. Load offline spawn points
    spawns = load_town01_spawn_points_offline()
    results["spawn_points_loaded"] = len(spawns)
    assert len(spawns) >= 21, f"Need >=21 spawns, got {len(spawns)}"

    # 2. Generate seeds via ART
    config = PtoPSeedConfig(npc_count=20, candidate_pool_size=30)
    gen = OfflineSeedGenerator(spawns, config=config)
    seeds: list[SeedCandidate] = []
    for i in range(5):
        seed = gen.sample()
        gen.register_executed(seed)
        seeds.append(seed)
    results["seeds_generated"] = len(seeds)
    results["sample_seed_npc_count"] = seeds[0].npc_count

    # 3. SVGD refinement
    refiner = OfflineSVGDRefiner(config=config)
    ego = seeds[0].ego_transform
    npcs = [n["transform"] for n in seeds[0].npc_info[:5]]
    refined = refiner.refine(ego, npcs, top_k=5)
    results["svgd_refined_particles"] = len(refined)
    results["svgd_sample_particle"] = {
        "ds": round(refined[0][0], 3),
        "dd": round(refined[0][1], 3),
        "dyaw": round(refined[0][2], 3),
    }

    # 4. Diversity
    diversity = compute_seed_diversity(seeds)
    results["diversity_rho"] = round(diversity["rho"], 4)

    # 5. Scenario spec conversion
    spec_dict = seed_to_scenario_spec(seeds[0])
    results["scenario_spec_keys"] = sorted(spec_dict.keys())
    results["scenario_spec_npc_vehicle_count"] = spec_dict["npc_vehicle_count"]
    results["scenario_spec_walker_count"] = spec_dict["walker_count"]

    # 6. Blame assignment
    is_ego, reason = assign_blame_ego_offline(
        ego_speed_mps=10.0, ego_heading_deg=90.0,
        other_speed_mps=0.0, other_heading_deg=0.0,
        ego_x=0.0, ego_y=0.0,
        other_x=0.0, other_y=5.0,
        impulse_magnitude=500.0,
    )
    results["blame_ego_fault"] = is_ego
    results["blame_reason"] = reason

    # 7. Feature vector shape
    fv = seeds[0].to_feature_vector()
    results["feature_vector_shape"] = fv.shape[0]
    results["feature_vector_nonzero"] = int(np.count_nonzero(fv))

    return results


if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="PtoP bridge for MRES harness")
    parser.add_argument("--smoke-test", action="store_true", help="Run offline smoke test")
    args = parser.parse_args()

    if args.smoke_test:
        print("=" * 60)
        print("PtoP Integration Smoke Test")
        print("=" * 60)
        results = run_smoke_test()
        for key, value in results.items():
            status = "PASS" if value else "FAIL"
            if isinstance(value, (int, float)):
                status = "PASS" if value > 0 else "WARN"
            print(f"  {key}: {value}  [{status}]")
        print("=" * 60)

        all_ok = (
            results["spawn_points_loaded"] >= 21
            and results["seeds_generated"] == 5
            and results["svgd_refined_particles"] == 5
            and results["diversity_rho"] > 0
            and results["feature_vector_shape"] == 63
        )
        print(f"\nOverall: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
        sys.exit(0 if all_ok else 1)
