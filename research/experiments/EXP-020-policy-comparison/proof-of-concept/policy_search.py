#!/usr/bin/env python
"""Matched-budget policy search over the threshold-crossing candidate space.

Two execution modes:

- ``server`` (default): evaluations run against a long-lived CARLA server on
  ``--server-port``. The server is (re)started automatically if it is not
  reachable. Each evaluation still runs in a fresh Python process
  (``run_shakedown.py``), which is the isolation the paper's protocol requires.
- ``diagnostics``: legacy hard-restart mode; boots and tears down CARLA per
  candidate via ``run_inter_session_diagnostics.py``. Slower and flakier on
  Linux/OpenGL; kept for compatibility.

Selection policies (all share the same candidate space and dedup rule):

- ``random``   : uniform sampling over the search space.
- ``lsa``      : elite-guided mutation around the best LSA fitness seen.
- ``kmnc``     : elite-guided mutation around the best KMNC fitness seen.
- ``semantic`` : elite-guided mutation around the best semantic candidate. With
  ``--engine-metrics --oracle PATH`` the elite is the candidate that closed the
  most *new* oracle obligations for the arm's suite (ties broken by obligations
  witnessed in the run; collisions are not rewarded). Without the engine flags
  the archived behaviour is kept: most fulfilled obligations plus a collision
  bonus.

Engine scoring path (opt-in, additive; default off): ``--engine-metrics``
requires ``--oracle PATH`` (an EXP-018 inventory). After each evaluation the
arm's semantic stream is scored by ``research/harness/coverage_engine.py`` and
the row receives ``engine_cov_v/a/e/h`` (suite-level, mapped subset),
``engine_new_obligations``, ``engine_run_covered_count``,
``engine_uncovered_count``, ``engine_first_uncover`` and ``engine_error``.
Missing or failed streams log nulls and never abort the search.

``--search-space campaign`` (the pre-registered four-dimensional space) is the
campaign configuration; ``--search-space legacy`` reproduces the pilot's
one-dimensional ``trigger_radius_m`` space so archived sessions and seeds stay
reproducible. Every policy arm must be launched with the same flag and
``--evals`` for the matched-budget comparison.

Every evaluation appends one JSON row to ``rows.jsonl`` so runs can resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.coverage_engine import (  # noqa: E402
    Oracle,
    compute_cov,
    load_ego_route,
    load_oracle,
    load_semantic_trace,
)
from research.harness.criticality import criticality_score  # noqa: E402
from research.harness.hazard_search import (  # noqa: E402
    AdaptiveMutation,
    ExploitTracker,
    HAZARD_TEMPLATES,
    OBLIGATION_TEMPLATE_MAP,
    ObligationScheduler,
)
from research.harness.safety_outcomes import classify_run  # noqa: E402
from research.harness.shared_suite import SharedSuiteStore  # noqa: E402
from research.harness.search_space import (  # noqa: E402
    SearchCandidate,
    SearchSpace,
    SearchSpaceError,
    make_campaign_space,
    make_legacy_space,
)

DIAGNOSTICS_SCRIPT = WORKSPACE_ROOT / "research" / "harness" / "run_inter_session_diagnostics.py"
SHAKEDOWN_SCRIPT = WORKSPACE_ROOT / "research" / "harness" / "run_shakedown.py"
DEFAULT_PYTHON = WORKSPACE_ROOT / "research" / ".venv" / "bin" / "python"
DEFAULT_CARLA_ROOT = Path(os.environ.get("CARLA_ROOT", "/mnt/DevDrive/carla-0.9.16"))
DEFAULT_OBLIGATIONS = [
    "stationary(ego)",
    "in_front_of(pedestrian,ego)",
    "crossing_path(pedestrian,ego)",
    "jaywalking(pedestrian)",
    "colliding(ego,pedestrian)",
]
# "critonly" is an ablation: criticality-guided search with no semantic gap scheduling.
DEFAULT_TEMPLATE_NAME = "pedestrian_crossing"
POLICIES = ("random", "lsa", "kmnc", "semantic", "critonly")
PROTOCOL_VERSION = "scout-search-v3"


def _validate_protocol(args: argparse.Namespace, rows_path: Path, hazard_payloads: dict) -> None:
    """Do not mix corrected search/observer/ADS behavior with archived rows."""
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path is not None and path.is_file() else None

    protocol = {
        "version": PROTOCOL_VERSION, "policy": args.policy, "seed": args.seed,
        "agent_kind": args.agent_kind, "pcla_agent": args.pcla_agent,
        "agent_repo": str(args.agent_repo_path), "agent_config": str(args.agent_config),
        "max_ticks": args.max_ticks, "control": args.control,
        "paired_controls": args.paired_controls,
        "frozen_base": args.frozen_base,
        "engine_metrics": args.engine_metrics, "search_space": _build_search_space(args).signature(),
        "hazard_search": args.hazard_search, "stall_limit": args.hazard_stall_limit,
        "base_spec": digest(args.base_spec), "oracle": digest(args.oracle),
        "coverage_profile": digest(args.coverage_profile),
        "hazard_specs": hazard_payloads,
    }
    # Record search-algorithm markers that changed semantics, so rows produced by
    # a different targeting/aggregation scheme cannot be resumed into this one.
    if getattr(args, "hazard_search", False):
        protocol["hazard_elite"] = "per-obligation-adaptive"
        protocol["criticality"] = args.criticality
        protocol["hazard_epsilon"] = args.hazard_epsilon
        protocol["exploit"] = [args.hazard_exploit_patience, args.hazard_exploit_cap]
    if getattr(args, "shared_suite", False):
        protocol["shared_suite"] = True
    path = args.output_dir / "campaign_protocol.json"
    if rows_path.exists() and rows_path.stat().st_size:
        if not path.exists() or _load_json(path) != protocol:
            raise ValueError("Incompatible campaign checkpoint. Use a new --output-dir; archived results must not be resumed with the corrected protocol.")
    elif path.exists() and _load_json(path) != protocol:
        raise ValueError("Output directory belongs to another campaign configuration; use a new --output-dir.")
    _write_json(path, protocol)


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _carla_ready(python_executable: Path, port: int) -> bool:
    probe = (
        "import carla,sys\n"
        f"c=carla.Client('127.0.0.1',{port});c.set_timeout(10.0)\n"
        "w=c.get_world();print('READY',w.get_map().name)\n"
    )
    try:
        result = subprocess.run(
            [str(python_executable), "-c", probe],
            cwd=WORKSPACE_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0 and "READY" in result.stdout


def _is_carla_process(entry: Path) -> bool:
    """True only for the actual CARLA binary, never for shells that merely mention it."""
    try:
        exe = os.readlink(entry / "exe")
    except (OSError, ProcessLookupError):
        return False
    return os.path.basename(exe) == "CarlaUE4-Linux-Shipping"


def _kill_carla_on_port(port: int) -> None:
    """SIGKILL any CARLA server bound to ``port`` and wait for the port to close."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or not _is_carla_process(entry):
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if f"-carla-rpc-port={port}" in cmdline.split():
            try:
                os.kill(int(entry.name), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    deadline = time.time() + 20
    while time.time() < deadline and _port_open(port):
        time.sleep(1.0)


def _ensure_server(args: argparse.Namespace, port: int) -> None:
    if os.environ.get("SCOUT_CARLA_SUPERVISED"):
        # An external supervisor (research/scripts/carla_supervisor.sh) owns the
        # fleet: it restarts crashed servers as the carla user and drops a
        # warming flag until a warm-up episode has run. Wait for it instead of
        # launching our own server.
        flag = Path(f"/tmp/carla_warming_{port}")
        deadline = time.time() + float(os.environ.get("SCOUT_CARLA_WAIT_S", "900"))
        while time.time() < deadline:
            if not flag.exists() and _carla_ready(args.python_executable, port):
                return
            time.sleep(5.0)
        raise RuntimeError(f"supervised CARLA server on port {port} not ready in time")
    if _carla_ready(args.python_executable, port):
        return
    if _port_open(port):
        # Port is bound but the server is not answering: wedged process.
        _kill_carla_on_port(port)
    launcher = args.carla_root / "CarlaUE4.sh"
    if not launcher.exists():
        raise FileNotFoundError(f"CARLA launcher not found at {launcher}")
    command = [
        str(launcher),
        f"-carla-rpc-port={port}",
        f"-quality-level={args.carla_quality}",
        "-RenderOffScreen",
        "-opengl",
        "-nosound",
    ]
    if args.graphics_adapter is not None:
        command.append(f"-graphicsadapter={args.graphics_adapter}")
    logs_dir = args.output_dir / "server_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"carla_{port}.log"
    log_handle = log_path.open("ab")
    subprocess.Popen(
        command,
        cwd=str(args.carla_root),
        stdout=log_handle,
        stderr=log_handle,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.time() + args.boot_timeout_seconds
    while time.time() < deadline:
        if _carla_ready(args.python_executable, port):
            return
        time.sleep(3.0)
    raise RuntimeError(f"CARLA server on port {port} did not become ready in time")


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _load_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _lead_braking_spec_candidates(base_spec: Path) -> list[Path]:
    """Locate the lead-braking base spec for ``base_spec``.

    Campaign base specs carry the benign-seed suffix (``*_benign_seed0.json``)
    while lead-braking specs are generated per route (``*_lead_braking.json``),
    so the route-level name is checked as well as the suffixed one.
    """
    candidates = [base_spec.with_name(base_spec.stem + "_lead_braking.json")]
    stem = base_spec.stem
    marker = "_benign_seed"
    index = stem.find(marker)
    if index != -1:
        candidates.append(base_spec.with_name(stem[:index] + "_lead_braking.json"))
    return candidates


ENGINE_AXES = ("V", "A", "E", "H")


def _latest_semantic_stream(semantic_dir: Path) -> Path | None:
    """Newest semantic-stream JSONL written under one evaluation's directory."""
    candidates = sorted(
        semantic_dir.glob("*-semantic-stream.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _engine_null_metrics(message: str) -> dict:
    """All engine columns as nulls plus a diagnostic; used on missing/failed streams."""
    return {
        "engine_cov_v": None,
        "engine_cov_a": None,
        "engine_cov_e": None,
        "engine_cov_h": None,
        "engine_new_obligations": None,
        "engine_run_covered_count": None,
        "engine_uncovered_count": None,
        "engine_first_uncover": None,
        "engine_error": message[-500:],
    }


class EngineCoverageState:
    """Suite-level obligation bookkeeping for the additive engine scoring path.

    One instance per arm. Each evaluation's semantic stream is scored with the
    oracle-matched coverage engine; the covered obligations of the run are
    unioned into the arm's suite set. That union is exactly
    ``compute_suite_cov``'s rule: node/attribute/relation obligations union
    across traces, hazard conjunctions credited only within a single trace.

    Failures (missing or malformed stream) return null metrics and an
    ``engine_error`` string instead of raising, so the search never aborts.
    """

    def __init__(self, oracle: Oracle) -> None:
        self.oracle = oracle
        self.suite_covered: set[str] = set()
        self.first_uncover: dict[str, int] = {}
        self._mapped_counts: dict[str, int] | None = None
        self._axis_by_signature: dict[str, str] = {}

    def _learn_mapping(self, report: dict) -> None:
        """Cache the (trace-independent) mapped obligation counts and axis index."""
        if self._mapped_counts is not None:
            return
        counts: dict[str, int] = {}
        for axis in ENGINE_AXES:
            section = report["dimensions"][axis]
            counts[axis] = int(section["mapped_obligations"])
            for row in section["obligations"]:
                if row.get("mapped"):
                    self._axis_by_signature[str(row["signature"])] = axis
        self._mapped_counts = counts

    @property
    def mapped_total(self) -> int:
        return sum(self._mapped_counts.values()) if self._mapped_counts else 0

    def observe_stream(self, semantic_dir: Path, eval_index: int) -> dict:
        """Score one evaluation's stream and return the additive engine columns."""
        stream_path = _latest_semantic_stream(semantic_dir)
        if stream_path is None:
            return _engine_null_metrics(f"no semantic stream under {semantic_dir}")
        try:
            ticks = load_semantic_trace(stream_path)
            ego_route = load_ego_route(stream_path)
            report = compute_cov(
                self.oracle,
                ticks,
                trace_label=str(stream_path),
                ego_route=ego_route,
            )
        except Exception as exc:  # noqa: BLE001 - metric logging must never abort the search
            return _engine_null_metrics(f"coverage engine failed for {stream_path}: {exc!r}")

        self._learn_mapping(report)
        run_covered: set[str] = set()
        for axis in ENGINE_AXES:
            for row in report["dimensions"][axis]["obligations"]:
                if row.get("covered"):
                    run_covered.add(str(row["signature"]))

        new_obligations = sorted(run_covered - self.suite_covered)
        for signature in new_obligations:
            self.first_uncover.setdefault(signature, eval_index)
        self.suite_covered.update(run_covered)

        metrics = {
            "engine_run_obligations": sorted(run_covered),
            "engine_new_obligations": len(new_obligations),
            "engine_run_covered_count": len(run_covered),
            "engine_uncovered_count": self.mapped_total - len(self.suite_covered),
            "engine_first_uncover": dict(sorted(self.first_uncover.items())),
            "engine_error": None,
        }
        counts = self._mapped_counts or {axis: 0 for axis in ENGINE_AXES}
        for axis in ENGINE_AXES:
            covered = sum(
                1
                for signature in self.suite_covered
                if self._axis_by_signature.get(signature) == axis
            )
            total = counts.get(axis, 0)
            metrics[f"engine_cov_{axis.lower()}"] = (covered / total) if total else None
        return metrics

    def restore_from_row(self, row: dict) -> None:
        """Rebuild suite state from previously written engine rows on resume."""
        first_uncover = row.get("engine_first_uncover")
        if not isinstance(first_uncover, dict):
            return
        for signature, eval_index in first_uncover.items():
            if not isinstance(signature, str) or isinstance(eval_index, bool):
                continue
            if not isinstance(eval_index, int):
                continue
            if signature not in self.first_uncover:
                self.first_uncover[signature] = eval_index
                self.suite_covered.add(signature)


class PolicyState:
    """Elite memory for the guided policies over one shared SearchSpace."""

    def __init__(self, policy: str, space: SearchSpace) -> None:
        self.policy = policy
        self.space = space
        self.best_values: dict[str, float] | None = None
        self.best_fitness: float = float("-inf")
        self.best_tiebreak: float = float("-inf")
        self.best_witnesses: set[str] | None = None

    def observe(
        self,
        values: dict[str, float] | None,
        fitness: float | None,
        tiebreak: float | None = None,
        *,
        run_obligations: set[str] | None = None,
        covered_before: set[str] | None = None,
    ) -> None:
        if not values or fitness is None:
            return
        if self.policy == "semantic" and run_obligations is not None and covered_before is not None:
            if self.best_witnesses is not None:
                self.best_fitness = float(len(self.best_witnesses - covered_before))
            fitness = float(len(run_obligations - covered_before))
        if tiebreak is None:
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_values = dict(values)
                self.best_witnesses = set(run_obligations) if run_obligations is not None else None
            return
        if fitness > self.best_fitness or (
            fitness == self.best_fitness and tiebreak > self.best_tiebreak
        ):
            self.best_fitness = fitness
            self.best_tiebreak = tiebreak
            self.best_values = dict(values)
            self.best_witnesses = set(run_obligations) if run_obligations is not None else None

    def next_candidate(self, rng: random.Random, seen: set[tuple]) -> SearchCandidate:
        if self.policy == "random" or self.best_values is None or rng.random() < 0.25:
            return self._sample_uniform(rng, seen)
        for _ in range(64):
            candidate = self._mutate_elite(rng)
            if self.space.key(candidate) not in seen:
                return candidate
        return self._sample_uniform(rng, seen)

    def _mutate_elite(self, rng: random.Random) -> SearchCandidate:
        mutated: dict[str, float] = {}
        assert self.best_values is not None
        for dimension in self.space.dimensions():
            base = float(self.best_values.get(dimension.name, dimension.missing_value()))
            mutated[dimension.name] = base + rng.gauss(0.0, dimension.sigma_or_default)
        return self.space.clamp(mutated)

    def _sample_uniform(self, rng: random.Random, seen: set[tuple]) -> SearchCandidate:
        candidate = self.space.sample(rng)
        for _ in range(200):
            if self.space.key(candidate) not in seen:
                return candidate
            candidate = self.space.sample(rng)
        return candidate


def _values_from_row(space: SearchSpace, row: dict) -> SearchCandidate | None:
    payload = row.get("candidate")
    if isinstance(payload, dict):
        try:
            return space.validate(payload)
        except SearchSpaceError:
            return None
    radius = _as_float(row.get("radius"))
    if radius is not None and "trigger_radius_m" in space.names():
        values = {dimension.name: dimension.missing_value() for dimension in space.dimensions()}
        values["trigger_radius_m"] = radius
        return space.clamp(values)
    return None


def _iter_rows(rows_path: Path):
    if not rows_path.exists():
        return
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _load_done(rows_path: Path, space: SearchSpace) -> tuple[set[tuple], int]:
    done: set[tuple] = set()
    count = 0
    for row in _iter_rows(rows_path):
        count += 1
        candidate = _values_from_row(space, row)
        if candidate is not None:
            done.add(space.key(candidate))
    return done, count


def _semantic_fitness(row: dict, policy: str, *, engine_active: bool = False) -> float | None:
    if policy == "lsa":
        return _as_float(row.get("coverage_lsa_max"))
    if policy == "kmnc":
        return _as_float(row.get("coverage_kmnc"))
    if policy == "critonly":
        return float(row.get("criticality", criticality_score(row)))
    if policy == "semantic":
        if engine_active:
            return _as_float(row.get("engine_new_obligations"))
        fulfilled = row.get("semantic_fulfilled_obligations")
        if isinstance(fulfilled, list):
            return float(len(fulfilled)) + (1.0 if row.get("terminated_by_collision") else 0.0)
        return None
    return None


def _semantic_tiebreak(row: dict, policy: str, *, engine_active: bool = False) -> float | None:
    """Second selection key for the engine-driven semantic elite: run-witnessed count."""
    if policy != "semantic" or not engine_active:
        return None
    return _as_float(row.get("engine_run_covered_count"))


def _inject_obligations(payload: dict, obligations: list[str]) -> None:
    controller_params = payload.setdefault("controller_params", {})
    if not controller_params.get("coverage_obligations"):
        controller_params["coverage_obligations"] = list(obligations)


def _strip_adversary(spec: dict) -> dict:
    stripped = json.loads(json.dumps(spec))
    stripped["walker_count"] = 0
    stripped["controller"] = "route_only"
    stripped["controller_params"] = {}
    stripped["scenario_id"] = f"{stripped.get('scenario_id', 'scenario')}-control"
    stripped["description"] = f"{stripped.get('description', '')} No-adversary control."
    return stripped


def _row_from_run(run_payload: dict, duration: float) -> dict:
    metadata = run_payload.get("metadata") or {}
    coverage = metadata.get("coverage") or {}
    semantic = metadata.get("semantic") or {}
    obligation_credit = semantic.get("obligation_credit") or {}
    collisions = run_payload.get("collisions") or []
    safety = classify_run(run_payload)
    safety_fields = {f"safety_{name}": value for name, value in safety.items() if name != "raw"}
    return {
        "duration_s": round(duration, 2),
        "ticks_executed": run_payload.get("ticks_executed"),
        "reached_goal": bool(run_payload.get("reached_goal")),
        "collision_count": run_payload.get("collision_count"),
        "terminated_by_collision": bool(run_payload.get("terminated_by_collision")),
        "collision_actors": sorted({str(c.get("actor_type")) for c in collisions if c.get("actor_type")}),
        "collision_events": collisions,
        "injected_actor_contact": any(c.get("injected_actor") is True for c in collisions),
        "safety_metrics": run_payload.get("safety_metrics"),
        "safety_raw": safety.get("raw"),
        **safety_fields,
        "coverage_status": coverage.get("status"),
        "coverage_kmnc": _as_float(coverage.get("kmnc")),
        "coverage_lsa_max": _as_float(coverage.get("lsa_max")),
        "coverage_lsa_mean": _as_float(coverage.get("lsa_mean")),
        "semantic_fulfilled_obligations": list(obligation_credit.get("fulfilled_obligations") or []),
        "semantic_missing_obligations": list(obligation_credit.get("missing_obligations") or []),
        "semantic_covered_predicates": list(semantic.get("covered_predicates") or []),
        "semantic_covered_signatures": list(semantic.get("covered_signatures") or []),
        "run_error": None,
    }


def _failure_row(duration: float, message: str) -> dict:
    return {
        "duration_s": round(duration, 2),
        "ticks_executed": None,
        "reached_goal": False,
        "collision_count": None,
        "terminated_by_collision": False,
        "coverage_kmnc": None,
        "coverage_lsa_max": None,
        "semantic_fulfilled_obligations": [],
        "semantic_missing_obligations": [],
        "semantic_covered_predicates": [],
        "run_json_path": None,
        "run_error": message[-2000:],
        "subprocess_returncode": None,
    }


def _run_evaluation_server(args: argparse.Namespace, spec_payload: dict, candidate_id: str, work_dir: Path, port: int) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_path = work_dir / f"{candidate_id}.json"
    _write_json(spec_path, spec_payload)
    started = time.time()
    try:
        _ensure_server(args, port)
    except Exception as exc:  # noqa: BLE001 - record and continue the matrix
        return _failure_row(time.time() - started, f"server unavailable on port {port}: {exc!r}")

    command = [
        str(args.python_executable),
        str(SHAKEDOWN_SCRIPT),
        "--scenario-spec",
        str(spec_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--agent-kind",
        args.agent_kind,
        "--reload-world",
        "--seed",
        str(getattr(args, "execution_seed", args.seed)),
        "--max-ticks",
        str(args.max_ticks),
        "--output-dir",
        str(work_dir / "runs"),
        "--semantic-observer",
        "--semantic-stream-every",
        "1",
        "--semantic-trace-output",
        str(work_dir / "semantic"),
        "--semantic-anomaly-output",
        str(work_dir / "semantic_dumps"),
        "--coverage-observer",
        "--coverage-profile",
        str(args.coverage_profile),
        "--coverage-trace-output",
        str(work_dir / "coverage"),
        "--telemetry-observer",
        "--telemetry-output",
        str(work_dir / "telemetry"),
    ]
    if args.agent_kind == "pcla" and args.pcla_agent:
        command.extend(["--pcla-agent", args.pcla_agent])
    if args.agent_kind == "autovla":
        if args.agent_repo_path is not None:
            command.extend(["--agent-repo-path", str(args.agent_repo_path)])
        if args.agent_config is not None:
            command.extend(["--agent-config", str(args.agent_config)])
        if getattr(args, "autovla_backend", "torch") != "torch":
            command.extend(["--autovla-backend", str(args.autovla_backend)])
            if args.autovla_endpoint:
                command.extend(["--autovla-endpoint", str(args.autovla_endpoint)])
            command.extend([
                "--autovla-timeout", str(args.autovla_timeout),
                "--autovla-model", str(args.autovla_model),
            ])

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(getattr(args, "execution_seed", args.seed))
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    started = time.time()
    last_error = "no attempt executed"
    for attempt in range(2):
        try:
            _ensure_server(args, port)
        except Exception as exc:  # noqa: BLE001 - retry once, then record
            _kill_carla_on_port(port)
            last_error = f"server unavailable on port {port}: {exc!r}"
            continue
        try:
            completed = subprocess.run(
                command,
                cwd=WORKSPACE_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=args.eval_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            _kill_carla_on_port(port)
            last_error = f"evaluation timed out after {args.eval_timeout_seconds}s (attempt {attempt + 1})"
            continue
        duration = time.time() - started
        run_files = sorted((work_dir / "runs").glob("*.json"), key=lambda p: p.stat().st_mtime)
        run_payload = _load_json(run_files[-1]) if run_files else None
        if isinstance(run_payload, dict):
            row = _row_from_run(run_payload, duration)
            row["run_json_path"] = str(run_files[-1])
            row["subprocess_returncode"] = completed.returncode
            if completed.returncode != 0:
                row["run_error"] = completed.stderr[-2000:]
            return row
        last_error = (completed.stderr or completed.stdout)[-2000:]
        _kill_carla_on_port(port)
    return _failure_row(time.time() - started, last_error)


def _run_evaluation_diagnostics(args: argparse.Namespace, spec_payload: dict, candidate_id: str, work_dir: Path, port: int) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_path = work_dir / f"{candidate_id}.json"
    _write_json(spec_path, spec_payload)
    command = [
        str(args.python_executable),
        str(DIAGNOSTICS_SCRIPT),
        "--scenario-spec",
        str(spec_path),
        "--runs",
        "1",
        "--carla-root",
        str(args.carla_root),
        "--agent-kind",
        args.agent_kind,
        "--max-ticks",
        str(args.max_ticks),
        "--port",
        str(port),
        "--boot-timeout-seconds",
        str(args.boot_timeout_seconds),
        "--quality-level",
        str(args.carla_quality),
        "--output-dir",
        str(work_dir / "diagnostics"),
        "--run-output-dir",
        str(work_dir / "runs"),
        "--telemetry-output",
        str(work_dir / "telemetry"),
        "--coverage-trace-output",
        str(work_dir / "coverage"),
        "--semantic-output",
        str(work_dir / "semantic"),
        "--semantic-anomaly-output",
        str(work_dir / "semantic_dumps"),
        "--coverage-profile",
        str(args.coverage_profile),
        "--label",
        candidate_id,
    ]
    if args.agent_kind == "pcla" and args.pcla_agent:
        command.extend(["--pcla-agent", args.pcla_agent])
    if args.agent_kind == "autovla":
        if args.agent_repo_path is not None:
            command.extend(["--agent-repo-path", str(args.agent_repo_path)])
        if args.agent_config is not None:
            command.extend(["--agent-config", str(args.agent_config)])
    env = os.environ.copy()
    if args.graphics_adapter is not None:
        env["MRES_CARLA_GRAPHICS_ADAPTER"] = str(args.graphics_adapter)
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    started = time.time()
    completed = subprocess.run(command, cwd=WORKSPACE_ROOT, env=env, text=True, capture_output=True, check=False)
    duration = time.time() - started
    _reap_orphan_carla(keep_port=port)
    run_files = sorted((work_dir / "runs").glob("*.json"), key=lambda p: p.stat().st_mtime)
    run_payload = _load_json(run_files[-1]) if run_files else None
    if isinstance(run_payload, dict):
        row = _row_from_run(run_payload, duration)
        row["run_json_path"] = str(run_files[-1])
        row["subprocess_returncode"] = completed.returncode
        return row
    return {
        "duration_s": round(duration, 2),
        "ticks_executed": None,
        "reached_goal": False,
        "collision_count": None,
        "terminated_by_collision": False,
        "coverage_kmnc": None,
        "coverage_lsa_max": None,
        "semantic_fulfilled_obligations": [],
        "semantic_missing_obligations": [],
        "semantic_covered_predicates": [],
        "run_json_path": None,
        "run_error": (completed.stderr or completed.stdout)[-2000:],
        "subprocess_returncode": completed.returncode,
    }


def _reap_orphan_carla(keep_port: int) -> None:
    killed = False
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "CarlaUE4-Linux-Shipping" not in cmdline:
            continue
        if f"carla-rpc-port={keep_port}" in cmdline:
            continue
        try:
            os.kill(int(entry.name), signal.SIGTERM)
            killed = True
        except (ProcessLookupError, PermissionError):
            pass
    if killed:
        time.sleep(2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matched-budget policy search over scenario candidates.")
    parser.add_argument("--policy", required=True, choices=POLICIES)
    parser.add_argument("--base-spec", type=Path, required=True)
    parser.add_argument("--route-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("server", "diagnostics"), default="server")
    parser.add_argument("--server-port", type=int, default=2000)
    parser.add_argument("--carla-root", type=Path, default=DEFAULT_CARLA_ROOT)
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--search-space",
        choices=("campaign", "legacy"),
        default="legacy",
        help="Candidate space: 'campaign' is the pre-registered four-dimensional space; 'legacy' is the pilot's 1-D trigger_radius_m space.",
    )
    parser.add_argument("--mutation-path", default="controller_params.trigger_radius_m", help="Legacy space only.")
    parser.add_argument("--mutation-min", type=float, default=5.0, help="Legacy space only.")
    parser.add_argument("--mutation-max", type=float, default=35.0, help="Legacy space only.")
    parser.add_argument("--mutation-sigma", type=float, default=4.0, help="Legacy space only.")
    parser.add_argument("--mutation-decimals", type=int, default=2, help="Legacy space only.")
    parser.add_argument("--evals", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--agent-kind", default="pcla")
    parser.add_argument("--pcla-agent", default="if_if")
    parser.add_argument("--hazard-stall-limit", type=int, default=12,
                        help="Skip a hazard target after this many attempts without a credit.")
    parser.add_argument("--criticality", action=argparse.BooleanOptionalAction, default=True,
                        help="Semantic policy: rank candidates within a target by run criticality "
                             "(min TTC / distance / deceleration / collision). --no-criticality is the ablation.")
    parser.add_argument("--hazard-exploit-patience", type=int, default=3,
                        help="Semantic policy: after a target is witnessed, keep searching it until this many "
                             "consecutive evals fail to raise criticality (0 = advance immediately).")
    parser.add_argument("--hazard-exploit-cap", type=int, default=8,
                        help="Maximum evals spent exploiting one covered target.")
    parser.add_argument("--hazard-epsilon", type=float, default=0.2,
                        help="Probability of a uniform restart sample instead of mutating the elite (guided policies).")
    parser.add_argument("--hazard-search", action="store_true",
                        help="Hazard-obligation-specific search: per-template parameter spaces with an obligation scheduler.")
    parser.add_argument("--shared-suite", action="store_true",
                        help="Share the covered-obligation suite across routes for this policy arm, so target "
                             "selection accounts for coverage found on other routes/maps (cross-map scheduler).")
    parser.add_argument("--shared-suite-path", type=Path, default=None,
                        help="Explicit shared-suite JSON path (default: <campaign root>/shared_suite/<policy>.json).")
    parser.add_argument("--agent-repo-path", type=Path, default=None,
                        help="Agent repository path for external kinds (e.g., autovla).")
    parser.add_argument("--agent-config", type=Path, default=None,
                        help="Agent config/checkpoint path for external kinds (e.g., autovla).")
    parser.add_argument("--autovla-backend", default="torch", choices=["torch", "http", "openai"],
                        help="AutoVLA serving stack passed through to run_shakedown.py.")
    parser.add_argument("--autovla-endpoint", default=None,
                        help="URL for remote AutoVLA backends (http: /plan server; openai: chat-completions).")
    parser.add_argument("--autovla-timeout", type=float, default=60.0)
    parser.add_argument("--autovla-model", default="autovla")
    parser.add_argument("--coverage-profile", type=Path, default=Path("research/logs/coverage/if-if-safe-prefix-profile.joblib"))
    parser.add_argument("--max-ticks", type=int, default=500)
    parser.add_argument("--boot-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--eval-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--graphics-adapter", type=int, default=None)
    parser.add_argument(
        "--carla-quality",
        default=os.environ.get("SCOUT_CARLA_QUALITY", "Epic"),
        choices=("Low", "Medium", "High", "Epic"),
        help="CARLA -quality-level used when booting the simulator "
             "(default: $SCOUT_CARLA_QUALITY or Epic).",
    )
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--obligations", nargs="*", default=None)
    parser.add_argument("--control", action="store_true", help="Run no-adversary controls instead of search.")
    parser.add_argument("--paired-controls", action="store_true", help="Execute and archive a fresh-world no-adversary control for every candidate; control episodes are additional to --evals.")
    parser.add_argument("--frozen-base", action="store_true", help="Repeat the base scenario unchanged (no search); used for frozen-seed ADS validation.")
    parser.add_argument(
        "--oracle",
        type=Path,
        default=None,
        help="EXP-018 oracle inventory JSON; with --engine-metrics enables the additive engine scoring path.",
    )
    parser.add_argument(
        "--engine-metrics",
        action="store_true",
        help=(
            "Score semantic candidates by suite gap closure and log engine_* metrics per row "
            "(requires --oracle). Default off keeps archived behaviour."
        ),
    )
    return parser


def _build_search_space(args: argparse.Namespace) -> SearchSpace:
    if args.search_space == "campaign":
        return make_campaign_space(seed=args.seed)
    if args.mutation_min > args.mutation_max:
        raise ValueError("--mutation-min must be less than or equal to --mutation-max.")
    return make_legacy_space(
        spec_path=args.mutation_path,
        low=args.mutation_min,
        high=args.mutation_max,
        decimals=args.mutation_decimals,
        sigma=args.mutation_sigma,
        seed=args.seed,
    )


class _HazardPolicyState:
    """Per-obligation elitist selection for the hazard-scheduled search.

    Every policy keys its elite by the *selected obligation* (the scheduler
    target) when one is active, so all methods run the same target-driven loop:
    the parameter search tunes the predicate currently being realised and keeps
    that memory if the search returns to it. With no active target (universe
    exhausted, stalled, or the criticality-only ablation) the template name is
    the key, so exploration still reuses an elite.

    Guided policies mutate the elite with an adaptive step size (shrinks on
    improvement, widens on stagnation), mix in ``epsilon`` uniform samples, and
    restart from a uniform sample after a long stall.
    """

    def __init__(self, policy: str, epsilon: float = 0.2) -> None:
        self.policy = policy
        self.epsilon = epsilon
        self.elites: dict[str, tuple[float, float, dict]] = {}  # key -> (fitness, tiebreak, candidate)
        self.adapt: dict[str, AdaptiveMutation] = {}

    def _key(self, template_name: str, target: str | None) -> str:
        return f"target:{target}" if target is not None else template_name

    def next_candidate(self, rng: random.Random, template_name: str, target: str | None = None) -> dict:
        template = HAZARD_TEMPLATES[template_name]
        key = self._key(template_name, target)
        if self.policy == "random" or key not in self.elites:
            return template.space.sample(rng)
        adapt = self.adapt.setdefault(key, AdaptiveMutation())
        if adapt.should_restart():
            adapt.reset()
            return template.space.sample(rng)
        if rng.random() < self.epsilon:
            return template.space.sample(rng)
        return template.space.mutate(self.elites[key][2], rng, scale=adapt.scale)

    def observe(
        self,
        template_name: str,
        target: str | None,
        candidate: dict,
        fitness: float | None,
        tiebreak: float | None,
    ) -> None:
        if fitness is None:
            return
        key = self._key(template_name, target)
        tie = tiebreak if tiebreak is not None else 0.0
        current = self.elites.get(key)
        improved = current is None or (fitness, tie) > (current[0], current[1])
        self.adapt.setdefault(key, AdaptiveMutation()).update(improved)
        if improved:
            self.elites[key] = (float(fitness), float(tie), dict(candidate))


def _target_progress(oracle: Oracle, target: str | None, row: dict, *, use_criticality: bool = False) -> tuple[float, float]:
    """Reward the active obligation, with criticality then constituent witnesses as guidance.

    Lexicographic (covered, tiebreak): coverage of the target always dominates;
    among equally covering runs the more critical one wins, and supporting
    witnesses (0.01 each) break remaining ties and guide the pre-coverage stage.
    """
    witnessed = set(row.get("engine_run_obligations") or [])
    obligation = next((o for o in oracle.obligations if o.signature == target), None)
    if obligation is None:
        return 0.0, 0.0
    supporting = {o.signature for o in oracle.obligations if o.predicate in obligation.required_predicates and o.node_types and o.node_types[0] in obligation.node_types}
    support = float(len(witnessed & supporting))
    if use_criticality:
        return float(target in witnessed), float(row.get("criticality", criticality_score(row))) + 0.01 * support
    return float(target in witnessed), support


def _mapped_hazard_universe(oracle: object) -> set[str]:
    """Obligations the hazard-specific search knows how to target."""
    universe: set[str] = set()
    ungrounded = set(getattr(oracle, "ungrounded", {}) or {})
    for obligation in getattr(oracle, "obligations", []):
        signature = str(obligation.signature)
        stripped = signature[len("hazard("):-1] if signature.startswith("hazard(") else signature
        predicate = str(getattr(obligation, "predicate", ""))
        if predicate in ungrounded:
            continue
        if signature in OBLIGATION_TEMPLATE_MAP or stripped in OBLIGATION_TEMPLATE_MAP:
            universe.add(signature)
    return universe


def _pick_target(
    scheduler: ObligationScheduler,
    covered: set[str],
    universe: set[str],
    attempts: dict[str, int] | None = None,
    stall_limit: int = 12,
    hold: str | None = None,
) -> tuple[str | None, str]:
    """Pick the highest-priority uncovered obligation.

    Targets attempted ``stall_limit`` times without being credited are skipped
    so the search advances across obligations instead of stalling on one that
    the available templates/observer cannot realise. ``hold`` pins a covered
    target for stage-2 criticality exploitation.
    """
    uncovered = universe - covered
    if attempts:
        attempted = {sig for sig, n in attempts.items() if n >= stall_limit}
        uncovered -= attempted
    if hold is not None:
        # Stage-2 exploitation: keep the credited target active (neither
        # covered nor stalled) so the scheduler does not advance past it.
        uncovered.add(hold)
        covered = covered - {hold}
    return scheduler.select(uncovered, covered)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.engine_metrics and args.oracle is None:
        parser.error("--engine-metrics requires --oracle PATH")
    if args.hazard_search and not args.engine_metrics:
        parser.error("--hazard-search requires --engine-metrics and --oracle")
    if args.paired_controls and args.mode != "server":
        parser.error("--paired-controls requires --mode server")
    args.base_spec = args.base_spec if args.base_spec.is_absolute() else (WORKSPACE_ROOT / args.base_spec).resolve()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else (WORKSPACE_ROOT / args.output_dir).resolve()
    args.coverage_profile = args.coverage_profile if args.coverage_profile.is_absolute() else (WORKSPACE_ROOT / args.coverage_profile).resolve()
    if args.oracle is not None:
        args.oracle = args.oracle if args.oracle.is_absolute() else (WORKSPACE_ROOT / args.oracle).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "rows.jsonl"
    obligations = list(args.obligations) if args.obligations else list(DEFAULT_OBLIGATIONS)

    engine_active = bool(args.engine_metrics)
    engine_state: EngineCoverageState | None = None
    if engine_active:
        try:
            engine_state = EngineCoverageState(load_oracle(args.oracle))
        except (OSError, ValueError) as exc:
            parser.error(f"--oracle could not be loaded: {exc}")

    base_payload = json.loads(args.base_spec.read_text(encoding="utf-8"))
    space = _build_search_space(args)

    hazard_payloads: dict[str, dict] = {}
    hazard_universe: set[str] = set()
    hazard_state: _HazardPolicyState | None = None
    hazard_scheduler: ObligationScheduler | None = None
    hazard_target_attempts: dict[str, int] = {}
    exploit = ExploitTracker(patience=args.hazard_exploit_patience, cap=args.hazard_exploit_cap)
    if args.hazard_search:
        hazard_payloads["pedestrian_crossing"] = base_payload
        lead_spec = next((path for path in _lead_braking_spec_candidates(args.base_spec) if path.exists()), None)
        if lead_spec is not None:
            hazard_payloads["lead_vehicle_braking"] = json.loads(lead_spec.read_text(encoding="utf-8"))
        else:
            print(json.dumps({"hazard_search_note": f"lead-braking spec missing for {args.base_spec.stem}; crossing template only"}))
        if engine_state is not None:
            hazard_universe = _mapped_hazard_universe(engine_state.oracle)
            hazard_universe = {sig for sig in hazard_universe if ObligationScheduler(set()).template_for(sig) in hazard_payloads}
        hazard_state = _HazardPolicyState(args.policy, epsilon=args.hazard_epsilon)
        hazard_scheduler = ObligationScheduler(uncovered=set(hazard_universe))

    try:
        _validate_protocol(args, rows_path, hazard_payloads)
    except ValueError as exc:
        parser.error(str(exc))

    run_evaluation = _run_evaluation_server if args.mode == "server" else _run_evaluation_diagnostics
    seen, done_count = _load_done(rows_path, space)
    rng = random.Random(args.seed)
    state = PolicyState(args.policy, space)
    for row in _iter_rows(rows_path):
        if row.get("rng_state"):
            version, internal, gaussian = row["rng_state"]
            rng.setstate((version, tuple(internal), gaussian))
        previous = _values_from_row(space, row)
        state.observe(
            previous.to_dict() if previous is not None else None,
            _semantic_fitness(row, args.policy, engine_active=engine_active),
            _semantic_tiebreak(row, args.policy, engine_active=engine_active),
            run_obligations=set(row.get("engine_run_obligations") or []) if engine_active else None,
            covered_before=set(engine_state.suite_covered) if engine_state is not None else None,
        )
        if engine_state is not None:
            engine_state.restore_from_row(row)
    if args.hazard_search and hazard_state is not None:
        for row in _iter_rows(rows_path):
            if row.get("hazard_target"):
                hazard_target_attempts[str(row["hazard_target"])] = hazard_target_attempts.get(str(row["hazard_target"]), 0) + 1
            template = row.get("template")
            candidate = row.get("candidate")
            if not template or not isinstance(candidate, dict):
                continue
            seen.add((str(template), json.dumps(candidate, sort_keys=True)))
            if hazard_scheduler is not None:
                hazard_scheduler.current = row.get("hazard_target")
            if args.policy == "semantic" and engine_state is not None:
                fitness, tiebreak = _target_progress(engine_state.oracle, row.get("hazard_target"), row, use_criticality=args.criticality)
            else:
                fitness = _semantic_fitness(row, args.policy, engine_active=engine_active)
                tiebreak = _semantic_tiebreak(row, args.policy, engine_active=engine_active)
            hazard_state.observe(str(template), row.get("hazard_target"), candidate, fitness, tiebreak)
            if args.policy == "semantic":
                exploit.observe(
                    row.get("hazard_target"),
                    str(row.get("hazard_target")) in set(row.get("engine_run_obligations") or []),
                    float(row.get("criticality", criticality_score(row))),
                )

    # Cross-route suite: union in coverage found on other routes/maps for this
    # policy arm so target selection reflects the campaign-wide uncovered set.
    shared_store: SharedSuiteStore | None = None
    if args.shared_suite and engine_state is not None and not args.control and not args.frozen_base:
        if args.shared_suite_path is not None:
            shared_path = args.shared_suite_path
            if not shared_path.is_absolute():
                shared_path = (WORKSPACE_ROOT / shared_path).resolve()
        else:
            # <campaign root>/<route>/<policy> -> campaign root
            shared_path = args.output_dir.parent.parent / "shared_suite" / f"{args.policy}.json"
        shared_store = SharedSuiteStore(shared_path)
        engine_state.suite_covered |= shared_store.load()

    if not args.control and not args.frozen_base and args.base_spec.exists() and not obligations:
        obligations = list(DEFAULT_OBLIGATIONS)

    for index in range(done_count, args.evals):
        evaluated_values: dict[str, float] | None = None
        if args.frozen_base:
            payload = _load_json(args.base_spec)
            candidate_id = f"frozen-{index:04d}"
            row = {
                "policy": "frozen",
                "route": args.route_label,
                "eval_index": index,
                "space": "frozen",
                "space_signature": "frozen-base-v1",
                "candidate": {},
            }
        elif args.control:
            candidate_id = f"control-{index:04d}"
            payload = _strip_adversary(base_payload)
            row = {"policy": "control", "route": args.route_label, "eval_index": index}
        else:
            payload = None
            if args.hazard_search and hazard_state is not None and hazard_scheduler is not None:
                covered = set(engine_state.suite_covered) if engine_state is not None else set()
                if args.policy == "critonly":
                    target, template_name = None, DEFAULT_TEMPLATE_NAME
                else:
                    hold = exploit.target if args.policy == "semantic" and exploit.holding(exploit.target) else None
                    target, template_name = _pick_target(
                        hazard_scheduler, covered, hazard_universe,
                        attempts=hazard_target_attempts, stall_limit=args.hazard_stall_limit, hold=hold,
                    )
                if target is None:
                    template_name = sorted(hazard_payloads)[index % len(hazard_payloads)]
                if target is not None:
                    hazard_target_attempts[target] = hazard_target_attempts.get(target, 0) + 1
                template = HAZARD_TEMPLATES.get(template_name, HAZARD_TEMPLATES["pedestrian_crossing"])
                if template.name not in hazard_payloads:
                    raise RuntimeError(f"No executable base specification for target {target}: {template.name}")
                candidate_dict = hazard_state.next_candidate(rng, template.name, target)
                key = (template.name, json.dumps(candidate_dict, sort_keys=True))
                for _ in range(64):
                    if key not in seen:
                        break
                    candidate_dict = template.space.mutate(candidate_dict, rng)
                    key = (template.name, json.dumps(candidate_dict, sort_keys=True))
                else:
                    raise RuntimeError(f"Could not produce a unique candidate for {template.name}")
                seen.add(key)
                payload = template.apply(hazard_payloads[template.name], candidate_dict)
                _inject_obligations(payload, obligations)
                payload["scenario_id"] = f"{base_payload.get('scenario_id', 'scenario')}-{args.policy}-{index:04d}"
                payload["description"] = (
                    f"{base_payload.get('description', '')} Hazard search target {target} via {template.name}; "
                    f"candidate {index} ({json.dumps(candidate_dict, sort_keys=True)})."
                )
                candidate_id = f"{args.policy}-{index:04d}"
                evaluated_values = dict(candidate_dict)
                row = {
                    "policy": args.policy,
                    "route": args.route_label,
                    "eval_index": index,
                    "candidate": dict(candidate_dict),
                    "space": f"hazard:{template.name}",
                    "space_signature": f"hazard-{template.name}-v1",
                    "template": template.name,
                    "hazard_target": target,
                }
                if "trigger_radius_m" in candidate_dict:
                    row["radius"] = candidate_dict["trigger_radius_m"]
            if payload is None:
                candidate: SearchCandidate | None = None
                if index == 0:
                    base_candidate = space.from_payload(base_payload, fill_missing=args.search_space != "legacy")
                    if base_candidate is not None and space.key(base_candidate) not in seen:
                        candidate = base_candidate
                if candidate is None:
                    candidate = state.next_candidate(rng, seen)
                seen.add(space.key(candidate))
                payload = space.apply_to_payload(base_payload, candidate)
                _inject_obligations(payload, obligations)
                payload["scenario_id"] = f"{base_payload.get('scenario_id', 'scenario')}-{args.policy}-{index:04d}"
                payload["description"] = (
                    f"{base_payload.get('description', '')} Policy {args.policy} candidate {index} "
                    f"({space.name}: {json.dumps(candidate.to_dict(), sort_keys=True)})."
                )
                candidate_id = f"{args.policy}-{index:04d}"
                evaluated_values = {name: float(value) for name, value in candidate.to_dict().items()}
                row = {
                    "policy": args.policy,
                    "route": args.route_label,
                    "eval_index": index,
                    "candidate": candidate.to_dict(),
                    "space": space.name,
                    "space_signature": space.signature(),
                }
                radius = candidate.values.get("trigger_radius_m")
                if radius is not None:
                    row["radius"] = radius

        work_dir = args.output_dir / "evaluations" / candidate_id
        port = args.server_port if args.mode == "server" else args.server_port + index
        args.execution_seed = args.seed + index
        if args.paired_controls and not args.control:
            nominal_dir = work_dir / "paired_control"
            nominal = run_evaluation(args=args, spec_payload=_strip_adversary(payload), candidate_id=candidate_id + "-nominal", work_dir=nominal_dir, port=port)
            if engine_state is not None:
                nominal.update(EngineCoverageState(engine_state.oracle).observe_stream(nominal_dir / "semantic", index))
            row["paired_control"] = nominal
        row.update(run_evaluation(args=args, spec_payload=payload, candidate_id=candidate_id, work_dir=work_dir, port=port))
        row["execution_seed"] = args.execution_seed
        row["criticality"] = criticality_score(row)
        covered_before = set(engine_state.suite_covered) if engine_state is not None else None
        if engine_state is not None:
            row.update(engine_state.observe_stream(work_dir / "semantic", index))
            if shared_store is not None:
                shared_store.merge(engine_state.suite_covered)
        row["rng_state"] = rng.getstate()
        row["protocol_version"] = PROTOCOL_VERSION
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if not args.control:
            fitness_value = _semantic_fitness(row, args.policy, engine_active=engine_active)
            tiebreak_value = _semantic_tiebreak(row, args.policy, engine_active=engine_active)
            if args.hazard_search and hazard_state is not None and row.get("template"):
                if args.policy == "semantic" and engine_state is not None:
                    fitness_value, tiebreak_value = _target_progress(engine_state.oracle, row.get("hazard_target"), row, use_criticality=args.criticality)
                hazard_state.observe(
                    str(row.get("template")),
                    row.get("hazard_target"),
                    dict(row.get("candidate") or {}),
                    fitness_value,
                    tiebreak_value,
                )
                if args.policy == "semantic":
                    exploit.observe(
                        row.get("hazard_target"),
                        str(row.get("hazard_target")) in set(row.get("engine_run_obligations") or []),
                        float(row["criticality"]),
                    )
            else:
                state.observe(evaluated_values, fitness_value, tiebreak_value,
                              run_obligations=set(row.get("engine_run_obligations") or []) if engine_active else None,
                              covered_before=covered_before)
        print(json.dumps({
            "eval": row.get("eval_index"),
            "radius": row.get("radius"),
            "candidate": row.get("candidate"),
            "ticks": row.get("ticks_executed"),
            "coll": row.get("collision_count"),
            "goal": row.get("reached_goal"),
            "kmnc": row.get("coverage_kmnc"),
            "lsa_max": row.get("coverage_lsa_max"),
            "fulfilled": len(row.get("semantic_fulfilled_obligations") or []),
            "sec": row.get("duration_s"),
            "err": (row.get("run_error") or "")[:100] or None,
        }), flush=True)

    print(f"complete: {args.output_dir}")


if __name__ == "__main__":
    main()
