"""Run a resumable SCOUT comparison queue within a wall-clock budget."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import runpy
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def stop_job(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_queue(manifest: dict, output: Path, hours: float) -> dict:
    if hours <= 0:
        raise ValueError("Hours must be positive")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "queue_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": {}}
    # Prevent two queues from driving the same synchronous simulator.
    with (Path("/tmp") / f"scout-carla-{manifest['port']}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + hours * 3600
        env = os.environ.copy()
        env.update({"WANDB_MODE": "disabled", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        env.update(manifest.get("env", {}))
        ensure_server = None
        server_args = None
        if manifest.get("server"):
            server = manifest["server"]
            namespace = runpy.run_path(str(ROOT / "research/experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py"))
            ensure_server = namespace["_ensure_server"]
            server_args = argparse.Namespace(
                python_executable=ROOT / "research/.venv/bin/python",
                carla_root=Path(server["carla_root"]), graphics_adapter=server.get("graphics_adapter", 0),
                quality_level=server.get("quality_level", "Epic"), output_dir=output,
                boot_timeout_seconds=240,
            )
        for job in manifest["jobs"]:
            previous = state["jobs"].get(job["id"], {})
            if previous.get("command") not in (None, job["command"]):
                raise ValueError(f"Command changed for {job['id']}; use a new queue output directory")
            if previous.get("status") in {"complete", "nominal_failed"}:
                continue
            requirements = job.get("requires", [])
            if any(state["jobs"].get(key, {}).get("status") != "complete" for key in requirements):
                state["jobs"][job["id"]] = {"command": job["command"], "status": "blocked_by_gate"}
                save_state(state_path, state)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if shutil.disk_usage(output).free < manifest.get("min_free_gb", 10) * 1024**3:
                state["stop_reason"] = "disk_reserve"
                save_state(state_path, state)
                break
            record = {"command": job["command"], "status": "running", "started": time.time()}
            state["jobs"][job["id"]] = record
            save_state(state_path, state)
            print(f"Starting {job['id']}", flush=True)
            started = time.monotonic()
            if ensure_server is not None:
                try:
                    ensure_server(server_args, manifest["port"])
                except Exception as exc:
                    record.update(status="server_failed", error=repr(exc))
                    save_state(state_path, state)
                    break
            with (output / f"{job['id']}.log").open("a") as log:
                process = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=min(remaining, job.get("timeout_seconds", remaining)))
                    record.update(status="complete" if code == 0 else "failed", returncode=code)
                    if code == 0 and job.get("nominal_results"):
                        results = sorted((ROOT / job["nominal_results"]).glob("*.json"),
                                         key=lambda path: path.stat().st_mtime)
                        payload = json.loads(results[-1].read_text()) if results else {}
                        if not payload.get("reached_goal") or payload.get("collision_count") != 0:
                            record["status"] = "nominal_failed"
                except subprocess.TimeoutExpired:
                    stop_job(process)
                    record["status"] = "budget_expired" if time.monotonic() >= deadline else "timeout"
                except BaseException:
                    stop_job(process)
                    record["status"] = "interrupted"
                    save_state(state_path, state)
                    raise
            record["wall_seconds"] = time.monotonic() - started
            save_state(state_path, state)
        return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hours", type=float, default=8)
    args = parser.parse_args()
    run_queue(json.loads(args.manifest.read_text()), args.output_dir, args.hours)


if __name__ == "__main__":
    main()
