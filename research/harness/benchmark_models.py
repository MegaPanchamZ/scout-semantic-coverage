"""Measure native ADS runs with all SCOUT observers on an existing CARLA server."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.harness.make_overnight_queue import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["interfuser", "lav", "transfuser"])
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--cuda-device", default="1")
    parser.add_argument("--scenario-spec", type=Path, default=Path("research/logs/nominal_policy_validation_strict/scenario1.json"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    lock = (Path("/tmp") / f"scout-carla-{args.port}.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"CARLA port {args.port} is already owned by a campaign queue")
    summary = []
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.cuda_device, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               WANDB_MODE="disabled", SCOUT_KEEP_SYNCHRONOUS="1")
    for model in args.models:
        python, flags = MODELS[model]
        output = args.output_dir / model
        output.mkdir(parents=True, exist_ok=True)
        command = [python, "research/harness/run_shakedown.py", "--scenario-spec", str(args.scenario_spec),
                   "--port", str(args.port), "--reload-world", "--seed", "13", "--max-ticks", "600",
                   "--output-dir", str(output / "runs"), "--coverage-observer", "--coverage-trace-output", str(output / "coverage"),
                   "--semantic-observer", "--semantic-trace-output", str(output / "semantic"),
                   "--semantic-anomaly-output", str(output / "semantic_dumps"), "--telemetry-observer",
                   "--telemetry-output", str(output / "telemetry")] + flags
        started = time.perf_counter()
        with (output / "process.log").open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        row = {"model": model, "port": args.port, "cuda_device": args.cuda_device,
               "wall_seconds": time.perf_counter() - started, "returncode": completed.returncode}
        paths = list((output / "runs").glob("*.json"))
        if paths and completed.returncode == 0:
            result = json.loads(max(paths, key=lambda path: path.stat().st_mtime).read_text())
            row.update({key: result[key] for key in ("ticks_executed", "reached_goal", "collision_count", "metadata")})
        summary.append(row)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({key: value for key, value in row.items() if key != "metadata"}), flush=True)


if __name__ == "__main__":
    main()
