"""Mine failure-causing semantic obligations/instances from archived semantic streams.

For every failing execution (collision_count > 0) in a search root, score its
semantic stream with the oracle-matched coverage engine and union the witnessed
obligations. Per arm (route x policy) we report:
  - failure_causing_obligations: distinct obligations witnessed in failing runs
  - failure_causing_instances:  distinct (obligation, trigger-radius bucket) pairs
    (5 m buckets of the candidate trigger radius, as a documented proxy for
    "the same obligation exercised at different staging distances")

Usage:
  research/.venv/bin/python research/scripts/mine_failure_obligations.py \
      --search-root research/logs/fse_search --ads interfuser \
      --oracle research/experiments/EXP-018-.../oracle_inventory_v1.0-trainval.json \
      --output research/logs/failure_obligations/interfuser.json
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

WORKSPACE = pathlib.Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from research.harness.coverage_engine import (  # noqa: E402
    compute_cov,
    load_ego_route,
    load_oracle,
    load_semantic_trace,
)

ORACLE = None


def _init(oracle_path: str) -> None:
    global ORACLE
    ORACLE = load_oracle(oracle_path)


def _score_one(task: tuple[str, int, float]) -> tuple[int, list[str], list[str]]:
    """task = (eval_dir, eval_index, radius). Returns (eval_index, obligations, instances)."""
    eval_dir, eval_index, radius = task
    streams = sorted(glob.glob(str(pathlib.Path(eval_dir) / "semantic" / "*.jsonl")))
    if not streams:
        return eval_index, [], []
    stream = streams[-1]
    try:
        ticks = load_semantic_trace(stream)
        route = load_ego_route(stream)
        report = compute_cov(ORACLE, ticks, trace_label=stream, ego_route=route)
    except Exception:
        return eval_index, [], []
    covered: list[str] = []
    for axis in ("V", "A", "E", "H"):
        for row in report["dimensions"][axis]["obligations"]:
            if row.get("covered"):
                covered.append(str(row["signature"]))
    bucket = int(round(radius / 5.0) * 5) if radius else 0
    instances = [f"{sig}@{bucket}m" for sig in covered]
    return eval_index, sorted(set(covered)), sorted(set(instances))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=pathlib.Path, required=True)
    ap.add_argument("--ads", required=True)
    ap.add_argument("--oracle", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    tasks: list[tuple[str, int, float]] = []
    arm_index: list[tuple[str, str, str]] = []  # (route, policy, arm_dir)
    for rows_path in sorted(glob.glob(str(args.search_root / "*" / "*" / "rows.jsonl"))):
        arm_dir = str(pathlib.Path(rows_path).parent)
        route = pathlib.Path(arm_dir).parent.name
        policy = pathlib.Path(arm_dir).name
        if policy == "control":
            continue
        for line in open(rows_path):
            try:
                row = json.loads(line)
            except Exception:
                continue
            if (row.get("collision_count") or 0) <= 0:
                continue
            idx = row.get("eval_index")
            cand = row.get("candidate") or {}
            radius = float(cand.get("trigger_radius_m") or 0.0)
            eval_dir = str(pathlib.Path(arm_dir) / "evaluations" / f"{policy}-{int(idx):04d}")
            tasks.append((eval_dir, int(idx), radius))
            arm_index.append((route, policy, arm_dir))

    print(f"failing executions to score: {len(tasks)}", flush=True)
    results: dict[str, dict[str, set[str]]] = {}
    obs_by_eval: dict[tuple[str, str], set[str]] = {}
    inst_by_eval: dict[tuple[str, str], set[str]] = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(str(args.oracle),)) as ex:
        for (route, policy, _), (idx, obligations, instances) in zip(
            arm_index, ex.map(_score_one, tasks, chunksize=4)
        ):
            key = f"{route}/{policy}"
            obs_by_eval.setdefault(key, set()).update(obligations)
            inst_by_eval.setdefault(key, set()).update(instances)

    arms = {}
    pooled: dict[str, dict[str, set[str]]] = {}
    for key in sorted(obs_by_eval):
        route, policy = key.split("/")
        arms[key] = {
            "route": route,
            "policy": policy,
            "failure_causing_obligations": sorted(obs_by_eval[key]),
            "failure_causing_instances": sorted(inst_by_eval[key]),
            "n_obligations": len(obs_by_eval[key]),
            "n_instances": len(inst_by_eval[key]),
        }
        pooled.setdefault(policy, {"obligations": set(), "instances": set()})
        pooled[policy]["obligations"].update(obs_by_eval[key])
        pooled[policy]["instances"].update(inst_by_eval[key])

    summary = {
        "ads": args.ads,
        "search_root": str(args.search_root),
        "n_failing_executions": len(tasks),
        "arms": arms,
        "pooled": {
            policy: {
                "n_obligations": len(v["obligations"]),
                "n_instances": len(v["instances"]),
                "obligations": sorted(v["obligations"]),
            }
            for policy, v in pooled.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print("pooled (distinct obligations / instances over failing runs):")
    for policy in ("semantic", "random", "lsa", "kmnc"):
        if policy in summary["pooled"]:
            p = summary["pooled"][policy]
            print(f"  {policy:9s} obligations {p['n_obligations']:3d} | instances {p['n_instances']:3d}")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
