#!/usr/bin/env python3
"""RQ3 campaign on Autoware: every search method x route x seed with the SAME evaluation budget, search space,
initial scenario, CARLA map, ADS and tick budget. Seed-major order so a partial campaign is still balanced.
Resumable: a finished (method, route, seed) leaves `complete` in its log; policy_search itself resumes from rows.jsonl.

usage: run_autoware_campaign.py OUTROOT [--evals 30] [--seeds 1 2 3] [--methods random semantic avfuzzer causal]
"""
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "research/experiments/EXP-020-policy-comparison/artifacts/base_specs"
ROUTES = {"town01_spawn0_goal82": "town01_spawn0_goal82.json",
          "town01_spawn115_goal206": "town01_spawn115_goal206.json",
          "town01_spawn68_goal218": "town01_spawn68_goal218.json"}
PS = ROOT / "research/experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py"

ap = argparse.ArgumentParser()
ap.add_argument("outroot", type=Path)
ap.add_argument("--evals", type=int, default=30)
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--methods", nargs="+", default=["random", "semantic", "avfuzzer", "causal"])
ap.add_argument("--routes", nargs="+", default=["town01_spawn0_goal82", "town01_spawn115_goal206"])
a = ap.parse_args()

for seed in a.seeds:
    for route in a.routes:
        for method in a.methods:
            out = a.outroot / method / route / f"seed{seed}"
            log = out.parent / f"seed{seed}.log"
            out.parent.mkdir(parents=True, exist_ok=True)
            if log.exists() and "complete:" in log.read_text()[-400:]:
                print(f"skip {method}/{route}/seed{seed} (complete)", flush=True); continue
            t0 = time.time()
            cmd = [sys.executable, str(PS), "--policy", method, "--base-spec", str(SPECS / ROUTES[route]),
                   "--route-label", route, "--output-dir", str(out), "--evals", str(a.evals), "--seed", str(seed),
                   "--search-space", "campaign", "--backend", "autoware", "--agent-kind", "autoware", "--max-ticks", "500"]
            print(f"[{time.strftime('%H:%M:%S')}] START {method}/{route}/seed{seed}", flush=True)
            with log.open("ab") as fh:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
            print(f"[{time.strftime('%H:%M:%S')}] END   {method}/{route}/seed{seed} rc={rc} {time.time()-t0:.0f}s", flush=True)
