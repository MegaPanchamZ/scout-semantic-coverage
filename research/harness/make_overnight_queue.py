"""Build the full EXP-020 matrix with nominal gates and model-specific profiles."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEARCH = "research/experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py"
BASE = ROOT / "research/experiments/EXP-020-policy-comparison/artifacts/base_specs"
ORACLE = "research/experiments/EXP-018-nuscenes-oracle-inventory/artifacts/oracle_inventory_v1.0-trainval.json"
MODELS = {
    "interfuser": ("research/.venv/bin/python", ["--agent-kind", "pcla", "--pcla-agent", "if_if"]),
    "lav": ("research/.venv/bin/python", ["--agent-kind", "pcla", "--pcla-agent", "lav_lav"]),
    "transfuser": ("research/.venv/bin/python", ["--agent-kind", "leaderboard-module", "--agent-repo-path", "research/models/transfuser_cvpr2021", "--agent-module", "transfuser_agent", "--agent-config", "research/models/transfuser/model_ckpt/transfuser"]),
    "simlingo": ("research/.venv-simlingo/bin/python", ["--agent-kind", "pcla", "--pcla-agent", "simlingo_simlingo"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["interfuser", "lav", "transfuser"])
    parser.add_argument("--routes", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13])
    parser.add_argument("--evals", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=5, help="Advance each resumable arm in small batches for a balanced overnight sample.")
    parser.add_argument("--max-ticks", type=int, default=500)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--quality-level", choices=("Low", "Epic"), default="Epic")
    parser.add_argument("--output-root", type=Path, default=Path("research/logs/overnight_campaign"))
    parser.add_argument("--manifest", type=Path, default=Path("research/logs/overnight_campaign/manifest.json"))
    args = parser.parse_args()
    if args.evals <= 0 or args.chunk_size <= 0:
        raise ValueError("Evaluation budget and chunk size must be positive")
    default_routes = "town01_spawn195_goal197 town01_spawn82_goal200 town01_spawn68_goal218 town01_spawn115_goal206 town01_spawn55_goal154 town01_spawn0_goal82 town03_spawn121_goal2 town03_spawn125_goal223 town05_spawn0_goal124 town05_spawn218_goal257 town05_spawn239_goal100 town10hd_spawn0_goal44 town10hd_spawn1_goal63 town10hd_spawn43_goal100".split()
    routes = [BASE / f"{name}.json" for name in (args.routes or default_routes)]
    if not routes or any(not path.is_file() for path in routes):
        raise ValueError("All requested route specs must exist")
    jobs = []
    # Interleave models and policies so one large arm cannot monopolise a night.
    for route in routes:
        nominal_spec = args.output_root / "nominal_specs" / route.name
        payload = json.loads(route.read_text())
        payload.update(controller="route_only", controller_params={}, walker_count=0, npc_vehicle_count=0)
        nominal_spec.parent.mkdir(parents=True, exist_ok=True)
        nominal_spec.write_text(json.dumps(payload, indent=2))
        for model in args.models:
            python, flags = MODELS[model]
            profile = args.output_root / model / "nominal_profile.joblib"
            profile_job = f"{model}-profile"
            if not any(job["id"] == profile_job for job in jobs):
                trace_label = args.quality_level + "_native_wrapper" if model == "transfuser" else args.quality_level
                trace_glob = f"research/logs/overnight_benchmark/{trace_label}/{model}/coverage/*.npz"
                jobs.append({"id": profile_job, "command": [python, "research/harness/build_coverage_profile.py", "--input-glob", trace_glob, "--output", str(profile)]})
            gates = []
            for seed in (13, 29, 47):
                gate = f"{model}-{route.stem}-nominal-{seed}"
                path = args.output_root / model / route.stem / f"nominal-{seed}"
                jobs.append({"id": gate, "command": [python, "research/harness/run_shakedown.py", "--scenario-spec", str(nominal_spec), "--reload-world", "--port", str(args.port), "--seed", str(seed), "--max-ticks", str(args.max_ticks), "--output-dir", str(path)] + flags,
                             "nominal_results": str(path), "timeout_seconds": 900})
                gates.append(gate)
            for seed in args.seeds:
                for policy in ("random", "lsa", "kmnc", "semantic"):
                    path = args.output_root / model / f"seed-{seed}" / route.stem / policy
                    command = [python, SEARCH, "--policy", policy, "--python-executable", python, "--base-spec", str(route), "--route-label", route.stem, "--output-dir", str(path), "--evals", str(args.evals), "--seed", str(seed), "--paired-controls", "--search-space", "campaign", "--max-ticks", str(args.max_ticks), "--server-port", str(args.port), "--cuda-visible-devices", "1", "--graphics-adapter", "0", "--quality-level", args.quality_level, "--coverage-profile", str(profile), "--engine-metrics", "--oracle", ORACLE] + flags
                    jobs.append({"id": f"{model}-{route.stem}-{seed}-{policy}", "requires": gates + [profile_job], "command": command})
    setup_jobs = [job for job in jobs if "requires" not in job]
    search_jobs = [job for job in jobs if "requires" in job]
    progressive_jobs = []
    for target in range(args.chunk_size, args.evals + args.chunk_size, args.chunk_size):
        target = min(target, args.evals)
        for job in search_jobs:
            command = list(job["command"])
            command[command.index("--evals") + 1] = str(target)
            progressive_jobs.append(dict(job, id=f"{job['id']}-through-{target}", command=command))
    jobs = setup_jobs + progressive_jobs
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"port": args.port, "server": {"carla_root": "/mnt/DevDrive/carla-0.9.16", "graphics_adapter": 0, "quality_level": args.quality_level}, "env": {"CUDA_VISIBLE_DEVICES": "1", "SCOUT_KEEP_SYNCHRONOUS": "1"}, "jobs": jobs}, indent=2) + "\n")
    print(f"Wrote {len(jobs)} jobs for {len(routes)} routes to {args.manifest}")


if __name__ == "__main__":
    main()
