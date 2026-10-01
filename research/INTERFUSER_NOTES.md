# Parallel Interfuser campaigns: applied patches and notes

`research/models/` is gitignored, so the in-place edits to the vendored PCLA /
Interfuser code are recorded here rather than committed. Apply them to a fresh
PCLA checkout to reproduce the parallel campaign.

## Inference speedups (PCLA, in place)

- `pcla_agents/interfuser/interfuser_agent.py` —
  `torch.set_num_threads(int(os.environ.get("INTERFUSER_TORCH_THREADS", "4")))`.
  With the default of 64 threads the four camera `ToTensor`+`Normalize`
  transforms took ~1,365 ms/tick; at 4 threads ~7 ms. The dominant per-tick cost
  was CPU thread thrash, **not** the model (eager forward ~27 ms). This is the
  ~10x per-tick win (per-tick ~1.4 s -> ~0.15 s).
- `pcla_agents/interfuser/interfuser_agent.py` — image normalization moved to
  the GPU (only resize/crop stay on CPU); differs from the old path by <=7e-7.
- `pcla_agents/interfuser/base_agent.py`, `interfuser_agent.py` — the debug
  `save_path` (`eval/<agent>_<timestamp>`) is suffixed with the PID and created
  with `exist_ok=True`; concurrent runs previously failed with
  `FileExistsError` on the second-resolution directory name.
- `pcla_agents/interfuser/timm/models/interfuser.py` — position-encoding pow base
  kept on `x.device`, and a GRU `flatten_parameters` guard, so `torch.export`
  passes. Note: torch-tensorrt 2.4 / TensorRT 10.1 still fail on a FakeTensor
  device-propagation error, and since the forward is only ~27 ms TensorRT is not
  the bottleneck.

## Multi-instance CARLA / harness (tracked code)

- `research/harness/scenario_runtime.py` — `set_autopilot(False)` is best-effort.
  Several CARLA servers share the default traffic-manager port (8000), so the
  call raised a bind error that killed every episode on all but one instance.
- `research/harness/config.py` — CARLA client timeout raised 10 s -> 60 s so
  `reload_world` survives a loaded GPU instead of failing the evaluation.
- `research/experiments/.../run_fse_hazard_workers.sh` — worker overrides
  `SCOUT_GPU`, `SCOUT_PORT`, `SCOUT_ROUTES`, `SCOUT_POLICIES`, `SCOUT_EVALS`,
  `SCOUT_CONTROLS`, `SCOUT_EVAL_TIMEOUT` (single-GPU host: `SCOUT_GPU=0`).
- `research/scripts/start_carla_fleet.sh` — start N headless CARLA servers as the
  `carla` user, one port per worker.
- `research/scripts/collect_nominal_and_build_profile.sh` — parallel nominal
  episodes + `build_coverage_profile.py` to fit the per-ADS coverage profile.

## Observed throughput

Per-eval is now ~40 s single / ~280 s under 7-way load on one H100 (GPU-bound at
7 workers). Full Interfuser = 14 routes x (4 x 50 + 8) = 2,912 evals -> ~28-32 h
on one GPU regardless of worker count (the GPU saturates).
