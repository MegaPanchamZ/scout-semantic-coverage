# Patches to the third-party agent repositories

`research/models/` is not tracked. The two agents SCOUT evaluates are cloned
there from upstream and patched with the files in this directory.

| Repo | Upstream | Commit | Patch |
|---|---|---|---|
| PCLA (InterFuser) | https://github.com/MasoudJTehrani/PCLA | `07d4d5d69f975d3c8988be8559bcd5d80c32f26d` | `pcla.patch` |
| AutoVLA | https://github.com/ucla-mobility/AutoVLA | `ba34eed74ce6729e7986592d0e66cbaca397b4fa` | `autovla.patch` |

```bash
cd research/models
git clone https://github.com/MasoudJTehrani/PCLA.git
git -C PCLA checkout 07d4d5d69f975d3c8988be8559bcd5d80c32f26d
git -C PCLA apply ../../patches/pcla.patch
git clone https://github.com/ucla-mobility/AutoVLA.git
git -C AutoVLA checkout ba34eed74ce6729e7986592d0e66cbaca397b4fa
git -C AutoVLA apply ../../patches/autovla.patch
```

Model weights come from the upstream instructions (PCLA's
`pcla_agents/interfuser_pretrained/`; AutoVLA's `AutoVLA_PDMS_89.ckpt`, converted
to a HF directory with `python research/scripts/convert_autovla_hf.py`, which
expects `AutoVLA/checkpoints/AutoVLA_PDMS_89.ckpt` and
`AutoVLA/Qwen2.5-VL-3B-Instruct/` and writes `AutoVLA/checkpoints/AutoVLA-hf`).

## What the patches change

`pcla.patch` (`pcla_agents/interfuser/{base_agent,interfuser_agent}.py`):
- Torch CPU threads capped (`INTERFUSER_TORCH_THREADS`, default 4); with 64
  threads the per-tick image transforms took ~1.4 s instead of ~7 ms.
- Image normalisation on the GPU (differs from the CPU path by <=7e-7).
- CUDA-graph forward that stays enabled when forward hooks are registered
  (SCOUT's coverage observer hooks the network). Hooks are swapped for tap hooks
  during capture and called on the tapped outputs after each replay; the graph
  is recaptured if the hook set changes. Outputs and coverage are bit-identical
  to eager mode. Global hooks, pre-hooks and kwargs hooks fall back to eager.
- Debug frames (one 1200x600 JPEG per tick) and the BEV render are written only
  when `SAVE_PATH` is set; the debug directory gets a PID suffix so parallel
  runs do not collide.

`autovla.patch` (`models/autovla.py`): the training-only navsim/nuplan import
in `models.utils.score` is wrapped in try/except so inference runs without the
navsim stack.

## Harness-side workarounds (tracked in this repo, listed for reference)
- `run_shakedown.py` exits with `os._exit(0)` after artifacts are written,
  because some agent stacks leave non-joinable threads that abort teardown.
- CARLA render quality defaults to `Epic` (`SCOUT_CARLA_QUALITY` /
  `--carla-quality` override). Keep quality fixed within a campaign.
- `scenario_runtime.py` makes `set_autopilot(False)` best-effort (servers share
  traffic-manager port 8000).
- Idle CARLA servers are parked in synchronous mode (`carla_utils.restore_world_settings`;
  `SCOUT_IDLE_ASYNC=1` restores the old behaviour). An idle async server keeps
  rendering and slows every other server on the same GPU.
- The coverage observer pins BLAS to one thread (`threadpoolctl`); OpenBLAS
  spin-waits otherwise steal CPU from the CARLA/agent processes.
