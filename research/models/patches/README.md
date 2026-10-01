# Local patches to vendored agent repositories

These patches make the vendored agents importable/runnable in this workspace.
They are applied in-place in the working copies; recorded here for reproducibility.

1. `research/models/PCLA/PCLA.py`
   - The lmdrive custom `timm` directory was force-prepended to `sys.path` at
     import time, shadowing the environment's `timm` for every PCLA agent and
     breaking TransFuser v6 (`torch_scatter` import error). The path insertion
     is now scoped to LMDrive agents inside `setup_agent` (sys.path restored
     after each agent load).

2. `research/models/AutoVLA/models/autovla.py`
   - `models.utils.score` imports navsim/nuplan (training-only). The import is
     guarded with a try/except so inference does not require the navsim stack.

3. `research/harness/run_shakedown.py`
   - Some agent stacks (PCLA/TransFuser) leave non-joinable worker threads that
     abort the interpreter during teardown after results are written. The
     runner exits explicitly (os._exit(0)) once artifacts are on disk.

4. CARLA render quality (harness default changed to `Epic`).
   - `research/experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py`
     hard-coded `-quality-level=Low` in the CARLA boot command. It is now
     `--carla-quality` and threaded through the diagnostics path.
   - `research/harness/{optimiser,run_inter_session_batch,run_inter_session_diagnostics}.py`
     `--quality-level` likewise.
   - Default is now `Epic`. Override per run with `--carla-quality`/`--quality-level`
     or by exporting `SCOUT_CARLA_QUALITY` (e.g. `SCOUT_CARLA_QUALITY=Low`).
   - Rationale: on a GPU-backed host the extra fidelity is nearly free — with the
     AutoVLA camera rig (3x 800x450) an H100 renders ~9.7 sim-Hz at Epic vs ~12.2
     at Low. Note: results captured at a different quality are not directly
     comparable with the paper's Low-quality runs, so keep quality fixed within a
     campaign.

Config conversions and artifacts (not patches):
- AutoVLA checkpoint converted from `AutoVLA_PDMS_89.ckpt` (Lightning, fp32,
  `autovla.vlm.` prefix, vision keys under `model.visual.*`, LLM keys under
  `model.language_model.*`) into a merged HF directory
  `research/models/AutoVLA/checkpoints/AutoVLA-hf` (bf16, 2 shards).
