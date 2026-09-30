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

Config conversions and artifacts (not patches):
- AutoVLA checkpoint converted from `AutoVLA_PDMS_89.ckpt` (Lightning, fp32,
  `autovla.vlm.` prefix, vision keys under `model.visual.*`, LLM keys under
  `model.language_model.*`) into a merged HF directory
  `research/models/AutoVLA/checkpoints/AutoVLA-hf` (bf16, 2 shards).
