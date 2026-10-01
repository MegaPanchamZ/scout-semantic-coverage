# Corrected SCOUT campaign protocol

The `scout-search-v2` protocol fixes scheduler advancement, changing novelty
fitness, template eligibility, AutoVLA trajectory tracking and longitudinal
control, and temporal hazard matching. Existing paper results were produced
by earlier code and have **not** been regenerated or validated by these fixes.
Do not combine old and new results. The search rejects incompatible checkpoints.

The executable scenario representation used by this campaign is JSON
`ScenarioSpec`, with two generator templates: pedestrian crossing (five
parameters) and lead braking (four). The nine inventory hazard classes are
not nine implemented generator templates. A formal TARGET grammar/compiler
and held-out manual grounding validation remain separate work.

## Environment and offline verification

Run from the repository root:

```bash
uv sync --directory research --python 3.10 --group dev
research/.venv/bin/python -m pytest research/tests -q
research/.venv/bin/python research/harness/run_shakedown.py --help
research/.venv/bin/python research/harness/hazard_search.py --list
```

CARLA 0.9.16, the PCLA repository/checkpoints for InterFuser, and the upstream
AutoVLA repository/model/codebook are external prerequisites. Model packages
also require their upstream inference environments (including PyTorch).
Weights and GPU packages are not installed by the lightweight test environment.
Place PCLA under `research/models/PCLA`, or pass `--agent-repo-path` directly.
Place AutoVLA under `research/models/AutoVLA`, with its Hugging Face checkpoint
under `checkpoints/AutoVLA-hf` and `codebook_cache/agent_vocab.pkl` available.
The checkpoint used in the original study was converted from the upstream
merged Lightning release; its conversion is not part of this repository's
verified reproduction workflow. Supply a compatible converted checkpoint
before attempting AutoVLA execution.

Set `CARLA_ROOT` to the extracted simulator directory containing `CarlaUE4.sh`.
The simulator/agent integration must be validated on nominal routes before
interpreting failure rates; offline tests do not establish driving competence.

## Nominal calibration

After installing an ADS, record multiple successful no-adversary episodes
with `run_shakedown.py --coverage-observer --semantic-observer` and the ADS's
agent flags. Collect raw activation traces **without** supplying a coverage
profile, then fit a fresh nominal profile:

```bash
research/.venv/bin/python research/harness/build_coverage_profile.py \
  --input-glob 'research/logs/nominal-v2/coverage/*.npz' \
  --output research/logs/coverage/autovla-nominal-profile-v2.joblib
```

Use separate profiles for each ADS. The AutoVLA profile must be rebuilt after
the control fixes; an old-profile file is not evidence that nominal behavior
is correct. This document does not claim an existing valid new profile.

## Hazard campaigns

`run_fse_hazard.sh` generates missing lead-braking specifications and then
starts the two InterFuser workers. Generation uses a live CARLA server and
may change its map: run it only on a server reserved for that preparation,
before starting evaluations. The generation script explicitly names output
files using the campaign's route labels (including optimized CARLA maps).

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.16
SCOUT_SEED=13 bash research/experiments/EXP-020-policy-comparison/proof-of-concept/run_fse_hazard.sh
```

Alternatively invoke `run_fse_hazard_workers.sh A` and `B` after generating
the specs. For AutoVLA, set `SCOUT_PROFILE` to the newly fitted profile and
invoke `run_fse_multimap_autovla.sh A` and `B`. Both ADS campaign launchers
now use a 500-tick cap and execute one matched nominal episode per candidate.
These paired controls are additional to the requested adversarial budget;
the eight route-level controls are retained as additional nominal diagnostics.

For independent repeated searches, rerun with different `SCOUT_SEED` values
(e.g. 13, 23, 37). Default output roots include the seed and `_v2`. Keep each
replicate's statistics separate or explicitly model route/replicate dependence;
do not count adaptive evaluations as independent replicate searches.
`SCOUT_OUTPUT_ROOT` overrides the output root. Launchers must not share a
simulator port/GPU with another concurrently running campaign.

## Recorded evidence and interpretation

- Each candidate records its actual per-run obligation signatures, cumulative
  first witnesses, target/template, RNG state, and execution seed. Restarting
  a compatible checkpoint restores candidate selection and RNG state.
- Semantic hazard fitness rewards the active obligation, with its constituent
  witnesses as a tie-break. The semantic elite resets when the target changes.
  Targets without available compatible specifications are excluded. Stall
  limits defer unsuccessful targets; after exhaustion, the remaining budget
  explores available templates with no active target.
- Each candidate and its nominal control start in a reloaded CARLA world and
  receive the same execution seed. `paired_control` archives the nominal
  outcome and independently scored semantic evidence. Control witnesses never
  contribute to adversarial suite coverage.
- Collision events explicitly identify contacts with injected actors. This
  is direct-contact evidence, not attribution of every failure to a hazard.
  Static collisions, nominal failures, observer witnesses, and injected-actor
  contacts must be inspected separately.
- The coverage engine preserves ordered cut-in events and a minimum 0.5 s
  stationary interval for waiting hazards. Its default window is 30 ticks
  at 10 Hz; `--tick-seconds` records another trace rate for persistence.
  Existing inventories receive these known constraints through the loader;
  rebuilt inventories explicitly serialize them. Primitive grounding rules
  remain documented geometric proxies, not manually validated labels.
- `mine_failure_obligations.py` now labels its output as obligations/instances
  **witnessed in collision runs**, with schema version 2. It does not prove
  causation and now fails explicitly if a collision trace cannot be scored.

The supplied aggregation computes route-level comparisons within one campaign
root. Old manuscript numbers, RQ1 manual validation, RQ4 observer map studies,
and stronger causal/statistical conclusions require additional empirical work.
