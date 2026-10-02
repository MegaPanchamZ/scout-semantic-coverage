# Corrected SCOUT campaign protocol

The `scout-search-v2` protocol fixes scheduler advancement, changing novelty
fitness, template eligibility, AutoVLA trajectory tracking and longitudinal
control, and temporal hazard matching. Existing paper results were produced
by earlier code and have **not** been regenerated or validated by these fixes,
except the RQ3 InterFuser safety campaign, which was rerun with `scout-search-v2`
(see [RQ3 safety campaign](#rq3-safety-campaign-interfuser)). The current code
is `scout-search-v3`, which adds criticality guidance, stage-2 exploitation and
adaptive mutation ([GAP_DRIVEN_SEARCH.md](GAP_DRIVEN_SEARCH.md)). Do not combine old and
new results. The search rejects incompatible checkpoints.

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
Clone PCLA and AutoVLA into `research/models/` at the pinned commits and apply
the patches in [`research/patches/`](patches/README.md) (CUDA-graph forward with
coverage hooks, thread caps, debug frames off by default, navsim-free AutoVLA
import). PCLA can also be passed with `--agent-repo-path`. AutoVLA needs its
Hugging Face checkpoint under `checkpoints/AutoVLA-hf` and
`codebook_cache/agent_vocab.pkl`. The checkpoint was converted from the
upstream Lightning release with `research/scripts/convert_autovla_hf.py`.
The AutoVLA planning server is in `research/serving/autovla/` (see
[MODEL_BACKENDS.md](MODEL_BACKENDS.md)).

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
- Safety is evaluated by named outcomes beyond collision (near collision /
  low TTC, unsafe proximity, red-light and lane-departure violations, stuck or
  incomplete routes, harsh/emergency braking). The schema-2 `safety_metrics`
  are recorded live and classified by `research/harness/safety_outcomes.py`;
  see [SAFETY_OUTCOMES.md](SAFETY_OUTCOMES.md) and
  `research/scripts/safety_outcome_report.py`. The utility evaluation also
  links obligations to issue types and measures time-to-first-issue per
  independent run (mean/std across `seed-*` replicates, paired against the
  baselines) via `research/scripts/safety_utility_report.py`. Legacy schema-1
  logs carry only collision/goal evidence, so the new outcomes require a rerun
  and must not be mixed with old rows.
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

## Benign seed scenarios and frozen-seed validation

All experiments start from benign randomised seed scenarios (per review).
Generate them with `research/scripts/generate_benign_seeds.py` (writes
`<route>_benign_seed{0,1,2}.json` and the shared `benign_seeds.json` seed set).
Workers default to `_benign_seed0` (`SCOUT_BASE_SUFFIX` overrides).

Before full runs, validate one frozen benign seed:
`bash research/scripts/validate_frozen_seed.sh <spec.json> 20` - 20 identical
runs, no search; warns if the collision rate is >= 40%. The script defaults to
the PCLA/InterFuser agent; set `SCOUT_AGENT_KIND=autovla` plus
`SCOUT_AGENT_REPO`/`SCOUT_AGENT_CONFIG` to validate AutoVLA instead. The CARLA
PythonAPI agents are located through `CARLA_ROOT` (falling back to
`/opt/carla`), so export it before a run. A missing nominal coverage profile no
longer aborts a run: coverage metrics are logged as null and collision/goal
outcomes are still recorded.

`run_fse_hazard.sh` launches whichever workers are listed in `SCOUT_WORKERS`
(default `A B`). Workers `A..H` map to CARLA ports `2000..2070`.

## CARLA fleet

`research/scripts/start_carla_fleet.sh` starts one CARLA server per worker
(`WORKERS`, default `A B`; `QUALITY`, default `Epic`; `CARLA_DIR`, default
`/opt/carla`) as the unprivileged `CARLA_USER` and writes server logs to
`LOG_DIR` (default `research/logs`). `research/scripts/carla_supervisor.sh`
restarts crashed servers and runs one discarded warm-up episode on each fresh
server, because the first episode on a cold server is not representative.

Several CARLA servers on one GPU share its rendering throughput. On the H100
used for the study, camera rendering is the bottleneck, so throughput is about
the same from 2 to 8 servers (about 19 s per 400-tick InterFuser episode with
two servers). Use two servers per GPU and add GPUs to scale. Idle servers are
left in synchronous mode (`SCOUT_IDLE_ASYNC=1` disables this); an idle
asynchronous server keeps rendering and slows the others. InterFuser debug
frames are written only when `SAVE_PATH` is set.

## RQ3 safety campaign (InterFuser)

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.16
WORKERS="A B" bash research/scripts/start_carla_fleet.sh
WORKERS="A B" bash research/scripts/carla_supervisor.sh &
SCOUT_EVALS=10 PORTS="2000 2010" bash research/scripts/run_rq3_queue.sh
research/.venv/bin/python research/scripts/rq3_analysis.py \
  --root interfuser=research/logs/rq3_if --budget 10 --out-dir research/logs/rq3_results
```

The queue runs seeds 13, 14 and 15 over four Town01 routes, with one search arm
per policy (SCOUT, Random, LSA, KMNC) and one control arm per (seed, route).
The control arm uses the same execution seeds and the adversary-stripped spec,
so candidate evaluation *i* is paired with control evaluation *i*. An outcome
counts as attributable when the candidate shows it and its paired control does
not. The queue is resumable: rerunning the command continues unfinished arms.
The baselines use the nominal profile
`research/logs/coverage/if-if-safe-prefix-profile.joblib`. A copy is in
`research/results/coverage/`; copy it to that path or rebuild it with
`research/scripts/collect_nominal_and_build_profile.sh`.

The 600 rows from the reported run and the reference tables are in
[`research/results/`](results/README.md). The tables can be regenerated
offline from those rows. Those rows were produced by `scout-search-v2`; running
the queue with the current code runs `scout-search-v3`, a different search, so
its numbers will differ from the reported ones.

Known limitation: the LSA baseline scores with scikit-learn's tree-based
`KernelDensity`. On a few low-density ticks its score differs from an exact
Gaussian KDE, and the selected value depends on float32 rounding in the PCA
projection (BLAS thread count). The coverage observer pins BLAS to one thread
so the scores are deterministic for a given setup. The reported LSA numbers
were computed this way.
