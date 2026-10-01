# SCOUT — Semantic Coverage-Guided Testing of Autonomous Driving Systems

Review artifact containing scenario generation, simulator-state semantic monitoring,
coverage scoring, ADS adapters, policy search, tests, and statistical aggregation.
Read [REVIEW_AUDIT.md](REVIEW_AUDIT.md) before interpreting campaign results.

## Where to review

| Component | Source |
|---|---|
| Oracle inventory construction | `research/experiments/EXP-018-nuscenes-oracle-inventory/proof-of-concept/build_oracle_inventory.py` |
| Scenario generation and execution | `research/harness/scenario_gen.py`, `scenario_runtime.py`, `runner.py` |
| Legacy/campaign parameter spaces | `research/harness/search_space.py` |
| Hazard-specific spaces and scheduler | `research/harness/hazard_search.py` |
| Search objectives and candidate selection | `research/experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py` |
| Semantic monitoring and obligation credit | `research/harness/observers/semantic.py`, `coverage_engine.py`, `obligation_credit.py` |
| ADS integration | `research/harness/pcla_bridge.py`, `autovla_bridge.py` |
| Statistics and hazard witnesses | EXP-020 `aggregate_fse_search.py`, `hazard_witness_metrics.py`; `research/scripts/mine_failure_obligations.py` |
| Supplementary definitions | [Standalone appendix](appendix/appendix.pdf), with separate A-numbering |

`--search-space campaign` runs the original five-dimensional pedestrian-crossing
search. `--hazard-search` adds crossing (five dimensions) and lead braking (four
dimensions). They are distinct experiment variants; original multi-map results
must not be described as results of the later hazard-scheduled variant.

## Offline verification

Python 3.10 and `uv` are required. From the repository root:

```bash
uv sync --directory research --python 3.10
research/.venv/bin/python -m pytest research/tests -q
```

The declared development group supplies pytest and plotting dependencies. The
ADS-specific GPU environments require additional upstream dependencies; the uv
lockfile alone does not provide a complete InterFuser or AutoVLA installation.

## Running campaigns

Install CARLA 0.9.16 separately and set its installation path. Launchers resolve
this repository from their own location rather than requiring the original
workspace path. `PYTHON_EXECUTABLE` can override the worker interpreter.

```bash
export CARLA_ROOT=/path/to/carla-0.9.16
bash research/experiments/EXP-020-policy-comparison/proof-of-concept/run_fse_matrix.sh A
bash research/experiments/EXP-020-policy-comparison/proof-of-concept/run_fse_matrix.sh B
bash research/experiments/EXP-020-policy-comparison/proof-of-concept/run_fse_multimap.sh A
bash research/experiments/EXP-020-policy-comparison/proof-of-concept/run_fse_multimap.sh B
```

These launchers expect PCLA/InterFuser source and weights under
`research/models/PCLA` and a fitted nominal coverage profile at
`research/logs/coverage/if-if-safe-prefix-profile.joblib`. See
[agent patch notes](research/models/patches/README.md). The reviewed workspace
used PCLA revision `84552c1582cae2a01ff34fb959a192e1761a9352` plus those patches.

AutoVLA uses `run_fse_multimap_autovla.sh A` and `B`, upstream source under
`research/models/AutoVLA`, the converted HF checkpoint under
`research/models/AutoVLA/checkpoints/AutoVLA-hf`, and
`research/logs/coverage/autovla-nominal-profile.joblib`. The reviewed workspace
used AutoVLA revision `ba34eed74ce6729e7986592d0e66cbaca397b4fa` plus the documented
patch. Conversion code and nominal trace provenance are not yet packaged; this
remains a reproduction gap.

To fit a profile from independently collected nominal traces:

```bash
research/.venv/bin/python research/harness/build_coverage_profile.py \
  --input /path/to/nominal-traces.npz --output /path/to/profile.joblib
```

This command explains the fitting interface; it does not reproduce the original
profile without the original trace selection and fitting settings.

The hazard campaign uses `run_fse_hazard_workers.sh A` and `B`. Missing lead-braking
base specs can be generated with `hazard_search.py --generate-lead-spec` against
an available CARLA server. `run_fse_hazard.sh` generates missing specs and launches
both workers; it expects a running server at `CARLA_HOST`/`CARLA_PORT` (defaults
127.0.0.1:2000) for generation. Review the scheduler issue in the audit first.

## Aggregation

```bash
research/.venv/bin/python research/experiments/EXP-020-policy-comparison/proof-of-concept/aggregate_fse_search.py \
  --search-root research/logs/fse_search
research/.venv/bin/python research/scripts/per_map_summary.py \
  --summary research/logs/fse_search/fse_summary.json --out-dir research/logs/fse_search
research/.venv/bin/python research/scripts/make_fse_figures.py \
  --search-root research/logs/fse_search --output-dir research/logs/fse_figures
```

Keep InterFuser, AutoVLA, and hazard-campaign output roots separate. Raw execution
logs, fitted profiles, model weights, dataset metadata, and credentials are not
included. Consequently the repository supports source review and offline tests,
but does not independently reproduce the manuscript statistics yet. PtoP and
DoTA adapters are included as supporting source; their external data/vendor
packages are required only for those paths. Cosmos is not a required component
of the documented search campaigns.
