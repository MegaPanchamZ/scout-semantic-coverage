# SCOUT — Semantic Coverage-Guided Testing of Autonomous Driving Systems

This repository contains the semantic inventory builder, simulator-state
coverage engine, CARLA harness, policy search, analysis scripts and offline tests.

The corrected `scout-search-v2` code fixes scheduler and AutoVLA control defects.
The RQ3 InterFuser safety campaign was rerun with that protocol
(`scout-search-v2`); its raw rows and tables are in `research/results/`.
In addition, the 900-episode RQ3 Autoware Universe campaign (`5 methods × 2 Town01 routes × 3 matched seeds × 30 evaluations`, comparing `semantic`, post-hoc `semantic_crit`, `avfuzzer`, `causal`, and `random` over the modular Autoware Universe stack with live 3D LiDAR `CenterPoint` perception in CARLA 0.9.15) is provided under `results/autoware/rq3/`, with bridge/service scripts in `scripts/autoware_bridge/`, campaign orchestration and statistical aggregation in `scripts/rq3/`, and protocol/baseline specifications in `research/docs/rq3-protocol.md` and `research/docs/rq3-baselines.md`.
The search has since moved to `scout-search-v3` (criticality guidance, stage-2
exploitation, adaptive mutation; see `research/GAP_DRIVEN_SEARCH.md`).
The implementation currently provides two generation templates: pedestrian
crossing and lead-vehicle braking, with separate parameter spaces.

Read [the reproduction guide](research/REPRODUCING_SCOUT.md) for setup,
external simulator/model prerequisites (with the patches to the third-party
agents in `research/patches/`), nominal-profile calibration, the CARLA fleet,
the RQ3 InterFuser and Autoware Universe campaigns, matched controls and remaining validation limitations.

```bash
uv sync --directory research --python 3.10 --group dev
research/.venv/bin/python -m pytest research/tests -q
research/.venv/bin/python research/harness/run_shakedown.py --help
research/.venv/bin/python scripts/rq3/rq3_aggregate.py results/autoware/rq3
```

`appendix/appendix.pdf` contains the historical supplementary document. Its
experimental tables and implementation description have not been updated to
claim results from the corrected protocol. No credentials, model weights or
simulator traces are included.
