# SCOUT — Semantic Coverage-Guided Testing of Autonomous Driving Systems

This repository contains the semantic inventory builder, simulator-state
coverage engine, CARLA harness, policy search, analysis scripts and offline tests.

The corrected `scout-search-v2` code fixes scheduler and AutoVLA control defects.
The RQ3 InterFuser safety campaign was rerun with that protocol
(`scout-search-v2`); its raw rows and tables are in `research/results/`. Other
paper results were produced by earlier code and have not been rerun. The
search has since moved to `scout-search-v3` (criticality guidance, stage-2
exploitation, adaptive mutation; see `research/GAP_DRIVEN_SEARCH.md`).
The implementation currently provides two generation templates: pedestrian
crossing and lead-vehicle braking, with separate parameter spaces.

Read [the reproduction guide](research/REPRODUCING_SCOUT.md) for setup,
external simulator/model prerequisites (with the patches to the third-party
agents in `research/patches/`), nominal-profile calibration, the CARLA fleet,
the RQ3 campaign, matched controls and remaining validation limitations.

```bash
uv sync --directory research --python 3.10 --group dev
research/.venv/bin/python -m pytest research/tests -q
research/.venv/bin/python research/harness/run_shakedown.py --help
```

`appendix/appendix.pdf` contains the historical supplementary document. Its
experimental tables and implementation description have not been updated to
claim results from the corrected protocol. No credentials, model weights or
simulator traces are included.
