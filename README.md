# SCOUT — Semantic Coverage-Guided Testing of Autonomous Driving Systems

This repository contains the semantic inventory builder, simulator-state
coverage engine, CARLA harness, policy search, analysis scripts and offline tests.

The corrected `scout-search-v2` code fixes scheduler and AutoVLA control defects.
Existing paper results were produced by earlier code and have not been rerun.
The implementation currently provides two generation templates: pedestrian
crossing and lead-vehicle braking, with separate parameter spaces.

Read [the reproduction guide](research/REPRODUCING_SCOUT.md) for setup,
external simulator/model prerequisites, nominal-profile calibration, repeated
seed campaigns, matched controls and remaining validation limitations.

```bash
uv sync --directory research --python 3.10 --group dev
research/.venv/bin/python -m pytest research/tests -q
research/.venv/bin/python research/harness/run_shakedown.py --help
```

`appendix/appendix.pdf` contains the historical supplementary document. Its
experimental tables and implementation description have not been updated to
claim results from the corrected protocol. No credentials, model weights or
execution logs are included.
