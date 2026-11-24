# Controlled outcome-detector validation suite

This suite tests positive, negative, and uncertainty controls for every outcome
category defined in [ads-outcome-specification.md](../../docs/ads-outcome-specification.md).

## Modes

- `--mode offline-synthetic`: executes deterministic synthetic-evidence controls
  for every declared case in `cases.json` without launching CARLA, verifying that
  each detector's live/offline replay matches expected `present`, `absent`,
  `unknown`, and `not_applicable` states.
- `--mode live`: executes the native CARLA `ScenarioSpec` files in `specs/`
  using scripted ego controls (`scripted_cases.py`, explicitly marked
  `detector_validation`, never ADS effectiveness results), verifies live vs
  replayed equivalence, and writes timestamped evidence/manifests under
  `artifacts/<run-id>/`.

## Usage

```bash
research/.venv/bin/python research/experiments/outcome-validation/run_validation.py \
  --mode offline-synthetic \
  --output-dir research/experiments/outcome-validation/artifacts/offline-check
```

Pass gate: all mandatory positive/negative/uncertainty controls for a reported
detector must pass and live vs. replayed outcomes must agree before admitting
that detector into a pilot comparison.
