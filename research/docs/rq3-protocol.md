# RQ3 experimental protocol (Autoware)

Compare against search-based ADS testing methods under a fairness protocol that aligns only
what is common, keeps each method's own mechanism, documents common vs method-specific parameters, and compares on
method-independent outcomes only.

## Common to every method (aligned)

| Item | Value |
|---|---|
| ADS | Autoware Universe 0.50.0 (`autoware:universe-cuda`), real lidar perception (CenterPoint base, TensorRT 10.8), CARLA bridge `autoware_carla_interface` |
| Simulator | CARLA 0.9.15 (the version the bridge supports), synchronous, 0.05 s tick, ClearNoon |
| Map / routes | Town01; routes `spawn0->goal82`, `spawn115->goal206` (3rd route `spawn68->goal218` optional) |
| Initial scenario | the route's threshold-crossing base spec (`EXP-020/.../base_specs/<route>.json`) |
| Parameter space | the pre-registered campaign space (5 dims): `trigger_radius_m` [5,35], `subject_speed_mps` [0.8,4.0], `staging_lateral_offset_m` [1,14], `staging_longitudinal_offset_m` [-10,30], `trigger_tick` [10,400] |
| Episode budget | `max_ticks`=500 at 0.1 s = 50 simulated s (the bridge ticks at 0.05 s; budgets are in simulated time) |
| Testing budget | **30 evaluations per (method, route, seed)**; seeds 1,2,3 (matched across methods) |
| Termination | first collision, goal within 3 m, or budget |
| Evaluation pipeline | identical (`policy_search.py --backend autoware`): same service, ego reset, harness controllers/oracles |

## Method-specific (NOT aligned; as defined by each approach)

| Method | Search mechanism | Feedback signal | Parameters (`baselines.py`, `policy_search.py`) |
|---|---|---|---|
| **Ours (semantic)** | elite mutation + 25% uniform restarts over the shared space | obligation credit from the simulator-state STSG (+1 if collision) | gaussian sigma per dimension (`SearchDimension.sigma`), restart prob 0.25 |
| **Random** | uniform sampling | none | - |
| **AV-Fuzzer (re-impl.)** | GA: tournament selection, uniform crossover, gaussian mutation, elitism, local iterative search, restarts | closeness to violation = -(min ego-adversary gap); collision = optimum | `AVFUZZER_PARAMS`: pop 6, tournament 2, crossover 0.8, mutation 0.4/dim, elite 1, LIS after 2 stagnant gens (sigma x0.25), restart after 4 |
| **Causal-Fuzzer-style (re-impl.)** | causality-weighted mutation of feedback-selected seeds | newly activated (param-bin -> behaviour -> outcome) relations | `CAUSAL_PARAMS`: 3 bins/dim, effect estimated after 6 archive points, ridge 1.0, uniform floor 0.1 |

Not run on Autoware, and why:
* **LSA / KMNC** (coverage baselines): need the ADS's neuron activations; Autoware is a modular stack, so there is no single activation space to instrument. They remain in the end-to-end ADS campaigns (InterFuser etc.).
* **CoCoMagic**: differential testing across *versions* of an ADS with metamorphic relations; needs two versions and a different outcome (behavioural divergence). No public code. Not included in this campaign; see Threats.

## Method-independent outcomes (all methods judged identically)

1. violations found within the budget (evaluations ending in a collision with an injected actor);
2. evaluations / wall-clock / simulated time to the first violation;
3. distinct violations (violating candidates > 0.15 apart in normalised parameter space).

Statistics: one-sided Mann-Whitney U (ours > baseline) over the (route x seed) runs, Holm-corrected (`scripts/rq3/rq3_aggregate.py`).

## Threats to validity (to carry into the paper)

* AV-Fuzzer and Causal-Fuzzer-style are **our re-implementations** of the search loops, not the authors' code. The original AV-Fuzzer targets LGSVL/Apollo; the Causal-Fuzzer implementation was not obtainable and is built from the published description of its two mechanisms, so it may be weaker than the original.
* Each method keeps its own feedback, so absolute search efficiency also reflects the signal design; the shared space and budget mean differences are attributable to the search+signal as a package.
* Autoware (and the sim) is not bit-deterministic: repeated runs of one scenario agree on outcome in our checks but sim-time of violation varies by about +-0.2 s.
* Single map and two routes for this campaign; the hazardous spawn-0 site dominates violations on Town01.
* Perception is a single model (CenterPoint base); results may not transfer to other perception stacks.
* CARLA 0.9.15 here vs 0.9.16 for the other four ADSs: cross-ADS comparisons mix simulator versions.
* Autoware infrastructure failures (routing/reset, ~4% of episodes) are retried once and reported as errors, never as results.

## Post-hoc addition: criticality-stratified semantic coverage (`semantic_crit`)

**Disclosure.** `semantic_crit` was designed and added *after* the first completed (route, seed) round of the pre-planned campaign
(2026-10-19) showed that the original signal (`semantic`, obligation credit) gave no guidance on this scenario family:
every method covered the same 21 semantic signatures and the obligation credit was 3 for almost every evaluation, so the
elite search degenerated to random-like behaviour (the same saturation noted in the earlier InterFuser pilot). The campaign
was NOT restarted or re-tuned; `semantic_crit` is queued to run on exactly the same routes, seeds and budget after the four
pre-planned methods finish, and the paper must report:

* the four pre-planned methods as the primary, pre-specified comparison;
* `semantic_crit` as a **post-hoc** method (it was shaped by looking at early data), with its results labelled exploratory;
* that it keeps the search mechanism of `semantic` (elite mutation, 25% uniform restarts) and changes only the coverage signal:
  credit = newly covered (semantic signature, criticality bin) pairs, bins from the minimum ego-adversary gap
  (collision | <1 m | 1-3 m | 3-6 m | >6 m), a closer encounter also covering all looser bins; ties broken by smaller gap.

Free parameters introduced (all chosen without tuning on outcomes): the four gap-bin edges (1, 3, 6 m).

## Measured Campaign Results (Completed 2026-10-19)

30 runs total (5 methods × 2 Town01 routes × 3 matched seeds × 30 evaluations = 900 Autoware episodes, 0 infrastructure errors).
Metrics below are over the 29 search evaluations per run after excluding the shared initial scenario (`eval_index = 0`):

| Method | Violations (mean ± SD) | Evals to 1st (median) | Wall s to 1st (median) | Distinct Violations (mean) | Runs w/ >=1 Violation | Infra Errors |
|---|---|---|---|---|---|---|
| `semantic` (SCOUT v1) | 1.33 ± 2.21 | 4.0 | 260 | 1.17 | 2/6 | 0 |
| `semantic_crit` (post-hoc) | 1.50 ± 2.14 | 5.0 | 336 | 1.33 | 2/6 | 0 |
| `avfuzzer` (re-impl.) | 2.50 ± 1.89 | 22.0 | 1323 | 1.33 | 5/6 | 0 |
| `causal` (Causal-Fuzzer-style) | 1.50 ± 1.89 | 3.0 | 196 | 1.50 | 3/6 | 0 |
| `random` | 0.33 ± 0.47 | 7.5 | 545 | 0.33 | 2/6 | 0 |

