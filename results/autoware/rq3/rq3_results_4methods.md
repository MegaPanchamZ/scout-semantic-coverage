# RQ3 results (rq3)

Runs compared: 6 (route x seed) per method; budget = 29 search evaluations after the shared initial scenario; distinct-violation radius = 0.15 (normalised parameter space).

| method | violations (mean±sd) | evals to 1st violation (median) | wall s to 1st (median) | distinct violations (mean) | runs w/ >=1 violation | infra errors |
|---|---|---|---|---|---|---|
| semantic | 1.33 ± 2.21 | 4.0 | 260 | 1.17 | 2/6 | 0 |
| avfuzzer | 2.50 ± 1.89 | 22.0 | 1323 | 1.33 | 5/6 | 0 |
| causal | 1.50 ± 1.89 | 3.0 | 196 | 1.50 | 3/6 | 0 |
| random | 0.33 ± 0.47 | 7.5 | 545 | 0.33 | 2/6 | 0 |

## semantic vs baselines (one-sided Mann-Whitney U on violations, Holm-corrected)

| baseline | P(ours > baseline) | p | p (Holm) |
|---|---|---|---|
| avfuzzer | 0.28 | 0.922 | 1.000 |
| causal | 0.44 | 0.672 | 1.000 |
| random | 0.56 | 0.387 | 1.000 |
