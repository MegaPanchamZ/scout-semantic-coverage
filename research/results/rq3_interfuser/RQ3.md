# RQ3 utility results

## interfuser

### 1. Attributable safety issues per independent run (mean ± sd)

| Policy | Runs | Valid pairs | Invalid | Collision | Low TTC | Unsafe proximity | Rule violation | Stuck / route failure | Harsh braking | Any |
|---|---|---|---|---|---|---|---|---|---|---|
| SCOUT | 12 | 118 | 2 | 0.08 ± 0.29 | 0.17 ± 0.58 | 0.00 ± 0.00 | 0.17 ± 0.39 | 0.50 ± 0.80 | 1.33 ± 2.19 | 0.25 ± 0.62 |
| Random | 12 | 113 | 7 | 0.33 ± 0.49 | 0.50 ± 0.52 | 0.00 ± 0.00 | 0.17 ± 0.39 | 0.58 ± 1.00 | 1.42 ± 1.73 | 0.75 ± 0.62 |
| LSA | 12 | 117 | 3 | 0.08 ± 0.29 | 0.08 ± 0.29 | 0.00 ± 0.00 | 0.08 ± 0.29 | 0.42 ± 0.67 | 1.08 ± 1.88 | 0.17 ± 0.39 |
| KMNC | 12 | 111 | 9 | 0.17 ± 0.39 | 0.25 ± 0.45 | 0.00 ± 0.00 | 0.17 ± 0.39 | 0.75 ± 1.29 | 0.92 ± 1.38 | 0.42 ± 0.67 |

`Any` = runs with any of Collision, Low TTC, Unsafe proximity, Rule violation. Totals over all runs:

| Policy | Collision | Low TTC | Unsafe proximity | Rule violation | Stuck / route failure | Harsh braking | Any |
|---|---|---|---|---|---|---|---|
| SCOUT | 1 | 2 | 0 | 2 | 6 | 16 | 3 |
| Random | 4 | 6 | 0 | 2 | 7 | 17 | 9 |
| LSA | 1 | 1 | 0 | 1 | 5 | 13 | 2 |
| KMNC | 2 | 3 | 0 | 2 | 9 | 11 | 5 |

### 3. Evals to first attributable safety issue (any hard type)

| Policy | Runs | Found | Find rate | Evals to first (found runs, mean ± sd) | Censored mean (not found = budget+1) |
|---|---|---|---|---|---|
| SCOUT | 12 | 2 | 17% | 1.00 ± 0.00 | 9.33 ± 3.89 |
| Random | 12 | 8 | 67% | 4.88 ± 2.64 | 6.92 ± 3.68 |
| LSA | 12 | 2 | 17% | 5.50 ± 4.95 | 10.08 ± 2.61 |
| KMNC | 12 | 4 | 33% | 5.75 ± 2.87 | 9.25 ± 2.99 |

### 2. Semantic obligations -> safety issues

Attributable issues by hazard target (the semantic obligation the eval was aimed at):

| Template / hazard target | Evals | Collision | Low TTC | Unsafe proximity | Rule violation | Stuck / route failure | Harsh braking |
|---|---|---|---|---|---|---|---|
| pedestrian_crossing / hazard(other: pedestrian) | 262 | 1 | 6 | 0 | 1 | 12 | 29 |
| pedestrian_crossing / None | 85 | 2 | 4 | 0 | 2 | 9 | 8 |
| lead_vehicle_braking / None | 63 | 3 | 0 | 0 | 2 | 2 | 12 |
| lead_vehicle_braking / hazard(other: ahead_or_waiting) | 49 | 2 | 2 | 0 | 2 | 4 | 8 |

Scenario-added obligations (covered by the run, not by its paired control) in runs with a hard attributable issue (n=19) vs without (n=440), top by lift:

| Obligation | In issue runs | In clean runs | Lift |
|---|---|---|---|
| `lane_changing(ego)` | 5% | 0% | ∞ |
| `crossing_path(vehicle,ego)` | 11% | 2% | 4.2 |
| `adjacent_lane(vehicle,ego)` | 5% | 1% | 3.9 |
| `hazard(other: pedestrian)` | 42% | 13% | 3.1 |
| `crossing_path(pedestrian,ego)` | 42% | 16% | 2.6 |
| `hazard(pedestrian_in_path)` | 42% | 16% | 2.6 |
| `obstructing(pedestrian,ego)` | 53% | 39% | 1.4 |
| `same_lane(pedestrian,ego)` | 63% | 47% | 1.4 |
| `jaywalking(pedestrian)` | 63% | 47% | 1.3 |
| `on_road(pedestrian)` | 63% | 47% | 1.3 |
| `approaching(vehicle,ego)` | 32% | 24% | 1.3 |
| `on_road(vehicle)` | 32% | 24% | 1.3 |
| `same_lane(vehicle,ego)` | 32% | 24% | 1.3 |
| `braking(vehicle)` | 32% | 24% | 1.3 |
| `hazard(other: start_stop_or_stationary)` | 32% | 24% | 1.3 |

