# RQ3 utility results

## InterFuser

### 1. Attributable safety issues per independent run (mean ± sd)

| Policy | Runs | Valid pairs | Invalid | Collision | Low TTC | Unsafe proximity | Rule violation | Stuck / route failure | Harsh braking | Any |
|---|---|---|---|---|---|---|---|---|---|---|
| SCOUT_crit | 12 | 60 | 0 | 1.25 ± 1.22 | 2.50 ± 1.31 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.92 ± 1.38 | 0.00 ± 0.00 | 2.58 ± 1.31 |
| SCOUT | 12 | 60 | 0 | 1.00 ± 1.04 | 1.58 ± 1.31 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.67 ± 0.98 | 0.00 ± 0.00 | 1.58 ± 1.31 |
| Criticality-only | 12 | 60 | 0 | 1.25 ± 1.60 | 2.00 ± 1.76 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 1.76 | 0.00 ± 0.00 | 2.08 ± 1.73 |
| Random | 12 | 60 | 0 | 1.08 ± 1.16 | 1.50 ± 1.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.92 ± 1.38 | 0.00 ± 0.00 | 1.50 ± 1.00 |

`Any` = runs with any of Collision, Low TTC, Unsafe proximity, Rule violation. Totals over all runs:

| Policy | Collision | Low TTC | Unsafe proximity | Rule violation | Stuck / route failure | Harsh braking | Any |
|---|---|---|---|---|---|---|---|
| SCOUT_crit | 15 | 30 | 0 | 0 | 11 | 0 | 31 |
| SCOUT | 12 | 19 | 0 | 0 | 8 | 0 | 19 |
| Criticality-only | 15 | 24 | 0 | 0 | 12 | 0 | 25 |
| Random | 13 | 18 | 0 | 0 | 11 | 0 | 18 |

### 3. Evals to first attributable safety issue (any hard type)

| Policy | Runs | Found | Find rate | Evals to first (found runs, mean ± sd) | Censored mean (not found = budget+1) |
|---|---|---|---|---|---|
| SCOUT_crit | 12 | 12 | 100% | 2.00 ± 1.21 | 2.00 ± 1.21 |
| SCOUT | 12 | 10 | 83% | 2.40 ± 1.58 | 3.00 ± 2.00 |
| Criticality-only | 12 | 10 | 83% | 2.50 ± 1.51 | 3.08 ± 1.93 |
| Random | 12 | 10 | 83% | 2.10 ± 1.37 | 2.75 ± 1.96 |

### 2. Semantic obligations -> safety issues

Attributable issues by hazard target (the semantic obligation the eval was aimed at):

| Template / hazard target | Evals | Collision | Low TTC | Unsafe proximity | Rule violation | Stuck / route failure | Harsh braking |
|---|---|---|---|---|---|---|---|
| lead_vehicle_braking / None | 96 | 28 | 41 | 0 | 0 | 22 | 0 |
| lead_vehicle_braking / hazard(other: turning) | 22 | 9 | 12 | 0 | 0 | 5 | 0 |
| lead_vehicle_braking / hazard(other: ahead_or_waiting) | 24 | 7 | 10 | 0 | 0 | 7 | 0 |
| lead_vehicle_braking / hazard(other: oncoming) | 22 | 7 | 9 | 0 | 0 | 3 | 0 |
| lead_vehicle_braking / adjacent_lane(vehicle,ego) | 5 | 3 | 4 | 0 | 0 | 2 | 0 |
| pedestrian_crossing / hazard(other: pedestrian) | 24 | 0 | 7 | 0 | 0 | 2 | 0 |
| pedestrian_crossing / hazard(pedestrian_in_path) | 15 | 0 | 3 | 0 | 0 | 0 | 0 |
| lead_vehicle_braking / crossing_path(vehicle,ego) | 3 | 1 | 1 | 0 | 0 | 0 | 0 |
| pedestrian_crossing / None | 24 | 0 | 1 | 0 | 0 | 1 | 0 |
| pedestrian_crossing / braking(pedestrian) | 2 | 0 | 2 | 0 | 0 | 0 | 0 |
| pedestrian_crossing / adjacent_lane(pedestrian,ego) | 3 | 0 | 1 | 0 | 0 | 0 | 0 |

Scenario-added obligations (covered by the run, not by its paired control) in runs with a hard attributable issue (n=93) vs without (n=147), top by lift:

| Obligation | In issue runs | In clean runs | Lift |
|---|---|---|---|
| `obstructing(vehicle,ego)` | 78% | 52% | 1.5 |
| `same_lane(vehicle,ego)` | 82% | 61% | 1.3 |
| `approaching(vehicle,ego)` | 85% | 63% | 1.3 |
| `braking(vehicle)` | 85% | 63% | 1.3 |
| `hazard(other: start_stop_or_stationary)` | 85% | 63% | 1.3 |
| `in_front_of(vehicle,ego)` | 85% | 63% | 1.3 |
| `moving(vehicle)` | 85% | 63% | 1.3 |
| `node(vehicle)` | 85% | 63% | 1.3 |
| `stationary(vehicle)` | 85% | 63% | 1.3 |
| `on_road(vehicle)` | 82% | 62% | 1.3 |
| `hazard(other: ahead_or_waiting)` | 83% | 63% | 1.3 |
| `obstructing(pedestrian,ego)` | 14% | 18% | 0.8 |
| `adjacent_lane(pedestrian,ego)` | 11% | 14% | 0.8 |
| `hazard(other: pedestrian)` | 11% | 15% | 0.7 |
| `crossing_path(pedestrian,ego)` | 12% | 17% | 0.7 |

