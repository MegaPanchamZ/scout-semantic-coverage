# Safety outcomes beyond collision

The utility question — *does the search find safety issues?* — must not be
answered by collision alone. A method can surface a near miss, an unsafe gap, a
rule violation, a stuck vehicle or a harsh emergency manoeuvre that never
becomes a contact, and those are exactly the failures a semantic search is
supposed to expose. This layer turns recorded simulator logs into a fixed set
of named unsafe behaviours.

## Outcome taxonomy

| Outcome | Evidence (recorded in the run log) |
| --- | --- |
| `collision` | collision-sensor contact (existing ground truth) |
| `near_collision` | rising-edge entry into a near band, or `min_ttc` below threshold |
| `unsafe_proximity` | minimum pedestrian or vehicle gap below a threshold |
| `red_light_violation` | ego at a red light above a speed threshold |
| `lane_departure` | lane-centre offset above a threshold while moving |
| `traffic_rule_violation` | `red_light_violation` or `lane_departure` |
| `stuck` | stationary streak past a time threshold |
| `no_progress` | episode ended short with negligible route progress |
| `route_incomplete` | goal not reached within the tick budget |
| `harsh_braking` | longitudinal deceleration above a threshold |
| `emergency_manoeuvre` | very hard deceleration, or hard brake plus hard steer |

`unsafe` is the disjunction of all of the above. The first four map directly to
the behaviours requested in review; `harsh_braking`/`emergency_manoeuvre` cover
harsh manoeuvres.

## Where the evidence is recorded

Detection happens live in `research/harness/oracles.py::SafetyOracle` (schema-2
`SafetyMetrics`), so no post-hoc guess about other actors is needed:

- `safety_metrics` in each run JSON now carries `min_ttc`,
  `min_pedestrian_distance_m`, `min_vehicle_distance_m`, `near_collisions`,
  `red_light_violations`, `lane_departure_events`, `max_lane_offset_m`,
  `max_stationary_streak_ticks`, `max_deceleration_mps2`,
  `harsh_braking_events` and `emergency_manoeuvre_events`.
- Each telemetry tick now also carries a compact `safety` block (current
  `min_ttc_s`, nearest actor type/distance, lane offset, deceleration), so the
  per-tick trace is auditable.
- `research/harness/safety_outcomes.py::classify_run` applies named thresholds
  to a run log and returns the booleans plus the raw values.
- `policy_search._row_from_run` copies `safety_*` fields and the raw values
  into every `rows.jsonl` row.

## Reporting

Aggregate a campaign by safety outcome instead of collision:

```bash
research/.venv/bin/python research/scripts/safety_outcome_report.py \
  --search-root research/logs/policy_search \
  --out-dir research/logs/safety_report \
  --include-paired-controls
```

It writes `safety_report.json` and `safety_report.md`: per-policy unsafe-run
rate, per-outcome counts/rates, the number of distinct issue types each method
found, and the first unsafe evaluation. Because it joins each row to its
`run_json_path`, it scores rows produced before these fields existed, but those
legacy logs only have schema-1 evidence (collision, goal, coarse stuck), so the
new outcomes require a re-run.

## Utility evaluation: how many, from what, how fast

`research/scripts/safety_utility_report.py` produces the three discussion
quantities for the utility question. It expands a campaign root's `seed-*`
children automatically, so several repeats can be analysed at once:

```bash
research/.venv/bin/python research/scripts/safety_utility_report.py \
  --search-root research/logs/fse_search_gap_v1 \
  --out-dir research/logs/safety_utility \
  --primary-policy semantic \
  --budget 50
```

1. **How many safety issues.** Per-policy unsafe-run rate and the distinct
   outcome types the method discovered.
2. **Which obligations lead to which issues.** `obligation_association` joins
   the obligations witnessed in a run (or the actively targeted obligation,
   `--association-source target`) to the run's outcomes. For each obligation it
   reports the outcome rate when the obligation is present versus absent, the
   risk ratio and the absolute lift. The markdown reports the top obligations
   overall and the leading obligation per issue type. These are associations,
   not causal effects.
3. **How fast the first issue appears.** One independent run is a
   `(policy, replicate, route)` search arm; `first_eval` is the smallest
   evaluation index whose run had *any* safety issue. The report gives the
   mean/std (and median, found-rate and Top-1 rate) across arms and across
   replicates, plus paired primary-vs-baseline differences over shared
   `(replicate, route)` pairs. Arms that never find an issue within the budget
   are right-censored and enter the censored mean/std as `budget + 1`; always
   read the found rate next to any found-only mean. Lower iteration counts are
   better; `--budget` fixes the censoring horizon so arms are compared under
   the same testing budget.

The report also writes `safety_utility.json`, which carries every arm and the
full per-obligation contingency tables for downstream statistics.

`research/harness/optimiser.py` also records `safety_unsafe` and the full
`safety_outcome` in its evaluation summaries. This is reporting only: the
search score and `collision_bonus` are unchanged, so existing fitness values
stay comparable.

## Interpreting the numbers

- **Proxies, not labels.** Lane offset, TTC and deceleration thresholds are
  engineering approximations. Report sensitivity to
  `--near-collision-ttc-s`, `--lane-offset-m`, `--harsh-deceleration-mps2` and
  `--stuck-seconds` alongside the headline rates, and do not present them as
  validated ground truth.
- **No attribution.** A `near_collision` or `unsafe_proximity` outcome says an
  unsafe state occurred; it does not by itself prove the injected hazard caused
  it. Pair with the paired no-adversary control (`--include-paired-controls`).
- **`route_incomplete` is not a collision.** A nominal but slow run may fail to
  reach the goal; keep the outcome breakdown separate rather than collapsing it
  into one "failure" bit.
- **Re-runs only.** Use the same rule as the rest of the corrected protocol: do
  not mix schema-1 and schema-2 safety metrics in one comparison.

## Suggested utility-question wording

For each method and route, report *unsafe-run rate* and *distinct issue types
found* over the named outcomes, and treat collision as one outcome among many
in the paired comparison. This makes the utility claim ("the search finds
safety issues") measurable without depending on a contact that may never occur.
