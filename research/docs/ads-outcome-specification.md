# ADS outcome specification — schema v2 (`scout-outcomes-2.2-draft`)

Implementation status: complete software stack implemented (offline evaluators, traffic-rule/lane/hazard-braking detectors, live physics-step capture adapter, v2 ledger/reporting, sensitivity grid, semantic-to-outcome linkage, controlled validation suite, and matched 5-evaluation pilot harness). The evidence schema remains 2; the definition identifier is `scout-outcomes-2.2-draft`. The default profile remains `draft` until live CARLA validation runs pass and its `validation_refs` are recorded; `require_pilot_ready()` enforces that gate.

## Interfaces and provenance

`research/harness/safety_outcomes.py` provides `OutcomeEvaluator.observe(FrameEvidence)`, `record_sensor_event(SensorEvidence)`, and `finish(CaptureManifest)`. Frames must have strictly increasing simulator frame numbers and simulation timestamps. Callbacks can arrive in any order before finalization. `finish()` is deterministic and does not mutate the recorded evidence. Live physics-step capture is wired into `HarnessRunner` via `_LiveOutcomeCapture` when `cfg.outcomes.enabled=True` (`--enable-outcome-capture`), persisting the JSONL trace first and evaluating the persisted trace so live and offline replay agree identically.

`classify_run(payload)` now returns schema 2. Without normalized evidence it conservatively adapts historical logs. `classify_run(payload, evidence=records)` requires `payload['capture_manifest']`. The iterable must contain normalized model instances. No referenced file is implicitly opened by classification.

The historical classifier is frozen in `research/harness/legacy_outcomes.py`. Existing policy search, paper scripts, and historical reports default to schema 1 for backward compatibility, while opting into schema 2 via `--outcome-schema 2` (and `--feedback-version v2_normalized` for search feedback). `classify_run(payload, schema_version=1)` also reproduces the legacy defaults. Do not feed new nested assessments into old boolean checks.

The draft profile is `research/harness/outcome_profiles/default-v2.json`. All settings include units, provenance and rationale. Profiles are immutable, validated and hashed by canonical content; results also include a digest of the implementation files, so uncommitted code is identifiable. Reordering settings does not change a profile digest. Changing a threshold or its provenance does. Draft profiles are rejected by `require_pilot_ready()`; live-validation references are required to mark a profile validated.

Task-window/progress and interaction-quality settings are optional, all-or-none extension groups. Older profiles load without acquiring new defaults; detectors requiring an absent group return unknown. The current default explicitly includes both groups. New frame/manifest context fields are optional, preserving reads of earlier captures without inventing context.

## Meaning of assessment states

- `present`: supported positive evidence was observed. A later simulator failure does not erase it.
- `absent`: no qualifying event was found within the explicitly complete, supported observation domain. This is not proof of general safety outside that domain.
- `unknown`: evidence is missing, incomplete, insufficient for the definition, or the detector is not implemented.
- `not_applicable`: explicit complete context establishes no active objective throughout the observation interval for a task metric. Missing evidence never establishes inapplicability.

`MetricAssessment` refuses boolean coercion; use `assessment_is_present()`. Serialized assessments must be queried by their `status`. An OR aggregate is present if any constituent is present, absent if all applicable constituents are absent, not applicable if there are no applicable constituents, otherwise unknown. Safety, behaviour, rules, task and quality remain separate groups. There is no global `unsafe` boolean in v2.

Complete negative assessments require explicit channel coverage, adequate duration, frame coverage at both interval endpoints, no capture discontinuity, available required channels at every frame, and no sample gap above the profile limit. The manifest's declared channel coverage must be established by future capture instrumentation; this offline library cannot independently attest to sensor health. The projection/scan domain must support geometry-dependent negative claims.

Execution is recorded independently as completed, partial, failed, not executed, dry run, or unknown. The last state is needed for archive rows whose execution cannot be established. Scenario invalidity is never inferred just from an execution error.

## Implemented measurements and labels

| Measurement/label | Operational definition and initial draft setting | Required evidence / exclusions | Regression coverage |
|---|---|---|---|
| Footprint clearance | Minimum horizontal distance between actual oriented rectangular footprints, including local box offset/yaw; separate vertical layers yield no applicable horizontal gap. | Finite box geometry and world transform. Yaw-only and vertically static projection domain. | `test_interaction_geometry.py`: rotated boxes, offsets, overlap, road decks. |
| TTC | Earliest overlap of two translating fixed-heading rectangles, with interval intersection over separating axes. Initial horizon 3 s. Contact prediction, no predicted contact in horizon and initial overlap are distinguished. | Horizontal velocities, footprint geometry and vertical intervals. Constant heading/velocity assumption; no turning/vertical-motion model. | Analytic rear-end, crossing vs time-separated paths, diverging/parallel actors, short contact intervals and horizon edges. |
| Collision | Sensor callback is positive evidence; group same-actor callbacks separated by no more than 0.25 s. Preserve callback count and peak impulse norm in N·s. | Collision channel; callback phase must be evaluation. A geometrically predicted overlap does not create a collision event. Final contact remains censored if sufficient subsequent sensor coverage is missing. Impulse is not injury severity. | Callback grouping, late/duplicate callbacks, partial-run positives, checksum/replay tests. |
| Near miss | Begin an encounter when TTC < 1.5 s. Close after at least 0.5 s continuously observed without low TTC or clearance < 2.5 m. Confirm a near miss only if the encounter closes without same-actor sensor contact and collision coverage spans it. | Missing actor, missing frame/channel, phase break or horizon truncation censors the encounter. Same-encounter collision suppresses near miss. Separate encounters can produce both run labels. | Resolved encounter, same/different actor collision, missing actor, horizon, gap and absent sensor controls. |
| Conflict exposure | Preserve low-TTC encounter evidence even if it collides or remains unresolved. | Diagnostic event only; does not substitute for completed near miss. | Censoring and contact tests. |
| Close proximity | Footprint gap < 2.5 m; group consecutive qualifying samples by actor. | Complete actor geometry/motion for negative assessment. Close parallel traffic can be proximity without a near miss. | Parallel adjacent motion, grouping and projected-path negative controls. |
| Hard braking | Recorded longitudinal acceleration < −8 m/s², positive brake command, forward motion, pre-event speed > 5 m/s; qualifying sample span >= 0.1 s. | Requires acceleration, controls and motion. Discontinuities break streaks; contact during the streak or preceding sample interval excludes impact deceleration. Complete collision-channel evidence is required to exclude impacts. | Threshold equality, speed gate, control absence, missing fields, impact exclusion. |
| Hazard-associated braking | Commanded `hard_braking` episode overlapping or following a predicted low-TTC conflict encounter within `hazard_braking_antecedent_s` (1.5 s); passive parallel `close_proximity` alone does not qualify. | Observational association only (records actor IDs, hazard kind and lag), never causal fault attribution. Hard braking with complete actor/collision coverage and no nearby conflict gives `absent`. | `test_traffic_rule_outcomes.py`: hazard-linked vs. non-hazard braking. |
| Lane crossing | `lane_invasion` sensor callbacks grouped within `max_sample_interval_s`. Descriptive only; does not imply illegality. | `lane_invasion` channel coverage. Disabled sensor gives `unknown`. | `test_traffic_rule_outcomes.py`: permitted vs. prohibited crossings, missing sensor. |
| Illegal lane change | `lane_crossing` outside a junction across prohibited markings (`Solid`, `DoubleSolid`, `SolidSolid`, `Curb`, `Grass`, `BottsDots`) or when `lane_change_permission == "none"`. | `lane_invasion` + `road_context` (`in_junction`, `lane_change_permission`). Junction transitions excluded; missing or ambiguous markings (`SolidBroken`) give `unknown`. | `test_traffic_rule_outcomes.py`: broken vs. solid, junction exclusion, disabled sensor. |
| Unsafe lane departure | Entry onto a prohibited road surface (`sidewalk`, `shoulder`, `median`, `offroad`, `restricted`, `biking`, `parking`) outside a junction, OR a non-junction `lane_crossing` overlapping an active actor conflict. | `road_context` (`lane_type`, `in_junction`), `lane_invasion`, and actor conflict coverage. | `test_traffic_rule_outcomes.py`: sidewalk departure, clean road control. |
| Red-light violation | Ego crosses an applicable directed stop line (`stop_line_applicable=True`, `stop_line_crossed=True`) while `red_light_state == "red"`. | `road_context` + directed swept front-footprint stop-line test (`road_context.crosses_directed_line`). Approaching on red without crossing, unrelated-lane signals, and green crossings give `absent`; unobserved transitions or missing signal state give `unknown`. | `test_traffic_rule_outcomes.py`, `test_outcome_capture.py`. |
| Stop-sign violation | Ego crosses an applicable stop-sign line without first dwelling at speed <= `stop_sign_speed_mps` (0.5 m/s) for >= `stop_sign_dwell_s` (1.0 s) inside the approach zone. | `road_context` + `ego_motion`. Stopping outside the approach zone does not count as compliance. | `test_traffic_rule_outcomes.py`: compliant dwell, roll-through, early stop outside approach. |
| Speeding | Speed strictly exceeds `speed_limit_mps + speeding_tolerance_mps` (1.0 m/s) continuously for >= `speeding_duration_s` (1.0 s). | `road_context.speed_limit_mps` + `ego_motion`. Handles mid-run speed-limit transitions; unknown speed limit gives `unknown`. | `test_traffic_rule_outcomes.py`: tolerance, duration, limit drop, missing limit. |
| Wrong-way | Opposing velocity component along `permitted_travel_yaw_rad` strictly exceeds `wrong_way_speed_mps` (1.0 m/s) continuously for >= `wrong_way_duration_s` (1.0 s). | `road_context` + `ego_motion` + `reverse`. Excludes permitted reverse manoeuvres (`reverse=True`) and junctions (`in_junction=True`). Missing `reverse` gives `unknown`. | `test_traffic_rule_outcomes.py`: opposing motion, legal reverse, junction, missing reverse. |
| Prolonged immobilization | Speed < 0.2 m/s for observed span >= 10 s. | Uses simulation timestamps; never extrapolates before first sample. Startup hold and gaps break the streak. A qualifying streak still present at termination is censored but proves the observed duration. | Long stationary interval, phase exclusion, short/incomplete trace. |
| Stuck | Speed < 0.2 m/s throughout an observed window >= 10 s, with an active unfinished objective and all applicable waiting reasons explicitly ruled out. | `ego_motion` and `task_context`. Red-light/stop-sign dwell, legitimate blockage/queue and yielding break eligible windows. Sensor waits and missing context are unknown. This is a contextual stationary label; route-based lack of progress is assessed separately. | Legitimate waits, missing context, sensor wait, threshold equality, continuous windows and later evidence loss. |
| No progress | Route high-water arc length advances by <= 1 m over an eligible window >= 10 s. | `task_context` and available `route_progress`. Requires the same route and continuity ID, nondecreasing high water, continuous capture and no legitimate wait. Reversing or retracing already covered route does not create new progress. | Forward advance, creeping, reversing, route changes, reacquisition, phase breaks and simulation-time windows. |
| Route incomplete | Explicit `reached_goal=False` on an executed run. True gives absent. | Completion is descriptive; no traffic or safety implication. Historical goal state is preserved with legacy provenance. | Missing state, legacy runs, partial execution. |
| Timeout | Explicit completion deadline exhausted, `reached_goal=False`, termination reason `horizon_exhausted`, executed status and duration >= deadline. | A short diagnostic horizon without a declared deadline does not establish timeout. Otherwise unknown in this increment. | Deadline vs short horizon vs collision termination. |
| Invalid scenario | Any explicit failed configuration, initial geometry, required actor spawn, or route-feasibility check. | `ScenarioChecks` on the manifest. All feasibility checks and controller readiness must pass for absent. An execution/controller error alone is unknown, not invalidity. A later execution error does not erase successfully established setup validity. | Spawn failure, controller failure, execution error and context round trip. |
| Low interaction | Valid scenario with >= 10 s complete supported actor/motion/collision capture, no observed contact and no nearby/projected interaction opportunity. | Opportunity proxy: vertically overlapping footprints within 15 m, or predicted contact within 3 s. Safe nearby interactions count. Positive opportunity/contact proves low-interaction absent; insufficient capture or unestablished validity gives unknown. | Empty valid scene, safe parallel actor, fast approaching actor beyond radius, contact outside actor capture, insufficient duration/coverage. |

Current numerical choices beyond initial supervisor proposals are proposed experimental settings for offline tests. No threshold has been tuned against policy performance. Sensor contact overrides a near-miss interpretation for the same encounter; footprint overlap alone never overrides the sensor channel.

`hard_braking` is a behaviour measure; it does not imply a hazardous cause, an ADS defect, or a rule violation. `prolonged_immobilization` is a duration measure; it does not establish unjustified stopping. Road-rule, lane-legality, and hazard-associated-braking detectors are implemented in `research/harness/traffic_rule_outcomes.py` and stay `unknown` whenever required road-context or sensor channels are missing or incomplete. Low interaction is a spatial/kinematic proxy within the declared observation domain, not a certificate of triviality or absence of every semantic challenge.

## Waiting context and route tracking

`FrameEvidence.task_context` carries `TaskContext`: `objective_active`, `red_light_stop`, `stop_sign_stop`, `blocked_route`, `yielding`, `sensor_wait`, and supporting `evidence_refs`. Each flag is true, false, or unknown (`None`). A true wait flag must refer to a reason applicable to the ego's current task: a red light elsewhere in the map or an arbitrary nearby actor does not establish legitimate waiting. `blocked_route` specifically denotes a legitimate observed queue/blockage reason, not an inferred excuse for every failure to move. Sensor wait denotes an execution limitation, not a traffic reason. Live extraction from CARLA is implemented in `research/harness/road_context.py` (`RoadContextIndex.sample()`), leaving flags as `None` whenever required map or actor channels are unavailable.

`RouteProgressTracker` in `route_progress.py` projects the ego reference position onto a planar route polyline. `update()` records current arc length, high-water arc length and cross-track offset. Its experimental defaults are a 3 m corridor, 60 m/s projection continuity bound, 1 m numerical/position slack and 0.11 s maximum interval. The first observation must match the supplied initial route anchor; subsequent projections are bounded by elapsed time. Gaps, explicit discontinuities and unmatched projections require `reset()` with a trusted anchor. Each reset changes `continuity_id`; no-progress windows cannot bridge it. Loops and intersections cannot trigger unrestricted nearest-point jumps. The planar model is not sufficient by itself for stacked roads or ambiguous map matching. Capture instrumentation should store `tracker.provenance()` with the route source and anchor evidence.

Task windows retain the last sample at or before their time boundary. Their measured span may exceed the requested window by at most one allowed sample interval; there is no interpolation or extrapolation. Qualifying adjacent windows form one event; events still active at termination or interrupted by missing evidence are censored. Their already-observed duration still establishes presence. Known exclusions reset windows without making a fully observed run unknown. An observed positive remains present after later evidence loss. Stuck uses speed/context directly; no-progress separately requires route evidence.

## Input identity and observed-event diversity

`scenario_quality.py` provides three separate input comparisons:

- `canonical_scenario_digest(spec, execution_config)` hashes the effective requested spec plus explicit map, ADS and route digests and integer seed. Callers must resolve defaults and supply remaining effective execution settings. It removes only top-level `scenario_id`, `description`, `output_dir`, and `output_path`; nested fields are retained. Equivalent integral JSON numbers normalize consistently. This does not prove identical realized execution.
- `realized_initial_state_digest(actors, execution_config)` hashes observed actor geometry, pose and velocity using unique stable roles. It ignores ephemeral simulator actor IDs and preserves a separate criterion from requested input identity. Missing/duplicate roles are rejected. This captures actor state, not every hidden simulator state.
- `parameter_similarity(a, b, space)` returns range-normalized RMS distance over the space's numeric dimensions, with dimension/range metadata. It neither rounds nor clips candidates. Zero-width dimensions contribute zero; missing/nonfinite/out-of-bounds inputs are rejected. This is an input-distance measure, not behavioural equivalence. Existing `SearchSpace.key()` remains a grid-coerced search key.

`outcome_diversity.py` implements the deliberately coarse `ordered_event_role_pattern_v1` criterion: event labels, closed/censored states, stable actor types/roles, ordering and simultaneous groups. Absolute event times, ephemeral actor IDs and run IDs do not affect the signature; source event IDs/times remain available as evidence. Assessment states and execution/termination status are retained so observed negatives cannot be equated with unknown outcomes. Profiles, definition versions, detector code and signature code form separate comparison scopes. Road geometry, manoeuvre classes, severity bands and causal/bug identity are not inferred.

Missing or reassigned actor roles make a run signature unknown. Runs with no event episodes remain unclustered; that does not mean they are safe. Exact signatures form deterministic groups with source membership and a representative. Reports count source records and repeated observed patterns within each scope; they do not infer unique campaign attempts or compare matched testing budgets. The CLI rejects supplying the same run/evidence twice. New identities/signatures do not change search selection or deduplication.

## Evidence files and replay

`outcome_io.py` writes append-only strict JSONL with schema/kind envelopes and run identity checks. `EvidenceWriter.finish()` produces a manifest with trace checksum and record count. Summary JSON writes are atomic. Existing trace files cannot be clobbered by creating another writer. Strict normalized JSON rejects NaN/Infinity; the explicit legacy adapter alone permits reading old log formats that contain them.

Corrupt/truncated evidence raises `EvidenceReadError` with line number (and byte offset on torn UTF-8). Records yielded before that error remain available to a caller, and `recover_partial_trace()` can explicitly recover valid prefix records into a fresh `"partial"` trace and manifest with explicit corruption provenance, while standard replay refuses to finalize a corrupted capture as complete. The CLI verifies checksums before replay.

Replay a normalized capture (when one has been recorded or supplied):

```bash
research/.venv/bin/python research/scripts/reclassify_outcomes.py \
  --capture-manifest /path/to/trace.jsonl.manifest.json \
  --output-dir /path/to/new-replay-output
```

Audit a historical campaign into fresh sidecars:

```bash
research/.venv/bin/python research/scripts/reclassify_outcomes.py \
  --search-root research/logs/fse_smoke \
  --output-dir /path/to/new-audit-output
```

Replay declared profiles on the same normalized capture:

```bash
research/.venv/bin/python research/scripts/outcome_sensitivity_report.py \
  --capture-manifest /path/to/trace.jsonl.manifest.json \
  --profile research/harness/outcome_profiles/default-v2.json \
  --profile /path/to/alternative-profile.json \
  --output-dir /path/to/new-sensitivity-output
```

Sensitivity replays geometry and event grouping from raw normalized records, reusing cached evidence across variants and marking derived grid variants as `draft` with cleared `validation_refs`. It rejects unsupported capture bounds and reports status, aggregate, episode-count, and event-membership diffs. It cannot establish how adaptive search would behave under changed feedback. Use `--use-default-grid` for the frozen pre-pilot one-factor-at-a-time + joint sensitivity grid.

Replay normalized captures into a diversity sidecar:

```bash
research/.venv/bin/python research/scripts/outcome_diversity_report.py \
  --capture-manifest /path/to/first.jsonl.manifest.json \
  --capture-manifest /path/to/second.jsonl.manifest.json \
  --output-dir /path/to/new-diversity-output
```

The reporter verifies trace checksums, re-evaluates raw evidence and writes `diversity.json`. Legacy aggregate logs lack the stable actor/time evidence needed for this criterion and cannot be passed as normalized captures.

## Historical-data interpretation

The audit retains corrupt rows, missing/unreadable artifacts and unassessable metrics. Counts are explicitly **source-row counts**, not deduplicated execution counts; matching and attempt-identity migration remain pending. Per-metric rates condition on present/absent assessments and include unknown counts. Positive collision records can be available when negative sensor-health evidence is missing, so these conditional rates must not be used as policy performance comparisons.

Explicit legacy collision records support contact occurrence with an unresolved phase annotation. Old collision counters alone do not establish v2 sensor evidence. Old near-collision counts, radial TTC, centre distances, lane offsets and braking extrema cannot establish the new contextual labels. Legacy empty/missing fields remain unknown, including collision absence.

## Live capture, reporting, linkage, validation and pilot

- **Live CARLA physics-step capture**: `research/harness/runner.py` (`_LiveOutcomeCapture`), `oracles.py` (`SafetyOracle.capture_frame_evidence`, `CollisionOracle.drain_events`, `LaneInvasionOracle`), `road_context.py` (`RoadContextIndex`, `crosses_directed_line`), and `run_shakedown.py` (`--enable-outcome-capture`, `--outcome-profile`, `--outcome-evidence-output`) observe every physics substep without altering ADS control cadence, isolate per-channel errors, drain sensor queues, and write partial envelopes on exceptions.
- **V2 reporting & ledger**: `research/harness/safety_utility.py` (`schema_version=2`), `research/scripts/safety_outcome_report.py --outcome-schema 2`, and `research/scripts/safety_utility_report.py --outcome-schema 2` retain every attempted evaluation (including missing/unreadable/failed runs), count malformed and duplicate rows, enforce budget cutoffs, report separate category tables with `present / absent / unknown / not_applicable` denominators, and display complete-execution strata alongside all observed evidence.
- **Frozen sensitivity grid**: `research/scripts/outcome_sensitivity_report.py --use-default-grid` replays the pre-pilot one-factor-at-a-time grid (TTC 1.0/1.5/2.0 s; proximity 1.5/2.5/3.5 m; braking 6/8/10 m/s²; stationary speed 0.1/0.2/0.3 m/s; immobilization 5/10/15 s) plus declared joint strict/lenient variants directly from normalized evidence.
- **Semantic linkage & guidance diagnostics**: `research/harness/outcome_linkage.py` and `research/scripts/guidance_diagnostics_report.py` require same-run, same-actor/role, and antecedent time windows while separating circular outcome-derived predicates (e.g., `colliding`) from predictive antecedent claims.
- **Controlled validation & matched pilot**: `research/experiments/outcome-validation/run_validation.py` defines 18 positive/negative/uncertainty controls across all outcome families; `research/experiments/EXP-020-policy-comparison/outcome-pilot/run_pilot.py` enforces validated-profile gating (`require_pilot_ready()`), collision-free baseline admission, matched 5-evaluation budgets with failed attempts charged, shared eval-0 accounting, and no automatic budget extension.

## Literature grounding and validation limits

| Source | Section / Artifact | Supported Claim | Departures & Sensitivity Range |
|---|---|---|---|
| [FHWA SSAM final report](https://www.fhwa.dot.gov/publications/research/safety/08051/02.cfm) | Chapter 2 (Surrogate Safety Measures) | Trajectory projection and 1.5 s TTC conflict threshold as a standard surrogate safety starting reference. | Fixed-heading oriented-rectangle constant-velocity projection (no turning/vertical model); swept at 1.0 / 1.5 / 2.0 s in the frozen sensitivity grid. |
| [ISO 34502:2022](https://www.iso.org/standard/78951.html) | Scenario-based safety evaluation framework | Motivates separating scenario validity, operational domain coverage, and safety-critical event assessment. | Catalog framing only; no clause-level legal conformity claim is asserted. |
| [DriveFuzz (Kim et al., CCS 2022)](https://drivefuzz.s3lab.io/ccs22-drivefuzz.pdf) | Driving-quality & misbehaviour oracles | Motivates tracking driving-quality signals (hard braking, lane crossings, immobilization) separately from collision outcomes. | We separate descriptive quality signals (`hard_braking`, `lane_crossing`, `prolonged_immobilization`) from contextual violations (`hazard_associated_braking`, `illegal_lane_change`, `stuck`); our 8 m/s² threshold follows supervisor proposal and is swept at 6 / 8 / 10 m/s². |
| [CARLA 0.9.16 sensor reference](https://carla.readthedocs.io/en/0.9.16/ref_sensors/) | Collision & lane-invasion sensors | Supplies frame-stamped impulse vectors and crossed OpenDRIVE lane markings. | Lane crossings are enriched offline/online with map topology (`in_junction`, `lane_change_permission`, `lane_type`) so permitted crossings never become violations. |

Unit and offline-synthetic tests establish software correctness and live/replay equivalence; graduating the draft profile (`scout-outcomes-2.2-draft`) to `validated` still requires executing the live CARLA controls and recording their `validation_refs`.
