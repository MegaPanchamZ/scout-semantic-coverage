# Implementation and artifact audit — 2026-10-01

Reviewed anonymous main at `27310be` against workspace source at `fa30f221`
plus local files. This update changes artifact packaging and documentation;
it does not repair the search algorithm or replace experiment results.

## Confirmed present before this update

AutoVLA adapter and campaign launcher, hazard-specific search wiring, constructed
nuScenes inventories, and the standalone appendix PDF/source were already on
main. Most apparent source differences were line endings. The top-level README
had not caught up with those additions.

## Packaging changes in this update

- Restore omitted `optimiser.py`, its DoTA dependency, and included modules'
  missing DoTA, PtoP, and debug-viewer imports. Before restoration, pytest stopped
  at collection with an ImportError for `research.harness.optimiser`.
- Include base-spec generation, hazard-witness statistics, failure-obligation
  mining, legacy aggregation, and the locally available lead-braking base specs.
- Sync the local lead-spec generator's reuse of an already-loaded CARLA town
  and the per-map summary fix.
- Resolve campaign roots relative to the launcher, expose CARLA_ROOT and the
  worker interpreter, and create the hazard launcher's log directory before
  redirection. Remove tracked Python bytecode and add ignore rules.
- Document ADS/profile prerequisites, distinguish original campaigns from the
  newer hazard variant, and declare offline test/plot dependencies.

## Methodology findings requiring resolution

### 1. Scheduler remains on a realised target — confirmed in saved rows

`policy_search.py::_pick_target` replaces `scheduler.uncovered` with the remaining
set, then calls `scheduler.observe(set())`. `ObligationScheduler.observe` advances
only when current is None or belongs to the passed covered set. Once current
has been credited, it is absent from uncovered but the passed covered set is
empty, so the old current persists.

Observed in local `fse_search_hazard/town01_spawn0_goal82/semantic/rows.jsonl`:
evaluation 0 records `hazard(other: ahead_or_waiting)` in `engine_first_uncover`
with index 0. Evaluations 1 and 2 retain that same `hazard_target` and
`lead_vehicle_braking` template, with zero new obligations. All 50 rows of that
arm retain the same target. The present implementation does not establish
“move to the next uncovered obligation after realisation.” These logs are not
included in this artifact; the example records the local audit observation.

Do not apply a scheduler repair midway through the active campaign. A corrected
variant requires separately labelled outputs and an explicit rerun decision.

### 2. Objective and selection are per-template, not per-obligation

`_HazardPolicyState` keeps one best candidate per template and samples uniformly
until an elite exists, then applies bounded Gaussian mutation. Random always
samples uniformly. LSA maximises `coverage_lsa_max`; KMNC maximises
`coverage_kmnc`; engine-enabled semantic search maximises newly discovered
suite obligations, breaking ties with run-covered obligation count. The older
semantic objective counts fulfilled obligations with a collision bonus.

`hazard_target` chooses a template and is recorded in rows; it does not define
an obligation-specific distance or fitness. Maps/routes run as separate arms,
with no shared cross-map scheduler. A missing template base spec silently falls
back to crossing. When no mapped targets remain, crossing continues until the
execution budget is exhausted. These details need to be reflected in comments
on the manuscript, or changed in a separately evaluated implementation.

### 3. Failure attribution needs matched-control analysis

No-adversary controls remove the generated controller/walkers but retain the
route. A high control collision rate alone does not prove the ADS or semantic
method is responsible. Inspect collision counterpart, timing/location, agent
initialisation, route validity, and route completion for matched generated and
control executions. Report the matched difference with uncertainty and separate
hazard-associated failures from background/static-object failures. This audit
does not independently validate the quoted 50% control collision figure.

### 4. Evidence and reproduction are incomplete

- The repo contains no execution rows, traces, fitted nominal profiles, or final
  statistics. Table 6 completeness and manuscript numbers cannot be verified
  from this repository alone.
- AutoVLA conversion is described in patch notes but its executable conversion
  script and exact nominal profile provenance are absent.
- Third-party patches are prose notes, not applicable diff files. Upstream source
  and weights remain external; GPU environments need additional dependencies.
- Deterministic inventory construction is not grounding validation. Publish the
  sampling protocol, reviewed obligation sample, oracle labels, and outcomes.
- Multi-map results do not supply independent repeated seeds. Verify the actual
  unit of pairing/replication and intervals/tests before claiming statistical
  significance across ADSs, maps, and repeated runs.

## Active experiment snapshot

At the read-only status check, both hazard workers and both CARLA processes were
active; worker logs had been modified within 15 seconds. There were 2,387 saved
rows, 56 completed arms, and 58 started arms out of 70 planned arms / 2,912 rows
(14 routes, four policies × 50 executions plus eight controls per route).
These are progress counts, not a success or validity check. The campaign is
managed separately by another agent; this update does not touch its workspace.

## Validation

After restoring source, offline suite: **101 passed, 1 skipped**, both with the
existing workspace interpreter and with a fresh Python 3.10 environment installed
using `uv sync --directory research --python 3.10 --frozen`. The baseline suite
failed during collection. All shell scripts pass syntax checks; all ten worker
launcher/worker combinations pass from outside the checkout using a non-executing
interpreter stub. Included Python sources have no missing absolute `research.*`
imports. GPU/CARLA reproduction and original statistics were not rerun. The
lockfile was regenerated against the artifact manifest, removing stale Runpod
dependencies absent from that manifest.
