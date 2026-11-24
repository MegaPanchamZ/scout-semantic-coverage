**SCTrans source replay and admission checks**

This runner reads the public CommonRoad XML directly. It converts source maps
with CommonRoad Scenario Designer 0.8.5, preserves source actor IDs, classes,
positions, headings and timestamps, and uses the source planning problem for
the ego's initial pose, speed and goal. Coordinates change handedness for CARLA
(`x, -y, -yaw`); they are not transplanted onto Town01/Town05 routes.

The [2026-11-02 report](../artifacts/sctrans-validation-2026-11-02/report.md)
records **zero admitted scenarios**. These files do not establish a successful
SCOUT comparison. The legacy task-346 entry point has been disabled and its
outputs marked invalid.

NPCs follow recorded samples using transform resets and finite-difference
velocities through CARLA physics. Position and heading errors are measured
after every tick. Actors disappear at the end of their recorded intervals;
trajectories are never extrapolated. CARLA blueprints approximate the source
shapes; both sets of dimensions are recorded. Converted geometry, weather,
ego blueprint, and blueprint dimensions prevent claiming full visual or physical
equivalence to the source. A converted map is not automatically a valid replay.

InterFuser runs through the existing PclaAdapter. The runner applies its returned
controls without changing throttle, brake, private agent state or model weights.
Setting the source initial velocity happens once, before evaluation. Collision
outcomes come only from `sensor.other.collision`. Minimum gap is the distance
between the horizontal polygons of the actual oriented CARLA bounding boxes.
A zero gap never creates a collision event.

Admission requires recorded provenance, all actors placed, all expected ticks,
ego displacement of at least 2 m, the declared intended hazard and its constituent
predicate witnesses on a mapped source actor, maximum replay error <= 0.25 m
and heading error <= 5 degrees, map alignment checks, collision-sensor evidence,
and no baseline collision. Diagnostic runs and perturbed baselines are rejected.
The movement threshold is a feasibility screen, not a route-completion claim.

The current map alignment screen samples each tenth actor state, checks its
distance outside the nearest driving lane, and checks the ego and goal. It
does not prove complete lane topology or traffic-control equivalence. Detailed
per-run metrics and failure traces are retained for repair and review.

Run the tests from the repository root:

```bash
research/.venv/bin/python -m unittest discover \
  -s research/experiments/EXP-020-policy-comparison/sctrans -v
```

Convert maps in an isolated environment, keeping the ADS environment unchanged:

```bash
uv venv --python 3.11 /tmp/sctrans-converter
uv pip install --python /tmp/sctrans-converter/bin/python \
  -r research/experiments/EXP-020-policy-comparison/sctrans/converter-requirements.txt
/tmp/sctrans-converter/bin/python \
  research/experiments/EXP-020-policy-comparison/sctrans/convert_maps.py \
  /path/to/SCTrans-docker/all_source_xml --output /tmp/sctrans-maps
```

Start CARLA 0.9.16 on a dedicated port, then audit and validate sequentially:

```bash
research/.venv/bin/python \
  research/experiments/EXP-020-policy-comparison/sctrans/run_validation.py \
  --sources /path/to/SCTrans-docker/all_source_xml \
  --maps /tmp/sctrans-maps --output /tmp/sctrans-validation --port 2510
```

The current source files and converted maps remain available at the absolute
paths recorded in the evidence manifest. Use a fresh output directory. The
batch command performs one feasibility attempt per usable source, not policy
search. Malformed maps are rejected before contacting the simulator. Initial
versions lacked that check; the Ffb conversion crashed CARLA during loading.

`--safe-route-topology` is an optional, explicitly recorded diagnostic repair
to PCLA's route construction. It stops topology sampling at a dead end and
never invents a connection. Its Peach test progressed to a genuine no-path
failure; it did not make the source admissible. It is disabled by default.

The policy adapter imports the actual `PolicyState`, scoring functions and
`make_policy_state` from EXP-020's `policy_search.py` and `baselines.py`. The
aliases are random -> random, criticality -> critonly, scout -> semantic, and
scout_crit -> semantic_crit. Tests compare feedback and candidate sequences
against direct calls to those implementations. This uses EXP-020's shared
SearchSpace mode; its hazard-search scheduler is not enabled. This distinction
must be retained in any later experimental description.

The gated comparison command is prepared for an admitted original replay:

```bash
research/.venv/bin/python \
  research/experiments/EXP-020-policy-comparison/sctrans/compare_policies.py \
  --admission-result /path/to/admitted/result.json \
  --output /tmp/sctrans-comparison --port 2510
```

It refuses failed, incomplete, altered-source, altered-code and altered-oracle
admission records before launching a campaign. It counts one shared admitted
eval 0 plus four new evaluations per arm, with no automatic extension. Search
changes one intended source actor's playback rate, time offset and lateral
offset; those runs are labelled source-derived variants. The full live search
path remains unvalidated because no baseline passed admission. Do not use this
adapter's unit checks as evidence of SCOUT effectiveness or completed live
comparison.

Converter reference: [CommonRoad Scenario Designer](https://github.com/CommonRoad/commonroad-scenario-designer).
Simulation API reference: [CARLA Python API](https://carla.readthedocs.io/en/latest/python_api/).
