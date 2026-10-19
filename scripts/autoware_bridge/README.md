# Autoware <-> CARLA bridge

Uses the upstream `autoware_carla_interface` (Autoware Universe, ROS 2 Humble). It is already inside `ghcr.io/autowarefoundation/autoware:universe`, so nothing was built. This is a wrapper and a test around it.

## Result (2026-10-18, RTX 5060 Ti x2)
`PERCEPTION=real ./bridge_test.sh` passes on Town01 **with real Autoware perception** (lidar CenterPoint, TensorRT 10.8 / CUDA 12.8 on Blackwell):

```json
{"localized": true, "route_set": true, "engaged": true, "engage_wait_s": 2.0,
 "distance_m": 77.12, "peak_speed_mps": 4.5, "goal_dist_m": 2.91, "reached_goal": true}
```

Autoware drove the CARLA ego 80 m to the goal with its own perception, planning and control. `PERCEPTION=stub ./bridge_test.sh` (empty world, `perception:=false`) is only a fast actuation check and is invalid for experiments.

## Layout
| Piece | Where | Why |
|---|---|---|
| CARLA **0.9.15** server | native, `run_carla_native.sh` (`${CARLA_0915_ROOT:-$HOME/carla-0.9.15}`) | The bridge only supports 0.9.15 (the harness uses 0.9.16). The docker image crashes on RTX 50xx (old Vulkan loader, and OpenGL segfaults). |
| Autoware + bridge | docker `ghcr.io/autowarefoundation/autoware:universe-cuda` (0.50.0, CUDA 12.8, TRT 10.8), `run_autoware.sh`, GPU 1 | Ubuntu 22.04 / Humble / py3.10. The host is Mint 22 (24.04). The non-CUDA `universe` image cannot run perception. |
| Models | `~/autoware_data/ml_models/lidar_centerpoint` = HF `AutowareFoundation/lidar_centerpoint` **v3.0** (flat layout this image expects; v4.1 uses subfolders and fails). Launched with `data_path:=/autoware_data/ml_models`. | TensorRT engines are built on first run (a few minutes) and cached next to the models. |
| CARLA ROS 2 wheel | `~/autoware_data/wheels` (`gezp/carla_ros` 0.9.15 cp310) | Needed by the bridge. |
| Maps | `~/autoware_data/maps/TownXX/` from `carla-simulator/autoware-contents` (y-inverted) | Only Town01 is installed so far. |

Run: `./run_carla_native.sh` in one terminal, then `./run_autoware.sh Town01 [ros2 launch args]`. For the full check, run `./bridge_test.sh`.

## Running harness scenarios on Autoware (scenario injection)
`./run_session.sh OUTDIR SPEC...` runs (one town per session) unmodified harness `ScenarioSpec` against Autoware (real perception) on a fresh CARLA 0.9.15:

1. `dump_spawn_points.sh` (once) caches the 0.9.15 spawn-point table per town (`~/autoware_data/spawn_points_0.9.15.json`). Specs reference spawn points by index; Town01 indices were checked against the specs' route polylines (<= 1 m).
2. The bridge is launched with `carla_map:=<town>` and `spawn_point:=<ego spawn>`; it owns the ego and the clock.
3. `scenario_runner.py` runs inside the container as a **second CARLA client**: it finds the ego, then uses the harness's own `build_scenario_controller`, `CollisionOracle` and `SafetyOracle`, and the Autoware AD API (initialise, route to the goal spawn point, engage). Output is a harness `RunResult`-shaped JSON.
4. **Budget is in simulated time.** Specs count ticks at 0.1 s (`harness.fixed_delta_seconds`); the bridge ticks at 0.05 s. `max_ticks * 0.1` sim-seconds is used, and controllers get the equivalent 0.1 s tick index.

First result, `town01_spawn55_goal154_lead_braking`: adversary triggered at harness tick 104, min TTC 3.29 s, no collision, goal not reached in 50 sim-s (ego waits behind the stopped lead: 661/1000 frames stuck). Goal completion is therefore *not* a success criterion for this scenario family; use safety outcomes.

**Maps (CARLA 0.9.15 here).** Plain `Town03` segfaults on load every time (Shipping build, no log). The `_Opt` variants the specs use are the ones to use; spawn tables exist for Town01, Town03_Opt, Town05_Opt, Town10HD_Opt (all lead-braking specs match the table to <= 1 m). `_Opt` towns use the base Autoware map directory (`Town05_Opt` -> `maps/Town05`).

**Adversary-type check (2026-10-18, one spec each):**

| Spec | Outcome |
|---|---|
| Town01 `threshold_crossing_adversary` spawn0->82 | **Collision** with the injected pedestrian at 17.7 sim-s (adversary triggered at 19.8 m) |
| Town10HD_Opt `threshold_crossing_adversary` spawn0->44 | Triggered at 7.9 m, no collision, 50 sim-s |
| Town05_Opt `threshold_crossing_adversary` spawn0->124 | **Cannot start**: `set_route_points` -> "planned route is empty" (see below) |
| Town03_Opt `lead_vehicle_braking` spawn121->2 | **CARLA segfault** during the run (rc 139), cause unknown (dump of the same map worked) |

**Town05 routing:** `route_probe.py` showed that on Town05 a goal 30 m ahead of the ego routes, but 80 m ahead does not, and neither does the spec goal in either orientation. So the provided Town05 lanelet2 map is disconnected near spawn 0 (a map-data problem, not a goal-pose or yaw problem; the same conversion works on Town01). Needs a different map source, or per-spec routability filtering, before Town05 can be used.

**Caveat on the Town01 collision:** only CenterPoint *tiny* is in use; a pedestrian miss may be a perception-model weakness. Re-check with the base/sigma model before reporting it as an Autoware planning failure.

## Known limits
- **Traffic-light recognition is not working** (models for `traffic_light_classifier`, `traffic_light_fine_detector` and `tensorrt_yolox` are not downloaded, so those nodes fail to init). Lidar driving is unaffected. Needed before any signalised-junction scenario.
- **Only the CenterPoint *tiny* engine was built and used** in the default launch. Check accuracy before relying on it for actor-sensitive results.
- **Never run two smoke tests at once**: duplicate `/bridge_smoke_test` nodes trip Autoware's duplicated-node check and make autonomous mode unavailable.
- **CARLA 0.9.15 segfaults on a second bridge session** against the same server. Start a fresh server for every run (`bridge_test.sh` does).
- **Version split:** Autoware runs on CARLA 0.9.15 and the other four ADSs run on 0.9.16. Fairness needs the same maps, scenarios and budgets across both, so the scenario format has to be portable.
- **Scenario control:** the bridge spawns its own ego (`spawn_point` param). Upstream PR #13410 ("attach to an ego another client spawned") would let the harness own the scenario. It is still open and is not in this image.
- Maps installed in `~/autoware_data/maps`: Town01, Town03, Town05, Town10HD. Only Town01 has been driven.
