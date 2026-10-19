#!/usr/bin/env python3
"""Run harness ScenarioSpecs against Autoware on CARLA 0.9.15, MANY specs per Autoware session. Runs INSIDE the container.

Startup (CARLA + Autoware + TensorRT) costs minutes, so one session serves every spec on one town: between episodes
the ego is stopped, re-spawned (teleported) at the next spec's ego spawn point and Autoware is re-localised and
re-routed. Episodes are independent apart from that reset (Autoware's perception/tracker state is warm).

The bridge (autoware_carla_interface) owns the ego and the simulation clock. This script is a *second* CARLA client:
it spawns the scenario's adversaries with the unmodified harness controllers (research.harness.scenario_runtime), watches
collisions/safety with the unmodified harness oracles, and drives Autoware through the AD API.

usage: scenario_runner.py TABLE.json OUTDIR SPEC1.json [SPEC2.json ...]      (all specs must be on the same CARLA town)
       scenario_runner.py --serve TABLE.json TOWN INBOX OUTDIR                  (persistent service, see serve())
Writes OUTDIR/<scenario_id>.json (harness RunResult shape + timing) and OUTDIR/_session.json.
"""
import json, math, os, sys, time, traceback
from pathlib import Path
sys.path[:0] = ["/repo", "/tmp"]

import carla
import rclpy
from geometry_msgs.msg import Pose
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from autoware_adapi_v1_msgs.msg import LocalizationInitializationState, OperationModeState, RouteState
from autoware_adapi_v1_msgs.srv import (ChangeOperationMode, ClearRoute, InitializeLocalization, SetRoutePoints)

from research.harness.models import RunResult, ScenarioSpec
from research.harness.oracles import CollisionOracle, SafetyOracle
from research.harness.scenario_runtime import build_scenario_controller
from smoke_test import T  # Node with AD-API helpers

GOAL_TOL_M = float(os.environ.get("GOAL_TOL_M", 3.0))
ENGAGE_WAIT_S = float(os.environ.get("ENGAGE_WAIT_S", 180))
# Harness specs count ticks at harness.fixed_delta_seconds (0.1 s); the bridge ticks at 0.05 s. Budgets and the tick index
# given to controllers are therefore expressed in SIMULATED time so every stack gets the same sim-seconds.
SPEC_DT = float(os.environ.get("SPEC_DT", 0.1))
SEMANTIC = os.environ.get("SEMANTIC", "1") == "1"   # run the harness SemanticObserver (simulator-state STSG stream) per episode


def carla_to_autoware(x, y, yaw_deg):
    """CARLA is left-handed (y south); the y-inverted Autoware maps use y -> -y, yaw -> -yaw."""
    return x, -y, -math.radians(yaw_deg)


def find_ego(world, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for a in world.get_actors().filter("vehicle.*"):
            if a.attributes.get("role_name") == "ego_vehicle":
                return a
        time.sleep(1)
    raise RuntimeError("ego_vehicle not found (is the bridge running?)")


def tf_from_row(r, dz=0.3):
    return carla.Transform(carla.Location(r["x"], r["y"], r["z"] + dz), carla.Rotation(r["pitch"], r["yaw"], r["roll"]))


def telemetry_for(ego, goal_loc):
    """The telemetry dict the harness observers expect, built from simulator state (no agent internals available)."""
    v = ego.get_velocity(); speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
    c = ego.get_control()
    return {"speed_mps": speed, "speed_kph": speed * 3.6,
            "distance_to_goal_m": float(ego.get_location().distance(goal_loc)),
            "control": {"throttle": c.throttle, "brake": c.brake, "steer": c.steer},
            "agent_step": {"route_progress": {}}}


def odom_err(n, ego):
    el = ego.get_location(); od = n.odom.pose.pose.position
    return math.hypot(el.x - od.x, -el.y - od.y)


def autoware_stop(n):
    """Leave autonomous mode and wait for STOP. The route is deliberately NOT cleared: clearing it and setting a new one
    crashed Autoware's behavior_planning_container ("failed to add guard condition to wait set"), after which no
    trajectory is ever produced. The next episode re-routes with change_route_points instead."""
    r = n.call(ChangeOperationMode, "/api/operation_mode/change_to_stop", ChangeOperationMode.Request(), timeout=15)
    print(f"[reset] change_to_stop success={r.status.success} msg={r.status.message!r}", flush=True)
    if not n.spin_until(lambda: n.op_mode == OperationModeState.STOP, 30):
        raise RuntimeError(f"Autoware did not enter STOP mode (mode={n.op_mode})")


def relocalise(n, ego, world, tf):
    """Teleport the ego to tf and bring Autoware's localisation to the new pose (init by GNSS)."""
    ego.set_target_velocity(carla.Vector3D(0, 0, 0)); ego.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
    ego.set_transform(tf)
    for _ in range(40):                        # let the bridge publish fresh sensors at the new pose
        world.wait_for_tick(30.0)
    n.loc_state = None
    n.call(InitializeLocalization, "/api/localization/initialize", InitializeLocalization.Request())
    n.spin_until(lambda: n.loc_state == LocalizationInitializationState.INITIALIZED, 60)
    ok = n.spin_until(lambda: odom_err(n, ego) < 4.0, 60)
    if not ok:
        raise RuntimeError(f"localisation did not converge on the teleported ego (err {odom_err(n, ego):.1f} m)")


def engage_to(n, goal_sp):
    gx, gy, gyaw = carla_to_autoware(goal_sp["x"], goal_sp["y"], goal_sp["yaw"])
    req = SetRoutePoints.Request(); req.header.frame_id = "map"; req.goal = Pose()
    req.goal.position.x, req.goal.position.y = gx, gy
    req.goal.orientation.z, req.goal.orientation.w = math.sin(gyaw / 2), math.cos(gyaw / 2)
    first_svc = "/api/routing/change_route_points" if n.route_state in (RouteState.SET, getattr(RouteState, "ARRIVED", 3)) else "/api/routing/set_route_points"
    alt_svc = "/api/routing/set_route_points" if first_svc.endswith("change_route_points") else "/api/routing/change_route_points"
    r = n.call(SetRoutePoints, first_svc, req)
    print(f"[route] {first_svc.rsplit('/', 1)[1]} success={r.status.success} msg={r.status.message!r}", flush=True)
    if not r.status.success:
        r = n.call(SetRoutePoints, alt_svc, req)
        print(f"[route-fallback] {alt_svc.rsplit('/', 1)[1]} success={r.status.success} msg={r.status.message!r}", flush=True)
    if not r.status.success:
        raise RuntimeError(f"route request failed: {r.status.message}")
    if not n.spin_until(lambda: n.route_state == RouteState.SET, 30):
        raise RuntimeError("route not set")
    t_e = time.time()
    while True:
        r = n.call(ChangeOperationMode, "/api/operation_mode/change_to_autonomous", ChangeOperationMode.Request())
        if r.status.success:
            return time.time() - t_e
        if time.time() - t_e > ENGAGE_WAIT_S:
            raise RuntimeError(f"could not engage: {r.status.message}")
        n.spin_until(lambda: False, 2)


def run_episode(n, client, world, ego, spec, table, first, outdir):
    notes = []
    sp = table[spec.town]
    goal_sp = sp[spec.goal_spawn_index]
    goal_loc = carla.Location(goal_sp["x"], goal_sp["y"], goal_sp["z"])
    if spec.weather_preset and hasattr(carla.WeatherParameters, spec.weather_preset):
        world.set_weather(getattr(carla.WeatherParameters, spec.weather_preset))
    if not first:
        autoware_stop(n)
        relocalise(n, ego, world, tf_from_row(sp[spec.ego_spawn_index]))
    elif n.loc_state != LocalizationInitializationState.INITIALIZED:
        n.call(InitializeLocalization, "/api/localization/initialize", InitializeLocalization.Request())
        assert n.spin_until(lambda: n.loc_state == LocalizationInitializationState.INITIALIZED, 60), "localisation not initialised"
    notes.append(f"CARLA->Autoware frame check: |ego - odom| = {odom_err(n, ego):.2f} m")
    notes.append(f"engaged after {engage_to(n, goal_sp):.1f}s")

    controller = build_scenario_controller(spec)
    context = {"world": world, "ego_vehicle": ego, "carla": carla, "scenario_notes": notes}
    controller.setup(context)
    collisions, safety = CollisionOracle(), SafetyOracle()
    collisions.injected_actor_ids = {a.id for a in controller.spawned_actors}
    collisions.attach(world, ego)

    observer = None
    if SEMANTIC:
        try:
            from research.harness.observers.semantic import SemanticObserver, SemanticObserverConfig
            sem_dir = Path(outdir) / "semantic" / spec.scenario_id
            observer = SemanticObserver(SemanticObserverConfig(trace_output_dir=sem_dir / "trace",
                                                               stream_output_dir=sem_dir / "stream"))
            context.update(scenario=spec, collision_oracle=collisions, telemetry=telemetry_for(ego, goal_loc))
            observer.on_run_start(spec, context)
        except Exception as e:
            observer = None
            notes.append(f"SemanticObserver disabled: {type(e).__name__}: {e}")

    reached = False; terminated = False; ticks = 0; last_h = 0
    min_gap = float("inf"); first_violation_s = None   # method-independent outcome metrics (used by RQ3 for every search method)
    budget_s = spec.max_ticks * SPEC_DT
    t0 = world.wait_for_tick(30.0).timestamp.elapsed_seconds
    elapsed = 0.0
    try:
        while elapsed < budget_s:
            snap = world.wait_for_tick(30.0)
            ticks += 1
            elapsed = snap.timestamp.elapsed_seconds - t0
            h = int(elapsed / SPEC_DT) + 1                     # harness tick index (0.1 s), 1-based here
            controller.on_tick(h - 1, context)
            if observer is not None and h != last_h:           # observers run once per harness tick, as in the harness
                last_h = h
                try:
                    context["telemetry"] = telemetry_for(ego, goal_loc)
                    observer.on_tick(h - 1, context)       # harness passes 0-based ticks
                except Exception as e:
                    notes.append(f"SemanticObserver error at tick {h - 1}: {type(e).__name__}: {e}"); observer = None
            safety.tick(world, ego)
            for a in controller.spawned_actors:                # closest approach to any injected actor (AV-Fuzzer's fitness signal)
                try: min_gap = min(min_gap, ego.get_location().distance(a.get_location()))
                except RuntimeError: pass
            rclpy.spin_once(n, timeout_sec=0)
            if collisions.events:
                first_violation_s = round(elapsed, 2)
                terminated = True
                notes.append(f"Terminated on first collision with {collisions.events[0].actor_type}")
                break
            if ego.get_location().distance(goal_loc) < GOAL_TOL_M:
                reached = True
                break
    finally:
        try:
            controller.teardown(context)
        finally:
            for a in controller.spawned_actors:
                try: a.destroy()
                except RuntimeError: pass
            collisions.destroy()

    res = RunResult(
        scenario_id=spec.scenario_id, town=spec.town, agent_kind="autoware", succeeded=True, dry_run=False,
        ticks_executed=ticks, reached_goal=reached, terminated_by_collision=terminated,
        collision_count=len(collisions.events), collisions=list(collisions.events), safety_metrics=safety.metrics,
        notes=notes, metadata={"carla_version": client.get_server_version(), "sim_seconds": round(elapsed, 2),
                               "budget_sim_seconds": budget_s, "bridge_dt": world.get_settings().fixed_delta_seconds,
                               "spec_dt": SPEC_DT, "goal_spawn_index": spec.goal_spawn_index,
                               "min_gap_m": None if min_gap == float("inf") else round(min_gap, 3),
                               "first_violation_sim_s": first_violation_s,
                               "controller": spec.controller})
    if observer is not None:
        try: observer.on_run_end(res, context)             # writes trace/stream dumps, fills res.metadata["semantic"]
        except Exception as e: res.notes.append(f"SemanticObserver on_run_end failed: {type(e).__name__}: {e}")
    return res


def _setup_session(table_path, first_town):
    table = json.load(open(table_path))
    client = carla.Client("localhost", 2000); client.set_timeout(60)
    world = client.get_world()
    cur = world.get_map().name.split("/")[-1]
    if cur != first_town:
        raise RuntimeError(f"CARLA map is {cur}, want {first_town} (launch bridge with carla_map:=...)")
    ego = find_ego(world)
    rclpy.init(); n = T()
    n.op_mode = None
    n.create_subscription(OperationModeState, "/api/operation_mode/state", lambda m: setattr(n, "op_mode", m.mode),
                          QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    assert n.spin_until(lambda: n.odom is not None, 180), "no /localization/kinematic_state"
    return table, client, world, ego, n


def _run_one(n, client, world, ego, spec, table, first, outdir):
    """One episode with one retry; writes OUTDIR/<scenario_id>.json (or .error.json) and returns the status record."""
    t_start = time.time(); rec = {"scenario_id": spec.scenario_id}
    try:
        for attempt in (1, 2):             # one retry: transient reset/routing failures (~1 in 25 episodes) are infra, not results
            try:
                res = run_episode(n, client, world, ego, spec, table, first=(first and attempt == 1), outdir=outdir)
                break
            except Exception as e1:
                if attempt == 2:
                    raise
                print(f"[retry] {spec.scenario_id}: {type(e1).__name__}: {e1}", flush=True)
                rec["retried"] = f"{type(e1).__name__}: {e1}"
                try: autoware_stop(n)
                except Exception: pass
        d = res.to_dict(); d["metadata"]["wall_seconds"] = round(time.time() - t_start, 1)
        d["metadata"]["retried"] = rec.get("retried")
        json.dump(d, open(os.path.join(outdir, f"{spec.scenario_id}.json"), "w"), indent=1, default=str)
        rec.update(ok=True, reached_goal=res.reached_goal, collision=res.terminated_by_collision,
                   sim_s=d["metadata"]["sim_seconds"], wall_s=d["metadata"]["wall_seconds"])
    except Exception as e:  # keep the session alive; record why this one failed
        traceback.print_exc()
        rec.update(ok=False, error=f"{type(e).__name__}: {e}", wall_s=round(time.time() - t_start, 1))
        json.dump(rec, open(os.path.join(outdir, f"{spec.scenario_id}.error.json"), "w"), indent=1)
    print(json.dumps(rec), flush=True)
    return rec


def serve(table_path, town, inbox, outdir):
    """Persistent session: evaluate every spec dropped into INBOX (any search process can drive it); stop on INBOX/STOP.
    A request is INBOX/<id>.json (a ScenarioSpec dict, scenario_id should equal <id>); result is OUTDIR/<id>.json."""
    os.makedirs(inbox, exist_ok=True); os.makedirs(outdir, exist_ok=True)
    table, client, world, ego, n = _setup_session(table_path, town)
    first = True
    print(f"[serve] ready on {town}, watching {inbox}", flush=True)
    open(os.path.join(outdir, "_READY"), "w").write(str(time.time()))
    while not os.path.exists(os.path.join(inbox, "STOP")):
        reqs = sorted(p for p in os.listdir(inbox) if p.endswith(".json"))
        if not reqs:
            rclpy.spin_once(n, timeout_sec=0.2); continue
        path = os.path.join(inbox, reqs[0])
        try:
            spec = ScenarioSpec.from_dict(json.load(open(path)))
        except Exception as e:
            os.rename(path, path + ".bad"); print(f"[serve] bad request {reqs[0]}: {e}", flush=True); continue
        os.rename(path, path + ".taken")
        if spec.town != town:
            json.dump({"scenario_id": spec.scenario_id, "ok": False, "error": f"service is on {town}, spec wants {spec.town}"},
                      open(os.path.join(outdir, f"{spec.scenario_id}.error.json"), "w")); continue
        _run_one(n, client, world, ego, spec, table, first, outdir)
        first = False
    return 0


def main(table_path, outdir, spec_paths):
    os.makedirs(outdir, exist_ok=True)
    specs = [ScenarioSpec.from_dict(json.load(open(p))) for p in spec_paths]
    towns = {s.town for s in specs}
    assert len(towns) == 1, f"one town per session, got {towns}"
    table = json.load(open(table_path))

    client = carla.Client("localhost", 2000); client.set_timeout(60)
    world = client.get_world()
    cur = world.get_map().name.split("/")[-1]
    if cur != specs[0].town:
        raise RuntimeError(f"CARLA map is {cur}, specs want {specs[0].town} (launch bridge with carla_map:=...)")
    ego = find_ego(world)
    rclpy.init(); n = T()
    n.op_mode = None
    n.create_subscription(OperationModeState, "/api/operation_mode/state", lambda m: setattr(n, "op_mode", m.mode),
                          QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    assert n.spin_until(lambda: n.odom is not None, 180), "no /localization/kinematic_state"

    session = []
    for i, spec in enumerate(specs):
        t_start = time.time(); rec = {"scenario_id": spec.scenario_id}
        try:
            for attempt in (1, 2):             # one retry: transient reset/routing failures (~1 in 25 episodes) are infra, not results
                try:
                    res = run_episode(n, client, world, ego, spec, table, first=(i == 0 and attempt == 1), outdir=outdir)
                    break
                except Exception as e1:
                    if attempt == 2:
                        raise
                    print(f"[retry] {spec.scenario_id}: {type(e1).__name__}: {e1}", flush=True)
                    rec["retried"] = f"{type(e1).__name__}: {e1}"
                    try: autoware_stop(n)
                    except Exception: pass
            d = res.to_dict(); d["metadata"]["wall_seconds"] = round(time.time() - t_start, 1)
            d["metadata"]["retried"] = rec.get("retried")
            json.dump(d, open(os.path.join(outdir, f"{spec.scenario_id}.json"), "w"), indent=1, default=str)
            rec.update(ok=True, reached_goal=res.reached_goal, collision=res.terminated_by_collision,
                       sim_s=d["metadata"]["sim_seconds"], wall_s=d["metadata"]["wall_seconds"])
        except Exception as e:  # keep the session alive for the remaining specs; record why this one failed
            traceback.print_exc()
            rec.update(ok=False, error=f"{type(e).__name__}: {e}", wall_s=round(time.time() - t_start, 1))
            json.dump(rec, open(os.path.join(outdir, f"{spec.scenario_id}.error.json"), "w"), indent=1)
        print(json.dumps(rec), flush=True)
        session.append(rec)
    json.dump(session, open(os.path.join(outdir, "_session.json"), "w"), indent=1)
    return 0 if all(r["ok"] for r in session) else 1


if __name__ == "__main__":
    if sys.argv[1] == "--serve":      # --serve TABLE.json TOWN INBOX OUTDIR
        sys.exit(serve(*sys.argv[2:6]))
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
