#!/usr/bin/env python3
"""Diagnose 'The planned route is empty': try several goal poses against the live Autoware stack.
Runs inside the container like scenario_runner.py (args: TABLE.json OUTDIR SPEC.json; use RUNNER=route_probe.py run_session.sh)."""
import json, math, sys, time
sys.path[:0] = ["/repo", "/tmp"]
import rclpy
from geometry_msgs.msg import Pose
from autoware_adapi_v1_msgs.msg import LocalizationInitializationState
from autoware_adapi_v1_msgs.srv import InitializeLocalization, SetRoutePoints
from autoware_adapi_v1_msgs.srv import ClearRoute
from smoke_test import T

spec = json.load(open(sys.argv[3])); table = json.load(open(sys.argv[1]))[spec["town"]]
rclpy.init(); n = T()
assert n.spin_until(lambda: n.odom is not None, 180)
if n.loc_state != LocalizationInitializationState.INITIALIZED:
    n.call(InitializeLocalization, "/api/localization/initialize", InitializeLocalization.Request())
n.spin_until(lambda: n.loc_state == LocalizationInitializationState.INITIALIZED, 60)
time.sleep(20)  # let map/planning settle


def pose(x, y, yaw):
    p = Pose(); p.position.x, p.position.y = x, y
    p.orientation.z, p.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return p


def try_goal(label, p):
    req = SetRoutePoints.Request(); req.header.frame_id = "map"; req.goal = p
    r = n.call(SetRoutePoints, "/api/routing/set_route_points", req)
    out = {"label": label, "ok": bool(r.status.success), "msg": r.status.message}
    print(json.dumps(out), flush=True)
    if r.status.success:
        try: n.call(ClearRoute, "/api/routing/clear_route", ClearRoute.Request()); time.sleep(3)
        except Exception: pass
    return out


res = []
o = n.odom.pose.pose; yaw0 = math.atan2(2 * (o.orientation.w * o.orientation.z), 1 - 2 * o.orientation.z ** 2)
for d in (30, 80):  # control: straight ahead of the ego must be routable
    res.append(try_goal(f"ahead_{d}m", pose(o.position.x + d * math.cos(yaw0), o.position.y + d * math.sin(yaw0), yaw0)))
g = table[spec["goal_spawn_index"]]; gx, gy, gyaw = g["x"], -g["y"], -math.radians(g["yaw"])
for lbl, yw in (("goal_-yaw", gyaw), ("goal_-yaw+pi", gyaw + math.pi), ("goal_+yaw", math.radians(g["yaw"]))):
    res.append(try_goal(lbl, pose(gx, gy, yw)))
json.dump(res, open(sys.argv[3], "w"), indent=1)
