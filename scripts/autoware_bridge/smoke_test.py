#!/usr/bin/env python3
"""Smoke test for the CARLA<->Autoware bridge. Runs INSIDE the autoware container (needs rclpy).

Checks, in order: (1) bridge publishes /localization/kinematic_state, (2) localization is initialised,
(3) a route can be set to a goal AHEAD_M metres in front of the ego, (4) autonomous mode engages,
(5) the CARLA ego actually moves (distance, peak speed) and approaches the goal.
Exit 0 only if all pass. Prints one JSON line (machine-readable result) at the end.

usage: docker cp smoke_test.py autoware:/tmp/ && docker exec autoware bash -c \
  'source /opt/ros/humble/setup.bash; source /opt/autoware/local_setup.bash; \
   ROS_DOMAIN_ID=7 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp python3 /tmp/smoke_test.py'
"""
import json, math, os, sys, time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from autoware_adapi_v1_msgs.srv import ChangeOperationMode, SetRoutePoints, InitializeLocalization
from autoware_adapi_v1_msgs.msg import LocalizationInitializationState, RouteState

AHEAD_M = float(os.environ.get("AHEAD_M", 80))
RUN_S = float(os.environ.get("RUN_S", 90))


class T(Node):
    def __init__(self):
        super().__init__("bridge_smoke_test")
        self.odom = None
        self.loc_state = None
        self.route_state = None
        self.create_subscription(Odometry, "/localization/kinematic_state", self._o, 10)
        self.create_subscription(LocalizationInitializationState, "/api/localization/initialization_state", self._l, 1)
        self.create_subscription(RouteState, "/api/routing/state", self._r, 1)

    def _o(self, m): self.odom = m
    def _l(self, m): self.loc_state = m.state
    def _r(self, m): self.route_state = m.state

    def spin_until(self, cond, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if cond():
                return True
        return False

    def call(self, srv_type, name, req, timeout=15):
        cli = self.create_client(srv_type, name)
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service {name} unavailable")
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        if fut.result() is None:
            raise RuntimeError(f"service {name} timed out")
        return fut.result()


def pos(o): return o.pose.pose.position.x, o.pose.pose.position.y


def yaw(o):
    q = o.pose.pose.orientation
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def main():
    rclpy.init()
    n = T()
    res = {"ahead_m": AHEAD_M}
    ok = n.spin_until(lambda: n.odom is not None, float(os.environ.get("ODOM_WAIT_S", 180)))
    res["kinematic_state"] = ok
    if not ok:
        print(json.dumps(res)); return 1

    if n.loc_state != LocalizationInitializationState.INITIALIZED:
        r = n.call(InitializeLocalization, "/api/localization/initialize", InitializeLocalization.Request())
        res["init_call"] = bool(r.status.success)
    res["localized"] = n.spin_until(lambda: n.loc_state == LocalizationInitializationState.INITIALIZED, 30)

    start = pos(n.odom); yw = yaw(n.odom)
    goal = PoseStamped(); goal.header.frame_id = "map"
    goal.pose.position.x = start[0] + AHEAD_M * math.cos(yw)
    goal.pose.position.y = start[1] + AHEAD_M * math.sin(yw)
    goal.pose.orientation = n.odom.pose.pose.orientation
    req = SetRoutePoints.Request(); req.header.frame_id = "map"
    req.goal = goal.pose
    r = n.call(SetRoutePoints, "/api/routing/set_route_points", req)
    res["route_set"] = bool(r.status.success); res["route_msg"] = r.status.message
    res["route_ready"] = n.spin_until(lambda: n.route_state == RouteState.SET, 20)

    # Autonomous mode only becomes available once planning has produced a trajectory for the route: retry.
    t_e = time.time()
    while True:
        r = n.call(ChangeOperationMode, "/api/operation_mode/change_to_autonomous", ChangeOperationMode.Request())
        if r.status.success or time.time() - t_e > float(os.environ.get("ENGAGE_WAIT_S", 90)):
            break
        n.spin_until(lambda: False, 2)
    res["engaged"] = bool(r.status.success); res["engage_msg"] = r.status.message
    res["engage_wait_s"] = round(time.time() - t_e, 1)

    vmax, t0, last = 0.0, time.time(), start
    dist = 0.0
    while time.time() - t0 < RUN_S:
        rclpy.spin_once(n, timeout_sec=0.1)
        p = pos(n.odom)
        dist += math.dist(p, last); last = p
        vmax = max(vmax, n.odom.twist.twist.linear.x)
        if math.dist(p, (goal.pose.position.x, goal.pose.position.y)) < 3.0:
            break
    res.update(distance_m=round(dist, 2), peak_speed_mps=round(vmax, 2),
               goal_dist_m=round(math.dist(last, (goal.pose.position.x, goal.pose.position.y)), 2),
               duration_s=round(time.time() - t0, 1))
    res["moved"] = dist > 5.0 and vmax > 0.5
    res["reached_goal"] = res["goal_dist_m"] < 3.0
    print(json.dumps(res))
    return 0 if all(res.get(k) for k in ("kinematic_state", "localized", "route_set", "engaged", "moved")) else 1


if __name__ == "__main__":
    sys.exit(main())
