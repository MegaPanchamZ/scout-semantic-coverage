#!/usr/bin/env python3
"""BRIDGE-ONLY stub: publishes an EMPTY world (no objects, empty obstacle cloud, free occupancy grid) so Autoware's
planning/control diagnostics go green when launched with perception:=false. Use ONLY to validate the CARLA<->Autoware
actuation path. It is NOT a valid configuration for ADS-testing experiments (Autoware would be blind to actors);
those need the real perception stack (universe-cuda image + models)."""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from autoware_perception_msgs.msg import PredictedObjects, TrafficLightGroupArray
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid


class Stub(Node):
    def __init__(self):
        super().__init__("stub_perception")
        q = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.obj = self.create_publisher(PredictedObjects, "/perception/object_recognition/objects", q)
        self.pc = self.create_publisher(PointCloud2, "/perception/obstacle_segmentation/pointcloud", q)
        self.og = self.create_publisher(OccupancyGrid, "/perception/occupancy_grid_map/map", q)
        self.tl = self.create_publisher(TrafficLightGroupArray, "/perception/traffic_light_recognition/traffic_signals", q)
        self.create_timer(0.1, self.tick)

    def tick(self):
        now = self.get_clock().now().to_msg()
        o = PredictedObjects(); o.header.stamp = now; o.header.frame_id = "map"; self.obj.publish(o)
        p = PointCloud2(); p.header.stamp = now; p.header.frame_id = "base_link"
        p.height = 1; p.width = 0; p.is_dense = True; p.point_step = 16
        p.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
        p.fields.append(PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1))
        self.pc.publish(p)
        g = OccupancyGrid(); g.header.stamp = now; g.header.frame_id = "base_link"
        g.info.resolution = 0.5; g.info.width = 200; g.info.height = 200
        g.info.origin.position.x = -50.0; g.info.origin.position.y = -50.0; g.info.origin.orientation.w = 1.0
        g.data = [0] * (200 * 200); self.og.publish(g)
        t = TrafficLightGroupArray(); t.stamp = now; self.tl.publish(t)


def main():
    rclpy.init(); rclpy.spin(Stub())


if __name__ == "__main__":
    main()
