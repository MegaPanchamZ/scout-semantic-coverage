#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CARLA-free control-clip renderer (EXP-023).

Compiles a DSL scenario variant + an OpenDRIVE map into Cosmos-transfer controls:
  seg.mp4   Cityscapes-palette semantic view from the ego camera
  edge.mp4  edge control (class boundaries + lane markings + crosswalk)
  depth.mp4 linear depth (grayscale)
  meta.json ego poses, actor tracks, bindings, camera intrinsics

No 3D engine: lane geometry is sampled from OpenDRIVE planView + lane widths and
projected through a pinhole camera. Actors are projected 3D boxes.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import imageio.v3 as iio
import numpy as np

CITYSCAPES = {
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "building": (70, 70, 70),
    "vegetation": (107, 142, 35),
    "terrain": (152, 251, 152),
    "sky": (70, 130, 180),
    "person": (220, 20, 60),
    "vehicle": (0, 0, 142),
    "pole": (153, 153, 153),
    "traffic light": (250, 170, 30),
}
LANE_TYPE_CLASS = {
    "driving": "road",
    "sidewalk": "sidewalk",
    "biking": "sidewalk",
    "parking": "road",
    "shoulder": "road",
    "median": "road",
    "restricted": "road",
    "none": "road",
}
VEHICLE_DIMS = (4.6, 1.9, 1.55)
PEDESTRIAN_DIMS = (0.55, 0.55, 1.75)


@dataclass
class Geometry:
    s0: float
    x: float
    y: float
    hdg: float
    length: float
    kind: str = "line"
    curvature: float = 0.0
    curv_start: float = 0.0
    curv_end: float = 0.0


@dataclass
class Lane:
    lane_id: int
    lane_type: str
    width: float


@dataclass
class Road:
    road_id: int
    junction: int
    length: float
    geoms: list[Geometry]
    lanes: list[Lane]
    signals: list[tuple[str, float, float]] = field(default_factory=list)

    def sample_centerline(self, ds: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        pts: list[tuple[float, float]] = []
        hdgs: list[float] = []
        for geom in self.geoms:
            n = max(1, int(math.ceil(geom.length / ds)))
            steps = np.linspace(0.0, geom.length, n + 1)
            if geom.kind == "arc" and abs(geom.curvature) > 1e-9:
                r = 1.0 / geom.curvature
                cx = geom.x - r * math.sin(geom.hdg)
                cy = geom.y + r * math.cos(geom.hdg)
                for d in steps:
                    phi = geom.hdg + geom.curvature * d
                    pts.append((cx + r * math.sin(phi), cy - r * math.cos(phi)))
                    hdgs.append(phi)
            elif geom.kind == "spiral" and abs(geom.curv_end - geom.curv_start) > 1e-9:
                n_sub = max(1, int(math.ceil(geom.length / 0.25)))
                x, y, h = geom.x, geom.y, geom.hdg
                sub = geom.length / n_sub
                for i in range(n_sub + 1):
                    if i:
                        frac = (i - 0.5) / n_sub
                        k = geom.curv_start + (geom.curv_end - geom.curv_start) * frac
                        x += sub * math.cos(h)
                        y += sub * math.sin(h)
                        h += k * sub
                    if i % max(1, int(round(0.25 / sub))) == 0 or i == n_sub:
                        pts.append((x, y))
                        hdgs.append(h)
            else:
                h = geom.hdg
                for d in steps:
                    pts.append((geom.x + d * math.cos(h), geom.y + d * math.sin(h)))
                    hdgs.append(h)
        return np.array(pts, dtype=np.float64), np.array(hdgs, dtype=np.float64)

    def lane_bands(self, ds: float = 1.0) -> dict[int, tuple[str, float, float, np.ndarray]]:
        """lane_id -> (class_name, t_inner, t_outer, polygon [N,2]).

        Uses the constant width of the first laneSection (CARLA maps keep widths
        constant per section; multi-section roads are truncated to the first for v0).
        """
        pts, hdgs = self.sample_centerline(ds)
        normals = np.stack([-np.sin(hdgs), np.cos(hdgs)], axis=1)
        out: dict[int, tuple[str, float, float, np.ndarray]] = {}
        t_pos = 0.0
        t_neg = 0.0
        for lane in sorted(self.lanes, key=lambda l: l.lane_id):
            if lane.lane_id <= 0:
                continue
            t_inner, t_outer = t_pos, t_pos + lane.width
            t_pos = t_outer
            out[lane.lane_id] = (LANE_TYPE_CLASS.get(lane.lane_type, "road"), t_inner, t_outer,
                                 (pts + normals * t_inner, pts + normals * t_outer))
        for lane in sorted(self.lanes, key=lambda l: l.lane_id, reverse=True):
            if lane.lane_id >= 0:
                continue
            t_inner, t_outer = t_neg, t_neg - lane.width
            t_neg = t_outer
            out[lane.lane_id] = (LANE_TYPE_CLASS.get(lane.lane_type, "road"), t_inner, t_outer,
                                 (pts + normals * t_inner, pts + normals * t_outer))
        return out


def parse_opendrive(path: Path) -> dict[int, Road]:
    root = ET.parse(path).getroot()
    roads: dict[int, Road] = {}
    for road_el in root.findall("road"):
        geoms: list[Geometry] = []
        plan = road_el.find("planView")
        if plan is not None:
            for g in plan.findall("geometry"):
                kind, curvature, cs, ce = "line", 0.0, 0.0, 0.0
                for child in g:
                    kind = child.tag
                    if child.tag == "arc":
                        curvature = float(child.get("curvature"))
                    elif child.tag == "spiral":
                        cs = float(child.get("curvStart"))
                        ce = float(child.get("curvEnd"))
                geoms.append(Geometry(float(g.get("s")), float(g.get("x")), float(g.get("y")), float(g.get("hdg")), float(g.get("length")), kind, curvature, cs, ce))
        lanes: list[Lane] = []
        lanes_el = road_el.find("lanes")
        if lanes_el is not None:
            sec = lanes_el.find("laneSection")
            if sec is not None:
                for side in ("left", "right"):
                    side_el = sec.find(side)
                    if side_el is None:
                        continue
                    for lane_el in side_el.findall("lane"):
                        width_el = lane_el.find("width")
                        width = float(width_el.get("a")) if width_el is not None else 3.5
                        lanes.append(Lane(int(lane_el.get("id")), lane_el.get("type", "driving"), width))
        signals: list[tuple[str, float, float]] = []
        sig_el = road_el.find("signals")
        if sig_el is not None:
            for sig in sig_el.findall("signal"):
                signals.append((sig.get("id", "0"), float(sig.get("s", "0")), float(sig.get("t", "0"))))
        roads[int(road_el.get("id"))] = Road(int(road_el.get("id")), int(road_el.get("junction", "-1")), float(road_el.get("length")), geoms, lanes, signals)
    return roads


@dataclass
class Camera:
    width: int
    height: int
    fov_deg: float = 70.0
    height_m: float = 1.42
    pitch_deg: float = -2.5

    def __post_init__(self) -> None:
        self.f = (self.width / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.horizon = self.cy + self.f * math.tan(math.radians(self.pitch_deg))

    def project(self, points: np.ndarray, pose: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        """points [N,3] world (z up) -> (uv [N,2], z_cam [N]). Forward is +x at hdg."""
        ego_x, ego_y, ego_hdg = pose
        theta = math.radians(-self.pitch_deg)  # positive = looking down
        c, s = math.cos(ego_hdg), math.sin(ego_hdg)
        rel = points - np.array([ego_x, ego_y, self.height_m])
        fwd = c * rel[:, 0] + s * rel[:, 1]
        right = s * rel[:, 0] - c * rel[:, 1]
        up = rel[:, 2]
        z_cam = fwd * math.cos(theta) - up * math.sin(theta)
        y_cam = -fwd * math.sin(theta) - up * math.cos(theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + self.f * right / z_cam
            v = self.cy + self.f * y_cam / z_cam
        return np.stack([u, v], axis=1), z_cam

def select_road(roads: dict[int, Road], min_length: float = 55.0) -> Road:
    best: tuple[float, Road] | None = None
    for road in roads.values():
        if road.length < min_length or road.junction >= 0:
            continue
        types = {lane.lane_type for lane in road.lanes}
        if "driving" not in types or "sidewalk" not in types:
            continue
        straight = sum(g.length for g in road.geoms if g.kind == "line") / road.length
        score = straight * road.length
        if best is None or score > best[0]:
            best = (score, road)
    if best is None:
        raise RuntimeError("no suitable road found")
    return best[1]


@dataclass
class Track:
    ego: list[tuple[float, float, float]]
    pedestrian: list[tuple[float, float, float]]
    crossing_s: float
    stop_t: float
    bindings: dict[str, str]
    lane_width: float
    s_track: list[float] = field(default_factory=list)


def build_track(road: Road, bindings: dict[str, str], frames: int, fps: int, speed: float = 8.0, ped_trigger: float = 18.0, ped_speed: float = 1.4) -> Track:
    pts, hdgs = road.sample_centerline(0.5)
    width = next((lane.width for lane in road.lanes if lane.lane_id == -1), 3.5)
    t_lane = -width / 2.0
    s_end = road.length - 8.0
    s_start = max(0.5, s_end - speed * (frames / fps) - 6.0)
    s_track = [s_start + speed * (i / fps) for i in range(frames)]
    crossing_s = s_end - 12.0

    def pose_at(s: float, t: float) -> tuple[float, float, float]:
        idx = int(np.clip(round(s / 0.5), 0, len(pts) - 1))
        x, y, h = pts[idx, 0], pts[idx, 1], hdgs[idx]
        return x - math.sin(h) * t, y + math.cos(h) * t, h

    ego = [pose_at(s, t_lane) for s in s_track]
    right_edge = -sum(lane.width for lane in road.lanes if lane.lane_id < 0)
    kerb_t = right_edge + 0.7
    far_kerb_t = 0.7
    crossing = bindings.get("pedestrian_1.behavior") == "crossing"
    trigger, walker = ped_trigger, ped_speed
    ped: list[tuple[float, float, float]] = []
    start_frame = None
    for i, s in enumerate(s_track):
        if crossing and (crossing_s - s) < trigger:
            if start_frame is None:
                start_frame = i
            t = max(far_kerb_t, kerb_t - walker * ((i - start_frame) / fps))
        else:
            t = kerb_t
        ped.append(pose_at(crossing_s, t))
    return Track(ego, ped, crossing_s, t_lane, bindings, width, s_track)


def render_frame(road: Road, bands: dict, track: Track, frame: int, cam: Camera, actors: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]) -> tuple[np.ndarray, np.ndarray]:
    """-> (seg BGR, depth uint8)."""
    seg = np.zeros((cam.height, cam.width, 3), np.uint8)
    seg[: int(max(0, min(cam.height, cam.horizon)))] = CITYSCAPES["sky"][::-1]
    seg[int(max(0, min(cam.height, cam.horizon))) :] = CITYSCAPES["terrain"][::-1]
    depth = np.full((cam.height, cam.width), 255, np.uint8)
    pose = track.ego[frame]

    s_now = track.s_track[frame] if track.s_track else 0.0

    def draw_strip(inner: np.ndarray, outer: np.ndarray, class_name: str, color_override=None) -> None:
        step = 3  # sample stride (~1.5 m)
        color = color_override or CITYSCAPES[class_name][::-1]
        for i in range(0, len(inner) - step, step):
            j = i + step
            quad = np.array([inner[i], outer[i], outer[j], inner[j]])
            zs = np.zeros((4, 1))
            uv, d = cam.project(np.concatenate([quad, zs], axis=1), pose)
            if np.any(d < 0.5) or not np.isfinite(uv).all():
                continue
            cv2.fillPoly(seg, [uv.astype(np.int32)], color, lineType=cv2.LINE_8)
            cv2.fillPoly(depth, [uv.astype(np.int32)], int(np.clip(np.median(d), 1, 120) / 120.0 * 255.0), lineType=cv2.LINE_8)

    ds_sample = 0.5
    near = int(max(0.0, s_now - 4.0) / ds_sample)
    far = int(min(road.length, s_now + 95.0) / ds_sample)
    for lane_id, (class_name, t_inner, t_outer, (inner_poly, outer_poly)) in bands.items():
        draw_strip(inner_poly[near:far], outer_poly[near:far], class_name)

    for kind, pose_actor, dims in actors:
        if kind == "pedestrian":
            draw_pedestrian(seg, depth, cam, pose, pose_actor, dims)
        else:
            draw_box3d(seg, depth, cam, pose, pose_actor[:2], pose_actor[2], dims, "vehicle", z_center=dims[2] / 2.0)
    return seg, depth


def box_corners(x: float, y: float, hdg: float, dims: tuple[float, float, float], z_center: float) -> np.ndarray:
    length, width, height = dims
    c, s = math.cos(hdg), math.sin(hdg)
    corners = []
    for along in (-length / 2.0, length / 2.0):
        for side in (-width / 2.0, width / 2.0):
            for z in (z_center - height / 2.0, z_center + height / 2.0):
                corners.append((x + c * along - s * side, y + s * along + c * side, z))
    return np.array(corners)


def draw_box3d(seg, depth, cam, pose, xy, hdg, dims, class_name, z_center):
    uv, d = cam.project(box_corners(xy[0], xy[1], hdg, dims, z_center), pose)
    if np.any(d < 0.3) or not np.isfinite(uv).all():
        return
    hull = cv2.convexHull(uv.astype(np.int32))
    cv2.fillPoly(seg, [hull], CITYSCAPES[class_name][::-1], lineType=cv2.LINE_8)
    cv2.fillPoly(depth, [hull], int(np.clip(np.median(d), 1, 120) / 120.0 * 255.0), lineType=cv2.LINE_8)


def draw_pedestrian(seg, depth, cam, pose, pose_actor, dims):
    x, y, h = pose_actor
    width = dims[1]
    color = CITYSCAPES["person"][::-1]
    draw_box3d(seg, depth, cam, pose, (x, y), h, (0.32, width, 0.82), "person", z_center=0.41)          # legs
    draw_box3d(seg, depth, cam, pose, (x, y), h, (0.30, width + 0.12, 0.62), "person", z_center=1.13)    # torso
    head_pt = np.array([[x, y, 1.62]])
    uv, d = cam.project(head_pt, pose)
    if np.isfinite(uv).all() and d[0] > 0.5:
        r = max(2, int(cam.f * 0.13 / float(d[0])))
        cv2.circle(seg, tuple(uv[0].astype(int)), r, color, -1, lineType=cv2.LINE_8)


def marking_layer(road: Road, bands: dict, track: Track, frame: int, cam: Camera, state: str) -> np.ndarray:
    """White ground markings (lane lines, crosswalk, stop line) for the edge control."""
    layer = np.zeros((cam.height, cam.width), np.uint8)
    pose = track.ego[frame]
    pts, hdgs = road.sample_centerline(0.5)

    def polyline_ground(t_values: list[float], s_range: tuple[float, float]) -> np.ndarray:
        idx0, idx1 = int(s_range[0] / 0.5), int(s_range[1] / 0.5)
        out = []
        for idx in range(max(0, idx0), min(len(pts), idx1)):
            x, y = pts[idx]
            h = hdgs[idx]
            for t in t_values:
                out.append((x - math.sin(h) * t, y + math.cos(h) * t))
        return np.array(out)

    s_now = track.s_track[frame] if track.s_track else 0.0
    s_lo, s_hi = max(0.0, s_now - 4.0), min(road.length, s_now + 95.0)

    def draw_lines(offsets: list[float], s_range: tuple[float, float], thickness: int = 3) -> None:
        idx0, idx1 = int(s_range[0] / 0.5), int(s_range[1] / 0.5)
        for idx in range(max(0, idx0), min(len(pts) - 1, idx1)):
            x0, y0, h0 = pts[idx][0], pts[idx][1], hdgs[idx]
            x1, y1, h1 = pts[idx + 1][0], pts[idx + 1][1], hdgs[idx + 1]
            seg_pts = []
            for (xx, yy, hh) in ((x0, y0, h0), (x1, y1, h1)):
                for t in offsets:
                    seg_pts.append((xx - math.sin(hh) * t, yy + math.cos(hh) * t, 0.0))
            arr = np.array(seg_pts)
            uv, d = cam.project(arr, pose)
            if np.any(d < 1.0) or not np.isfinite(uv).all():
                continue
            n_off = len(offsets)
            for k in range(n_off):
                a = tuple(uv[k].astype(int))
                b = tuple(uv[n_off + k].astype(int))
                cv2.line(layer, a, b, 255, thickness, lineType=cv2.LINE_8)

    offsets = sorted({round(v, 3) for _k, (_c, ti, to, _p) in bands.items() for v in (ti, to)})
    draw_lines(offsets, (s_lo, s_hi), thickness=2)

    # crosswalk zebra across the ego lane at crossing_s
    stripe_offsets = [round(v, 2) for v in np.arange(-0.6, -track.lane_width - 0.6, -1.1)]
    draw_lines(stripe_offsets, (max(0.0, track.crossing_s - 2.0), track.crossing_s + 2.0), thickness=6)
    # stop line just before the crosswalk
    draw_lines([round(v, 2) for v in np.arange(-0.5, -track.lane_width - 0.5, -0.6)], (track.crossing_s - 4.2, track.crossing_s - 3.8), thickness=8)

    return cv2.dilate(layer, np.ones((2, 2), np.uint8))


def draw_traffic_light(seg, road: Road, track: Track, frame: int, cam: Camera, state: str) -> None:
    """Kerbside pole-mounted signal at the crossing (no lane-spanning mast arm).

    The lane-spanning arm variant was collapsed by Cosmos into a pillar standing
    in the driving lane; a kerbside pole + head reads correctly.
    """
    pose = track.ego[frame]
    pts, hdgs = road.sample_centerline(0.5)
    s_sig = track.crossing_s + 2.0
    right_edge = -sum(lane.width for lane in road.lanes if lane.lane_id < 0)
    pole_t = right_edge + 0.9
    idx = int(np.clip(round(s_sig / 0.5), 0, len(pts) - 1))
    x, y, h = pts[idx][0], pts[idx][1], hdgs[idx]
    px, py = x - math.sin(h) * pole_t, y + math.cos(h) * pole_t
    dummy = np.zeros((cam.height, cam.width), np.uint8)
    draw_box3d(seg, dummy, cam, pose, (px, py), h, (0.20, 0.20, 4.6), "pole", z_center=2.3)
    draw_box3d(seg, dummy, cam, pose, (px, py), h, (0.34, 0.30, 0.95), "traffic light", z_center=5.05)
    head_pt = np.array([[px, py, 5.35]])
    uv, d = cam.project(head_pt, pose)
    if np.isfinite(uv).all() and d[0] > 0.5:
        color = {"green": (60, 200, 60), "yellow": (40, 210, 230), "red": (40, 40, 220)}.get(state, (60, 200, 60))
        cv2.circle(seg, tuple(uv[0].astype(int)), max(3, int(cam.f * 0.14 / float(d[0]))), color, -1, lineType=cv2.LINE_8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="out.json")
    ap.add_argument("--variant-index", type=int, default=1)
    ap.add_argument("--xodr", default="CarlaUE/CarlaUE4/Content/Carla/Maps/OpenDrive/Town01.xodr")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames", type=int, default=93)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--ped-trigger", type=float, default=18.0)
    ap.add_argument("--ped-speed", type=float, default=1.4)
    ap.add_argument("--ped-width", type=float, default=0.55)
    args = ap.parse_args()

    width, height = (int(v) for v in args.size.lower().split("x"))
    plan = json.loads(Path(args.plan).read_text())
    variant = next((v for v in plan["variants"] if v["variant_index"] == args.variant_index), plan["variants"][0])
    bindings = variant.get("bindings") or {}
    roads = parse_opendrive(Path(args.xodr))
    road = select_road(roads)
    bands = road.lane_bands(0.5)
    track = build_track(road, bindings, args.frames, args.fps, args.speed, args.ped_trigger, args.ped_speed)
    ped_dims = (args.ped_width, args.ped_width, 1.8)
    cam = Camera(width, height)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[raster] road {road.road_id} len {road.length:.1f} lanes {sorted(bands)} bindings {bindings}", flush=True)

    seg_frames, edge_frames, depth_frames = [], [], []
    state = str(bindings.get("traffic_light_main.state", "green"))
    for i in range(args.frames):
        ped_pose = track.pedestrian[i]
        actors = [("pedestrian", ped_pose, ped_dims)]
        seg, depth = render_frame(road, bands, track, i, cam, actors)
        draw_traffic_light(seg, road, track, i, cam, state)
        marks = marking_layer(road, bands, track, i, cam, state)
        gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
        edge = cv2.bitwise_or(cv2.Canny(gray, 40, 90), marks)
        seg_frames.append(seg)
        edge_frames.append(cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR))
        depth_frames.append(cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR))

    for name, frames in (("seg.mp4", seg_frames), ("edge.mp4", edge_frames), ("depth.mp4", depth_frames)):
        iio.imwrite(out_dir / name, [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=args.fps, codec="libx264", plugin="pyav")
    meta = {
        "variant_id": variant.get("variant_id"),
        "bindings": bindings,
        "road_id": road.road_id,
        "crossing_s": track.crossing_s,
        "ego_track": track.ego,
        "pedestrian_track": track.pedestrian,
        "camera": {"width": width, "height": height, "fov_deg": cam.fov_deg, "height_m": cam.height_m, "pitch_deg": cam.pitch_deg},
        "frames": args.frames,
        "fps": args.fps,
        "speed_mps": args.speed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[raster] wrote {out_dir}/seg.mp4 edge.mp4 depth.mp4 meta.json", flush=True)


if __name__ == "__main__":
    main()
