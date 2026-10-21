#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bulk-render Cosmos control clips for DSL scenario variants (EXP-023 factory).

Reads a compiled plan (out.json), rasterizes the selected variants with
control_rasterizer, writes one directory per variant with seg/edge/depth/meta,
and optionally stages everything to the RunPod network volume in a single pod
trip so the Cosmos endpoint can consume the control paths.

Usage:
  research/.venv/bin/python research/harness/render_scenario_controls.py \
      --plan out.json --variants 1,3 --output-root research/logs/cosmos/control_inputs/exp023
  # add --stage to push to the volume (pod mount /workspace -> worker /runpod-volume)
  # add --ped-trigger 34 --ped-speed 1.1 --ped-width 0.8 for prominent crossers
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import control_rasterizer as cr  # noqa: E402

REPO = HERE.parents[1]
VOLUME_ID = "bvj6so6y8r"
DC = "US-MO-2"
SSH_OPTS = ["-i", str(Path.home() / ".ssh/id_runpod"), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20"]


def render_variant(plan_path: Path, variant_index: int, xodr: Path, out_dir: Path, frames: int, fps: int,
                   ped_trigger: float, ped_speed: float, ped_width: float) -> dict:
    cmd = [
        sys.executable, str(HERE / "control_rasterizer.py"),
        "--plan", str(plan_path), "--variant-index", str(variant_index), "--xodr", str(xodr),
        "--output-dir", str(out_dir), "--frames", str(frames), "--fps", str(fps),
        "--ped-trigger", str(ped_trigger), "--ped-speed", str(ped_speed), "--ped-width", str(ped_width),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    ok = result.returncode == 0 and (out_dir / "seg.mp4").exists()
    return {"variant_index": variant_index, "dir": str(out_dir), "ok": ok, "log": (result.stdout + result.stderr).strip()[-400:]}


def stage(output_dirs: list[Path]) -> str:
    load_dotenv(REPO / "research" / ".env")
    api_key = os.environ["RUNPOD_API_KEY"]
    ah = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    pub_key = Path(str(Path.home() / ".ssh/id_runpod") + ".pub").read_text().strip()

    def gql(query: str, variables: dict | None = None) -> dict:
        body = requests.post("https://api.runpod.io/graphql", json={"query": query, "variables": variables or {}}, headers=ah, timeout=60).json()
        if "errors" in body:
            raise RuntimeError(json.dumps(body["errors"])[:220])
        return body.get("data") or {}

    pod_input = {"cloudType": "SECURE", "gpuCount": 1, "name": "mres-controls-stage", "imageName": "runpod/base:1.3.1-ubuntu2204",
                 "ports": "22/tcp", "containerDiskInGb": 20, "volumeInGb": 0, "networkVolumeId": VOLUME_ID,
                 "env": [{"key": "PUBLIC_KEY", "value": pub_key}], "dataCenterId": DC}
    pod_id = None
    deadline = time.time() + 1800
    while time.time() < deadline and not pod_id:
        for gpu in ["NVIDIA L4", "NVIDIA RTX A4000", "NVIDIA RTX 2000 Ada Generation", "NVIDIA RTX 4000 Ada Generation", "NVIDIA A40", "NVIDIA GeForce RTX 3090"]:
            pod_input["gpuTypeId"] = gpu
            try:
                pod_id = gql("mutation P($input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: $input) { id } }", {"input": pod_input})["podFindAndDeployOnDemand"]["id"]
                break
            except RuntimeError:
                continue
        if not pod_id:
            time.sleep(60)
    if not pod_id:
        return "staging failed: no pod capacity"
    host = port = None
    deadline = time.time() + 900
    while time.time() < deadline and not host:
        time.sleep(12)
        q = 'query { pod(input: {podId: "%s"}) { runtime { ports { ip isIpPublic privatePort publicPort } } } }' % pod_id
        runtime = (gql(q).get("pod") or {}).get("runtime") or {}
        ssh = next((p for p in (runtime.get("ports") or []) if p.get("privatePort") == 22 and p.get("isIpPublic")), None)
        if ssh:
            time.sleep(8)
            host, port = ssh["ip"], ssh["publicPort"]
    if not host:
        return "staging failed: ssh not ready"
    staged = []
    for d in output_dirs:
        for name in ("seg.mp4", "edge.mp4", "depth.mp4", "meta.json"):
            src = d / name
            if not src.exists():
                continue
            dest = f"{d.name}_{name}"
            subprocess.run(["scp", *SSH_OPTS, "-P", str(port), str(src), f"root@{host}:/workspace/controls/{dest}"], capture_output=True, text=True)
            staged.append(dest)
    gql('mutation T($input: PodTerminateInput!) { podTerminate(input: $input) }', {"input": {"podId": pod_id}})
    return f"staged {len(staged)} files (pod {pod_id} terminated): " + ", ".join(staged[:8]) + (" ..." if len(staged) > 8 else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(REPO / "out.json"))
    ap.add_argument("--xodr", default=str(REPO / "CarlaUE/CarlaUE4/Content/Carla/Maps/OpenDrive/Town01.xodr"))
    ap.add_argument("--variants", default="1,3")
    ap.add_argument("--output-root", default=str(REPO / "research/logs/cosmos/control_inputs/exp023"))
    ap.add_argument("--frames", type=int, default=93)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--ped-trigger", type=float, default=34.0)
    ap.add_argument("--ped-speed", type=float, default=1.1)
    ap.add_argument("--ped-width", type=float, default=0.8)
    ap.add_argument("--stage", action="store_true")
    args = ap.parse_args()

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    indices = [int(v) for v in args.variants.split(",") if v.strip()]
    results, dirs = [], []
    for idx in indices:
        variant = next((v for v in plan["variants"] if v["variant_index"] == idx), None)
        if variant is None:
            results.append({"variant_index": idx, "ok": False, "log": "variant not found"})
            continue
        out_dir = root / variant["variant_id"]
        results.append(render_variant(plan_path, idx, Path(args.xodr), out_dir, args.frames, args.fps,
                                      args.ped_trigger, args.ped_speed, args.ped_width))
        dirs.append(out_dir)

    index = {"plan": str(plan_path), "variants": results, "frames": args.frames, "fps": args.fps}
    (root / "index.json").write_text(json.dumps(index, indent=1))
    for r in results:
        print(f"[factory] variant {r['variant_index']}: {'ok' if r.get('ok') else 'FAILED'} -> {r.get('dir')}", flush=True)
    if args.stage and dirs:
        print("[factory]", stage(dirs), flush=True)


if __name__ == "__main__":
    main()
