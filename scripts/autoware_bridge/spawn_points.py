#!/usr/bin/env python3
"""Dump CARLA map spawn points per town (run inside the autoware container against a CARLA 0.9.15 server).
Usage: python3 spawn_points.py OUT.json TownXX   (ONE town per fresh CARLA 0.9.15 server: a second load_world crashes it;
results are merged into OUT.json)
The harness specs refer to spawn points by INDEX, and indices are only meaningful for the CARLA build they were
generated on, so we cache the 0.9.15 table and compare it to specs (see scenario_runner.check_spec_consistency)."""
import json, sys
import carla

out, towns = sys.argv[1], sys.argv[2:]
client = carla.Client("localhost", 2000); client.set_timeout(300)
import os
res = json.load(open(out)) if os.path.exists(out) else {}
res["carla_version"] = client.get_server_version()
for t in towns:
    w = client.load_world(t)
    res[t] = [
        {"x": s.location.x, "y": s.location.y, "z": s.location.z,
         "roll": s.rotation.roll, "pitch": s.rotation.pitch, "yaw": s.rotation.yaw}
        for s in w.get_map().get_spawn_points()
    ]
    print(t, len(res[t]), flush=True)
json.dump(res, open(out, "w"), indent=1)
