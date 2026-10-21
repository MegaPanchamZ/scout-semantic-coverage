"""Partition complete routes between two independent CARLA workers."""
import argparse
import copy
import json
from pathlib import Path


def split_manifest(manifest: dict) -> tuple[dict, dict]:
    routes = []
    for job in manifest["jobs"]:
        if job.get("nominal_results"):
            route = Path(job["nominal_results"]).parent.name
            if route not in routes:
                routes.append(route)
    workers = [copy.deepcopy(manifest), copy.deepcopy(manifest)]
    for index, worker in enumerate(workers):
        assigned = routes[index::2]
        worker["jobs"] = [copy.deepcopy(job) for job in manifest["jobs"]
                          if job["id"].endswith("-profile") or any(f"-{route}-" in job["id"] for route in assigned)]
    second = workers[1]
    second["port"] = manifest["port"] + 10
    second["server"]["graphics_adapter"] = 1
    second["env"]["CUDA_VISIBLE_DEVICES"] = "0"
    for job in second["jobs"]:
        command = job["command"]
        if job["id"].endswith("-profile"):
            profile = command[command.index("--output") + 1]
            job["command"] = [command[0], "-c", f"import joblib; joblib.load({profile!r})"]
            continue
        for flag in ("--port", "--server-port"):
            if flag in command:
                command[command.index(flag) + 1] = str(second["port"])
        if "--graphics-adapter" in command:
            command[command.index("--graphics-adapter") + 1] = "1"
        if "--cuda-visible-devices" in command:
            command[command.index("--cuda-visible-devices") + 1] = "0"
    return tuple(workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    for label, worker in zip(("A", "B"), split_manifest(json.loads(args.manifest.read_text()))):
        path = args.manifest.with_name(f"manifest-{label}.json")
        path.write_text(json.dumps(worker, indent=2) + "\n")
        print(f"{label}: {len(worker['jobs'])} jobs, port {worker['port']}: {path}")


if __name__ == "__main__":
    main()
