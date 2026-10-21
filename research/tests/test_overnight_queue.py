import json
from pathlib import Path
import sys

from research.harness.run_overnight import run_queue
from research.harness.split_overnight_queue import split_manifest


def test_resume_skips_completed_jobs_and_blocks_failed_nominal(tmp_path):
    counter = tmp_path / "counter"
    command = [sys.executable, "-c", f"from pathlib import Path; p=Path({str(counter)!r}); p.write_text(p.read_text()+'x' if p.exists() else 'x')"]
    failed = [sys.executable, "-c", "raise SystemExit(2)"]
    manifest = {"port": 59999, "jobs": [
        {"id": "ok", "command": command},
        {"id": "gate", "command": failed},
        {"id": "search", "command": command, "requires": ["gate"]},
    ]}
    run_queue(manifest, tmp_path / "queue", 1)
    state = run_queue(manifest, tmp_path / "queue", 1)
    assert counter.read_text() == "x"
    assert state["jobs"]["search"]["status"] == "blocked_by_gate"


def test_timeout_is_recorded_and_queue_continues(tmp_path):
    manifest = {"port": 59998, "jobs": [
        {"id": "timeout", "command": [sys.executable, "-c", "import time; time.sleep(30)"], "timeout_seconds": 0.05},
        {"id": "next", "command": [sys.executable, "-c", "pass"]},
    ]}
    state = run_queue(manifest, tmp_path, 1)
    assert state["jobs"]["timeout"]["status"] == "timeout"
    assert state["jobs"]["next"]["status"] == "complete"
    assert json.loads((tmp_path / "queue_state.json").read_text()) == state


def test_workers_have_disjoint_routes_and_independent_ports():
    jobs = []
    for route in ("town01_first", "town03_second"):
        jobs.append({"id": f"lav-{route}-nominal-13", "command": ["python", "run", "--port", "2000"],
                     "nominal_results": f"logs/lav/{route}/nominal-13"})
        jobs.append({"id": f"lav-{route}-semantic", "command": ["python", "search", "--server-port", "2000", "--graphics-adapter", "0", "--cuda-visible-devices", "1"]})
    a, b = split_manifest({"port": 2000, "server": {"graphics_adapter": 0}, "env": {"CUDA_VISIBLE_DEVICES": "1"}, "jobs": jobs})
    assert {job["id"] for job in a["jobs"]}.isdisjoint(job["id"] for job in b["jobs"])
    assert len(a["jobs"]) + len(b["jobs"]) == len(jobs)
    assert (a["port"], b["port"]) == (2000, 2010)
    assert b["jobs"][0]["command"][-1] == "2010"
    assert b["jobs"][1]["command"][-1] == "0"
    assert b["env"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_successful_process_with_failed_driving_gate_cannot_unlock_search(tmp_path):
    results = tmp_path / "runs"
    results.mkdir()
    (results / "result.json").write_text(json.dumps({"reached_goal": False, "collision_count": 0}))
    manifest = {"port": 59997, "jobs": [
        {"id": "nominal", "command": [sys.executable, "-c", "pass"], "nominal_results": str(results)},
        {"id": "search", "command": [sys.executable, "-c", "raise AssertionError('must not run')"], "requires": ["nominal"]},
    ]}
    state = run_queue(manifest, tmp_path / "queue", 1)
    assert state["jobs"]["nominal"]["status"] == "nominal_failed"
    assert state["jobs"]["search"]["status"] == "blocked_by_gate"
