#!/usr/bin/env python3
"""Host-side client for the persistent Autoware scenario service (see scenario_runner.py --serve).

    svc = AutowareService("Town01")      # starts CARLA 0.9.15 + Autoware (real perception) if not already running
    res = svc.evaluate(spec_dict)        # blocks; returns the harness RunResult-shaped dict (with metadata.semantic)

Any search process (our policy search, a re-implemented AV-Fuzzer, ...) can call evaluate(); the service resets the ego
between episodes, so a call costs ~30-75 s instead of a ~6 min cold start. Requests are files in SPOOL/in, results in SPOOL/out.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
DEFAULT_SPOOL = Path(os.path.expanduser("~/autoware_data/spool"))


class ServiceError(RuntimeError):
    pass


class AutowareService:
    def __init__(self, town: str, spool: Path = DEFAULT_SPOOL, perception: str = "centerpoint/centerpoint",
                 start_timeout: float = 900.0):
        self.town, self.spool, self.perception, self.start_timeout = town, Path(spool), perception, start_timeout
        self.inbox, self.outbox = self.spool / "in", self.spool / "out"
        self.proc: subprocess.Popen | None = None
        self.ensure()

    # ---------- lifecycle ----------
    def _ready(self) -> bool:
        return (self.outbox / "_READY").exists() and self._alive()

    def _alive(self) -> bool:
        r = subprocess.run(["docker", "ps", "-q", "-f", "name=^autoware$"], capture_output=True, text=True)
        return bool(r.stdout.strip())

    def ensure(self) -> None:
        if self._ready():
            return
        self.stop()
        for d in (self.inbox, self.outbox):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, AW_ARGS=f"lidar_detection_model:={self.perception}")
        self.proc = subprocess.Popen(["bash", str(SCRIPTS / "serve_autoware.sh"), self.town], env=env,
                                     stdout=open("/tmp/autoware_service.log", "ab"), stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < self.start_timeout:
            if self._ready():
                return
            if self.proc.poll() is not None:
                raise ServiceError(f"service exited early (rc={self.proc.returncode}); see /tmp/autoware_service.log")
            time.sleep(3)
        raise ServiceError("service did not become ready; see /tmp/autoware_service.log")

    def stop(self) -> None:
        try:
            (self.inbox).mkdir(parents=True, exist_ok=True)
            (self.inbox / "STOP").write_text("1")
        except OSError:
            pass
        subprocess.run(["docker", "rm", "-f", "autoware"], capture_output=True)
        subprocess.run(["pkill", "-x", "CarlaUE4-Linux-"], capture_output=True)
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None
        (self.outbox / "_READY").unlink(missing_ok=True)

    # ---------- evaluation ----------
    def evaluate(self, spec: dict, timeout: float = 600.0, restarts: int = 1) -> dict:
        """Run one ScenarioSpec dict. Raises ServiceError on infrastructure failure (after `restarts` service restarts)."""
        sid = spec["scenario_id"]
        last_err = None
        for attempt in range(restarts + 1):
            self.ensure()
            for suffix in (".json", ".error.json"):
                (self.outbox / f"{sid}{suffix}").unlink(missing_ok=True)
            tmp = self.inbox / f".{sid}.tmp"
            tmp.write_text(json.dumps(spec))
            tmp.rename(self.inbox / f"{sid}.json")          # atomic publish
            t0 = time.time()
            while time.time() - t0 < timeout:
                ok, err = self.outbox / f"{sid}.json", self.outbox / f"{sid}.error.json"
                if ok.exists():
                    time.sleep(0.2)
                    return json.loads(ok.read_text())
                if err.exists():
                    time.sleep(0.2)
                    last_err = json.loads(err.read_text()).get("error", "unknown error")
                    break
                if not self._alive():
                    break
                time.sleep(1)
            if attempt < restarts:                           # hung, crashed, or runner error: cold-restart once
                self.stop()
        raise ServiceError(last_err or f"evaluation of {sid} failed/timed out after {restarts + 1} attempt(s)")

    def semantic_stream(self, scenario_id: str) -> Path | None:
        files = sorted((self.outbox / "semantic" / scenario_id / "stream").glob("*-semantic-stream.jsonl"))
        return files[-1] if files else None
