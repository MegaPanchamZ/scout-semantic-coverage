from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_P = Path(__file__).resolve().parents[1] / "scripts/route_audit.py"
_s = importlib.util.spec_from_file_location("route_audit_under_test", _P)
ra = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ra)


def _write(tmp_path, name, length):
    poly = [[float(x), 0.0] for x in range(0, int(length) + 1, 2)]
    (tmp_path / f"{name}_lead_braking.json").write_text(json.dumps({"controller_params": {"route_polyline": poly}}))


def test_audit_flags_short_and_long_routes(tmp_path):
    _write(tmp_path, "short", 40)
    _write(tmp_path, "good", 300)
    _write(tmp_path, "long", 1200)
    res = ra.audit(tmp_path, speed=4.0, margin=1.5, min_length=125, max_length=700)
    assert not res["short"]["suitable"] and not res["long"]["suitable"]
    assert res["good"]["suitable"]
    assert res["good"]["ticks"] == 1125  # 300 m / 4 m/s / 0.1 s * 1.5


def test_tick_budget_is_clamped(tmp_path):
    _write(tmp_path, "tiny", 10)
    res = ra.audit(tmp_path, 4.0, 1.5, 125, 700)
    assert res["tiny"]["ticks"] == 300
