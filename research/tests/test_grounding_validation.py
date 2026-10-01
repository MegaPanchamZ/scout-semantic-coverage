"""Tests for the grounding-validation workflow."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = ROOT / "research" / "scripts" / "grounding_validation.py"


def _load():
    spec = importlib.util.spec_from_file_location("grounding_validation", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sheet_then_report(tmp_path):
    gv = _load()
    sheet = tmp_path / "groundtruth.csv"
    assert gv._cmd_sheet(SimpleNamespace(out=sheet)) == 0
    rows = list(csv.DictReader(sheet.open()))
    assert rows, "sheet should list crosswalk entries"
    assert any(r["grounding"] == "proxy" for r in rows)

    # unlabelled proxies block the report
    assert gv._cmd_report(SimpleNamespace(labels=sheet, allow_unvalidated=False)) == 1
    assert gv._cmd_report(SimpleNamespace(labels=sheet, allow_unvalidated=True)) == 0

    # labelling every row clears it
    for row in rows:
        row["label"] = "yes"
    with sheet.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    assert gv._cmd_report(SimpleNamespace(labels=sheet, allow_unvalidated=False)) == 0
