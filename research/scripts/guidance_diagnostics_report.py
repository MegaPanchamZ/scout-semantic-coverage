#!/usr/bin/env python3
"""Build guidance-effectiveness diagnostic reports from search decision logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.outcome_io import atomic_json
from research.harness.outcome_linkage import build_guidance_diagnostics


def _protocol_outcome_schema(rows_file: Path) -> int | None:
    proto_path = rows_file.parent / "campaign_protocol.json"
    if not proto_path.is_file():
        return None
    try:
        proto = json.loads(proto_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(proto, dict):
        return None
    schema = proto.get("outcome_schema")
    if isinstance(schema, int) and not isinstance(schema, bool):
        return schema
    return None


def load_decisions(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    files = sorted(path.rglob("rows.jsonl")) if path.is_dir() else [path]
    rows: list[dict[str, Any]] = []
    for f in files:
        proto_schema = _protocol_outcome_schema(f)
        for raw_line in f.read_bytes().decode("utf-8").split("\n"):
            line = raw_line.rstrip("\r")
            if not line.strip(" \t\r"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                if proto_schema == 2:
                    obj["outcome_schema_version"] = 2
                rows.append(obj)
    return rows


def build_report(source: Path) -> dict[str, Any]:
    rows = load_decisions(source)
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_policy.setdefault(str(row.get("policy") or "unknown"), []).append(row)
    return {
        "source": str(source),
        "overall": build_guidance_diagnostics(rows),
        "by_policy": {
            policy: build_guidance_diagnostics(items)
            for policy, items in sorted(by_policy.items())
        },
    }


def write_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "guidance_diagnostics.json"
    atomic_json(target, report)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="rows.jsonl file or campaign directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args.source)
    write_report(report, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
