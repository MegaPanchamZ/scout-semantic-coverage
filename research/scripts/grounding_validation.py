#!/usr/bin/env python3
"""Manual grounding-validation workflow for the simulator->oracle crosswalk.

The coverage engine documents how each simulator predicate grounds an oracle
predicate, and marks `proxy` where the grounding is an engineering approximation
rather than a measured equivalence. This script makes that auditable:

    # 1. emit a labelling sheet (one row per crosswalk entry)
    python research/scripts/grounding_validation.py sheet --out groundtruth.csv

    # 2. a human fills the `label` column with yes/no/unsure and can edit `notes`

    # 3. summarise agreement and flag still-unvalidated proxies
    python research/scripts/grounding_validation.py report --labels groundtruth.csv

`report` exits non-zero (unless --allow-unvalidated) when any `proxy` mapping
has not been labelled, so the campaign cannot silently claim validated grounding.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.harness.coverage_engine import (  # noqa: E402
    CROSSWALK,
    GROUNDING_PROXY,
)

FIELDS = [
    "sim_predicate",
    "oracle_predicate",
    "grounding",
    "nodes",
    "evidence",
    "reason",
    "label",  # yes | no | unsure  (human-filled)
    "notes",  # human-filled
]

POSITIVE = {"yes", "y", "true", "correct", "agree", "valid", "ok"}
NEGATIVE = {"no", "n", "false", "incorrect", "disagree", "invalid", "wrong"}


def _rows() -> list[dict[str, str]]:
    rows = []
    for mapping in CROSSWALK:
        rows.append(
            {
                "sim_predicate": mapping.sim_predicate,
                "oracle_predicate": mapping.oracle_predicate or "",
                "grounding": mapping.grounding,
                "nodes": "|".join(sorted(mapping.nodes)),
                "evidence": mapping.evidence,
                "reason": mapping.reason or "",
                "label": "",
                "notes": "",
            }
        )
    return rows


def _cmd_sheet(args: argparse.Namespace) -> int:
    rows = _rows()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    proxies = sum(1 for r in rows if r["grounding"] == GROUNDING_PROXY)
    print(f"wrote {len(rows)} crosswalk rows ({proxies} proxy) to {out}")
    print("fill the `label` column with yes/no/unsure, then run `report`.")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    with Path(args.labels).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("no rows in labels file")
        return 1

    by_grounding: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    unvalidated: list[dict[str, str]] = []
    proxy_unvalidated: list[dict[str, str]] = []
    for row in rows:
        grounding = (row.get("grounding") or "").strip()
        by_grounding[grounding] += 1
        label = (row.get("label") or "").strip().lower()
        labels[label or "<empty>"] += 1
        if not label:
            unvalidated.append(row)
            if grounding == GROUNDING_PROXY:
                proxy_unvalidated.append(row)

    positive = sum(labels[k] for k in POSITIVE)
    negative = sum(labels[k] for k in NEGATIVE)
    decided = positive + negative
    agreement = (positive / decided) if decided else None

    print(f"crosswalk rows: {len(rows)}")
    for grounding, count in sorted(by_grounding.items()):
        print(f"  grounding={grounding or '<none>'}: {count}")
    print(f"labelled: {len(rows) - len(unvalidated)}/{len(rows)}  "
          f"(yes={positive} no={negative} unsure={labels.get('unsure', 0)} empty={len(unvalidated)})")
    if agreement is not None:
        print(f"agreement on labelled rows: {agreement:.2%}")
    if proxy_unvalidated:
        print(f"UNVALIDATED PROXY MAPPINGS: {len(proxy_unvalidated)}")
        for row in proxy_unvalidated[:20]:
            print(f"  proxy: {row.get('sim_predicate')} -> {row.get('oracle_predicate')}")
        if not args.allow_unvalidated:
            print("grounding validation incomplete; label the proxy rows or pass --allow-unvalidated.")
            return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sheet = sub.add_parser("sheet", help="emit a labelling sheet for the crosswalk")
    sheet.add_argument("--out", type=Path, default=Path("research/logs/grounding/groundtruth.csv"))
    sheet.set_defaults(func=_cmd_sheet)
    report = sub.add_parser("report", help="summarise a filled labelling sheet")
    report.add_argument("--labels", type=Path, required=True)
    report.add_argument("--allow-unvalidated", action="store_true")
    report.set_defaults(func=_cmd_report)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
