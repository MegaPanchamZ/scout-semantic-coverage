#!/usr/bin/env python
"""RQ3 utility analysis over a queue campaign (research/scripts/run_rq3_queue.sh).

Layout: <root>/seed-<s>/<route>/<policy|control>/rows.jsonl. Candidate eval i
of (seed, route, policy) is paired with control eval i of (seed, route): both
use execution seed s + i and the same adversary-stripped spec. An outcome is
*attributable* when the candidate shows it and its paired control does not.

An independent run is one (seed, route, policy) search arm. Reported:

1. Safety issues by type: attributable issues per independent run, mean +/- sd.
2. Semantic obligations -> safety issues: which hazard targets / obligations
   the attributable issues come from.
3. Time to first issue: evals until the first attributable issue of any type
   (1-based), mean +/- sd over runs that found one, plus the find rate.

Runs whose run_error is set, that never ticked, or whose agent raised on more
than 10% of ticks (e.g. CUDA OOM) are invalid; their pairs are skipped.

Usage:
    research/.venv/bin/python research/scripts/rq3_analysis.py \
        --root research/logs/rq3_if --ads interfuser --out-dir research/logs/rq3_results
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.harness.safety_outcomes import classify_run  # noqa: E402

POLICIES = ("semantic", "random", "lsa", "kmnc")
POLICY_LABEL = {"semantic": "SCOUT", "random": "Random", "lsa": "LSA", "kmnc": "KMNC"}
# reported issue types -> constituent outcome names
ISSUE_TYPES = {
    "Collision": ("collision",),
    "Low TTC": ("near_collision",),
    "Unsafe proximity": ("unsafe_proximity",),
    "Rule violation": ("red_light_violation", "lane_departure"),
    "Stuck / route failure": ("stuck", "no_progress", "route_incomplete"),
    "Harsh braking": ("harsh_braking", "emergency_manoeuvre"),
}
# "Any" counts the first four (hard safety) types; route failure and harsh
# braking are reported but kept out of the headline, since a braking lead
# vehicle legitimately causes both.
HARD_TYPES = ("Collision", "Low TTC", "Unsafe proximity", "Rule violation")
AGENT_ERROR_LIMIT = 0.10


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _agent_error_fraction(row: dict) -> float:
    # exported rows (research/results) carry the fraction; telemetry is not shipped
    if "agent_error_fraction" in row:
        return float(row["agent_error_fraction"])
    run_path = row.get("run_json_path")
    if not run_path:
        return 0.0
    telemetry_dir = Path(run_path).parent.parent / "telemetry"
    frames = []
    for path in telemetry_dir.glob("*-telemetry.json"):
        try:
            frames = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0.0
    if not frames:
        return 0.0
    errors = sum(1 for f in frames if isinstance(f.get("agent_step"), dict) and f["agent_step"].get("error"))
    return errors / len(frames)


def _issues(row: dict) -> dict[str, bool] | None:
    """Issue types for one run, or None when the run is invalid."""
    result = classify_run(row)
    if not result.get("valid", True) or row.get("run_error"):
        return None
    if _agent_error_fraction(row) > AGENT_ERROR_LIMIT:
        return None
    return {t: any(bool(result.get(n)) for n in names) for t, names in ISSUE_TYPES.items()}


def _mean_sd(values: list[float]) -> str:
    if not values:
        return "n/a"
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.fmean(values):.2f} ± {sd:.2f}"


def analyse(root: Path, budget: int) -> dict:
    runs = []          # one entry per independent run (seed, route, policy)
    pairs_out = []     # every valid pair, for the obligation analysis
    invalid = Counter()
    for seed_dir in sorted(root.glob("seed-*")):
        for route_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            controls = {r.get("eval_index"): r for r in _rows(route_dir / "control" / "rows.jsonl")}
            control_issues = {i: _issues(r) for i, r in controls.items()}
            control_obs = {i: set(r.get("engine_run_obligations") or []) for i, r in controls.items()}
            for policy in POLICIES:
                cand_rows = sorted(_rows(route_dir / policy / "rows.jsonl"), key=lambda r: r.get("eval_index", 0))
                # skip arms still running, so every run has the same budget
                if len(cand_rows) < budget:
                    continue
                run = {"seed": seed_dir.name, "route": route_dir.name, "policy": policy,
                       "evals": 0, "pairs": 0, "counts": Counter(), "first": None}
                for row in cand_rows:
                    index = row.get("eval_index", 0)
                    run["evals"] += 1
                    cand = _issues(row)
                    ctrl = control_issues.get(index)
                    if cand is None or ctrl is None:
                        invalid[policy] += 1
                        continue
                    run["pairs"] += 1
                    attributable = [t for t in ISSUE_TYPES if cand[t] and not ctrl[t]]
                    for t in attributable:
                        run["counts"][t] += 1
                    hard = [t for t in attributable if t in HARD_TYPES]
                    if hard:
                        run["counts"]["Any"] += 1
                        if run["first"] is None:
                            run["first"] = index + 1
                    pairs_out.append({
                        "policy": policy, "route": route_dir.name, "seed": seed_dir.name, "eval": index,
                        "template": row.get("template"), "hazard_target": row.get("hazard_target"),
                        # obligations the scenario added over its paired control
                        "obligations": sorted(set(row.get("engine_run_obligations") or []) - control_obs.get(index, set())),
                        "new_obligations": row.get("engine_new_obligations"),
                        "attributable": attributable,
                    })
                runs.append(run)
    return {"runs": runs, "pairs": pairs_out, "invalid": invalid}


def report(ads: str, data: dict) -> list[str]:
    runs, pairs, invalid = data["runs"], data["pairs"], data["invalid"]
    lines = [f"## {ads}", ""]
    by_policy = defaultdict(list)
    for run in runs:
        by_policy[run["policy"]].append(run)

    lines += ["### 1. Attributable safety issues per independent run (mean ± sd)", ""]
    cols = list(ISSUE_TYPES) + ["Any"]
    lines.append("| Policy | Runs | Valid pairs | Invalid | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 4))
    for p in POLICIES:
        rs = by_policy.get(p, [])
        cells = [_mean_sd([r["counts"][c] for r in rs]) for c in cols]
        lines.append(f"| {POLICY_LABEL[p]} | {len(rs)} | {sum(r['pairs'] for r in rs)} | {invalid[p]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"`Any` = runs with any of {', '.join(HARD_TYPES)}. Totals over all runs:")
    lines.append("")
    lines.append("| Policy | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for p in POLICIES:
        rs = by_policy.get(p, [])
        lines.append(f"| {POLICY_LABEL[p]} | " + " | ".join(str(sum(r["counts"][c] for r in rs)) for c in cols) + " |")
    lines.append("")

    lines += ["### 3. Evals to first attributable safety issue (any hard type)", ""]
    lines.append("| Policy | Runs | Found | Find rate | Evals to first (found runs, mean ± sd) | Censored mean (not found = budget+1) |")
    lines.append("|---|---|---|---|---|---|")
    for p in POLICIES:
        rs = by_policy.get(p, [])
        found = [r["first"] for r in rs if r["first"] is not None]
        censored = [r["first"] if r["first"] is not None else r["evals"] + 1 for r in rs]
        rate = f"{len(found) / len(rs):.0%}" if rs else "n/a"
        lines.append(f"| {POLICY_LABEL[p]} | {len(rs)} | {len(found)} | {rate} | {_mean_sd(found)} | {_mean_sd(censored)} |")
    lines.append("")

    lines += ["### 2. Semantic obligations -> safety issues", ""]
    lines.append("Attributable issues by hazard target (the semantic obligation the eval was aimed at):")
    lines.append("")
    targets = defaultdict(Counter)
    totals = Counter()
    for pair in pairs:
        key = f"{pair['template']} / {pair['hazard_target']}"
        totals[key] += 1
        for t in pair["attributable"]:
            targets[key][t] += 1
    lines.append("| Template / hazard target | Evals | " + " | ".join(ISSUE_TYPES) + " |")
    lines.append("|" + "---|" * (len(ISSUE_TYPES) + 2))
    for key in sorted(totals, key=lambda k: (-sum(targets[k].values()), k)):
        lines.append(f"| {key} | {totals[key]} | " + " | ".join(str(targets[key][t]) for t in ISSUE_TYPES) + " |")
    lines.append("")
    # obligations over-represented in issue runs
    issue_ob, clean_ob = Counter(), Counter()
    n_issue = n_clean = 0
    for pair in pairs:
        hard = any(t in HARD_TYPES for t in pair["attributable"])
        bucket = issue_ob if hard else clean_ob
        bucket.update(set(pair["obligations"]))
        if hard:
            n_issue += 1
        else:
            n_clean += 1
    if n_issue:
        lines.append(f"Scenario-added obligations (covered by the run, not by its paired control) in runs with a hard attributable issue (n={n_issue}) vs without (n={n_clean}), top by lift:")
        lines.append("")
        lines.append("| Obligation | In issue runs | In clean runs | Lift |")
        lines.append("|---|---|---|---|")
        scored = []
        for ob, k in issue_ob.items():
            p_issue = k / n_issue
            p_clean = clean_ob[ob] / n_clean if n_clean else 0.0
            scored.append((p_issue / max(p_clean, 1e-9) if p_clean else float("inf"), ob, p_issue, p_clean))
        for lift, ob, pi, pc in sorted(scored, key=lambda s: (-s[0], -s[2], s[1]))[:15]:
            lines.append(f"| `{ob}` | {pi:.0%} | {pc:.0%} | {'∞' if lift == float('inf') else f'{lift:.1f}'} |")
        lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="ads=path (repeatable)")
    ap.add_argument("--budget", type=int, default=5, help="evals per run; shorter (unfinished) runs are skipped")
    ap.add_argument("--out-dir", type=Path, default=Path("research/logs/rq3_results"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# RQ3 utility results", ""]
    dump = {}
    for item in args.root:
        ads, _, path = item.partition("=")
        data = analyse(Path(path), args.budget)
        lines += report(ads, data)
        dump[ads] = {"runs": [{**r, "counts": dict(r["counts"])} for r in data["runs"]],
                     "pairs": data["pairs"], "invalid": dict(data["invalid"])}
    (args.out_dir / "RQ3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out_dir / "rq3_data.json").write_text(json.dumps(dump, indent=1), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
