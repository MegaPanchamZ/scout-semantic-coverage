#!/usr/bin/env python3
"""Aggregate an RQ3 campaign into method-independent outcome tables (+ figure).

Method-independent outcomes only (they do not depend on any method's own coverage/fitness notion):
  * violations found within the budget (evaluations ending in a collision with an injected actor)
  * evaluations to first violation (and wall-clock / simulated time to first violation)
  * distinct violations: violating candidates that are > D apart in normalised parameter space (greedy clustering)
Statistics: one-sided Mann-Whitney U (ours > baseline) over the runs (route x seed pairs), with Holm correction.

usage: rq3_aggregate.py CAMPAIGN_DIR [--ours semantic] [--out DIR] [--distinct-d 0.15]
"""
import argparse, json, math, statistics
from collections import defaultdict
from pathlib import Path


def load_runs(root: Path, include_initial: bool = False):
    runs = defaultdict(dict)                       # method -> {(route, seed): rows}
    for rows_path in sorted(root.glob("*/*/seed*/rows.jsonl")):
        seed_dir = rows_path.parent
        method, route, seed = seed_dir.parts[-3], seed_dir.parts[-2], seed_dir.name
        rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
        rows = sorted(rows, key=lambda r: r.get("eval_index", 0))
        # evaluation 0 is the provided initial scenario: identical for all methods, so it is not a search outcome
        runs[method][(route, seed)] = rows if include_initial else [r for r in rows if r.get("eval_index", 0) != 0]
    return runs


def is_violation(r):
    return bool(r.get("terminated_by_collision") or (r.get("collision_count") or 0) > 0) and not r.get("run_error")


def vec(r):
    c = r.get("candidate") or {}
    return [c[k] for k in sorted(c)]


def distinct(rows, bounds, d_thr):
    centres = []
    for r in rows:
        if not is_violation(r):
            continue
        v = [(x - lo) / ((hi - lo) or 1) for x, (lo, hi) in zip(vec(r), bounds)]
        if all(math.dist(v, c) > d_thr for c in centres):
            centres.append(v)
    return len(centres)


def run_stats(rows, bounds, d_thr):
    viol = [i for i, r in enumerate(rows) if is_violation(r)]
    errs = sum(1 for r in rows if r.get("run_error"))
    first = viol[0] + 1 if viol else None
    wall = sum(r.get("duration_s") or 0 for r in rows[: first]) if first else None
    simt = rows[viol[0]].get("first_violation_sim_s") if viol else None
    curve, c = [], 0
    for r in rows:
        c += is_violation(r); curve.append(c)
    return {"n": len(rows), "errors": errs, "violations": len(viol), "first": first, "wall_first_s": wall,
            "sim_first_s": simt, "distinct": distinct(rows, bounds, d_thr), "curve": curve}


def mwu_greater(a, b):
    """One-sided Mann-Whitney U p-value for a > b (normal approx. with tie correction; exact enough for n>=6)."""
    n1, n2 = len(a), len(b)
    allv = sorted([(x, 0) for x in a] + [(x, 1) for x in b])
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    r1 = sum(ranks[k] for k, (_, g) in enumerate(allv) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    ties = defaultdict(int)
    for x, _ in allv:
        ties[x] += 1
    n = n1 + n2
    sigma = math.sqrt(n1 * n2 / 12 * ((n + 1) - sum(t ** 3 - t for t in ties.values()) / (n * (n - 1)))) or 1e-9
    z = (u1 - mu - 0.5) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2)), u1 / (n1 * n2)       # p, common-language effect size P(a>b)+0.5P(=)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [None] * len(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); out[i] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", type=Path); ap.add_argument("--ours", default="semantic")
    ap.add_argument("--out", type=Path, default=None); ap.add_argument("--distinct-d", type=float, default=0.15)
    ap.add_argument("--include-initial", action="store_true", help="count evaluation 0 (the provided initial scenario, identical for every method)")
    a = ap.parse_args()
    out = a.out or a.campaign
    runs = load_runs(a.campaign, a.include_initial)
    if not runs:
        raise SystemExit(f"no rows.jsonl under {a.campaign}")
    # parameter bounds from the shared campaign space (identical for every method)
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from research.harness.search_space import make_campaign_space
    dims = sorted(make_campaign_space(seed=0).dimensions(), key=lambda d: d.name)
    bounds = [(d.low, d.high) for d in dims]

    stats = {m: {k: run_stats(rows, bounds, a.distinct_d) for k, rows in rs.items()} for m, rs in runs.items()}
    common = set.intersection(*[set(s) for s in stats.values()])           # only (route, seed) pairs every method completed
    budget = min(st["n"] for s in stats.values() for k, st in s.items() if k in common) if common else 0
    lines = [f"# RQ3 results ({a.campaign.name})", "",
             f"Runs compared: {len(common)} (route x seed) per method; budget = {budget} search evaluations"
             f"{' (initial scenario included)' if a.include_initial else ' after the shared initial scenario'}; "
             f"distinct-violation radius = {a.distinct_d} (normalised parameter space).", "",
             "| method | violations (mean±sd) | evals to 1st violation (median) | wall s to 1st (median) | distinct violations (mean) | runs w/ >=1 violation | infra errors |",
             "|---|---|---|---|---|---|---|"]
    per = {}
    for m in sorted(stats, key=lambda m: (m != a.ours, m)):
        ss = [stats[m][k] for k in sorted(common)]
        for s in ss:
            s["violations_b"] = s["curve"][budget - 1] if len(s["curve"]) >= budget else s["curve"][-1]
        v = [s["violations_b"] for s in ss]; firsts = [s["first"] for s in ss if s["first"] and s["first"] <= budget]
        walls = [s["wall_first_s"] for s in ss if s["first"] and s["first"] <= budget]
        per[m] = v
        lines.append(f"| {m} | {statistics.mean(v):.2f} ± {statistics.pstdev(v):.2f} | "
                     f"{(statistics.median(firsts) if firsts else float('nan')):.1f} | "
                     f"{(statistics.median(walls) if walls else float('nan')):.0f} | "
                     f"{statistics.mean(s['distinct'] for s in ss):.2f} | {len(firsts)}/{len(ss)} | {sum(s['errors'] for s in ss)} |")
    if a.ours in per:
        lines += ["", f"## {a.ours} vs baselines (one-sided Mann-Whitney U on violations, Holm-corrected)", "",
                  "| baseline | P(ours > baseline) | p | p (Holm) |", "|---|---|---|---|"]
        bl = [m for m in per if m != a.ours]
        res = [mwu_greater(per[a.ours], per[m]) for m in bl]
        adj = holm([p for p, _ in res])
        for m, (p, es), pa in zip(bl, res, adj):
            lines.append(f"| {m} | {es:.2f} | {p:.3f} | {pa:.3f} |")
    # LaTeX table for the paper (numbers come straight from the data; nothing is typed by hand)
    tex = ["\\begin{table}[t]", "\\centering", "\\small",
           f"\\caption{{RQ3 on Autoware: method-independent outcomes after the shared initial scenario "
           f"({len(common)} runs per method = routes $\\times$ seeds, {budget} search evaluations each). "
           f"$p$: one-sided Mann--Whitney U on violations, \\toolname{{}} vs.\\ the method, Holm-corrected.}}",
           "\\label{tab:rq3-results}", "\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
           "Method & Violations & Evals to 1st & Wall s to 1st & Distinct & Runs w/ viol. & $p$ \\\\", "\\midrule"]
    pvals = {}
    if a.ours in per:
        bl_ = [m for m in per if m != a.ours]
        adj_ = holm([mwu_greater(per[a.ours], per[m])[0] for m in bl_])
        pvals = dict(zip(bl_, adj_))
    tex_names = {
        "semantic": "\\toolname{} (v1)",
        "semantic_crit": "\\toolname{}\\textsubscript{crit} (post-hoc)",
        "avfuzzer": "AV-Fuzzer (re-impl.)",
        "causal": "Causal-Fuzzer-style",
        "random": "Random",
    }
    plot_names = {
        "semantic": "SCOUT (v1)",
        "semantic_crit": "SCOUT-crit (post-hoc)",
        "avfuzzer": "AV-Fuzzer (re-impl.)",
        "causal": "Causal-Fuzzer-style",
        "random": "Random",
    }
    method_order = ["semantic", "semantic_crit", "avfuzzer", "causal", "random"]
    ordered_methods = [m for m in method_order if m in stats] + [m for m in sorted(stats) if m not in method_order]
    for m in ordered_methods:
        ss = [stats[m][k] for k in sorted(common)]
        v = per[m]; firsts = [s["first"] for s in ss if s["first"] and s["first"] <= budget]
        walls = [s["wall_first_s"] for s in ss if s["first"] and s["first"] <= budget]
        name = tex_names.get(m, m.replace("_", "\\_"))
        pcell = "--" if m == a.ours else f"{pvals.get(m, float('nan')):.3f}"
        tex.append(f"{name} & {statistics.mean(v):.2f}$\\pm${statistics.pstdev(v):.2f} & "
                   f"{(statistics.median(firsts) if firsts else float('nan')):.1f} & {(statistics.median(walls) if walls else float('nan')):.0f} & "
                   f"{statistics.mean(s['distinct'] for s in ss):.2f} & {len(firsts)}/{len(ss)} & {pcell} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "rq3_results.tex").write_text("\n".join(tex) + "\n")
    (out / "rq3_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        for m in ordered_methods:
            cs = [stats[m][k]["curve"][:budget] for k in sorted(common)]
            xs = range(1, budget + 1)
            mean = [statistics.mean(c[i] for c in cs if len(c) > i) for i in range(budget)]
            ls = "--" if m == "semantic_crit" else "-"
            lw = 2.0 if m in (a.ours, "semantic_crit") else 1.3
            ax.plot(xs, mean, label=plot_names.get(m, m), lw=lw, ls=ls)
        ax.set_xlabel("Search evaluations (after shared initial scenario)")
        ax.set_ylabel("Cumulative violations (mean over 6 runs)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(out / "rq3_cumulative_violations.pdf"); fig.savefig(out / "rq3_cumulative_violations.png", dpi=160)
    except Exception as e:  # plotting is optional
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
