"""Search-based baselines for RQ3, sharing PolicyState's interface and the SAME SearchSpace as our semantic policy.

* ``avfuzzer``  -- re-implementation of AV-Fuzzer's search loop (Li et al., ISSRE 2020): a genetic algorithm over the
  scenario parameter vector with fitness = closeness to a safety violation (minimum ego-to-adversary gap; a collision is
  the optimum), tournament selection, crossover, mutation, elitism, a *local iterative search* around the incumbent when the
  population stagnates, and random restarts. Method-specific parameters are listed in ``AVFUZZER_PARAMS`` and recorded in the
  RQ3 protocol document.
* ``semantic_crit`` -- OUR coverage signal v2 (added POST HOC, see research/docs/rq3-protocol.md): semantic coverage stratified by
  criticality. The v1 signal (obligation credit) saturates after the first evaluations on the threshold-crossing family, so it
  stops guiding the search. v2 counts (semantic signature, criticality bin) pairs, where the bin comes from the run's closest
  approach to the adversary (collision / <1 m / 1-3 m / 3-6 m / >6 m): a scenario earns credit for exercising a known semantic
  situation *more critically* than before. Selection = number of newly covered pairs, tiebreak = smaller gap.
* ``causal``    -- Causal-Fuzzer-STYLE search. Implemented from the published description of its two mechanisms (causality-
  based feedback; causality-driven mutation), NOT from the authors' code, which we could not obtain. Causal graph nodes here
  are discretised scenario-parameter bins, an ego-behaviour proxy and violation outcomes; feedback counts newly activated
  (parameter-bin -> behaviour -> outcome) relations; mutation concentrates on the parameters with the largest estimated
  causal (regression) effect on violation risk. Report it as "Causal-Fuzzer-style" in the paper.

Both are *search* strategies only: candidates are evaluated through exactly the same pipeline (harness or Autoware service)
and judged by the same method-independent outcomes (violations found, evaluations to first violation).
"""
from __future__ import annotations

import math
import random
from typing import Any

AVFUZZER_PARAMS = {
    "population_size": 6,          # individuals per generation
    "tournament_k": 2,
    "crossover_prob": 0.8,
    "mutation_prob_per_dim": 0.4,
    "mutation_sigma_scale": 1.0,   # x the dimension's default sigma
    "elite_count": 1,
    "lis_trigger_stagnant_generations": 2,
    "lis_sigma_scale": 0.25,       # local iterative search perturbation, x default sigma
    "restart_stagnant_generations": 4,
}
CAUSAL_PARAMS = {
    "bins_per_dim": 3,
    "min_archive_for_effect": 6,
    "ridge_lambda": 1.0,
    "uniform_floor": 0.1,          # probability floor so no parameter is starved
    "seed_pool": "feedback-weighted",
}

COLLISION_FITNESS = 1000.0


def gap_fitness(row: dict) -> float | None:
    """Higher is closer to a safety violation. Collision with an injected actor is the optimum."""
    if row.get("run_error"):
        return None
    if row.get("terminated_by_collision") or (row.get("collision_count") or 0) > 0:
        return COLLISION_FITNESS
    gap = row.get("min_gap_m")
    if gap is None:
        return None
    return -float(gap)


def _normalised(space, values: dict[str, float]) -> list[float]:
    out = []
    for d in space.dimensions():
        span = (d.high - d.low) or 1.0
        out.append((float(values.get(d.name, d.low)) - d.low) / span)
    return out


def make_policy_state(policy: str, space: Any, base_cls: type):
    """Factory used by policy_search: the baselines subclass the supplied PolicyState so shared behaviour is unchanged."""
    if policy == "avfuzzer":
        return _make_avfuzzer(space, base_cls)
    if policy == "causal":
        return _make_causal(space, base_cls)
    if policy == "semantic_crit":
        return _make_semantic_crit(space, base_cls)
    return base_cls(policy, space)


def _make_avfuzzer(space, base_cls):
    P = AVFUZZER_PARAMS

    class AVFuzzerState(base_cls):
        def __init__(self, policy, space):
            super().__init__(policy, space)
            self.pop: list[tuple[dict[str, float], float]] = []     # evaluated individuals of the current generation
            self.pending: list[dict[str, float]] = []               # queued candidates (current generation)
            self.generation = 0
            self.incumbent: tuple[dict[str, float], float] | None = None
            self.stagnant = 0
            self.mode = "ga"                                        # ga | lis

        # ---- feedback ----
        def observe(self, values, fitness, tiebreak=None, **kw):
            if not values or fitness is None:
                return
            self.pop.append((dict(values), float(fitness)))
            if len(self.pop) >= P["population_size"]:
                self._end_generation()

        def _end_generation(self):
            best = max(self.pop, key=lambda t: t[1])
            if self.incumbent is None or best[1] > self.incumbent[1] + 1e-9:
                self.incumbent, self.stagnant = best, 0
            else:
                self.stagnant += 1
            self.generation += 1
            self._parents = list(self.pop)
            self.pop = []
            self.pending = []
            if self.stagnant >= P["restart_stagnant_generations"]:
                self.mode, self.incumbent, self.stagnant, self._parents = "ga", None, 0, []   # restart
            elif self.stagnant >= P["lis_trigger_stagnant_generations"] and self.incumbent is not None:
                self.mode = "lis"
            else:
                self.mode = "ga"

        # ---- generation ----
        def next_candidate(self, rng, seen):
            if not self.pending:
                self.pending = self._plan_generation(rng)
            for _ in range(len(self.pending)):
                cand = self.pending.pop(0)
                if self.space.key(cand) not in seen:
                    return cand
                if not self.pending:
                    break
            return self._sample_uniform(rng, seen)

        def _plan_generation(self, rng):
            n = P["population_size"]
            parents = getattr(self, "_parents", [])
            if not parents:                                         # first generation / after restart: random
                return [self.space.sample(rng) for _ in range(n)]
            if self.mode == "lis" and self.incumbent is not None:   # local iterative search around the incumbent
                return [self._gauss_from(rng, self.incumbent[0], P["lis_sigma_scale"]) for _ in range(n)]
            children = []
            elite = sorted(parents, key=lambda t: -t[1])[: P["elite_count"]]
            children.extend(self.space.clamp(dict(e[0])) for e in elite)
            while len(children) < n:
                a, b = self._tournament(rng, parents), self._tournament(rng, parents)
                if rng.random() < P["crossover_prob"]:
                    child = {d.name: (a[d.name] if rng.random() < 0.5 else b[d.name]) for d in self.space.dimensions()}
                else:
                    child = dict(a)
                children.append(self._gauss_from(rng, child, P["mutation_sigma_scale"], P["mutation_prob_per_dim"]))
            return children

        def _tournament(self, rng, parents):
            cands = rng.sample(parents, min(P["tournament_k"], len(parents)))
            return dict(max(cands, key=lambda t: t[1])[0])

        def _gauss_from(self, rng, base, scale, prob=1.0):
            out = {}
            for d in self.space.dimensions():
                v = float(base.get(d.name, d.missing_value()))
                if rng.random() < prob:
                    v += rng.gauss(0.0, d.sigma_or_default * scale)
                out[d.name] = v
            return self.space.clamp(out)

    return AVFuzzerState("avfuzzer", space)


def _make_causal(space, base_cls):
    P = CAUSAL_PARAMS

    class CausalFuzzerStyleState(base_cls):
        def __init__(self, policy, space):
            super().__init__(policy, space)
            self.archive: list[dict] = []                 # {x: normalised vector, values, risk, violation, edges, score}
            self.edges: set[tuple] = set()                # activated causal relations (parameter bins -> behaviour -> outcome)

        # ---- causality-based feedback ----
        def _bins(self, xs):
            k = P["bins_per_dim"]
            return tuple(min(int(x * k), k - 1) for x in xs)

        def _behaviour(self, row):
            if row.get("reached_goal"):
                return "completes"
            gap = row.get("min_gap_m")
            if gap is not None and gap < 3.0:
                return "close_call"
            return "stalls_or_timeout"

        def observe(self, values, fitness, tiebreak=None, row=None, **kw):
            if not values or fitness is None or row is None:
                return
            xs = _normalised(self.space, values)
            beh = self._behaviour(row)
            outcome = "violation" if (row.get("terminated_by_collision") or (row.get("collision_count") or 0) > 0) else "safe"
            bins = self._bins(xs)
            new_edges = set()
            for i, b in enumerate(bins):                  # relation: (dimension i, bin) -> behaviour -> outcome
                e = (i, b, beh, outcome)
                if e not in self.edges:
                    new_edges.add(e)
            self.edges |= new_edges
            risk = gap_fitness(row)
            self.archive.append({"x": xs, "values": dict(values), "risk": float(risk if risk is not None else -50.0),
                                 "violation": outcome == "violation", "score": len(new_edges) + (2.0 if outcome == "violation" else 0.0)})

        # ---- causality-driven mutation ----
        def _effects(self):
            """|ridge-regression coefficient| of violation risk on each normalised parameter (>= uniform floor)."""
            n, d = len(self.archive), len(self.space.dimensions())
            if n < P["min_archive_for_effect"]:
                return [1.0 / d] * d
            ys = [a["risk"] for a in self.archive]
            ym = sum(ys) / n
            ys = [min(max(y, -50.0), 50.0) - ym for y in ys]                       # clip the collision spike
            X = [a["x"] for a in self.archive]
            xm = [sum(r[j] for r in X) / n for j in range(d)]
            beta = []
            for j in range(d):                                                       # marginal ridge slope (diagonal approx.)
                sxx = sum((r[j] - xm[j]) ** 2 for r in X) + P["ridge_lambda"] * 0.01
                sxy = sum((r[j] - xm[j]) * ys[i] for i, r in enumerate(X))
                beta.append(abs(sxy / sxx))
            tot = sum(beta) or 1.0
            floor = P["uniform_floor"] / d
            return [(1 - P["uniform_floor"]) * b / tot + floor for b in beta]

        def next_candidate(self, rng, seen):
            if len(self.archive) < 3:
                return self._sample_uniform(rng, seen)
            weights = [a["score"] + 0.1 for a in self.archive]                       # feedback-weighted seed selection
            seed = rng.choices(self.archive, weights=weights, k=1)[0]
            eff = self._effects()
            dims = self.space.dimensions()
            for _ in range(64):
                out = {}
                for j, d in enumerate(dims):
                    v = float(seed["values"].get(d.name, d.missing_value()))
                    if rng.random() < min(1.0, 0.25 + 1.5 * eff[j]):                 # high-impact parameters mutate more often/more
                        v += rng.gauss(0.0, d.sigma_or_default * (0.5 + 2.0 * eff[j] * len(dims)) * 0.5)
                    out[d.name] = v
                cand = self.space.clamp(out)
                if self.space.key(cand) not in seen:
                    return cand
            return self._sample_uniform(rng, seen)

    return CausalFuzzerStyleState("causal", space)


CRIT_EDGES = (0.0, 1.0, 3.0, 6.0)          # gap (m) bin edges: collision | <1 | 1-3 | 3-6 | >6


def criticality_bin(row: dict) -> int:
    if row.get("terminated_by_collision") or (row.get("collision_count") or 0) > 0:
        return 0
    gap = row.get("min_gap_m")
    if gap is None:
        return len(CRIT_EDGES)
    for i, e in enumerate(CRIT_EDGES[1:], start=1):
        if gap < e:
            return i
    return len(CRIT_EDGES)


def critical_pairs(row: dict) -> set[tuple]:
    """(semantic signature, criticality bin) pairs exercised by a run: a signature seen at bin b also covers all bins > b
    (a closer encounter subsumes a looser one), so credit measures progress toward critical situations only."""
    b = criticality_bin(row)
    sigs = row.get("semantic_covered_signatures") or []
    return {(s, k) for s in sigs for k in range(b, len(CRIT_EDGES) + 1)}


def _make_semantic_crit(space, base_cls):
    class SemanticCritState(base_cls):
        def __init__(self, policy, space):
            super().__init__(policy, space)
            self.covered: set[tuple] = set()
            self.best_new = -1.0
            self.best_gap_key = float("-inf")

        def observe(self, values, fitness, tiebreak=None, row=None, **kw):
            if not values or row is None or row.get("run_error"):
                return
            pairs = critical_pairs(row)
            new = len(pairs - self.covered)
            self.covered |= pairs
            gap = gap_fitness(row)                      # higher = closer to a violation
            key = (float(new), float(gap if gap is not None else -50.0))
            if key > (self.best_new, self.best_gap_key):
                self.best_new, self.best_gap_key, self.best_values = key[0], key[1], dict(values)
                self.best_fitness = float(new)
    return SemanticCritState("semantic_crit", space)
