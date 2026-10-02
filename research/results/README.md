# Released results

`rq3_interfuser/` holds the raw rows of the RQ3 safety campaign on InterFuser
(PCLA), as written by `research/scripts/run_rq3_queue.sh`:

- seeds 13, 14, 15; routes `town01_spawn0_goal82`, `town01_spawn195_goal197`,
  `town01_spawn68_goal218`, `town01_spawn82_goal200`;
- per (seed, route): one search arm per policy (`semantic` = SCOUT, `random`,
  `lsa`, `kmnc`) and one `control` arm; 10 evaluations per arm, 600 rows.
- search protocol `scout-search-v2` (every row's `protocol_version`).

Rows are unchanged except that absolute paths were made repo-relative and an
`agent_error_fraction` field was added (fraction of ticks on which the agent
raised, computed from the per-run telemetry, which is not shipped). Per-run
traces, telemetry and recordings are not included.

`coverage/if-if-safe-prefix-profile.joblib` is the InterFuser nominal coverage
profile used by the LSA/KMNC baselines and the coverage observer.

Regenerate the tables offline (no CARLA or GPU needed):

```bash
research/.venv/bin/python research/scripts/rq3_analysis.py \
  --root interfuser=research/results/rq3_interfuser --budget 10 \
  --out-dir research/logs/rq3_results
diff research/logs/rq3_results/RQ3.md research/results/rq3_interfuser/RQ3.md
```

`rq3_interfuser/RQ3.md` is the reference output of that command.
