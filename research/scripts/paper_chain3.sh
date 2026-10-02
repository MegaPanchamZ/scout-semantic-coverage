#!/bin/bash
# Finish the AutoVLA RQ3 campaign (resume) and produce all report bundles.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
LOG_DIR="${LOG_DIR:-research/logs}"; mkdir -p "$LOG_DIR"

PY="research/.venv/bin/python"
IF_ROOT="research/logs/paper_safety_if"
AUTO_ROOT="research/logs/paper_safety_autovla"
OUT="research/logs/paper_results"
VA=research/experiments/EXP-020-policy-comparison/proof-of-concept/run_fse_multimap_autovla.sh

# clear orphaned evaluations from the killed parent (avoid two writers per arm)
pids=$(ps -eo pid,cmd | grep '[p]aper_safety_autovla' | awk '{print $1}')
[ -n "$pids" ] && kill -9 $pids 2>/dev/null
sleep 3

export SCOUT_GPU=0 SCOUT_EVAL_TIMEOUT=1500 CARLA_ROOT="${CARLA_ROOT:-/opt/carla}"
export SCOUT_EVALS=1 SCOUT_CONTROLS=0 SCOUT_ENSURE_LEAD_SPECS=0
export SCOUT_OUTPUT_ROOT="${AUTO_ROOT}/seed-13"

echo "[chain3] resuming AutoVLA ($(date -Is)); arms so far: $(find "$AUTO_ROOT" -name rows.jsonl | wc -l)"
SCOUT_PORT=2000 SCOUT_ROUTES="town01_spawn0_goal82 town01_spawn55_goal154" bash "$VA" A > ${LOG_DIR}/paper_autovla_A.log 2>&1 &
SCOUT_PORT=2010 SCOUT_ROUTES="town03_spawn121_goal2 town05_spawn0_goal124" bash "$VA" B > ${LOG_DIR}/paper_autovla_B.log 2>&1 &
SCOUT_PORT=2020 SCOUT_ROUTES="town05_spawn218_goal257 town10hd_spawn0_goal44" bash "$VA" C > ${LOG_DIR}/paper_autovla_C.log 2>&1 &
wait
echo "[chain3] AutoVLA done ($(date -Is)); arms: $(find "$AUTO_ROOT" -name rows.jsonl | wc -l)"

echo "[chain3] reports ($(date -Is))"
mkdir -p "$OUT/interfuser" "$OUT/autovla"
"$PY" research/scripts/safety_outcome_report.py --search-root "$IF_ROOT" --out-dir "$OUT/interfuser" --include-paired-controls || true
"$PY" research/scripts/safety_utility_report.py --search-root "$IF_ROOT" --out-dir "$OUT/interfuser" --include-paired-controls || true
"$PY" research/scripts/safety_outcome_report.py --search-root "$AUTO_ROOT" --out-dir "$OUT/autovla" --include-paired-controls || true
"$PY" research/scripts/safety_utility_report.py --search-root "$AUTO_ROOT" --out-dir "$OUT/autovla" --include-paired-controls || true
"$PY" research/scripts/paper_bundle.py --root "interfuser=$IF_ROOT" --root "autovla=$AUTO_ROOT" --out-dir "$OUT" || true
echo "[chain3] files:"; find "$OUT" -type f | sort
echo CHAIN3_DONE
