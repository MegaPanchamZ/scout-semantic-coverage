#!/bin/bash
# Paper safety pipeline: wait for the Interfuser safety campaign, run the
# AutoVLA safety campaign, then produce the RQ3 report bundles.
#
# Both campaigns run at SCOUT_EVALS=1 with paired controls (the paper needs
# attributable safety issues, i.e. generated-but-not-matched-control).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
LOG_DIR="${LOG_DIR:-research/logs}"; mkdir -p "$LOG_DIR"

PY="research/.venv/bin/python"
IF_ROOT="research/logs/paper_safety_if"
AUTO_ROOT="research/logs/paper_safety_autovla"
OUT="research/logs/paper_results"
AUTOVLA=research/experiments/EXP-020-policy-comparison/proof-of-concept/run_fse_multimap_autovla.sh

echo "[pipeline] waiting for the Interfuser safety campaign ($(date -Is))"
while pgrep -f 'paper_safety_if' >/dev/null 2>&1; do sleep 60; done
echo "[pipeline] Interfuser done ($(date -Is)); launched rows:"
find "$IF_ROOT" -name rows.jsonl 2>/dev/null | wc -l

echo "[pipeline] starting AutoVLA safety campaign ($(date -Is))"
export SCOUT_GPU=0 SCOUT_EVAL_TIMEOUT=900 CARLA_ROOT="${CARLA_ROOT:-/opt/carla}"
export SCOUT_EVALS=1 SCOUT_OUTPUT_ROOT="${AUTO_ROOT}/seed-13"
for w in A B; do
  bash "$AUTOVLA" "$w" > "${LOG_DIR}/paper_autovla_${w}.log" 2>&1 &
done
wait
echo "[pipeline] AutoVLA done ($(date -Is))"

mkdir -p "$OUT/interfuser" "$OUT/autovla"
for pair in "interfuser:$IF_ROOT" "autovla:$AUTO_ROOT"; do
  name="${pair%%:*}"; root="${pair#*:}"
  echo "[pipeline] reports for $name ($root)"
  "$PY" research/scripts/safety_outcome_report.py --search-root "$root" \
      --out-dir "$OUT/$name" --include-paired-controls || true
  "$PY" research/scripts/safety_utility_report.py --search-root "$root" \
      --out-dir "$OUT/$name" --include-paired-controls || true
done
echo "[pipeline] files:"
find "$OUT" -type f | sort
echo "PIPELINE_DONE"
