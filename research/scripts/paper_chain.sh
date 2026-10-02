#!/bin/bash
# Paper RQ3 chain: finish Interfuser (14 routes), run AutoVLA across four maps,
# then produce the safety report/utility bundles and the paper bundle.
#
# Both campaigns: SCOUT_EVALS=1, 8 no-adversary controls/route, and a paired
# matched control per generated candidate (needed for attributable safety).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

PY="research/.venv/bin/python"
IF_ROOT="research/logs/paper_safety_if"
AUTO_ROOT="research/logs/paper_safety_autovla"
OUT="research/logs/paper_results"
HZ=research/experiments/EXP-020-policy-comparison/proof-of-concept/run_fse_hazard_workers.sh
VA=research/experiments/EXP-020-policy-comparison/proof-of-concept/run_fse_multimap_autovla.sh

echo "[chain] restarting crashed CARLA ($(date -Is))"
WORKERS="A B C D E F G H" QUALITY=Epic bash research/scripts/start_carla_fleet.sh || true

export SCOUT_GPU=0 SCOUT_EVAL_TIMEOUT=1500 CARLA_ROOT=/opt/carla
export SCOUT_EVALS=1 SCOUT_ENSURE_LEAD_SPECS=0 SCOUT_CONTROLS=0

echo "[chain] resuming Interfuser safety campaign ($(date -Is))"
export SCOUT_OUTPUT_ROOT="${IF_ROOT}/seed-13"
for w in A B C D E; do bash "$HZ" "$w" > "/root/work/paper_if_${w}.log" 2>&1 & done
SCOUT_ROUTES="town01_spawn115_goal206 town10hd_spawn43_goal100" bash "$HZ" G > /root/work/paper_if_G.log 2>&1 &
SCOUT_ROUTES="town01_spawn195_goal197 town01_spawn82_goal200" bash "$HZ" H > /root/work/paper_if_H.log 2>&1 &
wait
echo "[chain] Interfuser done ($(date -Is)); arms with rows: $(find "$IF_ROOT" -name rows.jsonl | wc -l)"

echo "[chain] freeing VRAM: keep ports 2000-2030 for AutoVLA"
for p in 2040 2050 2060 2070; do
  pid=$(ps -eo pid,cmd | grep '[C]arlaUE4-Linux-Shipping' | grep "carla-rpc-port=$p" | awk '{print $1}' | head -1)
  [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
done
sleep 5

echo "[chain] AutoVLA safety campaign, 8 routes / 4 maps / 4 workers ($(date -Is))"
export SCOUT_OUTPUT_ROOT="${AUTO_ROOT}/seed-13" SCOUT_EVAL_TIMEOUT=1200
SCOUT_PORT=2000 SCOUT_ROUTES="town01_spawn0_goal82 town01_spawn55_goal154" bash "$VA" A > /root/work/paper_autovla_A.log 2>&1 &
SCOUT_PORT=2010 SCOUT_ROUTES="town03_spawn121_goal2 town03_spawn125_goal223" bash "$VA" B > /root/work/paper_autovla_B.log 2>&1 &
SCOUT_PORT=2020 SCOUT_ROUTES="town05_spawn0_goal124 town05_spawn218_goal257" bash "$VA" C > /root/work/paper_autovla_C.log 2>&1 &
SCOUT_PORT=2030 SCOUT_ROUTES="town10hd_spawn0_goal44 town10hd_spawn1_goal63" bash "$VA" D > /root/work/paper_autovla_D.log 2>&1 &
wait
echo "[chain] AutoVLA done ($(date -Is)); arms with rows: $(find "$AUTO_ROOT" -name rows.jsonl 2>/dev/null | wc -l)"

echo "[chain] reports ($(date -Is))"
mkdir -p "$OUT/interfuser" "$OUT/autovla"
"$PY" research/scripts/safety_outcome_report.py --search-root "$IF_ROOT" --out-dir "$OUT/interfuser" --include-paired-controls || true
"$PY" research/scripts/safety_utility_report.py --search-root "$IF_ROOT" --out-dir "$OUT/interfuser" --include-paired-controls || true
"$PY" research/scripts/safety_outcome_report.py --search-root "$AUTO_ROOT" --out-dir "$OUT/autovla" --include-paired-controls || true
"$PY" research/scripts/safety_utility_report.py --search-root "$AUTO_ROOT" --out-dir "$OUT/autovla" --include-paired-controls || true
"$PY" research/scripts/paper_bundle.py --root "interfuser=$IF_ROOT" --root "autovla=$AUTO_ROOT" --out-dir "$OUT" || true
echo "[chain] files:"; find "$OUT" -type f | sort
echo CHAIN_DONE
