#!/bin/bash
# FSE'27 multi-map extension matrix.
#
# 8 new routes across Town03 (2), Town05 (3), Town10HD (3):
# 4 policies x 50 evals per arm + 8 no-adversary controls per route,
# identical 5-D campaign space, paired seeds, engine metrics on.
# Town01's six routes were already run at the same budget.
#
# Usage: run_fse_multimap.sh {A|B}
#   A -> GPU 0, CARLA port 2000, first 4 routes
#   B -> GPU 1, CARLA port 2010, last 4 routes
#
# Resumable: policy_search.py appends rows.jsonl and skips completed rows.

set -u

# Cap CPU thread pools per worker. The box reports 64 cores, and with several
# CARLA servers + workers running, default 64-thread torch/BLAS pools thrash on
# tiny tensors (Interfuser preprocessing measured ~200x slower at 64 threads).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

WORKER="${1:-}"
if [ -z "$WORKER" ]; then
  echo "usage: run_fse_multimap.sh A or B" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1

PY="${SCOUT_PYTHON:-research/.venv/bin/python}"
SEARCH=research/experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py
BASE=research/experiments/EXP-020-policy-comparison/artifacts/base_specs
OUT="${SCOUT_OUTPUT_ROOT:-research/logs/fse_search_hazard_v2/seed-${SCOUT_SEED:-13}}"
ORACLE=research/experiments/EXP-018-nuscenes-oracle-inventory/artifacts/oracle_inventory_v1.0-trainval.json
EVALS="${SCOUT_EVALS:-50}"
CONTROLS="${SCOUT_CONTROLS:-8}"

case "$WORKER" in
  A) PORT=2000; CUDA=0
     ROUTES="town01_spawn0_goal82 town01_spawn55_goal154" ;;
  B) PORT=2010; CUDA=1
     ROUTES="town01_spawn68_goal218 town03_spawn121_goal2" ;;
  C) PORT=2020; CUDA=2
     ROUTES="town03_spawn125_goal223 town05_spawn0_goal124" ;;
  D) PORT=2030; CUDA=3
     ROUTES="town05_spawn218_goal257 town05_spawn239_goal100" ;;
  E) PORT=2040; CUDA=4
     ROUTES="town10hd_spawn0_goal44 town10hd_spawn1_goal63" ;;
  F) PORT=2050; CUDA=5
     ROUTES="town10hd_spawn43_goal100 town01_spawn82_goal200" ;;
  G) PORT=2060; CUDA=6
     ROUTES="town01_spawn115_goal206" ;;
  H) PORT=2070; CUDA=7
     ROUTES="town01_spawn195_goal197" ;;
  *) echo "usage: run_fse_hazard_workers.sh A..H" >&2; exit 2 ;;
esac

GPU="${SCOUT_GPU:-$CUDA}"
ROUTES="${SCOUT_ROUTES:-$ROUTES}"
PORT="${SCOUT_PORT:-$PORT}"
POLICIES="${SCOUT_POLICIES:-random lsa kmnc semantic}"

mkdir -p "$OUT"

for route in $ROUTES; do
  SPEC="$BASE/${route}${SCOUT_BASE_SUFFIX:-_benign_seed0}.json"
  [ -f "$SPEC" ] || SPEC="$BASE/${route}.json"
  for policy in $POLICIES; do
    echo "=== $(date -Is) worker ${WORKER} route ${route} policy ${policy} ==="
    CUDA_VISIBLE_DEVICES="${GPU}" "$PY" "$SEARCH" \
      --policy "$policy" --python-executable "$PY" \
      --seed "${SCOUT_SEED:-13}" --paired-controls \
      --search-space campaign \
      --hazard-search \
      --base-spec "$SPEC" \
      --route-label "$route" \
      --output-dir "$OUT/${route}/${policy}" \
      --evals "$EVALS" \
      --server-port "$PORT" \
      --eval-timeout-seconds "${SCOUT_EVAL_TIMEOUT:-300}" \
      --max-ticks 500 \
      --engine-metrics --oracle "$ORACLE" \
      --cuda-visible-devices "$GPU" --graphics-adapter "$GPU"
  done
  echo "=== $(date -Is) worker ${WORKER} route ${route} controls ==="
  CUDA_VISIBLE_DEVICES="${GPU}" "$PY" "$SEARCH" \
    --policy random --control --python-executable "$PY" \
    --seed "${SCOUT_SEED:-13}" \
    --base-spec "$SPEC" \
    --route-label "$route" \
    --output-dir "$OUT/${route}/control" \
    --evals "$CONTROLS" \
    --server-port "$PORT" \
    --eval-timeout-seconds "${SCOUT_EVAL_TIMEOUT:-300}" \
    --max-ticks 500 \
    --engine-metrics --oracle "$ORACLE" \
    --cuda-visible-devices "$GPU" --graphics-adapter "$GPU"
done

echo "=== worker ${WORKER} DONE $(date -Is) ==="
