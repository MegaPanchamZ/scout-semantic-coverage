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

WORKER="${1:-}"
if [ -z "$WORKER" ]; then
  echo "usage: run_fse_multimap.sh A or B" >&2
  exit 2
fi

ARTIFACT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ARTIFACT_ROOT" || exit 1

PY="${PYTHON_EXECUTABLE:-research/.venv/bin/python}"
SEARCH=research/experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py
BASE=research/experiments/EXP-020-policy-comparison/artifacts/base_specs
OUT=research/logs/fse_search_hazard
ORACLE=research/experiments/EXP-018-nuscenes-oracle-inventory/artifacts/oracle_inventory_v1.0-trainval.json
EVALS=50
CONTROLS=8

if [ "$WORKER" = "A" ]; then
  PORT=2000
  CUDA=0
  ROUTES="town01_spawn0_goal82 town01_spawn55_goal154 town01_spawn68_goal218 town03_spawn121_goal2 town03_spawn125_goal223 town05_spawn0_goal124 town05_spawn218_goal257"
else
  PORT=2010
  CUDA=1
  ROUTES="town05_spawn239_goal100 town10hd_spawn0_goal44 town10hd_spawn1_goal63 town10hd_spawn43_goal100 town01_spawn82_goal200 town01_spawn115_goal206 town01_spawn195_goal197"
fi

mkdir -p "$OUT"

for route in $ROUTES; do
  for policy in random lsa kmnc semantic; do
    echo "=== $(date -Is) worker ${WORKER} route ${route} policy ${policy} ==="
    CUDA_VISIBLE_DEVICES="${CUDA}" "$PY" "$SEARCH" \
      --policy "$policy" \
      --search-space campaign \
      --hazard-search \
      --base-spec "$BASE/${route}.json" \
      --route-label "$route" \
      --output-dir "$OUT/${route}/${policy}" \
      --evals "$EVALS" \
      --server-port "$PORT" \
      --carla-root "${CARLA_ROOT:?Set CARLA_ROOT to the CARLA installation}" \
      --python-executable "$PY" \
      --max-ticks 500 \
      --engine-metrics --oracle "$ORACLE" \
      --cuda-visible-devices "$CUDA" --graphics-adapter "$CUDA"
  done
  echo "=== $(date -Is) worker ${WORKER} route ${route} controls ==="
  CUDA_VISIBLE_DEVICES="${CUDA}" "$PY" "$SEARCH" \
    --policy random --control \
    --base-spec "$BASE/${route}.json" \
    --route-label "$route" \
    --output-dir "$OUT/${route}/control" \
    --evals "$CONTROLS" \
    --server-port "$PORT" \
      --carla-root "${CARLA_ROOT:?Set CARLA_ROOT to the CARLA installation}" \
      --python-executable "$PY" \
    --max-ticks 500 \
    --engine-metrics --oracle "$ORACLE" \
    --cuda-visible-devices "$CUDA" --graphics-adapter "$CUDA"
done

echo "=== worker ${WORKER} DONE $(date -Is) ==="
