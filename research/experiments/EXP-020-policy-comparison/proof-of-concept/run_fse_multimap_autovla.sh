#!/bin/bash
# FSE'27 AutoVLA campaign: the second ADS.
#
# 12 routes across four maps (Town01 4, Town03 2, Town05 3, Town10HD 3),
# 4 policies x 20 evals per arm + 8 no-adversary controls per route,
# identical 5-D campaign space, paired seeds, engine metrics on.
#
# Usage: run_fse_multimap_autovla.sh {A|B}
#   A -> GPU 0, CARLA port 2000   B -> GPU 1, CARLA port 2010
#
# Resumable: policy_search.py appends rows.jsonl and skips completed rows.

set -u

WORKER="${1:-}"
if [ -z "$WORKER" ]; then
  echo "usage: run_fse_multimap_autovla.sh A or B" >&2
  exit 2
fi

ARTIFACT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ARTIFACT_ROOT" || exit 1

PY="${PYTHON_EXECUTABLE:-research/.venv/bin/python}"
SEARCH=research/experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py
BASE=research/experiments/EXP-020-policy-comparison/artifacts/base_specs
OUT=research/logs/fse_search_autovla
ORACLE=research/experiments/EXP-018-nuscenes-oracle-inventory/artifacts/oracle_inventory_v1.0-trainval.json
PROFILE=research/logs/coverage/autovla-nominal-profile.joblib
AUTOVLA_REPO=research/models/AutoVLA
AUTOVLA_CKPT=research/models/AutoVLA/checkpoints/AutoVLA-hf
EVALS=20
CONTROLS=8

if [ "$WORKER" = "A" ]; then
  PORT=2000
  CUDA=0
  ROUTES="town01_spawn0_goal82 town01_spawn55_goal154 town01_spawn68_goal218 town03_spawn121_goal2 town03_spawn125_goal223 town05_spawn0_goal124"
else
  PORT=2010
  CUDA=1
  ROUTES="town05_spawn218_goal257 town05_spawn239_goal100 town10hd_spawn0_goal44 town10hd_spawn1_goal63 town10hd_spawn43_goal100 town01_spawn115_goal206"
fi

mkdir -p "$OUT"

for route in $ROUTES; do
  for policy in random lsa kmnc semantic; do
    echo "=== $(date -Is) worker ${WORKER} route ${route} policy ${policy} ==="
    CUDA_VISIBLE_DEVICES="${CUDA}" "$PY" "$SEARCH" \
      --policy "$policy" \
      --search-space campaign \
      --base-spec "$BASE/${route}.json" \
      --route-label "$route" \
      --output-dir "$OUT/${route}/${policy}" \
      --evals "$EVALS" \
      --server-port "$PORT" \
      --carla-root "${CARLA_ROOT:?Set CARLA_ROOT to the CARLA installation}" \
      --python-executable "$PY" \
      --max-ticks 300 \
      --agent-kind autovla \
      --agent-repo-path "$AUTOVLA_REPO" \
      --agent-config "$AUTOVLA_CKPT" \
      --coverage-profile "$PROFILE" \
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
    --max-ticks 300 \
    --agent-kind autovla \
    --agent-repo-path "$AUTOVLA_REPO" \
    --agent-config "$AUTOVLA_CKPT" \
    --coverage-profile "$PROFILE" \
    --engine-metrics --oracle "$ORACLE" \
    --cuda-visible-devices "$CUDA" --graphics-adapter "$CUDA"
done

echo "=== worker ${WORKER} DONE $(date -Is) ==="
