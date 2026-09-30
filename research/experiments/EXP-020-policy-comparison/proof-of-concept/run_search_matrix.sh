#!/bin/bash
# Matched-budget policy search matrix for EXP-020.
#
# Usage: run_search_matrix.sh {A|B}
#   A -> GPU 0, CARLA port 2000, first four routes
#   B -> GPU 1, CARLA port 2010, last four routes
#
# Each job is resumable: policy_search.py appends rows.jsonl and skips
# already-completed rows on restart. All four policies run the identical
# four-dimensional campaign space (--search-space campaign).

set -u

WORKER="${1:-}"
if [ -z "$WORKER" ]; then
  echo "usage: run_search_matrix.sh A or B" >&2
  exit 2
fi
PY=/mnt/DevDrive/development/MRES/research/.venv/bin/python
SEARCH=/mnt/DevDrive/development/MRES/research/experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py
BASE=/mnt/DevDrive/development/MRES/research/experiments/EXP-020-policy-comparison/artifacts/base_specs
OUT=/mnt/DevDrive/development/MRES/research/logs/policy_search
EVALS_PER_POLICY=30
CONTROLS=8

cd /mnt/DevDrive/development/MRES || exit 1

if [ "$WORKER" = "A" ]; then
  PORT=2000
  CUDA=0
  ROUTES="town01_spawn0_goal82 town01_spawn115_goal206 town01_spawn195_goal197 town01_spawn55_goal154"
else
  PORT=2010
  CUDA=1
  ROUTES="town01_spawn68_goal218 town01_spawn82_goal200 town02_spawn40_goal33 town02_spawn98_goal35"
fi

for route in $ROUTES; do
  for policy in random lsa kmnc semantic; do
    echo "=== $(date -Is) worker ${WORKER} route ${route} policy ${policy} ==="
    CUDA_VISIBLE_DEVICES="${CUDA}" "$PY" "$SEARCH" \
      --policy "$policy" \
      --search-space campaign \
      --base-spec "$BASE/${route}.json" \
      --route-label "$route" \
      --output-dir "$OUT/${route}/${policy}" \
      --evals "$EVALS_PER_POLICY" \
      --server-port "$PORT" \
      --max-ticks 500 \
      --cuda-visible-devices "$CUDA" --graphics-adapter "$CUDA"
  done
  echo "=== $(date -Is) worker ${WORKER} route ${route} controls ==="
  CUDA_VISIBLE_DEVICES="${CUDA}" "$PY" "$SEARCH" \
    --policy random \
    --base-spec "$BASE/${route}.json" \
    --route-label "$route" \
    --output-dir "$OUT/${route}/control" \
    --evals "$CONTROLS" \
    --server-port "$PORT" \
    --max-ticks 500 \
    --cuda-visible-devices "$CUDA" --graphics-adapter "$CUDA" \
    --control
done

echo "=== worker ${WORKER} DONE $(date -Is) ==="
