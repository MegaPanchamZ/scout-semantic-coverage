#!/bin/bash
# FSE'27 gap-driven policy matrix (James review items 1-2).
#
# Benign randomised seed scenarios + hazard/obligation scheduler: the semantic
# arm selects an uncovered obligation, maps it to a scenario template
# (crossing_path -> pedestrian crossing, braking/in_front_of -> lead braking)
# and searches that template's parameter space. All policies start from the
# same benign seed for a matched comparison.
#
# 4 policies x 6 Town01 routes x 50 evals per arm + 8 no-adversary controls per
# route, identical parameter spaces, paired seeds (--seed 13 default), engine
# metrics on (oracle-matched Cov_V/A/E/H + gap-closure logging per row).
#
# Usage: run_fse_matrix.sh {A|B}
#   A -> GPU 0, CARLA port 2000, routes 1-3
#   B -> GPU 1, CARLA port 2010, routes 4-6
#
# Resumable: policy_search.py appends rows.jsonl and skips completed rows.
# Lead-braking specs are generated first unless SCOUT_ENSURE_LEAD_SPECS=0.

set -u

WORKER="${1:-}"
if [ -z "$WORKER" ]; then
  echo "usage: run_fse_matrix.sh A or B" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1

PY="${SCOUT_PYTHON:-research/.venv/bin/python}"
SEARCH=research/experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py
BASE=research/experiments/EXP-020-policy-comparison/artifacts/base_specs
OUT="${SCOUT_OUTPUT_ROOT:-research/logs/fse_search_gap_v1/seed-${SCOUT_SEED:-13}}"
ORACLE=research/experiments/EXP-018-nuscenes-oracle-inventory/artifacts/oracle_inventory_v1.0-trainval.json
EVALS=50
CONTROLS=8
BASE_SUFFIX="${SCOUT_BASE_SUFFIX:-_benign_seed0}"

if [ "$WORKER" = "A" ]; then
  PORT=2000
  CUDA=0
  ROUTES="town01_spawn0_goal82 town01_spawn115_goal206 town01_spawn195_goal197"
else
  PORT=2010
  CUDA=1
  ROUTES="town01_spawn55_goal154 town01_spawn68_goal218 town01_spawn82_goal200"
fi

mkdir -p "$OUT"

# Realise the lead-braking template for braking(vehicle)/in_front_of targets.
if [ "${SCOUT_ENSURE_LEAD_SPECS:-1}" = "1" ]; then
  SCOUT_ROUTES="$ROUTES" CARLA_PORT="$PORT" SCOUT_PYTHON="$PY" \
    bash research/scripts/ensure_lead_braking_specs.sh || true
fi

for route in $ROUTES; do
  SPEC="$BASE/${route}${BASE_SUFFIX}.json"
  [ -f "$SPEC" ] || SPEC="$BASE/${route}.json"
  for policy in random lsa kmnc semantic; do
    echo "=== $(date -Is) worker ${WORKER} route ${route} policy ${policy} ==="
    CUDA_VISIBLE_DEVICES="${CUDA}" "$PY" "$SEARCH" \
      --policy "$policy" --python-executable "$PY" \
      --seed "${SCOUT_SEED:-13}" --paired-controls \
      --search-space campaign \
      --hazard-search \
      --shared-suite \
      --base-spec "$SPEC" \
      --route-label "$route" \
      --output-dir "$OUT/${route}/${policy}" \
      --evals "$EVALS" \
      --server-port "$PORT" \
      --max-ticks 500 \
      --engine-metrics --oracle "$ORACLE" \
      --cuda-visible-devices "$CUDA" --graphics-adapter "$CUDA"
  done
  echo "=== $(date -Is) worker ${WORKER} route ${route} controls ==="
  CUDA_VISIBLE_DEVICES="${CUDA}" "$PY" "$SEARCH" \
    --policy random --control --python-executable "$PY" \
    --seed "${SCOUT_SEED:-13}" \
    --base-spec "$SPEC" \
    --route-label "$route" \
    --output-dir "$OUT/${route}/control" \
    --evals "$CONTROLS" \
    --server-port "$PORT" \
    --max-ticks 500 \
    --engine-metrics --oracle "$ORACLE" \
    --cuda-visible-devices "$CUDA" --graphics-adapter "$CUDA"
done

echo "=== worker ${WORKER} DONE $(date -Is) ==="
