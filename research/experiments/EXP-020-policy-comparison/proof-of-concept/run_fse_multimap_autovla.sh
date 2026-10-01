#!/bin/bash
# FSE'27 AutoVLA campaign: the second ADS (gap-driven protocol).
#
# 12 routes across four maps (Town01 4, Town03 2, Town05 3, Town10HD 3), benign
# randomised seeds and the hazard/obligation scheduler enabled (review items 1-2).
# 4 policies x 20 evals per arm + 8 no-adversary controls per route, paired seeds,
# engine metrics on.
#
# Usage: run_fse_multimap_autovla.sh {A|B}
#   A -> GPU 0, CARLA port 2000   B -> GPU 1, CARLA port 2010
#
# Resumable: policy_search.py appends rows.jsonl and skips completed rows.
# Lead-braking specs are generated first unless SCOUT_ENSURE_LEAD_SPECS=0.

set -u

WORKER="${1:-}"
if [ -z "$WORKER" ]; then
  echo "usage: run_fse_multimap_autovla.sh A or B" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1

PY="${SCOUT_PYTHON:-research/.venv/bin/python}"
SEARCH=research/experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py
BASE=research/experiments/EXP-020-policy-comparison/artifacts/base_specs
OUT="${SCOUT_OUTPUT_ROOT:-research/logs/fse_search_autovla_gap_v1/seed-${SCOUT_SEED:-13}}"
ORACLE=research/experiments/EXP-018-nuscenes-oracle-inventory/artifacts/oracle_inventory_v1.0-trainval.json
PROFILE="${SCOUT_PROFILE:-research/logs/coverage/autovla-nominal-profile-v2.joblib}"
AUTOVLA_REPO="${SCOUT_AUTOVLA_REPO:-research/models/AutoVLA}"
AUTOVLA_CKPT="${SCOUT_AUTOVLA_CHECKPOINT:-$AUTOVLA_REPO/checkpoints/AutoVLA-hf}"
BASE_SUFFIX="${SCOUT_BASE_SUFFIX:-_benign_seed0}"
# Optional remote serving backend (torch is the default, in-process path).
AUTOVLA_ARGS=()
if [ -n "${SCOUT_AUTOVLA_BACKEND:-}" ]; then
  AUTOVLA_ARGS+=(--autovla-backend "$SCOUT_AUTOVLA_BACKEND")
  if [ -n "${SCOUT_AUTOVLA_ENDPOINT:-}" ]; then
    AUTOVLA_ARGS+=(--autovla-endpoint "$SCOUT_AUTOVLA_ENDPOINT")
  fi
fi
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
      --search-space campaign --hazard-search \
      --base-spec "$SPEC" \
      --route-label "$route" \
      --output-dir "$OUT/${route}/${policy}" \
      --evals "$EVALS" \
      --server-port "$PORT" \
      --max-ticks 500 \
      --agent-kind autovla \
      --agent-repo-path "$AUTOVLA_REPO" \
      --agent-config "$AUTOVLA_CKPT" \
      ${AUTOVLA_ARGS[@]+"${AUTOVLA_ARGS[@]}"} \
      --coverage-profile "$PROFILE" \
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
    --agent-kind autovla \
    --agent-repo-path "$AUTOVLA_REPO" \
    --agent-config "$AUTOVLA_CKPT" \
    ${AUTOVLA_ARGS[@]+"${AUTOVLA_ARGS[@]}"} \
    --coverage-profile "$PROFILE" \
    --engine-metrics --oracle "$ORACLE" \
    --cuda-visible-devices "$CUDA" --graphics-adapter "$CUDA"
done

echo "=== worker ${WORKER} DONE $(date -Is) ==="
