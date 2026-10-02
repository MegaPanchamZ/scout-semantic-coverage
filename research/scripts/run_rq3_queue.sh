#!/bin/bash
# RQ3 safety campaign as a shared job queue over a CARLA fleet.
#
# Units are (seed, route, policy) search arms plus one (seed, route) control
# arm. Controls run with the same execution seeds (seed + index) and the same
# adversary-stripped spec as --paired-controls would, so one control arm per
# (seed, route) pairs every policy's eval i with control i at 1/4 the cost.
# Each worker owns one CARLA port and pulls the next unit under a lock. The
# queue is ordered seed-major, so a partial finish still gives balanced arms.
#
# Usage: PORTS="2000 2010 ..." bash research/scripts/run_rq3_queue.sh
set -u
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

PY="${SCOUT_PYTHON:-research/.venv/bin/python}"
SEARCH=research/experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py
BASE=research/experiments/EXP-020-policy-comparison/artifacts/base_specs
ORACLE=research/experiments/EXP-018-nuscenes-oracle-inventory/artifacts/oracle_inventory_v1.0-trainval.json
OUT="${SCOUT_OUTPUT_ROOT:-research/logs/rq3_if}"
EVALS="${SCOUT_EVALS:-5}"
SEEDS="${SCOUT_SEEDS:-13 14 15}"
ROUTES="${SCOUT_ROUTES:-town01_spawn0_goal82 town01_spawn195_goal197 town01_spawn68_goal218 town01_spawn82_goal200}"
POLICIES="${SCOUT_POLICIES:-semantic random lsa kmnc}"
PORTS="${PORTS:-2000 2010}"
MAX_TICKS="${SCOUT_MAX_TICKS:-400}"
TIMEOUT="${SCOUT_EVAL_TIMEOUT:-900}"
GPU="${SCOUT_GPU:-0}"
RECYCLE_MIB="${SCOUT_RECYCLE_MIB:-7500}"
# CARLA servers are owned by research/scripts/carla_supervisor.sh
export SCOUT_CARLA_SUPERVISED=1

mkdir -p "$OUT"
QUEUE="$OUT/queue.txt"
LOCK="$OUT/queue.lock"
if [ ! -f "$QUEUE" ]; then
  for seed in $SEEDS; do
    for route in $ROUTES; do echo "$seed $route control"; done
    for policy in $POLICIES; do
      for route in $ROUTES; do echo "$seed $route $policy"; done
    done
  done > "$QUEUE"
fi

next_unit() {
  # pop the first line of the queue under an exclusive lock
  (
    flock -x 9
    line="$(head -n 1 "$QUEUE")"
    [ -n "$line" ] && sed -i '1d' "$QUEUE"
    echo "$line"
  ) 9>"$LOCK"
}

server_mib() {
  # GPU memory held by the CARLA server on port $1
  local pids; pids=" $(pgrep -u carla -f "carla-rpc-port=${1}( |$)" | tr '\n' ' ') "
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits |
    awk -F', ' -v p="$pids" 'index(p, " "$1" ") {s += $2} END {print s + 0}'
}

worker() {
  local port="$1"
  while :; do
    local unit; unit="$(next_unit)"
    [ -n "$unit" ] || break
    # CARLA's GPU memory grows over hours; past RECYCLE_MIB eight servers plus
    # their agents no longer fit and agents fail with CUDA OOM. Recycle the
    # server between units; carla_supervisor.sh restarts and warms it.
    if [ "$(server_mib "$port")" -gt "$RECYCLE_MIB" ]; then
      echo "=== $(date -Is) port ${port} RECYCLE server ($(server_mib "$port") MiB) ==="
      pkill -u carla -f "carla-rpc-port=${port}( |$)"
      sleep 15
    fi
    read -r seed route policy <<<"$unit"
    local spec="$BASE/${route}_benign_seed0.json"
    local dir="$OUT/seed-${seed}/${route}/${policy}"
    local extra=(--policy "$policy" --hazard-search --search-space campaign)
    [ "$policy" = control ] && extra=(--policy random --control)
    mkdir -p "$(dirname "$dir")"
    echo "=== $(date -Is) port ${port} START ${unit} ==="
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$SEARCH" "${extra[@]}" \
      --python-executable "$PY" --seed "$seed" \
      --base-spec "$spec" --route-label "$route" --output-dir "$dir" \
      --evals "$EVALS" --server-port "$port" \
      --eval-timeout-seconds "$TIMEOUT" --max-ticks "$MAX_TICKS" \
      --engine-metrics --oracle "$ORACLE" \
      --cuda-visible-devices "$GPU" --graphics-adapter "$GPU" \
      > "$dir.log" 2>&1 || echo "=== port ${port} FAILED ${unit} (see $dir.log) ==="
    if grep -q "server unavailable\|did not become ready" "$dir/rows.jsonl" 2>/dev/null; then
      # a server crash ate evals: drop rows from the first failed eval on and
      # resume the arm later (earlier evals and the search state survive)
      "$PY" - "$dir" <<'EOF'
import json, shutil, sys
from pathlib import Path
d = Path(sys.argv[1]); rows_path = d / "rows.jsonl"
keep = []
for line in rows_path.read_text().splitlines():
    if "server unavailable" in line or "did not become ready" in line:
        break
    keep.append(line)
rows_path.write_text("".join(l + "\n" for l in keep))
for e in (d / "evaluations").glob("*-*"):
    if e.name.rsplit("-", 1)[1].isdigit() and int(e.name.rsplit("-", 1)[1]) >= len(keep):
        shutil.rmtree(e)
EOF
      ( flock -x 9; echo "$unit" >> "$QUEUE" ) 9>"$LOCK"
      echo "=== port ${port} REQUEUED ${unit} (server failure) ==="
    fi
    echo "=== $(date -Is) port ${port} DONE ${unit} ==="
  done
}

for port in $PORTS; do
  mkdir -p "$OUT"
  worker "$port" &
  sleep 20   # stagger startups so model loads don't collide
done
wait
echo "=== RQ3 queue DONE $(date -Is) ==="
