#!/bin/bash
# Ensure every route has a lead-braking base spec, so the gap-driven scheduler
# can realise braking(vehicle) / in_front_of(vehicle, ego) targets and not just
# pedestrian crossings.
#
# Lead-braking specs bake the route polyline sampled from a live CARLA world, so
# this connects to a running server and may change its loaded map. Run it on a
# CARLA server reserved for preparation, BEFORE starting campaign workers.
#
# Usage:
#   CARLA_PORT=2000 bash research/scripts/ensure_lead_braking_specs.sh
#   SCOUT_ROUTES="town01_spawn0_goal82 town01_spawn55_goal154" \
#     CARLA_PORT=2010 bash research/scripts/ensure_lead_braking_specs.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

PY="${SCOUT_PYTHON:-research/.venv/bin/python}"
BASE=research/experiments/EXP-020-policy-comparison/artifacts/base_specs
HOST="${CARLA_HOST:-127.0.0.1}"
PORT="${CARLA_PORT:-2000}"
LOG="${SCOUT_LEAD_SPEC_LOG:-research/logs/lead_braking_specgen.log}"
mkdir -p "$(dirname "$LOG")"

if [ -n "${SCOUT_ROUTES:-}" ]; then
  ROUTES="$SCOUT_ROUTES"
else
  # Every base route spec, excluding benign variants and already-generated leads.
  ROUTES=""
  for f in "$BASE"/*.json; do
    b="$(basename "$f" .json)"
    case "$b" in
      *_benign_seed*|*_lead_braking|benign_seeds) continue ;;
    esac
    ROUTES="$ROUTES $b"
  done
fi

echo "=== ensuring lead-braking specs (host ${HOST}:${PORT}) at $(date -Is) ===" | tee -a "$LOG"
generated=0
for route in $ROUTES; do
  spec="$BASE/${route}.json"
  [ -f "$spec" ] || { echo "skip ${route}: no base spec" | tee -a "$LOG"; continue; }
  [ -f "$BASE/${route}_lead_braking.json" ] && continue
  town="$($PY -c "import json;print(json.load(open('$spec'))['town'])")"
  ego="$($PY -c "import json;print(json.load(open('$spec'))['ego_spawn_index'])")"
  goal="$($PY -c "import json;print(json.load(open('$spec'))['goal_spawn_index'])")"
  if "$PY" research/harness/hazard_search.py --generate-lead-spec \
      --host "$HOST" --port "$PORT" --town "$town" \
      --ego-spawn-index "$ego" --goal-spawn-index "$goal" \
      --output "$BASE/${route}_lead_braking.json" >> "$LOG" 2>&1; then
    echo "generated ${route}_lead_braking.json" | tee -a "$LOG"
    generated=$((generated + 1))
  else
    echo "FAILED ${route} (see $LOG)" | tee -a "$LOG"
  fi
done
echo "=== lead-braking specs: generated ${generated}; present $(ls "$BASE"/*_lead_braking.json 2>/dev/null | wc -l) ===" | tee -a "$LOG"
