#!/bin/bash
# Keep a CARLA fleet alive during a campaign.
#
# CARLA 0.9.16 segfaults now and then. When a server's port stops listening,
# this restarts it as the carla user (start_carla_fleet.sh) and then runs one
# throwaway control episode on it before releasing it: a cold server's first
# episode is not representative (Interfuser ran into static objects on the
# first run of every freshly started server). While that happens,
# /tmp/carla_warming_<port> exists and workers running with
# SCOUT_CARLA_SUPERVISED=1 wait for it to disappear.
#
# Usage: WORKERS="A B C D E F G H" bash research/scripts/carla_supervisor.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

PY="${SCOUT_PYTHON:-research/.venv/bin/python}"
WORKERS="${WORKERS:-A B}"
WARMUP_SPEC="${WARMUP_SPEC:-research/experiments/EXP-020-policy-comparison/artifacts/base_specs/town01_spawn0_goal82_benign_seed0.json}"
SEARCH=research/experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py

port_for() {
  case "$1" in
    A) echo 2000 ;; B) echo 2010 ;; C) echo 2020 ;; D) echo 2030 ;;
    E) echo 2040 ;; F) echo 2050 ;; G) echo 2060 ;; H) echo 2070 ;;
  esac
}

warm() {
  local w="$1" port="$2"
  local flag="/tmp/carla_warming_${port}"
  touch "$flag"
  echo "$(date -Is) port $port down; restarting"
  # SIGKILL: a hung server can ignore SIGTERM and keep its port open
  pkill -9 -u carla -f "carla-rpc-port=${port}( |$)" 2>/dev/null
  sleep 5
  WORKERS="$w" bash research/scripts/start_carla_fleet.sh >/dev/null 2>&1
  rm -rf "/tmp/carla_warmup_${port}"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 "$PY" "$SEARCH" --policy random --control \
    --python-executable "$PY" --seed 999 --base-spec "$WARMUP_SPEC" --route-label warmup \
    --output-dir "/tmp/carla_warmup_${port}" --evals 1 --server-port "$port" \
    --eval-timeout-seconds 600 --max-ticks 250 --cuda-visible-devices 0 --graphics-adapter 0 \
    >/dev/null 2>&1
  rm -f "$flag"
  echo "$(date -Is) port $port restarted and warmed"
}

probe() {
  # a hung server keeps listening but stops answering RPCs; count consecutive
  # failed probes and restart it after three (about three minutes)
  local w="$1" port="$2" fails="/tmp/carla_probe_fails_${port}"
  if timeout 45 "$PY" -c "import carla; c = carla.Client('127.0.0.1', ${port}); c.set_timeout(30.0); c.get_world().get_map().name" >/dev/null 2>&1; then
    rm -f "$fails"
    return
  fi
  echo x >> "$fails"
  if [ "$(wc -l < "$fails")" -ge 3 ] && [ ! -f "/tmp/carla_warming_${port}" ]; then
    rm -f "$fails"
    echo "$(date -Is) port $port hung (3 failed probes)"
    warm "$w" "$port"
  fi
}

tick=0
while :; do
  for w in $WORKERS; do
    port="$(port_for "$w")"
    if ! ss -ltn | grep -q ":${port} " && [ ! -f "/tmp/carla_warming_${port}" ]; then
      warm "$w" "$port" &
    elif [ $((tick % 6)) -eq 0 ] && [ ! -f "/tmp/carla_warming_${port}" ]; then
      probe "$w" "$port" &
    fi
  done
  tick=$((tick + 1))
  sleep 10
done
