#!/usr/bin/env bash
# One fresh CARLA 0.9.15 server per town (reloading a world on a used server segfaults it).
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-/autoware_data/spawn_points_0.9.15.json}
for T in ${@:-Town01 Town03 Town05 Town10HD}; do
  pkill -x 'CarlaUE4-Linux-' 2>/dev/null; sleep 3
  setsid nohup "$D/run_carla_native.sh" > /tmp/carla915.log 2>&1 < /dev/null &
  until ss -ltn | grep -q ':2000 '; do sleep 2; done; sleep 8
  docker run --rm --network host -v ~/autoware_data:/autoware_data -v "$D":/bridge:ro --entrypoint bash \
    ghcr.io/autowarefoundation/autoware:universe-cuda -c "pip install -q --no-deps /autoware_data/wheels/carla-0.9.15-cp310-cp310-linux_x86_64.whl 2>&1 | grep -v WARNING; python3 /bridge/spawn_points.py $OUT $T"
done
pkill -x 'CarlaUE4-Linux-'
