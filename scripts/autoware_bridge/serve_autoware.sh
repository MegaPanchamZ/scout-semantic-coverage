#!/usr/bin/env bash
# Start CARLA 0.9.15 + Autoware (real perception) and leave a persistent scenario service running.
# usage: serve_autoware.sh TOWN          (TOWN as in the specs, e.g. Town01 or Town05_Opt)
# Requests: ~/autoware_data/spool/in/<id>.json   Results: ~/autoware_data/spool/out/<id>.json   Stop: touch spool/in/STOP
# env: AW_ARGS (extra ros2 launch args), EGO_INDEX (initial spawn, default 0), WARMUP_S (default 60)
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); TABLE=$HOME/autoware_data/spawn_points_0.9.15.json; TOWN=$1
ENVX='source /opt/ros/humble/setup.bash; source /opt/autoware/local_setup.bash; export ROS_DOMAIN_ID=7 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'
EGO=$(python3 - "$TOWN" "${EGO_INDEX:-0}" "$TABLE" <<'PY'
import json,sys
t=json.load(open(sys.argv[3]))[sys.argv[1]][int(sys.argv[2])]
print(f'{t["x"]},{t["y"]},{t["z"]+0.3},{t["roll"]},{t["pitch"]},{t["yaw"]}')
PY
)
mkdir -p "$HOME/autoware_data/spool/in" "$HOME/autoware_data/spool/out"
pkill -x 'CarlaUE4-Linux-' 2>/dev/null; sleep 3
setsid nohup "$D/run_carla_native.sh" > /tmp/carla915.log 2>&1 < /dev/null &
until ss -ltn | grep -q ':2000 '; do sleep 2; done; sleep 5
docker rm -f autoware >/dev/null 2>&1
# shellcheck disable=SC2086
"$D/run_autoware.sh" "${TOWN%_Opt}" carla_map:="$TOWN" spawn_point:="$EGO" ${AW_ARGS:-} > /tmp/autoware.log 2>&1 &
until docker exec autoware true 2>/dev/null; do sleep 2; done
until grep -q "Registered sensor: imu" /tmp/autoware.log 2>/dev/null; do sleep 2; done; sleep "${WARMUP_S:-60}"
docker cp "$D/smoke_test.py" autoware:/tmp/smoke_test.py; docker cp "$D/scenario_runner.py" autoware:/tmp/runner.py
docker cp "$TABLE" autoware:/tmp/table.json
docker exec autoware bash -c "$ENVX; python3 /tmp/runner.py --serve /tmp/table.json $TOWN /autoware_data/spool/in /autoware_data/spool/out"
rc=$?
docker rm -f autoware >/dev/null 2>&1; pkill -x 'CarlaUE4-Linux-' 2>/dev/null
exit $rc
