#!/usr/bin/env bash
# Run several harness ScenarioSpecs (ONE town) against Autoware (real perception) in ONE CARLA 0.9.15 + Autoware session.
# usage: run_session.sh OUTDIR SPEC.json [SPEC.json ...]
# env:   AW_ARGS="lidar_detection_model:=centerpoint/centerpoint"  extra `ros2 launch` args (e.g. pick the base CenterPoint)
#        WARMUP_S (default 60), ENGAGE_WAIT_S, SPEC_DT
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); TABLE=$HOME/autoware_data/spawn_points_0.9.15.json
OUT=$(readlink -f "$1"); shift; mkdir -p "$OUT"
SPECS=(); for s in "$@"; do SPECS+=("$(readlink -f "$s")"); done
ENVX='source /opt/ros/humble/setup.bash; source /opt/autoware/local_setup.bash; export ROS_DOMAIN_ID=7 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'
read -r TOWN EGO < <(python3 - "${SPECS[0]}" "$TABLE" <<'PY'
import json,sys
s=json.load(open(sys.argv[1])); t=json.load(open(sys.argv[2]))[s["town"]][s["ego_spawn_index"]]
print(s["town"], f'{t["x"]},{t["y"]},{t["z"]+0.3},{t["roll"]},{t["pitch"]},{t["yaw"]}')
PY
)
pkill -x 'CarlaUE4-Linux-' 2>/dev/null; sleep 3          # 0.9.15 crashes on a 2nd world load: always a fresh server
setsid nohup "$D/run_carla_native.sh" > /tmp/carla915.log 2>&1 < /dev/null &
until ss -ltn | grep -q ':2000 '; do sleep 2; done; sleep 5
docker rm -f autoware >/dev/null 2>&1
# shellcheck disable=SC2086
"$D/run_autoware.sh" "${TOWN%_Opt}" carla_map:="$TOWN" spawn_point:="$EGO" ${AW_ARGS:-} > /tmp/autoware.log 2>&1 &
until docker exec autoware true 2>/dev/null; do sleep 2; done
until grep -q "Registered sensor: imu" /tmp/autoware.log 2>/dev/null; do sleep 2; done; sleep "${WARMUP_S:-60}"
docker cp "$D/smoke_test.py" autoware:/tmp/smoke_test.py; docker cp "$D/${RUNNER:-scenario_runner.py}" autoware:/tmp/runner.py
docker cp "$TABLE" autoware:/tmp/table.json
docker exec autoware mkdir -p /tmp/specs /tmp/out
I=0; INS=(); for s in "${SPECS[@]}"; do I=$((I+1)); docker cp "$s" autoware:/tmp/specs/$(printf %04d $I).json; INS+=("/tmp/specs/$(printf %04d $I).json"); done
docker exec autoware bash -c "$ENVX; python3 /tmp/runner.py /tmp/table.json /tmp/out ${INS[*]}"; rc=$?
docker cp autoware:/tmp/out/. "$OUT"/ 2>/dev/null
docker rm -f autoware >/dev/null 2>&1; pkill -x 'CarlaUE4-Linux-' 2>/dev/null
exit $rc
