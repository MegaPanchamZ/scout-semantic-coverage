#!/usr/bin/env bash
# Full bridge validation: Autoware container (perception off) + stub world + smoke drive. Needs run_carla_native.sh running.
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); ENVX='source /opt/ros/humble/setup.bash; source /opt/autoware/local_setup.bash; export ROS_DOMAIN_ID=7 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'
# CARLA 0.9.15 segfaults when a second bridge session reloads the world on a used server -> always start fresh.
pkill -x 'CarlaUE4-Linux-' 2>/dev/null; sleep 3
setsid nohup "$D/run_carla_native.sh" > /tmp/carla915.log 2>&1 < /dev/null &
until ss -ltn | grep -q ':2000 '; do sleep 2; done; sleep 5
"$D/run_autoware.sh" "${TOWN:-Town01}" $([ "${PERCEPTION:-stub}" = stub ] && echo perception:=false) > /tmp/autoware.log 2>&1 &
until docker exec autoware true 2>/dev/null; do sleep 2; done
until grep -q "Registered sensor: imu" /tmp/autoware.log 2>/dev/null; do sleep 2; done; sleep ${WARMUP_S:-40}
docker cp "$D/stub_perception.py" autoware:/tmp/stub_perception.py; docker cp "$D/smoke_test.py" autoware:/tmp/smoke_test.py
[ "${PERCEPTION:-stub}" = stub ] && docker exec -d autoware bash -c "$ENVX; python3 /tmp/stub_perception.py --ros-args -p use_sim_time:=true > /tmp/stub.log 2>&1"
sleep 20
docker exec -e ENGAGE_WAIT_S=${ENGAGE_WAIT_S:-90} autoware bash -c "$ENVX; python3 /tmp/smoke_test.py"; rc=$?
docker rm -f autoware >/dev/null 2>&1; exit $rc
