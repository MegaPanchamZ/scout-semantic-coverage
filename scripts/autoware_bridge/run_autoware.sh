#!/usr/bin/env bash
# Autoware Universe (Humble) container + autoware_carla_interface, talking to a CARLA 0.9.15 server on the host.
# IMAGE=...universe (no CUDA) is only usable with perception:=false. AW_GPU picks the GPU (CARLA uses GPU 0).
# usage: run_autoware.sh [TownXX] [extra ros2 launch args...]
set -euo pipefail
TOWN=${1:-Town01}; shift || true
DATA=${DATA:-$HOME/autoware_data}
WHEEL=carla-0.9.15-cp310-cp310-linux_x86_64.whl
IMAGE=${IMAGE:-ghcr.io/autowarefoundation/autoware:universe-cuda}
docker rm -f autoware >/dev/null 2>&1 || true
exec docker run --rm --name autoware --network host --ipc host --gpus "device=${AW_GPU:-1}" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v "$DATA":/autoware_data -v "$(cd "$(dirname "$0")/../.." && pwd)":/repo:ro \
  -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-7} -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  --entrypoint bash "$IMAGE" -c '
    set -e
    pip install -q --no-deps /autoware_data/wheels/'"$WHEEL"'
    source /opt/ros/humble/setup.bash; source /opt/autoware/local_setup.bash
    exec ros2 launch autoware_launch e2e_simulator.launch.xml \
      map_path:=/autoware_data/maps/'"$TOWN"' data_path:=/autoware_data/ml_models \
      vehicle_model:=sample_vehicle sensor_model:=carla_sensor_kit simulator_type:=carla \
      rviz:=false '"$*"'
'
