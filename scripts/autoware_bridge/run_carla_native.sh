#!/usr/bin/env bash
# Native CARLA 0.9.15 server (docker image's old Vulkan loader crashes on RTX 50xx).
set -euo pipefail
CARLA_ROOT=${CARLA_ROOT:-${HOME}/carla-0.9.15}; PORT=${PORT:-2000}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
exec "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -quality-level=Low -nosound -carla-rpc-port="$PORT" "$@"
