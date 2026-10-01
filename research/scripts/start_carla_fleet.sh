#!/bin/bash
# Start a fleet of headless CARLA servers for parallel campaign workers.
#
# The box has many CPUs but the CARLA/ADS process is CPU-bound, so throughput
# scales with the number of independent (CARLA + ADS) workers rather than with
# a single persistent server. Each worker gets its own CARLA on its own port.
#
# CARLA must run as the non-root `carla` user (it refuses root), so this script
# launches them that way. Ports match the worker layout in the campaign
# launchers (A=2000, B=2010, ...).
#
# Usage:
#   WORKERS="A B C D E F" bash research/scripts/start_carla_fleet.sh
#   QUALITY=Epic bash research/scripts/start_carla_fleet.sh
set -u

CARLA_DIR="${CARLA_DIR:-/opt/carla}"
QUALITY="${QUALITY:-Epic}"
WORKERS="${WORKERS:-A B C D E F}"
CARLA_USER="${CARLA_USER:-carla}"
LOG_DIR="${LOG_DIR:-/root/work}"
VULKAN_ICD="${VULKAN_ICD:-/usr/share/vulkan/icd.d/nvidia_egl_icd.json}"

port_for() {
  case "$1" in
    A) echo 2000 ;; B) echo 2010 ;; C) echo 2020 ;; D) echo 2030 ;;
    E) echo 2040 ;; F) echo 2050 ;; G) echo 2060 ;; H) echo 2070 ;;
    *) echo "" ;;
  esac
}

mkdir -p "$LOG_DIR"

for w in $WORKERS; do
  port="$(port_for "$w")"
  [ -n "$port" ] || { echo "unknown worker $w"; continue; }
  # clear any wedged server on the port
  if ss -ltn 2>/dev/null | grep -q ":$port"; then
    echo "port $port already in use; skipping $w"
    continue
  fi
  log="$LOG_DIR/carla_${w}_${port}.log"
  echo "starting CARLA $w on port $port ($QUALITY) -> $log"
  (cd "$CARLA_DIR" && setsid nohup runuser -u "$CARLA_USER" -- \
      env HOME="/home/$CARLA_USER" XDG_RUNTIME_DIR=/tmp/carla-runtime \
      VK_ICD_FILENAMES="$VULKAN_ICD" \
      ./CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 \
      -RenderOffScreen -quality-level="$QUALITY" -opengl -nosound \
      -carla-rpc-port="$port" -stdout -FullStdOutLogOutput > "$log" 2>&1 < /dev/null &)
done

# wait for each port to accept a connection
python_bin="${SCOUT_PYTHON:-research/.venv/bin/python}"
for w in $WORKERS; do
  port="$(port_for "$w")"
  [ -n "$port" ] || continue
  timeout 180 "$python_bin" - "$port" <<'PY' || echo "worker $w (port $port) NOT READY"
import sys, time, carla
port = int(sys.argv[1]); t0 = time.time()
while time.time() - t0 < 170:
    try:
        c = carla.Client("127.0.0.1", port); c.set_timeout(5)
        print(f"port {port} READY {c.get_world().get_map().name}", flush=True); break
    except Exception:
        time.sleep(3)
else:
    sys.exit(1)
PY
done
echo "=== fleet up: $(ss -ltn | grep -cE ':20[0-7]0') CARLA ports listening ==="
