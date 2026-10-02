#!/bin/bash
# Collect nominal (no-adversary) Interfuser episodes in parallel and fit the
# activation coverage profile the lsa/kmnc arms need.
#
# At the post-thread-fix speed each episode is ~30-40 s, so a handful run in
# parallel across CARLA ports in well under a minute.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
LOG_DIR="${LOG_DIR:-research/logs}"; mkdir -p "$LOG_DIR"

PY="${SCOUT_PYTHON:-research/.venv/bin/python}"
SPEC="${SCOUT_SPEC:-research/experiments/EXP-020-policy-comparison/artifacts/base_specs/town01_spawn0_goal82_benign_seed0.json}"
OUTDIR="${SCOUT_NOMINAL_OUT:-research/logs/nominal-v4/coverage}"
RUNDIR="${SCOUT_NOMINAL_RUNS:-research/logs/nominal-v4/runs}"
PROFILE="${SCOUT_PROFILE:-research/logs/coverage/if-if-safe-prefix-profile.joblib}"
PORTS="${SCOUT_PORTS:-2010 2020 2030 2040 2050 2060 2070}"

mkdir -p "$OUTDIR" "$RUNDIR" "$(dirname "$PROFILE")"

echo "=== nominal episodes on ports: $PORTS ==="
i=0
for p in $PORTS; do
  CUDA_VISIBLE_DEVICES=0 "$PY" research/harness/run_shakedown.py \
    --scenario-spec "$SPEC" --host 127.0.0.1 --port "$p" \
    --agent-kind pcla --pcla-agent if_if --reload-world --seed "$i" \
    --max-ticks 500 --output-dir "$RUNDIR" \
    --coverage-observer --coverage-trace-output "$OUTDIR" \
    --telemetry-observer --telemetry-output "$RUNDIR/telemetry" \
    > "${LOG_DIR}/nominal4_${p}.log" 2>&1 &
  i=$((i+1))
done
wait

traces=$(ls "$OUTDIR"/*.npz 2>/dev/null | wc -l)
echo "=== collected $traces coverage traces ==="
if [ "$traces" -lt 2 ]; then
  echo "too few traces; check ${LOG_DIR}/nominal4_*.log"; exit 1
fi
"$PY" research/harness/build_coverage_profile.py \
  --input-glob "$OUTDIR/*.npz" --output "$PROFILE"
echo "=== profile written: $PROFILE ==="
