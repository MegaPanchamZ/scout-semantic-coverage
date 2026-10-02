#!/bin/bash
# Frozen-seed ADS validation.
# Repeats one identical benign scenario N times with no search and reports
# collision/goal rates. Run before any full campaign: if collisions stay near
# ~50% on a benign frozen seed, debug the ADS/CARLA bridge first.
#
# Usage: bash research/scripts/validate_frozen_seed.sh <spec.json> [repeats] [port] [cuda]
set -u
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKSPACE_ROOT" || exit 1
PY="${SCOUT_PYTHON:-research/.venv/bin/python}"
SPEC="${1:?usage: validate_frozen_seed.sh <spec.json> [repeats] [port] [cuda]}"
REPEATS="${2:-20}"
PORT="${3:-2000}"
CUDA="${4:-0}"
OUT="${SCOUT_OUTPUT_DIR:-research/logs/frozen_seed_validation/$(basename "$SPEC" .json)}"
SEARCH=research/experiments/EXP-020-policy-comparison/proof-of-concept/policy_search.py

# Optional ADS selection. Defaults to the InterFuser/PCLA path used by the
# campaign. For AutoVLA set SCOUT_AGENT_KIND=autovla and point
# SCOUT_AGENT_REPO/SCOUT_AGENT_CONFIG at the converted checkpoint.
AGENT_ARGS=()
if [ -n "${SCOUT_AGENT_KIND:-}" ]; then AGENT_ARGS+=(--agent-kind "$SCOUT_AGENT_KIND"); fi
if [ -n "${SCOUT_AGENT_REPO:-}" ]; then AGENT_ARGS+=(--agent-repo-path "$SCOUT_AGENT_REPO"); fi
if [ -n "${SCOUT_AGENT_CONFIG:-}" ]; then AGENT_ARGS+=(--agent-config "$SCOUT_AGENT_CONFIG"); fi

echo "=== frozen-seed validation: $SPEC x$REPEATS on GPU $CUDA port $PORT (agent ${SCOUT_AGENT_KIND:-pcla}) ==="
CUDA_VISIBLE_DEVICES="$CUDA" "$PY" "$SEARCH" \
  --policy random --frozen-base \
  --python-executable "$PY" \
  --base-spec "$SPEC" \
  --route-label "$(basename "$SPEC" .json)" \
  --output-dir "$OUT" \
  --evals "$REPEATS" \
  --server-port "$PORT" \
  --max-ticks 500 \
  --seed "${SCOUT_SEED:-0}" \
  "${AGENT_ARGS[@]}" \
  --cuda-visible-devices "$CUDA" --graphics-adapter "$CUDA"

"$PY" - "$OUT/rows.jsonl" <<'PYEOF'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
n = len(rows)
coll = sum(1 for r in rows if (r.get("collision_count") or 0) > 0)
goal = sum(1 for r in rows if r.get("reached_goal"))
err = sum(1 for r in rows if (r.get("run_error") or "").strip())
rate = 100 * coll / max(n, 1)
print(f"frozen-seed validation: {n} runs | collisions {coll} ({rate:.0f}%) | reached_goal {goal} | errors {err}")
if rate >= 40:
    print("WARNING: collision rate on a benign frozen seed is high; debug the ADS/CARLA bridge before full experiments.")
PYEOF
