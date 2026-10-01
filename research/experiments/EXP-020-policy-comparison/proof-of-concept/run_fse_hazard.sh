#!/bin/bash
# Generate lead-braking base specs (all routes, grouped by town) then run the
# hazard-scheduled campaign: per-template parameter spaces + obligation scheduler.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1
PY="${SCOUT_PYTHON:-research/.venv/bin/python}"
BASE=research/experiments/EXP-020-policy-comparison/artifacts/base_specs
mkdir -p research/logs/fse_search_hazard_v2 research/logs
for route in town01_spawn0_goal82 town01_spawn55_goal154 town01_spawn68_goal218 town01_spawn82_goal200 town01_spawn115_goal206 town01_spawn195_goal197 town03_spawn121_goal2 town03_spawn125_goal223 town05_spawn0_goal124 town05_spawn218_goal257 town05_spawn239_goal100 town10hd_spawn0_goal44 town10hd_spawn1_goal63 town10hd_spawn43_goal100; do
  spec="$BASE/${route}.json"
  [ -f "$BASE/${route}_lead_braking.json" ] && continue
  town=$($PY -c "import json;print(json.load(open('$spec'))['town'])")
  ego=$($PY -c "import json;print(json.load(open('$spec'))['ego_spawn_index'])")
  goal=$($PY -c "import json;print(json.load(open('$spec'))['goal_spawn_index'])")
  $PY research/harness/hazard_search.py --generate-lead-spec --town "$town" --ego-spawn-index "$ego" --goal-spawn-index "$goal" --output "$BASE/${route}_lead_braking.json" >> research/logs/hazard_specgen.log 2>&1
  echo "generated $route" >> research/logs/hazard_specgen.log
done
echo "SPECS_DONE $(ls $BASE/*_lead_braking.json | wc -l)" >> research/logs/hazard_specgen.log
# then launch the two campaign workers
setsid nohup bash research/experiments/EXP-020-policy-comparison/proof-of-concept/run_fse_hazard_workers.sh A > research/logs/fse_search_hazard_v2/worker-A.log 2>&1 < /dev/null &
sleep 2
setsid nohup bash research/experiments/EXP-020-policy-comparison/proof-of-concept/run_fse_hazard_workers.sh B > research/logs/fse_search_hazard_v2/worker-B.log 2>&1 < /dev/null &
