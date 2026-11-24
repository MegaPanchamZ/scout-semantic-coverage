#!/usr/bin/env python3
"""Five evaluations per EXP-020 arm, gated by an admitted recorded baseline.

One admitted eval 0 is shared verbatim across arms (one observation each), then
four fresh-process evaluations per arm. Search variants are explicitly labelled
source-derived. This command never extends a budget automatically.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from paths import workspace_root
from source_replay import validation_verdict,load_source
from variants import DEFAULTS


class AdmissionError(ValueError):pass


def admitted(path):
    row=json.loads(path.read_text());verdict=validation_verdict(row)
    if not verdict['passed']:raise AdmissionError('Baseline is not admitted: '+', '.join(verdict['reasons']))
    if row.get('candidate') is not None:raise AdmissionError('Admission requires an unperturbed replay')
    source=Path(row['source']['source']);map_path=Path(row['map'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=row['source']['sha256']:
        raise AdmissionError('Source changed since validation')
    if hashlib.sha256(map_path.read_bytes()).hexdigest()!=row['map_sha256']:
        raise AdmissionError('Converted map changed since validation')
    root=workspace_root()
    checks={'validate_scene.py':Path(__file__).with_name('validate_scene.py'),
            'source_replay.py':Path(__file__).with_name('source_replay.py'),
            'variants.py':Path(__file__).with_name('variants.py'),
            'pcla_bridge.py':root/'research/harness/pcla_bridge.py',
            'coverage_engine.py':root/'research/harness/coverage_engine.py',
            'semantic.py':root/'research/harness/observers/semantic.py',
            'interfuser_agent.py':root/'research/models/PCLA/pcla_agents/interfuser/interfuser_agent.py'}
    for name,p in checks.items():
        if row['code_sha256'].get(name)!=hashlib.sha256(p.read_bytes()).hexdigest():
            raise AdmissionError('Code changed since validation: '+name)
    return row


def campaign(args):
    # Fail before loading policies, launching evaluations, or creating a campaign.
    baseline=admitted(args.admission_result)
    from real_policies import RealPolicy,provenance,ALIASES
    from research.harness.criticality import criticality_score
    oracle=args.oracle or workspace_root()/'research/experiments/EXP-018-nuscenes-oracle-inventory/artifacts/oracle_inventory_v1.0-trainval.json'
    if hashlib.sha256(oracle.read_bytes()).hexdigest()!=baseline['oracle_sha256']:
        raise AdmissionError('Oracle changed since validation')
    witness=baseline['intended_witnesses'][0]['witness_actor'].split('#')[-1]
    actor_id=next(k for k,v in baseline['actor_mapping'].items() if str(v['carla_id'])==witness)
    protocol={'budget_per_arm':5,'eval_0':'shared admitted baseline; counted in each arm',
              'eval_0_result_sha256':hashlib.sha256(args.admission_result.read_bytes()).hexdigest(),
              'seed':args.seed,'mutated_source_actor':actor_id,'source':baseline['source'],
              'policies':provenance(),'space':RealPolicy('random',args.seed).space.to_dict(),
              'variants_fidelity':'source-derived perturbations; eval 0 is the original recording',
              'budget_extensions':[]}
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use a fresh output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    all_rows=[]
    for name in ALIASES:
        policy=RealPolicy(name,args.seed)
        with (args.output/(name+'.jsonl')).open('w') as log:
            for index in range(5):
                values=dict(DEFAULTS) if index==0 else policy.next()
                if index==0:
                    row=dict(baseline,shared_eval_0=str(args.admission_result.resolve()))
                else:
                    output=args.output/name/f'eval{index:02d}';output.mkdir(parents=True)
                    candidate=output/'candidate.json'
                    candidate.write_text(json.dumps(dict(values,actor_id=actor_id),indent=2)+'\n')
                    command=[sys.executable,str(Path(__file__).with_name('validate_scene.py')),
                             '--source',baseline['source']['source'],'--map',baseline['map'],
                             '--output',str(output),'--port',str(args.port),'--candidate',str(candidate),
                             '--oracle',str(oracle)]
                    if baseline.get('route_compatibility_fix'):command.append('--safe-route-topology')
                    try:
                        with (output/'process.log').open('w') as process_log:
                            result=subprocess.run(command,stdout=process_log,stderr=subprocess.STDOUT,timeout=300)
                        result_path=output/'result.json'
                        row=json.loads(result_path.read_text()) if result_path.exists() else {'run_error':'missing_result','exit_code':result.returncode}
                    except subprocess.TimeoutExpired:
                        row={'run_error':'evaluation_timeout'}
                    # Collision, missing hazard, or an ego stop are outcomes of search.
                    # Broken replay/placement is explicitly invalid and earns no coverage.
                    if not row.get('run_error') and (row.get('spawn_failures') or row.get('max_replay_error_m',float('inf'))>.25 or row.get('max_replay_yaw_error_deg',float('inf'))>5):
                        row['run_error']='invalid_replay'
                row=policy.observe(values,row)
                row.update(policy=name,implementation=ALIASES[name],eval_index=index,candidate=values,
                           source_actor_id=actor_id)
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();all_rows.append(row)
                print(name,index,'collision',row.get('collision_count'),'error',row.get('run_error'),flush=True)
                if row.get('run_error'):
                    # A broken simulator/replay must not silently seed later arms.
                    raise RuntimeError('Campaign stopped after execution failure; preserve partial evidence')
    summary={name:{'evals':5,'sensor_collision_evals':sum(bool(r.get('collision_count')) for r in all_rows if r['policy']==name),
                    'covered_obligations':sorted(set().union(*(set(r.get('covered_obligations') or []) for r in all_rows if r['policy']==name)))}
             for name in ALIASES}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--admission-result',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=2510);p.add_argument('--seed',type=int,default=20)
    p.add_argument('--oracle',type=Path)
    try:campaign(p.parse_args())
    except AdmissionError as e:p.exit(2,str(e)+'\n')


if __name__=='__main__':main()
