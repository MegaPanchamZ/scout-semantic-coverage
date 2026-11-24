#!/usr/bin/env python3
"""Audit all public sources, then run each usable source once, sequentially."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from source_replay import load_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--maps',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=2510)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):p.error('Use a fresh output directory')
    args.output.mkdir(parents=True,exist_ok=True);rows=[]
    for source_path in sorted(args.sources.glob('*.xml')):
        source=load_source(source_path);row=source.manifest();reasons=[]
        if not source.goals:reasons.append('missing_spatial_goal')
        if source.end_time<2:reasons.append('recorded_trajectory_shorter_than_2s')
        map_path=args.maps/(source_path.name+'.xodr')
        if not reasons and not map_path.exists():reasons.append('missing_converted_map')
        if reasons:
            row.update(status='excluded_source',reasons=reasons)
        else:
            out=args.output/source_path.stem;out.mkdir()
            command=[sys.executable,str(Path(__file__).with_name('validate_scene.py')),
                     '--source',str(source_path),'--map',str(map_path),
                     '--output',str(out),'--port',str(args.port)]
            try:
                with (out/'process.log').open('w') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                row.update(exit_code=result.returncode,result=str(out/'result.json'))
                if (out/'result.json').exists():
                    result_row=json.loads((out/'result.json').read_text())
                    row.update(status='admitted' if result_row['validation']['passed'] else 'excluded_validation',
                               reasons=result_row['validation']['reasons'])
                else:row.update(status='execution_failed',reasons=['missing_result'])
            except subprocess.TimeoutExpired:
                row.update(status='execution_failed',reasons=['timeout'])
        rows.append(row)
        (args.output/'summary.json').write_text(json.dumps({'sources':rows,'admitted':sum(r['status']=='admitted' for r in rows),
              'comparison_evaluations':0,'note':'Validation only. No policy comparison or automatic budget extension.'},indent=2)+'\n')
        print(source_path.name,row['status'],row['reasons'],flush=True)
    if not rows:p.error('No source XML files found')


if __name__=='__main__':main()
