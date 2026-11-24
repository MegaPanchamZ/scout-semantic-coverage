"""Use the upstream CR2ODR converter in a separate process per source map."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from source_replay import load_source


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source_dir',type=Path)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'maps')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for p in sorted(args.source_dir.glob('*.xml')):
        src=load_source(p)
        row={'source':str(p),'source_sha256':src.manifest()['sha256']}
        if not src.goals or src.end_time<2:
            row['status']='excluded_source';rows.append(row);continue
        out=args.output/(p.name+'.xodr')
        with out.with_suffix('.log').open('w') as log:
            try:
                proc=subprocess.run([sys.executable,'-c',
                    'import sys;from crdesigner.map_conversion.opendrive.cr2odr.converter import Converter;Converter(sys.argv[1],center=False).convert(sys.argv[2])',
                    str(p),str(out)],stdout=log,stderr=subprocess.STDOUT,timeout=120)
                row.update(status='converted' if proc.returncode==0 and out.is_file() else 'conversion_failed',
                           exit_code=proc.returncode,map=str(out))
            except subprocess.TimeoutExpired:
                row.update(status='conversion_timeout',map=str(out))
        rows.append(row);print(p.name,row['status'],flush=True)
        (args.output/'conversion_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'conversion_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
