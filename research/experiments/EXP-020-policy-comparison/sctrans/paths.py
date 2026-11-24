"""Find the SCOUT checkout without depending on the invoking directory."""
import os
from pathlib import Path


def workspace_root():
    candidates=[]
    if os.environ.get('SCOUT_ROOT'):
        candidates.append(Path(os.environ['SCOUT_ROOT']))
    candidates.extend(Path(__file__).resolve().parents)
    for path in candidates:
        if (path/'research/harness/coverage_engine.py').is_file():return path
    raise RuntimeError('Set SCOUT_ROOT to the SCOUT checkout')
