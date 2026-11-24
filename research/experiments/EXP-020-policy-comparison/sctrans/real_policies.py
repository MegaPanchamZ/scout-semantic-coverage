"""Use EXP-020's shared-space policies directly; no local scoring heuristics."""
import hashlib
import importlib.util
from pathlib import Path
import random
import sys

from paths import workspace_root
from variants import DEFAULTS,BOUNDS

ROOT=workspace_root()
POC=ROOT/'research/experiments/EXP-020-policy-comparison/proof-of-concept'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(POC))
spec=importlib.util.spec_from_file_location('sctrans_exp020_policy_search',POC/'policy_search.py')
policy_search=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=policy_search
spec.loader.exec_module(policy_search)
import baselines
from research.harness.search_space import SearchDimension,SearchSpace

if Path(baselines.__file__).resolve() != (POC/'baselines.py').resolve():
    raise ImportError('A different baselines module is already loaded')

ALIASES={'random':'random','criticality':'critonly','scout':'semantic','scout_crit':'semantic_crit'}


def make_space():
    sigmas={'time_scale':.04,'time_offset_s':.1,'lateral_offset_m':.15}
    return SearchSpace([SearchDimension(name=k,spec_path='replay.'+k,
                 low=BOUNDS[k][0],high=BOUNDS[k][1],default=v,sigma=sigmas[k])
                 for k,v in DEFAULTS.items()],name='sctrans-recorded-actor-variants-v1')


def provenance():
    paths=[POC/'policy_search.py',POC/'baselines.py',ROOT/'research/harness/search_space.py',
           ROOT/'research/harness/criticality.py']
    return {'mode':'EXP-020 shared SearchSpace (hazard-search scheduler disabled)',
            'aliases':ALIASES,'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


class RealPolicy:
    def __init__(self,name,seed):
        self.name=ALIASES[name];self.space=make_space();self.rng=random.Random(seed)
        self.state=baselines.make_policy_state(self.name,self.space,policy_search.PolicyState)
        self.seen=set();self.covered=set()

    def next(self):
        candidate=self.state.next_candidate(self.rng,self.seen)
        self.seen.add(self.space.key(candidate))
        return candidate.to_dict()

    def observe(self,values,row):
        self.seen.add(self.space.key(self.space.validate(values)))
        witnessed=set(row.get('covered_obligations') or []) if not row.get('run_error') else set()
        scored=dict(row,engine_run_obligations=sorted(witnessed),
                    engine_new_obligations=len(witnessed-self.covered),engine_run_covered_count=len(witnessed))
        scored['criticality']=policy_search.criticality_score(scored)
        self.state.observe(values,policy_search._semantic_fitness(scored,self.name,engine_active=True),
                           policy_search._semantic_tiebreak(scored,self.name,engine_active=True),
                           run_obligations=witnessed,covered_before=set(self.covered),row=scored)
        self.covered|=witnessed
        return scored
