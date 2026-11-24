"""Explicit source-derived perturbations; identity is the recorded baseline.

These are search parameters, never claimed to be the original recorded scene.
Time outside the actor's recorded interval always returns None.
"""
import math
from source_replay import State

DEFAULTS={'time_scale':1.0,'time_offset_s':0.0,'lateral_offset_m':0.0}
BOUNDS={'time_scale':(.85,1.15),'time_offset_s':(-.3,.3),'lateral_offset_m':(-.5,.5)}


def validate_candidate(candidate,source):
    if set(candidate)!=set(DEFAULTS)|{'actor_id'}:raise ValueError('Invalid candidate fields')
    if str(candidate['actor_id']) not in {a.actor_id for a in source.actors}:
        raise ValueError('Candidate actor is absent from source')
    for k,(low,high) in BOUNDS.items():
        value=candidate[k]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError('Invalid candidate '+k)
    return candidate


def sample_actor(actor,t,candidate=None):
    if candidate is None or actor.actor_id!=str(candidate['actor_id']):return actor.at(t)
    state=actor.at(t*candidate['time_scale']+candidate['time_offset_s'])
    if state is None:return None
    offset=candidate['lateral_offset_m']
    return State(t,state.x-offset*math.sin(state.yaw),state.y+offset*math.cos(state.yaw),
                 state.yaw,state.speed*candidate['time_scale'])
