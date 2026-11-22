"""Fixed historical cohorts and explicit observable-state support; no simulation."""
import hashlib
from collections import Counter

import numpy as np

from .partial_observation import validate
from .partial_phase5c_coverage import coordinates,direction,state_bins


def ray_for_action(action):
    """Raw observation rays are N,S,E,W; actions are U,D,L,R,Wait."""
    if action not in range(5): raise ValueError('Invalid action')
    return (0,1,3,2,None)[action]


def state_hash(observation):
    value=validate(np.asarray(observation,dtype=np.float32))
    return hashlib.sha256(value.tobytes()).hexdigest()


def relation(observation):
    r,c,gr,gc=coordinates(observation)
    return direction((gr-r,gc-c)),int(abs(gr-r)+abs(gc-c))


def select_cohort(steps,per_group=6):
    groups={(d,outcome):[] for d in ('Up','Down','Left','Right') for outcome in ('goal','other-near1')}
    seen={key:set() for key in groups}
    for s in steps:
        d,distance=relation(s['observation'])
        if distance!=1 or d not in ('Up','Down','Left','Right'): continue
        key=(d,'goal' if s['reward']>0 else 'other-near1'); h=state_hash(s['observation'])
        if h in seen[key]: continue
        seen[key].add(h);groups[key].append(s)
    selected=[]; support=[]
    for (d,outcome),values in groups.items():
        n=min(per_group,len(values))
        indices=np.floor(np.linspace(0,len(values)-1,n)).astype(int).tolist() if n else []
        for i in indices:
            s=values[i]
            selected.append({'global_step':s['global_training_interaction'],'direction':d,'cohort':outcome,
                'observation_sha256':state_hash(s['observation']),'next_observation_sha256':state_hash(s['next_observation'])})
        support.append({'direction':d,'cohort':outcome,'distinct_available':len(values),'selected':n})
    return {'selected':selected,'support':support,'window':6000,
            'provenance':'Historical outcome-stratified, deliberately balanced cohort; not population success rate'}


def support_bin(observation):
    o=np.asarray(observation,dtype=np.float32);b=state_bins(o)
    known=o[8:108]!=-1;age=float(np.mean(o[108:208][known])) if known.any() else 0.
    age_bin='zero' if age==0 else '(0,.25]' if age<=.25 else '>.25'
    return tuple(b[k] for k in ('direction','distance','quadrant','boundary','visits','known','elapsed'))+(age_bin,)


def match_states(steps,observations,max_l1=.05):
    matrix=np.asarray([s['observation'] for s in steps],dtype=np.float32) if steps else np.empty((0,409),dtype=np.float32)
    hashes=[state_hash(o) for o in matrix]; bins=[support_bin(o) for o in matrix]
    pairs=[tuple(coordinates(o)) for o in matrix]; result=[]
    for case_id,observation in observations:
        o=np.asarray(observation,dtype=np.float32);h=state_hash(o); b=support_bin(o);p=tuple(coordinates(o))
        exact=[i for i,key in enumerate(hashes) if key==h]
        same_pair=[i for i,key in enumerate(pairs) if key==p]
        coarse=[i for i,key in enumerate(bins) if key==b]
        distances=np.mean(np.abs(matrix[coarse]-o),axis=1) if coarse else np.empty(0)
        near=[coarse[j] for j,v in enumerate(distances) if v<=max_l1]
        result.append({'case_id':case_id,'exact_observation_count':len(exact),'exact_position_goal_count':len(same_pair),
            'observation_sha256':h,'position_goal':list(map(int,p)),
            'observable_bins':list(b),'approximate_mean_raw_l1_ceiling':max_l1,
            'coarse_transition_support':len(coarse),'coarse_distinct_observations':len({hashes[i] for i in coarse}),
            'approximate_transition_support':len(near),'approximate_distinct_observations':len({hashes[i] for i in near}),
            'nearest_mean_raw_l1':float(distances.min()) if len(distances) else None,
            'approximate_global_steps':[steps[i]['global_training_interaction'] for i in near],
            'training_denominator':len(steps),'provenance':'Exact hash versus explicitly approximate support; not a causal match'})
    return result


def check_goal_only_pair(first,second):
    a=validate(np.asarray(first,dtype=np.float32));b=validate(np.asarray(second,dtype=np.float32))
    if not np.array_equal(a[:2],b[:2]) or not np.array_equal(a[4:],b[4:]):
        raise ValueError('Goal-only comparison changed visible history or position')
    for o in (a,b):
        r,c,gr,gc=coordinates(o)
        if (r,c)==(gr,gc) or o[208+10*gr+gc]>0:
            raise ValueError('Swapped goal is current or was already visited in this history')
    return True
