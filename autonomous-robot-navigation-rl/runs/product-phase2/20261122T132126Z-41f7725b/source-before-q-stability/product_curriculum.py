"""Seeded product-only curriculum and disjoint development layouts (no historical data)."""
from functools import lru_cache
from copy import deepcopy
import numpy as np
import gymnasium as gym
from .core import GridMap,digest
from .partial_dynamic_env import make_scenario,PartialDynamicEnv
from .partial_contract import configuration
KINDS=('short-open','turns','permanent-racks','maze-detour','temporary-wait','temporary-detour')
VERSION='product-curriculum-v2';TRAIN_SEED=82101;DEV_SEED=82901

def _rotate(cell,k):
    r,c=cell
    for _ in range(k):r,c=9-c,r
    return [int(r),int(c)]

def _case(rng,kind,index,split):
    cells=np.full((10,10),'.',dtype='<U1');rotation=int(rng.integers(4));start=None;goal=None;event=None
    if kind=='temporary-wait':
        cells[:,:]='#';row=int(rng.integers(2,8));cells[row,1:9]='.'
        for c in rng.choice(np.arange(1,9),size=3,replace=False):cells[row-1,c]='.'
        start=[row,1];goal=[row,int(rng.integers(4,9))];event=[row,2]
    else:
        if kind in ('permanent-racks','maze-detour'):
            for col in (3,6):
                cells[1:9,col]='#';cells[int(rng.integers(1,9)),col]='.'
        elif kind=='turns':
            row=int(rng.integers(2,8));cells[row,2:8]='#';cells[row,int(rng.integers(2,8))]='.'
        # Storage locations vary genuine geometry hashes, including nominal open floors.
        for _ in range(int(rng.integers(2,9))):cells[int(rng.integers(10)),int(rng.integers(10))]='#'
        if kind=='temporary-detour':
            row=int(rng.integers(2,8));start=[row,1];goal=[row,int(rng.integers(4,9))];cells[row,1:9]='.';event=[row,3]
        else:
            free=np.argwhere(cells=='.');start=list(map(int,free[int(rng.integers(len(free)))]));grid=GridMap('sample',tuple(''.join(x) for x in cells));dist=grid.distances(start)
            limits=(1,3) if kind=='short-open' else (3,7) if kind=='turns' else (6,18)
            options=[p for p,d in dist.items() if limits[0]<=d<=limits[1]]
            if not options:return None
            goal=list(map(int,options[int(rng.integers(len(options)))]))
    rows=[''.join(row) for row in np.rot90(cells,rotation)];start=_rotate(start,rotation);goal=_rotate(goal,rotation)
    events=[] if event is None else [{'id':'product-pallet','cell':_rotate(event,rotation),'start':0,'clear':int(rng.integers(2,13))}]
    try:
        case=make_scenario(rows,start,goal,events,scenario_id=f'product-{split}-{index:03d}',layout_id=f'product-{split}-layout-{index:03d}',event_seed=int(rng.integers(100000,999999)),split='validation' if split=='development' else split,condition=kind)
    except ValueError:return None
    case['product_kind']=kind;case['curriculum_stage']=KINDS.index(kind);case['base_distance']=GridMap('distance',tuple(rows)).distances(start)[tuple(goal)];case['distance_band']='short' if case['base_distance']<=5 else 'longer'
    return case

@lru_cache(maxsize=1)
def _datasets():
    used=set();result={}
    for split,seed,per_kind in [('train',TRAIN_SEED,32),('development',DEV_SEED,4)]:
        rng=np.random.default_rng(seed);cases=[]
        for kind in KINDS:
            for _ in range(per_kind):
                for attempt in range(10000):
                    case=_case(rng,kind,len(cases),split)
                    if case and case['layout_hash'] not in used:break
                else:raise RuntimeError('Could not generate unique reachable layout')
                used.add(case['layout_hash']);cases.append(case)
        result[split]=cases
    return result

def cases(split):return deepcopy(_datasets()[split])

def manifest():
    d=_datasets();return {'version':VERSION,'seeds':{'train':TRAIN_SEED,'development':DEV_SEED},'training_cases':d['train'],'development_cases':d['development'],'training_hash':digest(d['train']),'development_hash':digest(d['development']),'layout_overlap':sorted({c['layout_hash'] for c in d['train']}&{c['layout_hash'] for c in d['development']}),'note':'Fresh product development comparison; development selected, not scientific unseen accuracy. Geometry is used only for generation/reachability, never supplied to learners.'}

class CurriculumEnv(gym.Wrapper):
    def __init__(self,budget,seed,shaping):
        super().__init__(PartialDynamicEnv(cases('train'),configuration()));self.budget=budget;self.actions=0;self.rng=np.random.default_rng(seed);self.shaping=shaping;self.previous=None;self.pool=cases('train')
    def reset(self,*,seed=None,options=None):
        if seed is not None:self.rng=np.random.default_rng(seed)
        fraction=self.actions/max(1,self.budget);stage=0 if fraction<.15 else 2 if fraction<.35 else 3 if fraction<.55 else 5
        pool=[c for c in self.pool if c['curriculum_stage']<=stage];case=pool[int(self.rng.integers(len(pool)))];obs,info=self.env.reset(seed=seed,options={'scenario':case});self.previous=obs.copy();return obs,info
    def step(self,action):
        obs,reward,term,trunc,info=self.env.step(action);old_distance=float(np.abs(self.previous[2:4]-self.previous[:2]).sum()*9);new_distance=float(np.abs(obs[2:4]-obs[:2]).sum()*9)
        shaped=reward+(.2*(old_distance-new_distance) if self.shaping=='progress-v1' else 0.)
        self.previous=obs.copy();self.actions+=1
        return obs,shaped,term,trunc,{**info,'official_reward':reward,'training_shaping':shaped-reward}
