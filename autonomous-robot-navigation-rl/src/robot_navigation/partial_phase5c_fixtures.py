"""Legal score probes and outcome-independent schedule simulation; no learning."""
import numpy as np

from .core import digest
from .partial_contract import MOVES
from .partial_dynamic_env import PartialDynamicEnv, make_scenario
from .partial_memory import AgentMemory
from .partial_sensor import scan
from .partial_observation import encode, validate

HISTORY_ACTIONS=(0,)*4+(3,)*4+(1,)*2+(2,)*2+(1,)*2+(2,)*2+(3,)*2+(2,)*2+(3,)*2


def uniform_schedule(count,seed):
    if type(count) is not int or count<0: raise ValueError('Nonnegative count required')
    return np.random.default_rng(seed).integers(8,size=count).tolist()


def balanced_schedule(count,seed):
    if type(count) is not int or count<0: raise ValueError('Nonnegative count required')
    rng=np.random.default_rng(seed); result=[]
    while len(result)<count: result.extend(rng.permutation(8).tolist())
    return result[:count]


def transform_cell(cell,turns=0,mirror=False):
    r,c=map(int,cell)
    if mirror: c=9-c
    for _ in range(turns%4): r,c=c,9-r
    return [r,c]


def action_permutation(turns=0,mirror=False):
    result=[]
    for dr,dc in MOVES:
        if mirror: dc=-dc
        for _ in range(turns%4): dr,dc=dc,-dr
        result.append(MOVES.index((dr,dc)))
    return result


def transform_observation(observation,turns=0,mirror=False):
    value=validate(observation).copy()
    for lo in (0,2): value[lo:lo+2]=np.array(transform_cell(np.rint(value[lo:lo+2]*9),turns,mirror))/9
    for lo in (8,108,208,308):
        matrix=observation[lo:lo+100].reshape(10,10)
        if mirror: matrix=np.fliplr(matrix)
        value[lo:lo+100]=np.rot90(matrix,-turns%4).ravel()
    rays=[(-1,0),(1,0),(0,1),(0,-1)]
    for i,(dr,dc) in enumerate(rays):
        if mirror: dc=-dc
        for _ in range(turns%4): dr,dc=dc,-dr
        value[4+rays.index((dr,dc))]=observation[4+i]
    return validate(value)


def history_observation(length):
    position=np.array([4,4]); goal=[9,9]; memory=AgentMemory(position)
    blocked=np.zeros((10,10),dtype=bool); reading=scan(blocked,position)
    memory.update(reading,0)
    for step,action in enumerate(HISTORY_ACTIONS[:length],1):
        position=position+MOVES[action]
        if np.any(position<0) or np.any(position>=10) or position.tolist()==goal:
            raise ValueError('Illegal probe history')
        memory.enter(position); reading=scan(blocked,position); memory.update(reading,step)
    return encode(memory,position,goal,reading.distances,length)


def probe_manifest():
    probes=[]
    for r,c in ((4,4),(2,2),(7,7)):
        for distance in (1,2):
            for action,(dr,dc) in enumerate(MOVES[:4]):
                goal=[r+distance*dr,c+distance*dc]
                obs,_=PartialDynamicEnv([make_scenario(['.'*10]*10,[r,c],goal)]).reset(seed=0)
                probes.append({'id':f'reset-{r}-{c}-{distance}-{action}','kind':'reset-goal-distance-translation',
                    'position':[r,c],'goal':goal,'goal_action':action,'distance':distance,'observation':obs.tolist()})
    for length in (8,22):
        original=history_observation(length)
        for mirror in (False,True):
            for turns in range(4):
                obs=transform_observation(original,turns,mirror)
                probes.append({'id':f'history-{length}-{int(mirror)}-{turns}','kind':'D4-history',
                    'base_id':f'history-{length}-0-0','turns':turns,'mirror':mirror,
                    'action_permutation':action_permutation(turns,mirror),'observation':obs.tolist()})
    return {'version':'phase5c-legal-score-probes-v1','new_policy_environment_actions':0,
            'probes':probes,'sha256':digest(probes)}
