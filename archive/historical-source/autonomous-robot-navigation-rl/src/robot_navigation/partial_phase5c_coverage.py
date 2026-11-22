"""Historical counts only. No learner update, action selection or env.step."""
from collections import Counter,defaultdict
import hashlib

import numpy as np

from .partial_contract import ACTION_NAMES,MOVES
from .partial_observation import validate
from .partial_phase5c_fixtures import uniform_schedule,balanced_schedule

DIRECTIONS=('Up','Down','Left','Right','Up-Left','Up-Right','Down-Left','Down-Right','At-goal')
DISTANCES=('0','1','2','3-5','6+')


def coordinates(observation):
    o=validate(np.asarray(observation,dtype=np.float32))
    return np.rint(o[:4]*9).astype(int)


def direction(delta):
    r,c=map(int,delta)
    if r==0 and c==0: return 'At-goal'
    if r==0: return 'Right' if c>0 else 'Left'
    if c==0: return 'Down' if r>0 else 'Up'
    return ('Down' if r>0 else 'Up')+'-'+('Right' if c>0 else 'Left')


def distance_band(n): return str(n) if n<=2 else '3-5' if n<=5 else '6+'


def state_bins(observation):
    o=np.asarray(observation,dtype=np.float32); r,c,gr,gc=coordinates(o)
    visits=int(np.rint(o[208+10*r+c]*5))
    if visits not in range(1,6): raise ValueError('Current cell must have a legal positive visit count')
    known=int(np.count_nonzero(o[8:108]!=-1)); d=abs(gr-r)+abs(gc-c)
    return {'direction':direction((gr-r,gc-c)),'quadrant':str(int(r>=5)*2+int(c>=5)),
        'boundary':'edge' if r in (0,9) or c in (0,9) else 'interior','distance':distance_band(d),
        'visits':'first' if visits==1 else 'saturated-five' if visits==5 else 'revisited',
        'known':'<25' if known<25 else '25-74' if known<75 else '75+',
        'elapsed':'0' if o[408]==0 else '1-19' if o[408]<.1 else '20-99' if o[408]<.5 else '100-199',
        'blocked_streak':'zero' if not np.any(o[308:408]) else 'positive'}


def reducing_actions(observation):
    r,c,gr,gc=coordinates(observation); d=abs(gr-r)+abs(gc-c)
    return [a for a,(dr,dc) in enumerate(MOVES[:4]) if abs(gr-r-dr)+abs(gc-c-dc)<d]


def score_record(observation,decision):
    scores=np.asarray(decision.get('action_values',decision.get('action_probabilities')),dtype=float)
    if scores.shape!=(5,) or not np.isfinite(scores).all(): raise ValueError('Missing/nonfinite five-action values')
    good=reducing_actions(observation); others=[a for a in range(5) if a not in good]
    return {'scores':scores.tolist(),'score_kind':'Q' if 'action_values' in decision else 'probability',
        'first_argmax':int(np.argmax(scores)),'maximizers':np.flatnonzero(scores==scores.max()).tolist(),
        'reducing_actions':good,'goal_margin':float(scores[good].max()-scores[others].max()) if good else None,
        'entropy':float(-np.sum(np.maximum(scores,1e-30)*np.log(np.maximum(scores,1e-30)))) if 'action_probabilities' in decision else None}


def bucket():
    return {'decisions':0,'actions':Counter(),'actual_distance_change':Counter(),
        'attempted_distance_change':Counter(),'collisions':0,'successful_moves':0,'waits':0,
        'selected_reducing':0,'greedy_reducing':0,'selected_first_argmax':0,'selected_in_max_set':0,
        'greedy_wait':0,'goal_rewards':0,'states':set()}


def change(a,b): return 'reduced' if b<a else 'increased' if b>a else 'preserved'


def add(b,record,quality):
    obs=np.asarray(record['observation'],dtype=np.float32); r,c,gr,gc=coordinates(obs)
    nr,nc,_,_=coordinates(record['next_observation']); action=int(record['action'])
    dr,dc=MOVES[action]; before=abs(gr-r)+abs(gc-c)
    b['decisions']+=1; b['actions'][ACTION_NAMES[action]]+=1
    b['actual_distance_change'][change(before,abs(gr-nr)+abs(gc-nc))]+=1
    b['attempted_distance_change'][change(before,abs(gr-r-dr)+abs(gc-c-dc))]+=1
    b['collisions']+=int(record['info']['outcome']=='collision')
    b['successful_moves']+=int((nr,nc)!=(r,c)); b['waits']+=int(action==4)
    b['selected_reducing']+=int(action in quality['reducing_actions'])
    b['greedy_reducing']+=int(quality['first_argmax'] in quality['reducing_actions'])
    b['selected_first_argmax']+=int(action==quality['first_argmax'])
    b['selected_in_max_set']+=int(action in quality['maximizers'])
    b['greedy_wait']+=int(quality['first_argmax']==4); b['goal_rewards']+=int(record['reward']>0)
    b['states'].add(hashlib.sha256(obs.tobytes()).hexdigest())


def finish(b):
    n=b['decisions']
    result={k:v for k,v in b.items() if k not in ('states','actions','actual_distance_change','attempted_distance_change')}
    result.update(observed=bool(n),distinct_states=len(b['states']),
        actions={name:b['actions'][name] for name in ACTION_NAMES},
        actual_distance_change={name:b['actual_distance_change'][name] for name in ('reduced','preserved','increased')},
        attempted_distance_change={name:b['attempted_distance_change'][name] for name in ('reduced','preserved','increased')},
        selected_reducing_fraction=b['selected_reducing']/n if n else None,
        greedy_reducing_fraction=b['greedy_reducing']/n if n else None)
    return result


def coverage(records,cases):
    ids=[c['id'] for c in cases]; lookup={c['id']:c for c in cases}
    case_counts=Counter(); reset_directions=Counter(); reset_distances=Counter(); reset_indices=[]
    dimensions={'direction':DIRECTIONS,'quadrant':('0','1','2','3'), 'boundary':('edge','interior'),
        'distance':DISTANCES,'visits':('first','revisited','saturated-five'),
        'known':('<25','25-74','75+'),'elapsed':('0','1-19','20-99','100-199'),
        'blocked_streak':('zero','positive')}
    marginal={k:{str(v):bucket() for v in values} for k,values in dimensions.items()}
    joint=defaultdict(bucket); near={str(d):{v:bucket() for v in DIRECTIONS} for d in (1,2)}
    total=bucket(); goals=Counter(); goal_cases=Counter(); goal_pairs=Counter(); episodes=[]
    epsilon=[]; reset_seeds=Counter(); position_pairs=set(); episode=None
    for record in records:
        if record['kind']=='reset':
            if episode is not None: episodes.append(episode)
            case=record['scenario']; case_id=case['id']
            if case_id not in lookup or case!=lookup[case_id]: raise ValueError('Undeclared training case')
            case_counts[case_id]+=1; reset_indices.append(ids.index(case_id)); reset_seeds[str(record['seed'])]+=1
            delta=np.asarray(case['goal'])-case['start']; reset_directions[direction(delta)]+=1
            reset_distances[str(int(np.abs(delta).sum()))]+=1
            episode={'id':record['episode'],'case_id':case_id,'actions':0,'outcome':'empty_reset'}
            continue
        if record['kind']!='step' or episode is None: raise ValueError('Decision without reset')
        if record['episode']!=episode['id']: raise ValueError('Wrong episode identity')
        o=np.asarray(record['observation'],dtype=np.float32); bins=state_bins(o)
        if bins['direction']=='At-goal': raise ValueError('Action after an A goal')
        q=score_record(o,record['behavior_context']['decision']); add(total,record,q)
        for name,value in bins.items(): add(marginal[name][value],record,q)
        key='|'.join(bins[k] for k in ('direction','quadrant','boundary','distance','visits','known'))
        add(joint[key],record,q)
        r,c,gr,gc=coordinates(o); distance=abs(gr-r)+abs(gc-c); position_pairs.add((r,c,gr,gc))
        if distance in (1,2): add(near[str(distance)][bins['direction']],record,q)
        eps=record['behavior_context'].get('epsilon_used')
        if eps is not None: epsilon.append(float(eps))
        episode['actions']+=1
        if record['terminated'] or record['truncated']: episode['outcome']=record['info']['outcome']
        else: episode['outcome']='incomplete_prefix'
        if record['reward']>0:
            if not record['terminated'] or record['reward']!=9.99 or distance!=1 or record['info']['outcome']!='success':
                raise ValueError('Invalid positive goal transition')
            goals[ACTION_NAMES[record['action']]]+=1; goal_cases[episode['case_id']]+=1
            goal_pairs[f'{r},{c}->{gr},{gc}']+=1
    if episode is not None: episodes.append(episode)
    for key in dimensions:
        if sum(b['decisions'] for b in marginal[key].values())!=total['decisions']: raise ValueError('Overlapping/lost marginal')
    if sum(b['decisions'] for b in joint.values())!=total['decisions']: raise ValueError('Lost joint decision')
    n=len(reset_indices)
    return {'total':finish(total),'reset_count':n,'reset_indices':reset_indices,
        'reset_cases':{i:case_counts[i] for i in ids},
        'reset_directions':{d:reset_directions[d] for d in DIRECTIONS[:4]},
        'reset_direction_shares':{d:reset_directions[d]/n if n else None for d in DIRECTIONS[:4]},
        'reset_distances':{d:reset_distances[d] for d in ('1','2')},'reset_seeds':dict(reset_seeds),
        'episode_outcomes':dict(Counter(e['outcome'] for e in episodes)),
        'distinct_position_goal_pairs':len(position_pairs),
        'marginals':{k:{v:finish(b) for v,b in groups.items()} for k,groups in marginal.items()},
        'joint_observed':{k:finish(b) for k,b in sorted(joint.items())},
        'joint_unobserved_cartesian_bin_count':9*4*2*5*3*3-len(joint),
        'joint_zero_note':'Unobserved includes structurally impossible bins; probability is not estimated as zero.',
        'near_goal':{d:{v:finish(b) for v,b in groups.items()} for d,groups in near.items()},
        'goal_rewards_by_arrival_action':{a:goals[a] for a in ACTION_NAMES},
        'goal_rewards_by_reset_case':{i:goal_cases[i] for i in ids},'goal_rewards_by_pre_action_pair':dict(goal_pairs),
        'epsilon':{'n':len(epsilon),'min':min(epsilon),'max':max(epsilon),'mean':float(np.mean(epsilon))} if epsilon else None,
        'exploration_label_limit':'Chosen vs current argmax is observable. Realized epsilon coin is not. PPO actions are stochastic; SARSA is carried pre-update.'}


def schedule_comparison(count,seed):
    uniform=uniform_schedule(count,seed); balanced=balanced_schedule(count,seed)
    def counts(indices):
        return {'cases':{str(i):indices.count(i) for i in range(8)},
                'directions':{d:sum(i%4==a for i in indices) for a,d in enumerate(DIRECTIONS[:4])}}
    return {'episodes':count,'seed':seed,'uniform':counts(uniform),'balanced':counts(balanced),
        'balanced_full_blocks':count//8,'balanced_remainder':balanced[count//8*8:]}


def sampling_gate(finals,config):
    rule=config['sampling_gate']; results={}; candidates=[]
    for family,node in finals.items():
        cv=node['coverage']; n=cv['reset_count']; sim=schedule_comparison(n,rule['simulation_seed'])
        directions={}
        for a,d in enumerate(DIRECTIONS[:4]):
            cases=[a,a+4]; case_counts=list(cv['reset_cases'].values())
            deficient=[i for i in cases if n and case_counts[i]/n<rule['underexposed_case_share_below']]
            under=bool(n and (cv['reset_directions'][d]/n<rule['underexposed_direction_share_below'] or deficient))
            near=cv['near_goal']['1'][d]['decisions']; goals=cv['goal_rewards_by_arrival_action'][d]
            scarce=near<rule['sparse_near1_direction_decisions_below'] or goals<rule['sparse_direction_goal_rewards_below']
            direction_gain=(sim['balanced']['directions'][d]-sim['uniform']['directions'][d])/n if n else None
            case_gain=max(((sim['balanced']['cases'][str(i)]-sim['uniform']['cases'][str(i)])/n for i in deficient),default=0.) if n else None
            checks={'enough_resets':n>=rule['minimum_episode_starts'],'reset_deficit':under,
                'impaired_same_direction':node['expanded_direction_goals'][d]<=rule['impaired_expanded_goals_at_most'],
                'scarce_near_goal_or_arrival_signal':scarce,
                'meaningful_planned_schedule_change':bool(n and (direction_gain>=rule['schedule_direction_share_gain_at_least'] or case_gain>=rule['schedule_case_share_gain_at_least']))}
            passed=all(checks.values())
            directions[d]={'checks':checks,'candidate':passed,'reset_count':cv['reset_directions'][d],
                'reset_share':cv['reset_direction_shares'][d],'near1_decisions':near,'goal_arrivals':goals,
                'expanded_goals':node['expanded_direction_goals'][d],
                'planned_direction_share_gain':direction_gain,'planned_deficient_case_share_gain':case_gain}
            if passed: candidates.append({'family':family,'direction':d})
        results[family]={'directions':directions,'schedule_simulation':sim}
    return {'coverage_intervention_justified':bool(candidates),'candidates':candidates,'families':results,
        'reason':'All predeclared deficit, failure, scarcity and schedule-change conditions must coincide. Uneven action coverage alone does not establish a reset-sampling mechanism.'}
