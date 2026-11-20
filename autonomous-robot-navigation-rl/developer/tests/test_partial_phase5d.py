"""Read-only study safeguards and independently worked credit fixtures."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_navigation.core import ROOT,read_json
from robot_navigation.partial_contract import configuration,MOVES
from robot_navigation.partial_agents import LinearPolicy
from robot_navigation.partial_features_scaled import ARMS,encode_scaled
from robot_navigation.partial_phase5d_states import (state_hash,relation,select_cohort,match_states,check_goal_only_pair,ray_for_action)
from robot_navigation.partial_phase5d_credit import (ForwardBudget,FrozenScores,scientific_guard,protect_inference,
    attribution,snapshot_td,require_historical_target,ppo_statistics,audit_linear_logs,audit_ppo_rollout)
from robot_navigation.partial_phase5d import Evidence,checked_hash,settings,unique_training_total


@pytest.fixture
def probes():
    return read_json(ROOT/'artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/probe-manifest.json')['probes']


@pytest.mark.parametrize('action,ray',[(0,0),(1,1),(2,3),(3,2),(4,None)])
def test_ray_order_is_independent_from_action_order(action,ray):
    assert ray_for_action(action)==ray
    if action<4:
        assert MOVES[action]==((-1,0),(1,0),(0,-1),(0,1))[action]


def test_snapshot_even_same_counter_cannot_be_historical():
    row=snapshot_td(9.99,True,False,None,-2,6000,6000)
    assert row['target']==9.99 and row['residual']==11.99
    with pytest.raises(ValueError):require_historical_target(row)
    row=snapshot_td(-.01,False,True,[1,2,3,4,5],2,6000,1000)
    assert row['target']==pytest.approx(4.94) and not row['historical_optimizer_target_verified']
    with pytest.raises(ValueError):snapshot_td(0,False,False,None,0,6000,1000)


def test_ppo_not_Q_and_zero_probability_entropy():
    row=ppo_statistics([0,0,0,1,0]);assert row['Q_values'] is None and row['entropy']==0
    with pytest.raises(ValueError):ppo_statistics([1]*5)
    with pytest.raises(ValueError):ppo_statistics([-.1,.1,.2,.3,.5])


def test_ppo_missing_rollout_history_never_fabricated():
    b=SimpleNamespace(full=False,pos=0)
    row=audit_ppo_rollout({'replay':b},[])
    assert row['full_history_GAE']=='not recorded' and row['saved_rollout_transitions']==0


def test_ppo_interior_GAE_and_unknown_nonterminal_boundary():
    values=np.array([2.,3.]);rewards=np.array([1.,1.]);advantages=np.array([1+.99*3-2+.99*.95*4,4.])
    b=SimpleNamespace(full=True,buffer_size=2,values=values,log_probs=np.log([.2,.2]),advantages=advantages,
        returns=advantages+values,rewards=rewards,episode_starts=np.array([1.,0.]))
    steps=[{'action':0,'reward':1.,'behavior_context':{'decision':{'action_probabilities':[.2]*5,'state_value':v}}} for v in values]
    row=audit_ppo_rollout({'replay':b,'last_episode_starts':np.array([False])},steps)
    assert not row['full_saved_rollout_independent_GAE'] and 'not recorded' in row['boundary']
    b.advantages[0]+=1
    with pytest.raises(ValueError):audit_ppo_rollout({'replay':b,'last_episode_starts':np.array([False])},steps)


def test_duplicate_prefix_summation_rejected():
    assert unique_training_total([6000]*8)==48000
    with pytest.raises(ValueError):unique_training_total([0,2000,6000]*8)


def test_checkpoint_corruption_rejected(tmp_path):
    from robot_navigation.release import sha
    p=tmp_path/'policy.zip';p.write_bytes(b'checkpoint');h=sha(p);checked_hash(p,h)
    p.write_bytes(b'corrupt')
    with pytest.raises(ValueError):checked_hash(p,h)


@pytest.mark.parametrize('folder',['final','test','tests','heldout'])
def test_final_test_reads_rejected(tmp_path,folder):
    p=tmp_path/folder/'data.json';p.parent.mkdir();p.write_text('{}')
    with pytest.raises(ValueError):Evidence(tmp_path,{}).json(p)


def test_unregistered_evidence_and_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError):Evidence(tmp_path,{}).json(tmp_path/'data.json')
    with pytest.raises(ValueError):Evidence(tmp_path,{}).json(tmp_path.parent/'outside.json')


def test_invalid_goal_swap_history_rejected(probes):
    a,b=deepcopy(probes[0]['observation']),deepcopy(probes[1]['observation'])
    assert check_goal_only_pair(a,b)
    b[208+5*10+4]=.2
    with pytest.raises(ValueError):check_goal_only_pair(a,b)
    b=deepcopy(a);b[2:4]=a[:2]
    with pytest.raises(ValueError):check_goal_only_pair(a,b)


def test_model_mutation_detected(probes):
    p=LinearPolicy('q-learning',configuration(ROOT))
    with pytest.raises(RuntimeError,match='mutated'):
        with protect_inference(p):p.weights[0,0]=1
    p=LinearPolicy('q-learning',configuration(ROOT))
    with protect_inference(p):
        with pytest.raises(RuntimeError):p.act(probes[0]['observation'])


def test_scientific_runtime_guard_blocks_all_mutators():
    from stable_baselines3 import DQN,PPO
    from robot_navigation.partial_dynamic_env import PartialDynamicEnv
    import socket
    with scientific_guard():
        for method in (DQN.learn,DQN.train,PPO.learn,PPO.train,LinearPolicy.update,PartialDynamicEnv.step):
            with pytest.raises(RuntimeError):method(None)
        with pytest.raises(RuntimeError):socket.create_connection(('example.com',443))


def test_cache_reuse_and_count_before_forward(probes):
    p=LinearPolicy('q-learning',configuration(ROOT));b=ForwardBudget(ceiling=1)
    e=FrozenScores(p,ARMS['B0-original'],b,'toy');o=probes[0]['observation'];e.reuse(o,[0]*5)
    assert e.get(o).tolist()==[0]*5 and b.calls==0
    e.get(o,'components');assert b.calls==1
    with pytest.raises(RuntimeError):e.get(probes[1]['observation'])
    assert b.calls==1


def test_linear_exact_contributions_and_finite_checks(probes):
    p=LinearPolicy('q-learning',configuration(ROOT));p.weights[:]=np.arange(2060).reshape(5,412)*.001
    e=FrozenScores(p,ARMS['B0-original'],ForwardBudget(),'toy')
    o=probes[0]['observation'];row=attribution(e.get(o),e.get(o,'components'))
    assert row['sum_error']<1e-9
    with pytest.raises(ValueError):attribution([1]*5,{'bias':[0]*5})


@pytest.mark.parametrize('algorithm',['q-learning','sarsa'])
def test_linear_logged_next_action_mechanics_without_applying_update(probes,algorithm):
    o=probes[0]['observation'];phi=encode_scaled(np.asarray(o,dtype=np.float32),ARMS['B0-original'])
    # Synthetic records only; no learner update is called. Terminal repeated rows isolate the one bootstrap fixture.
    steps=[];logs=[]
    for i in range(6000):
        steps.append({'episode':i,'observation':o,'next_observation':o,'action':0,'reward':9.99,
            'terminated':True,'truncated':False,'behavior_context':{'decision':{'action_values':[0]*5}}})
        logs.append({'update':i+1,'target':9.99,'td_error':9.99,'update_norm':.1*9.99/np.linalg.norm(phi),'next_behavior_action':None,
            'finite':True,'terminated':True,'feature_norm':float(np.linalg.norm(phi))})
    bootstrap=2 if algorithm=='sarsa' else 8;target=-.01+.99*bootstrap
    steps[0]['terminated']=False;steps[0]['reward']=-.01;steps[1]['episode']=0
    steps[1]['behavior_context']['decision']['action_values']=[.1*target,2,8,0,0]
    logs[0]=dict(logs[0],target=target,td_error=target,update_norm=.1*abs(target)/np.linalg.norm(phi),next_behavior_action=1 if algorithm=='sarsa' else None,terminated=False)
    error=9.99-.1*target
    logs[1]=dict(logs[1],td_error=error,update_norm=.1*abs(error)/np.linalg.norm(phi))
    rows=audit_linear_logs(steps,logs,ARMS['B0-original'],algorithm)
    assert rows[0]['computed_target']==pytest.approx(target)
    if algorithm=='sarsa':
        logs[0]['next_behavior_action']=2
        with pytest.raises(ValueError):audit_linear_logs(steps,logs,ARMS['B0-original'],algorithm)


def test_distinct_cohort_and_unsupported_state_null(probes):
    o=probes[3]['observation'];s={'global_training_interaction':1,'observation':o,'next_observation':o,'reward':9.99}
    cohort=select_cohort([s,dict(s,global_training_interaction=2)])
    assert len(cohort['selected'])==1 and cohort['selected'][0]['direction']=='Right'
    row=match_states([], [('case',o)])[0]
    assert row['exact_observation_count']==0 and row['nearest_mean_raw_l1'] is None
    assert state_hash(o)!=state_hash(probes[0]['observation'])


def test_scientific_sources_forbid_training_calls_and_network_imports():
    for p in (ROOT/'src/robot_navigation').glob('partial_phase5d*.py'):
        tree=ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                assert node.func.attr not in {'learn','train','update','step','act','explore','sample'},(p,node.lineno)
            if isinstance(node,ast.Import):assert not any(n.name in ('requests','urllib','httpx') for n in node.names)
    assert settings()['new_training_actions']==settings()['new_policy_environment_actions']==0
