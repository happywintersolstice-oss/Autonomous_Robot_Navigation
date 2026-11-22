"""Focused H tests; one shared 20-step smoke, six mechanical optimizer calls."""
import copy,json,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure
from robot_navigation.core import ROOT,write_json
from robot_navigation.partial_contract import configuration,MOVES
from robot_navigation.partial_dynamic_env import PartialDynamicEnv
from robot_navigation.partial_features_scaled import encode_scaled
from robot_navigation.partial_phase5g_io import cases
from robot_navigation.partial_phase5g_assess import global_rng
from robot_navigation.partial_phase5h import settings
from robot_navigation.partial_phase5h_auxiliary import labels,LeanLearning,SEED,parameter_counts
from robot_navigation.partial_phase5h_evidence import TrainingEnv,Stream,model,numerical_check,records


def test_direction_conventions_diagonal_ties_border_goal():
    pairs=[([5,5],[4,5],0),([5,5],[6,5],1),([5,5],[5,4],2),([5,5],[5,6],3),
        ([5,5],[3,4],0),([5,5],[6,3],2),([5,5],[3,7],0),([5,5],[7,3],1),
        ([0,0],[0,9],3),([9,9],[0,9],0),([4,4],[4,4],-1)]
    obs=np.zeros((len(pairs),409),np.float32)
    for i,(a,b,_) in enumerate(pairs):obs[i,:4]=np.array(a+b)/9
    assert labels(torch.as_tensor(obs)).tolist()==[p[2] for p in pairs]
    assert MOVES==((-1,0),(1,0),(0,-1),(0,1),(0,0))


def test_feature_contract():
    obs=np.zeros(409,np.float32);obs[:4]=np.array([4,5,2,7])/9
    features=encode_scaled(obs);assert features.shape==(412,)
    np.testing.assert_allclose(features[-2:],4*(obs[2:4]-obs[:2]))


def test_frozen_scope_and_panels():
    c=settings();assert c['smoke_transitions']==20 and c['mechanical_optimizer_cap']==6
    p=cases();assert [len(p[k]) for k in ('training_cases','original_cases','expanded_cases')]==[8,4,16]
    assert c['full_regression_suite']=='DEFERRED_TO_MAJOR_MILESTONE'


@pytest.fixture(scope='module')
def mechanics():
    folder=Path(os.environ['PHASE5H_MECHANICS']);budget={'transitions':0,'optimizer_steps':0}
    ledger=folder/'mechanical-ledger.jsonl';data=[]
    def debit(kind):
        key='transitions' if kind=='transition' else 'optimizer_steps';cap=20 if kind=='transition' else 6
        assert budget[key]<cap;budget[key]+=1
        with ledger.open('a') as f:f.write(json.dumps({'kind':kind,'reserved_before_operation':True,'count':budget[key]})+'\n')
    stream=Stream(folder/'smoke-actions.jsonl.gz');updates=[]
    class SmokeEnv(TrainingEnv):
        def step(self,action):
            if self.n<self.cap:debit('transition')
            before=self.obs.copy();result=super().step(action)
            data.append((before,result[0].copy(),action,result[1],result[2],result[3]));return result
    cfg=__import__('yaml').safe_load((ROOT/'configs/partial_feasibility_v1.yaml').read_text())['model']
    testcfg=dict(cfg,learning_starts=4,batch_size=4)
    env=SmokeEnv(PartialDynamicEnv(cases()['training_cases'],configuration()),stream,'synthetic-H1',cap=20)
    policy=model(env,'H1',testcfg);env.model=policy.model
    with LeanLearning(policy.model,.05,updates.append,lambda:debit('optimizer'),lambda:env.last) as tap:
        policy.model.learn(total_timesteps=20,reset_num_timesteps=False)
    assert env.n==20 and tap.updates==4
    with pytest.raises(RuntimeError):env.step(4)
    stream.write({'event':'closed','mechanical':True});stream.close()
    for row in updates:
        if row['event']=='update':numerical_check(row)
    models=[];initial=[];fixed=[]
    for arm in ('H0','H1'):
        e=PartialDynamicEnv(cases()['training_cases'],configuration());p=model(e,arm,cfg);m=p.model;m.set_logger(configure(None,[]));m.num_timesteps=104
        initial.append({k:v.detach().clone() for k,v in m.q_net.state_dict().items()})
        if arm=='H1':
            x=torch.as_tensor(np.stack([v[0] for v in data[:4]]));latent=m.q_net.q_net[:-1](m.q_net.extract_features(x,m.q_net.features_extractor))
            loss=torch.nn.functional.cross_entropy(m.policy.goal_direction_head(latent),labels(x))
            shared=list(m.q_net.q_net[0].parameters())+list(m.q_net.q_net[2].parameters());qhead=list(m.q_net.q_net[4].parameters());head=list(m.policy.goal_direction_head.parameters())
            grads=torch.autograd.grad(loss,shared+qhead+head,allow_unused=True)
            assert any(g is not None and bool((g!=0).any()) for g in grads[:len(shared)])
            assert all(g is None for g in grads[len(shared):len(shared)+len(qhead)])
            assert all(g is not None for g in grads[-len(head):])
        latest={};rows=[]
        with LeanLearning(m,0.,rows.append,lambda:debit('optimizer'),lambda:latest) as observer:
            for i,(o,n,a,r,term,trunc) in enumerate(data[:4],1):
                latest={'transition_id':i};m.replay_buffer.add(o[None],n[None],np.array([a]),np.array([r]),np.array([term or trunc]),[{'TimeLimit.truncated':trunc}])
            np.random.seed(SEED);m.train(gradient_steps=1,batch_size=4)
        fixed.append(next(r for r in rows if r['event']=='update'));models.append(p)
    assert all(torch.equal(initial[0][k],initial[1][k]) for k in initial[0])
    assert fixed[0]['sample_ids']==fixed[1]['sample_ids']
    for k,v in models[0].model.q_net.state_dict().items():torch.testing.assert_close(v,models[1].model.q_net.state_dict()[k],atol=2e-5,rtol=2e-6)
    assert any(not torch.equal(v,initial[0][k]) for k,v in models[0].model.q_net.state_dict().items())
    with patch.object(models[1].model.policy.goal_direction_head,'forward',side_effect=AssertionError('Auxiliary head must not select actions')):
        models[1].act(data[0][0])
    receipt={**budget,'scope':'one H1-only 20-step smoke plus two fixed-batch lambda0 updates; no research',
        'lambda0_Q_update_match':True,'initial_common_Q_equal':True,'same_fixed_batch_ids':True,'aux_gradient_to_shared_and_head_only':True,
        'Q_only_action_selection':True,'parameter_counts':{a:parameter_counts(p.model) for a,p in zip(('H0','H1'),models)}}
    write_json(folder/'mechanics.json',receipt)
    write_json(folder/'mechanics-evidence.json',{'classification':'synthetic_mechanical_only','smoke_updates':updates,'fixed_lambda0_updates':fixed})
    return receipt,updates,fixed


def test_one_bounded_smoke_and_lambda0_equivalence(mechanics):
    r,_,_=mechanics;assert r['transitions']==20 and r['optimizer_steps']==6 and r['lambda0_Q_update_match']


def test_extra_head_parameter_count_and_gradient_routing(mechanics):
    r,_,_=mechanics;a,b=r['parameter_counts']['H0'],r['parameter_counts']['H1']
    assert a['online_q']==b['online_q'] and a['auxiliary']==0 and b['auxiliary']==132
    assert b['optimizer_parameters']-a['optimizer_parameters']==132


def test_actual_target_loss_auxiliary_arithmetic(mechanics):
    _,updates,fixed=mechanics
    for r in updates+fixed:
        if r['event']=='update':numerical_check(r)


def test_corrupt_consumed_tensor_rejected(mechanics):
    _,_,fixed=mechanics;r=copy.deepcopy(fixed[0]);r['actual_target']['data'][0]+=1
    with pytest.raises(ValueError):numerical_check(r)


def test_no_hidden_geometry_or_development_training_source():
    src=(ROOT/'src/robot_navigation/partial_phase5h_auxiliary.py').read_text()
    assert '_true_state' not in src and 'snapshot(' not in src and 'replay_buffer.sample(' not in src
    assert 'frame.f_code is not DQN.train.__code__' in src and 'self.coefficient*ce' in src
    assert "head(self.latent)" in src and 'labels(r.observations)' in src
