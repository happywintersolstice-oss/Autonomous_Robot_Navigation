"""New cumulatively bounded synthetic mechanics; run via Phase 5E validate."""
import ast
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robot_navigation.core import ROOT,read_json
from robot_navigation.partial_contract import configuration,MOVES
from robot_navigation.partial_agents import LinearPolicy
from robot_navigation.partial_features_scaled import ARMS,encode_scaled,RelativeScale4Extractor,load_scaled_policy
from robot_navigation.partial_neural import PartialVisibleExtractor
from robot_navigation.partial_observation import SPACE
from robot_navigation.partial_instrumentation_trace import (TraceWriter,ReplayIdentity,MechanicalLedger,numeric,array,read_trace,checksum)
from robot_navigation.partial_training_instrumentation import observe,check_compatibility,library_contract
from robot_navigation.partial_instrumentation_audit import audit_records,validate_dqn,validate_gae,validate_ppo
from robot_navigation.partial_phase5e_mechanics import paired,MechanicalEnvironment,fixture_observation


@pytest.fixture(scope='session')
def output():
    path=os.environ.get('PHASE5E_OUTPUT')
    if not path:raise RuntimeError('Use python -m robot_navigation.partial_phase5e validate; cumulative budget/seal required')
    return Path(path)


@pytest.fixture(scope='session')
def runs(output):
    manifest=read_json(output/'attempt-plan.json');result={}
    for algorithm in ('q-learning','sarsa','dqn','ppo'):
        for arm in ARMS:
            if [algorithm,arm] not in manifest['selected_pairs']:continue
            path=output/'pairs'/(algorithm+'-'+arm)
            receipt=paired(algorithm,arm,path,os.environ['PHASE5E_LEDGER'],manifest['source_digest'],manifest['library_digest'],output/'model-config.json')
            result[algorithm,arm]={'path':path,'equivalence':receipt,'records':read_trace(path/'on/trace'),
                'on':read_json(path/'on/witness.json'),'off':read_json(path/'off/witness.json')}
    return result


@pytest.mark.parametrize('algorithm',['q-learning','sarsa','dqn','ppo'])
@pytest.mark.parametrize('arm',list(ARMS))
def test_independent_on_off_bitwise_equality_all_models_and_arms(runs,algorithm,arm):
    r=runs[algorithm,arm]
    assert r['on']==r['off'] and r['equivalence']['bitwise_equal'] and r['equivalence']['tolerance']==0
    for mode in ('on','off'):
        receipt=read_json(r['path']/mode/'receipt.json')
        assert receipt['checkpoint_loading']=='passed' and receipt['hooks_restored']


@pytest.mark.parametrize('arm',list(ARMS))
def test_actual_dqn_samples_targets_timeouts_duplicates_and_sync(runs,arm):
    r=runs['dqn',arm];rows=r['records'];a=audit_records(rows)
    assert a['counts']['DQN']==1 and a['optimizer_steps']==1
    sample=next(x for x in rows if x['event']=='replay_sample')
    assert len(sample['sample'])==50 and len({s['transition_id'] for s in sample['sample']})==4
    assert len({s['transition_id'] for s in sample['sample']})<len(sample['sample'])
    actual=next(x for x in rows if x['event']=='dqn_loss')
    near=validate_dqn(actual);assert near['target_max_error']==0
    kinds={(s['terminated'],s['truncated']) for s in actual['sample']}
    assert {(True,False),(False,True),(False,False)}.issubset(kinds)
    targets=array(actual['actual_target']).reshape(-1);rewards=array(actual['reward']).reshape(-1)
    for s,t,reward in zip(actual['sample'],targets,rewards):
        if s['terminated']:assert t==reward
    sync=[x for x in rows if x['event']=='target_sync'];assert len(sync)==1 and sync[0]['call']==250
    assert sync[0]['after']['target']==sync[0]['after']['online']
    assert sample['indices']['data']==r['on']['sample_order'][0]['indices']


@pytest.mark.parametrize('arm',list(ARMS))
def test_actual_ppo_bootstrap_GAE_permutation_normalization_losses_stop(runs,arm):
    r=runs['ppo',arm];rows=r['records'];a=audit_records(rows)
    assert a['counts']['GAE']==1 and a['counts']['PPO']==11 and a['optimizer_steps']==10
    assert len([x for x in rows if x['event']=='ppo_behavior'])==100
    assert len([x for x in rows if x['event']=='ppo_timeout_bootstrap'])==20
    gae=next(x for x in rows if x['event']=='gae_input');assert array(gae['dones']).tolist()==[False]
    assert len(gae['rows'])==100 and len({s['transition_id'] for s in gae['rows']})==100
    for p in (x for x in rows if x['event']=='ppo_permutation'):
        assert sorted(array(p['indices']).tolist())==list(range(100))
    losses=[x for x in rows if x['event']=='ppo_loss'];assert losses[-1]['early_stop'] and losses[-1]['optimizer_step'] is None
    assert r['on']['train_returns'][-1]['early_stop'] and not r['on']['train_returns'][0]['early_stop']
    for row in losses:
        assert row['Q_values']=='not_applicable';validate_ppo(row)
    assert len(r['on']['optimizer_steps'])==10


@pytest.mark.parametrize('algorithm',['q-learning','sarsa'])
@pytest.mark.parametrize('arm',list(ARMS))
def test_actual_linear_normalized_update_and_sampled_SARSA_action(runs,algorithm,arm):
    rows=runs[algorithm,arm]['records'];a=audit_records(rows)
    assert a['counts']['linear']==8
    updates=[x for x in rows if x['event']=='linear_update']
    assert all(x['neural_fields']=='not_applicable' for x in updates)
    if algorithm=='sarsa':assert all(x['sampled_next_action'] in range(5) for x in updates if not x['terminated'])
    else:assert all(x['sampled_next_action'] is None for x in updates)


def metadata():
    return {'source_kind':'synthetic_mechanical','study_id':'phase5e-training-instrumentation-v1','run_id':'injected-fixture',
        'algorithm':'dqn','arm':'B0-original','seed':23101,'source_hash':'test','config_hash':'test','dependency_hash':'test','split':'synthetic_mechanical'}


def test_ring_slot_generation_and_replacement_order():
    from stable_baselines3.common.buffers import ReplayBuffer
    import gymnasium as gym
    b=ReplayBuffer(3,SPACE,gym.spaces.Discrete(5),device='cpu')
    obs=fixture_observation().reshape(1,-1)
    side=ReplayIdentity(3)
    for i in range(1,11):
        side.add(b.pos,0,i)
        b.add(obs,obs,np.array([i%5]),np.array([float(i)]),np.array([False]),[{}])
    assert b.full and b.pos==1
    assert b.rewards[:,0].tolist()==[10.,8.,9.]
    rows=side.sampled([0,2,0,1],[0]*4)
    assert [x['transition_id'] for x in rows]==[10,9,10,8]
    assert [x['generation'] for x in rows]==[4,3,4,3]
    with pytest.raises(ValueError):side.add(1,0,10)
    with pytest.raises(ValueError):side.sampled([-1],[0])
    with pytest.raises(ValueError):side.sampled([0],[1])
    with pytest.raises(ValueError):side.sampled([0,1],[0])


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_tensors_fail_closed(value):
    with pytest.raises(ValueError):numeric([value])


def test_numeric_shape_dtype_hash_preserved_and_corruption_rejected():
    for a in (np.float32(1),np.arange(6,dtype=np.float32).reshape(2,3),np.array([True,False])):
        record=numeric(a);actual=array(record)
        assert actual.shape==np.asarray(a).shape and actual.dtype==np.asarray(a).dtype
        assert np.array_equal(actual,a)
    r=numeric([1.,2.]);r['data'][0]=3
    with pytest.raises(ValueError):array(r)


def test_trace_corruption_interruption_order_and_quotas(output):
    base=output/'injections';base.mkdir(exist_ok=True)
    tiny=TraceWriter(base/'quota',metadata(),byte_cap=100)
    with pytest.raises(RuntimeError):tiny.emit('large',values=numeric(np.ones(20)))
    tiny.close()
    partial=TraceWriter(base/'interrupted',metadata());partial.emit('observer_opened');partial.close()
    with pytest.raises(ValueError):read_trace(base/'interrupted')
    writer=TraceWriter(base/'corrupt',metadata());writer.emit('observer_closed');writer.close()
    p=next((base/'corrupt').glob('*.gz'))
    rows=gzip.decompress(p.read_bytes()).decode().splitlines();item=json.loads(rows[0]);item['record']['sequence']=9
    p.write_bytes(gzip.compress((json.dumps(item)+'\n').encode()))
    with pytest.raises(ValueError):read_trace(base/'corrupt')
    with pytest.raises(ValueError):TraceWriter(base/'forbidden-split',dict(metadata(),split='final'))
    writer=TraceWriter(base/'reserved-provenance',metadata())
    with pytest.raises(ValueError):writer.emit('fake',source_kind='historical_logged')
    writer.close()


def test_missing_bootstrap_rollout_identity_and_fake_PPO_Q_rejected(runs):
    rows=runs['ppo','B1-relative-x4']['records']
    inp=deepcopy(next(x for x in rows if x['event']=='gae_input'));result=next(x for x in rows if x['event']=='gae_result')
    inp.pop('last_values')
    with pytest.raises((ValueError,KeyError)):validate_gae(inp,result)
    inp=deepcopy(next(x for x in rows if x['event']=='gae_input'));inp['rollout_id']+=1
    with pytest.raises(ValueError):validate_gae(inp,result)
    loss=deepcopy(next(x for x in rows if x['event']=='ppo_loss'));loss['Q_values']=[1]*5
    with pytest.raises(ValueError):validate_ppo(loss)


def test_mismatched_actual_network_and_target_tensor_rejected(runs):
    rows=deepcopy(runs['dqn','B0-original']['records'])
    opt=next(x for x in rows if x['event']=='optimizer');opt['network_before']['online']='wrong-network'
    with pytest.raises(ValueError):audit_records(rows)
    loss=deepcopy(next(x for x in rows if x['event']=='dqn_loss'))
    loss['actual_target']=numeric(array(loss['actual_target'])+1)
    with pytest.raises(ValueError):validate_dqn(loss)


def test_disabled_is_no_op_and_hooks_restore_on_exception(output,monkeypatch):
    import robot_navigation.partial_training_instrumentation as module
    with monkeypatch.context() as m:
        m.setattr(module,'check_compatibility',lambda:pytest.fail('Disabled must not inspect or hook'))
        with observe(object()):pass
    ledger=MechanicalLedger(output/'injections/cleanup-ledger.jsonl');env=MechanicalEnvironment(ledger,'cleanup')
    p=LinearPolicy('q-learning',configuration());writer=TraceWriter(output/'injections/cleanup',metadata())
    with pytest.raises(RuntimeError,match='injected'):
        with observe(p,writer,environment=env):raise RuntimeError('injected observer interruption')
    writer.close();assert sys.gettrace() is None
    with pytest.raises(ValueError):read_trace(output/'injections/cleanup')


def test_unsupported_library_fails_before_hooks():
    contract=deepcopy(library_contract());contract['versions']['torch']='unsupported'
    with pytest.raises(RuntimeError,match='UNSUPPORTED_INSTRUMENTATION_PATH'):check_compatibility(contract)
    assert sys.gettrace() is None


def test_resource_ledger_rejects_next_operation_before_execution(output):
    p=output/'injections/fake-exhausted-ledger.jsonl'
    p.write_text('\n'.join(json.dumps({'kind':'optimizer','sequence':i+1}) for i in range(500))+'\n')
    with pytest.raises(RuntimeError):MechanicalLedger(p).record('optimizer')
    assert len(p.read_text().splitlines())==500


@pytest.mark.parametrize('arm',list(ARMS))
def test_feature_parity_original_contract_and_action_ray_mapping(arm):
    obs=fixture_observation();cls=PartialVisibleExtractor if arm=='B0-original' else RelativeScale4Extractor
    actual=cls(SPACE)(torch.tensor(obs).unsqueeze(0)).detach().numpy()[0]
    np.testing.assert_allclose(actual,encode_scaled(obs,ARMS[arm]),atol=1e-7,rtol=0)
    assert len(obs)==409 and len(actual)==412 and MOVES==((-1,0),(1,0),(0,-1),(0,1),(0,0))
    # Rays N,S,E,W; corresponding action IDs U,D,R,L.
    assert [0,1,3,2]==[MOVES.index(m) for m in ((-1,0),(1,0),(0,1),(0,-1))]


def test_checkpoint_corruption_rejected_without_prediction(runs,output):
    source=runs['q-learning','B0-original']['path']/'on/policy.npz'
    target=output/'injections/corrupt-policy.npz';target.write_bytes(source.read_bytes()+b'corruption')
    target.with_suffix('.json').write_bytes(source.with_suffix('.json').read_bytes())
    with pytest.raises(ValueError,match='checksum'):load_scaled_policy(target,configuration(),'B0-original')


def test_source_information_boundary_and_no_experiment_cli():
    observer=(ROOT/'src/robot_navigation/partial_training_instrumentation.py').read_text()
    assert all(name not in observer for name in ('_true_state','_scenario','diagnostic_snapshot','sealed-final.json'))
    tree=ast.parse(observer)
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
            assert n.func.attr not in ('learn','train','sample','predict','predict_values','get_distribution','step','update')
    from robot_navigation.partial_phase5e import checked_input
    for path in ('data/unseen/final.json','data/partial/test.json','../outside.json','data/final/maps.json'):
        with pytest.raises(ValueError):checked_input(ROOT,path,{})
