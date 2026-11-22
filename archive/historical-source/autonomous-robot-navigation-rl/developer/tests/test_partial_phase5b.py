"""Phase 5B mechanical, isolation and injected-failure gates; no study reruns."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from robot_navigation.core import ROOT, read_json, write_json
from robot_navigation.release import sha
from robot_navigation.partial_agents import LinearPolicy, load_policy
from robot_navigation.partial_contract import ACTION_NAMES, MOVES, configuration
from robot_navigation.partial_dynamic_env import PartialDynamicEnv, make_scenario
from robot_navigation.partial_features import encode_features
from robot_navigation.partial_features_scaled import (ARMS, SCALED, ScaledLinearPolicy,
    RelativeScale4Extractor, build_policy, encode_scaled, load_scaled_policy)
from robot_navigation.partial_neural import PartialVisibleExtractor
from robot_navigation.partial_feasibility_fixtures import cases, _rotate, _turn_action
from robot_navigation.partial_observation import SPACE
from robot_navigation.partial_pilot_diagnostics import preserve_random_state
from robot_navigation.partial_pilot_ledger import PilotLimit, PilotRecorder, ContinuousLinear
from robot_navigation.partial_commands import read_records
from robot_navigation.partial_phase5b import (ArtifactQuota, TrialBudget, ArmBudget, QuotaStream,
    settings, screening, reconcile_ledger, checked_manifest)
from robot_navigation.partial_phase5b_fixtures import expanded_cases, audit_cases, goal_action
from robot_navigation.partial_phase5b_diagnostics import (bellman_checks, continuity_state,
    continuity_hash, decision_diagnostic, feature_statistics, UpdateRecorder, LinearUpdateRecorder,two_step_checks)


def small_settings():
    c=deepcopy(settings()); c.update(per_arm_actions=8,total_training_actions=64,
        evaluation_action_ceiling=20,suite_wall_seconds=100,per_arm_wall_seconds=10,
        artifact_byte_ceiling=8*1024**2)
    return c


@pytest.mark.parametrize('turns',range(4))
@pytest.mark.parametrize('action',range(5))
def test_action_ids_ray_alignment_and_rotated_coordinate_normalization(turns,action):
    start=[4,4]; dr,dc=MOVES[action]
    goal=[4+dr,4+dc] if action!=4 else [4,6]
    env=PartialDynamicEnv([make_scenario(['.'*10]*10,_rotate(start,turns),_rotate(goal,turns))])
    obs,_=env.reset(seed=0); actual_action=_turn_action(action,turns)
    nxt,reward,terminal,cutoff,_=env.step(actual_action)
    expected=_rotate([4+dr,4+dc],turns)
    np.testing.assert_allclose(nxt[:2],np.array(expected)/9,atol=1e-7)
    assert ACTION_NAMES[actual_action] in ('Up','Down','Left','Right','Wait')
    assert terminal==(action!=4) and not cutoff
    assert reward==(9.99 if action!=4 else -.01)
    if action!=4: assert goal_action(obs)==actual_action
    # East is ray index 2/raw 6, whereas Left is action 2.
    row=['.'*10]*10; row[4]='.....#....'
    reading=PartialDynamicEnv([make_scenario(row,[4,4],[2,4])]).reset()[0]
    assert reading[6]==.5 and MOVES[2]==(0,-1)


@pytest.mark.parametrize('algorithm',['q-learning','sarsa'])
@pytest.mark.parametrize('arm',list(ARMS))
@pytest.mark.parametrize('outcome,reward,terminated',[
    ('nonterminal',-.01,False),('goal',9.99,True),('collision',-5.01,True),('truncation',-.01,False)])
def test_analytic_bellman_targets_against_both_real_encoders(algorithm,arm,outcome,reward,terminated):
    cls=LinearPolicy if arm=='B0-original' else ScaledLinearPolicy
    p=cls(algorithm,configuration()); obs,_=PartialDynamicEnv(cases('A')).reset(seed=0)
    p.weights[1,0]=2.; p.weights[3,0]=3.
    target=reward if terminated else reward+.99*(2 if algorithm=='sarsa' else 3)
    p.update(obs,2,reward,obs,terminated,.5,.99,1 if algorithm=='sarsa' and not terminated else None)
    assert p.scores(obs)[2]==pytest.approx(.5*target,abs=1e-12)
    assert np.isfinite(p.weights).all()
    if terminated:
        p.weights[1:]=1000
        p.weights[0]=0
        p.update(obs,0,reward,obs,True,1.,.99)
        assert p.scores(obs)[0]==pytest.approx(reward)


def test_two_step_worked_targets_are_executed_and_capacity_witness_is_not_a_policy():
    rows=bellman_checks()
    assert len(rows)==10
    assert [r['target'] for r in rows if r['case']=='two-step-backup-to-goal']==pytest.approx([9.8801]*2)
    two=two_step_checks(); assert len(two)==8
    assert {r['second_transition'] for r in two}=={'goal','collision','nonterminal','external_truncation'}
    assert next(r['first_target'] for r in two if r['algorithm']=='sarsa' and r['second_transition']=='collision')==pytest.approx(-4.9699)
    for case in cases('A'):
        env=PartialDynamicEnv([case]); obs,_=env.reset()
        assert np.rint(obs[:2]*9).astype(int).tolist()==case['start']
        assert tuple(np.array(case['start'])+MOVES[goal_action(obs)])==tuple(case['goal'])
        assert env.diagnostic_snapshot()['step']==0


@pytest.mark.parametrize('arm',list(ARMS))
def test_numpy_torch_parity_zero_random_edge_corner_and_goal_near(arm):
    rng=np.random.default_rng(34)
    samples=[np.zeros(409,dtype=np.float32)]
    for _ in range(12):
        x=rng.random(409,dtype=np.float32); x[:4]=rng.integers(0,10,4)/9
        x[4:8]=rng.choice([0,.5,1],4); x[8:108]=rng.choice([-1,0,1],100)
        samples.append(x)
    for case in cases('A')+expanded_cases(): samples.append(PartialDynamicEnv([case]).reset()[0])
    for start,goal in [([0,0],[9,9]),([9,9],[0,0]),([0,9],[0,8]),([9,0],[8,0])]:
        samples.append(PartialDynamicEnv([make_scenario(['.'*10]*10,start,goal)]).reset()[0])
    cls=PartialVisibleExtractor if arm=='B0-original' else RelativeScale4Extractor
    actual=cls(SPACE)(torch.tensor(np.stack(samples))).detach().numpy()
    expected=np.stack([encode_scaled(x,ARMS[arm]) for x in samples])
    np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)
    assert expected.shape==(len(samples),412)
    if arm=='B0-original':
        assert np.array_equal(expected,np.stack([encode_features(x) for x in samples]))
    else:
        assert abs(expected[-4:,-2:]).max()==4
        assert np.array_equal(expected[:,:410],np.stack([encode_features(x) for x in samples])[:,:410])


def test_hidden_topology_events_equivalence_and_feature_source_boundary(monkeypatch):
    rows=['.'*10]*10; changed=list(rows); changed[9]='.........#'
    cases_hidden=[make_scenario(rows,[5,5],[1,1]),
        make_scenario(changed,[5,5],[1,1],[{'id':'future','cell':[9,8],'start':0,'clear':50}],scenario_id='secret')]
    envs=[PartialDynamicEnv([c]) for c in cases_hidden]
    observations=[env.reset(seed=5)[0] for env in envs]
    for action in (4,0,4):
        assert np.array_equal(*observations)
        assert np.array_equal(encode_scaled(observations[0]),encode_scaled(observations[1]))
        observations=[env.step(action)[0] for env in envs]
    assert list(inspect.signature(encode_scaled).parameters)==['observation','feature_version']
    with pytest.raises(TypeError): encode_scaled(observations[0],scenario_id='secret')
    with pytest.raises(ValueError): encode_scaled({'true_map':rows})
    def forbidden(*args,**kwargs): pytest.fail('Hidden diagnostics reached learned policy')
    monkeypatch.setattr(PartialDynamicEnv,'diagnostic_snapshot',forbidden)
    for arm in ARMS:
        for algorithm in ('q-learning','sarsa','dqn','ppo'):
            p=build_policy(envs[0],algorithm,settings()['model'],23021,arm)
            assert p.act(observations[0])==p.act(observations[1])


@pytest.mark.parametrize('algorithm',list(('q-learning','sarsa','dqn','ppo')))
def test_strict_versions_hyperparameter_parity_initial_weights_and_corruption(tmp_path,algorithm):
    policies=[]
    for arm in ARMS:
        p=build_policy(PartialDynamicEnv(cases('A','train')),algorithm,settings()['model'],23021,arm)
        path=tmp_path/(arm+('.npz' if hasattr(p,'weights') else '.zip'))
        p.save(path,{'arm':arm}); policies.append(p)
        with preserve_random_state(): loaded=load_scaled_policy(path,configuration(),arm)
        assert loaded.state_fingerprint()==p.state_fingerprint()
        other='B1-relative-x4' if arm=='B0-original' else 'B0-original'
        with pytest.raises(ValueError,match='version mismatch'): load_scaled_policy(path,configuration(),other)
        if arm=='B0-original': assert load_policy(path,configuration()).state_fingerprint()==p.state_fingerprint()
        else:
            with pytest.raises(ValueError,match='mismatch'): load_policy(path,configuration())
        path.write_bytes(path.read_bytes()+b'corrupt')
        with pytest.raises(ValueError,match='checksum'): load_scaled_policy(path,configuration(),arm)
    assert policies[0].state_fingerprint()==policies[1].state_fingerprint()
    if algorithm in ('dqn','ppo'):
        for key in ('learning_rate','gamma','batch_size','seed'):
            assert getattr(policies[0].model,key)==getattr(policies[1].model,key)
        if algorithm=='dqn':
            for key in ('learning_starts','buffer_size','exploration_fraction','train_freq','target_update_interval'):
                assert getattr(policies[0].model,key)==getattr(policies[1].model,key)
        else:
            for key in ('n_steps','n_epochs','ent_coef'):
                assert getattr(policies[0].model,key)==getattr(policies[1].model,key)


def test_phase5a_checkpoints_still_load_under_original_signature():
    for a in ('q-learning','sarsa','dqn','ppo'):
        p=ROOT/f'artifacts/partial-dynamic-navigation-v1/feasibility-pilot-v1/{a}/A-10000'
        path=p/('policy.npz' if a in ('q-learning','sarsa') else 'policy.zip')
        with preserve_random_state():
            old=load_policy(path,configuration()); loaded=load_scaled_policy(path,configuration(),'B0-original')
        assert loaded.state_fingerprint()==old.state_fingerprint()


def test_deterministic_expanded_registration_unique_pairs_balanced_without_new_layouts():
    a,b=expanded_cases(),expanded_cases()
    assert a==b
    result=audit_cases(a)
    assert result['pair_overlap']==0
    assert result['counts']['distance']=={'1':8,'2':8}
    assert sorted(result['counts']['quadrant'].values())==[4]*4
    a[0]=deepcopy(cases('A','train')[0])
    with pytest.raises(ValueError,match='Duplicate'): audit_cases(a)


@pytest.mark.parametrize('algorithm',['q-learning','sarsa'])
@pytest.mark.parametrize('arm',list(ARMS))
def test_chunk_continuation_evaluation_rng_and_sampled_sarsa_are_uncontaminated(tmp_path,algorithm,arm):
    outcomes=[]
    for split in (False,True):
        folder=tmp_path/str(split); folder.mkdir()
        c=small_settings(); budget=TrialBudget(c,folder); budget.start(f'{algorithm}/{arm}')
        adapter=ArmBudget(budget,algorithm,arm); stream=QuotaStream(folder/'trace.gz',budget.quota)
        env=PilotRecorder(cases('A','train'),configuration(),adapter,algorithm,stream,'A',23021)
        p=build_policy(env,algorithm,c['model'],23021,arm)
        trainer=ContinuousLinear(p,adapter,{**c,'per_algorithm_actions':10000}); trainer.set_env(env)
        stream.begin(); trainer.train(4 if split else 8); stream.close()
        if split:
            before=continuity_hash(continuity_state(p,trainer,env)); prefix=stream.path.read_bytes()
            path=folder/'model.npz'; p.save(path,{})
            with preserve_random_state():
                loaded=load_scaled_policy(path,configuration(),arm)
                for case in cases('A')+expanded_cases(): loaded.act(PartialDynamicEnv([case]).reset()[0])
            assert continuity_hash(continuity_state(p,trainer,env))==before
            stream.begin(); trainer.train(4); stream.close()
            assert stream.path.read_bytes().startswith(prefix)
        records=read_records(stream.path)
        outcomes.append((p.weights.copy(),[r['action'] for r in records if r['kind']=='step']))
        budget.close()
    assert np.array_equal(outcomes[0][0],outcomes[1][0]) and outcomes[0][1]==outcomes[1][1]


def test_budget_interruptions_no_retry_and_exact_accounting():
    c=small_settings(); now=[0.]; b=TrialBudget(c,clock=lambda:now[0]); key='q-learning/B0-original'
    b.start(key)
    for _ in range(8): b.reserve('training',key)
    with pytest.raises(PilotLimit): b.reserve('training',key)
    with pytest.raises(PilotLimit,match='retry'): b.start(key)
    assert b.training[key]==8
    now[0]=11.
    with pytest.raises(PilotLimit,match='Per-arm'): b.reserve('evaluation',key)
    now[0]=101.
    with pytest.raises(PilotLimit,match='trial wall'): b.reserve('evaluation')
    training={key:1}; rows=[{'kind':'training','run_id':key,'count':1}]
    assert reconcile_ledger(rows,training,0,True)['training_total']==1
    with pytest.raises(ValueError,match='Wrong actual'): reconcile_ledger(rows,{key:0},0,True)
    with pytest.raises(ValueError,match='count/ceiling'): reconcile_ledger(rows*2,training,0,False)


def test_compressed_stream_quota_and_incomplete_prefix_survive_exception(tmp_path):
    c=small_settings(); b=TrialBudget(c,tmp_path); key='q-learning/B0-original'; b.start(key)
    stream=QuotaStream(tmp_path/'prefix.gz',b.quota)
    stream.begin(); stream.write({'kind':'reset','fixture':'synthetic'})
    try:
        for i in range(3):
            b.reserve('training',key); stream.write({'kind':'step','action':4,'terminated':False,'truncated':False})
        raise RuntimeError('injected interrupted run')
    except RuntimeError: stream.close()
    records=read_records(stream.path)
    assert len(records)==4 and not records[-1]['terminated'] and not records[-1]['truncated']
    assert b.training[key]==3
    b.close()
    quota=ArtifactQuota(tmp_path,1048576+1000)
    with pytest.raises(PilotLimit,match='Artifact ceiling'): quota.write_bytes(tmp_path/'oversize',b'x'*1001)
    assert not (tmp_path/'oversize').exists() or (tmp_path/'oversize').stat().st_size==0


def test_corrupt_manifest_and_stale_hash_are_rejected(tmp_path,monkeypatch):
    import robot_navigation.partial_phase5b as module
    folder=module.location(tmp_path); folder.mkdir(parents=True)
    write_json(folder/'manifest.json',{'inputs':{}})
    write_json(folder/'protocol-seal.json',{'sha256':'bad'})
    with pytest.raises(ValueError,match='Corrupt'): checked_manifest(tmp_path)
    write_json(folder/'protocol-seal.json',{'sha256':sha(folder/'manifest.json')})
    monkeypatch.setattr(module,'source_inputs',lambda root:{'changed':'source'})
    with pytest.raises(ValueError,match='Stale'): checked_manifest(tmp_path)


def test_failure_attempt_is_preserved_and_cannot_be_executed_again(tmp_path,monkeypatch):
    import robot_navigation.partial_phase5b as module
    folder=module.location(tmp_path); folder.mkdir(parents=True)
    write_json(folder/'status.json',{'status':'registered'})
    c=small_settings()
    monkeypatch.setattr(module,'checked_manifest',lambda root:{'settings':c,'original_cases':cases('A'),'expanded_cases':expanded_cases()})
    monkeypatch.setattr(module,'configuration',lambda root:configuration())
    def fail(*args,**kwargs): raise RuntimeError('injected mechanics failure')
    monkeypatch.setattr(module,'assess',fail)
    r=module.run(tmp_path)
    assert r['status']=='interrupted' and r['budget']['training_total']==0
    assert (folder/'execution.lock').exists() and read_json(folder/'summary.json')['traceback']
    with pytest.raises(ValueError,match='no free retry'): module.run(tmp_path)


def test_fixed_screen_requires_every_condition_and_rejects_tuned_protocol(tmp_path):
    def cp(original,expanded): return {'panels':{'original':{'counts':{'success':original}},'expanded':{'counts':{'success':expanded}}}}
    def arm(*cps): return {'status':'completed','checkpoints':list(cps)}
    systems={a:{'B0-original':arm(cp(1,4),cp(2,8)),
                'B1-relative-x4':arm(cp(2,7),cp(3,12))} for a in ('q-learning','sarsa','dqn','ppo')}
    assert all(v['passed'] for v in screening(systems,settings()).values())
    systems['q-learning']['B0-original']['checkpoints'][-1]=cp(3,8)
    assert not screening(systems,settings())['q-learning']['passed']
    configs=tmp_path/'configs'; configs.mkdir()
    import yaml
    data=yaml.safe_load((ROOT/'configs/partial_phase5b_v1.yaml').read_text())
    data['screen']['expanded_successes']=11
    (configs/'partial_phase5b_v1.yaml').write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError,match='envelope changed'): settings(tmp_path)


def test_training_only_statistics_and_known_invalid_goal_margin():
    observations=[PartialDynamicEnv(cases('A','train')).reset(seed=i)[0] for i in range(8)]
    stats=feature_statistics(observations)
    assert stats['training_observations_only'] and stats['count']==8
    assert len(stats['per_feature']['mean'])==412 and stats['relative_goal_energy_fraction']['max']<.01
    env=PartialDynamicEnv([make_scenario(['.'*10]*10,[0,0],[0,1])]); obs,_=env.reset()
    p=LinearPolicy('q-learning',configuration()); d=decision_diagnostic(p,obs)
    assert d['chosen_action']==0 and d['known_invalid_attempt'] and d['goal_actions']==[3]


@pytest.mark.parametrize('algorithm',['dqn','ppo'])
@pytest.mark.parametrize('arm',list(ARMS))
def test_real_neural_gradients_buffers_optimizer_rng_survive_copy_assessment(tmp_path,algorithm,arm):
    from robot_navigation.partial_training import TrainingRecorder
    env=TrainingRecorder(PartialDynamicEnv(cases('A','train')),cases('A','train'))
    policy=build_policy(env,algorithm,settings()['model'],23021,arm)
    path=tmp_path/'updates.gz'; observer=UpdateRecorder(policy,path,lambda:policy.model.num_timesteps)
    try: policy.model.learn(total_timesteps=104 if algorithm=='dqn' else 100,reset_num_timesteps=False)
    finally: observer.close()
    updates=read_records(path)
    assert updates and all(r['finite'] and r['parameter_update_norm']>0 for r in updates)
    assert observer.steps==(1 if algorithm=='dqn' else 10)
    before=continuity_hash(continuity_state(policy,None,env))
    model_path=tmp_path/'policy.zip'; policy.save(model_path,{})
    with preserve_random_state():
        loaded=load_scaled_policy(model_path,configuration(),arm)
        for case in cases('A')+expanded_cases(): loaded.act(PartialDynamicEnv([case]).reset()[0])
    assert continuity_hash(continuity_state(policy,None,env))==before


def test_b1_sarsa_samples_before_update_and_discards_truncation_action(tmp_path):
    from robot_navigation.partial_training import TrainingRecorder,train_linear
    config={**configuration(),'max_steps':1}
    case=make_scenario(['.'*10]*10,[5,5],[9,9])
    class Spy(ScaledLinearPolicy):
        def __init__(self):
            super().__init__('sarsa',config); self.samples=iter([3,4,2,1]); self.received=[]
        def explore(self,*args): return next(self.samples)
        def update(self,*args):
            self.received.append((args[4],args[-1])); return super().update(*args)
    p=Spy(); env=TrainingRecorder(PartialDynamicEnv([case],config),[case])
    train_linear(p,env,2,23021,settings()['model'])
    assert [r['action'] for r in env.records if r['kind']=='step']==[3,2]
    assert p.received==[(False,4),(False,1)]
