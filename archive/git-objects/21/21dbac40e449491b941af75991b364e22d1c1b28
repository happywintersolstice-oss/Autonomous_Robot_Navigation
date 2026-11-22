"""Feature semantics, real collector targets, non-invasive traces and pairing."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from robot_navigation.agents import QLearning, SARSA, load_policy
from robot_navigation.core import GridMap, NavigationEnv, Task, ROOT, load_task, read_json, write_json
from robot_navigation.dqn import CoordinateObservation, DQNPolicy, build_model
from robot_navigation.observation_encoding import COORDINATES, ABSOLUTE_RELATIVE, encode
from robot_navigation.part_a import TrainingLedger
from robot_navigation.representation import summarize, validate_training_prefix, SEEDS
from robot_navigation.training import train_loop
from robot_navigation.training_traces import TrainingTraces, trace_limit


def fixture():
    task=Task(GridMap('representation-fixture',('...',)),{**load_task().config,'max_steps':1})
    scenario={'id':'fixture','start':[0,0],'goal':[0,2],'shortest_length':2}
    config=dict(algorithm='dqn',run_id='fixture',seed=7,interactions=200,monitor_every=100,
                learning_rate=.001,discount=.99,buffer_size=1000,learning_starts=4,batch_size=8,
                train_freq=1,gradient_steps=1,target_update_interval=20,epsilon_start=1.,epsilon_end=.1,
                epsilon_decay_fraction=.8,net_arch=[16,16],torch_threads=1,training_trace_episodes=3)
    return task,scenario,config


def test_relative_features_retain_absolute_context_and_transition_contract():
    first=encode([0,0,0,2],(3,3),ABSOLUTE_RELATIVE)
    second=encode([1,0,1,2],(3,3),ABSOLUTE_RELATIVE)
    assert np.array_equal(first[4:],second[4:]) and not np.array_equal(first[:4],second[:4])
    assert np.array_equal(encode([2,2,0,0],(3,3),ABSOLUTE_RELATIVE),[1,1,0,0,-1,-1])
    task,scenario,_=fixture()
    raw=NavigationEnv(task,[scenario]);wrapped=CoordinateObservation(NavigationEnv(task,[scenario]),ABSOLUTE_RELATIVE)
    raw_obs,_=raw.reset(seed=7);obs,_=wrapped.reset(seed=7)
    assert wrapped.observation_space.contains(obs) and obs.dtype==np.float32
    assert np.array_equal(obs,encode(raw_obs,task.grid.shape,ABSOLUTE_RELATIVE))
    actual,*flags=raw.step(3);encoded,*wrapped_flags=wrapped.step(3)
    assert flags==wrapped_flags and np.array_equal(encoded,encode(actual,task.grid.shape,ABSOLUTE_RELATIVE))
    with pytest.raises(ValueError,match='encoding'):
        encode(raw_obs,task.grid.shape,'unknown')


@pytest.mark.parametrize('goal,done,timeout',[([0,2],0.,1.),([0,1],1.,0.)])
def test_relative_sb3_collector_preserves_terminal_observation_and_bootstrap(goal,done,timeout):
    task,scenario,config=fixture();scenario={**scenario,'goal':goal,'shortest_length':goal[1]}
    model=build_model(NavigationEnv(task,[scenario]),{**config,'observation_encoding':ABSOLUTE_RELATIVE})
    model._sample_action=lambda *args:(np.array([3]),np.array([3]))
    model.learn(total_timesteps=1)
    assert model.replay_buffer.timeouts[0,0]==timeout
    assert model.replay_buffer.sample(1).dones.item()==done
    assert np.array_equal(model.replay_buffer.next_observations[0,0],encode([0,1,*goal],task.grid.shape,ABSOLUTE_RELATIVE))


def test_neural_trace_capture_does_not_change_seeded_weights_replay_or_ledger(tmp_path):
    task,scenario,config=fixture();config={**config,'observation_encoding':ABSOLUTE_RELATIVE}
    models=[];logs=[];visits=[]
    for enabled in (False,True):
        trace=TrainingTraces(task,tmp_path,config) if enabled else None
        rows=[];ledger=TrainingLedger(NavigationEnv(task,[scenario]),rows.append,trace)
        model=build_model(ledger,config);model.learn(total_timesteps=64)
        if trace:
            saved=trace.finish()
            assert len(saved['records'])==3
            for item in saved['records']:
                validate_training_prefix(task,read_json(tmp_path/item['path']))
        models.append(model);logs.append(rows);visits.append(ledger.visitation)
    assert logs[0]==logs[1] and np.array_equal(visits[0],visits[1])
    assert all(torch.equal(models[0].q_net.state_dict()[k],v) for k,v in models[1].q_net.state_dict().items())
    assert np.array_equal(models[0].replay_buffer.actions,models[1].replay_buffer.actions)
    policy=DQNPolicy(task,models[1],'fixture');path=tmp_path/'relative.zip'
    before=policy.state_fingerprint();scores=policy.scores([0,0,0,2])
    assert policy.state_fingerprint()==before
    policy.save(path,{'training_interactions':64})
    loaded=load_policy(path,task)
    assert loaded.encoding==ABSOLUTE_RELATIVE and np.array_equal(loaded.scores([0,0,0,2]),scores)
    metadata=read_json(path.with_suffix('.json'));metadata['encoding']=COORDINATES
    write_json(path.with_suffix('.json'),metadata)
    with pytest.raises(ValueError,match='encoding mismatch'):
        load_policy(path,task)


@pytest.mark.parametrize('family',[QLearning,SARSA])
def test_tabular_trace_capture_preserves_training_actions_updates_and_outcomes(tmp_path,family):
    task,scenario,config=fixture();task=replace(task,config={**task.config,'max_steps':4})
    config={**config,'algorithm':family.algorithm,'learning_rate':.3}
    policies=[];records=[];visits=[]
    for enabled in (False,True):
        policy=family(task);rows=[];trace=TrainingTraces(task,tmp_path,config) if enabled else None
        visitation,_=train_loop(task,[scenario],policy,config,on_episode=rows.append,trace=trace)
        if trace:
            assert len(trace.finish()['records'])==3
        policies.append(policy);records.append(rows);visits.append(visitation)
    assert np.array_equal(policies[0].q,policies[1].q)
    assert records[0]==records[1] and np.array_equal(visits[0],visits[1])


def test_incomplete_training_trace_is_not_fabricated_as_terminal_episode(tmp_path):
    task,scenario,config=fixture();task=replace(task,config={**task.config,'max_steps':4})
    trace=TrainingTraces(task,tmp_path,config);env=NavigationEnv(task,[scenario]);obs,_=env.reset(seed=7)
    trace.start(obs,scenario,0);nxt,reward,terminated,truncated,info=env.step(3)
    trace.transition(obs,3,reward,nxt,terminated,truncated,info,1)
    records=trace.finish()['records'];record=read_json(tmp_path/records[0]['path'])
    assert not record['complete'] and 'metrics' not in record
    assert record['cutoff_reason']=='budget_cutoff'
    validate_training_prefix(task,record)
    broken=deepcopy(record);broken['metrics']={'outcome':'timeout'}
    with pytest.raises(ValueError,match='terminal metrics'):
        validate_training_prefix(task,broken)
    assert trace.finish()['records']==records


@pytest.mark.parametrize('limit',[-1,11,True,1.5])
def test_training_trace_limit_is_bounded_and_integer(limit):
    with pytest.raises(ValueError,match='0 to 10'):
        trace_limit({'training_trace_episodes':limit})


def test_representation_summary_uses_all_fresh_paired_seeds_and_rejects_incomplete_matrix():
    originals=[r for r in read_json(ROOT/'artifacts/studies/part-a-v1/run_results.json') if r['algorithm']=='dqn']
    rows=[]
    for variant,encoding in (('control',COORDINATES),('relative',ABSOLUTE_RELATIVE)):
        for seed,source in zip(SEEDS,originals):
            rows.append({**deepcopy(source),'variant':variant,'encoding':encoding,'seed':seed})
    summary=summarize(rows,load_task().signature)
    assert summary['paired_change']['success_rate']['mean']==0.
    assert summary['paired_change']['success_rate']['sample_sd']==0.
    assert summary['final_test_evaluated'] is False
    with pytest.raises(ValueError,match='ten'):
        summarize(rows[:-1],load_task().signature)
    rows[-1]['seed']=101
    with pytest.raises(ValueError,match='paired seeds'):
        summarize(rows,load_task().signature)


def test_real_relative_pilot_replays_with_encoding_label_and_training_trace_downloads():
    from streamlit.testing.v1 import AppTest
    directory=ROOT/'artifacts/runs/m8-relative-dqn-pilot-seed1701'
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=20).run()
    next(w for w in app.selectbox if w.label=='Trained model').set_value(directory).run()
    assert not app.exception
    assert any(ABSOLUTE_RELATIVE in c.value and '6 features' in c.value for c in app.caption)
    next(b for b in app.button if b.label=='Next step').click().run()
    player=app.session_state['playback']
    assert player.index==1 and player.current['position']==player.episode['transitions'][0]['next_observation'][:2]


def test_published_representation_dashboard_keeps_original_comparison_separate():
    from streamlit.testing.v1 import AppTest
    from robot_navigation.representation import saved_summary
    summary=saved_summary()
    assert summary is not None and summary['original_final_selection_changed'] is False
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=20).run()
    view=lambda:next(w for w in app.selectbox if w.label=='Dashboard view')
    view().set_value('Optimization and robustness').run()
    assert not app.exception
    inputs=app.dataframe[0].value
    assert list(inputs['Inputs'])==['control','relative']
    assert list(inputs['Online parameters'])==[4740,4868]
    assert list(app.dataframe[1].value['seed'])==SEEDS
    assert any('Four baseline algorithms' in heading.value for heading in app.subheader)
    view().set_value('Learning analysis').run()
    assert not app.exception
    assert any('DQN input representation' in heading.value for heading in app.subheader)
    view().set_value('Agent comparison').run()
    assert not app.exception
    assert list(app.dataframe[0].value['Algorithm'])==['Q-learning','SARSA','DQN','PPO']
