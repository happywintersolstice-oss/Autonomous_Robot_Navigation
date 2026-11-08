"""Scheduled actual-training captures must not alter learning or select outcomes."""
from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml

from robot_navigation.agents import QLearning, SARSA
from robot_navigation.core import ROOT, GridMap, Task, NavigationEnv, load_task, read_json
from robot_navigation.learning_replays import demo_configs
from robot_navigation.part_a import TrainingLedger
from robot_navigation.representation import validate_training_prefix
from robot_navigation.training import train_loop
from robot_navigation.training_traces import TrainingTraces, trace_limit


def fixture():
    task = Task(GridMap('recording-fixture', ('...',)), {**load_task().config,'max_steps':1})
    scenario = {'id':'fixture','start':[0,0],'goal':[0,2],'shortest_length':2}
    config = dict(algorithm='q-learning',run_id='fixture',seed=1109,interactions=80,monitor_every=20,
                  learning_rate=.3,discount=.99,epsilon_start=1.,epsilon_end=.1,epsilon_decay_fraction=.8,
                  training_trace_episodes=4,training_trace_starts=[0,40])
    return task, scenario, config


@pytest.mark.parametrize('family', [QLearning,SARSA])
def test_recording_windows_observe_all_stages_without_changing_tabular_learning(tmp_path, family):
    task, scenario, config = fixture()
    config['algorithm'] = family.algorithm
    outputs = []
    for enabled in (False, True):
        policy = family(task)
        rows = []
        trace = TrainingTraces(task, tmp_path, config) if enabled else None
        visits, _ = train_loop(task,[scenario],policy,config,on_episode=rows.append,trace=trace)
        outputs.append((policy.q.copy(), visits, rows))
        if trace:
            saved = trace.finish()
            assert saved['window_starts']==[0,40] and saved['episodes_per_window']==2
            records = [read_json(tmp_path/item['path']) for item in saved['records']]
            assert [r['training_start_interaction'] for r in records]==[0,1,40,41]
            assert [r['recording_window_index'] for r in records]==[0,0,1,1]
            assert all(r['transitions'][0]['scores'] is None for r in records)
            for record in records:
                validate_training_prefix(task,record)
    assert np.array_equal(outputs[0][0], outputs[1][0])
    assert np.array_equal(outputs[0][1], outputs[1][1])
    assert outputs[0][2] == outputs[1][2]


@pytest.mark.parametrize('algorithm', ['dqn','ppo'])
def test_scheduled_neural_recordings_preserve_seeded_training_weights_and_actions(tmp_path, algorithm):
    from robot_navigation.dqn import build_model as dqn_model
    from robot_navigation.ppo import build_model as ppo_model
    task, scenario, config = fixture()
    config.update(algorithm=algorithm, learning_rate=.001,net_arch=[16,16],torch_threads=1,
                  buffer_size=1000,learning_starts=4,batch_size=8,train_freq=1,gradient_steps=1,
                  target_update_interval=20,n_steps=8,n_epochs=2,gae_lambda=.95,clip_range=.2,
                  ent_coef=0.,vf_coef=.5,max_grad_norm=.5)
    outputs = []
    for enabled in (False,True):
        rows=[]
        trace=TrainingTraces(task,tmp_path,config) if enabled else None
        ledger=TrainingLedger(NavigationEnv(task,[scenario]), rows.append, trace)
        model=(dqn_model if algorithm=='dqn' else ppo_model)(ledger,config)
        model.learn(total_timesteps=80)
        outputs.append((deepcopy(model.policy.state_dict()),ledger.visitation.copy(),rows))
        if trace:
            records=[read_json(tmp_path/item['path']) for item in trace.finish()['records']]
            assert [r['training_start_interaction'] for r in records]==[0,1,40,41]
            for record in records:
                validate_training_prefix(task,record)
    assert all(torch.equal(outputs[0][0][key], value) for key,value in outputs[1][0].items())
    assert np.array_equal(outputs[0][1],outputs[1][1]) and outputs[0][2]==outputs[1][2]


@pytest.mark.parametrize('starts,limit', [([],4),([1,40],4),([0,40,40],6),([0,80],4),([0,True],4),([0,40],3),([0],0)])
def test_invalid_recording_schedules_fail_before_training(starts,limit):
    _,_,config=fixture()
    with pytest.raises(ValueError,match='Trace windows'):
        trace_limit({**config,'training_trace_starts':starts,'training_trace_episodes':limit})


def test_demo_plan_uses_baseline_settings_and_separate_identities():
    plan=yaml.safe_load((ROOT/'configs/learning_replays.yaml').read_text())
    configs=demo_configs(plan,ROOT)
    assert [c['algorithm'] for c in configs]==['q-learning','sarsa','dqn','ppo']
    changed={'run_id','seed','stage','training_trace_episodes','training_trace_starts','log_progress','max_wall_seconds'}
    for config in configs:
        old=read_json(ROOT/'artifacts/runs'/plan['sources'][config['algorithm']]/'configuration.json')['training']
        assert all(config[key]==value for key,value in old.items() if key not in changed)
        assert config['run_id'].startswith('learning-replay-') and config['seed']==1109
        assert config['training_trace_episodes']==10 and config['interactions']==200000


@pytest.mark.parametrize('algorithm', ['q-learning','sarsa','dqn','ppo'])
def test_actual_demo_family_and_late_training_recording_are_playable(algorithm, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from robot_navigation.dashboard import training_paths
    import robot_navigation.simulator_ui as ui
    directory=ROOT/f'artifacts/runs/learning-replay-{algorithm}-seed1109'
    viewer=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=30).run()
    def forbidden(*args,**kwargs):
        pytest.fail('Learning replay must use recorded actions, not trained-model inference')
    monkeypatch.setattr(ui,'load_policy',forbidden)
    viewer.session_state['replay_tab']='Learning replay';viewer.run()
    next(w for w in viewer.selectbox if w.label=='Training run').set_value(directory).run()
    assert not viewer.exception
    paths=training_paths(directory)
    assert len(paths)==10
    selector=next(w for w in viewer.selectbox if w.label=='Training episode')
    assert any('95% training stage' in option for option in selector.options)
    selector.set_value(paths[-1]).run()
    expected=read_json(paths[-1])
    assert expected['training_start_interaction']>=190000
    assert viewer.session_state['playback'].episode==expected
    next(w for w in viewer.button if w.label=='Next step').click().run()
    assert not viewer.exception
    state=viewer.session_state['playback'].current
    assert state['index']==1 and state['position']==expected['transitions'][0]['next_observation'][:2]
    assert state['return']==expected['transitions'][0]['reward']
    assert all('Q score' not in df.value.columns and 'Action probability' not in df.value.columns for df in viewer.dataframe)


def test_all_recorded_demo_artifacts_pass_declared_window_audit():
    from robot_navigation.learning_replays import audit_demo
    status=read_json(ROOT/'artifacts/studies/learning-replays-v1/manifest.json')
    assert status['status']=='verified' and status['total_training_interactions']==800000
    assert status['recorded_episodes']==40 and status['new_final_test_policy_executions']==0
    for algorithm in ('q-learning','sarsa','dqn','ppo'):
        audited=audit_demo(ROOT/f'artifacts/runs/learning-replay-{algorithm}-seed1109',ROOT)
        assert len(audited['recordings'])==10
        assert [r['window_start'] for r in audited['recordings']]==[0,0,50000,50000,100000,100000,150000,150000,190000,190000]


def test_demo_audit_accepts_only_floating_accumulation_noise(tmp_path):
    import csv
    import shutil
    from robot_navigation.learning_replays import audit_demo
    source=ROOT/'artifacts/runs/learning-replay-sarsa-seed1109'
    directory=tmp_path/'demo'
    directory.mkdir()
    for name in ('manifest.json','configuration.json','pilot.json','training.csv','visitation.npz'):
        shutil.copy2(source/name,directory/name)
    for name in ('source','training_traces'):
        shutil.copytree(source/name,directory/name)
    with (directory/'training.csv').open(newline='',encoding='utf-8') as file:
        rows=list(csv.DictReader(file)); fields=list(rows[0])
    original=float(rows[0]['return'])
    def save(value):
        rows[0]['return']=str(value)
        with (directory/'training.csv').open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    save(original+5e-14)
    assert len(audit_demo(directory,ROOT)['recordings'])==10
    save(original+.01)
    with pytest.raises(ValueError,match='chronological'):
        audit_demo(directory,ROOT)
