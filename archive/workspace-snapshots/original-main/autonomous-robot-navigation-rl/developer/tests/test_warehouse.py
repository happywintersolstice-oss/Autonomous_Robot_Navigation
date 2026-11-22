"""The default is a real warehouse task; historical artifacts stay incompatible."""
from copy import deepcopy

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from robot_navigation.agents import QLearning
from robot_navigation.core import ROOT
from robot_navigation.episodes import validate_replay
from robot_navigation.warehouse import MAP_ID,saved


def app():
    return AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=30).run()


def test_floor_fixture_geometry_jobs_and_separate_training_split():
    task,layout,routes,summary,records,directory = saved()
    assert task.grid.map_id==MAP_ID and task.grid.shape==(13,17)
    assert len(task.grid.distances(layout['stations'][0]['cell']))==len(task.grid.free_cells)
    for fixture in layout['fixtures']:
        r1,c1,r2,c2 = fixture['bounds']
        assert all(not task.grid.is_free((r,c)) for r in range(r1,r2+1) for c in range(c1,c2+1))
    assert all(task.grid.is_free(s['cell']) for s in layout['stations'])
    train = {s['id'] for s in routes['train']}
    validation = {s['id'] for s in routes['validation']}
    assert not train&validation
    assert {j['scenario']['id'] for j in routes['jobs']}<=validation
    assert len(records)==summary['validation']['episodes']==188
    for record in records:
        validate_replay(task,record)
    q = QLearning.load(directory/'final.npz',task)
    assert np.isfinite(q.q).all()
    with pytest.raises(ValueError,match='mismatch'):
        QLearning.load(ROOT/'artifacts/runs/m4-tuned-q-learning-seed101/selected.npz',task)


def test_default_project_explains_the_warehouse_and_switching_jobs_resets_playback(monkeypatch):
    import robot_navigation.warehouse as warehouse
    def forbidden(*args,**kwargs):
        pytest.fail('Opening the warehouse must not train')
    monkeypatch.setattr(warehouse,'train',forbidden)
    viewer = app()
    assert not viewer.exception and viewer.session_state['demo_map']==MAP_ID
    assert len(next(w for w in viewer.selectbox if w.label=='Navigation map').options)==6
    assert any('Rack B' in m.value and 'packing station' in m.value for m in viewer.markdown)
    assert viewer.session_state['warehouse_playback'].episode['policy_id'].startswith('q-learning/warehouse-q-learning-seed19411')
    next(b for b in viewer.button if b.label=='Play').click().run()
    previous = viewer.session_state['warehouse_playback']
    next(w for w in viewer.selectbox if w.label=='Delivery job').set_value('ship').run()
    assert not viewer.exception and not previous.playing
    assert viewer.session_state['warehouse_playback'].index==0
    assert viewer.session_state['warehouse_playback'].episode['scenario']['start']==[10,8]
    assert viewer.session_state['warehouse_playback'].episode['scenario']['goal']==[10,14]


def test_warehouse_comparison_custom_job_and_3d_use_the_same_floor():
    viewer = app()
    viewer.session_state['replay_tab']='Compare robots';viewer.run()
    assert not viewer.exception
    player = viewer.session_state['warehouse_playback']
    assert len(player.episodes)==4
    assert {e['policy_id'].split('/')[0] for e in player.episodes}=={'q-learning','sarsa','dqn','ppo'}
    assert all(e['scenario']==player.episodes[0]['scenario'] for e in player.episodes)
    assert player.episodes[0]['signature']==player.episodes[1]['signature']
    viewer.toggle[0].set_value(True).run()
    assert len(viewer.get('deck_gl_json_chart'))==4
    viewer.session_state['replay_tab']='Choose a route';viewer.run()
    next(w for w in viewer.selectbox if w.label=='Pickup point').set_value('receiving')
    next(w for w in viewer.selectbox if w.label=='Delivery point').set_value('dispatch')
    next(b for b in viewer.button if b.label=='Create delivery route').click().run()
    assert not viewer.exception
    custom = viewer.session_state['warehouse_playback'].episode
    assert custom['mode']=='custom demonstration'
    assert custom['scenario']['start']==[10,2] and custom['scenario']['goal']==[10,14]
    assert len(viewer.get('deck_gl_json_chart'))==1
    viewer.session_state['replay_tab']='Learning history';viewer.run()
    assert not viewer.exception and not viewer.session_state['warehouse_playback'].playing
    assert viewer.session_state['warehouse_playback'].episode['mode']=='training'
    assert any('chronological exploratory actions' in c.value for c in viewer.caption)


def test_results_are_for_the_selected_warehouse_and_not_the_old_benchmark():
    viewer = app()
    next(b for b in viewer.button if b.label=='Results').click().run()
    assert not viewer.exception and viewer.title[0].value=='Four-algorithm warehouse comparison'
    assert viewer.metric[0].label=='Algorithms compared' and viewer.metric[0].value=='4'
    assert any('188 development routes' in c.value and 'three training seeds' in c.value for c in viewer.caption)
    assert any('familiar warehouse' in i.value for i in viewer.info)


def test_all_four_warehouse_policies_and_three_seeds_are_accessible_and_reset():
    viewer = app()
    def select(label):
        return next(w for w in viewer.selectbox if w.label==label)
    assert select('Algorithm').options==['Q-learning','SARSA','DQN','PPO']
    assert select('Training seed').options==['19411','19421','19431']
    for family in ('sarsa','dqn','ppo'):
        select('Algorithm').set_value(family).run()
        assert not viewer.exception
        assert viewer.session_state['warehouse_playback'].episode['policy_id'].startswith(family+'/')
    next(b for b in viewer.button if b.label=='Next step').click().run()
    previous=viewer.session_state['warehouse_playback']
    select('Training seed').set_value(19421).run()
    assert not viewer.exception and previous.index==0
    assert viewer.session_state['warehouse_playback'].index==0
    assert 'seed19421' in viewer.session_state['warehouse_playback'].episode['policy_id']
    next(c for c in viewer.checkbox if c.label=='Choose from all 188 assessment routes').set_value(True).run()
    assert len(select('Assessment route').options)==188


def test_warehouse_learning_recording_never_queries_a_policy(monkeypatch):
    import robot_navigation.warehouse_ui as ui
    from robot_navigation.core import read_json
    from robot_navigation.warehouse_study import checked_run
    viewer=app()
    def forbidden(*args,**kwargs):
        pytest.fail('Recorded learning playback must not query a model')
    monkeypatch.setattr(ui,'load_policy',forbidden)
    viewer.session_state['replay_tab']='Learning history';viewer.run()
    run,folder=checked_run('q-learning',19411)
    assert not viewer.exception
    assert viewer.session_state['warehouse_playback'].episode==read_json(folder/run['training_traces']['records'][0]['path'])
    assert len(next(w for w in viewer.selectbox if w.label=='Training recording').options)==6
