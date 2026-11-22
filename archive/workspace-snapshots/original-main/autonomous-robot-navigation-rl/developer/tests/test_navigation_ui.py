"""One simulator owns all maps and rejects incompatible reward comparisons."""

from navigation import benchmark_app
from copy import deepcopy

import pytest
from streamlit.testing.v1 import AppTest

from robot_navigation.core import ROOT
from robot_navigation.dashboard import PAGES
from robot_navigation.navigation_lab import load_layouts,configuration,make_task,RewardEnv
from robot_navigation.agents import BFSPolicy
from robot_navigation.episodes import rollout


def app():
    return benchmark_app(str(ROOT/'app/streamlit_app.py'),default_timeout=30).run()


def choose_map(viewer):
    layout = load_layouts()[0]
    next(w for w in viewer.selectbox if w.label=='Navigation map').set_value(layout['map_id']).run()
    return layout


def test_one_simulator_has_six_maps_and_resets_the_old_clock():
    viewer = app()
    assert 'Map & reward lab' not in PAGES
    assert len(next(w for w in viewer.selectbox if w.label=='Navigation map').options)==6
    next(b for b in viewer.button if b.label=='Play').click().run()
    previous = viewer.session_state['playback']
    choose_map(viewer)
    assert not viewer.exception and not previous.playing and previous.index==0
    assert viewer.session_state['dashboard_page']=='Navigation simulator'
    assert viewer.session_state['lab_playback'].index==0
    assert not any(b.label in ('Explore new maps and rewards','Back to original demo') for b in viewer.button)


def test_structured_compare_custom_and_learning_use_the_same_simulator():
    viewer = app()
    layout = choose_map(viewer)
    viewer.session_state['replay_tab']='Compare robots';viewer.run()
    assert not viewer.exception
    player = viewer.session_state['lab_comparison_playback']
    assert player.episodes[0]['scenario']==player.episodes[1]['scenario']
    assert player.episodes[0]['signature']['reward_version']!=player.episodes[1]['signature']['reward_version']
    next(b for b in viewer.button if b.label=='Next step').click().run()
    assert player.index==1
    viewer.session_state['replay_tab']='Choose a route';viewer.run()
    assert not player.playing
    # Form inputs are submitted together, as in a browser; don't force a rerun first.
    next(w for w in viewer.selectbox if w.label=='Frozen policy').set_value('BFS')
    next(b for b in viewer.button if b.label=='Generate demonstration').click().run()
    assert not viewer.exception
    assert viewer.session_state['lab_playback'].episode['mode']=='custom demonstration'
    assert viewer.session_state['lab_playback'].episode['metrics']['outcome']=='success'
    viewer.session_state['replay_tab']='Learning history';viewer.run()
    assert not viewer.exception
    assert any('not recorded' in info.value for info in viewer.info)


def test_reward_comparison_checks_dynamics_and_retains_real_signatures():
    from robot_navigation.reward_playback import RewardComparisonPlayback
    layout = load_layouts()[0];config = configuration()
    tasks = [make_task(layout,c,config) for c in ('original','time-cost')]
    scenario = layout['splits']['validation'][0]
    episodes = [rollout(RewardEnv(t,[scenario]),BFSPolicy(t.grid),scenario) for t in tasks]
    player = RewardComparisonPlayback(episodes,tasks)
    assert player.episodes[0]['signature']!=player.episodes[1]['signature']
    player.seek(1000)
    assert all(s['status']=='success' for s in player.current)
    from dataclasses import replace
    wrong = replace(tasks[1],config={**tasks[1].config,'max_steps':199})
    with pytest.raises(ValueError,match='dynamics'):
        RewardComparisonPlayback(episodes,[tasks[0],wrong])
    altered = deepcopy(episodes)
    altered[1]['scenario'] = {**scenario,'id':'wrong'}
    with pytest.raises(ValueError,match='scenario'):
        RewardComparisonPlayback(altered,tasks)
