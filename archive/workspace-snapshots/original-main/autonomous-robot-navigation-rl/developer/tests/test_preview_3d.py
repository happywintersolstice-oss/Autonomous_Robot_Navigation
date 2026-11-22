"""Temporary 3D rendering preserves the grid, visible history and playback state."""

from navigation import benchmark_app
from copy import deepcopy
import json

import pytest
from streamlit.testing.v1 import AppTest

from robot_navigation.core import ROOT,load_task,read_json
from robot_navigation.navigation_lab import load_layouts
from robot_navigation.preview_3d import scene_data,make_deck


def test_scene_uses_exact_walls_and_only_the_displayed_replay_prefix():
    grid = load_task().grid
    episode = read_json(ROOT/'artifacts/runs/m4-tuned-q-learning-seed101/replays/q_learning_success.json')
    original = deepcopy(episode)
    data = scene_data(grid,episode,1,visits=True)
    observed = deepcopy(episode)
    observed['transitions'] = episode['transitions'][:1]
    assert data==scene_data(grid,observed,1,visits=True)
    assert episode==original
    height,width = grid.shape
    assert len(data['floor'])==height*width
    assert {(cell['polygon'][0][0],height-cell['polygon'][0][1]-1) for cell in data['walls']}=={
        (column,row) for row,values in enumerate(grid.rows) for column,value in enumerate(values) if value=='#'}
    expected = episode['transitions'][0]['next_observation'][:2]
    assert data['state']['position']==expected
    assert len(data['paths'][0]['path'])==2
    assert data['markers'][-1]['position'][:2]==[expected[1]+.5,height-expected[0]-.5]
    assert not scene_data(grid,episode,0)['paths']


def test_collision_appears_only_at_its_recorded_action_and_scene_has_no_basemap():
    episode = read_json(ROOT/'artifacts/runs/smoke-seed7/replays/q_learning_collision.json')
    grid = load_task().grid
    assert not scene_data(grid,episode,0)['collision']
    data = scene_data(grid,episode,1)
    assert len(data['collision'])==1
    assert data['state']['position']==episode['initial_observation'][:2]
    assert data['state']['attempted_position']!=data['state']['position']
    spec = json.loads(make_deck(grid,episode,1).to_json())
    assert spec['views']==[{'@@type':'OrbitView','controller':True,'orbitAxis':'Z'}]
    assert 'mapProvider' not in spec and 'mapStyle' not in spec
    assert all(layer['coordinateSystem']=='cartesian' for layer in spec['layers'])
    layers = {layer['id']:layer for layer in spec['layers']}
    assert layers['floor']['lineWidthUnits']=='pixels'
    assert layers['trail']['widthUnits']=='pixels'
    assert layers['markers']['radiusUnits']=='common'
    assert layers['labels']['fontFamily']=='Arial'


@pytest.mark.parametrize('structured',[False,True])
def test_temporary_view_switch_pauses_without_resetting_or_changing_a_route(structured):
    app = benchmark_app(str(ROOT/'app/streamlit_app.py'),default_timeout=30).run()
    assert not app.toggle[0].value
    assert not app.get('deck_gl_json_chart')
    if structured:
        next(w for w in app.selectbox if w.label=='Navigation map').set_value(load_layouts()[0]['map_id']).run()
    key = 'lab_playback' if structured else 'playback'
    next(b for b in app.button if b.label=='Next step').click().run()
    player = app.session_state[key]
    expected = deepcopy(player.episode)
    next(b for b in app.button if b.label=='Play').click().run()
    before = player.index
    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert not player.playing and player.index==before and player.episode==expected
    assert len(app.get('deck_gl_json_chart'))==1
    assert any('3D' in text.value and 'TEMPORARY' in text.value for text in app.markdown)
    app.toggle[0].set_value(False).run()
    assert not app.exception and not app.get('deck_gl_json_chart')
    assert player.index==before and player.episode==expected


def test_comparison_has_independent_3d_panels_and_returns_to_2d():
    app = benchmark_app(str(ROOT/'app/streamlit_app.py'),default_timeout=30).run()
    app.toggle[0].set_value(True).run()
    app.session_state['replay_tab'] = 'Compare robots'
    app.run()
    assert not app.exception and len(app.get('deck_gl_json_chart'))==2
    player = app.session_state['comparison_playback']
    assert player.episodes[0]['scenario']==player.episodes[1]['scenario']
    next(b for b in app.button if b.label=='Next step').click().run()
    assert player.index==1
    app.toggle[0].set_value(False).run()
    assert not app.exception and not app.get('deck_gl_json_chart')
    assert player.index==1 and not player.playing
