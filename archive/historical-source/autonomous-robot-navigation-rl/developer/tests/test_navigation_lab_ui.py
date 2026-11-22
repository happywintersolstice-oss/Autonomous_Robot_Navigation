"""New maps retain route identity and recorded rewards while models stay fixed."""
from streamlit.testing.v1 import AppTest

from navigation import navigate
from robot_navigation.core import ROOT


def test_lab_replays_new_maps_and_preserves_route_on_reward_switch(monkeypatch):
    import robot_navigation.navigation_lab as lab
    app = AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=30).run()
    def forbidden(*args,**kwargs):
        raise AssertionError('Opening the map lab must not train')
    monkeypatch.setattr(lab,'run',forbidden)
    layout = lab.load_layouts()[0]
    next(w for w in app.selectbox if w.label=='Navigation map').set_value(layout['map_id']).run()
    assert not app.exception and app.session_state['dashboard_page']=='Navigation simulator'
    assert len(next(w for w in app.selectbox if w.label=='Navigation map').options)==5
    player = app.session_state['lab_playback']
    assert player.episode['metrics']['outcome']=='success'
    route = player.episode['scenario']
    next(b for b in app.button if b.key=='lab_next').click().run()
    assert app.session_state['lab_playback'].index==1
    next(w for w in app.selectbox if w.label=='Reward design').set_value('original').run()
    assert not app.exception
    player = app.session_state['lab_playback']
    assert player.index==0 and player.episode['scenario']==route
    assert player.episode['signature']['reward_version']=='structured-original-v1'
    next(b for b in app.button if b.key=='lab_example_timeout').click().run()
    assert app.session_state['lab_playback'].episode['metrics']['outcome']=='timeout'
    assert app.session_state['lab_playback'].index==0
    next(b for b in app.button if b.key=='lab_example_success').click().run()
    player = app.session_state['lab_playback']
    assert player.episode['metrics']['outcome']=='success'
    next(b for b in app.button if b.key=='lab_play').click().run()
    assert player.playing
    navigate(app,'Methodology')
    assert not app.exception and not player.playing


def test_map_lab_is_independent_of_original_validation(monkeypatch):
    import robot_navigation.benchmark as benchmark
    def absent(*args,**kwargs):
        raise FileNotFoundError('old validation absent')
    monkeypatch.setattr(benchmark,'load_scenarios',absent)
    app = AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=30)
    app.session_state['dashboard_page'] = 'Map & reward lab'
    app.run()
    assert not app.exception and app.dataframe
