"""Recovery comparisons use saved actions, separate RNG labels and paused clocks."""

from navigation import benchmark_app
from streamlit.testing.v1 import AppTest

from navigation import navigate
from robot_navigation.core import ROOT


def test_recovery_comparison_uses_recordings_and_pauses_on_route_switch(monkeypatch):
    import robot_navigation.unseen_agents as agents
    def forbidden(*args,**kwargs):
        raise AssertionError('Recovery playback cannot query a policy')
    monkeypatch.setattr(agents,'load_map_policy',forbidden)
    app = benchmark_app(str(ROOT/'app/streamlit_app.py'),default_timeout=30).run()
    navigate(app,'Optimization and robustness')
    assert not app.exception
    player = app.session_state['recovery_playback']
    assert player.index==0 and len(player.episodes)==2
    assert player.episodes[0]['scenario']==player.episodes[1]['scenario']
    assert player.episodes[0]['metrics']['outcome']=='timeout'
    assert player.episodes[1]['metrics']['outcome']=='success'
    next(b for b in app.button if b.key=='recovery_next').click().run()
    assert app.session_state['recovery_playback'].index==1
    next(b for b in app.button if b.key=='recovery_play').click().run()
    player = app.session_state['recovery_playback']
    assert player.playing
    navigate(app,'Methodology')
    assert not app.exception and not player.playing
