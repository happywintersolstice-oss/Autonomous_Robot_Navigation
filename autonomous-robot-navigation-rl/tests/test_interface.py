"""Presentation checks using actual saved validation evidence."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

import pytest
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.errors import AppTestError

from robot_navigation.core import ROOT, read_json
from robot_navigation.interface import preferred_runs, repeated_pattern, run_label


def test_loop_indicator_uses_only_displayed_actions_and_learned_policies():
    episode=read_json(ROOT/'artifacts/runs/smoke-seed7/replays/q_learning_timeout.json')
    assert repeated_pattern(episode,6) is None
    assert repeated_pattern(episode,7)==2
    prefix=deepcopy(episode)
    prefix['transitions']=prefix['transitions'][:7]
    assert repeated_pattern(prefix,7)==repeated_pattern(episode,7)
    prefix['policy_id']='random/seed7'
    assert repeated_pattern(prefix,7) is None
    assert repeated_pattern(episode,0) is None


def test_run_inventory_identifies_the_predeclared_system():
    runs=list((ROOT/'artifacts/runs').glob('*'))
    selected=preferred_runs(runs)[0]
    assert selected.name=='m4-tuned-q-learning-seed101'
    assert 'Selected system' in run_label(selected)
    assert 'Baseline' in run_label(ROOT/'m4-control-q-learning-seed101')
    assert 'seed 101' in run_label(selected)


def test_default_comparison_holds_collision_while_selected_system_moves():
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=20).run()
    next(r for r in app.radio if r.label=='Episode source').set_value('Matched comparison').run()
    assert not app.exception
    second=next(b for b in app.selectbox if b.label=='Compare with run')
    assert second.value.name=='m4-control-q-learning-seed101'
    player=app.session_state['comparison_playback']
    assert [len(e['transitions']) for e in player.episodes]==[3,1]
    app.slider[0].set_value(2).run()
    assert not app.exception
    player=app.session_state['comparison_playback']
    assert [s['index'] for s in player.current]==[2,1]
    assert [s['status'] for s in player.current]==['running','collision']
    assert any('Terminal frame held' in m.value for m in app.markdown)
    assert [m.value for m in app.metric]==['2 / 200','-0.02','1 / 200','-5.01']
    next(b for b in app.button if b.label=='Reset').click().run()
    assert app.session_state['comparison_playback'].index==0


def test_continuous_playback_ignores_event_seek_until_paused():
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=20).run()
    next(b for b in app.button if b.label=='Play').click().run()
    for index in (1,2,3):
        app.session_state['last_tick']=0.
        app.run()
        assert not app.exception
        assert app.session_state['playback'].index==index
        event=next(b for b in app.selectbox if b.label=='Seek displayed event')
        assert event.disabled==(index<3)
        if index==1:
            with pytest.raises(AppTestError,match='disabled'):
                event.set_value(0)
            assert app.session_state['playback'].playing
            assert app.session_state['playback'].index==1
    assert not app.session_state['playback'].playing
    assert app.session_state['playback'].current['status']=='success'
    next(b for b in app.selectbox if b.label=='Seek displayed event').set_value(0).run()
    assert app.session_state['playback'].index==0


def test_parallel_map_images_are_isolated_and_share_fixed_geometry(monkeypatch):
    import robot_navigation.interface as ui
    from robot_navigation.core import load_task

    grid=load_task().grid
    episodes=[read_json(ROOT/f'artifacts/runs/m4-tuned-sarsa-seed{seed}/replays/sarsa_success.json')
              for seed in (503,211)]
    original_draw,original_close=ui.draw,ui.plt.close
    guard=Lock()
    active=peak=0
    positions=[]
    def observed_draw(*args,**kwargs):
        nonlocal active,peak
        with guard:
            active+=1
            peak=max(peak,active)
        sleep(.01)  # Overlap competing session requests without serialization.
        return original_draw(*args,**kwargs)
    def observed_close(figure):
        nonlocal active
        if hasattr(figure,'axes'):
            assert len(figure.axes)==1 and len(figure.axes[0].images)==1
            assert figure.get_layout_engine() is None
            positions.append(figure.axes[0].get_position().bounds)
            with guard:
                active-=1
        original_close(figure)
    monkeypatch.setattr(ui,'draw',observed_draw)
    monkeypatch.setattr(ui.plt,'close',observed_close)
    jobs=[(episode,index) for index in (0,1) for episode in episodes]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pngs=list(pool.map(lambda job:ui.map_png.__wrapped__(grid,*job),jobs))
    assert peak==1 and active==0
    assert all(p==positions[0] for p in positions)
    # Same map, start, goal and prefix must produce the same map image,
    # regardless of which checkpoint supplied the episode.
    assert pngs[0]==pngs[1]
