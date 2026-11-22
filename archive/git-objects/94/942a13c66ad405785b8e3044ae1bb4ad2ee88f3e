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


def test_control_room_readings_and_reward_chart_use_only_the_visible_prefix():
    from robot_navigation.interface import replay_statistics, reward_history
    episode = read_json(ROOT/'artifacts/runs/smoke-seed7/replays/q_learning_collision.json')
    assert replay_statistics(episode, 0) == {
        'actions':0, 'return':0, 'traveled':0,
        'goal_distance':sum(abs(a-b) for a,b in zip(episode['scenario']['start'], episode['scenario']['goal']))}
    assert reward_history(episode, 0) == {'Step':[0], 'Reward':[0.]}
    observed = deepcopy(episode)
    observed['transitions'] = episode['transitions'][:1]
    assert replay_statistics(episode, 1) == replay_statistics(observed, 1)
    assert reward_history(episode, 1) == reward_history(observed, 1)
    readings = replay_statistics(episode, 1)
    assert readings['return'] == -5.01 and readings['traveled'] == 0
    assert reward_history(episode, 1) == {'Step':[0,1], 'Reward':[0.,-5.01]}


def test_workspace_navigation_opens_existing_pages_and_pauses_replay():
    app = AppTest.from_file(str(ROOT/'app/streamlit_app.py'), default_timeout=30).run()
    next(b for b in app.button if b.label == 'Play').click().run()
    player = app.session_state['playback']
    next(b for b in app.button if b.label == 'Compare results').click().run()
    assert not app.exception and not player.playing
    assert app.session_state['dashboard_page'] == 'Agent comparison'
    next(b for b in app.button if b.label == 'Live map').click().run()
    assert not app.exception and app.session_state['dashboard_page'] == 'Navigation simulator'
    assert not app.session_state['playback'].playing


def test_default_comparison_holds_collision_while_selected_system_moves():
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=20).run()
    app.session_state['replay_tab']='Compare agents'; app.run()
    assert not app.exception
    second=next(b for b in app.selectbox if b.label=='Second trained model')
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


@pytest.mark.parametrize('comparison',[False,True])
def test_delayed_timeline_events_cannot_rewind_or_pause_newer_frame(comparison):
    from robot_navigation.dashboard import seek_from_widget
    from robot_navigation.episodes import Playback, ComparisonPlayback
    episode=read_json(ROOT/'artifacts/runs/m4-tuned-q-learning-seed101/replays/q_learning_success.json')
    player=ComparisonPlayback([episode,deepcopy(episode)]) if comparison else Playback(episode)
    player.play(); player.tick()
    state={'player':player,'position':1,'active':'timeline:1','timeline:0':0,'timeline:1':1}
    # The browser echoed frame zero after the timer had already advanced to one.
    assert not seek_from_widget(state,'timeline:0',player_key='player',position_key='position',active_key='active')
    assert player.index==1 and player.playing and state['position']==1
    # A redundant current-frame event must also leave Play running.
    assert not seek_from_widget(state,'timeline:1',player_key='player',position_key='position',active_key='active')
    assert player.playing
    # A real seek from the currently displayed widget still pauses and rewinds.
    state['timeline:1']=0
    assert seek_from_widget(state,'timeline:1',player_key='player',position_key='position',active_key='active')
    assert player.index==0 and not player.playing and state['position']==0


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
