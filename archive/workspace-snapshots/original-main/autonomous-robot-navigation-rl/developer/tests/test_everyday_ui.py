"""Everyday flows use valid cells and retain the scientific meaning of results."""

from navigation import benchmark_app
import pytest
from streamlit.testing.v1 import AppTest

from robot_navigation.core import ROOT,load_task
from robot_navigation.navigation_lab import load_layouts,make_task,configuration
from robot_navigation.release import final_summary


@pytest.mark.parametrize('structured',[False,True])
def test_free_square_picker_generates_the_selected_route_without_training(structured):
    app = benchmark_app(str(ROOT/'app/streamlit_app.py'),default_timeout=30).run()
    task = load_task()
    if structured:
        layout = load_layouts()[0]
        next(w for w in app.selectbox if w.label=='Navigation map').set_value(layout['map_id']).run()
        task = make_task(layout,'time-cost',configuration())
    app.session_state['replay_tab'] = 'Choose a route'
    app.run()
    cells = task.grid.free_cells
    def select(label):
        return next(w for w in app.selectbox if w.label==label)
    assert len(select('Starting square').options)==len(cells)
    assert len(select('Destination').options)==len(cells)
    start,goal = cells[1],cells[-2]
    select('Starting square').set_value(start).run()
    select('Destination').set_value(goal).run()
    select('Frozen policy').set_value('BFS')
    next(b for b in app.button if b.label=='Generate demonstration').click().run()
    assert not app.exception and not app.error
    player = app.session_state['lab_playback' if structured else 'playback']
    assert player.episode['scenario']['start']==list(start)
    assert player.episode['scenario']['goal']==list(goal)
    assert player.episode['metrics']['outcome']=='success'
    assert not player.playing and player.index==0


def test_results_intro_uses_selected_model_final_metrics_and_labels_the_cohort():
    app = benchmark_app(str(ROOT/'app/streamlit_app.py'),default_timeout=30).run()
    next(b for b in app.button if b.label=='Results').click().run()
    summary = final_summary()
    selected = next(row for system in summary['systems'].values() for row in system['seed_results']
                    if row['id']==summary['selected_system'])['evaluation']
    assert not app.exception
    assert [m.value for m in app.metric][:3]==[
        f"{100*selected[key]:.1f}%" for key in ('success_rate','collision_rate','timeout_rate')]
    assert any('held-out start' in c.value and 'familiar original map' in c.value for c in app.caption)


def test_old_tab_session_migrates_to_visible_watch_page():
    app = benchmark_app(str(ROOT/'app/streamlit_app.py'),default_timeout=30)
    app.session_state['replay_tab'] = 'Trained replay'
    app.run()
    assert not app.exception
    assert app.session_state['replay_tab']=='Watch the robot'
    assert app.session_state['playback'].index==0
