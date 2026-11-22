"""One map selector and tab set for original and structured navigation models."""
import streamlit as st

from .benchmark import load_scenarios
from .core import ROOT,load_task
from .dashboard_ui import render_comparison
from .evidence import EXPECTED_ERRORS
from .interface import preferred_runs,run_selector,workspace_header
from .simulator_ui import pause_players,render_replay

ORIGINAL = 'bottleneck-v1'
REPLAY_TABS = ['Watch the robot','Learning history','Compare robots','Choose a route']


def reset_demo_players():
    pause_players()
    for key in ('playback','comparison_playback','lab_playback','lab_comparison_playback'):
        player = st.session_state.get(key)
        if player is not None:
            player.reset()


def render_demo(root=ROOT):
    layouts = []
    try:
        from .navigation_lab import BENCHMARK,load_layouts
        if (root/'data/navigation-lab'/BENCHMARK/'manifest.json').is_file():
            layouts = load_layouts(root)
    except EXPECTED_ERRORS as exc:
        st.info(f'Structured maps unavailable ({exc}). The original map remains available.')
    by_id = {m['map_id']:m for m in layouts}
    options = [ORIGINAL,*by_id]
    if st.session_state.get('demo_map') not in options:
        st.session_state['demo_map'] = ORIGINAL
    workspace_header('', 'Navigation simulator')
    st.write('Choose a map, then press **Play** to watch the robot find its way to the goal.')
    names = {ORIGINAL:'Original map - narrow passage','dead-end-maze':'Maze with dead ends',
             'looped-corridors':'Looped corridors','rooms-and-doorways':'Rooms and doorways',
             'deep-cul-de-sacs':'Long dead ends'}
    selected = st.selectbox('Navigation map',options,key='demo_map',on_change=reset_demo_players,
        format_func=lambda value:names[ORIGINAL if value==ORIGINAL else by_id[value]['family']])
    if st.session_state.get('demo_active_map')!=selected:
        reset_demo_players()
        st.session_state['demo_active_map'] = selected
    descriptions = {ORIGINAL:'An open map with a narrow bottleneck. This uses the original selected model.',
        'dead-end-maze':'Narrow passages and wrong turns that require the robot to backtrack.',
        'looped-corridors':'Connected corridors with several ways around and some dead ends.',
        'rooms-and-doorways':'Open rooms linked by narrow doorways.',
        'deep-cul-de-sacs':'Long dead ends that make planning and backtracking difficult.'}
    st.caption(descriptions.get(ORIGINAL if selected==ORIGINAL else by_id[selected]['family'], ''))
    st.caption('R = robot | S = start | G = goal | dark squares = walls. Playback does not train the model.')
    task,validation,runs = None,None,None
    if selected==ORIGINAL:
        try:
            task = load_task(root)
            validation = load_scenarios(task,'validation',root)
            runs = preferred_runs(p for p in (root/'artifacts/runs').glob('*') if (p/'manifest.json').exists())
        except EXPECTED_ERRORS as exc:
            st.error(f'Fixed-layout validation unavailable: {exc}. Restore the matching map and validation data, or select a structured map.')
            return
        if not runs:
            st.info('No saved models are available yet.')
            return
    old_tabs = {'Trained replay':'Watch the robot','Learning replay':'Learning history',
                'Compare agents':'Compare robots','Try a route':'Choose a route'}
    if st.session_state.get('replay_tab') in old_tabs:
        st.session_state['replay_tab'] = old_tabs[st.session_state['replay_tab']]
    tabs = st.tabs(REPLAY_TABS,key='replay_tab',on_change='rerun')
    if st.session_state.get('active_replay_tab')!=st.session_state['replay_tab']:
        pause_players()
        st.session_state['active_replay_tab'] = st.session_state['replay_tab']
    modes = {'Watch the robot':'trained','Learning history':'learning','Compare robots':'compare','Choose a route':'custom'}
    for label,tab in zip(REPLAY_TABS,tabs):
        if not tab.open:
            continue
        with tab:
            if selected!=ORIGINAL:
                from .structured_demo_ui import render_structured
                render_structured(by_id[selected],root,modes[label])
            elif label=='Compare robots':
                st.write('Watch different agents tackle the same start and goal. Their steps advance together.')
                count = st.radio('What would you like to compare?',['Two agents','Four algorithms'],
                                 horizontal=True,key='comparison_size',on_change=pause_players)
                if count=='Four algorithms':
                    render_comparison(task,None,validation,runs,root,four=True)
                else:
                    with st.container(key='model_setup'):
                        directory = run_selector(runs,label='First trained model')
                    if directory is not None:
                        render_comparison(task,directory,validation,runs,root)
            else:
                render_replay(task,validation,runs,root,mode=modes[label])
