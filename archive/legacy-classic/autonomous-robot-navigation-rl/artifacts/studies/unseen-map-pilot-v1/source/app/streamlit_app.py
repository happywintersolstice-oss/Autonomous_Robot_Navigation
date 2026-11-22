"""Local navigation viewer with separate trained, learning and comparison tabs."""
import streamlit as st

from robot_navigation.benchmark import load_scenarios
from robot_navigation.core import ROOT, load_task
from robot_navigation.dashboard import PAGES
from robot_navigation.dashboard_ui import render_page, render_comparison
from robot_navigation.interface import apply_theme, preferred_runs, run_selector, workspace_header
from robot_navigation.simulator_ui import render_replay, pause_players

st.set_page_config(page_title='Navigation Control Room', page_icon=':material/explore:', layout='wide',
                   initial_sidebar_state='collapsed')
apply_theme()


def open_page(value):
    pause_players()
    st.session_state['dashboard_page'] = value


with st.sidebar:
    st.markdown('<div class="lab-brand"><span class="lab-icon">smart_toy</span>'
                '<strong>NAV<span>RL</span></strong></div>', unsafe_allow_html=True)
    st.markdown('<div class="lab-nav-label">WORKSPACE</div>', unsafe_allow_html=True)
    active = st.session_state.get('dashboard_page', 'Navigation simulator')
    for label, value, icon in (
        ('Live map', 'Navigation simulator', 'map'),
        ('Unseen maps', 'Unseen maps', 'travel_explore'),
        ('Overview', 'Overview', 'grid_view'),
        ('Learning analysis', 'Learning analysis', 'monitoring'),
        ('Compare results', 'Agent comparison', 'compare_arrows'),
        ('Neural tuning', 'Neural tuning', 'tune'),
        ('Methodology', 'Methodology', 'database'),
    ):
        st.button(label, icon=f':material/{icon}:', key=f'nav_{value}',
                  type='primary' if active == value else 'secondary', width='stretch',
                  on_click=open_page, args=(value,))
    with st.expander('Research library'):
        page = st.selectbox('Dashboard view', PAGES, key='dashboard_page', on_change=pause_players)
    st.markdown('<div class="lab-sidebar-status"><span class="lab-eyebrow">LOCAL WORKSPACE</span>'
                '<div class="lab-kv"><span>Data source</span><b>Saved runs</b></div>'
                '<div class="lab-kv"><span>Environment</span><b>Fixed grid</b></div>'
                '<small>Playback uses recorded actions.<br>Policies stay fixed during replay.</small></div>',
                unsafe_allow_html=True)


@st.cache_resource
def task_resource():
    return load_task()


try:
    task = task_resource()
    validation = load_scenarios(task, 'validation')
except (OSError, ValueError, KeyError) as exc:
    st.error(f'Benchmark unavailable: {exc}. Run python -m robot_navigation.cli prepare first.')
    st.stop()

workspace_header('Unseen-map pilot' if page=='Unseen maps' else task.grid.map_id, page)
if page != 'Navigation simulator':
    render_page(page, task, ROOT)
    st.stop()

runs = preferred_runs(p for p in (ROOT / 'artifacts/runs').glob('*') if (p / 'manifest.json').exists())
if not runs:
    st.info('No saved models are available yet.')
    st.stop()

labels = ['Trained replay', 'Learning replay', 'Compare agents', 'Try a route']
tabs = st.tabs(labels, key='replay_tab', on_change='rerun')
if st.session_state.get('active_replay_tab') != st.session_state['replay_tab']:
    pause_players()
    st.session_state['active_replay_tab'] = st.session_state['replay_tab']
for label, tab in zip(labels, tabs):
    if tab.open:
        with tab:
            if label == 'Compare agents':
                st.write('Watch different agents tackle the same start and goal. Their steps advance together.')
                count = st.radio('What would you like to compare?', ['Two agents', 'Four algorithms'],
                                 horizontal=True, key='comparison_size', on_change=pause_players)
                if count == 'Four algorithms':
                    render_comparison(task, None, validation, runs, ROOT, four=True)
                else:
                    with st.container(key='model_setup'):
                        directory = run_selector(runs, label='First trained model')
                    if directory is not None:
                        render_comparison(task, directory, validation, runs, ROOT)
            else:
                render_replay(task, validation, runs, ROOT,
                              mode={'Trained replay':'trained', 'Learning replay':'learning', 'Try a route':'custom'}[label])
