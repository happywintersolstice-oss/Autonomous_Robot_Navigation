"""Local navigation viewer with separate trained, learning and comparison tabs."""
import streamlit as st

from robot_navigation.benchmark import load_scenarios
from robot_navigation.core import ROOT, load_task
from robot_navigation.dashboard import PAGES
from robot_navigation.dashboard_ui import render_page, render_comparison
from robot_navigation.interface import apply_theme, preferred_runs, run_selector
from robot_navigation.simulator_ui import render_replay, pause_players

st.set_page_config(page_title='Navigation Control Room', page_icon=':material/explore:', layout='wide',
                   initial_sidebar_state='collapsed')
apply_theme()
with st.sidebar:
    st.markdown('<div class="lab-brand"><div class="lab-eyebrow">NAVIGATION LAB</div>'
                '<strong>Robot navigation</strong></div>', unsafe_allow_html=True)
    page = st.selectbox('Dashboard view', PAGES, key='dashboard_page', on_change=pause_players)
    st.caption('Use the replay tabs to watch the robot. Other pages contain saved research results.')


@st.cache_resource
def task_resource():
    return load_task()


try:
    task = task_resource()
    validation = load_scenarios(task, 'validation')
except (OSError, ValueError, KeyError) as exc:
    st.error(f'Benchmark unavailable: {exc}. Run python -m robot_navigation.cli prepare first.')
    st.stop()

if page != 'Navigation simulator':
    render_page(page, task, ROOT)
    st.stop()

st.markdown('<div class="lab-eyebrow">ROBOT NAVIGATION LAB</div>', unsafe_allow_html=True)
st.title('Watch the robot navigate')
st.caption('Choose what you want to watch. Play animates the robot; Next step shows one decision at a time.')
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
