"""The everyday navigation app; research evidence remains available on demand."""
import streamlit as st

from robot_navigation.core import ROOT, load_task
from robot_navigation.dashboard import PAGES, NAVIGATION
from robot_navigation.dashboard_ui import render_page
from robot_navigation.interface import apply_theme
from robot_navigation.simulator_ui import pause_players

st.set_page_config(page_title='Robot Navigation', page_icon=':material/explore:', layout='wide',
                   initial_sidebar_state='collapsed')
apply_theme()


def open_page(value):
    pause_players()
    st.session_state['dashboard_page'] = value


def choose_result():
    open_page(st.session_state['results_view'])


def choose_benchmark():
    open_page('Unseen maps' if st.session_state['results_benchmark']=='Unseen layouts' else 'Overview')


page = st.session_state.get('dashboard_page', PAGES[0])
if page=='Map & reward lab':
    page = 'Navigation simulator'
    st.session_state['demo_map'] = 'structured-navigation-v2-dead-end-maze'
if page not in PAGES:
    page = PAGES[0]
st.session_state['dashboard_page'] = page
record = next(row for row in NAVIGATION if row[0]==page)
st.caption('ROBOT NAVIGATION | Learn by watching')
with st.container(key='app_navigation'):
    for column,route in zip(st.columns(3),dict.fromkeys(row[2] for row in NAVIGATION)):
        target = next(row[0] for row in NAVIGATION if row[2]==route)
        column.button(route,key=f'route_{route}',width='stretch',
                      type='primary' if record[2]==route else 'secondary',
                      on_click=open_page,args=(target,))


@st.cache_resource
def task_resource():
    return load_task()


if page=='Navigation simulator':
    from robot_navigation.navigation_demo import render_demo
    render_demo(ROOT)
    st.stop()

if page=='Methodology':
    from robot_navigation.user_guide_ui import render_guide
    render_guide()
    with st.expander('Research methods and technical background'):
        render_page(page,None,ROOT)
    with st.expander('Check this installation'):
        if st.button('Check saved demo assets'):
            from robot_navigation.demo_assets import verify
            try:
                result = verify(ROOT)
                st.success(f"Saved demo verified: {result['files']} files.")
            except (OSError,ValueError,KeyError) as exc:
                st.info(f'Saved demo unavailable: {exc}. Restore the named assets from the matching release.')
    st.stop()

from robot_navigation.user_guide_ui import render_results_intro
render_results_intro(ROOT)
with st.expander('Research results and comparisons',expanded=page!='Overview'):
    st.session_state['results_benchmark'] = record[3]
    st.radio('Benchmark',['Fixed layout','Unseen layouts'],key='results_benchmark',on_change=choose_benchmark,
             horizontal=True,format_func=lambda x:'Original map' if x=='Fixed layout' else 'Previously unseen maps')
    options = [row[0] for row in NAVIGATION if row[2]=='Results' and row[3]==record[3]]
    if len(options)>1:
        st.session_state['results_view'] = page
        st.selectbox('Results view',options,key='results_view',
                     format_func=lambda value:next(row[1] for row in NAVIGATION if row[0]==value),on_change=choose_result)
    if page=='Unseen maps':
        render_page(page,None,ROOT)
    else:
        try:
            render_page(page,task_resource(),ROOT)
        except (OSError, ValueError, KeyError) as exc:
            st.error(f'This research view is unavailable: {exc}. The saved-data guide is still available.')
