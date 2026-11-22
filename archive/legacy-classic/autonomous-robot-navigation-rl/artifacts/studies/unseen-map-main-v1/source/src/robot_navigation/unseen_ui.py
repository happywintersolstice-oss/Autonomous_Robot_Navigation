"""Read-only pilot results and the same centered recorded-frame presentation."""
import time

import pandas as pd
import streamlit as st

from .core import read_json
from .dashboard import seek_from_widget
from .episodes import Playback
from .interface import ALGORITHM_LABELS, playback_status, playback_toolbar
from .simulator_ui import render_frame
from .unseen_audit import load_recording
from .unseen_benchmark import MapBenchmark
from .unseen_study import STUDY


def render_unseen(root):
    from .unseen_main import STUDY as MAIN
    choices = [STUDY]
    main_summary = root/'artifacts/studies'/MAIN/'summary.json'
    if main_summary.exists() and read_json(main_summary).get('status')=='verified':
        choices.insert(0,MAIN)
    selected = st.selectbox('Unseen-map study',choices,
        format_func=lambda value:'Shared features · three seeds' if value==MAIN else 'Initial pilot · one seed',
        key='unseen_study')
    if st.session_state.get('unseen_active_study') != selected:
        player = st.session_state.get('unseen_playback')
        if player is not None:
            player.pause()
        st.session_state['unseen_active_study'] = selected
    directory = root/'artifacts/studies'/selected
    try:
        summary = read_json(directory/'summary.json')
        benchmark = MapBenchmark(root)
        verification = read_json(directory/'verification.json')
        if (summary['status'] != 'verified' or verification['status'] != 'passed'
                or summary['signature'] != benchmark.signature
                or verification['signature'] != benchmark.signature):
            raise ValueError('Pilot verification or benchmark identity is incomplete')
    except (OSError,ValueError,KeyError) as exc:
        st.info(f'Unseen-map pilot evidence unavailable: {exc}')
        st.json({'status':'Prepare, train and verify the separate unseen-map pilot first.'})
        return
    if selected==MAIN:
        st.caption('Shared-feature Q-learning, SARSA, DQN and PPO · three seeds · 200,000 training steps per run. '
                   'Development validation uses four layouts excluded from training. Final layouts remain reserved.')
    else:
        st.caption('Map-aware Q-learning, SARSA, DQN and PPO · one pilot seed · 25,000 training steps each. '
                   'Development validation uses four layouts excluded from training. Final layouts remain reserved.')
    tabs = st.tabs(['Results','Replay'], key='unseen_view', on_change='rerun')
    if st.session_state.get('unseen_active_view') != st.session_state['unseen_view']:
        player = st.session_state.get('unseen_playback')
        if player is not None:
            player.pause()
        st.session_state['unseen_active_view'] = st.session_state['unseen_view']
    if tabs[0].open:
        with tabs[0]:
            _results(summary)
    if tabs[1].open:
        with tabs[1]:
            _replay(benchmark, directory, summary)


def _results(summary):
    if 'runs' in summary:
        _main_results(summary)
        return
    rows = []
    for system in summary['systems']:
        validation = system['validation']
        rows.append({'Algorithm':ALGORITHM_LABELS[system['algorithm']],
                     'Familiar success (%)':100*system['familiar']['success_rate'],
                     'New-layout success (%)':100*validation['success_rate'],
                     'Collision (%)':100*validation['collision_rate'],
                     'Timeout (%)':100*validation['timeout_rate'], 'SPL':validation['spl'],
                     'Training steps':system['training_interactions'],
                     'Selected step':system['selected_interactions'],
                     'Training seconds':system['training_seconds']})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption('New-layout validation has 48 routes per layout, 192 routes per policy. '
               'The familiar-layout probe has 72 training routes. This pilot has one trained seed per algorithm.')
    st.info('Every algorithm receives the full obstacle map and robot/goal coordinates. '
            'Tabular policies use separate exact-state tables for each training layout. '
            'A new layout starts with zero Q values, so Up wins evaluation ties. '
            'DQN and PPO share their learned functions across layouts.')
    st.subheader('Results by validation layout')
    per_map = [{'Algorithm':ALGORITHM_LABELS[s['algorithm']], 'Layout':name,
                'Routes':values['episodes'], 'Success (%)':100*values['success_rate'],
                'SPL':values['spl'], 'Successes':values['counts']['success'],
                'Successful route length':values['mean_successful_route_length'],
                'Mean reward':values['mean_return']}
               for s in summary['systems'] for name, values in s['validation']['by_map'].items()]
    st.dataframe(pd.DataFrame(per_map), hide_index=True, width='stretch')
    st.subheader('Reference methods')
    st.dataframe(pd.DataFrame([{'Method':name.upper() if name=='bfs' else 'Random',
                               'Success (%)':100*record['success_rate'], 'SPL':record['spl'],
                               'Routes':record['episodes']}
                              for name,record in summary['references'].items()]),hide_index=True,width='stretch')
    st.caption('BFS plans with the obstacle map. Random is untrained. '
               'The main multi-seed study and independent final evaluation follow the pilot.')


def _main_results(summary):
    rows = []
    for system in summary['systems']:
        stats = system['validation']['statistics']
        rows.append({'Algorithm':ALGORITHM_LABELS[system['algorithm']],
                     'Seeds':system['seed_count'],
                     'Familiar success (%)':100*system['familiar']['statistics']['success_rate']['mean'],
                     'New-layout success (%)':100*stats['success_rate']['mean'],
                     'Success SD (points)':100*stats['success_rate']['sd'],
                     'Collision (%)':100*stats['collision_rate']['mean'],
                     'Timeout (%)':100*stats['timeout_rate']['mean'],
                     'SPL':stats['spl']['mean'],'SPL SD':stats['spl']['sd'],
                     'Successes / routes':f"{system['validation']['counts']['success']} / {system['validation']['episodes']}",
                     'Training steps':system['training_interactions']})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.caption('Means and sample SD across three training seeds. Each policy uses the same 192 routes '
               'on four validation layouts. The familiar probe uses 72 training routes. '
               'Routes from the same layout and policy are dependent observations.')
    st.info('All four algorithms use the same 1,234 map-and-goal features. Q-learning and SARSA '
            'use shared linear value functions. DQN and PPO use neural networks. '
            'These development results include checkpoint selection on validation layouts.')
    st.subheader('Each training seed')
    st.dataframe(pd.DataFrame([{'Algorithm':ALGORITHM_LABELS[r['algorithm']],
        'Seed':r['training_seed'],'Success (%)':100*r['validation']['success_rate'],
        'SPL':r['validation']['spl'],'Collision (%)':100*r['validation']['collision_rate'],
        'Selected step':r['selected_interactions'],'Parameters':r['optimized_parameter_count'],
        'Training seconds':r['training_seconds'],'Decision µs':r['mean_decision_microseconds']}
        for r in summary['runs'] if r['phase']=='main']),hide_index=True,width='stretch')
    st.subheader('Results by validation layout')
    st.dataframe(pd.DataFrame([{'Algorithm':ALGORITHM_LABELS[s['algorithm']],'Layout':name,
        'Mean success (%)':100*values['success_rate'],'Mean SPL':values['spl']}
        for s in summary['systems'] for name,values in s['validation']['by_map'].items()]),
        hide_index=True,width='stretch')
    with st.expander('Shared-feature engineering pilots'):
        st.dataframe(pd.DataFrame([{'Algorithm':ALGORITHM_LABELS[r['algorithm']],
            'Seed':r['training_seed'],'Training steps':r['training_interactions'],
            'Success (%)':100*r['validation']['success_rate'],'SPL':r['validation']['spl']}
            for r in summary['runs'] if r['phase']=='pilot']),hide_index=True,width='stretch')
    st.caption('The ten final layouts remain reserved for the next independent evaluation. '
               'The earlier pilot also used a different budget, value function and seeds.')


def _replay(benchmark, study, summary):
    systems = [s for s in summary.get('runs',summary['systems']) if s.get('phase','main')=='main']
    with st.container(key='unseen_setup'):
        family, recording_type, controls = st.columns([1.1,1.1,2.3], vertical_alignment='bottom')
    algorithm = family.selectbox('Map-aware algorithm', list(dict.fromkeys(s['algorithm'] for s in systems)),
                                format_func=lambda a:ALGORITHM_LABELS[a], key='unseen_algorithm')
    kind = recording_type.selectbox('Recording type', ['Trained policy','Recorded learning'], key='unseen_kind')
    candidates = [s for s in systems if s['algorithm']==algorithm]
    if len(candidates)>1:
        seed = st.selectbox('Training seed',[s['training_seed'] for s in candidates],key='unseen_seed')
        system = next(s for s in candidates if s['training_seed']==seed)
    else:
        system = candidates[0]
    directory = study/'runs'/system['run_id']
    manifest = read_json(directory/'manifest.json')
    items = manifest['replays'] if kind=='Trained policy' else manifest['training']['training_traces']

    def label(item):
        if 'window_start' in item:
            return f"Training window {item['window_start']:,} · {item['layout_id']}"
        return f"{item['layout_id']} · {item['outcome']}"

    item = st.selectbox('Map recording', items, format_func=label, key=f'unseen_recording:{system["run_id"]}:{kind}')
    try:
        task, episode = load_recording(benchmark, directory, item)
    except (OSError,ValueError,KeyError) as exc:
        st.error(f'Cannot open map recording: {exc}')
        return
    identity = f'{directory.name}/{item["path"]}/{item["sha256"]}'
    if st.session_state.get('unseen_identity') != identity:
        st.session_state['unseen_playback'] = Playback(episode)
        st.session_state['unseen_identity'] = identity
        st.session_state['unseen_tick'] = time.monotonic()
    player = st.session_state['unseen_playback']

    def command(name):
        current = st.session_state['unseen_playback']
        getattr(current,name)()
        st.session_state['unseen_seek'] = current.index
        st.session_state['unseen_tick'] = time.monotonic()

    def seek(key):
        seek_from_widget(st.session_state,key,player_key='unseen_playback',position_key='unseen_seek',
                         active_key='unseen_active_timeline')

    with controls:
        with st.container(key='single_playback_controls'):
            speed = playback_toolbar(command,prefix='unseen_',speed_key='unseen_speed',compact=True)
    options = {'trail':st.session_state.get('unseen_trail',True),
               'visits':st.session_state.get('unseen_visits',False),
               'bfs':st.session_state.get('unseen_bfs',False)}

    @st.fragment(run_every=.2 if player.playing else None)
    def viewer():
        if (st.session_state.get('unseen_identity') != identity
                or st.session_state.get('dashboard_page') != 'Unseen maps'
                or st.session_state.get('unseen_view') != 'Replay'):
            return
        current = st.session_state['unseen_playback']
        now = time.monotonic()
        if current.playing and now-st.session_state['unseen_tick'] >= 1/speed:
            current.tick()
            st.session_state['unseen_tick'] = now
            if not current.playing:
                st.rerun()
        playback_status(current.index,current.playing,training=episode['mode']=='training')
        render_frame(task,episode,current.index,current.playing,options)
        key = f'unseen-timeline:{identity}:{current.index}'
        st.session_state['unseen_active_timeline'] = key
        st.slider('Replay position',0,len(episode['transitions']),value=current.index,key=key,
                  on_change=seek,args=(key,))

    viewer()
    with st.expander('Run and policy information'):
        st.caption(f"{manifest['run_id']} · {episode['layout_split']} layout · {episode['scenario']['id']}")
        st.caption('Policy input includes the full obstacle map. The recording stores actual coordinate transitions.')
    with st.expander('Map display options'):
        st.checkbox('Traveled trail',value=True,key='unseen_trail')
        st.checkbox('Visit counts',key='unseen_visits')
        st.checkbox('BFS planner reference',key='unseen_bfs')
