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
    from .unseen_final import STUDY as FINAL, FinalBenchmark
    choices = [STUDY]
    main_summary = root/'artifacts/studies'/MAIN/'summary.json'
    if main_summary.exists() and read_json(main_summary).get('status')=='verified':
        choices.insert(0,MAIN)
    final_summary = root/'artifacts/studies'/FINAL/'summary.json'
    if final_summary.exists() and read_json(final_summary).get('status')=='verified':
        choices.insert(0,FINAL)
    labels = {FINAL:'Final evaluation · ten held-out layouts',MAIN:'Shared features · three seeds',
              STUDY:'Initial pilot · one seed'}
    selected = st.selectbox('Unseen-map study',choices,
        format_func=lambda value:labels[value],
        key='unseen_study')
    if st.session_state.get('unseen_active_study') != selected:
        player = st.session_state.get('unseen_playback')
        if player is not None:
            player.pause()
        st.session_state['unseen_active_study'] = selected
    directory = root/'artifacts/studies'/selected
    try:
        summary = read_json(directory/'summary.json')
        benchmark = FinalBenchmark(directory) if selected==FINAL else MapBenchmark(root)
        verification = read_json(directory/'verification.json')
        if (summary['status'] != 'verified' or verification['status'] != 'passed'
                or summary['signature'] != benchmark.signature
                or verification['signature'] != benchmark.signature):
            raise ValueError('Pilot verification or benchmark identity is incomplete')
        if selected==FINAL:
            from .release import sha
            if (verification['release_digest']!=benchmark.release_digest
                    or summary['release_digest']!=benchmark.release_digest
                    or verification['summary_sha256']!=sha(directory/'summary.json')):
                raise ValueError('Final release/result verification mismatch')
    except (OSError,ValueError,KeyError) as exc:
        st.info(f'Unseen-map pilot evidence unavailable: {exc}')
        st.json({'status':'Prepare, train and verify the separate unseen-map pilot first.'})
        return
    if selected==FINAL:
        st.caption('Independent final evaluation · ten held-out layouts · 480 routes per policy · '
                   'three trained seeds per algorithm. The frozen models use shared map-and-goal features.')
    elif selected==MAIN:
        st.caption('Shared-feature Q-learning, SARSA, DQN and PPO · three seeds · 200,000 training steps per run. '
                   'Development validation uses four layouts excluded from training. '
                   'The final evaluation has its own study entry when available.')
    else:
        st.caption('Map-aware Q-learning, SARSA, DQN and PPO · one pilot seed · 25,000 training steps each. '
                   'Development validation uses four layouts excluded from training. '
                   'This is the initial development pilot.')
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
    if summary.get('final_test_evaluated'):
        _final_results(summary)
        return
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
    st.caption('The M11 development study kept ten final layouts reserved. Final outcomes are presented separately. '
               'The earlier pilot also used a different budget, value function and seeds.')


def _final_results(summary):
    learned = [s for s in summary['systems'] if s['algorithm'] in ALGORITHM_LABELS]
    rows = []
    for system in learned:
        stats = system['statistics']
        interval = stats['success_rate']['layout_ci95']
        rows.append({'Algorithm':ALGORITHM_LABELS[system['algorithm']],
            'Final success (%)':100*stats['success_rate']['mean'],
            'Seed SD (points)':100*stats['success_rate']['sd'],
            'Layout 95% interval (%)':f'{100*interval[0]:.2f}–{100*interval[1]:.2f}',
            'Collision (%)':100*stats['collision_rate']['mean'],
            'Timeout (%)':100*stats['timeout_rate']['mean'],'SPL':stats['spl']['mean'],
            'Successes / routes':f"{system['counts']['success']} / {system['episodes']}"})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.caption('Three frozen training seeds per algorithm, 480 routes per policy. '
               'Seed SD measures training variation. The 95% interval resamples ten layouts, '
               'conditional on these policies. Routes within a layout/policy are dependent.')
    st.info('These results use the reserved final layouts after the complete model roster and analysis were sealed. '
            'The models made fixed decisions throughout evaluation. Future model changes require fresh evaluation evidence.')
    with st.expander('Every frozen policy and evaluation cost'):
        st.dataframe(pd.DataFrame([{'Algorithm':ALGORITHM_LABELS[r['algorithm']],'Training seed':r['training_seed'],
            'Development success (%)':100*r['record']['development']['success_rate'],
            'Final success (%)':100*r['final']['success_rate'],'SPL':r['final']['spl'],
            'Evaluation seconds':r['evaluation_seconds'],'Decision µs':r['decision_microseconds'],
            'Training seconds':r['record']['training_seconds'],'Checkpoint bytes':r['record']['checkpoint_bytes']}
            for r in summary['runs']]),hide_index=True,width='stretch')
    st.subheader('Results by final layout')
    st.dataframe(pd.DataFrame([{'Algorithm':ALGORITHM_LABELS[s['algorithm']],'Layout':name,
        'Success (%)':100*v['success_rate'],'Collision (%)':100*v['collision_rate'],
        'Timeout (%)':100*v['timeout_rate'],'SPL':v['spl'],'Routes':v['episodes']}
        for s in learned for name,v in s['by_map'].items()]),hide_index=True,width='stretch')
    st.subheader('Final reference methods')
    st.dataframe(pd.DataFrame([{'Method':s['algorithm'].upper() if s['algorithm']=='bfs' else 'Random',
        'Runs':s['seed_count'],'Success (%)':100*s['statistics']['success_rate']['mean'],
        'SPL':s['statistics']['spl']['mean'],'Successes':s['counts']['success'],'Routes':s['episodes'],
        'Evaluation seconds':s['evaluation_seconds']}
        for s in summary['systems'] if s['algorithm'] in ('bfs','random')]),hide_index=True,width='stretch')
    st.caption('BFS uses the known map. Random uses three fixed RNG seeds. '
               'Final outcomes cover 7,680 episodes across twelve learned policies and four reference runs.')


def _replay(benchmark, study, summary):
    systems = [s for s in summary.get('runs',summary['systems']) if s.get('phase','main') in ('main','final')]
    with st.container(key='unseen_setup'):
        family, recording_type, controls = st.columns([1.1,1.1,2.3], vertical_alignment='bottom')
    algorithm = family.selectbox('Map-aware algorithm', list(dict.fromkeys(s['algorithm'] for s in systems)),
                                format_func=lambda a:ALGORITHM_LABELS[a], key='unseen_algorithm')
    kinds = ['Final policy'] if summary.get('final_test_evaluated') else ['Trained policy','Recorded learning']
    kind = recording_type.selectbox('Recording type', kinds, key='unseen_kind')
    candidates = [s for s in systems if s['algorithm']==algorithm]
    if len(candidates)>1:
        seed = st.selectbox('Training seed',[s['training_seed'] for s in candidates],key='unseen_seed')
        system = next(s for s in candidates if s['training_seed']==seed)
    else:
        system = candidates[0]
    directory = study/'runs'/system['run_id']
    manifest = read_json(directory/'manifest.json')
    if summary.get('final_test_evaluated'):
        from .unseen_final import validate_result
        if system['record'] not in benchmark.release['records']:
            raise ValueError('Final replay policy is outside the sealed roster')
        validate_result(directory,system['record'],benchmark.release_digest)
    items = manifest['training']['training_traces'] if kind=='Recorded learning' else manifest['replays']

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
