"""Warehouse navigation, learning recordings and matched four-algorithm playback."""
import time

import numpy as np
import pandas as pd
import streamlit as st

from .agents import BFSPolicy,QLearning,load_policy
from .core import NavigationEnv,read_json
from .dashboard import seek_from_widget
from .episodes import Playback,ComparisonPlayback,rollout,validate_replay
from .evidence import EXPECTED_ERRORS
from .interface import playback_toolbar,playback_status,show_map,telemetry
from .simulator_ui import render_frame,pause_players
from .warehouse import MAP_ID,load as load_floor,scenario
from .warehouse_study import FAMILIES,saved,checked_run,episodes as assessment_episodes


def reset_player():
    pause_players()
    player = st.session_state.get('warehouse_playback')
    if player is not None:
        player.reset()


def player_view(task,episodes,identity,mode):
    if st.session_state.get('warehouse_identity')!=identity:
        pause_players()
        st.session_state['warehouse_playback'] = Playback(episodes[0]) if len(episodes)==1 else ComparisonPlayback(episodes)
        st.session_state['warehouse_identity'] = identity
        st.session_state['warehouse_tick'] = time.monotonic()
    player = st.session_state['warehouse_playback']
    def command(name):
        getattr(st.session_state['warehouse_playback'],name)()
        st.session_state['warehouse_tick'] = time.monotonic()
        st.session_state['warehouse_start_pending'] = name=='play'
    speed = playback_toolbar(command,prefix='warehouse_',speed_key='warehouse_speed')
    def seek(key):
        seek_from_widget(st.session_state,key,player_key='warehouse_playback',
                         position_key='warehouse_seek',active_key='warehouse_timeline')
    @st.fragment(run_every=.2 if player.playing else None)
    def viewer():
        labels = {'trained':'Watch the robot','compare':'Compare robots','custom':'Choose a route','learning':'Learning history'}
        if (st.session_state.get('dashboard_page')!='Navigation simulator'
                or st.session_state.get('demo_map')!=MAP_ID
                or st.session_state.get('replay_tab')!=labels[mode]
                or st.session_state.get('warehouse_identity')!=identity):
            return
        player = st.session_state['warehouse_playback']
        now = time.monotonic()
        starting = st.session_state.pop('warehouse_start_pending',False)
        if starting:
            st.session_state['warehouse_tick'] = now
        if player.playing and not starting and now-st.session_state['warehouse_tick']>=1/speed:
            player.tick()
            st.session_state['warehouse_tick'] = now
            if not player.playing:
                st.rerun()
        playback_status(player.index,player.playing,shared=len(episodes)>1)
        if len(episodes)==1:
            render_frame(task,episodes[0],player.index,player.playing,{'trail':True,'visits':False,'bfs':False})
            length = len(episodes[0]['transitions'])
        else:
            for offset in range(0,len(player.episodes),2):
                for i,column in zip(range(offset,min(offset+2,len(player.episodes))),st.columns(2)):
                    episode = player.episodes[i]
                    family = episode['policy_id'].split('/')[0]
                    label = FAMILIES.get(family,'BFS shortest-route reference')
                    with column:
                        st.markdown(f'**{label}**')
                        show_map(task.grid,episode,player.index,view_key=f'warehouse_comparison_{i}')
                        telemetry(episode,player.index,task.config['max_steps'],player.playing,
                                  shared_index=player.index,compact=len(episodes)==4)
            length = player.length
        key = f'warehouse-timeline:{identity}:{player.index}'
        st.session_state['warehouse_timeline'] = key
        st.slider('Replay position',0,length,value=player.index,key=key,on_change=seek,args=(key,))
    viewer()


def render_warehouse(root,mode):
    try:
        result = saved(root)
        task,layout,routes = load_floor(root)
    except EXPECTED_ERRORS as exc:
        if (root/'artifacts/warehouse/warehouse-four-algorithm-v1/published.json').exists():
            st.error(f'Warehouse comparison could not be verified: {exc}')
            return
        from .warehouse_results_ui import study_progress
        study_progress(root)
        return
    seed_column,algorithm_column = st.columns(2)
    seed = seed_column.selectbox('Training seed',result['systems']['q-learning']['seeds'],
                                key='warehouse_seed',on_change=reset_player,
                                help='Each seed identifies an independent training run. Compare the same seed across algorithms; Results averages all three.')
    algorithm = algorithm_column.selectbox('Algorithm',list(FAMILIES),format_func=lambda v:FAMILIES[v],
        key='warehouse_algorithm',on_change=reset_player,disabled=mode=='compare',
        help='The comparison shows all four algorithms. This choice controls individual playback, training recordings and custom routes.')
    stations = {item['id']:item for item in layout['stations']}
    jobs = {item['id']:item for item in routes['jobs']}
    if mode in ('trained','compare'):
        job_id = st.selectbox('Delivery job',list(jobs),key='warehouse_job',
            format_func=lambda value:jobs[value]['name'],on_change=reset_player)
        job = jobs[job_id]
        item = job['scenario']
        with st.expander('Browse every assessed route'):
            all_routes = st.checkbox('Choose from all 188 assessment routes',key='warehouse_all_routes',on_change=reset_player)
            if all_routes:
                item = st.selectbox('Assessment route',routes['validation'],key='warehouse_assessment_route',
                    format_func=lambda s:f"{s['start']} to {s['goal']} | shortest {s['shortest_length']} moves",on_change=reset_player)
        st.write('Recorded development route on the same warehouse floor.' if all_routes else job['description'])
        from .warehouse_scene import station_name
        a,b = st.columns(2)
        a.markdown(f"**Pickup:** {station_name(task.grid,item['start'])}")
        b.markdown(f"**Destination:** {station_name(task.grid,item['goal'])}")
    st.caption('Storage racks, the packing bench, staged pallets and outer walls block movement. Parcel handling is the task context; physical loading is not simulated.')
    run,folder = checked_run(algorithm,seed,root)
    if mode=='learning':
        st.write(f"Actual training actions recorded for {FAMILIES[algorithm]}, seed {seed}.")
        traces = run['training_traces']['records']
        entry = st.selectbox('Training recording',traces,key=f'warehouse_trace_{algorithm}_{seed}',
            format_func=lambda r:f"From training move {r['training_start_interaction']:,} | {r['actions']} recorded actions")
        episode = read_json(folder/entry['path'])
        st.caption('These are chronological exploratory actions, not decisions reconstructed from the final model. The recording samples two episodes at each registered training stage.')
        player_view(task,[episode],('learning',algorithm,seed,entry['path']),'learning')
        with st.expander('Learning curve for this run'):
            logs = read_json(folder/'monitoring.json')
            st.line_chart(pd.DataFrame([{'Training moves':r['training_interactions'],
                                        'Monitored completion rate':r['success_rate']} for r in logs]).set_index('Training moves'))
    elif mode=='custom':
        with st.form('warehouse_delivery'):
            pickup,target = st.columns(2)
            source = pickup.selectbox('Pickup point',list(stations),key='warehouse_pickup',format_func=lambda value:stations[value]['name'])
            destination = target.selectbox('Delivery point',list(stations),index=4,key='warehouse_destination',format_func=lambda value:stations[value]['name'])
            method = st.selectbox('Route method',['Selected trained algorithm','Shortest-route planner (BFS)'],key='warehouse_method')
            generate = st.form_submit_button('Create delivery route',type='primary')
        if generate:
            try:
                item = scenario(task.grid,stations[source]['cell'],stations[destination]['cell'])
                policy = load_policy(folder/run['checkpoint'],task) if method=='Selected trained algorithm' else BFSPolicy(task.grid)
                fingerprint = getattr(policy,'state_fingerprint',None)
                before = fingerprint() if fingerprint else policy.q.copy() if isinstance(policy,QLearning) else None
                episode = rollout(NavigationEnv(task,[item]),policy,item,mode='custom demonstration')
                validate_replay(task,episode)
                if before is not None and ((fingerprint and before!=fingerprint()) or
                                           (not fingerprint and not np.array_equal(before,policy.q))):
                    raise ValueError('Delivery changed the frozen model')
                st.session_state['warehouse_custom'] = episode
                st.session_state['warehouse_custom_context'] = (algorithm,seed)
                st.session_state['warehouse_custom_identity'] = (source,destination,method,algorithm,seed)
                reset_player()
                st.session_state.pop('warehouse_identity',None)
            except (ValueError,OSError,KeyError) as exc:
                st.error(f'Cannot create this delivery: {exc}')
        if st.session_state.get('warehouse_custom_context')==(algorithm,seed):
            st.caption('Showing the last created route. New station choices take effect after Create delivery route.')
            player_view(task,[st.session_state['warehouse_custom']],
                ('custom',*st.session_state['warehouse_custom_identity']),'custom')
        else:
            st.info('Assign a pickup and destination, then create a route using the selected algorithm and seed.')
    elif mode=='compare':
        records = [next(e for e in assessment_episodes(family,seed,root) if e['scenario']['id']==item['id']) for family in FAMILIES]
        st.caption('Four independently trained algorithms, one warehouse route and one shared move clock. Finished robots hold their final position. These are examples; Results compares all 188 routes across all three seeds.')
        player_view(task,records,('compare',seed,item['id']),mode)
    else:
        episode = next(e for e in assessment_episodes(algorithm,seed,root) if e['scenario']['id']==item['id'])
        player_view(task,[episode],('trained',algorithm,seed,item['id']),mode)
    with st.expander('Algorithm and experiment details'):
        st.write(f"{FAMILIES[algorithm]} | seed {seed} | {run['training']['training_interactions']:,} training actions")
        st.caption('Four algorithms, three independent seeds per algorithm, equal training-action budgets and the same assessment cases. Fixed final checkpoints; no selection based on monitoring.')
        st.caption(result['scope'])
        st.caption(layout['simulation_limits'])
