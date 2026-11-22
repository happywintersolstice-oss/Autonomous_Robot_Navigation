"""Structured-map modes inside the shared navigation Demo."""
import gzip
import json
import time

import numpy as np
import pandas as pd
import streamlit as st

from .agents import BFSPolicy,QLearning,RandomPolicy
from .core import read_json,validate_scenario
from .dashboard import seek_from_widget
from .episodes import Playback,rollout
from .evidence import EXPECTED_ERRORS
from .interface import playback_toolbar,playback_status,show_map,telemetry
from .navigation_lab import BENCHMARK,STUDY,configuration,make_task,validate_record,RewardEnv
from .release import sha
from .reward_playback import RewardComparisonPlayback
from .simulator_ui import render_frame

LABELS = {'original':'Original sparse reward','time-cost':'Stronger time cost','potential':'Time cost + potential shaping'}
TAB_MODES = {'trained':'Watch the robot','learning':'Learning history','compare':'Compare robots','custom':'Choose a route'}


@st.cache_data(show_spinner=False)
def recorded_episodes(path,checksum):
    if sha(path)!=checksum:
        raise ValueError(f'Changed map recording: {path.name}')
    with gzip.open(path,'rt',encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def model_record(root,study,layout,condition,seed,config,hashes):
    directory = study/'runs'/f'{layout["family"]}-{condition}-seed{seed}'
    manifest_path = directory/'manifest.json'
    if sha(manifest_path)!=hashes[manifest_path.relative_to(study).as_posix()]:
        raise ValueError(f'Changed model manifest: {manifest_path}')
    manifest = read_json(manifest_path)
    task = make_task(layout,condition,config)
    task.check(manifest['signature'])
    path = directory/'episodes.jsonl.gz'
    records = recorded_episodes(path,hashes[path.relative_to(study).as_posix()])
    return task,manifest,records,directory


def map_details(root,layout,summary):
    with st.expander('Map details and reward results'):
        stats = layout['topology']
        st.dataframe(pd.DataFrame([{'Free cells':stats['free_cells'],'Dead ends':stats['dead_ends'],
            'Deepest dead end':stats['deepest_dead_end'],'Junctions':stats['junctions'],
            'Independent loops':stats['cycle_rank'],'Longest shortest route':stats['diameter']}]),hide_index=True,width='stretch')
        path = root/'data/navigation-lab'/BENCHMARK/'gallery.png'
        if path.is_file():
            st.image(str(path),width='stretch')
        else:
            st.info(f'Map gallery unavailable: {path}')
        rows = [{'Reward design':LABELS[c],
                 'Success (%)':100*s['metrics']['success_rate']['mean'],
                 'Collision (%)':100*s['metrics']['collision_rate']['mean'],
                 'Timeout (%)':100*s['metrics']['timeout_rate']['mean'],
                 'SPL':s['metrics']['spl']['mean'],'Looping timeouts':s['looping_timeouts']}
                for c,s in summary['systems'].items()]
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        st.caption('Development results over all four familiar maps: equal map means within each of three training seeds, then equal seed means. Each map/reward/seed trains for 100,000 actions. These are separate from historical final cohorts.')
        st.info('Stronger time cost gave a modest improvement. Potential shaping increased collisions and remains a failed comparison; loops are not fully solved.')


def select_route(layout,records):
    ids = {e['scenario']['id']:e for e in records}
    scenarios = sorted(layout['splits']['validation'],key=lambda s:(-s['shortest_length'],s['id']))
    key = f'lab_route:{layout["family"]}'
    if st.session_state.get(key) not in scenarios:
        st.session_state[key] = next((s for s in scenarios if ids[s['id']]['metrics']['outcome']=='success'),scenarios[0])
    def example(outcome):
        selected = next((s for s in scenarios if ids[s['id']]['metrics']['outcome']==outcome),None)
        if selected is not None:
            for name in ('lab_playback','lab_comparison_playback'):
                previous = st.session_state.get(name)
                if previous is not None:
                    previous.pause()
            st.session_state[key] = selected
    for column,outcome in zip(st.columns(3),('success','timeout','collision')):
        column.button({'success':'Route that succeeds','timeout':'Route that gets stuck','collision':'Route that hits a wall'}[outcome],key=f'lab_example_{outcome}',on_click=example,args=(outcome,),
                      disabled=not any(e['metrics']['outcome']==outcome for e in records),width='stretch')
    st.caption('The initial success is illustrative; all failures and validation routes remain accessible. Switching reward designs keeps the selected route.')
    scenario = st.selectbox('Navigation route',scenarios,key=key,
        format_func=lambda s:f"{s['start']} to {s['goal']} / shortest {s['shortest_length']} actions / {'detour' if s['requires_detour'] else 'direct'}")
    return scenario,ids[scenario['id']]


def install_player(key,identity_key,identity,episode):
    if st.session_state.get(identity_key)!=identity:
        previous = st.session_state.get(key)
        if previous is not None:
            previous.pause()
        st.session_state[key] = Playback(episode)
        st.session_state[identity_key] = identity
        st.session_state['lab_tick'] = time.monotonic()


def single_player(task,episode,identity,mode):
    install_player('lab_playback','lab_identity',identity,episode)
    def command(name):
        getattr(st.session_state['lab_playback'],name)()
        st.session_state['lab_tick'] = time.monotonic()
    speed = playback_toolbar(command,prefix='lab_',speed_key='lab_speed')
    def seek(key):
        seek_from_widget(st.session_state,key,player_key='lab_playback',position_key='lab_seek',active_key='lab_timeline')
    player = st.session_state['lab_playback']
    @st.fragment(run_every=.2 if player.playing else None)
    def viewer():
        if (st.session_state.get('dashboard_page')!='Navigation simulator'
                or st.session_state.get('demo_map')!=task.grid.map_id
                or st.session_state.get('replay_tab')!=TAB_MODES[mode]
                or st.session_state.get('lab_identity')!=identity):
            return
        player = st.session_state['lab_playback']
        now = time.monotonic()
        if player.playing and now-st.session_state['lab_tick']>=1/speed:
            player.tick()
            st.session_state['lab_tick'] = now
            if not player.playing:
                st.rerun()
        playback_status(player.index,player.playing)
        render_frame(task,episode,player.index,player.playing,{'trail':True,'visits':False,'bfs':False})
        if player.index:
            values = episode['transitions'][player.index-1]['info']['reward_components']
            with st.expander('Last move scoring'):
                st.caption(f"Last reward: base {values['base']:+.3f}, potential {values['potential']:+.3f}; total {sum(values.values()):+.3f}.")
        key = f'lab-timeline:{identity}:{player.index}'
        st.session_state['lab_timeline'] = key
        st.slider('Replay position',0,len(episode['transitions']),value=player.index,key=key,on_change=seek,args=(key,))
    viewer()


def compare_players(tasks,episodes,identity,labels):
    if st.session_state.get('lab_comparison_identity')!=identity:
        for task,episode in zip(tasks,episodes,strict=True):
            validate_record(task,episode)
        previous = st.session_state.get('lab_comparison_playback')
        if previous is not None:
            previous.pause()
        st.session_state['lab_comparison_playback'] = RewardComparisonPlayback(episodes,tasks)
        st.session_state['lab_comparison_identity'] = identity
        st.session_state['lab_comparison_tick'] = time.monotonic()
    def command(name):
        getattr(st.session_state['lab_comparison_playback'],name)()
        st.session_state['lab_comparison_tick'] = time.monotonic()
    speed = playback_toolbar(command,prefix='lab_comparison_',speed_key='lab_comparison_speed')
    def seek(key):
        seek_from_widget(st.session_state,key,player_key='lab_comparison_playback',
                         position_key='lab_comparison_seek',active_key='lab_comparison_timeline')
    player = st.session_state['lab_comparison_playback']
    @st.fragment(run_every=.2 if player.playing else None)
    def panels():
        if (st.session_state.get('dashboard_page')!='Navigation simulator'
                or st.session_state.get('demo_map')!=tasks[0].grid.map_id
                or st.session_state.get('replay_tab')!='Compare robots'
                or st.session_state.get('lab_comparison_identity')!=identity):
            return
        player = st.session_state['lab_comparison_playback']
        now = time.monotonic()
        if player.playing and now-st.session_state['lab_comparison_tick']>=1/speed:
            player.tick()
            st.session_state['lab_comparison_tick'] = now
            if not player.playing:
                st.rerun()
        playback_status(player.index,player.playing,shared=True)
        for column,task,episode,label in zip(st.columns(2),tasks,player.episodes,labels):
            with column:
                st.write(label)
                show_map(task.grid,episode,player.index)
                telemetry(episode,player.index,task.config['max_steps'],player.playing,shared_index=player.index)
        key = f'lab-comparison-timeline:{identity}:{player.index}'
        st.session_state['lab_comparison_timeline'] = key
        st.slider('Replay position',0,player.length,value=player.index,key=key,on_change=seek,args=(key,))
    panels()


def custom_route(task,directory,selection):
    default = selection[-1]
    keys = ('lab_custom_sr','lab_custom_sc','lab_custom_gr','lab_custom_gc')
    remembered = st.session_state.get('lab_custom_form_values',{})
    if st.session_state.get('lab_custom_form_map')!=task.grid.map_id or any(key not in st.session_state for key in keys):
        for key,value in zip(keys,remembered.get(task.grid.map_id,[*default['start'],*default['goal']])):
            st.session_state[key] = value
        st.session_state['lab_custom_form_map'] = task.grid.map_id
    from .route_picker import pick_cells,coordinate_inputs
    pick_cells(task,keys,'structured',[*default['start'],*default['goal']])
    with st.form('structured_custom'):
        sr,sc,gr,gc = coordinate_inputs(keys,*task.grid.shape)
        model = st.selectbox('Frozen policy',['Q-learning','BFS','Random (untrained)'],key='lab_custom_policy',
            format_func=lambda value:'Shortest-path planner (BFS)' if value=='BFS' else 'Trained robot (Q-learning)' if value=='Q-learning' else value)
        with st.expander('Random movement settings'):
            seed = st.number_input('Demonstration seed',0,2147483647,7,key='lab_custom_rng')
        generate = st.form_submit_button('Generate demonstration',type='primary')
    st.caption('Generate demonstration creates the route attempt. Then press Play. Changing the squares does not change an existing replay until you generate again.')
    st.session_state['lab_custom_form_values'] = {**remembered,task.grid.map_id:[sr,sc,gr,gc]}
    context = tuple(selection[:-1])
    if generate:
        try:
            scenario = {'id':f'custom-{sr}-{sc}-{gr}-{gc}','start':[sr,sc],'goal':[gr,gc]}
            validate_scenario(task.grid,scenario)
            scenario['shortest_length'] = len(task.grid.path(scenario['start'],scenario['goal']))-1
            policy = BFSPolicy(task.grid) if model=='BFS' else RandomPolicy(seed) if model=='Random (untrained)' else QLearning.load(directory/'selected.npz',task)
            before = policy.q.copy() if isinstance(policy,QLearning) else None
            episode = rollout(RewardEnv(task,[scenario]),policy,scenario,seed,'custom demonstration')
            validate_record(task,episode)
            if before is not None and not np.array_equal(before,policy.q):
                raise ValueError('Custom demonstration changed model weights')
            st.session_state['lab_custom_episode'] = episode
            st.session_state['lab_custom_context'] = context
            st.session_state['lab_custom_id'] = ('custom',*context,scenario['id'],model,seed)
        except (*EXPECTED_ERRORS,RuntimeError) as exc:
            st.error(f'Invalid custom scenario: {exc}')
    st.caption('Generated with the selected map and reward version. No training or benchmark writes.')
    if st.session_state.get('lab_custom_context')!=context:
        st.info('Choose your start and goal, then select Generate demonstration.')
        return
    single_player(task,st.session_state['lab_custom_episode'],st.session_state['lab_custom_id'],'custom')


def render_structured(layout,root,mode='trained'):
    st.caption('This map uses a Q-learning robot trained specifically for its passages and walls.')
    study = root/'artifacts/studies'/STUDY
    try:
        config = configuration(root)
        if read_json(study/'status.json')['status']!='verified' or read_json(study/'verification.json')['status']!='passed':
            raise ValueError('Study has not passed verification')
        hashes = read_json(study/'outputs.json')
        if sha(study/'summary.json')!=hashes['summary.json']:
            raise ValueError('Reward-study summary checksum mismatch')
        summary = read_json(study/'summary.json')
        with st.expander('Robot settings and reward design'):
            model,seed_column = st.columns(2)
            condition = model.selectbox('Reward design',list(LABELS),index=1,format_func=lambda c:LABELS[c],key='lab_reward')
            seed = seed_column.selectbox('Model training seed',config['training_seeds'],key='lab_seed',
                format_func=lambda value:f"Robot version {config['training_seeds'].index(value)+1}")
            st.caption('Each version was trained independently. Reward design controls how training scores moves. Stronger time cost is the default; potential shaping increased collisions in the saved study.')
        task,manifest,records,directory = model_record(root,study,layout,condition,seed,config,hashes)
        with st.expander('Model identification'):
            st.caption(f"{layout['family']} / training seed {seed} / checkpoint {manifest['selected_interactions']:,} actions; no learning during playback.")
        if mode=='learning':
            st.info('Chronological training actions were not recorded for these map runs. The saved validation-monitoring curves below are learning analysis, not a reconstructed training replay.')
            path = directory/'monitoring.json'
            if sha(path)!=hashes[path.relative_to(study).as_posix()]:
                raise ValueError('Changed monitoring evidence')
            st.line_chart(pd.DataFrame(read_json(path)).set_index('step')[['success_rate','spl']])
        elif mode=='custom':
            custom_route(task,directory,(layout['family'],condition,seed,layout['splits']['validation'][0]))
        else:
            scenario,episode = select_route(layout,records)
            if mode=='compare':
                second,other_seed = st.columns(2)
                second_condition = second.selectbox('Second reward design',list(LABELS),
                    index=0 if condition!='original' else 1,format_func=lambda c:LABELS[c],key='lab_second_reward')
                second_seed = other_seed.selectbox('Second model training seed',config['training_seeds'],key='lab_second_seed')
                second_task,_,other_records,_ = model_record(root,study,layout,second_condition,second_seed,config,hashes)
                other = next(e for e in other_records if e['scenario']['id']==scenario['id'])
                st.caption('Same map, start, goal, action cap and shared action clock. Rewards can differ; compare success/SPL and route behavior rather than raw reward totals.')
                compare_players([task,second_task],[episode,other],
                    (layout['family'],condition,seed,second_condition,second_seed,scenario['id']),
                    [f'{LABELS[condition]} / seed {seed}',f'{LABELS[second_condition]} / seed {second_seed}'])
            else:
                validate_record(task,episode)
                single_player(task,episode,('trained',layout['family'],condition,seed,scenario['id']),'trained')
        map_details(root,layout,summary)
    except EXPECTED_ERRORS as exc:
        st.info(f'{STUDY}: unavailable or unverified ({exc}). Restore the matching map/model/recording files.')
