"""Streamlit views; all quantitative displays read saved experiment evidence."""
import time

import pandas as pd
import streamlit as st

from .core import ACTION_NAMES, read_json
from .dashboard import evidence, early_checkpoint, frozen_episode, seek_from_widget
from .episodes import ComparisonPlayback
from .interface import context_strip, panel_heading, show_map, telemetry, run_label, SELECTED_RUN
from .release import final_summary
from .four_way import saved_summary as four_way_summary,STUDY as FOUR_WAY_STUDY
from .four_way_analysis import saved_analysis, ANALYSIS_STUDY


def saved_json(path):
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        st.info(f'Saved evidence unavailable ({path.name}): {exc}')
        return None


def saved_image(path):
    if path.is_file():
        st.image(str(path), width='stretch')
    else:
        st.info(f'Saved figure unavailable: {path.name}')


def metric_rows(summary):
    rows=[]
    for name, system in summary['systems'].items():
        row={'System':name,'Seeds':system['seed_count']}
        for metric in ('success_rate','collision_rate','timeout_rate','spl','mean_successful_route_length','mean_return'):
            stats=system['aggregate'][metric]
            row[metric]=stats['mean']
            row[f'{metric} seed SD']=stats['sd']
        rows.append(row)
    return rows


def render_page(page, task, root):
    st.title(page)
    try:
        details = saved_analysis(root)
        if details:
            task.check(details['signature'])
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f'Extended comparison unavailable: {exc}')
        details = None
    fourth=four_way_summary(root)
    if fourth:
        task.check(fourth['signature'])
    if page=='Agent comparison':
        st.subheader('Four RL algorithms: development validation')
        if fourth:
            st.caption('Historical Part A Q-learning, SARSA and DQN plus the additional PPO baseline. Five seeds and 200,000 training interactions per seed. Post-release exploratory comparison; PPO is not part of the frozen final evaluation below.')
            rows=[{'Algorithm':system['label'],'Seeds':system['seed_count'],
                   **{metric:system['metrics'][metric]['mean'] for metric in ('success_rate','collision_rate','timeout_rate','spl')},
                   'Success seed SD':system['metrics']['success_rate']['sample_sd']}
                  for system in fourth['systems'].values()]
            if details:
                for row, system in zip(rows, details['systems'].values()):
                    route = system['metrics']['mean_successful_route_length']
                    row.update({'Successful route actions': route['mean'],
                                'Route seed SD': route['sample_sd'],
                                'Route contributing seeds': route['contributing_seeds'],
                                'Successes across five seeds': system['counts']['success'],
                                'Mean return': system['metrics']['mean_return']['mean']})
            st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
            if details:
                st.caption('Route length is conditional on success: equal-weight seed means, excluding undefined zero-success seed means. Success counts span 928 scenarios × five seeds; they are not independent training replicates. Rates are fractions (1 = 100%).')
                st.subheader('Recorded costs across five training seeds')
                cost_labels = {'training_seconds':'Training seconds', 'evaluation_seconds':'Validation seconds',
                               'mean_decision_microseconds':'Decision microseconds', 'checkpoint_bytes':'Checkpoint bytes'}
                costs = [{'Algorithm': system['label'],
                          **{label: system['costs'][key]['mean'] for key, label in cost_labels.items()},
                          **{label+' seed SD': system['costs'][key]['sample_sd'] for key, label in cost_labels.items()}}
                         for system in details['systems'].values()]
                st.dataframe(pd.DataFrame(costs),hide_index=True,width='stretch')
                st.caption('Recorded timings are descriptive, from separate runs on this machine. Training excludes monitoring/export; validation includes reset/recording/export. Latency is the historical 100-query mean, excluding rendering. Checkpoint bytes exclude metadata and do not measure RAM. SD is sample SD across training seeds.')
                with st.expander('Per-seed metrics and recorded costs'):
                    records = [{'Algorithm': system['label'], 'Seed': seed['seed'],
                                'Run': seed['run_id'], 'Selected interaction': seed['selected_interactions'],
                                **{k:v for k,v in seed['metrics'].items() if k!='counts'}, **seed['costs']}
                               for system in details['systems'].values() for seed in system['seed_results']]
                    st.dataframe(pd.DataFrame(records),hide_index=True,width='stretch')
                    st.download_button('Download four-algorithm metrics and costs',
                        (root/'artifacts/studies'/ANALYSIS_STUDY/'costs.csv').read_bytes(),
                        file_name='four-rl-costs.csv',mime='text/csv')
            saved_image(root/'artifacts/studies'/FOUR_WAY_STUDY/'comparison.png')
        else:
            st.info('The PPO extension has not produced a verified four-algorithm comparison yet. The existing results below cover the original release.')
        st.divider()
    final=final_summary(root)
    st.caption(f"{task.grid.map_id} · {task.config['environment_version']} · development validation · "
               + ('frozen final evaluation available' if final else 'final test sealed'))
    if final and page in ('Overview','Agent comparison'):
        task.check(final['signature'])
        st.subheader('Frozen final evaluation')
        st.caption('Held-out initial conditions on the same familiar map. All settings and checkpoints were frozen before this evaluation; no final-test selection. The simulator continues to use validation/custom scenarios.')
        final_rows=[{'System':name,'Replicates':system['replicates'],'Replicate unit':system['replicate_unit'],
                     **{m:system['metrics'][m]['mean'] for m in ('success_rate','collision_rate','timeout_rate','spl')}}
                    for name,system in final['systems'].items()]
        st.dataframe(pd.DataFrame(final_rows),hide_index=True,width='stretch')
        st.write(f"Predeclared selected system: **{final['selected_system']}**. It remains selected regardless of final ranking.")
        st.divider()
    if page=='Methodology':
        st.markdown('''The robot observes its row/column and goal row/column. Four discrete
actions move one cell. A wall or boundary attempt ends in collision; reaching the
goal ends in success; the cap of 200 actions truncates the episode. The original
reward is −0.01 per action, +10 at the goal and −5 on collision.

Q-learning and SARSA use coordinate-indexed tables. DQN scales the same four
coordinates by dimension minus one. BFS knows the obstacle map and supplies a
planner reference; learned agents receive no map, action mask or planner fallback.
The additional PPO baseline uses the same normalized coordinates, a categorical
four-action policy and separate actor/value networks. Its diagnostics are action
probabilities and a state-value estimate. It is a post-release exploratory
extension and is excluded from the original frozen final evaluation.
Voting uses three frozen Q-learning members, majority action and first-member
priority for three-way ties. Q-values are scores, never probabilities.

The saved train/validation/final manifests partition start–goal pairs on a familiar
map. States can overlap through training trajectories. Checkpoints and settings
were selected using development validation; these are not unseen-map or sealed
test results. The final test remains unopened by this dashboard.

Each learned run received 200,000 training interactions. Part A compared five
seeds per family. Part B spent an additional 12.4 million interactions across
search, controls, confirmations and reward ablations; voting reused three members.
Snapshots are inference-only and do not support exact training resume.

Success, collision and timeout partition completed episodes. SPL is success times
shortest distance / max(shortest distance, actions), averaged over every episode;
failures contribute zero. Successful route length is conditional on success and
must be read with its success count. Seed SD is sample SD, not a confidence interval.
Interactive demos/comparisons do not enter benchmark summaries.''')
        st.json(task.signature)
        st.caption('Protocol and full definitions: docs/milestone5_protocol.md. Final freeze/test is milestone 6. PPO is the additional fourth-algorithm extension; dynamics noise, unseen maps and hardware remain deferred.')
        return
    try:
        summary=evidence(root)
        task.check(summary['signature'])
    except (OSError,ValueError,KeyError) as exc:
        st.info(f'Saved dashboard evidence unavailable: {exc}. Run scripts/analyze_core_robustness.py once with a fresh suite ID.')
        return
    study=root / 'artifacts/studies/milestone5-v1'
    if page=='Overview':
        st.write('A local research dashboard for coordinate-only navigation on one fixed map. Explore recorded decisions, compare frozen checkpoints and inspect weak scenario groups.')
        a,b,c=st.columns(3)
        a.metric('Implemented learned families','4')
        b.metric('Validation pairs per seed','928')
        c.metric('Trained RL families','4' if fourth else '3')
        st.dataframe(pd.DataFrame(metric_rows(summary))[['System','Seeds','success_rate','collision_rate','timeout_rate','spl']],hide_index=True,width='stretch')
        st.info('Q-learning improved with tuning. SARSA still misses its declared targets. Voting lost five goals versus its predeclared single member while improving SPL. Removing the step cost hurt both tabular families.')
        st.write('Use Navigation simulator for saved replay, a frozen custom demonstration, matched agents or early-versus-selected checkpoints. Learning analysis uses saved logs; Optimization and robustness shows outcome-independent scenario groups.')
        st.caption('Evidence is local under artifacts/studies/milestone5-v1. Original experiment snapshots and canonical benchmark files are unchanged.')
    elif page=='Learning analysis':
        if fourth:
            st.subheader('Four-algorithm validation monitoring')
            st.caption('Additional PPO baseline alongside the original three Part A families; exploratory extension.')
            saved_image(root/'artifacts/studies'/FOUR_WAY_STUDY/'learning.png')
        st.write('Read-only 30-pair validation monitoring curves. The band is ±1 sample seed SD, clipped to [0,1]. Selected checkpoint results use the complete 928-pair validation set.')
        saved_image(study / 'figures/learning.png')
        names=[n for n,s in summary['systems'].items() if s['learning_curve']]
        name=st.selectbox('Learning system',names)
        frame=pd.DataFrame([{'interactions':r['interactions'],'success':r['success_rate']['mean'],
                             'success seed SD':r['success_rate']['sd'],'SPL':r['spl']['mean'],
                             'SPL seed SD':r['spl']['sd']} for r in summary['systems'][name]['learning_curve']])
        st.dataframe(frame,hide_index=True,width='stretch')
        st.dataframe(pd.DataFrame(summary['systems'][name]['seed_results']).drop(columns=['metrics']),hide_index=True,width='stretch')
    elif page=='Agent comparison':
        st.dataframe(pd.DataFrame(metric_rows(summary)),hide_index=True,width='stretch')
        st.caption('Rates use fractions (1 = 100%). Means and sample SD span five training seeds; voting is one fixed ensemble with no ensemble SD. DQN uses historical Part A seeds and is not a seed-matched Part B control.')
        name=st.selectbox('Inspect system',list(summary['systems']))
        rows=[]
        for seed in summary['systems'][name]['seed_results']:
            rows.append({k:v for k,v in {**seed,**seed['metrics']}.items() if k not in ('metrics','counts')})
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        vote=saved_json(root / 'artifacts/studies/part-b-v1/voting_comparison.json')
        if vote is None:
            return
        st.write(f"Voting Q arrays: {vote['voting_table_memory_bytes']:,} bytes; single: {vote['single_table_memory_bytes']:,} bytes, excluding Python overhead. Member training: {vote['member_training_interactions']:,} interactions, already counted in Part B.")
        st.dataframe(pd.DataFrame([{'policy':k,**v} for k,v in vote['timing'].items()]),hide_index=True)
        st.caption('Above latency uses the shared 1,000-query benchmark. Per-run evaluation runtimes include reset/recording/export; they are local measurements, not isolated hardware guarantees. Voting wrapper training is zero, but member cost is not zero.')
    elif page=='Optimization and robustness':
        if details:
            st.subheader('Four baseline algorithms: fixed-map robustness')
            st.caption('Original Part A Q-learning, SARSA and DQN plus PPO; five seeds per family. Unchanged predefined geometry labels on development validation. These are exploratory, overlapping scenario dimensions on one familiar map, not causal effects or independent test sets.')
            dimensions = list(dict.fromkeys(g['dimension'] for g in details['systems']['ppo']['groups']))
            dimension = st.selectbox('Four-algorithm scenario dimension', dimensions, key='four_way_dimension')
            rows = []
            for system in details['systems'].values():
                for group in system['groups']:
                    if group['dimension'] == dimension:
                        route = group['metrics']['mean_successful_route_length']
                        rows.append({'Algorithm':system['label'],'Group':group['group'],
                                     'Pairs per seed':group['scenarios_per_seed'],'Seeds':system['seed_count'],
                                     **{m:group['metrics'][m]['mean'] for m in ('success_rate','collision_rate','timeout_rate','spl')},
                                     'Success seed SD':group['metrics']['success_rate']['sample_sd'],
                                     'Successful route actions':route['mean'],
                                     'Route contributing seeds':route['contributing_seeds'],
                                     'Successes across five seeds':group['counts']['success']})
            st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch',
                         column_config={'Group':st.column_config.TextColumn(width='medium')})
            st.caption('Each dimension partitions all 928 pairs per seed. Route means exclude zero-success seeds; an empty value means no successful routes, never a zero-action solution. Corridor labels describe the canonical BFS route. Groups differ in distance and geometry.')
            st.download_button('Download four-algorithm robustness by seed',
                (root/'artifacts/studies'/ANALYSIS_STUDY/'groups.csv').read_bytes(),
                file_name='four-rl-robustness.csv',mime='text/csv')
            st.divider()
        st.subheader('Matched baseline versus refinement')
        paired=saved_json(root / 'artifacts/studies/part-b-v1/paired_improvements.json')
        if paired is not None:
            st.json(paired,expanded=False)
        st.caption('Original settings and tuned settings use the same fresh seeds and budgets. Hyperparameters were selected on separate development seeds. Reward-only ablation is diagnostic; returns across reward versions are not a quality ranking.')
        for filename in ('paired_refinement.png', 'reward_ablation.png', 'voting.png'):
            saved_image(root / 'artifacts/studies/part-b-v1/figures' / filename)
        st.subheader('Core robustness on the fixed map')
        st.write('Labels were fixed before aggregating learned outcomes. Each dimension partitions the same validation scenarios; dimensions overlap. Corridor means the canonical BFS route intersects a cell with exactly two opposite free neighbors, including endpoints. Near wall means the start has a blocked or out-of-bounds cardinal neighbor. Regions are map quadrants.')
        dimensions=list(dict.fromkeys(r['dimension'] for r in next(iter(summary['systems'].values()))['groups']))
        dimension=st.selectbox('Scenario dimension',dimensions)
        rows=[]
        for name,system in summary['systems'].items():
            for row in system['groups']:
                if row['dimension']==dimension:
                    rows.append({'System':name,'Group':row['group'],'Pairs per seed':row['scenarios_per_seed'],
                        'Seeds':system['seed_count'],**{m:row['aggregate'][m]['mean'] for m in ('success_rate','collision_rate','timeout_rate','spl')},
                        'success seed SD':row['aggregate']['success_rate']['sd']})
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        saved_image(study / 'figures/distance.png')
        saved_image(study / 'figures/groups.png')
        st.caption('Per-seed group metrics, conditional successful route lengths/counts and seed-matched group deltas are retained in groups.csv, summary.json and paired_groups.json. Small groups and five seeds limit interpretation; no causal or unseen-state claim is made.')


def overlays(prefix='single'):
    return {'trail':st.checkbox('Traveled trail',True,key=f'{prefix}_trail'),
            'visits':st.checkbox('Prefix visit counts',False,key=f'{prefix}_visits'),
            'bfs':st.checkbox('BFS planner reference',False,key=f'{prefix}_bfs')}


def diagnostics(episode,index,key):
    if index==0:
        st.caption('No action has been displayed. Decision scores are unavailable at the initial frame.')
        return
    transition=episode['transitions'][min(index,len(episode['transitions']))-1]
    st.caption(f'Recorded decision for displayed action {min(index,len(episode["transitions"]))}, at before-state {transition["observation"][:2]}. Scores describe the saved model output, not an explanation of reasoning.')
    if transition.get('scores') is not None:
        st.dataframe(pd.DataFrame({'Action':ACTION_NAMES,'Q score':transition['scores']}),hide_index=True)
        st.caption('Q-values are not probabilities.')
    elif transition.get('decision'):
        decision=transition['decision']
        if 'action_probabilities' in decision:
            st.dataframe(pd.DataFrame({'Action':ACTION_NAMES,'Action probability':decision['action_probabilities']}),hide_index=True)
            st.caption(f"PPO categorical action probabilities; selected action is the largest probability. State value: {decision['state_value']:.3f}. These are not Q-values.")
        else:
            st.json(decision)
    else:
        st.caption('Action scores unavailable for this policy.')


def event_log(episode,index,key,callback,disabled=False):
    # Prefix-only: selecting an event cannot reveal or draw future decisions.
    events=[{'Frame':0,'Action':'Initial state','Reward':0.,'Status':'ready'}]
    events += [{'Frame':i,'Action':ACTION_NAMES[t['action']],'Reward':t['reward'],'Status':t['info']['outcome']}
               for i,t in enumerate(episode['transitions'][:index],1)]
    st.dataframe(pd.DataFrame(events),hide_index=True,width='stretch')
    st.selectbox('Seek displayed event',range(index+1),index=index,key=key,on_change=callback,args=(key,),
                 format_func=lambda i:f'Frame {i}',disabled=disabled)
    if disabled:
        st.caption('Pause playback to seek a displayed event.')


def render_comparison(task,directory,validation,runs,root):
    with st.sidebar:
        st.header('Comparison setup')
        kind=st.radio('Comparison type',['Another agent','Early versus selected checkpoint'])
        scenario=st.selectbox('Validation scenario',validation,
                             format_func=lambda s:f"{s['id']} · shortest {s['shortest_length']}",
                             key='comparison_scenario')
        second=directory
        early=None
        if kind=='Another agent':
            preferred='m4-control-q-learning-seed101' if directory.name==SELECTED_RUN else SELECTED_RUN
            default=next((i for i,p in enumerate(runs) if p.name==preferred and p!=directory),
                         next((i for i,p in enumerate(runs) if p!=directory),0))
            second=st.selectbox('Compare with run',runs,index=default,format_func=run_label,key='comparison_second')
        else:
            try:
                early=early_checkpoint(directory,root)
                st.caption('Early: 10,000 interactions. Later: monitoring-selected snapshot, which may precede the full budget.')
            except (OSError,ValueError) as exc:
                st.info(str(exc))
                return
    identity=(str(directory),str(second),scenario['id'],kind)
    if st.session_state.get('comparison_identity')!=identity:
        try:
            records=[frozen_episode(task,directory,scenario,early),frozen_episode(task,second,scenario)]
            st.session_state['comparison_playback']=ComparisonPlayback(records)
            st.session_state['comparison_identity']=identity
            st.session_state['comparison_seek']=0
            st.session_state['comparison_tick']=time.monotonic()
        except (OSError,ValueError,KeyError,RuntimeError,ImportError) as exc:
            st.error(f'Comparison unavailable: {exc}')
            return
    player=st.session_state['comparison_playback']
    context_strip(player.episodes[0])
    with st.sidebar.expander('Map layers'):
        options=overlays('comparison')
    with st.sidebar.expander('Comparison details'):
        st.caption('Frozen policies on validation scenarios. No learning or benchmark result writes.')
        st.caption('Both panels share map, start, goal, reward, replay seed and action index. Finished agents hold their terminal frame.')
        st.caption('BFS knows the obstacle map. Learned agents see only robot and goal coordinates.')
    def command(name):
        player=st.session_state['comparison_playback']
        getattr(player,name)()
        st.session_state['comparison_playback']=player
        st.session_state['comparison_seek']=player.index
        st.session_state['comparison_tick']=time.monotonic()
    def seek(key):
        seek_from_widget(st.session_state, key, player_key='comparison_playback',
                         position_key='comparison_seek', active_key='comparison_timeline_key')
    with st.container(key='comparison_playback_controls'):
        controls=st.columns([1,1,1.2,1,2.2],vertical_alignment='center')
        for col,label,name,icon in zip(controls,['Play','Pause','Next step','Reset'],
                                      ['play','pause','next','reset'],
                                      ['play_arrow','pause','skip_next','restart_alt']):
            col.button(label,on_click=command,args=(name,),key=f'comparison_{name}',icon=f':material/{icon}:',
                       type='primary' if name=='play' else 'secondary',width='stretch')
        speed=controls[4].select_slider('Speed (actions / second)',options=[.5,1.,2.,4.],value=2.,key='comparison_speed')
    @st.fragment(run_every=.2 if player.playing else None)
    def panels():
        player=st.session_state['comparison_playback']
        now=time.monotonic()
        if player.playing and now-st.session_state['comparison_tick']>=1/speed:
            player.tick()
            st.session_state['comparison_playback']=player
            st.session_state['comparison_seek']=player.index
            st.session_state['comparison_tick']=now
            if not player.playing:
                st.rerun()
        timeline_key=f'comparison-timeline:{identity}:{player.index}'
        st.session_state['comparison_timeline_key']=timeline_key
        st.slider('Replay position',0,player.length,value=player.index,key=timeline_key,
                  on_change=seek,args=(timeline_key,))
        st.caption(f'Shared action index: {player.index} · '+('Playing' if player.playing else 'Paused'))
        for panel_index,(col,episode,run) in enumerate(zip(st.columns(2),player.episodes,(directory,second))):
            with col:
                age='Early · 10,000 interactions' if early and run==directory and episode is player.episodes[0] else 'Selected checkpoint'
                panel_heading(run,f"{age} · replay seed {episode['seed']}")
                show_map(task.grid,episode,player.index,**options)
                telemetry(episode,player.index,task.config['max_steps'],player.playing,shared_index=player.index)
                with st.container(key=f'comparison_decision_details_{panel_index}'):
                    with st.expander('Recorded decision and checkpoint'):
                        st.caption(episode['display_checkpoint'])
                        st.caption(episode['policy_id'])
                        diagnostics(episode,min(player.index,len(episode['transitions'])),'comparison')
    panels()
