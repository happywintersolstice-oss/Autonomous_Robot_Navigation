"""Task-based replay screens. Learning uses saved actions, never inferred history."""
import time

import streamlit as st

from .agents import BFSPolicy, RandomPolicy, load_policy, viewer_checkpoint
from .core import NavigationEnv, read_json, validate_scenario
from .dashboard import seek_from_widget, training_paths, training_episode
from .dashboard_ui import overlays, diagnostics, event_log
from .episodes import Playback, rollout, validate_replay
from .interface import (context_strip, show_map, telemetry, run_selector, card_heading,
                        playback_toolbar, playback_status, replay_statistics, reward_chart,
                        run_label, status_badge, repeated_pattern)
from .refinement import task_for_run


TAB_LABELS = {'trained':'Watch the robot', 'learning':'Learning history', 'custom':'Choose a route'}


def pause_players():
    """A hidden tab must never keep advancing a recording."""
    for key in ('playback', 'comparison_playback', 'unseen_playback', 'recovery_playback', 'lab_playback', 'lab_comparison_playback'):
        player = st.session_state.get(key)
        if player is not None:
            player.pause()
            st.session_state[key] = player


def render_replay(base_task, validation, runs, root, mode='trained'):
    learning = mode == 'learning'
    descriptions = {
        'trained':'Watch a saved model after training. It makes fixed decisions here; it does not learn during replay.',
        'learning':'Watch actions recorded while the robot was learning. These include exploration and changing policies.',
        'custom':'Choose a start and goal, then let a trained model try your route. This does not retrain the model.'}

    def command(name):
        current = st.session_state['playback']
        getattr(current, name)()
        st.session_state['playback'] = current
        st.session_state['seek'] = current.index
        st.session_state['last_tick'] = time.monotonic()

    if learning:
        available = [run for run in runs if training_paths(run)]
        if not available:
            st.info('No training episodes were recorded in these runs. Historical learning cannot be reconstructed from a trained model.')
            return
        family_order = {'q-learning':0, 'sarsa':1, 'dqn':2, 'ppo':3}
        runs = sorted(available, key=lambda p: (not p.name.startswith('learning-replay-'),
            family_order.get(read_json(p/'manifest.json').get('algorithm'), 4), 'pilot' in p.name, p.name))
        families = list(dict.fromkeys(read_json(p/'manifest.json').get('algorithm') for p in runs))
        from .interface import ALGORITHM_LABELS
        recording_help = ('Available training recordings: '+', '.join(ALGORITHM_LABELS.get(f,f) for f in families)+'. '
                   'New demo runs sample several stages of training. The older DQN input study saved only its first three episodes. '
                   'Older Q-learning, SARSA, DQN and PPO histories were not reconstructed. Use Learning analysis for saved monitoring curves.')

    with st.container(key='model_setup'):
        model_column = st.expander('Robot settings and saved models')
        episode_column = st.container()
        controls_column = st.container()
        with model_column:
            directory = run_selector(runs, key='learning_run' if learning else 'saved_run',
                                     label='Training run' if learning else 'Trained model')
        if directory is None:
            return
        try:
            task = task_for_run(base_task, directory)
            manifest = read_json(directory / 'manifest.json')
            # Learning history never queries a trained policy to invent past actions.
            policy = None if learning else load_policy(viewer_checkpoint(directory), task)
            with episode_column:
                if mode == 'custom':
                    st.caption('Set your route below. Rows and columns run from 0 to 9.')
                    path = None
                elif learning:
                    paths = training_paths(directory)
                    trace_items = {directory/item['path']:item for item in manifest['training_traces']['records']}
                    def training_label(path):
                        item = trace_items[path]
                        suffix = (f"from training step {item['training_start_interaction']:,}"
                                  if 'training_start_interaction' in item else 'recorded during training')
                        if 'recording_window_index' in item:
                            boundary = manifest['training_traces']['window_starts'][item['recording_window_index']]
                            percent = round(100*boundary/manifest['training_interactions'])
                            suffix = f"{percent}% training stage · {suffix}"
                        return f"Episode {int(path.stem.split('-')[-1])} · {suffix}"
                    path = st.selectbox('Training episode', paths, key=f'training_episode:{directory}',
                                       format_func=training_label)
                else:
                    paths = sorted((directory / 'replays').glob('*.json'))
                    learned_prefix = manifest.get('algorithm', 'q-learning').replace('-', '_')
                    paths.sort(key=lambda p: (not p.stem.startswith(learned_prefix), 'success' not in p.stem, p.name))
                    if not paths:
                        st.info('This model has no saved example episodes. Use Choose a route instead.')
                        return
                    path = st.selectbox('Recorded episode', paths, key=f'recorded_episode:{directory}',
                                        format_func=lambda p:p.stem.replace('_', ' ').replace('q learning', 'Q-learning').replace('bfs', 'Shortest-path planner').replace('timeout', '(runs out of moves)').replace('collision', '(hits a wall)').replace('success', '(reaches the goal)'),
                                        help='Saved examples include successes and failures. BFS and Random are reference methods, not trained RL agents.')
            identity = f'{mode}/{path}'
            if mode != 'custom' and st.session_state.get('episode_identity') != identity:
                if learning:
                    episode = training_episode(task, directory, path, root)
                else:
                    episode = read_json(path)
                    if episode['scenario'] not in validation or episode['mode'] != 'validation':
                        raise ValueError('Watch the robot accepts canonical validation episodes only')
                    validate_replay(task, episode)
                install_episode(episode, identity)
        except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
            st.error(f'Cannot open this recording: {exc}')
            return

    if mode == 'custom':
        custom_route(task, directory, policy)
        if not str(st.session_state.get('episode_identity', '')).startswith(f'custom/{directory.name}/'):
            st.info('Choose your start and goal, then select Generate demonstration.')
            return

    player = st.session_state['playback']
    episode = player.episode
    # The options live below the map. Widget state is available before rendering.
    options = {'trail': st.session_state.get('single_trail', True),
               'visits': st.session_state.get('single_visits', False),
               'bfs': st.session_state.get('single_bfs', False)}

    def seek(key):
        seek_from_widget(st.session_state, key, player_key='playback',
                         position_key='seek', active_key='active_timeline_key')

    def seek_event(key):
        if not st.session_state['playback'].playing:
            seek_from_widget(st.session_state, key, player_key='playback',
                             position_key='seek', active_key='active_event_key')

    with controls_column:
        with st.container(key='single_playback_controls'):
            speed = playback_toolbar(command, compact=True)
    expected_identity = st.session_state['episode_identity']

    @st.fragment(run_every=.2 if player.playing else None)
    def viewer():
        if (st.session_state.get('episode_identity') != expected_identity
                or st.session_state.get('demo_map','bottleneck-v1')!='bottleneck-v1'
                or st.session_state.get('replay_tab') != TAB_LABELS[mode]):
            return
        current = st.session_state['playback']
        now = time.monotonic()
        if current.playing and now - st.session_state['last_tick'] >= 1 / speed:
            current.tick()
            st.session_state['playback'] = current
            st.session_state['seek'] = current.index
            st.session_state['last_tick'] = now
            if not current.playing:
                st.rerun()
        playback_status(current.index, current.playing, training=learning)
        render_frame(task, episode, current.index, current.playing, options)
        timeline_key = f'timeline:{expected_identity}:{current.index}'
        st.session_state['active_timeline_key'] = timeline_key
        st.slider('Replay position', 0, len(episode['transitions']), value=current.index,
                  key=timeline_key, on_change=seek, args=(timeline_key,), help='Move to a recorded step. Seeking pauses playback.')
        with st.expander('Step-by-step history'):
            event_key = f'events:{expected_identity}:{current.index}'
            st.session_state['active_event_key'] = event_key
            event_log(episode, current.index, event_key, seek_event, disabled=current.playing)

    viewer()
    with st.expander('Map display options'):
        overlays()
    with st.expander('Model and recording details'):
        context_strip(episode)
        st.caption(f'Run ID: {directory.name}')
        st.caption(episode['policy_id'])
        st.caption(f"{task.grid.map_id} · replay seed {episode['seed']}")
        st.caption(f"Reward: {task.config['reward_version']} · step {task.config['step_reward']:+.2f}, goal +{task.config['goal_reward']}, collision {task.config['collision_reward']}")
        if learning:
            st.caption('Chronological training traces: actual exploratory actions, excluded from validation metrics.')
        else:
            st.caption(f"Checkpoint: {viewer_checkpoint(directory).name} · selected interaction: {manifest.get('selected_interactions', 'legacy snapshot')}")
            if policy.algorithm == 'dqn':
                st.caption(f'Input encoding: {policy.encoding} · {policy.model.observation_space.shape[0]} features from the same raw coordinates')
        st.caption('One action moves one cell. A wall or boundary attempt ends the episode. Repeated cells do not end an episode.')
        st.caption('BFS knows the obstacle map. Learned agents see robot and goal coordinates. Random is untrained.')
    with st.expander('How to use this replay'):
        st.write(descriptions[mode])
        st.caption('Play animates recorded decisions. Pause stops the clock. Next step advances one action. Reset returns to the beginning.')
        st.caption('The four cards describe the displayed episode, not aggregate benchmark performance. Distances use grid cells.')
        st.caption(run_label(directory))
        if learning:
            st.caption(recording_help)
            st.caption(f"Training interactions {episode['training_start_interaction']:,}–{episode['training_end_interaction']:,}. "
                       'This is a saved recording, not live training. No trained-checkpoint scores are substituted.')
        else:
            st.caption('Replay examples use validation scenarios. “Trained” refers to the saved model, not a final-test episode.')


def render_frame(task, episode, index, playing, options):
    """Map-first replay; technical scores are available without crowding the view."""
    from .episodes import frame
    state = frame(episode,index)
    readings = replay_statistics(episode,index)
    with st.container(key='single_replay_content'):
        a,b,c = st.columns(3)
        a.metric('Moves used',f"{readings['actions']} / {task.config['max_steps']}",
                 help='Moves shown so far / maximum allowed on this route.')
        b.metric('Distance travelled',f"{readings['traveled']} cells",
                 help='Squares successfully travelled. Trying to enter a wall adds no distance.')
        c.metric('Distance to goal',f"{readings['goal_distance']} cells",
                 help='Straight grid distance, ignoring walls. The actual route may be longer.')
        map_column, explanation = st.columns([3,1.3],gap='large')
        with map_column:
            with st.container(key='single_map'):
                card_heading("The robot's route",'Follow the blue robot R towards the green goal G.')
                show_map(task.grid,episode,index,**options)
        with explanation:
            status_badge(state['status'],playing)
            st.markdown('**Reading the map**')
            st.write('**R** Robot location  \n**S** Starting square  \n**G** Destination  \n**Dark squares** Walls  \n**Blue line** Route travelled')
            st.markdown('**What happens next?**')
            if state['status']=='success':
                st.success('The robot reached its destination. Reset to watch again or choose another route.')
            elif state['status']=='collision':
                st.error('The robot tried to move into a wall or outside the map. This route has ended.')
            elif state['status']=='timeout':
                st.warning('The robot used all its moves without reaching the goal. Try another route or compare robots.')
            elif repeated_pattern(episode,index):
                st.warning('The robot is repeating its moves. You can pause and compare another robot on this route. Playback follows the recorded decisions; it does not help the robot escape.')
            else:
                st.write('Press Play to watch, or Next step to see one decision at a time. You can pause at any point.')
            st.caption('Each move covers one square. The robot does not learn during replay.')
        with st.expander('Scores and decision details'):
            telemetry(episode,index,task.config['max_steps'],playing,paired=False,show_metrics=False)
            st.metric('Reward so far',f"{readings['return']:.2f}")
            st.caption('Reward is the training score: reaching the goal earns points, while moves and collisions cost points. Compare completion rates before comparing reward totals.')
            reward_chart(episode,index)
            diagnostics(episode,index,'single')


def install_episode(episode, identity):
    st.session_state['playback'] = Playback(episode)
    st.session_state['episode_identity'] = identity
    st.session_state['seek'] = 0
    st.session_state['last_tick'] = time.monotonic()


def custom_route(task, directory, policy):
    from .interface import ALGORITHM_LABELS
    from .route_picker import pick_cells,coordinate_inputs
    keys = ('custom_sr','custom_sc','custom_gr','custom_gc')
    pick_cells(task,keys,'custom',(*task.grid.free_cells[0],*task.grid.free_cells[-1]))
    with st.form('custom'):
        sr,sc,gr,gc = coordinate_inputs(keys,*task.grid.shape)
        model = st.selectbox('Frozen policy', [ALGORITHM_LABELS[policy.algorithm], 'BFS', 'Random (untrained)'],
                             format_func=lambda value:'Shortest-path planner (BFS)' if value=='BFS' else 'Trained robot ('+value+')' if value!='Random (untrained)' else value,
                             help='Choose the trained robot, a planner that knows the map, or an untrained random reference.')
        with st.expander('Random movement settings'):
            seed = st.number_input('Demonstration seed',0,2147483647,7,
                                   help='Repeat the same random moves with the same number. This does not train a robot.')
        generate = st.form_submit_button('Generate demonstration',type='primary')
    st.caption('Generate demonstration creates the route attempt. Then press Play. Changing the squares does not change an existing replay until you generate again.')
    if generate:
        try:
            scenario = {'id':f'custom-{sr}-{sc}-{gr}-{gc}', 'start':[sr, sc], 'goal':[gr, gc]}
            validate_scenario(task.grid, scenario)
            scenario['shortest_length'] = len(task.grid.path(scenario['start'], scenario['goal'])) - 1
            selected = BFSPolicy(task.grid) if model == 'BFS' else RandomPolicy(seed) if model == 'Random (untrained)' else policy
            episode = rollout(NavigationEnv(task, [scenario]), selected, scenario, seed, 'custom demonstration')
            validate_replay(task, episode)
            install_episode(episode, f'custom/{directory.name}/{scenario["id"]}/{model}/{seed}')
        except (ValueError, RuntimeError) as exc:
            st.error(f'Invalid custom scenario: {exc}')
    st.caption('Custom demonstrations are separate from the benchmark results.')
