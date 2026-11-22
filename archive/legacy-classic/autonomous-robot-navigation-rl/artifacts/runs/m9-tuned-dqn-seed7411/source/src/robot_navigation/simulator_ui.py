"""Task-based replay screens. Learning uses saved actions, never inferred history."""
import time

import streamlit as st

from .agents import BFSPolicy, RandomPolicy, load_policy, viewer_checkpoint
from .core import NavigationEnv, read_json, validate_scenario
from .dashboard import seek_from_widget, training_paths, training_episode
from .dashboard_ui import overlays, diagnostics, event_log
from .episodes import Playback, rollout, validate_replay
from .interface import (context_strip, panel_heading, show_map, telemetry, run_selector,
                        playback_toolbar, playback_status)
from .refinement import task_for_run


TAB_LABELS = {'trained':'Trained replay', 'learning':'Learning replay', 'custom':'Try a route'}


def pause_players():
    """A hidden tab must never keep advancing a recording."""
    for key in ('playback', 'comparison_playback'):
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
    st.write(descriptions[mode])
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
        st.caption('Available training recordings: '+', '.join(ALGORITHM_LABELS.get(f,f) for f in families)+'. '
                   'New demo runs sample several stages of training. The older DQN input study saved only its first three episodes. '
                   'Older Q-learning, SARSA, DQN and PPO histories were not reconstructed. Use Learning analysis for saved monitoring curves.')
    else:
        st.caption('Replay examples use validation scenarios. “Trained” refers to the saved model, not a final-test episode.')

    with st.container(key='model_setup'):
        model_column, episode_column = st.columns([1.4, 1], gap='large')
        with model_column:
            directory = run_selector(runs, key='learning_run' if learning else 'saved_run',
                                     label='Training run' if learning else 'Trained model')
        if directory is None:
            return
        try:
            task = task_for_run(base_task, directory)
            manifest = read_json(directory / 'manifest.json')
            # Learning replay never queries a trained policy to invent past actions.
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
                        st.info('This model has no saved example episodes. Use Try a route instead.')
                        return
                    path = st.selectbox('Recorded episode', paths, key=f'recorded_episode:{directory}',
                                        format_func=lambda p:p.stem.replace('_', ' '),
                                        help='Saved examples include successes and failures. BFS and Random are reference methods, not trained RL agents.')
            identity = f'{mode}/{path}'
            if mode != 'custom' and st.session_state.get('episode_identity') != identity:
                if learning:
                    episode = training_episode(task, directory, path, root)
                else:
                    episode = read_json(path)
                    if episode['scenario'] not in validation or episode['mode'] != 'validation':
                        raise ValueError('Trained replay accepts canonical validation episodes only')
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
    if learning:
        st.caption(f"Training interactions {episode['training_start_interaction']:,}–{episode['training_end_interaction']:,}. "
                   'This is a saved recording, not live training. No trained-checkpoint scores are substituted.')
    st.caption(f"Start {episode['scenario']['start']} → goal {episode['scenario']['goal']} · shortest route {episode['scenario']['shortest_length']} steps")
    # The options live below the map. Widget state is available before rendering.
    options = {'trail': st.session_state.get('single_trail', True),
               'visits': st.session_state.get('single_visits', False),
               'bfs': st.session_state.get('single_bfs', False)}

    def command(name):
        current = st.session_state['playback']
        getattr(current, name)()
        st.session_state['playback'] = current
        st.session_state['seek'] = current.index
        st.session_state['last_tick'] = time.monotonic()

    def seek(key):
        seek_from_widget(st.session_state, key, player_key='playback',
                         position_key='seek', active_key='active_timeline_key')

    def seek_event(key):
        if not st.session_state['playback'].playing:
            seek_from_widget(st.session_state, key, player_key='playback',
                             position_key='seek', active_key='active_event_key')

    with st.container(key='single_playback_controls'):
        speed = playback_toolbar(command)
    expected_identity = st.session_state['episode_identity']

    @st.fragment(run_every=.2 if player.playing else None)
    def viewer():
        if (st.session_state.get('episode_identity') != expected_identity
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
        timeline_key = f'timeline:{expected_identity}:{current.index}'
        st.session_state['active_timeline_key'] = timeline_key
        st.slider('Replay position', 0, len(episode['transitions']), value=current.index,
                  key=timeline_key, on_change=seek, args=(timeline_key,), help='Move to a recorded step. Seeking pauses playback.')
        playback_status(current.index, current.playing, training=learning)
        with st.container(key='single_replay_content'):
            map_column, details_column = st.columns([3, 1.15], gap='medium')
            with map_column:
                with st.container(key='single_map'):
                    panel_heading(directory, 'Recorded learning episode' if learning else 'After-training replay', episode['policy_id'])
                    show_map(task.grid, episode, current.index, **options)
            with details_column:
                with st.container(key='single_inspector'):
                    st.subheader('What is happening?')
                    telemetry(episode, current.index, task.config['max_steps'], current.playing, paired=False)
                    with st.expander('Action details'):
                        diagnostics(episode, current.index, 'single')
                    st.caption('Actions count robot steps. Reward is the total score earned so far. Revisited cells count returns to previous positions.')
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


def install_episode(episode, identity):
    st.session_state['playback'] = Playback(episode)
    st.session_state['episode_identity'] = identity
    st.session_state['seek'] = 0
    st.session_state['last_tick'] = time.monotonic()


def custom_route(task, directory, policy):
    from .interface import ALGORITHM_LABELS
    with st.form('custom'):
        start, goal, agent = st.columns(3)
        with start:
            st.markdown('**Start cell**')
            sr = st.number_input('Start row', 0, 9, 0)
            sc = st.number_input('Start column', 0, 9, 0)
        with goal:
            st.markdown('**Goal cell**')
            gr = st.number_input('Goal row', 0, 9, 9)
            gc = st.number_input('Goal column', 0, 9, 9)
        with agent:
            model = st.selectbox('Frozen policy', [ALGORITHM_LABELS[policy.algorithm], 'BFS', 'Random (untrained)'],
                                 help='The trained model is fixed. BFS is a known-map planner; Random is an untrained reference.')
            seed = st.number_input('Demonstration seed', 0, 2147483647, 7,
                                   help='Controls randomness for the Random reference. This is not a new training seed.')
        generate = st.form_submit_button('Generate demonstration', type='primary')
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
