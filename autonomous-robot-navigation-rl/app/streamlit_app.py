"""Local, read-only playback plus explicitly labeled frozen-policy custom demos."""
import time

from robot_navigation.agents import BFSPolicy, RandomPolicy, load_policy, viewer_checkpoint
from robot_navigation.benchmark import load_scenarios
from robot_navigation.core import ROOT, NavigationEnv, load_task, read_json, validate_scenario
from robot_navigation.episodes import Playback, rollout, validate_replay
from robot_navigation.refinement import task_for_run
from robot_navigation.interface import apply_theme, preferred_runs, run_label, context_strip, show_map, telemetry
from robot_navigation.dashboard import PAGES, seek_from_widget
from robot_navigation.dashboard_ui import render_page, render_comparison, overlays, diagnostics, event_log
from robot_navigation.release import final_summary

import streamlit as st

st.set_page_config(page_title="Navigation Control Room", page_icon=":material/explore:", layout="wide")
apply_theme()
with st.sidebar:
    st.markdown('<div class="lab-brand"><div class="lab-eyebrow">NAVIGATION LAB</div><strong>Autonomous Robot</strong><div class="lab-subtle">Local research · Frozen policies</div></div>', unsafe_allow_html=True)
    page = st.selectbox('Dashboard view', PAGES, key='dashboard_page')


@st.cache_resource
def task_resource():
    return load_task()


try:
    task = task_resource()
    validation = load_scenarios(task, "validation")
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Benchmark unavailable: {exc}. Run python -m robot_navigation.cli prepare first.")
    st.stop()

if page != 'Navigation simulator':
    render_page(page, task, ROOT)
    st.stop()

st.markdown('<div class="lab-eyebrow">SIMULATOR / FIXED-MAP NAVIGATION</div>', unsafe_allow_html=True)
st.title("Navigation control room")
runs = preferred_runs(p for p in (ROOT / "artifacts/runs").glob("*") if (p / "manifest.json").exists())
if not runs:
    st.info("No saved runs yet. Run python -m robot_navigation.cli smoke, then refresh.")
    st.stop()

with st.sidebar:
    st.header("Mission setup")
    directory = st.selectbox("Saved run", runs, format_func=run_label, key="saved_run")
    st.caption(directory.name)
    source = st.radio("Episode source", ["Saved replay", "Custom demonstration", "Matched comparison"])
    st.caption('Frozen final evaluation is complete. This simulator uses validation/custom scenarios only.'
               if final_summary(ROOT) else 'The final test holdout is sealed. This viewer never opens its manifest.')

try:
    # Check the run and the actual checkpoint even when opening its saved replays.
    task = task_for_run(task, directory)
    frozen_agent = load_policy(viewer_checkpoint(directory), task)
except (OSError, ValueError, KeyError, ImportError) as exc:
    st.error(f"Cannot open this fixed-map run: {exc}")
    st.stop()

if source == 'Matched comparison':
    render_comparison(task_resource(), directory, validation, runs, ROOT)
    st.stop()


def install_episode(episode, identity):
    validate_replay(task, episode)
    st.session_state["playback"] = Playback(episode)
    st.session_state["episode_identity"] = identity
    st.session_state["seek"] = 0
    st.session_state["last_tick"] = time.monotonic()


if source == "Saved replay":
    paths = sorted((directory / "replays").glob("*.json"))
    if not paths:
        st.warning("No saved replays in this run.")
        st.stop()
    # Put a verified start-to-goal reference first, and retain actual RL failures.
    paths.sort(key=lambda p: ("success" not in p.stem, p.name != "bfs_success.json", p.name))
    path = st.sidebar.selectbox("Recorded episode", paths, format_func=lambda p: p.stem.replace("_", " "),
                        key=f"recorded_episode:{directory}")
    identity = str(path)
    if st.session_state.get("episode_identity") != identity:
        try:
            episode = read_json(path)
            if episode["scenario"]["id"] not in {s["id"] for s in validation} or episode["mode"] != "validation":
                raise ValueError("Saved library accepts validation replays only")
            install_episode(episode, identity)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            st.error(f"Invalid replay: {exc}")
            st.stop()
else:
    st.info("Custom demonstration · runs a frozen policy once, then replays it. Results are excluded from the benchmark.")
    with st.sidebar.form("custom"):
        columns = st.columns(2)
        sr = columns[0].number_input("Start row", 0, 9, 0)
        sc = columns[1].number_input("Start column", 0, 9, 0)
        columns = st.columns(2)
        gr = columns[0].number_input("Goal row", 0, 9, 9)
        gc = columns[1].number_input("Goal column", 0, 9, 9)
        learned_label = {"q-learning": "Q-learning", "sarsa": "SARSA", "dqn": "DQN", "ppo": "PPO", "voting": "Policy voting"}[frozen_agent.algorithm]
        model = st.selectbox("Frozen policy", ["BFS", "Random (untrained)", learned_label])
        seed = st.number_input("Demonstration seed", 0, 2147483647, 7)
        generate = st.form_submit_button("Generate demonstration", type="primary", width="stretch")
    if generate:
        try:
            scenario = {"id": f"custom-{sr}-{sc}-{gr}-{gc}", "start": [sr, sc], "goal": [gr, gc]}
            validate_scenario(task.grid, scenario)
            scenario["shortest_length"] = len(task.grid.path(scenario["start"], scenario["goal"])) - 1
            policy = {"BFS": lambda: BFSPolicy(task.grid), "Random (untrained)": lambda: RandomPolicy(seed),
                      learned_label: lambda: frozen_agent}[model]()
            episode = rollout(NavigationEnv(task, [scenario]), policy, scenario, seed, "custom demonstration")
            install_episode(episode, f"custom/{directory.name}/{scenario['id']}/{model}/{seed}")
        except (ValueError, RuntimeError) as exc:
            st.error(f"Invalid custom scenario: {exc}")
    if not str(st.session_state.get("episode_identity", "")).startswith(f"custom/{directory.name}/"):
        st.info("Set a start, goal and frozen policy in Mission setup, then generate a demonstration.")
        st.stop()

playback = st.session_state["playback"]
episode = playback.episode
context_strip(episode)
with st.sidebar.expander("Map layers"):
    overlay_options = overlays()
with st.sidebar.expander("Model and task details"):
    st.caption(episode["policy_id"])
    st.caption(f"Checkpoint: {viewer_checkpoint(directory).name} · selected interaction: {read_json(directory / 'manifest.json').get('selected_interactions', 'legacy snapshot')}")
    st.caption(f"{task.grid.map_id} · replay seed {episode['seed']}")
    st.caption(f"Reward: {task.config['reward_version']} · step {task.config['step_reward']:+.2f}, goal +{task.config['goal_reward']}, collision {task.config['collision_reward']}")
    if frozen_agent.algorithm == "voting":
        st.caption(f"Frozen {frozen_agent.family} voting · member seeds {list(frozen_agent.seeds)} · ties use the first member")
    st.write("BFS knows the obstacle map. Learned agents see only robot and goal coordinates. Random is untrained.")
    st.caption("One action moves one grid cell. Walls and boundaries end the episode in collision. Repeated cells do not end an episode.")


def control(command):
    player = st.session_state["playback"]
    getattr(player, command)()
    st.session_state['playback'] = player
    st.session_state["seek"] = player.index
    st.session_state["last_tick"] = time.monotonic()


def seek(key):
    seek_from_widget(st.session_state, key, player_key='playback',
                     position_key='seek', active_key='active_timeline_key')


def seek_event(key):
    player = st.session_state['playback']
    # Stale event widgets must not rewind timer playback when its prefix grows.
    if player.playing:
        return
    seek_from_widget(st.session_state, key, player_key='playback',
                     position_key='seek', active_key='active_event_key')


# Keep action controls stable while the timer refreshes the display below.
with st.container(key="single_playback_controls"):
    controls = st.columns([1, 1, 1.2, 1, 2.2], vertical_alignment="center")
    for col, label, command, icon in zip(controls, ["Play", "Pause", "Next step", "Reset"],
                                       ["play", "pause", "next", "reset"],
                                       ["play_arrow", "pause", "skip_next", "restart_alt"]):
        col.button(label, on_click=control, args=(command,), icon=f":material/{icon}:",
                   type="primary" if command == "play" else "secondary", width="stretch")
    speed = controls[4].select_slider("Speed (actions / second)", options=[.5, 1., 2., 4.], value=2., key="speed")


@st.fragment(run_every=.2 if playback.playing else None)
def navigation_viewer():
    player = st.session_state["playback"]
    now = time.monotonic()
    if player.playing and now - st.session_state["last_tick"] >= 1 / speed:
        player.tick()
        st.session_state['playback'] = player
        st.session_state["seek"] = player.index
        st.session_state["last_tick"] = now
        if not player.playing:
            st.rerun()  # Rebuild with the paused timer disabled at the terminal frame.
    timeline_key = f'timeline:{st.session_state["episode_identity"]}:{player.index}'
    st.session_state['active_timeline_key'] = timeline_key
    st.slider("Replay position", 0, len(player.episode["transitions"]), value=player.index,
              key=timeline_key, on_change=seek, args=(timeline_key,))
    with st.container(key='single_replay_content'):
        left, right = st.columns([2, 1.2], gap="medium")
        with left:
            show_map(task.grid, player.episode, player.index, **overlay_options)
        with right:
            st.subheader("Episode telemetry")
            telemetry(player.episode, player.index, task.config["max_steps"], player.playing, paired=False)
            with st.container(key='single_decision_details'):
                with st.expander("Recorded decision"):
                    diagnostics(player.episode, player.index, 'single')
            st.caption("Trail and diagnostics show the displayed prefix. Future actions remain hidden.")
    with st.expander('Displayed step event log'):
        event_key = f'events:{st.session_state["episode_identity"]}:{player.index}'
        st.session_state['active_event_key'] = event_key
        event_log(player.episode, player.index, event_key, seek_event, disabled=player.playing)


navigation_viewer()

with st.expander("Saved training progress and measured validation"):
    try:
        pilot = read_json(directory / "pilot.json")
        validation_results = read_json(directory / "validation.json")
        st.write(f"{pilot['training_interactions']:,} real training interactions · "
                 f"{pilot['successful_training_episodes']} successful training episodes · "
                 f"{pilot['distinct_goals_discovered']} goals discovered")
        st.json({name: {key: result[key] for key in ("episodes", "counts", "spl", "mean_return")}
                 for name, result in validation_results.items() if name != "bfs_training_correctness"})
        st.caption("One saved seed on held-out initial conditions on a familiar map. No final-test results.")
    except (OSError, KeyError, ValueError) as exc:
        st.info(f"Progress summary unavailable: {exc}")
