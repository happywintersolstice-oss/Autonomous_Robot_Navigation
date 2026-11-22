"""Local, read-only playback plus explicitly labeled frozen-policy custom demos."""
import time

from robot_navigation.agents import BFSPolicy, RandomPolicy, load_policy
from robot_navigation.benchmark import load_scenarios
from robot_navigation.core import ROOT, NavigationEnv, load_task, read_json, validate_scenario
from robot_navigation.episodes import Playback, rollout, validate_replay
from robot_navigation.visualization import draw

import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Robot Navigation | Local Lab", page_icon="🧭", layout="wide")
st.markdown("""<style>
.block-container {max-width:1250px;padding-top:2rem;}
[data-testid="stMetric"] {background:#f0f6fa;border-radius:10px;padding:12px;}
</style>""", unsafe_allow_html=True)
st.caption("LOCAL RESEARCH LAB  /  TABULAR NAVIGATION")
st.title("Watch the next move")
st.write("Fixed-map robot navigation · recorded actions, visible outcomes.")


@st.cache_resource
def task_resource():
    return load_task()


try:
    task = task_resource()
    validation = load_scenarios(task, "validation")
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Benchmark unavailable: {exc}. Run python -m robot_navigation.cli prepare first.")
    st.stop()

runs = sorted((p for p in (ROOT / "artifacts/runs").glob("*") if (p / "manifest.json").exists()),
              key=lambda p: (p.name != "smoke-seed7", p.name))
if not runs:
    st.info("No saved runs yet. Run python -m robot_navigation.cli smoke, then refresh.")
    st.stop()

with st.sidebar:
    st.header("Episode library")
    directory = st.selectbox("Saved run", runs, format_func=lambda p: p.name)
    source = st.radio("Episode source", ["Saved replay", "Custom demonstration"])
    st.caption("The final test holdout is sealed. This viewer never opens its manifest.")
    st.divider()
    st.markdown("**Information available**")
    st.write("BFS knows the obstacle map. Q-learning and SARSA see only robot and goal coordinates. Random is untrained.")
    st.caption("One action moves one grid cell. Walls and boundaries end the episode in collision. Repeated cells do not end an episode.")

try:
    # Check the run and the actual checkpoint even when opening its saved replays.
    task.check(read_json(directory / "manifest.json")["signature"])
    frozen_agent = load_policy(directory / "checkpoint.npz", task)
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Cannot open this fixed-map run: {exc}")
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
    paths.sort(key=lambda p: (p.name != "bfs_success.json", p.name))
    path = st.selectbox("Recorded episode", paths, format_func=lambda p: p.stem.replace("_", " "))
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
    with st.form("custom"):
        columns = st.columns(4)
        sr = columns[0].number_input("Start row", 0, 9, 0)
        sc = columns[1].number_input("Start column", 0, 9, 0)
        gr = columns[2].number_input("Goal row", 0, 9, 9)
        gc = columns[3].number_input("Goal column", 0, 9, 9)
        learned_label = "SARSA" if frozen_agent.algorithm == "sarsa" else "Q-learning"
        model = st.selectbox("Frozen policy", ["BFS", "Random (untrained)", learned_label])
        seed = st.number_input("Demonstration seed", 0, 2147483647, 7)
        generate = st.form_submit_button("Generate demonstration")
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
        st.stop()

playback = st.session_state["playback"]
episode = playback.episode
st.caption(f"{episode['policy_id']}  ·  {task.grid.map_id}  ·  {episode['scenario']['id']}  ·  {episode['mode']}  ·  seed {episode['seed']}")
st.caption("Saved replay" if source == "Saved replay" else "Frozen-policy demonstration replay; no learning")


def control(command):
    player = st.session_state["playback"]
    getattr(player, command)()
    st.session_state["seek"] = player.index
    st.session_state["last_tick"] = time.monotonic()


def seek():
    player = st.session_state["playback"]
    player.pause()
    player.seek(st.session_state["seek"])


@st.fragment(run_every=.2)
def navigation_viewer():
    player = st.session_state["playback"]
    controls = st.columns([1, 1, 1.3, 1, 2.4])
    controls[0].button("Play", on_click=control, args=("play",), width="stretch")
    controls[1].button("Pause", on_click=control, args=("pause",), width="stretch")
    controls[2].button("Next step", on_click=control, args=("next",), width="stretch")
    controls[3].button("Reset", on_click=control, args=("reset",), width="stretch")
    speed = controls[4].select_slider("Speed (actions / second)", options=[.5, 1., 2., 4.], value=2., key="speed")
    now = time.monotonic()
    if player.playing and now - st.session_state["last_tick"] >= 1 / speed:
        player.tick()
        st.session_state["seek"] = player.index
        st.session_state["last_tick"] = now
    st.slider("Replay position", 0, len(player.episode["transitions"]), key="seek", on_change=seek)
    state = player.current
    left, right = st.columns([1.6, 1])
    with left:
        figure = draw(task.grid, player.episode, player.index)
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with right:
        st.subheader(state["status"].capitalize())
        st.caption("Playing" if player.playing else "Paused")
        a, b = st.columns(2)
        a.metric("Actions", f"{state['index']} / {task.config['max_steps']}")
        b.metric("Cumulative reward", f"{state['return']:.2f}")
        st.write(f"**Robot position:** {state['position']}")
        st.write(f"**Last attempted action:** {state['action']}")
        st.write(f"**Step reward:** {state['step_reward']:.2f}")
        st.write(f"**Revisited cells:** {state['revisited_cells']}")
        st.write(f"**Reference shortest distance:** {player.episode['scenario']['shortest_length']} actions")
        if state["status"] == "collision":
            st.error(f"Collision attempt at {state['attempted_position']}; robot remains in its actual cell.")
        if player.index == len(player.episode["transitions"]):
            st.write(f"**Completed SPL:** {player.episode['metrics']['spl']:.3f}")
        st.caption("Trail and diagnostics show the current replay prefix. Future actions remain hidden.")


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
