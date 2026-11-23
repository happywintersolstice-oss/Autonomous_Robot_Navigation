"""MediNav-RL prototype dashboard.

    streamlit run app/streamlit_app.py      (or double-click "Start App.cmd")

Tabs
    Simulator     build a hospital floor, pick an agent, watch it drive, inspect metrics
    Head-to-head  several agents on the SAME map, trajectories overlaid
    Results       tables and figures produced by the automated pipeline
    How it works  one-page explanation for the demo / Q&A
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from medinav.agents.registry import baseline, find_runs, load_agent, read_meta  # noqa: E402
from medinav.env import DIFFICULTY, make_env, map_seed  # noqa: E402
from medinav.evaluation.rollout import run_episode  # noqa: E402
from medinav.render import draw_world  # noqa: E402

st.set_page_config(page_title="MediNav-RL", page_icon="🏥", layout="wide")
RUNS = ROOT / "runs"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#8a8984"]


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def list_trained():
    """{display name: run dir} for every trained run (first seed of each label)."""
    out = {}
    if RUNS.exists():
        for rd in find_runs(RUNS):
            exp = rd.relative_to(RUNS).parent.parent.as_posix()     # e.g. "part_a" or "env_v1/part_a"
            if exp.split("/")[-1] in ("tuning", "pilot") or exp.startswith("env_v1"):
                continue  # tuning trials are temporary; env v1 models are kept as evidence, not for demos
            try:
                m = read_meta(rd)
            except Exception:  # noqa: BLE001 - a half-written run must not break the dropdown
                continue
            out[f"{m['label']} · seed {m['seed']} ({exp})"] = str(rd)
    return out


@st.cache_resource(show_spinner="Loading model…")
def get_agent(choice: str):
    if choice == "A* planner (baseline)":
        return baseline("astar"), {"action_mode": "discrete", "env": {}, "wrappers": {}}
    if choice == "Random (baseline)":
        return baseline("random"), {"action_mode": "discrete", "env": {}, "wrappers": {}}
    return load_agent(list_trained()[choice])


def safe_agent(choice: str):
    """get_agent with a friendly message instead of a crash (missing/corrupt/half-copied model)."""
    try:
        return get_agent(choice)
    except Exception as exc:  # noqa: BLE001 - shown to the user, app keeps running
        st.error(f"Could not load **{choice}**: {type(exc).__name__}: {exc}. "
                 "Re-train it or pick another agent.")
        return None, None


def agent_choices():
    return ["A* planner (baseline)", "Random (baseline)"] + list(list_trained())


def world_controls(prefix=""):
    st.sidebar.header("Hospital floor")
    split = st.sidebar.selectbox("Map set", ["test", "validation", "train"], key=prefix + "split",
                                 help="Test maps were never seen during training.")
    idx = st.sidebar.number_input("Map number", 0, 9999, 0, key=prefix + "idx")
    diff = st.sidebar.select_slider("Difficulty", options=list(DIFFICULTY), value=2, key=prefix + "diff",
                                    help="0 = open room … 3 = three walls, nine obstacles, four people")
    with st.sidebar.expander("Stress test", expanded=False):
        humans = st.slider("People (override)", -1, 8, -1, key=prefix + "hum", help="-1 = use difficulty preset")
        speed = st.slider("People speed (m/s)", 0.0, 1.2, 0.5, 0.1, key=prefix + "spd")
        late = st.slider("People start walking at step", 0, 100, 0, 5, key=prefix + "late")
        noise = st.slider("LiDAR noise σ (m)", 0.0, 0.15, 0.0, 0.01, key=prefix + "noise")
        drop = st.slider("LiDAR dropout probability", 0.0, 0.6, 0.0, 0.05, key=prefix + "drop")
        size = st.select_slider("Map size (m)", [10.0, 12.0, 14.0], 10.0, key=prefix + "size")
    cond = {"difficulty": diff, "human_speed": speed, "humans_start_step": late,
            "lidar_noise_std": noise, "dropout_prob": drop, "map_size": size,
            "n_humans": None if humans < 0 else humans}
    return map_seed(split, idx), cond


def env_for(meta, cond):
    cfg = {**(meta.get("env") or {}), **cond, "action_mode": meta["action_mode"]}
    return make_env(cfg, meta.get("wrappers"))


def frame_figure(snap, title=None):
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=100)
    draw_world(ax, snap)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


OUTCOME_TEXT = {"success": "✅ Delivered", "collision_wall": "💥 Hit a wall", "collision_human": "🚑 Hit a person",
                "timeout": "⏱️ Ran out of time", "stuck_oscillating": "🔁 Got stuck", "too_slow": "🐢 Too slow"}


# ------------------------------------------------------------------ tabs
tab_sim, tab_h2h, tab_res, tab_about = st.tabs(["Simulator", "Head-to-head", "Results", "How it works"])

with tab_sim:
    mseed, cond = world_controls()
    st.subheader("Medicine delivery run")
    c1, c2 = st.columns([1, 2])
    choice = c1.selectbox("Agent", agent_choices())
    run = c1.button("▶ Run episode", type="primary")
    if not list_trained():
        c1.info("No trained models found in runs/ yet - only the baselines are available. "
                "Train with `python -m medinav pipeline --config configs/smoke.yaml`.")
    if run:
        agent, meta = safe_agent(choice)
        if agent is not None:
            try:
                env = env_for(meta, cond)
                with st.spinner("Driving…"):
                    row, frames = run_episode(agent, env, mseed, 0, record=True)
                st.session_state["sim"] = (row, frames, choice)
            except Exception as exc:  # noqa: BLE001 - e.g. an impossible stress-test combination
                st.error(f"Episode failed: {type(exc).__name__}: {exc}")
    if "sim" in st.session_state:
        row, frames, who = st.session_state["sim"]
        # outcome as a full-width banner (a metric tile truncates long text on laptop screens)
        outcome = f"**{who}**: {OUTCOME_TEXT.get(row['failure_type'], row['failure_type'])}"
        (st.success if row["success"] else st.error)(outcome)
        m = st.columns(4)
        m[0].metric("Time", f"{row['steps'] * 0.2:.1f} s")
        m[1].metric("Distance driven", f"{row['path_length']:.1f} m", f"A* optimum {row['optimal_length']:.1f} m",
                    delta_color="off")
        m[2].metric("Closest person", "—" if np.isnan(row["min_human_dist"]) else f"{row['min_human_dist']:.2f} m")
        m[3].metric("Return", f"{row['return']:.1f}")
        left, right = c2.columns([3, 2])
        play = right.button("Play animation")
        fps = right.slider("Speed (frames/s)", 2, 30, 12)
        step = right.slider("Step", 0, len(frames) - 1, len(frames) - 1)
        terms = {k[2:]: v for k, v in row.items() if k.startswith("r_")}
        right.caption("Reward collected, by term")
        right.bar_chart(pd.Series(terms, name="reward"), height=180)
        holder = left.empty()
        if play:
            stride = max(1, len(frames) // 120)
            for i in range(0, len(frames), stride):
                fig = frame_figure(frames[i], f"{who} - t = {i * 0.2:.1f} s")
                holder.pyplot(fig)
                plt.close(fig)
                time.sleep(1 / fps)
        fig = frame_figure(frames[step], f"{who} - t = {step * 0.2:.1f} s")
        holder.pyplot(fig)
        plt.close(fig)

with tab_h2h:
    st.subheader("Same map, different brains")
    st.caption("Uses the map and stress settings from the sidebar.")
    picks = st.multiselect("Agents to compare", agent_choices(), default=["A* planner (baseline)"])
    n_maps = st.slider("Number of consecutive maps", 1, 20, 1)
    if st.button("Compare", type="primary") and picks:
        rows, fig, ax = [], *plt.subplots(figsize=(6, 6), dpi=100)
        for k, choice in enumerate(picks):
            agent, meta = safe_agent(choice)
            if agent is None:
                continue
            for j in range(n_maps):
                env = env_for(meta, cond)
                row, frames = run_episode(agent, env, mseed + j, 0, record=(j == 0))
                rows.append({"agent": choice, **{c: row[c] for c in
                             ("failure_type", "success", "steps", "path_length", "path_efficiency", "min_human_dist")}})
                if j == 0:
                    if k == 0:
                        first = dict(frames[0])
                        first["path"] = None
                        draw_world(ax, first, show_rays=False)
                    p = frames[-1]["path"]
                    ax.plot(p[:, 0], p[:, 1], color=PALETTE[k % 6], lw=2.2, label=f"{choice.split(' ·')[0]}: {row['failure_type']}")
        if not rows:
            st.stop()  # every selected agent failed to load; errors are already shown
        ax.legend(fontsize=7, loc="upper left")
        a, b = st.columns([3, 2])
        a.pyplot(fig)
        df = pd.DataFrame(rows)
        b.dataframe(df.groupby("agent")[["success", "steps", "path_efficiency", "min_human_dist"]].mean().round(2))
        b.caption(f"Averaged over {n_maps} map(s).")

with tab_res:
    st.subheader("Experiment results (generated by `python -m medinav report`)")
    # current experiments first, then the archived environment-v1 results (kept to show the iteration)
    exps = ([p.parent.relative_to(RUNS).as_posix() for p in sorted(RUNS.glob("*/results"))] +
            [p.parent.relative_to(RUNS).as_posix() for p in sorted(RUNS.glob("env_v1/*/results"))]) \
        if RUNS.exists() else []
    if not exps:
        st.info("No results yet. Run the pipeline first.")
    else:
        default = next((exps.index(e) for e in ("part_b", "part_a") if e in exps), 0)
        exp = st.selectbox("Experiment", exps, index=default,
                           help="env_v1/... = first environment version, before people yielded to the robot")
        res = RUNS / exp / "results"
        if (res / "summary.csv").exists():
            st.dataframe(pd.read_csv(res / "summary.csv").round(3))
        if (res / "significance.md").exists():
            st.markdown("**Is the difference real?** (permutation test on success)")
            st.markdown((res / "significance.md").read_text())
        for png in sorted(res.glob("*.png")):
            st.image(str(png), caption=png.stem.replace("_", " "))

with tab_about:
    st.markdown((ROOT / "docs" / "how_it_works.md").read_text() if (ROOT / "docs" / "how_it_works.md").exists()
                else "See README.md")
