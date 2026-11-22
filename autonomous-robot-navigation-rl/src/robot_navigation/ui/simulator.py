"""Warehouse simulator and passive playback of recorded product journeys."""

from html import escape
from pathlib import Path
import time

import streamlit as st

from ..app_registry import registry
from ..app_scenarios import NAMES, scenario as phase2_scenario
from ..app_service import cursor, frame, generate
from ..phase3_demo import NAMES as PHASE3_NAMES, scenario as phase3_scenario
from .components import (
    LABELS,
    friendly_checkpoint,
    page_header,
    panel_title,
    status_card,
)

ACTIONS = ["Up", "Down", "Left", "Right", "Wait"]


def scenario(name, *args, **kwargs):
    factory = phase3_scenario if name in PHASE3_NAMES else phase2_scenario
    return factory(name, *args, **kwargs)


def grid_svg(snapshot, truth=False, trail=()):
    """Draw recorded scene or public memory; never supply truth to a policy."""
    size, offset = 38, 24
    occupancy = snapshot["occupancy"] if truth else snapshot["memory"]["occupancy"]
    seen = snapshot["memory"]["currently_seen"]
    cells = []
    for r in range(10):
        for c in range(10):
            value = int(occupancy[r][c])
            color = "#23404e" if value == 1 else "#091a26" if value == -1 else "#d6e3e9"
            if truth and value == 1 and snapshot["true_rows"][r][c] != "#":
                color = "#e8ac61"
            stroke = "#00a7c9" if not truth and seen[r][c] else "#47616e"
            cells.append(
                f'<rect x="{c*size+offset}" y="{r*size+offset}" width="{size}" height="{size}" rx="1" fill="{color}" stroke="{stroke}" stroke-width=".7"/>'
            )
    for r, c in trail:
        cells.append(
            f'<circle cx="{c*size+43}" cy="{r*size+43}" r="3" fill="#5a9cbb"/>'
        )
    if snapshot.get("start"):
        r, c = snapshot["start"]
        cells.append(
            f'<text x="{c*size+29}" y="{r*size+36}" fill="#124763" font-size="10">S</text>'
        )
    r, c = snapshot["goal"]
    cells.append(
        f'<rect x="{c*size+28}" y="{r*size+28}" width="30" height="30" rx="8" fill="#14816f"/><text x="{c*size+43}" y="{r*size+48}" text-anchor="middle" fill="white" font-size="14">G</text>'
    )
    r, c = snapshot["position"]
    cells.append(
        f'<circle cx="{c*size+43}" cy="{r*size+43}" r="13" fill="#087ea4" stroke="#a8edff" stroke-width="2"/><text x="{c*size+43}" y="{r*size+48}" text-anchor="middle" fill="white" font-size="14" font-weight="600">R</text>'
    )
    for i in range(10):
        cells.append(
            f'<text x="{i*size+43}" y="16" text-anchor="middle" font-size="11" fill="#96b2c5">{i}</text>'
        )
        cells.append(
            f'<text x="12" y="{i*size+47}" text-anchor="middle" font-size="11" fill="#96b2c5">{i}</text>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Robot navigation grid" viewBox="0 0 426 410">'
        + "".join(cells)
        + "</svg>"
    )


def preview_snapshot(case):
    occupancy = [[int(ch == "#") for ch in row] for row in case["rows"]]
    for event in case["events"]:
        if event["start"] <= 0 < event["clear"]:
            r, c = event["cell"]
            occupancy[r][c] = 1
    return {
        "occupancy": occupancy,
        "true_rows": case["rows"],
        "memory": {
            "occupancy": [[-1] * 10 for _ in range(10)],
            "currently_seen": [[False] * 10 for _ in range(10)],
        },
        "position": case["start"],
        "start": case["start"],
        "goal": case["goal"],
    }


def simulator():
    page_header(
        "Live simulator",
        "Autonomous Warehouse Robot Navigation",
        "Choose a route and a trained robot. Run once, then watch its decisions unfold.",
        "PHASE 3 · 10 × 10",
    )
    items = sorted(
        registry(),
        key=lambda item: (
            not item.get("phase3_selected", False),
            list(LABELS).index(item["algorithm"]),
        ),
    )
    options = {item["label"]: item for item in items}
    options.update(
        {
            "Observed-Map Replanner (non-RL reference)": "Observed-Map Replanner",
            "Random (non-RL reference)": "Random reference",
        }
    )

    def format_choice(label):
        item = options[label]
        return friendly_checkpoint(item) if isinstance(item, dict) else label

    with st.container(key="route_setup"):
        route, obstacles, controller, run = st.columns(
            [2.1, 1.25, 2.4, 0.95], gap="medium"
        )
        presets = NAMES + PHASE3_NAMES
        name = route.selectbox(
            "Map preset", presets, index=presets.index("Temporary bridge - Wait")
        )
        mode = obstacles.radio(
            "Obstacle mode", ["Static", "Temporary"], horizontal=True, index=1
        )
        choice = controller.selectbox(
            "Controller / checkpoint", list(options), format_func=format_choice
        )
        run_requested = run.button("Run journey", type="primary", width="stretch")
        default = scenario(name, mode)
        with st.expander("Route settings & checkpoint details"):
            a, b = st.columns([1, 2])
            seed = a.number_input("Evaluation seed", 0, 2**31 - 1, value=0)
            custom = b.checkbox("Choose start and goal")
            start, goal = default["start"], default["goal"]
            if custom:
                fields = st.columns(4)
                start = [
                    fields[0].number_input("Start row", 0, 9, value=start[0]),
                    fields[1].number_input("Start column", 0, 9, value=start[1]),
                ]
                goal = [
                    fields[2].number_input("Goal row", 0, 9, value=goal[0]),
                    fields[3].number_input("Goal column", 0, 9, value=goal[1]),
                ]
            if isinstance(options[choice], dict):
                item = options[choice]
                st.caption(
                    f"{item['actions']:,} training actions · training seed {item['seed']} · {item['feature_version']}"
                )
                st.code(item["id"], language=None)
            else:
                st.caption(
                    "This is a separately labeled non-RL reference, not a trained RL checkpoint."
                )
    if run_requested:
        try:
            case = scenario(name, mode, start, goal)
            with st.spinner("Recording the robot’s observations and actions…"):
                record, path = generate(case, options[choice], int(seed))
            st.session_state.update(
                journey=record, journey_file=str(path), frame_index=0, playing=False
            )
        except Exception as exc:
            st.error("Unable to run: " + str(exc))
    if not items:
        st.info(
            "No native checkpoints are available. Select a reference controller or inspect Train Models."
        )
    if "journey" in st.session_state:
        playback()
    else:
        with st.container(key="simulator_workspace"):
            a, b = st.columns([3.4, 1])
            with a, st.container(key="primary_scene"):
                panel_title(
                    "Navigation environment",
                    f"{name} · {mode.lower()} obstacles",
                    "ROUTE PREVIEW",
                )
                preview = {**default, "start": start, "goal": goal}
                st.image(grid_svg(preview_snapshot(preview), True), width="stretch")
                legend()
            with b:
                st.markdown(
                    '<div class="warehouse-zone">READY TO EXPLORE</div>',
                    unsafe_allow_html=True,
                )
                st.write("Your next delivery route")
                st.caption(
                    "Press Run journey to record the selected robot. No model trains during simulation or playback."
                )
                st.metric("Start", f"{start[0]}, {start[1]}")
                st.metric("Goal", f"{goal[0]}, {goal[1]}")
                st.caption(
                    "The robot senses up to two cells away and remembers what it has seen. Occupancy does not reveal whether a blockage is permanent."
                )


def legend():
    st.markdown(
        '<div class="warehouse-legend"><span><i class="legend-dot"></i>Robot R</span><span><i class="legend-dot goal"></i>Goal G</span><span><i class="legend-dot wall"></i>Wall</span><span><i class="legend-dot blocker"></i>Temporary blocker</span></div>',
        unsafe_allow_html=True,
    )


def playback():
    @st.fragment(run_every=0.2 if st.session_state.get("playing") else None)
    def player():
        record = st.session_state.journey
        length = len(record["transitions"])
        index = st.session_state.get("frame_index", 0)
        with st.container(key="playback_controls"):
            controls = st.columns(6)
            for col, label, command in zip(
                controls,
                ["Play", "Pause", "Back", "Step", "Reset", "End"],
                ["play", "pause", "back", "step", "reset", "end"],
            ):
                if col.button(label, key="player_" + command, width="stretch"):
                    if command in ("play", "pause"):
                        st.session_state.playing = command == "play"
                        st.session_state.last_tick = time.monotonic()
                    else:
                        st.session_state.playing = False
                        st.session_state.frame_index = (
                            length
                            if command == "end"
                            else cursor(index, command, length)
                        )
                    st.rerun()
        if st.session_state.get("playing") and time.monotonic() - st.session_state.get(
            "last_tick", 0
        ) >= 1 / st.session_state.get("replay_speed", 4):
            index = cursor(index, "step", length)
            st.session_state.frame_index = index
            st.session_state.last_tick = time.monotonic()
            if index == length:
                st.session_state.playing = False
        # Seek is read before drawing, so both views represent the same frame.
        with st.container(key="replay_footer"):
            seek, speed = st.columns([4, 1])
            chosen = seek.slider(
                "Seek recorded step",
                0,
                length,
                value=int(st.session_state.get("frame_index", 0)),
                key="seek_" + str(st.session_state.get("frame_index", 0)),
            )
            speed.select_slider(
                "Playback speed (frames/second)",
                options=[1, 2, 4, 8],
                value=4,
                key="replay_speed",
            )
        if chosen != st.session_state.get("frame_index", 0):
            st.session_state.frame_index = chosen
            st.session_state.playing = False
        index = chosen
        snap = frame(record, index)
        actions = [t["action"] for t in record["transitions"][:index]]
        last_action = ACTIONS[actions[-1]] if actions else "—"
        waits = 0
        for action in reversed(actions):
            if action != 4:
                break
            waits += 1
        metrics = record["metrics"]
        with st.container(key="simulator_workspace"):
            left, scene, right = st.columns([1, 3.25, 1.35], gap="medium")
            with left, st.container(key="journey_metrics"):
                st.markdown(
                    '<div class="warehouse-zone">PERFORMANCE</div>',
                    unsafe_allow_html=True,
                )
                st.metric("Steps", f"{index} / {length}")
                st.metric(
                    "Total reward",
                    f"{metrics['return']:.2f}",
                    help="Official total for the complete recorded journey.",
                )
                st.metric(
                    "Wait actions",
                    metrics["wait_count"],
                    help="Total for the complete recorded journey.",
                )
            with scene, st.container(key="primary_scene"):
                panel_title(
                    "Navigation environment",
                    f"{record['scenario']['id']} · recorded journey",
                    "2D · REPLAY",
                )
                truth = st.toggle("Show scene truth — viewer only", value=True)
                trail = [record["initial_snapshot"]["position"]] + [
                    t["snapshot"]["position"] for t in record["transitions"][:index]
                ]
                st.image(grid_svg(snap, truth, trail), width="stretch")
                legend()
                st.caption(
                    "Scene truth is for the viewer. It is never supplied to the controller."
                    if truth
                    else "Robot memory only: dark cells are unknown; cyan borders are sensed now."
                )
            with right:
                status_card(metrics["outcome"], index, length)
                r, c = snap["position"]
                st.markdown(
                    f'<div class="agent-state"><span>Robot position</span><b>[{r}, {c}]</b><span>Last action</span><b>{escape(last_action)}</b><span>Consecutive Waits</span><b>{waits}</b></div>',
                    unsafe_allow_html=True,
                )
                with st.container(key="memory_scene"):
                    panel_title("Robot memory", "Only the robot’s public observations")
                    st.image(grid_svg(snap, False), width="stretch")
                st.caption(
                    "Sensors N / S / E / W: "
                    + " / ".join(
                        (
                            "clear within 2 cells"
                            if v == 0
                            else f"blocked at {int(v*2)} cell(s)"
                        )
                        for v in snap["scan"]["distances"]
                    )
                )
        st.caption(
            f"Recorded outcome: {metrics['outcome'].title()} · Total reward and Waits describe the full journey. Playback never trains or makes new decisions."
        )
        with st.expander("Journey details & replay download"):
            st.caption(
                "Controller: " + str(record.get("policy_id", "recorded controller"))
            )
            st.caption(
                f"Moving revisits: {metrics['repeated_movement_visits']} · Two-cell loops: {metrics['two_cell_oscillation_windows']} · Recoveries: {metrics['recoveries']} / {metrics['recovery_opportunities']} observed opportunities."
            )
            st.caption(
                "The controller uses public past decisions, never blockage identity or clearing time. A blocked attempted move ends in collision."
            )
            path = Path(st.session_state.journey_file)
            st.download_button(
                "Download this replay",
                path.read_bytes(),
                file_name=path.name,
                mime="application/gzip",
            )

    player()
