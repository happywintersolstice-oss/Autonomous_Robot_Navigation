"""Explicit training controls and saved checkpoint inspection."""

from pathlib import Path
import streamlit as st
from ..app_registry import registry
from ..app_service import start_job, job_status
from .components import LABELS, page_header


def training():
    page_header(
        "Model workshop",
        "Train Models",
        "Inspect saved models or explicitly start a new training job.",
        "FOUR RL METHODS",
    )
    preset = st.radio("Training preset", ["quick", "standard"], horizontal=True)
    feature_choice = st.selectbox(
        "Learning representation",
        ["Phase 3 public history", "Phase 2 local goal features"],
    )
    features = "product-v3" if feature_choice.startswith("Phase 3") else "product-v2"
    chosen = st.multiselect(
        "RL algorithms",
        list(LABELS),
        default=list(LABELS),
        format_func=lambda v: LABELS[v],
    )
    actions = st.number_input(
        "Actions per model",
        min_value=32,
        max_value=50000,
        value=256 if preset == "quick" else 10000,
        step=32 if preset == "quick" else 200,
    )
    shaping = st.selectbox(
        "Training feedback",
        (
            ["none", "liveness-v1"]
            if features == "product-v3"
            else ["none", "progress-v1"]
        ),
        help="Feedback is train-only. Phase3 liveness uses public distance progress, recent revisits and repeated Waits with an observed unvisited adjacent alternative. It knows no obstacle identity or clear time. Official evaluation reward is unchanged.",
    )
    st.caption(
        "Quick is a small starter. Phase3 uses 8 past public decision positions, the last action and a Wait counter: 409 public inputs, 431 internal state values and 520 derived features. Phase2 uses its original 480 features. Q-learning uses stabilized targets. Training feedback and all measured failures are recorded."
    )
    seed = st.number_input(
        "Training seed", min_value=0, max_value=2**31 - 1, value=74001
    )
    st.code(
        f"python -m robot_navigation.app_training --preset {preset} --models "
        + " ".join(chosen)
        + f" --actions {actions} --seed {seed} --shaping {shaping} --features {features}",
        language="powershell",
    )
    if preset == "standard":
        st.warning(
            "Standard is user-initiated development training. It may take minutes or longer on CPU; closing the browser does not stop it. No archived data is used."
        )
    status = job_status(st.session_state.job) if st.session_state.get("job") else None
    running = status and status["status"] in ("starting", "running")
    if st.button(
        "Start training", type="primary", disabled=bool(running) or not chosen
    ):
        from ..app_training import model_settings

        cfg = model_settings(preset)
        if (
            "ppo" in chosen
            and actions % cfg["n_steps"]
            or ("dqn" in chosen and actions % 4)
        ):
            st.error(
                "Use an exact PPO rollout multiple (32 quick / 200 standard), and a multiple of 4 for DQN."
            )
        else:
            try:
                st.session_state.job = start_job(
                    preset, chosen, int(actions), int(seed), shaping, features
                )
            except OSError as exc:
                st.error("Unable to start training: " + str(exc))
            else:
                st.rerun()
    if status:
        st.progress(
            float(status.get("fraction", 0)),
            text=f"{status['status'].title()} · {status.get('algorithm', '')} · {status.get('completed_actions', 0)} / {status.get('total_actions', '?')} actions",
        )
        if status.get("error"):
            st.error(status["error"])
        a, b = st.columns(2)
        if a.button("Refresh training status"):
            st.rerun()
        if b.button("Cancel training", disabled=not running):
            (Path(st.session_state.job) / "cancel.request").write_text("cancel")
            st.info("Cancellation requested. The worker preserves the partial run.")
        if status["status"] == "complete":
            st.success(
                "Checkpoints are ready in Simulator and Compare & Export. These are inference checkpoints, not exact-resume snapshots."
            )
    items = registry()
    st.subheader("Available native checkpoints")
    st.dataframe(
        [
            {
                "Algorithm": LABELS[i["algorithm"]],
                "Status": i["selection_scope"],
                "Features": i["feature_version"],
                "Agent state": i["agent_state_values"],
                "Actions": i["actions"],
                "Seed": i["seed"],
                "Run": i["id"],
            }
            for i in items
        ],
        width="stretch",
    )
    if items:
        target = st.selectbox(
            "Checkpoint integrity check",
            [i["label"] for i in items],
            key="integrity_checkpoint",
        )
        if st.button("Verify selected checkpoint"):
            from ..app_registry import load_model

            try:
                item = next((i for i in items if i["label"] == target))
                load_model(item)
                st.success(
                    "Task, feature/history version, five actions and SHA-256 verified: "
                    + item["checkpoint_sha256"]
                )
            except ValueError as exc:
                st.error(str(exc))
