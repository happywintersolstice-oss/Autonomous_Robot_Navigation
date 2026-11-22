"""Thin saved-result/demo helpers; viewing never trains or reruns held-out cases."""

from pathlib import Path
import gzip, json
from .core import read_json, ROOT
from .app_registry import RUNS, registry
from .app_service import identity, saved_comparison
from .phase3_evaluation import load_saved, export_comparison
from .phase3_data import train_dev

SCOPES = ("Phase 3 product held-out", "Phase 3 development", "Phase 2 development")


def saved(scope):
    """Read an existing comparison; this function never executes a policy."""
    if scope not in SCOPES:
        raise ValueError("Choose a registered development or held-out scope")
    if scope == "Phase 2 development":
        return saved_comparison()
    pub = (
        read_json(RUNS / "phase3-publication.json")
        if (RUNS / "phase3-publication.json").is_file()
        else {}
    )
    if scope == SCOPES[0]:
        relative = pub.get("heldout_comparison")
    else:
        selected = (
            read_json(RUNS / "phase3-selection.json")
            if (RUNS / "phase3-selection.json").is_file()
            else {}
        )
        relative = pub.get("development_comparison") or selected.get(
            "development_comparison"
        )
    if not relative:
        return None
    folder = (RUNS / relative).resolve()
    if not folder.is_relative_to(RUNS.resolve()):
        raise ValueError("Unsafe saved result path")
    return load_saved(folder)


def development_compare(selected, include_random=False):
    """Explicitly evaluate the development panel and preserve prior exports."""
    if set(selected) != {"q-learning", "sarsa", "dqn", "ppo"}:
        raise ValueError("Select all four genuine RL families")
    if any(item.get("algorithm") != algorithm for algorithm, item in selected.items()):
        raise ValueError("Each comparison slot must contain its matching RL algorithm")
    base = read_json(RUNS / "selection.json")["selected"]
    available = {i["id"]: i for i in registry()}
    choices = [
        ("Phase3 " + a, selected[a], "RL user-selected development")
        for a in ("q-learning", "sarsa", "dqn", "ppo")
    ]
    choices += [
        ("Phase2 " + a, available[record["id"]], "RL Phase2 baseline")
        for a, record in base.items()
    ]
    choices += [
        ("Observed-Map Replanner", "Observed-Map Replanner", "Non-RL reference")
    ]
    if include_random:
        choices.append(("Random reference", "Random reference", "Non-RL reference"))
    return export_comparison(
        choices,
        train_dev()["development"],
        RUNS / "phase3-user-comparisons" / identity(),
        "development",
        {"explicit_user_rerun": True, "heldout_evaluations": 0},
    )


def open_replay(path):
    """Open saved frames in Simulator without stepping an environment."""
    import streamlit as st

    path = Path(path).resolve()
    if not path.is_relative_to(RUNS.resolve()):
        raise ValueError("Replay outside new product output")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        record = json.load(f)
    st.session_state.journey = record
    st.session_state.journey_file = str(path)
    st.session_state.frame_index = 0
    st.session_state.playing = False
    st.session_state.workspace_page = "Robot Simulator"
