"""Shared presentation elements, with no policy or environment execution."""

from html import escape
from pathlib import Path

import streamlit as st

LABELS = {"q-learning": "Q-learning", "sarsa": "SARSA", "dqn": "DQN", "ppo": "PPO"}
PAGES = ["Robot Simulator", "Train Models", "Compare & Export", "Guide", "Archives"]


def apply_theme():
    css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def pause_playback():
    st.session_state.playing = False


def navigation():
    st.markdown(
        '<div class="warehouse-eyebrow">A COMPARATIVE STUDY OF Q-LEARNING, SARSA, DQN AND PPO</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="warehouse_navigation"):
        return st.radio(
            "Workspace",
            PAGES,
            horizontal=True,
            key="workspace_page",
            format_func=lambda page: "Simulator" if page == "Robot Simulator" else page,
            label_visibility="collapsed",
            on_change=pause_playback,
            width="stretch",
        )


def page_header(section, title, description, badge):
    st.markdown(
        f"""<header class="warehouse-header">
        <div><div class="warehouse-breadcrumb">Warehouse navigation / {escape(section)}</div>
        <h1>{escape(title)}</h1><p>{escape(description)}</p></div>
        <div class="warehouse-badge"><span class="status-dot"></span>{escape(badge)}</div>
        </header>""",
        unsafe_allow_html=True,
    )


def panel_title(title, detail="", badge=""):
    st.markdown(
        f'<div class="panel-heading"><div><strong>{escape(title)}</strong><small>{escape(detail)}</small></div><span class="panel-badge">{escape(badge)}</span></div>',
        unsafe_allow_html=True,
    )


def friendly_checkpoint(item):
    if item.get("arm") == "public-vote-v1":
        return "Four-model vote · optional untrained ensemble"
    label = f"{LABELS[item['algorithm']]} · {item['selection_scope']}"
    if not item.get("phase3_selected") and not item.get("phase2_baseline"):
        label += f" · {item['seed']} / {item['id'].split('/')[0][-8:]}"
    return label


def status_card(outcome, index, length):
    title = {
        "success": "Goal reached",
        "timeout": "Timed out",
        "collision": "Collision recorded",
    }[outcome]
    state = (
        "Ready to play"
        if index == 0
        else "At the final frame" if index == length else f"Frame {index} of {length}"
    )
    st.markdown(
        f'<div class="journey-status {escape(outcome)}"><small>RECORDED RESULT</small><strong>{title}</strong><span>{state}</span></div>',
        unsafe_allow_html=True,
    )
