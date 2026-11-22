"""Single product entry point. Presentation lives in the ui package."""

import streamlit as st
from .ui.components import apply_theme, navigation
from .ui.simulator import simulator, grid_svg, scenario
from .ui.training import training
from .ui.comparison import comparison
from .ui.reference import guide, archives


def render():
    st.set_page_config(
        page_title="Autonomous Warehouse Robot Navigation",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_theme()
    page = navigation()
    {
        "Robot Simulator": simulator,
        "Train Models": training,
        "Compare & Export": comparison,
        "Guide": guide,
        "Archives": archives,
    }[page]()
