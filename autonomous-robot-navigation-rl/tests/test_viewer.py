"""Streamlit wiring checks using its in-process app harness; no browser account."""
from streamlit.testing.v1 import AppTest

from robot_navigation.core import ROOT


def test_viewer_controls_and_failure_replay():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=10).run()
    assert not app.exception
    assert app.session_state["playback"].index == 0
    app.button[2].click().run()  # next
    assert not app.exception
    assert app.session_state["playback"].index == 1
    app.select_slider[0].set_value(4.).run()
    assert app.session_state["playback"].index == 1
    app.slider[0].set_value(2).run()
    assert app.session_state["playback"].index == 2
    app.button[0].click().run()  # play
    # Simulate a due fragment rerun with the same session, verifying the actual
    # animation branch advances and updates the visible counters together.
    app.session_state["last_tick"] = 0.
    app.run()
    assert app.session_state["playback"].index == 3
    assert app.metric[0].value == "3 / 200"
    app.button[1].click().run()  # pause
    assert not app.session_state["playback"].playing
    app.button[3].click().run()  # reset
    assert app.session_state["playback"].index == 0
    # Saved Q-learning failure: actual collision and synchronized attempted cell.
    next(box for box in app.selectbox if box.label == "Recorded episode").set_value(
        ROOT / "artifacts/runs/smoke-seed7/replays/q_learning_collision.json").run()
    assert not app.exception
    app.button[2].click().run()
    state = app.session_state["playback"].current
    assert state["status"] == "collision" and state["return"] == -5.01
    assert app.error


def test_custom_scenario_validation():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=10).run()
    app.radio[0].set_value("Custom demonstration").run()
    app.button[0].click().run()  # generate default BFS demo
    assert not app.exception
    assert app.session_state["playback"].episode["mode"] == "custom demonstration"
    assert app.session_state["playback"].episode["metrics"]["outcome"] == "success"
    app.number_input[0].set_value(1)
    app.number_input[1].set_value(3)  # wall
    app.button[0].click().run()
    assert app.error and "Invalid custom scenario" in app.error[0].value


def test_sarsa_run_replay_and_custom_policy():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=10).run()
    next(box for box in app.selectbox if box.label == "Saved run").set_value(
        ROOT / "artifacts/runs/m2-sarsa-seed7").run()
    assert not app.exception
    next(box for box in app.selectbox if box.label == "Recorded episode").set_value(
        ROOT / "artifacts/runs/m2-sarsa-seed7/replays/sarsa_success.json").run()
    player = app.session_state["playback"]
    assert player.episode["policy_id"].startswith("sarsa/")
    app.button[0].click().run()
    app.session_state["last_tick"] = 0.
    app.run()
    assert app.session_state["playback"].index == 1
    app.radio[0].set_value("Custom demonstration").run()
    next(box for box in app.selectbox if box.label == "Frozen policy").set_value("SARSA")
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["playback"].episode["policy_id"].startswith("sarsa/")
