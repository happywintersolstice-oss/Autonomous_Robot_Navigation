"""Streamlit wiring checks using its in-process app harness; no browser account."""
from streamlit.testing.v1 import AppTest

from robot_navigation.core import ROOT


def test_viewer_controls_and_failure_replay():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=10).run()
    assert not app.exception
    assert next(box for box in app.selectbox if box.label == 'Trained model').value.name == 'm4-tuned-q-learning-seed101'
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
    next(box for box in app.selectbox if box.label == "Trained model").set_value(
        ROOT / "artifacts/runs/smoke-seed7").run()
    next(box for box in app.selectbox if box.label == "Recorded episode").set_value(
        ROOT / "artifacts/runs/smoke-seed7/replays/q_learning_collision.json").run()
    assert not app.exception
    app.button[2].click().run()
    state = app.session_state["playback"].current
    assert state["status"] == "collision" and state["return"] == -5.01
    assert app.error


def test_custom_scenario_validation():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=10).run()
    app.session_state['replay_tab']='Try a route'; app.run()
    next(w for w in app.selectbox if w.label=='Frozen policy').set_value('BFS')
    next(button for button in app.button if button.label == 'Generate demonstration').click().run()
    assert not app.exception
    assert app.session_state["playback"].episode["mode"] == "custom demonstration"
    assert app.session_state["playback"].episode["metrics"]["outcome"] == "success"
    next(box for box in app.number_input if box.label == 'Start row').set_value(1)
    next(box for box in app.number_input if box.label == 'Start column').set_value(3)  # wall
    next(button for button in app.button if button.label == 'Generate demonstration').click().run()
    assert app.error and "Invalid custom scenario" in app.error[0].value


def test_sarsa_run_replay_and_custom_policy():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=10).run()
    next(box for box in app.selectbox if box.label == "Trained model").set_value(
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
    app.session_state['replay_tab']='Try a route'; app.run()
    next(box for box in app.selectbox if box.label == "Frozen policy").set_value("SARSA")
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["playback"].episode["policy_id"].startswith("sarsa/")


def test_dqn_selected_checkpoint_replay_and_custom_policy():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=15).run()
    next(box for box in app.selectbox if box.label == "Trained model").set_value(
        ROOT / "artifacts/runs/m3-dqn-pilot-seed7").run()
    assert not app.exception
    next(box for box in app.selectbox if box.label == "Recorded episode").set_value(
        ROOT / "artifacts/runs/m3-dqn-pilot-seed7/replays/dqn_success.json").run()
    assert app.session_state["playback"].episode["policy_id"].startswith("dqn/")

    app.button[2].click().run()
    assert not app.exception and app.session_state["playback"].index == 1
    app.session_state['replay_tab']='Try a route'; app.run()
    next(box for box in app.selectbox if box.label == "Frozen policy").set_value("DQN")
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["playback"].episode["policy_id"].startswith("dqn/")


def test_terminal_reward_variant_replay_is_labeled_and_synchronized():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=15).run()
    next(box for box in app.selectbox if box.label == "Trained model").set_value(
        ROOT / "artifacts/runs/m4-terminal-q-learning-seed101").run()
    assert not app.exception
    next(box for box in app.selectbox if box.label == "Recorded episode").set_value(
        ROOT / "artifacts/runs/m4-terminal-q-learning-seed101/replays/q_learning_success.json").run()
    episode = app.session_state["playback"].episode
    assert episode["signature"]["reward_version"] == "terminal-only-v1"
    assert any("terminal-only-v1" in caption.value for caption in app.caption)
    app.button[2].click().run()
    current = app.session_state["playback"].current
    assert not app.exception and current["return"] == episode["transitions"][0]["reward"]
    assert current["position"] == episode["transitions"][0]["next_observation"][:2]


def test_voting_replay_and_frozen_custom_demo():
    from robot_navigation.core import read_json
    study = ROOT / "artifacts/studies/part-b-v1"
    directory = ROOT / "artifacts/runs" / read_json(study / "manifest.json")["voting_run"]
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=15).run()
    next(box for box in app.selectbox if box.label == "Trained model").set_value(directory).run()
    assert not app.exception
    next(box for box in app.selectbox if box.label == "Recorded episode").set_value(
        directory / "replays/voting_success.json").run()
    episode = app.session_state["playback"].episode
    assert episode["policy_id"].startswith("voting/")
    assert len(episode["transitions"][0]["decision"]["member_actions"]) == 3
    assert episode["transitions"][0]["scores"] is None
    app.session_state['replay_tab']='Try a route'; app.run()
    next(box for box in app.selectbox if box.label == "Frozen policy").set_value("Policy voting")
    app.button[0].click().run()
    assert not app.exception and app.session_state["playback"].episode["policy_id"].startswith("voting/")


def test_new_run_inventory_preserves_selected_episode(tmp_path, monkeypatch):
    import shutil
    import robot_navigation.core as core
    source = ROOT / "artifacts/runs/smoke-seed7"
    runs = tmp_path / "artifacts/runs"
    for folder in ("configs", "data"):
        shutil.copytree(ROOT / folder, tmp_path / folder)
    def copy_run(name):
        directory = runs / name
        directory.mkdir(parents=True)
        for filename in ("manifest.json", "configuration.json", "checkpoint.json", "checkpoint.npz", "pilot.json", "validation.json"):
            shutil.copy2(source / filename, directory / filename)
        shutil.copytree(source / "replays", directory / "replays")
        return directory
    copy_run("smoke-seed7")
    selected = copy_run("zz-selected-run")
    monkeypatch.setattr(core, "ROOT", tmp_path)
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=15).run()
    next(box for box in app.selectbox if box.label == "Trained model").set_value(selected).run()
    next(button for button in app.button if button.label == "Next step").click().run()
    assert app.session_state["playback"].index == 1
    identity = app.session_state["episode_identity"]
    copy_run("aa-new-run")
    app.run()
    assert not app.exception
    assert next(box for box in app.selectbox if box.label == "Trained model").value == selected
    assert app.session_state["episode_identity"] == identity and app.session_state["playback"].index == 1
