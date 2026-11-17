from dataclasses import replace
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
from stable_baselines3.common.env_checker import check_env

from robot_navigation.agents import load_policy
from robot_navigation.core import GridMap, NavigationEnv, Task, load_task, read_json, write_json
from robot_navigation.dqn import CoordinateObservation, DQNPolicy, build_model, normalize
from robot_navigation.part_a import checkpoint_rank
from robot_navigation.workflow import evaluate


def fixture():
    task = Task(GridMap("dqn-fixture", ("...",)), {**load_task().config, "max_steps": 1})
    scenario = {"id": "fixture", "start": [0, 0], "goal": [0, 2], "shortest_length": 2}
    config = dict(seed=7, learning_rate=.001, discount=.99, buffer_size=1000,
        learning_starts=20, batch_size=16, train_freq=1, gradient_steps=1,
        target_update_interval=20, epsilon_start=1., epsilon_end=.1,
        epsilon_decay_fraction=.8, net_arch=[16, 16], torch_threads=1)
    return task, scenario, config


def test_normalization_checker_and_transition_parity():
    task, scenario, _ = fixture()
    raw = NavigationEnv(task, [scenario])
    wrapped = CoordinateObservation(NavigationEnv(task, [scenario]))
    check_env(wrapped, warn=True)
    raw_obs, _ = raw.reset(seed=7)
    obs, _ = wrapped.reset(seed=7)
    assert np.array_equal(obs, normalize(raw_obs, task.grid.shape))
    actual, *flags = raw.step(3)
    normalized, *wrapped_flags = wrapped.step(3)
    assert np.array_equal(normalized, normalize(actual, task.grid.shape))
    assert flags == wrapped_flags and normalized.dtype == np.float32


@pytest.mark.parametrize("goal,expected_done,expected_timeout", [([0, 2], 0., 1.), ([0, 1], 1., 0.)])
def test_sb3_stores_final_observation_and_bootstraps_timeouts(goal, expected_done, expected_timeout):
    task, scenario, config = fixture()
    scenario = {**scenario, "goal": goal, "shortest_length": goal[1]}
    model = build_model(NavigationEnv(task, [scenario]), config)
    # Force a single known RIGHT action through SB3's real vector/autoreset/storage path.
    model._sample_action = lambda *args: (np.array([3]), np.array([3]))
    model.learn(total_timesteps=1)
    buffer = model.replay_buffer
    assert buffer.timeouts[0, 0] == expected_timeout
    assert buffer.sample(1).dones.item() == expected_done
    assert np.array_equal(buffer.next_observations[0, 0], normalize([0, 1, *goal], task.grid.shape))
    assert buffer.next_observations[0, 0, 1] == .5  # final state, not reset column 0


def test_dqn_learns_fixture_inference_is_read_only_and_save_load(tmp_path):
    task, scenario, config = fixture()
    # One-step adjacent goal; larger fixture above checks the truncation path.
    scenario = {**scenario, "goal": [0, 1], "shortest_length": 1}
    model = build_model(NavigationEnv(task, [scenario]), config)
    model.learn(total_timesteps=1000)
    agent = DQNPolicy(task, model, "dqn-fixture")
    assert agent.act([0, 0, 0, 1]) == 3
    assert int(model.predict(normalize([0, 0, 0, 1], task.grid.shape), deterministic=True)[0]) == 3
    before = agent.state_fingerprint()
    numpy_rng, torch_rng, python_rng = np.random.get_state(), torch.get_rng_state(), random.getstate()
    result, _, _ = evaluate(task, [scenario], agent, 7, "fixture")
    assert result["success_rate"] == 1 and agent.state_fingerprint() == before
    after_rng = np.random.get_state()
    assert numpy_rng[0] == after_rng[0] and np.array_equal(numpy_rng[1], after_rng[1]) and numpy_rng[2:] == after_rng[2:]
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()
    path = tmp_path / "agent.zip"
    agent.save(path, {"training_interactions": 1000})
    loaded = load_policy(path, task)
    assert loaded.algorithm == "dqn" and np.array_equal(agent.scores([0, 0, 0, 1]), loaded.scores([0, 0, 0, 1]))
    with pytest.raises(ValueError, match="mismatch"):
        load_policy(path, replace(task, config={**task.config, "step_reward": -.1}))
    metadata = read_json(path.with_suffix(".json"))
    write_json(path.with_suffix(".json"), {**metadata, "encoding": "wrong"})
    with pytest.raises(ValueError, match="encoding mismatch"):
        load_policy(path, task)
    write_json(path.with_suffix(".json"), metadata)
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_policy(path, task)


def test_checkpoint_selection_is_success_then_spl_then_earliest():
    assert checkpoint_rank({"success_rate": .5, "spl": .3}, 10000) > checkpoint_rank({"success_rate": .4, "spl": .4}, 20000)
    assert checkpoint_rank({"success_rate": .5, "spl": .4}, 10000) > checkpoint_rank({"success_rate": .5, "spl": .3}, 10000)
    assert checkpoint_rank({"success_rate": .5, "spl": .4}, 10000) > checkpoint_rank({"success_rate": .5, "spl": .4}, 20000)
