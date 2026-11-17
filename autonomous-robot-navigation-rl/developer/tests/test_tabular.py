from dataclasses import replace
import gzip
import json

import numpy as np
import pytest

from robot_navigation.agents import QLearning, SARSA, load_policy
from robot_navigation.core import GridMap, Task, load_task
from robot_navigation.training import epsilon_at, train_loop, validate_config
from robot_navigation.workflow import evaluate


def fixture():
    task = Task(GridMap("line-fixture", ("...",)), {**load_task().config, "max_steps": 2})
    scenario = {"id": "line", "start": [0, 0], "goal": [0, 2], "shortest_length": 2}
    config = dict(seed=7, interactions=500, learning_rate=.3, discount=.99,
                  epsilon_start=.4, epsilon_end=.4, epsilon_decay_fraction=1., monitor_every=100,
                  algorithm="sarsa", run_id="fixture")
    return task, scenario, config


def test_sarsa_uses_behavior_action_and_terminal_target():
    task, _, _ = fixture()
    agent = SARSA(task)
    obs, nxt = [0, 0, 0, 2], [0, 1, 0, 2]
    agent.q[tuple(nxt)] = [2, 4, 3, 1]
    agent.update(obs, 3, 1, nxt, False, .5, .9, next_action=3)
    assert agent.q[(*obs, 3)] == pytest.approx(.95)  # not max-Q target 2.3
    agent.q[(*obs, 3)] = 0
    agent.update(obs, 3, -5.01, nxt, True, 1., .9)
    assert agent.q[(*obs, 3)] == -5.01
    with pytest.raises(ValueError):
        agent.update(obs, 3, -.01, nxt, False, 1., .9)


def test_sarsa_truncation_action_and_reset_boundary():
    task, scenario, config = fixture()
    config["interactions"] = 3
    class TracedSARSA(SARSA):
        def __init__(self):
            super().__init__(task)
            self.choices, self.observations, self.updates = iter([3, 2, 0, 3, 3]), [], []
        def explore(self, observation, epsilon, rng):
            self.observations.append(observation.tolist())
            return next(self.choices)
        def update(self, observation, action, reward, next_observation, terminated, alpha, gamma, next_action=None):
            self.updates.append((action, next_action, terminated))
            super().update(observation, action, reward, next_observation, terminated, alpha, gamma, next_action)
    policy, episodes = TracedSARSA(), []
    visits, report = train_loop(task, [scenario], policy, config, on_episode=episodes.append)
    assert policy.updates == [(3, 2, False), (2, 0, False), (3, 3, False)]
    assert [obs[1] for obs in policy.observations] == [0, 1, 0, 0, 1]
    assert episodes[0]["outcome"] == "timeout" and episodes[0]["steps"] == 2
    assert report["incomplete_budget_cutoff_episode"]["steps"] == 1
    assert visits.sum() == 3 and sum(report["sampled_training_goals"].values()) == 2


@pytest.mark.parametrize("cls", [QLearning, SARSA])
def test_training_repeats_and_monitoring_does_not_change_learning(cls):
    task, scenario, config = fixture()
    first, second = cls(task), cls(task)
    visits1, report1 = train_loop(task, [scenario], first, config)
    def monitor(interactions, visitation):
        before = second.q.copy()
        evaluate(task, [scenario], second, 91, "fixture monitoring")
        assert np.array_equal(before, second.q)
    visits2, report2 = train_loop(task, [scenario], second, config, monitor)
    assert np.array_equal(first.q, second.q) and np.array_equal(visits1, visits2)
    assert report1["successful_training_episodes"] == report2["successful_training_episodes"] > 0
    assert first.act([0, 0, 0, 2]) == first.act([0, 1, 0, 2]) == 3


def test_checkpoint_dispatch_and_single_pass_export(tmp_path):
    task, scenario, config = fixture()
    agent = SARSA(task)
    train_loop(task, [scenario], agent, config)
    path = tmp_path / "sarsa.npz"
    agent.save(path, {"training_interactions": 500})
    loaded = load_policy(path, task)
    assert isinstance(loaded, SARSA) and np.array_equal(agent.q, loaded.q)
    with pytest.raises(ValueError, match="algorithm mismatch"):
        QLearning.load(path, task)
    output = tmp_path / "episodes.jsonl.gz"
    summary, metrics, _ = evaluate(task, [scenario], loaded, 7, "fixture", trajectory_path=output)
    with gzip.open(output, "rt", encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    assert len(records) == 1 and summary["interactions"] == 2
    assert records[0]["metrics"]["return"] == metrics[0]["return"]
    assert np.array_equal(loaded.q, agent.q)
    with pytest.raises(FileExistsError):
        evaluate(task, [scenario], loaded, 7, "fixture", trajectory_path=output)


def test_parameters_and_epsilon_schedule():
    _, _, config = fixture()
    validate_config(config)
    config.update(epsilon_start=1., epsilon_end=.1, epsilon_decay_fraction=.8)
    assert epsilon_at(config, 0) == 1.
    assert epsilon_at(config, 400) == pytest.approx(.1)
    for key, bad in (("algorithm", "ppo"), ("seed", -1), ("run_id", "../bad"), ("interactions", 0)):
        with pytest.raises(ValueError):
            validate_config({**config, key: bad})
