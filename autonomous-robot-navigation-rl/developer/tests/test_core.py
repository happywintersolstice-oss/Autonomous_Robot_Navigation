from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from robot_navigation.core import GridMap, NavigationEnv, Task, load_task
from robot_navigation.agents import BFSPolicy, QLearning, RandomPolicy
from robot_navigation.episodes import Playback, aggregate, frame, rollout, validate_replay


@pytest.fixture
def fixture_task():
    return Task(GridMap("fixture", ("...", ".#.", "...")), {**load_task().config, "max_steps": 4})


@pytest.fixture
def scenario():
    return {"id": "fixture", "start": [0, 0], "goal": [0, 2], "shortest_length": 2}


def test_checker_and_seeded_reset(fixture_task, scenario):
    other = {"id": "other", "start": [2, 2], "goal": [0, 0], "shortest_length": 4}
    env = NavigationEnv(fixture_task, [scenario, other])
    check_env(env, skip_render_check=True)
    first = env.reset(seed=17)[0]
    second = env.reset(seed=17)[0]
    assert np.array_equal(first, second)
    assert env.observation_space.contains(first)


@pytest.mark.parametrize("action,attempt", [(0, [-1, 0]), (2, [0, -1])])
def test_boundary_collision(fixture_task, scenario, action, attempt):
    env = NavigationEnv(fixture_task, [scenario])
    env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.tolist() == [0, 0, 0, 2]
    assert reward == -5.01 and terminated and not truncated
    assert info["attempted_position"] == attempt and info["outcome"] == "collision"
    with pytest.raises(RuntimeError):
        env.step(3)


def test_obstacle_collision(fixture_task, scenario):
    env = NavigationEnv(fixture_task, [scenario])
    env.reset()
    env.step(1)
    obs, reward, term, trunc, info = env.step(3)
    assert obs[:2].tolist() == [1, 0] and reward == -5.01
    assert term and not trunc and info["attempted_position"] == [1, 1]


def test_timeout_and_terminal_priority_on_200(fixture_task, scenario):
    task = replace(fixture_task, config={**fixture_task.config, "max_steps": 200})
    env = NavigationEnv(task, [scenario])
    env.reset()
    for i in range(200):
        obs, reward, term, trunc, info = env.step(3 if i % 2 == 0 else 2)
    assert not term and trunc and reward == -.01 and info["outcome"] == "timeout"
    for final_action, outcome, reward in ((3, "success", 9.99), (1, "collision", -5.01)):
        env.reset()
        for i in range(199):
            env.step(3 if i % 2 == 0 else 2)
        _, actual, term, trunc, info = env.step(final_action)
        assert term and not trunc and info["outcome"] == outcome and actual == reward


def test_invalid_scenarios_and_actions(fixture_task, scenario):
    for start, goal in (([0, 0], [0, 0]), ([1, 1], [0, 0]), ([-1, 0], [0, 0]), ([0.5, 0], [0, 0])):
        with pytest.raises(ValueError):
            NavigationEnv(fixture_task, [{"id": "bad", "start": start, "goal": goal}])
    env = NavigationEnv(fixture_task, [scenario])
    env.reset()
    with pytest.raises(ValueError):
        env.step(4)


def test_bfs_hand_solvable_and_unreachable(fixture_task):
    assert len(fixture_task.grid.path((0, 0), (2, 2))) - 1 == 4
    assert fixture_task.grid.distances((0, 0))[(0, 2)] == 2
    with pytest.raises(ValueError, match="unreachable"):
        GridMap("disconnected", (".#.", "###", ".#.")).path((0, 0), (2, 2))
    for rows in (("..", "."), (".x", ".."), ("##", "##")):
        with pytest.raises(ValueError):
            GridMap("bad", rows)


def test_q_hand_calculated_terminal_and_truncation(fixture_task):
    agent = QLearning(fixture_task)
    obs, next_obs = np.array([0, 0, 0, 2]), np.array([0, 1, 0, 2])
    agent.q[tuple(next_obs)] = [2, 4, 3, 1]
    agent.update(obs, 3, 1, next_obs, False, .5, .9)
    assert agent.q[(*obs, 3)] == pytest.approx(2.3)
    agent.q[(*obs, 3)] = 0
    agent.update(obs, 3, 1, next_obs, True, .5, .9)
    assert agent.q[(*obs, 3)] == .5
    # Passing terminated=False on a timeout must bootstrap the next observation.
    agent.q[(*obs, 3)] = 0
    agent.update(obs, 3, -.01, next_obs, False, 1, .9)
    assert agent.q[(*obs, 3)] == pytest.approx(3.59)


def test_checkpoint_and_read_only_evaluation(tmp_path, fixture_task, scenario):
    from robot_navigation.workflow import evaluate
    agent = QLearning(fixture_task)
    agent.q[0, 0, 0, 2, 3] = 2
    path = tmp_path / "agent.npz"
    agent.save(path, {"training_interactions": 1})
    loaded = QLearning.load(path, fixture_task)
    assert np.array_equal(loaded.q, agent.q) and loaded.act([0, 0, 0, 2]) == 3
    before = loaded.q.copy()
    evaluate(fixture_task, [scenario], loaded, 7, "fixture")
    assert np.array_equal(before, loaded.q)
    mismatch = replace(fixture_task, grid=GridMap("other-map", fixture_task.grid.rows))
    with pytest.raises(ValueError, match="mismatch"):
        QLearning.load(path, mismatch)
    changed_reward = replace(fixture_task, config={**fixture_task.config, "step_reward": -.02})
    with pytest.raises(ValueError, match="mismatch"):
        QLearning.load(path, changed_reward)
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        QLearning.load(path, fixture_task)


def test_metrics_replay_and_playback(fixture_task, scenario):
    env = NavigationEnv(fixture_task, [scenario])
    success = rollout(env, BFSPolicy(fixture_task.grid), scenario)
    assert success["metrics"] == {"outcome": "success", "steps": 2, "return": 9.98,
                                  "successful_route_length": 2, "spl": 1.0}
    validate_replay(fixture_task, success)
    failed = rollout(env, QLearning(fixture_task), scenario)
    result = aggregate([success["metrics"], failed["metrics"]])
    assert result["success_rate"] == result["collision_rate"] == .5 and result["timeout_rate"] == 0
    assert result["spl"] == .5 and result["successful_route_count"] == 1
    assert aggregate([failed["metrics"]])["mean_successful_route_length"] is None
    player = Playback(success)
    player.play()
    player.tick()
    assert player.current["position"] == [0, 1] and player.current["return"] == -.01
    assert player.current["trail"] == [[0, 0], [0, 1]]  # no future route
    player.pause()
    player.tick()
    assert player.index == 1
    player.play()
    player.tick()
    assert player.index == 2 and not player.playing and player.current["status"] == "success"
    player.seek(1)
    player.next()
    assert player.current["return"] == pytest.approx(9.98)
    player.reset()
    assert player.index == 0 and player.current["return"] == 0 and not player.playing
    collision_frame = frame(failed, 1)
    assert collision_frame["position"] == [0, 0] and collision_frame["attempted_position"] == [-1, 0]
    success["transitions"][0]["reward"] = 2
    with pytest.raises(ValueError, match="transition mismatch"):
        validate_replay(fixture_task, success)


def test_nonoptimal_and_timeout_metrics(fixture_task, scenario):
    class Scripted:
        policy_id = "fixture-script"
        def __init__(self, actions):
            self.actions = iter(actions)
        def act(self, observation):
            return next(self.actions)
    env = NavigationEnv(fixture_task, [scenario])
    detour = rollout(env, Scripted([3, 2, 3, 3]), scenario)
    assert detour["metrics"]["spl"] == .5 and detour["metrics"]["steps"] == 4
    timeout = rollout(env, Scripted([3, 2, 3, 2]), scenario)
    assert timeout["metrics"]["spl"] == 0 and timeout["metrics"]["return"] == -.04
    assert aggregate([timeout["metrics"]])["timeout_rate"] == 1


def test_random_reproducible(fixture_task, scenario):
    env = NavigationEnv(fixture_task, [scenario])
    assert rollout(env, RandomPolicy(4), scenario) == rollout(env, RandomPolicy(4), scenario)


def test_learns_hand_solvable_fixture(fixture_task, scenario):
    policy = QLearning(fixture_task)
    env = NavigationEnv(fixture_task, [scenario])
    rng = np.random.default_rng(7)
    observation, _ = env.reset(seed=7)
    discoveries = 0
    for _ in range(500):
        action = policy.explore(observation, .4, rng)
        next_obs, reward, terminated, truncated, info = env.step(action)
        policy.update(observation, action, reward, next_obs, terminated, .3, .99)
        discoveries += info["outcome"] == "success"
        observation = next_obs
        if terminated or truncated:
            observation, _ = env.reset()
    episode = rollout(env, policy, scenario)
    assert discoveries > 0 and episode["metrics"]["outcome"] == "success"
    assert episode["metrics"]["steps"] == 2
