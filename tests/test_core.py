"""Fast checks of the parts every result depends on.  Run:  python -m pytest -q"""

import numpy as np
import pandas as pd
import pytest

from medinav.agents import AStarAgent, TabularAgent
from medinav.agents.tabular import Discretiser
from medinav.env import HospitalNavEnv, generate_layout, make_env, map_seed
from medinav.env.planning import astar, geodesic_field, inflate
from medinav.env.wrappers import ImputeLidar
from medinav.evaluation.metrics import iqm, permutation_test, stratified_bootstrap
from medinav.evaluation.rollout import evaluate, run_episode


def test_layout_is_deterministic_and_reachable():
    a, b = generate_layout(123, 2), generate_layout(123, 2)
    assert (a.grid == b.grid).all() and np.allclose(a.start, b.start)
    assert np.isfinite(a.geodesic[a.cell(a.start)])


def test_splits_are_disjoint():
    assert map_seed("train", 499) < map_seed("validation", 0) < map_seed("test", 0)


def test_astar_matches_geodesic_length():
    grid = np.zeros((10, 10), bool)
    grid[5, 1:9] = True
    path = astar(grid, (0, 0), (9, 0))
    field = geodesic_field(grid, (9, 0))
    assert path[0] == (0, 0) and path[-1] == (9, 0)
    assert field[0, 0] == pytest.approx(sum(np.hypot(*np.subtract(p, q)) for p, q in zip(path, path[1:])))


def test_inflate_covers_diagonals():
    g = np.zeros((5, 5), bool)
    g[2, 2] = True
    assert inflate(g, 1)[1, 1] and inflate(g, 1)[3, 3]


@pytest.mark.parametrize("mode", ["discrete", "continuous"])
def test_env_contract(mode):
    env = HospitalNavEnv({"action_mode": mode, "dropout_prob": 0.3, "lidar_noise_std": 0.05})
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.all(np.abs(obs) <= 1.0)
        assert r == pytest.approx(sum(info["reward_terms"].values()))
        if term or trunc:
            break


def test_astar_baseline_solves_static_maps():
    env = make_env({"difficulty": 3, "n_humans": 0})
    rows = evaluate(AStarAgent(), env, "validation", 15)
    assert np.mean([r["success"] for r in rows]) == 1.0


def test_same_map_seed_same_episode():
    env = make_env({"difficulty": 2})
    r1, _ = run_episode(AStarAgent(), env, map_seed("test", 3))
    r2, _ = run_episode(AStarAgent(), env, map_seed("test", 3))
    assert r1["steps"] == r2["steps"] and r1["outcome"] == r2["outcome"]


def test_imputation_removes_sentinel():
    for strategy in ("max_range", "zero", "last_valid", "neighbour"):
        env = ImputeLidar(HospitalNavEnv({"dropout_prob": 0.5}), strategy)
        obs, _ = env.reset(seed=1)
        for _ in range(5):
            obs, *_ = env.step(0)
        assert (obs[:24] >= 0).all()


def test_discretiser_range():
    d = Discretiser()
    env = HospitalNavEnv()
    obs, _ = env.reset(seed=0)
    assert 0 <= d(obs) < d.n_states


def test_tabular_learns_something():
    env = make_env({"difficulty": 0})
    agent = TabularAgent("q_learning").train(env, 150, seed=0)
    assert np.abs(agent.q).sum() > 0


def test_statistics():
    assert iqm([0, 1, 2, 3, 100]) == pytest.approx(2.0)
    df = pd.DataFrame({"seed": [0] * 50 + [1] * 50, "success": [1.0] * 100})
    assert stratified_bootstrap(df, "success") == (1.0, 1.0)
    assert permutation_test(np.ones(30), np.zeros(30), 500) < 0.01
