"""Interpretability: WHAT information does each agent actually use? (Part A step 3)

permutation_importance (model-agnostic, works for every agent incl. tabular)
    Break one group of inputs (e.g. the rear LiDAR rays) by replacing it with
    values taken from a random other moment, and measure how much success drops.
    Big drop = the agent relies on that input. Same idea as sklearn's
    permutation_importance, applied to an RL policy.

gradient_saliency (deep RL only)
    d(decision)/d(input) averaged over visited states: which individual LiDAR rays
    most change the steering command. Plotted as a polar "attention" chart.

tabular_policy_map
    The learned Q-table is small enough to read: for each goal direction and each
    front-obstacle level, which action does the agent choose?
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd

from ..env import make_env
from .rollout import evaluate


def feature_groups(n_rays=24, guidance=False) -> dict:
    ang = (np.degrees(np.linspace(0, 2 * np.pi, n_rays, endpoint=False)) + 180) % 360 - 180
    groups = {
        "lidar_front": np.flatnonzero(np.abs(ang) <= 45),
        "lidar_left": np.flatnonzero((ang > 45) & (ang < 135)),
        "lidar_back": np.flatnonzero(np.abs(ang) >= 135),
        "lidar_right": np.flatnonzero((ang < -45) & (ang > -135)),
        "goal_distance": np.array([n_rays]),
        "goal_bearing": np.array([n_rays + 1, n_rays + 2]),
        "own_velocity": np.array([n_rays + 3, n_rays + 4]),
    }
    if guidance:
        groups["astar_waypoint"] = np.array([n_rays + 5, n_rays + 6])
    return groups


class PermuteInputs(gym.ObservationWrapper):
    """Replace selected observation dims with the same dims from a random stored observation."""

    def __init__(self, env, idx, pool, seed=0):
        super().__init__(env)
        base = env.unwrapped.obs_dim
        k = env.observation_space.shape[0] // base           # frame stacking repeats the layout
        self.idx = np.concatenate([np.asarray(idx) + j * base for j in range(k)])
        self.pool, self.rng = pool, np.random.default_rng(seed)

    def observation(self, obs):
        obs = obs.copy()
        obs[self.idx] = self.pool[self.rng.integers(len(self.pool))][self.idx]
        return obs


class _Recorder(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.seen = []

    def observation(self, obs):
        self.seen.append(obs.copy())
        return obs


def collect_observations(agent, env_cfg, wrappers, n_episodes=10, split="validation"):
    env = _Recorder(make_env(env_cfg, wrappers))
    evaluate(agent, env, split, n_episodes)
    return np.array(env.seen)


def permutation_importance(agent, env_cfg, wrappers, n_episodes=30, split="validation"):
    pool = collect_observations(agent, env_cfg, wrappers, 10, split)
    groups = feature_groups(env_cfg.get("n_rays", 24), env_cfg.get("guidance", False))
    base = pd.DataFrame(evaluate(agent, make_env(env_cfg, wrappers), split, n_episodes))["success"].mean()
    rows = [{"group": "none (baseline)", "success": base, "drop": 0.0}]
    for name, idx in groups.items():
        env = PermuteInputs(make_env(env_cfg, wrappers), idx, pool)
        s = pd.DataFrame(evaluate(agent, env, split, n_episodes))["success"].mean()
        rows.append({"group": name, "success": s, "drop": base - s})
    return pd.DataFrame(rows)


def gradient_saliency(model, observations: np.ndarray, algo: str) -> np.ndarray:
    """Mean |d steering / d input| for each observation dimension (torch models only)."""
    import torch as th

    x = th.as_tensor(observations.astype(np.float32), device=model.device).requires_grad_(True)
    if algo in ("dqn", "double_dqn", "dueling_dqn", "dueling_double_dqn"):
        score = model.q_net(x).max(dim=1).values
    elif algo in ("sac",):
        score = model.actor(x, deterministic=True)[:, 1]          # angular velocity command
    elif algo in ("td3",):
        score = model.actor(x)[:, 1]
    elif algo == "recurrent_ppo":
        raise ValueError("saliency for recurrent policies not supported (needs LSTM state)")
    else:  # ppo / a2c
        dist = model.policy.get_distribution(x)
        d = dist.distribution
        score = d.mean[:, 1] if hasattr(d, "mean") and d.mean.dim() == 2 and d.mean.shape[1] == 2 else d.logits.max(dim=1).values
    score.sum().backward()
    return x.grad.abs().mean(dim=0).cpu().numpy()


def tabular_policy_map(agent) -> pd.DataFrame:
    """Greedy action for every (goal bearing bin, front LiDAR level), other sectors clear, goal far."""
    from ..env.nav_env import ACTION_NAMES

    rows = []
    for b in range(8):
        for front in range(3):
            s = ((b * 3 + 2) * 3 + front)
            s = ((s * 3 + 2) * 3 + 2) * 3 + 2
            q = agent.q[s]
            rows.append({"goal_bearing_deg": b * 45 - 180, "front": ["danger", "near", "clear"][front],
                         "action": ACTION_NAMES[int(np.argmax(q))] if np.abs(q).sum() else "(unvisited)"})
    return pd.DataFrame(rows)
