"""Run one or many episodes and turn them into per-episode metric rows.

Every algorithm is evaluated on the SAME list of map seeds (common random
numbers), so differences between algorithms come from the agents, not from
luckier maps.
"""

from __future__ import annotations

import time

import numpy as np

from ..env.maps import map_seed


def failure_type(outcome: str, path: np.ndarray) -> str:
    """Categorise how an episode ended, for the error analysis.

    success / collision_wall / collision_human come straight from the env.
    A timeout is split into:
      stuck_oscillating  robot moved < 0.5 m in its last 100 steps (dithering, dead end)
      too_slow           robot was still making progress but ran out of time
    """
    if outcome != "timeout":
        return outcome
    tail = path[-100:]
    if len(tail) > 1 and np.linalg.norm(tail.max(axis=0) - tail.min(axis=0)) < 0.5:
        return "stuck_oscillating"
    return "too_slow"


def run_episode(agent, env, map_seed_value: int, episode_seed: int = 0, record: bool = False):
    """Play one episode. Returns (metrics dict, frames list or None)."""
    obs, info = env.reset(seed=episode_seed, options={"map_seed": int(map_seed_value)})
    agent.reset()
    u = env.unwrapped
    total, steps, intrusions = 0.0, 0, 0
    min_h = float("inf")
    terms_sum: dict[str, float] = {}
    frames = [u.state_snapshot()] if record else None
    t0 = time.perf_counter()
    done = False
    while not done:
        action = agent.act(obs, env)
        obs, r, term, trunc, info = env.step(action)
        total += r
        steps += 1
        for k, v in info["reward_terms"].items():
            terms_sum[k] = terms_sum.get(k, 0.0) + v
        min_h = min(min_h, info["min_human_dist"])
        intrusions += int(info["min_human_dist"] < u.cfg["social_radius"])
        if record:
            frames.append(u.state_snapshot())
        done = term or trunc
    outcome = info["outcome"]
    path = np.array(u.path)
    success = outcome == "success"
    row = {
        "map_seed": int(map_seed_value),
        "episode_seed": int(episode_seed),
        "outcome": outcome,
        "failure_type": failure_type(outcome, path),
        "success": float(success),
        "collision": float(outcome.startswith("collision")),
        "collision_human": float(outcome == "collision_human"),
        "timeout": float(outcome == "timeout"),
        "return": total,
        "steps": steps,
        "time_to_goal_s": steps * u.cfg["dt"] if success else np.nan,
        "path_length": info["path_length"],
        "optimal_length": info["optimal_length"],
        # path efficiency in (0, 1]: 1 = drove the A* optimal distance
        "path_efficiency": min(1.0, info["optimal_length"] / max(info["path_length"], 1e-6)) if success else np.nan,
        "min_human_dist": min_h if np.isfinite(min_h) else np.nan,
        "social_intrusion_rate": intrusions / steps,
        "decision_ms": 1000 * (time.perf_counter() - t0) / steps,
        **{f"r_{k}": v for k, v in terms_sum.items()},
    }
    return row, frames


def evaluate(agent, env, split: str = "validation", n_episodes: int = 20, start_index: int = 0):
    """Evaluate on the first `n_episodes` maps of a split. Returns a list of rows."""
    rows = []
    for i in range(n_episodes):
        row, _ = run_episode(agent, env, map_seed(split, start_index + i), episode_seed=i)
        rows.append(row)
    return rows


def summarise(rows) -> dict:
    """Quick mean summary (used by training callbacks; full stats live in metrics.py)."""
    if not rows:
        return {}
    keys = ["success", "collision", "timeout", "return", "path_efficiency", "social_intrusion_rate"]
    return {k: float(np.nanmean([r[k] for r in rows])) if any(np.isfinite(r[k]) for r in rows) else float("nan")
            for k in keys}
