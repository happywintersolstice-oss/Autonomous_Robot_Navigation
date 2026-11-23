"""Tabular Q-learning and SARSA  (owner: Member 1)

Why include tabular methods?
----------------------------
They are the textbook starting point and fully interpretable: every decision is
one row of a table you can print. They also show *why* deep RL is needed: the
continuous LiDAR input has to be squashed into a small number of discrete states,
and information is lost.

State discretisation (1 944 states)
-----------------------------------
goal bearing     8 bins of 45 degrees
goal distance    3 bins: < 1 m, < 3 m, further
4 LiDAR sectors  front / left / back / right, each: danger (< 0.4 m), near (< 1 m), clear
=> 8 * 3 * 3^4 = 1944 states, 5 actions

Update rules
------------
Q-learning (off-policy): Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
SARSA      (on-policy):  Q(s,a) += alpha * [r + gamma * Q(s',a')        - Q(s,a)]
where a' is the action the epsilon-greedy policy will actually take. SARSA learns
the value of its own exploratory behaviour, so it is usually more cautious near
walls - a difference we measure in the collision rate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Agent

N_ACTIONS = 5


class Discretiser:
    """Observation -> state index.

    guidance=False  bearing bins describe the straight-line direction to the GOAL.
    guidance=True   bearing bins describe the direction to the next A* WAYPOINT (env config
                    `guidance: true` appends it to the observation). Same table size, but the
                    direction now "walks around" walls, which a 1944-state table cannot learn
                    by itself - a representation change, not a bigger model (Part B).
    """

    def __init__(self, n_rays=24, max_range=3.0, dist_norm=15.0, guidance=False):
        self.n_rays, self.max_range, self.dist_norm = n_rays, max_range, dist_norm
        self.guidance = bool(guidance)
        ang = np.degrees(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
        ang = (ang + 180) % 360 - 180  # [-180, 180), 0 = straight ahead
        self.sectors = [
            np.flatnonzero(np.abs(ang) <= 45),                 # front
            np.flatnonzero((ang > 45) & (ang < 135)),          # left
            np.flatnonzero(np.abs(ang) >= 135),                # back
            np.flatnonzero((ang < -45) & (ang > -135)),        # right
        ]
        self.n_states = 8 * 3 * 3 ** 4

    def __call__(self, obs) -> int:
        n = self.n_rays
        scan = np.where(obs[:n] < 0, 1.0, obs[:n]) * self.max_range   # dropped ray -> treat as clear
        dist = obs[n] * self.dist_norm
        if self.guidance:
            if len(obs) < n + 7:
                raise ValueError("guidance discretiser needs an env with `guidance: true`")
            bearing = np.arctan2(obs[n + 5], obs[n + 6])     # direction to the A* waypoint
        else:
            bearing = np.arctan2(obs[n + 1], obs[n + 2])     # straight-line direction to the goal
        b = int(((bearing + np.pi) / (2 * np.pi) * 8 + 0.5) % 8)
        d = 0 if dist < 1.0 else 1 if dist < 3.0 else 2
        s = b * 3 + d
        for idx in self.sectors:
            m = scan[idx].min()
            s = s * 3 + (0 if m < 0.4 else 1 if m < 1.0 else 2)
        return int(s)


class TabularAgent(Agent):
    action_mode = "discrete"

    def __init__(self, algo="q_learning", n_rays=24, max_range=3.0, dist_norm=15.0, q=None, guidance=False):
        if algo not in ("q_learning", "sarsa"):
            raise ValueError(f"tabular algo must be q_learning or sarsa, got {algo!r}")
        self.name, self.algo = algo, algo
        self.disc = Discretiser(n_rays, max_range, dist_norm, guidance)
        # optimistic-ish zero init; ties broken randomly during training
        self.q = np.zeros((self.disc.n_states, N_ACTIONS)) if q is None else q

    def act(self, obs, env=None):
        return int(np.argmax(self.q[self.disc(obs)]))

    def _explore(self, s, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(N_ACTIONS))
        row = self.q[s]
        return int(rng.choice(np.flatnonzero(row == row.max())))

    # ------------------------------------------------------------------ train
    def train(self, env, episodes, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05,
              eps_decay_frac=0.8, seed=0, callback=None, callback_every=200):
        """Classic episode loop. `callback(episode, steps, agent)` is used for validation curves."""
        rng = np.random.default_rng(seed)
        total_steps = 0
        for ep in range(episodes):
            frac = min(1.0, ep / max(1, eps_decay_frac * episodes))
            eps = eps_start + frac * (eps_end - eps_start)
            obs, _ = env.reset(seed=int(seed * 1_000_003 + ep))
            s = self.disc(obs)
            a = self._explore(s, eps, rng)
            done = False
            while not done:
                obs, r, term, trunc, _ = env.step(a)
                s2 = self.disc(obs)
                a2 = self._explore(s2, eps, rng)
                if term:
                    target = r                                    # no future after a terminal state
                elif self.algo == "q_learning":
                    target = r + gamma * self.q[s2].max()
                else:
                    target = r + gamma * self.q[s2, a2]
                # truncation (timeout) is NOT terminal: we still bootstrap from s2
                self.q[s, a] += alpha * (target - self.q[s, a])
                s, a = s2, a2
                total_steps += 1
                done = term or trunc
            if callback and (ep + 1) % callback_every == 0:
                callback(ep + 1, total_steps, self)
        return self

    # ---------------------------------------------------------------- save/load
    def save(self, run_dir):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "q_table.npy", self.q)

    @classmethod
    def load(cls, run_dir, algo, env_cfg):
        q = np.load(Path(run_dir) / "q_table.npy")
        return cls(algo, env_cfg.get("n_rays", 24), env_cfg.get("max_range", 3.0), env_cfg.get("dist_norm", 15.0), q=q,
                   guidance=env_cfg.get("guidance", False))

    def visited_fraction(self):
        return float((np.abs(self.q).sum(axis=1) > 0).mean())
