"""Stable-Baselines3 callbacks: validation learning curves, best-model saving, curriculum.

ValidationCallback
    Every `eval_freq` environment steps, play `n_episodes` on VALIDATION maps with
    the deterministic policy. Writes curve.csv (used for learning-curve plots and
    the sample-efficiency metric) and keeps best_model.zip = highest validation
    success. Optionally reports to an Optuna trial so bad trials are pruned early.

CurriculumCallback
    Starts training on easy maps and raises the difficulty level once the rolling
    training success rate passes a threshold (Part B "restructuring learning").
"""

from __future__ import annotations

import csv
import time
from collections import deque
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from ..agents.base import SB3Agent
from ..evaluation.rollout import evaluate, summarise


def call_envs(vec_env, method, *args):
    """Call a method on every underlying HospitalNavEnv (works across SB3/gymnasium versions)."""
    try:
        return vec_env.env_method(method, *args)
    except AttributeError:
        return [getattr(e.unwrapped, method)(*args) for e in vec_env.envs]


class ValidationCallback(BaseCallback):
    def __init__(self, eval_env, run_dir, label, action_mode, eval_freq=10_000, n_episodes=20,
                 recurrent=False, trial=None):
        super().__init__()
        self.eval_env, self.run_dir, self.label = eval_env, Path(run_dir) if run_dir else None, label
        self.action_mode, self.eval_freq, self.n_episodes = action_mode, eval_freq, n_episodes
        self.recurrent, self.trial = recurrent, trial
        self.best = (-1.0, -float("inf"))
        self.history = []
        self._next = eval_freq
        self._t0 = time.time()

    def _evaluate(self):
        agent = SB3Agent(self.model, self.label, self.action_mode, recurrent=self.recurrent)
        s = summarise(evaluate(agent, self.eval_env, "validation", self.n_episodes))
        row = {"timesteps": self.num_timesteps, "wall_s": round(time.time() - self._t0, 1), **s,
               "difficulty": call_envs(self.training_env, "get_difficulty")[0]}
        self.history.append(row)
        if self.run_dir:
            path = self.run_dir / "curve.csv"
            new = not path.exists()
            with path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            score = (s["success"], s["return"])
            if score > self.best:
                self.best = score
                self.model.save(self.run_dir / "best_model.zip")
        if self.trial is not None:
            import optuna

            self.trial.report(s["success"], step=self.num_timesteps)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.eval_freq
            self._evaluate()
        return True

    def _on_training_end(self):
        if not self.history or self.history[-1]["timesteps"] != self.num_timesteps:
            self._evaluate()


class CurriculumCallback(BaseCallback):
    def __init__(self, start_level=0, max_level=3, threshold=0.7, window=100):
        super().__init__()
        self.level, self.max_level, self.threshold = start_level, max_level, threshold
        self.outcomes = deque(maxlen=window)

    def _on_training_start(self):
        call_envs(self.training_env, "set_difficulty", self.level)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.outcomes.append(info.get("is_success", False))
        if (self.level < self.max_level and len(self.outcomes) == self.outcomes.maxlen
                and sum(self.outcomes) / len(self.outcomes) >= self.threshold):
            self.level += 1
            self.outcomes.clear()
            call_envs(self.training_env, "set_difficulty", self.level)
            self.logger.record("curriculum/level", self.level)
        return True
