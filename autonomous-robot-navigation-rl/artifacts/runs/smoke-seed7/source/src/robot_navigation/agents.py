"""Small policies sharing act(observation); only QLearning.update learns."""
from pathlib import Path
import hashlib

import numpy as np

from .core import MOVES, read_json, write_json


class RandomPolicy:
    policy_id = "random-untrained"

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        return int(self.rng.integers(4))


class BFSPolicy:
    policy_id = "bfs-known-map"

    def __init__(self, grid):
        self.grid = grid
        self.cache = {}

    def act(self, observation):
        r, c, gr, gc = map(int, observation)
        goal = (gr, gc)
        if goal not in self.cache:
            self.cache[goal] = self.grid.distances(goal)
        distances = self.cache[goal]
        for action, (dr, dc) in enumerate(MOVES):
            if distances.get((r + dr, c + dc)) == distances[(r, c)] - 1:
                return action
        raise ValueError("No BFS action from this state")


class QLearning:
    def __init__(self, task, policy_id="q-learning-untrained"):
        self.task, self.policy_id = task, policy_id
        h, w = task.grid.shape
        self.q = np.zeros((h, w, h, w, 4), dtype=np.float64)

    def scores(self, observation):
        return self.q[tuple(observation)]

    def act(self, observation):
        # Fixed evaluation tie break: Up, Down, Left, Right.
        return int(np.argmax(self.scores(observation)))

    def explore(self, observation, epsilon, rng):
        if rng.random() < epsilon:
            return int(rng.integers(4))
        scores = self.scores(observation)
        # Random tie breaking during training only; all actions remain eligible.
        return int(rng.choice(np.flatnonzero(scores == scores.max())))

    def update(self, observation, action, reward, next_observation, terminated, alpha, gamma):
        index = (*map(int, observation), int(action))
        target = reward + (0 if terminated else gamma * self.scores(next_observation).max())
        self.q[index] += alpha * (target - self.q[index])

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, q=self.q)
        write_json(path.with_suffix(".json"), {
            "signature": self.task.signature, "policy_id": self.policy_id,
            "kind": "inference-only; exact training resume unsupported",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **metadata})

    @classmethod
    def load(cls, path, task):
        path = Path(path)
        metadata = read_json(path.with_suffix(".json"))
        task.check(metadata["signature"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        agent = cls(task, metadata["policy_id"])
        with np.load(path, allow_pickle=False) as data:
            table = data["q"]
            if table.shape != agent.q.shape or not np.isfinite(table).all():
                raise ValueError("Invalid Q-table shape or nonfinite values")
            agent.q[:] = table
        return agent
