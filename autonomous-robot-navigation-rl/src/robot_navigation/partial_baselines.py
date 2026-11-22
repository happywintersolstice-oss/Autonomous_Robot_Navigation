"""Same-information references: every decision accepts only the 409-value input."""

from collections import deque

import numpy as np

from .partial_contract import MOVES
from .partial_observation import validate


class OnlinePlanner:
    """Replan optimistically on remembered occupancy, never on the true layout.

    Unknown cells are traversable hypotheses. Reconsider stale blocked cells
    after four steps, but never execute a move into a freshly sensed obstacle.
    Prefer an available route over waiting. If no route/frontier move exists,
    Wait and rescan; permanent walls are never secretly classified or cleared.
    """

    policy_id = "online-observed-map-replanner-v1"
    algorithm = "online-planner"

    def act(self, observation):
        value = validate(observation)
        r, c, gr, gc = np.rint(value[:4] * 9).astype(int)
        occupancy = value[8:108].reshape(10, 10)
        ages = value[108:208].reshape(10, 10)
        visits = value[208:308].reshape(10, 10)
        traversable = (occupancy != 1) | (ages >= 0.2)
        start, goal = (r, c), (gr, gc)
        todo, first_action = deque([start]), {start: None}
        while todo:
            cell = todo.popleft()
            if cell == goal:
                return int(first_action[cell]) if first_action[cell] is not None else 4
            candidates = []
            for action, (dr, dc) in enumerate(MOVES[:4]):
                nxt = (cell[0] + dr, cell[1] + dc)
                if (
                    all(0 <= v < 10 for v in nxt)
                    and traversable[nxt]
                    and nxt not in first_action
                ):
                    candidates.append((float(visits[nxt]), action, nxt))
            for _, action, nxt in sorted(candidates):
                # Sensor refreshes all adjacent cells: a stale optimistic graph
                # never overrides a currently blocked immediate move.
                if cell == start and occupancy[nxt] == 1:
                    continue
                first_action[nxt] = action if cell == start else first_action[cell]
                todo.append(nxt)
        # Reach an observed frontier (known free adjacent to unknown), if possible.
        candidates = []
        for action, (dr, dc) in enumerate(MOVES[:4]):
            nxt = (r + dr, c + dc)
            if all(0 <= v < 10 for v in nxt) and occupancy[nxt] == 0:
                candidates.append(
                    (float(visits[nxt]), abs(nxt[0] - gr) + abs(nxt[1] - gc), action)
                )
        # If every neighbor merely returns to a repeatedly visited dead end,
        # staying for a fresh scan is this reference's declared heuristic.
        if candidates and min(v[0] for v in candidates) == 0:
            return int(min(candidates)[2])
        return 4


class AlwaysWait:
    policy_id = "always-wait-v1"
    algorithm = "always-wait"

    def act(self, observation):
        validate(observation)
        return 4


class RandomPolicy:
    algorithm = "random"

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.policy_id = f"partial-random-seed{seed}"

    def act(self, observation):
        validate(observation)
        return int(self.rng.integers(5))
