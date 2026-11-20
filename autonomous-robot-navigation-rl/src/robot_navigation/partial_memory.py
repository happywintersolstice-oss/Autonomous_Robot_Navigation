"""Agent history; it has no environment, event schedule or true-map reference."""
import numpy as np


class AgentMemory:
    def __init__(self, start):
        self.occupancy = np.full((10, 10), -1, dtype=np.int8)
        self.last_seen_step = np.full((10, 10), -1, dtype=np.int64)
        self.visit_count = np.zeros((10, 10), dtype=np.int64)
        self.consecutive_blocked_observations = np.zeros((10, 10), dtype=np.int64)
        self.currently_seen = np.zeros((10, 10), dtype=bool)
        self.visit_count[tuple(start)] = 1
        self.scan_step = -1

    def enter(self, position):
        """Call only on a successful move into a cell; never on Wait/collision."""
        self.visit_count[tuple(position)] += 1

    def update(self, reading, step):
        if type(step) is not int or step != self.scan_step+1:
            raise ValueError('Memory requires one chronological scan per decision time')
        old_seen = self.currently_seen.copy()
        old_streak = self.consecutive_blocked_observations.copy()
        self.currently_seen.fill(False)
        self.consecutive_blocked_observations.fill(0)
        corrections = 0
        for r, c, blocked in reading.cells:
            previous = int(self.occupancy[r, c])
            corrections += previous != -1 and previous != blocked
            if blocked:
                continuous = old_seen[r, c] and self.last_seen_step[r, c] == step-1 and previous == 1
                self.consecutive_blocked_observations[r, c] = int(old_streak[r, c])+1 if continuous else 1
            self.occupancy[r, c] = blocked
            self.last_seen_step[r, c] = step
            self.currently_seen[r, c] = True
        self.scan_step = step
        return {'sensed_cells': len(reading.cells), 'occupancy_corrections': int(corrections)}

    def snapshot(self):
        return {name: getattr(self, name).tolist() for name in
                ('occupancy', 'last_seen_step', 'visit_count',
                 'consecutive_blocked_observations', 'currently_seen')}
