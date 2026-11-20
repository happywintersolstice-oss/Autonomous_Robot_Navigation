import numpy as np

from robot_navigation.partial_memory import AgentMemory
from robot_navigation.partial_sensor import scan


def test_memory_only_updates_visible_cells_and_wait_does_not_enter():
    walls = np.zeros((10, 10), dtype=bool)
    memory = AgentMemory((5, 5))
    assert (memory.occupancy == -1).all() and (memory.last_seen_step == -1).all()
    memory.update(scan(walls, (5, 5)), 0)
    assert memory.occupancy[5, 5] == 0 and memory.last_seen_step[9, 9] == -1
    assert memory.occupancy[9, 9] == -1 and memory.visit_count.sum() == 1
    memory.update(scan(walls, (5, 5)), 1)  # Wait: scan again, no entry.
    assert memory.visit_count.sum() == 1
    memory.enter((5, 6)); memory.enter((5, 5))
    assert memory.visit_count[5, 5] == 2
    assert AgentMemory((5, 5)).visit_count.sum() == 1


def test_stale_changes_and_interrupted_blockage_streak():
    walls = np.zeros((10, 10), dtype=bool); walls[5, 6] = True
    memory = AgentMemory((5, 5))
    memory.update(scan(walls, (5, 5)), 0)
    memory.update(scan(walls, (5, 5)), 1)
    assert memory.consecutive_blocked_observations[5, 6] == 2
    memory.update(scan(walls, (2, 2)), 2)
    assert memory.occupancy[5, 6] == 1 and memory.last_seen_step[5, 6] == 1
    assert memory.consecutive_blocked_observations[5, 6] == 0
    memory.update(scan(walls, (5, 5)), 3)
    assert memory.consecutive_blocked_observations[5, 6] == 1
    walls[5, 6] = False
    memory.update(scan(walls, (2, 2)), 4)
    assert memory.occupancy[5, 6] == 1
    memory.update(scan(walls, (5, 5)), 5)
    assert memory.occupancy[5, 6] == 0 and memory.consecutive_blocked_observations[5, 6] == 0
    walls[5, 6] = True
    memory.update(scan(walls, (2, 2)), 6)
    assert memory.occupancy[5, 6] == 0
    memory.update(scan(walls, (5, 5)), 7)
    assert memory.occupancy[5, 6] == 1
