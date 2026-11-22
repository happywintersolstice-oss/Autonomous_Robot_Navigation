import numpy as np
import pytest

from robot_navigation.partial_memory import AgentMemory
from robot_navigation.partial_sensor import scan
from robot_navigation.partial_observation import SPACE, encode, validate


def test_exact_slices_clipping_and_unknown_vs_fresh_free():
    walls = np.zeros((10, 10), dtype=bool); walls[5, 6] = True
    memory = AgentMemory((5, 5))
    for t in range(26):
        position = (5, 5) if t == 0 else (1, 1)
        reading = scan(walls, position); memory.update(reading, t)
    memory.visit_count[1, 1] = 12
    memory.consecutive_blocked_observations[5, 6] = 12  # Encoder clipping independent of transition tests.
    value = encode(memory, (1, 1), (9, 9), reading.distances, 25)
    assert SPACE.contains(value) and value.shape == (409,) and value.dtype == np.float32
    np.testing.assert_allclose(value[:4], np.array([1, 1, 9, 9])/9)
    assert value[8+56] == 1 and value[108+56] == 1
    assert value[8+99] == -1 and value[108+99] == 0
    assert value[8+11] == 0 and value[108+11] == 0
    assert value[208+11] == 1 and value[308+56] == 1
    assert value[408] == .125
    for invalid in (value.astype(np.float64), np.zeros(104, dtype=np.float32), value*float('nan')):
        with pytest.raises(ValueError):
            validate(invalid)


def test_equal_observable_history_is_identical_before_unseen_difference():
    a = np.zeros((10, 10), dtype=bool); b = a.copy(); b[9, 9] = True
    memories = [AgentMemory((5, 5)), AgentMemory((5, 5))]
    values = []
    for walls, memory in zip((a, b), memories):
        reading = scan(walls, (5, 5)); memory.update(reading, 0)
        values.append(encode(memory, (5, 5), (9, 8), reading.distances, 0))
    assert np.array_equal(*values) and memories[0].snapshot() == memories[1].snapshot()
