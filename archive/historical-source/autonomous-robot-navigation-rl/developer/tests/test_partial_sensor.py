import numpy as np
import pytest

from robot_navigation.partial_sensor import scan


@pytest.mark.parametrize('distance,encoded', [(1, .5), (2, 1), (3, 0)])
def test_range_and_occlusion(distance, encoded):
    walls = np.zeros((10, 10), dtype=bool)
    walls[5, 5+distance] = True
    reading = scan(walls, (5, 5))
    assert reading.distances[2] == encoded
    assert (5, 6, int(distance == 1)) in reading.cells
    assert all(c <= 5+min(distance, 2) for r, c, _ in reading.cells if r == 5 and c > 5)
    assert all((r, c) != (5, 8) for r, c, _ in reading.cells)


def test_cardinal_order_boundary_and_identical_occupancy():
    walls = np.zeros((10, 10), dtype=bool)
    walls[4, 5] = walls[5, 7] = True
    assert scan(walls, (5, 5)).distances == (.5, 0, 1, 0)
    assert scan(np.zeros_like(walls), (0, 0)).distances == (.5, 0, 0, .5)
    assert scan(np.zeros_like(walls), (1, 1)).distances == (1, 0, 0, 1)
    static = walls.copy()
    temporary = np.zeros_like(walls); temporary[4, 5] = temporary[5, 7] = True
    assert scan(static, (5, 5)) == scan(temporary, (5, 5))


def test_hidden_walls_are_not_revealed():
    a = np.zeros((10, 10), dtype=bool); a[5, 6] = True
    b = a.copy(); b[5, 7] = b[9, 9] = True
    assert scan(a, (5, 5)) == scan(b, (5, 5))
