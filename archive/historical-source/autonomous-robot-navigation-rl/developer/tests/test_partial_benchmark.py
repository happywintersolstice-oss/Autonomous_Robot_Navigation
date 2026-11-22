from copy import deepcopy

import numpy as np
import pytest

from robot_navigation.core import GridMap, read_json, write_json
from robot_navigation.partial_benchmark import DevelopmentBenchmark, event_schedule, geometry_hash, prepare
from robot_navigation.partial_contract import configuration, BENCHMARK


def small_config():
    return {**configuration(), 'layout_counts': {'train': 2, 'validation': 1}, 'routes_per_layout': 3}


def test_development_splits_are_deterministic_connected_d4_disjoint_and_final_absent(tmp_path):
    config = small_config()
    first = prepare(tmp_path/'a', config)
    second = prepare(tmp_path/'b', config)
    a, b = DevelopmentBenchmark(tmp_path/'a'), DevelopmentBenchmark(tmp_path/'b')
    assert first['files'] == second['files']
    assert a.audit() == {'layouts': 3, 'scenarios': 18, 'final_generated': False}
    identities = set()
    for split in ('train', 'validation'):
        for layout in a.layouts(split):
            assert layout['geometry_hash'] not in identities
            identities.add(layout['geometry_hash'])
            grid = GridMap(layout['layout_id'], tuple(layout['rows']))
            assert len(grid.distances(grid.free_cells[0])) == len(grid.free_cells)
            for dynamic in (False, True):
                rows = [c for c in layout['scenarios'] if bool(c['events']) == dynamic]
                assert len(rows) == 3
                assert all(c['condition'] == ('familiar' if split == 'train' else 'unseen')+('-dynamic' if dynamic else '-static') for c in rows)
                for case in rows:
                    if dynamic:
                        assert case['events'] == event_schedule(grid, case['start'], case['goal'], case['event_seed'], config)
                        assert all(3 <= e['clear']-e['start'] <= 8 for e in case['events'])
    assert not (a.folder/'test.json').exists()
    with pytest.raises(ValueError, match='Final'):
        a.layouts('test')
    assert prepare(tmp_path/'a', config) == first
    with pytest.raises(ValueError, match='configuration changed'):
        prepare(tmp_path/'a', {**config, 'generation_seed': 1})


def test_d4_and_historical_geometry_exclusion_and_checksum(tmp_path):
    config = small_config(); prepare(tmp_path/'a', config)
    layout = DevelopmentBenchmark(tmp_path/'a').layouts('train')[0]
    array = np.array([list(row) for row in layout['rows']])
    rotated = [''.join(row) for row in np.fliplr(np.rot90(array))]
    assert geometry_hash(rotated) == geometry_hash(layout['rows'])
    write_json(tmp_path/'b/data/maps/already-inspected.json', {'rows': rotated})
    manifest = prepare(tmp_path/'b', config)
    reader = DevelopmentBenchmark(tmp_path/'b')
    assert layout['geometry_hash'] in manifest['historical_excluded_geometry_hashes']
    assert layout['geometry_hash'] not in {m['geometry_hash'] for m in reader.layouts('train')}
    (reader.folder/'train.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        reader.layouts('train')


def test_generation_cannot_expose_a_final_split(tmp_path):
    with pytest.raises(ValueError, match='Only training'):
        prepare(tmp_path, {**small_config(), 'layout_counts': {'train': 1, 'validation': 1, 'test': 1}})
