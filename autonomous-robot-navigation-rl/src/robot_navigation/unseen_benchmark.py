"""Versioned layout splits and a full-map observation over the core dynamics."""
from collections import defaultdict
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import yaml

from .benchmark import distance_band
from .core import ROOT, GridMap, NavigationEnv, Task, digest, load_task, read_json, write_json
from .release import sha

OBSERVATION = 'full-occupancy-coordinates-v1'
DIMENSIONS = 104
BENCHMARK = 'unseen-maps-v1'


def configuration(root=ROOT):
    return yaml.safe_load((root/'configs/unseen_maps.yaml').read_text(encoding='utf-8'))


def geometry_hash(grid):
    """IDs do not affect equality, and D4 transformations share one identity."""
    array = np.array([list(row) for row in grid.rows])
    variants = [tuple(''.join(row) for row in value)
                for rotation in range(4)
                for value in (np.rot90(array, rotation), np.fliplr(np.rot90(array, rotation)))]
    return digest(min(variants))


def map_observation(coordinates, grid):
    if grid.shape != (10, 10) or len(coordinates) != 4:
        raise ValueError('Map-aware observation requires four coordinates and a 10x10 layout')
    walls = [int(cell == '#') for row in grid.rows for cell in row]
    return np.array([float(v)/9 for v in coordinates] + walls, dtype=np.float32)


def table_key(observation):
    observation = np.asarray(observation)
    if (observation.shape != (DIMENSIONS,) or not np.isfinite(observation).all()
            or (observation < 0).any() or (observation > 1).any()
            or not np.isin(observation[4:], (0, 1)).all()):
        raise ValueError('Invalid full-map observation')
    coordinates = tuple(np.rint(observation[:4]*9).astype(int))
    if not np.allclose(np.array(coordinates)/9, observation[:4], atol=1e-6):
        raise ValueError('Coordinates must identify discrete grid cells')
    return observation[4:].astype(np.uint8).tobytes(), coordinates


def _scenarios(grid, config, rng):
    bands = defaultdict(list)
    diameter = 0
    for goal in grid.free_cells:
        for start, length in sorted(grid.distances(goal).items()):
            if start == goal:
                continue
            diameter = max(diameter, length)
            bands[distance_band(length)].append({
                'id':f'{grid.map_id}-r{start[0]}c{start[1]}-r{goal[0]}c{goal[1]}',
                'start':list(start), 'goal':list(goal), 'shortest_length':length,
                'distance_band':distance_band(length)})
    if diameter < config['minimum_diameter']:
        return None
    if any(len(bands[band]) < config['scenarios_per_band'] for band in ('short','medium','long')):
        return None
    selected = []
    for band in ('short', 'medium', 'long'):
        selected.extend(rng.sample(bands[band], config['scenarios_per_band']))
    return sorted(selected, key=lambda s:s['id']), diameter


def prepare(root=ROOT):
    config = configuration(root)
    if (config['benchmark_id'] != BENCHMARK or config['shape'] != [10,10]
            or config['observation_version'] != OBSERVATION
            or config['generator_version'] != 'connected-obstacles-d4-v1'):
        raise ValueError('Unsupported unseen-map benchmark contract')
    directory = root/'data/unseen'/BENCHMARK
    if directory.exists():
        benchmark = MapBenchmark(root)
        if benchmark.record['configuration'] != config:
            raise ValueError('Existing benchmark configuration changed; use a new version')
        return audit_benchmark(root)
    # Generate completely before publication. An interrupted publication is not
    # accepted by MapBenchmark because the final manifest is written last.
    occupied = {geometry_hash(load_task(root).grid)}
    rng = random.Random(config['generation_seed'])
    splits = {}
    attempts = 0
    for split, count in config['layout_counts'].items():
        layouts = []
        while len(layouts) < count:
            attempts += 1
            if attempts > 100000:
                raise RuntimeError('Layout generation exceeded its registered rejection limit')
            blocked = set(rng.sample(range(100), config['blocked_cells']))
            rows = tuple(''.join('#' if r*10+c in blocked else '.' for c in range(10)) for r in range(10))
            grid = GridMap(f'{BENCHMARK}-{split}-{len(layouts)+1:03d}', rows)
            identity = geometry_hash(grid)
            if identity in occupied or len(grid.distances(grid.free_cells[0])) != len(grid.free_cells):
                continue
            result = _scenarios(grid, config, rng)
            if result is None:
                continue
            scenarios, diameter = result
            layouts.append({'map_id':grid.map_id, 'rows':list(rows), 'map_hash':grid.hash,
                            'geometry_hash':identity, 'diameter':diameter, 'scenarios':scenarios})
            occupied.add(identity)
        splits[split] = {'benchmark_id':BENCHMARK, 'split':split, 'layouts':layouts}
    environment = dict(load_task(root).config)
    signature = {'benchmark_id':BENCHMARK, 'configuration_hash':digest(config),
                 'environment_hash':digest(environment), 'observation_version':OBSERVATION,
                 'splits_hash':digest(splits)}
    directory.mkdir(parents=True, exist_ok=False)
    for split, record in splits.items():
        write_json(directory/f'{split}.json', record)
    manifest = {'signature':signature, 'configuration':config, 'environment':environment,
                'files':{f'{split}.json':sha(directory/f'{split}.json') for split in splits},
                'generation_attempts':attempts,
                'final_access':'sealed; no training, ordinary evaluation or dashboard access'}
    write_json(directory/'manifest.json', manifest)
    return audit_benchmark(root)


class MapBenchmark:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.directory = self.root/'data/unseen'/BENCHMARK
        self.record = read_json(self.directory/'manifest.json')
        self.config = self.record['configuration']
        self.signature = self.record['signature']
        if (self.config['benchmark_id'] != BENCHMARK or self.config['observation_version'] != OBSERVATION
                or digest(self.config) != self.signature['configuration_hash']
                or digest(self.record['environment']) != self.signature['environment_hash']):
            raise ValueError('Unseen benchmark contract hash mismatch')

    def layouts(self, split):
        if split not in ('train','validation'):
            raise ValueError('Final layouts are sealed; only train/validation are permitted')
        return self._read(split)

    def _read(self, split):
        filename = f'{split}.json'
        if sha(self.directory/filename) != self.record['files'][filename]:
            raise ValueError('Unseen split checksum mismatch')
        record = read_json(self.directory/filename)
        if record['benchmark_id'] != BENCHMARK or record['split'] != split:
            raise ValueError('Unseen split identity mismatch')
        return record['layouts']

    def task(self, layout):
        grid = GridMap(layout['map_id'], tuple(layout['rows']))
        if grid.hash != layout['map_hash']:
            raise ValueError('Layout identity mismatch')
        return Task(grid, {**self.record['environment'], 'map_id':grid.map_id})

    def cases(self, split, monitoring=False):
        cases = []
        for layout in self.layouts(split):
            scenarios = layout['scenarios']
            if monitoring:
                scenarios = [s for band in ('short','medium','long')
                             for s in [s for s in scenarios if s['distance_band']==band]
                             [:self.config['monitoring_per_band']]]
            cases.extend((layout, scenario) for scenario in scenarios)
        return cases


def audit_benchmark(root=ROOT):
    """Preparation-only split audit. No policy or final outcome is consulted."""
    from .core import validate_scenario
    benchmark = MapBenchmark(root)
    geometries = {geometry_hash(load_task(root).grid)}
    ids = set()
    splits = {}
    records = {}
    for split, count in benchmark.config['layout_counts'].items():
        layouts = benchmark._read(split)
        if len(layouts) != count:
            raise ValueError('Layout count mismatch')
        for layout in layouts:
            task = benchmark.task(layout)
            grid = task.grid
            identity = geometry_hash(grid)
            if identity in geometries or identity != layout['geometry_hash']:
                raise ValueError('Duplicate or changed layout geometry')
            if (len(grid.free_cells) != 100-benchmark.config['blocked_cells']
                    or len(grid.distances(grid.free_cells[0])) != len(grid.free_cells)):
                raise ValueError('Layout occupancy/connectivity mismatch')
            diameter = max(max(grid.distances(cell).values()) for cell in grid.free_cells)
            if diameter != layout['diameter'] or diameter < benchmark.config['minimum_diameter']:
                raise ValueError('Layout diameter mismatch')
            geometries.add(identity)
            if len(layout['scenarios']) != 3*benchmark.config['scenarios_per_band']:
                raise ValueError('Scenario support mismatch')
            if any(sum(s['distance_band']==band for s in layout['scenarios']) != benchmark.config['scenarios_per_band']
                   for band in ('short','medium','long')):
                raise ValueError('Distance-stratum support mismatch')
            for scenario in layout['scenarios']:
                validate_scenario(grid, scenario)
                if (scenario['id'] in ids or not scenario['id'].startswith(grid.map_id+'-')
                        or scenario['distance_band'] != distance_band(scenario['shortest_length'])):
                    raise ValueError('Scenario identity/distance mismatch')
                ids.add(scenario['id'])
        records[split] = {'benchmark_id':BENCHMARK,'split':split,'layouts':layouts}
        splits[split] = {'layouts':len(layouts), 'scenarios':sum(len(m['scenarios']) for m in layouts)}
    if digest(records) != benchmark.signature['splits_hash']:
        raise ValueError('Benchmark split-set hash mismatch')
    return {'signature':benchmark.signature, 'splits':splits, 'd4_unique':True,
            'fixed_map_excluded':True, 'final_policy_executions':0}


class MapNavigationEnv(gym.Env):
    """Choose a declared layout/route, then delegate every move to NavigationEnv."""
    metadata = {'render_modes':[]}

    def __init__(self, benchmark, cases):
        self.benchmark, self.cases = benchmark, list(cases)
        if not self.cases:
            raise ValueError('At least one declared training case is required')
        declared = {(layout['map_id'],scenario['id']):(layout,scenario)
                    for layout,scenario in benchmark.cases('train')}
        if any(declared.get((layout['map_id'],scenario['id'])) != (layout,scenario)
               for layout,scenario in self.cases):
            raise ValueError('Training requires declared training layouts and scenarios')
        self.observation_space = gym.spaces.Box(0, 1, (DIMENSIONS,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self.env = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.layout, scenario = self.cases[int(self.np_random.integers(len(self.cases)))]
        self.env = NavigationEnv(self.benchmark.task(self.layout), [scenario])
        raw, info = self.env.reset(seed=seed, options={'scenario':scenario})
        return map_observation(raw, self.env.grid), info

    def step(self, action):
        if self.env is None:
            raise RuntimeError('Reset before stepping')
        raw, reward, terminated, truncated, info = self.env.step(action)
        return map_observation(raw, self.env.grid), reward, terminated, truncated, info
