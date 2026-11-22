"""Read-only dashboard evidence and matched frozen-policy replay helpers."""
from pathlib import Path

from .agents import load_policy, viewer_checkpoint
from .benchmark import load_scenarios
from .core import NavigationEnv, ROOT, read_json
from .episodes import rollout, validate_replay
from .refinement import task_for_run

PAGES = ('Navigation simulator', 'Overview', 'Learning analysis', 'Agent comparison',
         'Optimization and robustness', 'Neural tuning', 'Methodology')
PART_B_SEEDS = (101, 211, 307, 401, 503)
PART_A_SEEDS = (7, 19, 43, 67, 91)


def baseline_lineup(seed, root=ROOT):
    """Declared baseline replicates, independent of the scenario or its outcome."""
    if seed not in PART_A_SEEDS:
        raise ValueError('Choose one of the five declared baseline training seeds')
    algorithms = ('q-learning', 'sarsa', 'dqn', 'ppo')
    runs = [root / 'artifacts/runs' / f'm{7 if a == "ppo" else 3}-{a}-seed{seed}' for a in algorithms]
    for directory, algorithm in zip(runs, algorithms):
        manifest = read_json(directory / 'manifest.json')
        if (manifest['status'] != 'complete' or manifest['algorithm'] != algorithm
                or manifest['training_seed'] != seed or manifest['training_interactions'] != 200000):
            raise ValueError(f'Baseline run identity or budget mismatch: {directory.name}')
    return runs


def training_paths(directory):
    """Only manifest-declared recordings, confined to this run's trace folder."""
    paths = []
    for item in read_json(directory / 'manifest.json').get('training_traces', {}).get('records', []):
        path = directory / item['path']
        if not path.resolve().is_relative_to((directory / 'training_traces').resolve()):
            raise ValueError('Training recording must stay inside its run')
        paths.append(path)
    return paths


def training_episode(task, directory, path, root=ROOT):
    """Read actual recorded training behavior, never a frozen-model reconstruction."""
    from .representation import validate_training_prefix
    from .release import sha
    paths = training_paths(directory)
    if path not in paths:
        raise ValueError('Training recording is not declared in the run manifest')
    manifest = read_json(directory / 'manifest.json')
    item = manifest['training_traces']['records'][paths.index(path)]
    if sha(path) != item['sha256']:
        raise ValueError('Training recording checksum mismatch')
    episode = read_json(path)
    scenarios = {s['id']: s for s in load_scenarios(task, 'train', root)}
    if (episode['scenario'] != scenarios.get(episode['scenario']['id'])
            or episode['seed'] != manifest['training_seed']
            or episode['complete'] != item['complete'] or len(episode['transitions']) != item['actions']):
        raise ValueError('Training recording identity mismatch')
    validate_training_prefix(task, episode)
    return episode


def seek_from_widget(state, key, *, player_key, position_key, active_key):
    """Ignore delayed events from a timeline/event widget no longer displayed."""
    if state.get(active_key) != key:
        return False
    player = state[player_key]
    target = state[key]
    if target == player.index:
        return False
    player.pause()
    player.seek(target)
    state[player_key] = player
    state[position_key] = player.index
    return True


def system_runs():
    systems = {}
    for stage, label in (('control', 'Baseline'), ('tuned', 'Refined')):
        for algorithm, name in (('q-learning', 'Q-learning'), ('sarsa', 'SARSA')):
            systems[f'{name} {label}'] = [f'm4-{stage}-{algorithm}-seed{s}' for s in PART_B_SEEDS]
    systems['DQN Part A'] = [f'm3-dqn-seed{s}' for s in PART_A_SEEDS]
    systems['Q-learning Voting'] = ['m4-voting-q-learning']
    return systems


def scenario_labels(grid, scenario):
    def neighbors(cell):
        r, c = cell
        return [grid.is_free((r+dr, c+dc)) for dr, dc in ((-1,0),(1,0),(0,-1),(0,1))]
    def quadrant(cell):
        return ('N' if cell[0] < grid.shape[0]/2 else 'S') + ('W' if cell[1] < grid.shape[1]/2 else 'E')
    def corridor(cell):
        up, down, left, right = neighbors(cell)
        return (up and down and not left and not right) or (left and right and not up and not down)
    route = grid.path(scenario['start'], scenario['goal'])
    start, goal = quadrant(scenario['start']), quadrant(scenario['goal'])
    distance = scenario['shortest_length']
    return {'distance': 'short' if distance <= 5 else 'medium' if distance <= 10 else 'long',
            'corridor': 'reference corridor' if any(corridor(c) for c in route) else 'no reference corridor',
            'wall proximity': 'near wall' if not all(neighbors(scenario['start'])) else 'open start',
            'start region': start, 'goal region': goal,
            'region crossing': 'same quadrant' if start == goal else 'different quadrant'}


def evidence(root=ROOT):
    path = root / 'artifacts/studies/milestone5-v1/summary.json'
    return read_json(path)


def early_checkpoint(directory, root=ROOT):
    if (read_json(directory/'manifest.json').get('algorithm')=='ppo' or directory.name.startswith('m8-')):
        checkpoint=directory/'checkpoints/step-0010000.zip'
        if checkpoint.is_file() and read_json(checkpoint.with_suffix('.json'))['training_interactions']==10000:
            return checkpoint
        raise ValueError('No 10,000-interaction neural checkpoint is available for this run')
    path = root / 'artifacts/studies/milestone5-v1/early_checkpoints' / directory.name
    candidates = sorted(path.glob('*.npz')) + sorted(path.glob('*.zip'))
    if not candidates:
        raise ValueError('No archived early checkpoint for this run; choose a declared exemplar')
    return candidates[0]


def frozen_episode(base_task, directory, scenario, checkpoint=None, seed=7):
    """Run the shared environment once; never write benchmark results or learn."""
    task = task_for_run(base_task, Path(directory))
    approved = {s['id']: s for s in load_scenarios(base_task, 'validation')}
    if scenario != approved.get(scenario['id']):
        raise ValueError('Matched comparisons accept canonical validation scenarios only')
    checkpoint = Path(checkpoint) if checkpoint else viewer_checkpoint(directory)
    policy = load_policy(checkpoint, task)
    fingerprint = getattr(policy, 'state_fingerprint', None)
    before = fingerprint() if fingerprint else policy.q.tobytes()
    episode = rollout(NavigationEnv(task, [scenario]), policy, scenario, seed, 'validation')
    after = fingerprint() if fingerprint else policy.q.tobytes()
    if before != after:
        raise RuntimeError('Frozen policy changed during comparison')
    validate_replay(task, episode)
    episode['display_checkpoint'] = str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint)
    return episode
