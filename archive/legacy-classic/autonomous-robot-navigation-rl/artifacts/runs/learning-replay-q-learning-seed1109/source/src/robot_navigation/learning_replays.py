"""New, separately labeled training demonstrations; no historical reconstruction."""
import csv

import numpy as np

from .core import load_task, read_json
from .dashboard import training_episode, training_paths
from .release import sha
from .training_traces import trace_limit
from .workflow import source_hash


FAMILIES = ('q-learning', 'sarsa', 'dqn', 'ppo')


def demo_configs(plan, root):
    if (list(plan['sources']) != list(FAMILIES) or type(plan['seed']) is not int
            or plan['seed'] < 0 or plan['interactions'] != 200000):
        raise ValueError('Recorded demos require all four families and the declared 200k budget')
    task = load_task(root)
    configs = []
    for algorithm, source in plan['sources'].items():
        saved = read_json(root/'artifacts/runs'/source/'configuration.json')
        task.check(saved['signature'])
        config = saved['training'].copy()
        if config['algorithm'] != algorithm or config['interactions'] != plan['interactions']:
            raise ValueError('Baseline configuration identity/budget mismatch')
        config.update(run_id=f'learning-replay-{algorithm}-seed{plan["seed"]}', seed=plan['seed'],
                      stage='learning-replay-demo', training_trace_episodes=plan['training_trace_episodes'],
                      training_trace_starts=plan['training_trace_starts'], log_progress=True,
                      max_wall_seconds=plan['limits']['run_wall_seconds'])
        trace_limit(config)
        configs.append(config)
    return configs


def audit_demo(directory, root):
    task = load_task(root)
    manifest = read_json(directory/'manifest.json')
    config = read_json(directory/'configuration.json')['training']
    if (manifest['status'] != 'complete' or manifest['stage'] != 'learning-replay-demo'
            or manifest['final_test_evaluated'] or manifest['training_interactions'] != 200000
            or manifest['training_seed'] != config['seed']):
        raise ValueError('Demo completion, identity or budget mismatch')
    task.check(manifest['signature'])
    if source_hash(directory/'source') != manifest['source_hash']:
        raise ValueError('Recorded demo source snapshot changed')
    with (directory/'training.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    partial = read_json(directory/'pilot.json')['incomplete_budget_cutoff_episode']
    interactions = sum(int(row['steps']) for row in rows) + (partial['steps'] if partial else 0)
    with np.load(directory/'visitation.npz', allow_pickle=False) as data:
        visits = int(data['state_action_counts'].sum())
    if interactions != 200000 or visits != interactions:
        raise ValueError('Demo training ledger does not reconcile')
    trace = manifest['training_traces']
    starts = config['training_trace_starts']
    per_window = config['training_trace_episodes'] // len(starts)
    if (trace['window_starts'] != starts or trace['episodes_per_window'] != per_window
            or trace['historical_training_reconstructed'] or trace['extra_environment_interactions'] != 0):
        raise ValueError('Recording schedule metadata mismatch')
    expected = []
    for index, start in enumerate(starts):
        stop = starts[index+1] if index+1 < len(starts) else interactions
        candidates = [row for row in rows if start <= int(row['end_interaction'])-int(row['steps']) < stop]
        if len(candidates) < per_window:
            raise ValueError('A declared training window has too few completed episodes')
        expected.extend((index, row) for row in candidates[:per_window])
    paths = training_paths(directory)
    if len(paths) != config['training_trace_episodes'] or len(set(paths)) != len(paths):
        raise ValueError('Missing or duplicated training recordings')
    records = []
    for path, (window, row) in zip(paths, expected, strict=True):
        episode = training_episode(task, directory, path, root)
        start = int(row['end_interaction'])-int(row['steps'])
        if (not episode['complete'] or episode['recording_window_index'] != window
                or episode['recording_window_start_interaction'] != starts[window]
                or episode['training_start_interaction'] != start
                or episode['training_end_interaction'] != int(row['end_interaction'])
                or episode['scenario']['id'] != row['scenario_id']
                or len(episode['transitions']) != int(row['steps'])
                or episode['metrics']['return'] != float(row['return'])
                or episode['metrics']['outcome'] != row['outcome']):
            raise ValueError('Recording is not the declared chronological training episode')
        records.append({'path':path.relative_to(directory).as_posix(), 'sha256':sha(path),
                        'window_start':starts[window], 'training_start':start,
                        'training_end':episode['training_end_interaction'],
                        'actions':len(episode['transitions']), 'outcome':episode['metrics']['outcome']})
    return {'run_id':directory.name, 'algorithm':manifest['algorithm'], 'seed':manifest['training_seed'],
            'training_interactions':interactions, 'recordings':records,
            'recorded_actions':sum(row['actions'] for row in records),
            'scope':'new recorded training demo; excluded from original benchmarks and final evaluation',
            'historical_training_reconstructed':False, 'final_test_evaluated':False}
