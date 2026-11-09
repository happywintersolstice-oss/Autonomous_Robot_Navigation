"""Saved validation metrics and costs; no policy execution or final split access."""
import csv
import math
import shutil

import numpy as np

from .agents import viewer_checkpoint
from .benchmark import load_scenarios
from .core import ROOT, read_json, write_json
from .dashboard import scenario_labels
from .episodes import aggregate
from .four_way import FAMILIES, STUDY, saved_summary
from .release import sha

ANALYSIS_STUDY = 'four-rl-analysis-v1'
METRICS = ('success_rate', 'collision_rate', 'timeout_rate', 'spl', 'mean_return',
           'mean_successful_route_length')
COSTS = ('training_seconds', 'evaluation_seconds', 'mean_decision_microseconds', 'checkpoint_bytes')


def outcome_totals(rows):
    return {outcome: sum(row['metrics']['counts'][outcome] for row in rows)
            for outcome in ('success', 'collision', 'timeout')}


def stats(values):
    values = [float(value) for value in values if value is not None]
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Nonfinite analysis metric')
    return {'mean': float(np.mean(values)) if values else None,
            'sample_sd': float(np.std(values, ddof=1)) if len(values) > 1 else None,
            'contributing_seeds': len(values)}


def grouped_metrics(metrics, labels):
    """Reject missing/duplicate coverage before grouping; preserve zero-success groups."""
    ids = [row['scenario_id'] for row in metrics]
    if len(ids) != len(set(ids)) or set(ids) != set(labels):
        raise ValueError('Validation metric coverage mismatch')
    overall = aggregate(metrics)
    groups = []
    for dimension in next(iter(labels.values())):
        partition = []
        for group in sorted({label[dimension] for label in labels.values()}):
            result = aggregate([row for row in metrics if labels[row['scenario_id']][dimension] == group])
            partition.append(result)
            groups.append({'dimension': dimension, 'group': group, 'metrics': result})
        for field in ('episodes', 'interactions', 'successful_route_count'):
            if sum(row[field] for row in partition) != overall[field]:
                raise ValueError('Group totals do not reconcile')
        if any(sum(row['counts'][outcome] for row in partition) != count
               for outcome, count in overall['counts'].items()):
            raise ValueError('Group outcomes do not reconcile')
    return groups


def build_analysis(task, study, root=ROOT):
    """Publish only into a fresh study; retain old evidence and hash every input."""
    if study.exists():
        raise FileExistsError('Analysis output already exists; choose a fresh study ID')
    original = saved_summary(root)
    if original is None:
        raise ValueError('Verified four-algorithm comparison is required')
    task.check(original['signature'])
    study.mkdir(parents=True)
    write_json(study/'status.json', {'status': 'running'})
    for source, name in ((root/'src/robot_navigation/four_way_analysis.py', 'analysis.py'),
                         (root/'scripts/analyze_four_way_details.py', 'runner.py'),
                         (root/'docs/four_way_analysis_protocol.md', 'protocol.md')):
        shutil.copy2(source, study/name)
    inputs = {}

    def track(path):
        inputs[path.relative_to(root).as_posix()] = sha(path)
        return read_json(path)

    track(root/f'artifacts/studies/{STUDY}/summary.json')
    track(root/f'artifacts/studies/{STUDY}/verification.json')
    track(root/'data/scenarios/validation.json')
    scenarios = load_scenarios(task, 'validation', root)
    labels = {s['id']: scenario_labels(task.grid, s) for s in scenarios}
    historical_labels = track(root/'artifacts/studies/milestone5-v1/scenario_labels.json')
    task.check(historical_labels['signature'])
    if labels != historical_labels['scenarios']:
        raise ValueError('Predeclared robustness labels changed')
    # Persist the unchanged historical label definitions before reading run outcomes.
    write_json(study/'scenario_labels.json', historical_labels)
    result = {'status': 'verified', 'signature': task.signature, 'systems': {},
              'validation_scenarios': len(scenarios),
              'scope': 'exploratory saved development validation; four Part A baseline families',
              'label_source': 'unchanged milestone5-v1 geometry labels',
              'new_training_interactions': 0, 'new_benchmark_evaluation_interactions': 0,
              'final_test_policy_executions': 0,
              'aggregation': 'unweighted seed means and sample SD; conditional route means omit zero-success seeds'}
    group_rows, cost_rows = [], []
    for algorithm, label in FAMILIES.items():
        source = original['systems'][algorithm]
        seeds, groups = [], {}
        for row in source['seed_results']:
            directory = root/'artifacts/runs'/row['run_id']
            manifest = track(directory/'manifest.json')
            task.check(manifest['signature'])
            if (manifest['status'] != 'complete' or manifest['training_seed'] != row['seed']
                    or manifest['algorithm'] != algorithm or manifest['training_interactions'] != 200000):
                raise ValueError('Run identity or budget mismatch')
            key = algorithm.replace('-', '_')
            metrics = track(directory/f'{key}_episodes.json')
            validation = track(directory/'validation.json')[key]
            overall = aggregate(metrics)
            if any(value != row['validation'][name] or value != validation[name]
                   for name, value in overall.items()):
                raise ValueError('Saved validation aggregate mismatch')
            checkpoint = viewer_checkpoint(directory)
            metadata = track(checkpoint.with_suffix('.json'))
            task.check(metadata['signature'])
            inputs[checkpoint.relative_to(root).as_posix()] = sha(checkpoint)
            if inputs[checkpoint.relative_to(root).as_posix()] != metadata['sha256']:
                raise ValueError('Selected checkpoint checksum mismatch')
            costs = {'training_seconds': manifest['training_seconds'],
                     'evaluation_seconds': validation['evaluation_seconds'],
                     'mean_decision_microseconds': validation['mean_decision_microseconds'],
                     'checkpoint_bytes': checkpoint.stat().st_size}
            if costs['checkpoint_bytes'] != manifest['checkpoint_bytes']:
                raise ValueError('Selected checkpoint size mismatch')
            seed = {'run_id': row['run_id'], 'seed': row['seed'], 'metrics': overall,
                    'costs': costs, 'selected_interactions': row['selected_interactions']}
            seeds.append(seed)
            cost_rows.append({'algorithm': algorithm, 'seed': row['seed'], 'run_id': row['run_id'],
                              **costs, **{name: overall[name] for name in METRICS},
                              'successful_route_count': overall['successful_route_count']})
            for group in grouped_metrics(metrics, labels):
                identity = group['dimension'], group['group']
                groups.setdefault(identity, []).append({'run_id': row['run_id'], 'seed': row['seed'],
                                                        'metrics': group['metrics']})
                group_rows.append({'algorithm': algorithm, 'run_id': row['run_id'], 'seed': row['seed'],
                                   'dimension': identity[0], 'group': identity[1],
                                   **{k: v for k, v in group['metrics'].items() if k != 'counts'},
                                   **group['metrics']['counts']})
        result['systems'][algorithm] = {
            'label': label, 'seed_count': len(seeds), 'seed_results': seeds,
            'metrics': {name: stats([s['metrics'][name] for s in seeds]) for name in METRICS},
            'costs': {name: stats([s['costs'][name] for s in seeds]) for name in COSTS},
            'counts': outcome_totals(seeds),
            'groups': [{'dimension': d, 'group': g, 'scenarios_per_seed': rows[0]['metrics']['episodes'],
                        'seed_results': rows,
                        'metrics': {name: stats([s['metrics'][name] for s in rows]) for name in METRICS},
                        'counts': outcome_totals(rows)}
                       for (d, g), rows in groups.items()]}
        # Existing results must remain identical after adding conditional metrics/costs.
        for name in source['metrics']:
            if result['systems'][algorithm]['metrics'][name]['mean'] != source['metrics'][name]['mean']:
                raise ValueError('Original four-way mean changed')
    for name, rows in (('groups.csv', group_rows), ('costs.csv', cost_rows)):
        with (study/name).open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_json(study/'summary.json', result)
    if any(sha(root/path) != value for path, value in inputs.items()):
        raise ValueError('Analysis inputs changed during execution')
    write_json(study/'inputs.json', inputs)
    outputs = {name: sha(study/name) for name in ('summary.json', 'groups.csv', 'costs.csv', 'scenario_labels.json')}
    write_json(study/'verification.json', {
        'status': 'passed', 'saved_episode_metrics_reconciled': len(scenarios)*20,
        'group_seed_rows': len(group_rows), 'input_files_verified': len(inputs),
        'input_files_unchanged': True, 'output_hashes': outputs,
        'new_training_interactions': 0, 'new_benchmark_evaluation_interactions': 0,
        'final_test_policy_executions': 0, 'historical_means_unchanged': True})
    write_json(study/'status.json', {'status': 'complete'})
    return result


def saved_analysis(root=ROOT):
    study = root/'artifacts/studies'/ANALYSIS_STUDY
    if not (study/'status.json').is_file():
        return None
    if read_json(study/'status.json')['status'] != 'complete':
        return None
    verification = read_json(study/'verification.json')
    if verification['status'] != 'passed' or any(sha(study/name) != value
            for name, value in verification['output_hashes'].items()):
        raise ValueError('Four-algorithm analysis verification failed')
    return read_json(study/'summary.json')
