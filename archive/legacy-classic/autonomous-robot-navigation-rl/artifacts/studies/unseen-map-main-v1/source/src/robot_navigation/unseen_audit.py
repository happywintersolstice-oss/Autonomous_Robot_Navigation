"""Saved-action and provenance audit for the development-only unseen-map pilot."""
import csv
from copy import deepcopy
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .agents import BFSPolicy, RandomPolicy
from .core import ROOT, read_json, write_json
from .episodes import aggregate, validate_replay
from .part_a import checkpoint_rank
from .release import sha, verify_freeze
from .representation import validate_training_prefix
from .unseen_agents import load_map_policy, map_adapter
from .unseen_benchmark import MapBenchmark, OBSERVATION, audit_benchmark
from .unseen_study import STUDY
from .workflow import source_hash


def replay_task(benchmark, episode):
    if (episode.get('benchmark_signature') != benchmark.signature
            or episode.get('policy_observation_version') != OBSERVATION):
        raise ValueError('Replay benchmark/input contract mismatch')
    split = episode['layout_split']
    layouts = benchmark.layouts(split)
    layout = next((m for m in layouts if m['map_id']==episode['layout_id']), None)
    if layout is None or episode['scenario'] not in layout['scenarios']:
        raise ValueError('Replay layout/scenario is not declared in its development split')
    task = benchmark.task(layout)
    if episode['mode'] == 'training':
        if split != 'train':
            raise ValueError('Training recording has the wrong layout split')
        # Earlier captures retained the same info dictionary returned to SB3.
        # Its vector wrapper appended this flag after collection. Check its
        # meaning before validating the original core fields, without rewriting
        # or changing any saved action, observation, reward or terminal flag.
        canonical = deepcopy(episode)
        for transition in canonical['transitions']:
            info = transition['info']
            if 'TimeLimit.truncated' in info:
                expected = bool(transition['truncated'] and not transition['terminated'])
                if info['TimeLimit.truncated'] is not expected:
                    raise ValueError('Stored SB3 timeout metadata disagrees with the transition flags')
                del info['TimeLimit.truncated']
        validate_training_prefix(task, canonical)
        if any(t['scores'] is not None or t.get('decision') is not None for t in episode['transitions']):
            raise ValueError('Training recordings cannot contain substituted checkpoint diagnostics')
    else:
        if episode['mode'] != 'validation':
            raise ValueError('Unseen pilot replays accept development validation/training only')
        validate_replay(task, episode)
    return task


def load_recording(benchmark, directory, item):
    directory = Path(directory)
    manifest = read_json(directory/'manifest.json')
    if (manifest['status'] != 'complete' or manifest['signature'] != benchmark.signature
            or item not in [*manifest['replays'], *manifest['training']['training_traces']]):
        raise ValueError('Recording is not registered by a completed compatible pilot run')
    path = (directory/item['path']).resolve()
    if not path.is_relative_to(directory.resolve()) or sha(path) != item['sha256']:
        raise ValueError('Recording path/checksum mismatch')
    episode = read_json(path)
    if item['layout_id'] != episode['layout_id']:
        raise ValueError('Recording layout identity mismatch')
    if episode.get('policy_feature_version') != manifest.get('representation'):
        raise ValueError('Recording feature version mismatch')
    return replay_task(benchmark, episode), episode


def _audit_validation(benchmark, path, policy, split, expected):
    cases = benchmark.cases(split, monitoring=split=='train')
    before = policy.state_fingerprint()
    rows, actions, cache = [], 0, {}
    with gzip.open(path, 'rt', encoding='utf-8') as file:
        for (layout, scenario), line in zip(cases, file, strict=True):
            episode = json.loads(line)
            if episode['scenario'] != scenario or episode['layout_split'] != split:
                raise ValueError('Stored evaluation cases are not in declared order')
            task = replay_task(benchmark, episode)
            adapter = map_adapter(policy, task.grid)
            if episode['policy_id'] != policy.policy_id:
                raise ValueError('Stored replay does not identify the selected checkpoint')
            if episode.get('policy_feature_version') != getattr(policy,'feature_version',None):
                raise ValueError('Stored replay feature version differs from the checkpoint')
            for transition in episode['transitions']:
                key = (layout['map_id'], tuple(transition['observation']))
                if key not in cache:
                    if policy.algorithm == 'ppo':
                        cache[key] = adapter.decision(transition['observation'])
                    else:
                        action, scores = adapter.action_and_scores(transition['observation'])
                        cache[key] = {'action':action, 'scores':scores.tolist()}
                decision = cache[key]
                if transition['action'] != decision['action']:
                    raise ValueError('Reloaded checkpoint action differs from stored action')
                if policy.algorithm == 'ppo':
                    if transition.get('decision') != decision:
                        raise ValueError('Reloaded PPO diagnostics differ from the stored decision')
                elif not np.array_equal(transition['scores'], decision['scores']):
                    raise ValueError('Reloaded Q scores differ from the stored decision')
                actions += 1
            rows.append(episode['metrics'])
    observed = aggregate(rows)
    if any(observed[k] != expected[k] for k in observed):
        raise ValueError('Stored evaluation summary does not reconcile with its episodes')
    if policy.state_fingerprint() != before:
        raise ValueError('Inference audit changed learned state')
    return actions


def audit_run(benchmark,study,system,settings,source_digest):
    directory = study/'runs'/system['run_id']
    run = read_json(directory/'manifest.json')
    family_config = settings['tabular' if system['algorithm'] in ('q-learning','sarsa') else system['algorithm']]
    expected_config = {**family_config, 'seed':settings['seed'], 'algorithm':system['algorithm'],
                       'interactions':settings['interactions'], 'monitor_every':settings['monitor_every'],
                       'run_wall_seconds':settings['run_wall_seconds']}
    if settings.get('representation'):
        expected_config['representation'] = settings['representation']
    if (run['status'] != 'complete' or run['source_hash'] != source_digest
            or run['signature'] != benchmark.signature or run['training_seed'] != settings['seed']
            or run['training_interactions'] != settings['interactions']
            or sum(run['training']['map_interactions'].values()) != settings['interactions']
            or run['configuration'] != expected_config):
        raise ValueError('Run provenance or exact training budget mismatch')
    with (directory/'training.csv').open(newline='', encoding='utf-8') as file:
        ledger = list(csv.DictReader(file))
    if (sum(int(r['steps']) for r in ledger)+run['training']['cutoff_steps'] != settings['interactions']
            or [int(r['episode']) for r in ledger] != list(range(1,len(ledger)+1))):
        raise ValueError('Training ledger does not reconcile with its budget')
    previous = 0
    for row in ledger:
        if int(row['start_interaction']) != previous or int(row['end_interaction'])-previous != int(row['steps']):
            raise ValueError('Training ledger has a gap or overlapping episode')
        previous = int(row['end_interaction'])
    monitoring = read_json(directory/'monitoring.json')
    if [m['training_interactions'] for m in monitoring] != list(
            range(settings['monitor_every'],settings['interactions']+1,settings['monitor_every'])):
        raise ValueError('Monitoring/checkpoint opportunity mismatch')
    selected_step = max(monitoring, key=lambda m:checkpoint_rank(m,m['training_interactions']))['training_interactions']
    if selected_step != run['selected_interactions']:
        raise ValueError('Selected checkpoint violates the predeclared rule')
    selected_path = directory/run['viewer_checkpoint']
    policy = load_map_policy(selected_path, benchmark)
    periodic = directory/f'checkpoints/step-{selected_step:07d}{selected_path.suffix}'
    if sha(periodic) != sha(selected_path):
        raise ValueError('Selected checkpoint differs from its registered periodic snapshot')
    for monitor in monitoring:
        path = directory/f'checkpoints/step-{monitor["training_interactions"]:07d}{selected_path.suffix}'
        load_map_policy(path, benchmark)
    validation = read_json(directory/'validation.json')
    familiar = read_json(directory/'familiar.json')
    if validation != system['validation'] or familiar != system['familiar']:
        raise ValueError('Study and run summaries differ')
    actions = _audit_validation(benchmark, directory/'validation.jsonl.gz', policy, 'validation', validation)
    familiar_actions = _audit_validation(benchmark, directory/'familiar.jsonl.gz', policy, 'train', familiar)
    trace_actions = 0
    traces = run['training']['training_traces']
    if [t['window_start'] for t in traces] != settings['trace_windows']:
        raise ValueError('Registered training windows were not recorded')
    for item in traces:
        _, episode = load_recording(benchmark, directory, item)
        start = episode['training_start_interaction']
        if not item['window_start'] <= start < item['window_start']+benchmark.record['environment']['max_steps']:
            raise ValueError('Training trace is outside its declared window')
        if episode['complete']:
            row = next((r for r in ledger if int(r['start_interaction']) == start), None)
            if (row is None or row['layout_id'] != episode['layout_id']
                    or row['scenario_id'] != episode['scenario']['id']
                    or int(row['steps']) != len(episode['transitions'])
                    or float(row['return']) != episode['metrics']['return']
                    or row['outcome'] != episode['metrics']['outcome']):
                raise ValueError('Training trace differs from its chronological ledger row')
        trace_actions += len(episode['transitions'])
    for item in run['replays']:
        load_recording(benchmark, directory, item)
    return {'run_id':run['run_id'], 'validation_actions':actions,
                          'familiar_actions':familiar_actions, 'training_trace_actions':trace_actions,
                          'training_traces':len(traces), 'inference_unchanged':True}


def verify(root=ROOT):
    started = perf_counter()
    benchmark = MapBenchmark(root)
    geometry = audit_benchmark(root)
    study = root/'artifacts/studies'/STUDY
    manifest = read_json(study/'manifest.json')
    summary = read_json(study/'summary.json')
    if (manifest['signature'] != benchmark.signature or summary['signature'] != benchmark.signature
            or source_hash(study/'source') != manifest['source_hash']
            or sha(study/'protocol.md') != manifest['protocol_hash']):
        raise ValueError('Pilot source or benchmark identity mismatch')
    settings = benchmark.config['pilot']
    if [s['algorithm'] for s in summary['systems']] != settings['algorithms']:
        raise ValueError('Four-family pilot support mismatch')
    verified_runs = []
    for system in summary['systems']:
        verified_runs.append(audit_run(benchmark,study,system,settings,manifest['source_hash']))
    references = summary['references']
    if references['bfs']['success_rate'] != 1 or references['bfs']['spl'] != 1:
        raise ValueError('BFS did not solve every reachable validation route optimally')
    reference_actions = 0
    for name in ('bfs','random'):
        rows = []
        random_policy = RandomPolicy(settings['seed'])
        with gzip.open(study/f'{name}.jsonl.gz','rt',encoding='utf-8') as file:
            for (layout,scenario), line in zip(benchmark.cases('validation'),file,strict=True):
                episode = json.loads(line)
                task = replay_task(benchmark,episode)
                if episode['scenario'] != scenario or episode['layout_id'] != layout['map_id']:
                    raise ValueError('Reference evaluation support mismatch')
                reference = BFSPolicy(task.grid) if name=='bfs' else random_policy
                if any(reference.act(t['observation']) != t['action'] for t in episode['transitions']):
                    raise ValueError('Stored reference actions differ from the registered policy')
                rows.append(episode['metrics'])
                reference_actions += len(episode['transitions'])
        observed = aggregate(rows)
        if any(observed[k] != references[name][k] for k in observed):
            raise ValueError('Reference outcomes do not reconcile')
    old_study = root/'artifacts/studies/final-v1'
    if old_study.exists():
        verify_freeze(old_study,read_json(old_study/'seal.json')['freeze_digest'],root=old_study/'snapshot')
    result = {'status':'passed', 'signature':benchmark.signature, 'benchmark':geometry,
              'runs':verified_runs, 'reference_actions':reference_actions,
              'final_policy_executions':0, 'verification_seconds':perf_counter()-started,
              'verification_source_hash':source_hash(root), 'training_source_hash':manifest['source_hash'],
              'scope':'one-seed pilot; no independent final evaluation or main-study claim'}
    write_json(study/'verification.json',result)
    summary['status'] = 'verified'
    manifest['status'] = 'verified'
    write_json(study/'summary.json',summary)
    write_json(study/'manifest.json',manifest)
    return result
