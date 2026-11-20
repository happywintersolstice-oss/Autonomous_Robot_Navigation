"""Phase 5A only: registered, single-seed, hard-bounded learning feasibility."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np
import yaml

from .core import ROOT, GridMap, digest, read_json, write_json
from .release import sha
from .partial_agents import ALGORITHMS, LinearPolicy, load_policy
from .partial_baselines import AlwaysWait, RandomPolicy, OnlinePlanner
from .partial_benchmark import DevelopmentBenchmark, path_without
from .partial_commands import read_records, require_gates, write_records
from .partial_contract import configuration, signature
from .partial_dynamic_env import PartialDynamicEnv
from .partial_evaluation import code_hashes, episode_metrics, validate_replay
from .partial_feasibility_fixtures import FIXTURES, cases, suite, ScriptedMechanics, hindsight_actions
from .partial_pilot_diagnostics import (decision_values, evaluation_summary, failure_examples, model_diagnostics,
    preserve_random_state, representation_probe, reward_sanity, training_summary)
from .partial_pilot_ledger import Budget, PilotLimit, ActionStream, PilotRecorder, ContinuousLinear
from .partial_training import validate_training_trace

STUDY = 'feasibility-pilot-v1'


def settings(root=ROOT):
    result = yaml.safe_load((root/'configs/partial_feasibility_v1.yaml').read_text(encoding='utf-8'))
    if (result['study_id'] != STUDY or result['fixture_version'] != FIXTURES
            or result['algorithms'] != list(ALGORITHMS) or result['primary_seed'] != 23001
            or result['confirmation_seeds'] or result['per_algorithm_actions'] != 10000
            or result['total_training_actions'] != 40000 or result['evaluation_action_ceiling'] != 100000
            or result['suite_wall_seconds'] != 1200 or result['per_algorithm_wall_seconds'] != 240
            or result['open_checkpoints'] != [2000, 6000, 10000] or result['later_stage_actions'] != 2000):
        raise ValueError('The registered pilot envelope changed; no unbounded or additional-seed study is allowed')
    task = configuration(root)
    model = result['model']
    if (any(n % model['n_steps'] for n in (2000, 4000, 10000))
            or model['n_steps'] % model['batch_size'] or any(n % model['train_freq'] for n in (2000, 4000, 10000))):
        raise ValueError('Neural rollout/training frequencies must divide registered action blocks exactly')
    if any(task[k] != v for k, v in {'max_steps': 200, 'step_reward': -.01, 'success_bonus': 10.,
                                    'collision_penalty': -5., 'extra_timeout_penalty': 0.}.items()):
        raise ValueError('Phase 5A baseline task/rewards must remain unchanged')
    return result


def location(root=ROOT):
    return root/'artifacts/partial-dynamic-navigation-v1'/STUDY


def pilot_inputs(root=ROOT):
    return {'code': code_hashes(), 'config': sha(root/'configs/partial_feasibility_v1.yaml'),
        'baseline_config': sha(root/'configs/partial_dynamic.yaml'),
        'protocol': sha(root/'docs/partial_feasibility_protocol.md'),
        'brief': sha(root.parent/'CODEX_PHASE5_LEARNING_FEASIBILITY_PILOT.md'),
        'fixtures_hash': digest(suite()), 'gate_inputs': require_gates(root)['inputs']}


def register(root=ROOT):
    c = settings(root); inputs = pilot_inputs(root)
    dataset = DevelopmentBenchmark(root); dataset.audit()
    preflight = root/'.runtime/phase5a-preflight-tests.xml'
    import xml.etree.ElementTree as ET
    result = ET.parse(preflight).getroot().find('testsuite')
    if result is None or any(int(result.attrib[k]) for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Pre-flight full regressions did not pass')
    validation = root/'.runtime/phase5a-final-tests.xml'
    final_suite = ET.parse(validation).getroot().find('testsuite')
    if final_suite is None or any(int(final_suite.attrib[k]) for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Final implementation regressions did not pass')
    inventory = read_json(root/'.runtime/phase5a-base-inventory.json')
    if any(sha(root.parent/name) != value for name, value in inventory['tracked_files'].items()):
        raise ValueError('A pre-existing tracked file changed before the pilot')
    folder = location(root)
    folder.mkdir(parents=True, exist_ok=False)
    snapshot = folder/'snapshot'
    shutil.copytree(root/'src/robot_navigation', snapshot/'src/robot_navigation',
                    ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    for name in ('configs/partial_feasibility_v1.yaml', 'configs/partial_dynamic.yaml',
                 'docs/partial_feasibility_protocol.md', 'requirements-lock.txt', 'pyproject.toml'):
        target = snapshot/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(root/name, target)
    for path in (root/'developer/tests').glob('test_partial*.py'):
        target = snapshot/'developer/tests'/path.name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    shutil.copytree(dataset.folder, snapshot/'data/partial/partial-dynamic-navigation-v1')
    shutil.copy2(preflight, folder/'preflight-regressions.xml')
    shutil.copy2(validation, folder/'implementation-regressions.xml')
    (snapshot/'briefs').mkdir()
    shutil.copy2(root.parent/'CODEX_PHASE5_LEARNING_FEASIBILITY_PILOT.md', snapshot/'briefs/CODEX_PHASE5_LEARNING_FEASIBILITY_PILOT.md')
    shutil.copy2(root/'.runtime/partial-validation.log', folder/'validation.log')
    shutil.copy2(root/'.runtime/phase5a-base-inventory.json', folder/'base-inventory.json')
    shutil.copy2(root/'.runtime/phase5a-manual-trace-v1.json', folder/'preflight-manual-trace.json')
    write_json(folder/'validation-gates.json', require_gates(root))
    protocol = {'study_id': STUDY, 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'settings': c, 'task_signature': signature(configuration(root)), 'inputs': inputs,
        'fixture_entries': suite(), 'evaluation_case_ids': [v['id'] for v in cases()],
        'training_cases_by_phase': {phase: [v['id'] for v in cases(categories, 'train')]
                                  for phase, categories in (('A', 'A'), ('B', 'B'), ('CD', 'CD'), ('E', 'E'))},
        'composition_case_ids': [v['id'] for split in ('train', 'validation') for v in dataset.scenarios(split, True)],
        'checkpoint_rule': 'fixed stage boundaries and declared gating; retain every checkpoint and failure',
        'confirmation': 'not authorized/exposed by this runner; one primary seed only',
        'runtime': {'python': platform.python_version(), 'platform': platform.platform(), 'processor': platform.processor(),
                    'logical_cpus': os.cpu_count(), 'torch_threads': 1, 'python_executable': sys.executable,
                    'working_directory': str(root), 'PYTHONPATH': os.environ.get('PYTHONPATH'),
                    'dependencies': {n: version(n) for n in ('numpy', 'gymnasium', 'stable-baselines3', 'torch')}},
        'git': {'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip(),
                'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'status_before_registration': subprocess.check_output(['git', 'status', '--porcelain=v1', '-uall'], cwd=root, text=True)},
        'commands': [f'{sys.executable} -m robot_navigation.partial_feasibility {action}'
                     for action in ('register', 'run', 'audit', 'report')]}
    write_json(folder/'phase5a_pilot_protocol.json', protocol)
    write_json(folder/'protocol-seal.json', {'protocol_sha256': sha(folder/'phase5a_pilot_protocol.json')})
    write_json(folder/'snapshot-hashes.json', {p.relative_to(snapshot).as_posix(): sha(p) for p in snapshot.rglob('*') if p.is_file()})
    write_json(folder/'status.json', {'status': 'registered', 'training_actions': 0})
    return {'status': 'registered', 'protocol_sha256': sha(folder/'phase5a_pilot_protocol.json')}


def checked_protocol(root=ROOT):
    folder = location(root)
    if sha(folder/'phase5a_pilot_protocol.json') != read_json(folder/'protocol-seal.json')['protocol_sha256']:
        raise ValueError('Registered pilot protocol changed')
    record = read_json(folder/'phase5a_pilot_protocol.json')
    if record['inputs'] != pilot_inputs(root): raise ValueError('Pilot source/config/fixture/test gates are stale')
    for name, value in read_json(folder/'snapshot-hashes.json').items():
        if sha(folder/'snapshot'/name) != value: raise ValueError('Pilot snapshot changed')
    return record


class EpisodeAbort(RuntimeError):
    def __init__(self, record, cause):
        super().__init__(str(cause)); self.record, self.cause = record, cause


def rollout(case, policy, task, budget):
    """Bounded equivalent of the existing evaluator, retaining interrupted prefixes."""
    if case.get('split') not in ('manual', 'train', 'validation'): raise ValueError('No final/test evaluation')
    env = PartialDynamicEnv([case], task); observation, _ = env.reset(seed=0)
    fingerprint = getattr(policy, 'state_fingerprint', lambda: None)
    before = fingerprint()
    record = {'schema_version': 1, 'task_signature': env.signature, 'configuration': task,
        'scenario': deepcopy(case), 'scenario_hash': digest(case), 'evaluation_seed': 0,
        'policy_id': policy.policy_id, 'algorithm': policy.algorithm, 'training_seed': getattr(policy, 'training_seed', None),
        'checkpoint_sha256': getattr(policy, 'checkpoint_sha256', None), 'parameter_fingerprint': before,
        'code_hashes': code_hashes(), 'initial_observation': observation.tolist(),
        'initial_snapshot': env.diagnostic_snapshot(), 'transitions': [], 'inference_seconds': 0.}
    try:
        while True:
            budget.evaluation_action(policy.algorithm)
            started = perf_counter()
            decision = decision_values(policy, observation.copy())
            action = policy.act(observation.copy())
            record['inference_seconds'] += perf_counter()-started
            nxt, reward, terminal, cutoff, info = env.step(action)
            record['transitions'].append({'observation': observation.tolist(), 'action': int(action), 'decision': decision,
                'reward': reward, 'next_observation': nxt.tolist(), 'terminated': terminal, 'truncated': cutoff,
                'info': info, 'snapshot': env.diagnostic_snapshot()})
            observation = nxt
            if terminal or cutoff: break
        if before != fingerprint(): raise ValueError('Evaluation changed learned state')
        record['metrics'] = episode_metrics(record); record['completion'] = 'complete'
        if case.get('fixture_category') == 'D':
            optimum = hindsight_actions(case)
            record['hindsight'] = {'clairvoyant_minimum_actions': optimum,
                'success_action_excess': len(record['transitions'])-optimum if record['metrics']['outcome'] == 'success' else None,
                'information_scope': 'Privileged post-hoc time-expanded search; not supplied to any operational policy'}
        return record
    except BaseException as exc:
        record.update(completion='incomplete', metrics=None, exception=f'{type(exc).__name__}: {exc}')
        raise EpisodeAbort(record, exc) from exc


def basic_gate(current, initial, c):
    return (current['counts']['success'] >= c['basic_min_successes']
            and current['counts']['success']-initial['counts']['success'] >= c['basic_min_gain'])


def grouped(records):
    return {category: evaluation_summary([r for r in records if r['scenario']['fixture_category'] == category])
            for category in sorted({r['scenario']['fixture_category'] for r in records})}


def evaluation(cases_to_run, policy, task, budget, path):
    records = []
    try:
        with gzip.open(path, 'xt', encoding='utf-8') as output:
            for case in cases_to_run:
                try: record = rollout(case, policy, task, budget)
                except EpisodeAbort as exc:
                    output.write(json.dumps(exc.record, allow_nan=False)+'\n'); raise
                validate_replay(record)
                output.write(json.dumps(record, allow_nan=False)+'\n'); output.flush()
                records.append(record)
    except BaseException:
        raise
    return records


def controls(folder, task, c, budget):
    folder.mkdir()
    summaries = {}
    entries = [e for e in suite() if e['case']['fixture_split'] == 'evaluation']
    for name in ('always-wait', 'random', 'online-planner', 'scripted-mechanics'):
        records = []
        with gzip.open(folder/f'{name}.jsonl.gz', 'xt', encoding='utf-8') as output:
            for i, entry in enumerate(entries):
                policy = (AlwaysWait() if name == 'always-wait' else RandomPolicy(c['random_control_seed']+i)
                          if name == 'random' else OnlinePlanner() if name == 'online-planner'
                          else ScriptedMechanics(entry['script']))
                try: record = rollout(entry['case'], policy, task, budget)
                except EpisodeAbort as exc:
                    output.write(json.dumps(exc.record)+'\n'); raise
                validate_replay(record); output.write(json.dumps(record, allow_nan=False)+'\n'); records.append(record)
                if name == 'always-wait' and (record['metrics']['outcome'] != 'timeout' or record['metrics']['wait_count'] != 200):
                    raise ValueError('Negative control failed')
                if name == 'scripted-mechanics' and record['metrics']['outcome'] != 'success':
                    raise ValueError('A handwritten mechanics case failed')
        summaries[name] = grouped(records)
    write_json(folder/'summary.json', summaries)
    return summaries


def _resource_usage():
    result = {'cpu_user_seconds': os.times().user, 'cpu_system_seconds': os.times().system}
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            class Memory(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)]+[(k, ctypes.c_size_t) for k in
                    ('peak_working', 'working', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]
            value = Memory(); value.cb = ctypes.sizeof(value)
            ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            function = ctypes.windll.psapi.GetProcessMemoryInfo
            function.argtypes = (wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD)
            if not function(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(value), value.cb):
                raise OSError('Process memory measurement failed')
            result.update(working_set_bytes=int(value.working), peak_working_set_bytes=int(value.peak_working))
        else:
            import resource
            result['peak_rss_platform_units'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, AttributeError): result['memory_measurement'] = 'unavailable'
    return result


def checkpoint(folder, phase, policy, stream, categories, task, c, budget, updates):
    count = budget.training[policy.algorithm]
    target = folder/f'{phase}-{count:05}'; target.mkdir()
    path = target/('policy.npz' if policy.algorithm in ('q-learning', 'sarsa') else 'policy.zip')
    policy.save(path, {'study_id': STUDY, 'phase': phase, 'training_interactions': count,
                      'master_protocol_sha256': sha(folder.parent/'phase5a_pilot_protocol.json')})
    before = policy.state_fingerprint()
    with preserve_random_state():
        loaded = load_policy(path, task)
        if loaded.state_fingerprint() != before: raise ValueError('Checkpoint reload state mismatch')
        records = evaluation(cases(categories), loaded, task, budget, target/'evaluation.jsonl.gz')
        probes = []
        for case in cases('AB'):
            obs, _ = PartialDynamicEnv([case], task).reset(seed=0)
            probes.append({'case_id': case['id'], 'observation': obs.tolist(), 'decision': decision_values(loaded, obs)})
    if policy.state_fingerprint() != before: raise ValueError('Evaluation perturbed active learner')
    training = training_summary(read_records(stream.path)) if stream.path.exists() else None
    if stream.path.exists():
        if validate_training_trace(read_records(stream.path), task) != count:
            raise ValueError('Actual training actions do not match reserved count')
    summary = {'phase': phase, 'training_actions': count, 'checkpoint': path.name, 'checkpoint_sha256': sha(path),
        'parameter_fingerprint': before, 'checkpoint_bytes': path.stat().st_size, 'training': training,
        'evaluation': grouped(records), 'model': model_diagnostics(policy), 'linear_updates': updates,
        'resource_usage': _resource_usage(), 'training_prefix_bytes': stream.path.stat().st_size if stream.path.exists() else 0,
        'training_prefix_sha256': sha(stream.path) if stream.path.exists() else None}
    if not summary['model']['parameters_finite']: raise FloatingPointError('Nonfinite learned parameters')
    write_json(target/'summary.json', summary)
    write_json(target/'value-probes.json', probes)
    write_json(target/'examples.json', {r['scenario']['id']: failure_examples(r) for r in records})
    if sum(p.stat().st_size for p in folder.parent.rglob('*') if p.is_file()) > c['artifact_byte_ceiling']:
        raise PilotLimit('Artifact byte ceiling at checkpoint boundary')
    budget.check(policy.algorithm)
    print(f'{policy.algorithm} {phase} {count:,}: '+', '.join(f"{k} {v['counts']} Wait={v['wait_fraction']:.3f}" for k, v in summary['evaluation'].items()), flush=True)
    return summary


def train_algorithm(algorithm, folder, task, c, budget):
    folder.mkdir()
    write_json(folder/'phase5a_pilot_protocol.json', {'algorithm': algorithm, 'seed': c['primary_seed'],
        'master_protocol_sha256': sha(folder.parent/'phase5a_pilot_protocol.json'), 'model_settings': c['model'],
        'per_algorithm_ceiling': c['per_algorithm_actions'], 'wall_seconds': c['per_algorithm_wall_seconds'],
        'attempt': 1, 'selection': 'Every registered stage checkpoint retained; no retries or seed selection'})
    budget.start_algorithm(algorithm)
    stream = ActionStream(folder/'training-actions.jsonl.gz')
    def environment(phase):
        return PilotRecorder(cases(phase, 'train'), task, budget, algorithm, stream, phase,
                             c['primary_seed']+c['stage_reset_seed_offsets'][phase])
    env = environment('A')
    if algorithm in ('q-learning', 'sarsa'):
        policy = LinearPolicy(algorithm, task, c['primary_seed'])
        trainer = ContinuousLinear(policy, budget, c); trainer.set_env(env)
    else:
        from .partial_neural import build_neural
        policy = build_neural(env, algorithm, c['model'], c['primary_seed']); trainer = None
        env.context = lambda: {'epsilon_used': float(policy.model.exploration_rate)} if algorithm == 'dqn' else {'sampling': 'PPO stochastic training action'}
    policy.policy_id = f'{algorithm}/phase5a-primary-seed{c["primary_seed"]}'
    summaries, phase = [], 'A'
    def fit(actions, new_phase=None):
        nonlocal env, phase
        if actions <= 0 or budget.training[algorithm]+actions > c['per_algorithm_actions']:
            raise PilotLimit('Requested block exceeds the registered action ceiling')
        stream.begin()
        try:
            if new_phase is not None and new_phase != phase:
                phase = new_phase; env = environment(phase)
                if trainer: trainer.set_env(env)
                else:
                    policy.model.set_env(env, force_reset=True)
                    env.context = lambda: {'epsilon_used': float(policy.model.exploration_rate)} if algorithm == 'dqn' else {'sampling': 'PPO stochastic training action'}
            if trainer: trainer.train(actions)
            else:
                expected = budget.training[algorithm]+actions
                policy.model.learn(total_timesteps=actions, reset_num_timesteps=False)
                if policy.model.num_timesteps != expected: raise ValueError('SB3 exceeded or missed the exact registered block')
        finally: stream.close()
    def assess(label, categories):
        result = checkpoint(folder, label, policy, stream, categories, task, c, budget, trainer.updates if trainer else None)
        summaries.append(result); return result
    try:
        initial = assess('untrained', 'A')['evaluation']['A']
        passed = False
        for boundary in c['open_checkpoints']:
            fit(boundary-budget.training[algorithm])
            current = assess('A', 'A')
            if basic_gate(current['evaluation']['A'], initial, c): passed = True; break
        result = {'algorithm': algorithm, 'basic_gate': passed, 'checkpoints': summaries, 'halt': None}
        if not passed: result['halt'] = 'Basic learning gate failed at the 10,000-action cap; no B/C/D/E progression'
        elif budget.training[algorithm]+2000 > c['per_algorithm_actions']: result['halt'] = 'Basic gate passed but no registered budget remains for obstacles'
        else:
            fit(2000, 'B'); current = assess('B', 'AB')
            if (current['evaluation']['B']['counts']['success'] < c['wall_min_successes']
                    or current['evaluation']['A']['counts']['success'] < c['retention_min_successes']):
                result['halt'] = 'Wall/retention gate failed; C/D/E training not run'
            elif budget.training[algorithm]+2000 > c['per_algorithm_actions']: result['halt'] = 'Budget stops before C/D'
            else:
                fit(2000, 'CD'); current = assess('CD', 'ABCD')
                if (current['evaluation']['C']['counts']['success'] < c['corridor_min_successes']
                        or current['evaluation']['D']['counts']['success'] < c['detour_min_successes']
                        or current['evaluation']['A']['counts']['success'] < c['retention_min_successes']):
                    result['halt'] = 'C/D/retention gate failed; E training not run'
                elif budget.training[algorithm]+2000 <= c['per_algorithm_actions']:
                    fit(2000, 'E'); assess('E', 'ABCDE')
                else: result['halt'] = 'Budget stops before E training; E mechanics remain controls only'
        result.update(status='completed', training_actions=budget.training[algorithm])
    except BaseException as exc:
        stream.close()
        if isinstance(exc, EpisodeAbort): write_records(folder/'interrupted-evaluation.jsonl.gz', [exc.record])
        exception = traceback.format_exc()
        (folder/'exception.txt').write_text(exception, encoding='utf-8')
        path = folder/('interrupted-policy.npz' if trainer else 'interrupted-policy.zip')
        try: policy.save(path, {'study_id': STUDY, 'scope': 'Interrupted prefix', 'training_interactions': budget.training[algorithm]})
        except BaseException: (folder/'checkpoint-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        result = {'algorithm': algorithm, 'status': 'interrupted', 'training_actions': budget.training[algorithm],
                  'basic_gate': False, 'checkpoints': summaries, 'exception': f'{type(exc).__name__}: {exc}',
                  'limit_abort': isinstance(exc, PilotLimit) or (isinstance(exc, EpisodeAbort) and isinstance(exc.cause, PilotLimit))}
    write_json(folder/'run-summary.json', result)
    return result


def run(root=ROOT):
    protocol = checked_protocol(root); folder = location(root); c = protocol['settings']; task = configuration(root)
    if read_json(folder/'status.json')['status'] != 'registered': raise ValueError('Pilot already attempted; no free retry')
    with (folder/'execution.lock').open('x') as lock: lock.write(datetime.now(timezone.utc).isoformat())
    write_json(folder/'status.json', {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat()})
    budget = Budget(c, folder); result = {'systems': {}}
    try:
        result['controls'] = controls(folder/'controls', task, c, budget)
        result['composition'] = composition(folder/'composition', root, budget)
        write_json(folder/'reward-sanity.json', reward_sanity(c['model']['discount']))
        write_json(folder/'representation-probe.json', representation_probe())
        for algorithm in ALGORITHMS:
            budget.check()
            result['systems'][algorithm] = train_algorithm(algorithm, folder/algorithm, task, c, budget)
            if (result['systems'][algorithm]['status'] != 'completed'
                    and not result['systems'][algorithm].get('limit_abort')):
                raise RuntimeError(f'{algorithm}: functional failure stops the remaining pilot')
            if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) > c['artifact_byte_ceiling']:
                raise PilotLimit('Artifact byte ceiling')
        checked_protocol(root)
        budget.check()
        result['verdict'] = ('IMPLEMENTATION_BLOCKED' if any(v['status'] != 'completed' for v in result['systems'].values())
                             else 'READY_FOR_PHASE5_PROTOCOL_REVIEW' if all(v['basic_gate'] for v in result['systems'].values())
                             else 'REQUIRES_REPRESENTATION_OR_TRAINING_DIAGNOSIS')
        result['status'] = 'completed'
    except BaseException as exc:
        if isinstance(exc, EpisodeAbort): write_records(folder/'interrupted-control-or-composition.jsonl.gz', [exc.record])
        (folder/'exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
        result.update(status='interrupted', exception=f'{type(exc).__name__}: {exc}', verdict='IMPLEMENTATION_BLOCKED')
    finally:
        result.update(budget=budget.summary(), ended_utc=datetime.now(timezone.utc).isoformat(), resources=_resource_usage())
        budget.close()
    write_json(folder/'summary.json', result)
    write_json(folder/'outputs.json', {p.relative_to(folder).as_posix(): sha(p) for p in folder.rglob('*')
        if p.is_file() and p.name not in ('status.json', 'outputs.json')})
    write_json(folder/'status.json', {'status': result['status'], 'verdict': result['verdict']})
    return result


def composition(folder, root, budget):
    """Observe the unchanged online reference on all 180 existing dynamic dev cases."""
    import csv
    folder.mkdir(); reader = DevelopmentBenchmark(root)
    selected = [v for split in ('train', 'validation') for v in reader.scenarios(split, True)]
    rows, durations, activations = [], Counter(), Counter()
    with gzip.open(folder/'online-reference.jsonl.gz', 'xt', encoding='utf-8') as output:
        for case in selected:
            try: record = rollout(case, OnlinePlanner(), configuration(root), budget)
            except EpisodeAbort as exc:
                output.write(json.dumps(exc.record)+'\n'); raise
            validate_replay(record); output.write(json.dumps(record, allow_nan=False)+'\n'); output.flush()
            grid = GridMap(case['layout_id'], tuple(case['rows']))
            path = grid.path(case['start'], case['goal'])
            original_path = set(path)
            near, active_at_arrival = 0, 0
            for event in case['events']:
                durations[event['clear']-event['start']] += 1; activations[event['start']] += 1
                # Timing/proximity proxy on the unimpeded base path. It ignores
                # dynamic delays/occlusion and is not a real agent observation.
                cell = tuple(event['cell'])
                near += any(event['start'] <= t < event['clear'] and (p[0] == cell[0] or p[1] == cell[1])
                            and 0 < abs(p[0]-cell[0])+abs(p[1]-cell[1]) <= 2 for t, p in enumerate(path[:-1]))
                active_at_arrival += any(tuple(p) == cell and event['start'] <= t-1 < event['clear']
                                         for t, p in enumerate(path) if t)
            relevant, visible, disconnected, detour, no_move, suppressed = set(), set(), False, False, False, set()
            snapshots = [record['initial_snapshot']]+[v['snapshot'] for v in record['transitions'][:-1]]
            for snap in snapshots:
                active = snap['event_log'][-1]['applied']
                observed = {(r, c) for r, c, b in snap['scan']['cells'] if b}
                remaining = set(grid.path(snap['position'], case['goal']))
                events = {e['id']: e for e in case['events']}
                blocked = {tuple(map(int, p)) for p in np.argwhere(snap['occupancy'])}
                if any(tuple(events[e]['cell']) in remaining for e in active):
                    disconnected |= path_without(grid, snap['position'], case['goal'], blocked) is None
                    detour |= path_without(grid, snap['position'], case['goal'], blocked) is not None
                    pos = tuple(snap['position'])
                    no_move |= all(not grid.is_free((pos[0]+dr, pos[1]+dc)) or (pos[0]+dr, pos[1]+dc) in blocked
                                   for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                for e in active:
                    if tuple(events[e]['cell']) in observed:
                        visible.add(e)
                        if tuple(events[e]['cell']) in remaining: relevant.add(e)
                suppressed.update(snap['event_log'][-1]['suppressed'])
            m = record['metrics']
            rows.append({'case_id': case['id'], 'split': case['split'], 'outcome': m['outcome'],
                'actions': m['actions'], 'nominal_events': len(case['events']),
                'initial_base_path_events': sum(tuple(e['cell']) in original_path for e in case['events']),
                'unimpeded_near_ray_timing_proxy': int(near), 'unimpeded_arrival_active_events': int(active_at_arrival),
                'single_cell_disconnecting_impacts': sum(x is None for x in case['difficulty']['single_event_detour_lengths']),
                'observed_at_executed_decision': len(visible), 'observed_relevant_at_executed_decision': len(relevant),
                'no_relevant_event_observed': not relevant, 'temporarily_disconnected_during_relevant_window': disconnected,
                'detour_available_during_relevant_window': detour, 'no_legal_move_during_relevant_window': no_move,
                'events_start_at_or_after_route_end': sum(e['start'] >= m['actions'] for e in case['events']),
                'events_with_suppression': len(suppressed),
                **{k: m[k] for k in ('applied_event_count', 'detected_event_count', 'events_never_observed',
                    'suppressed_activation_ticks', 'recovery_opportunities', 'recoveries')}})
    keys = [k for k in rows[0] if k not in ('case_id', 'split', 'outcome')]
    result = {'reference_cases': len(rows), 'reference_outcomes': dict(Counter(r['outcome'] for r in rows)),
        'totals': {k: sum(r[k] for r in rows) for k in keys},
        'duration_counts': {str(k): v for k, v in sorted(durations.items())},
        'activation_step_counts': {str(k): v for k, v in sorted(activations.items())},
        'scope': 'One unchanged observation-only OnlinePlanner trajectory per existing dynamic train/development case. '
            'Unimpeded timing is an evaluator counterfactual, not sensor visibility. Disconnection counts are cases, '
            'single-cell impacts are events; detour/disconnection flags can overlap across times. No final access.'}
    with (folder/'cases.csv').open('x', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(folder/'summary.json', result)
    return result


def audit_episode(record):
    if record.get('completion', 'complete') == 'complete':
        validate_replay(record)
    else:
        if record['metrics'] is not None or digest(record['scenario']) != record['scenario_hash']:
            raise ValueError('Invalid interrupted episode identity/metrics')
        if record['task_signature'] != signature(record['configuration']): raise ValueError('Partial task mismatch')
        if any(code_hashes().get(n) != v for n, v in record['code_hashes'].items()):
            raise ValueError('Partial replay engine changed')
        trace = [{'kind': 'reset', 'episode': 0, 'seed': record['evaluation_seed'], 'scenario': record['scenario'],
                  'observation': record['initial_observation'], 'snapshot': record['initial_snapshot']}]
        trace += [{'kind': 'step', 'episode': 0, **step} for step in record['transitions']]
        if validate_training_trace(trace, record['configuration']) != len(record['transitions']):
            raise ValueError('Interrupted action prefix mismatch')
    return len(record['transitions'])


def audit(root=ROOT):
    folder = location(root); checked_protocol(root)
    for name, value in read_json(folder/'outputs.json').items():
        target = (folder/name).resolve()
        if not target.is_relative_to(folder.resolve()) or sha(target) != value: raise ValueError(f'Pilot output changed: {name}')
    summary = read_json(folder/'summary.json'); actual_training = dict.fromkeys(ALGORITHMS, 0)
    actual_evaluation = 0; episodes = 0
    paths = list((folder/'controls').glob('*.jsonl.gz'))+list((folder/'composition').glob('*.jsonl.gz'))
    for algorithm in ALGORITHMS: paths += list((folder/algorithm).glob('*/evaluation.jsonl.gz'))
    # Extra interrupted-* diagnostic copies are intentionally not double-counted.
    for path in paths:
        for record in read_records(path):
            actual_evaluation += audit_episode(record); episodes += 1
    task = configuration(root)
    for algorithm, run_result in summary['systems'].items():
        path = folder/algorithm/'training-actions.jsonl.gz'
        records = read_records(path) if path.exists() else []
        actual_training[algorithm] = validate_training_trace(records, task)
        if [v['global_training_interaction'] for v in records if v['kind'] == 'step'] != list(range(1, actual_training[algorithm]+1)):
            raise ValueError('Non-continuous actual training counter')
        if actual_training[algorithm] != run_result['training_actions']: raise ValueError('Training count audit mismatch')
        for checkpoint_result in run_result['checkpoints']:
            target = folder/algorithm/f"{checkpoint_result['phase']}-{checkpoint_result['training_actions']:05}"
            prefix = path.read_bytes()[:checkpoint_result['training_prefix_bytes']] if path.exists() else b''
            if prefix and hashlib.sha256(prefix).hexdigest() != checkpoint_result['training_prefix_sha256']:
                raise ValueError('An earlier training prefix was rewritten')
            with preserve_random_state():
                policy = load_policy(target/checkpoint_result['checkpoint'], task)
                evaluated = read_records(target/'evaluation.jsonl.gz')
                for record in evaluated:
                    for transition in record['transitions']:
                        obs = np.array(transition['observation'], dtype=np.float32)
                        if policy.act(obs.copy()) != transition['action']: raise ValueError('Executed action does not match checkpoint')
                        decision = decision_values(policy, obs.copy())
                        for key in decision:
                            if not np.allclose(decision[key], transition['decision'][key], atol=1e-7, rtol=1e-6):
                                raise ValueError('Saved decision values do not match checkpoint')
                if digest(grouped(evaluated)) != digest(checkpoint_result['evaluation']):
                    raise ValueError('Evaluation summary mismatch')
            prefix_records = []
            if checkpoint_result['training_prefix_bytes']:
                import io
                with gzip.open(io.BytesIO(prefix), 'rt', encoding='utf-8') as file:
                    prefix_records = [json.loads(line) for line in file]
                if digest(training_summary(prefix_records)) != digest(checkpoint_result['training']):
                    raise ValueError('Training diagnostic summary mismatch')
    reserved_train, reserved_eval = dict.fromkeys(ALGORITHMS, 0), 0
    with (folder/'budget-reservations.jsonl').open(encoding='utf-8') as file:
        for line in file:
            row = json.loads(line)
            if row['kind'] == 'training_reserved':
                reserved_train[row['algorithm']] += 1
                if reserved_train[row['algorithm']] != row['count'] or row['count'] > 10000:
                    raise ValueError('Training reservation sequence/ceiling mismatch')
            else:
                reserved_eval += 1
                if reserved_eval != row['count'] or reserved_eval > 100000: raise ValueError('Evaluation reservation ceiling mismatch')
    if reserved_train != summary['budget']['training_reserved'] or reserved_eval != summary['budget']['evaluation_reserved']:
        raise ValueError('Budget journal and summary disagree')
    if any(actual_training[a] > reserved_train[a] for a in ALGORITHMS) or actual_evaluation > reserved_eval:
        raise ValueError('Executed actions exceed durable reservations')
    if summary['status'] == 'completed' and (actual_training != reserved_train or actual_evaluation != reserved_eval):
        raise ValueError('Completed study contains unused action reservations')
    result = {'status': 'passed' if summary['status'] == 'completed' else 'partial_prefixes_audited',
              'training_actions': actual_training, 'total_training_actions': sum(actual_training.values()),
              'evaluation_actions': actual_evaluation, 'evaluation_episodes': episodes, 'protocol_sha256': sha(folder/'phase5a_pilot_protocol.json')}
    # Verification is a new artifact, never a rewrite of measured outcomes.
    path = folder/'replay-verification.json'
    if path.exists():
        if read_json(path) != result: raise ValueError('Existing replay verification differs')
    else: write_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register', 'run', 'audit', 'report'))
    args = parser.parse_args()
    if args.command == 'register': result = register()
    elif args.command == 'run': result = run()
    elif args.command == 'audit': result = audit()
    else:
        from .partial_feasibility_report import report
        result = report()
    print(json.dumps({k: v for k, v in result.items() if k not in ('systems', 'controls', 'composition')}, indent=2))


if __name__ == '__main__': main()
