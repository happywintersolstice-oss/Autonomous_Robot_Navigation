"""Validation/development tools only. No large-training or final-evaluation command."""
import argparse
from datetime import datetime, timezone
import gzip
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

from .core import ROOT, read_json, write_json
from .release import sha
from .partial_agents import ALGORITHMS, LinearPolicy, load_policy
from .partial_benchmark import DevelopmentBenchmark, prepare
from .partial_contract import BENCHMARK, configuration
from .partial_dynamic_env import PartialDynamicEnv
from .partial_evaluation import code_hashes, grouped_summary, record_episode, validate_replay
from .partial_fixtures import corridor
from .partial_training import TrainingRecorder, train_linear, validate_training_trace


def gate_path(root=ROOT):
    return root/'.runtime/partial-dynamic-gates.json'


def validation_inputs(root=ROOT):
    return {'code': code_hashes(), 'configuration': sha(root/'configs/partial_dynamic.yaml'),
            'tests': {p.name: sha(p) for p in sorted((root/'developer/tests').glob('test_partial_*.py'))}}


def require_gates(root=ROOT):
    proof = read_json(gate_path(root))
    if proof.get('status') != 'passed' or proof.get('inputs') != validation_inputs(root):
        raise ValueError('Partial validation gates are failed/stale; run validate before smoke training')
    return proof


def validate(root=ROOT):
    before = validation_inputs(root)
    paths = sorted((root/'developer/tests').glob('test_partial_*.py'))
    if len(paths) < 8: raise ValueError('Required partial benchmark tests are missing')
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *[str(p) for p in paths],
        '--basetemp=.runtime/partial-validation-tests'], cwd=root, capture_output=True, text=True, timeout=180)
    log = result.stdout+result.stderr
    (root/'.runtime').mkdir(exist_ok=True)
    (root/'.runtime/partial-validation.log').write_text(log, encoding='utf-8')
    proof = {'status': 'passed' if result.returncode == 0 and before == validation_inputs(root) else 'failed',
        'inputs': before, 'returncode': result.returncode, 'log': '.runtime/partial-validation.log',
        'validated_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Implementation gates, not learned performance'}
    write_json(gate_path(root), proof)
    print(log, end='')
    if proof['status'] != 'passed': raise RuntimeError('Partial validation failed; smoke training remains blocked')
    return proof


def write_records(path, records):
    with gzip.open(path, 'wt', encoding='utf-8') as file:
        for record in records: file.write(json.dumps(record, allow_nan=False)+'\n')


def read_records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def manual_trace(output, root=ROOT):
    class Script:
        algorithm = policy_id = 'handwritten-wait-then-deliver-v1'
        def __init__(self): self.actions = iter([4]*3+[3]*7)
        def act(self, observation): return next(self.actions)
    output = Path(output)
    if output.exists(): raise FileExistsError('Trace already exists; choose a fresh path')
    record = record_episode(corridor(clear=3), Script(), configuration(root))
    validate_replay(record)
    write_json(output, record)
    return {'status': 'passed', 'actions': len(record['transitions']), 'outcome': record['metrics']['outcome']}


def smoke(output, root=ROOT):
    proof = require_gates(root)
    reader = DevelopmentBenchmark(root); reader.audit()
    config, output = configuration(root), Path(output)
    c = config['smoke']
    if c['interactions_per_algorithm'] != 256 or c['algorithms'] != list(ALGORITHMS):
        raise ValueError('Only the registered 4 x 256-action functional smoke matrix is allowed')
    if output.exists(): raise FileExistsError('Smoke directory exists; preserve it and choose a fresh path')
    output.mkdir(parents=True)
    write_json(output/'status.json', {'status': 'running', 'scope': 'Functional validation only'})
    started = perf_counter()
    try:
        snapshot = output/'snapshot'
        shutil.copytree(root/'src/robot_navigation', snapshot/'src/robot_navigation',
                        ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
        for name in ('configs/partial_dynamic.yaml', 'docs/partial_dynamic_environment_spec.md',
                     'docs/partial_dynamic_experiment_protocol.md', 'pyproject.toml', 'requirements-lock.txt'):
            target = snapshot/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root/name, target)
        shutil.copytree(reader.folder, snapshot/'data/partial'/BENCHMARK)
        write_json(output/'inputs.json', {p.relative_to(snapshot).as_posix(): sha(p)
            for p in snapshot.rglob('*') if p.is_file()})
        write_json(output/'validation-gates.json', proof)
        shutil.copy2(root/'.runtime/partial-validation.log', output/'validation.log')
        training_cases = reader.scenarios('train')[:2]
        assessment = [reader.scenarios(split, dynamic)[0]
                      for split in ('train', 'validation') for dynamic in (False, True)]
        summaries = {}
        for algorithm in ALGORITHMS:
            if perf_counter()-started > 120: raise TimeoutError('Bounded smoke wall limit exceeded')
            folder = output/algorithm; folder.mkdir()
            recorder = TrainingRecorder(PartialDynamicEnv(training_cases, config), training_cases)
            if algorithm in ('q-learning', 'sarsa'):
                policy = LinearPolicy(algorithm, config, c['seed'])
            else:
                from .partial_neural import build_neural
                policy = build_neural(recorder, algorithm, c, c['seed'])
            before = policy.state_fingerprint()
            tick = perf_counter()
            if algorithm in ('q-learning', 'sarsa'):
                training = train_linear(policy, recorder, 256, c['seed'], c)
            else:
                from stable_baselines3.common.callbacks import BaseCallback
                class Deadline(BaseCallback):
                    def _on_step(self):
                        if perf_counter()-started > 120: raise TimeoutError('Bounded smoke wall limit exceeded')
                        return True
                policy.model.learn(total_timesteps=256, callback=Deadline())
                training = {'interactions': int(policy.model.num_timesteps), 'training_seconds': perf_counter()-tick}
            if (training['interactions'] != 256 or recorder.interactions != 256
                    or before == policy.state_fingerprint()):
                raise ValueError(f'{algorithm}: missing interactions or parameter updates')
            if validate_training_trace(recorder.records, config) != 256:
                raise ValueError('Recorded training action count mismatch')
            training['completed_episodes'] = sum(r['kind'] == 'step' and (r['terminated'] or r['truncated']) for r in recorder.records)
            write_records(folder/'training-actions.jsonl.gz', recorder.records)
            extension = '.npz' if algorithm in ('q-learning', 'sarsa') else '.zip'
            checkpoint = folder/f'final{extension}'
            policy.save(checkpoint, {'training_interactions': 256,
                'development_manifest_sha256': sha(reader.folder/'manifest.json'), 'scope': 'Functional smoke only'})
            loaded = load_policy(checkpoint, config)
            if loaded.state_fingerprint() != policy.state_fingerprint(): raise ValueError('Reload changed learned state')
            records = [record_episode(case, loaded, config, evaluation_seed=0) for case in assessment]
            for record in records: validate_replay(record)
            write_records(folder/'assessment.jsonl.gz', records)
            summaries[algorithm] = {'status': 'functional_pass', 'training': training,
                'parameter_fingerprint': loaded.state_fingerprint(), 'checkpoint': checkpoint.name,
                'checkpoint_bytes': checkpoint.stat().st_size, 'assessment': grouped_summary(records),
                'training_actions_audited': 256, 'assessment_episodes_audited': len(records)}
            write_json(folder/'summary.json', summaries[algorithm])
        summary = {'status': 'verified', 'scope': 'Functional smoke; one seed, not a comparative performance study',
            'training_seed': c['seed'], 'training_actions': 1024, 'assessment_episodes': 16,
            'python': platform.python_version(), 'dependencies': {n: version(n) for n in
                ('numpy', 'gymnasium', 'stable-baselines3', 'torch')},
            'code_hashes': code_hashes(), 'systems': summaries, 'wall_seconds': perf_counter()-started,
            'large_study_run': False, 'new_final_generated_or_evaluated': False}
        write_json(output/'summary.json', summary)
        write_json(output/'outputs.json', {p.relative_to(output).as_posix(): sha(p)
            for p in output.rglob('*') if p.is_file() and p.name != 'status.json'})
        write_json(output/'status.json', {'status': 'verified', 'training_actions': 1024})
        return summary
    except BaseException as exc:
        write_json(output/'status.json', {'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'})
        raise


def audit(output, root=ROOT):
    output = Path(output).resolve()
    if read_json(output/'status.json')['status'] != 'verified': raise ValueError('Smoke bundle is not verified')
    for name, value in read_json(output/'outputs.json').items():
        path = (output/name).resolve()
        if not path.is_relative_to(output) or sha(path) != value: raise ValueError(f'Smoke output changed: {name}')
    for name, value in read_json(output/'inputs.json').items():
        path = (output/'snapshot'/name).resolve()
        if not path.is_relative_to(output/'snapshot') or sha(path) != value:
            raise ValueError(f'Snapshot input changed: {name}')
    summary = read_json(output/'summary.json')
    for algorithm in ALGORITHMS:
        folder = output/algorithm; saved = summary['systems'][algorithm]
        settings = configuration(root)
        loaded = load_policy(folder/saved['checkpoint'], settings)
        if loaded.state_fingerprint() != saved['parameter_fingerprint']: raise ValueError('Model fingerprint mismatch')
        if validate_training_trace(read_records(folder/'training-actions.jsonl.gz'), settings) != 256:
            raise ValueError('Training audit count mismatch')
        records = read_records(folder/'assessment.jsonl.gz')
        if len(records) != 4: raise ValueError('Smoke assessment count mismatch')
        for record in records: validate_replay(record)
        if grouped_summary(records) != saved['assessment']: raise ValueError('Smoke aggregation mismatch')
    return {'status': 'passed', 'training_actions_audited': 1024, 'assessment_episodes_audited': 16}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'prepare', 'trace', 'smoke', 'audit'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command in ('trace', 'smoke', 'audit') and args.output is None: parser.error('--output is required')
    if args.command == 'validate': result = validate()
    elif args.command == 'prepare':
        prepare(); result = DevelopmentBenchmark().audit()
    elif args.command == 'trace': result = manual_trace(args.output)
    elif args.command == 'smoke': result = smoke(args.output)
    else: result = audit(args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('systems', 'inputs', 'code_hashes')}, indent=2))


if __name__ == '__main__': main()
