"""Registered paired DQN input experiment; old evidence is read-only."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
from time import perf_counter

import yaml

from robot_navigation.core import ROOT, digest, load_task, read_json, write_json
from robot_navigation.observation_encoding import COORDINATES, ABSOLUTE_RELATIVE
from robot_navigation.part_a import train_part_a
from robot_navigation.release import sha
from robot_navigation.workflow import source_hash


def protected_files():
    return {p.relative_to(ROOT).as_posix(): sha(p) for folder in ('artifacts', 'data')
            for p in (ROOT/folder).rglob('*') if p.is_file()}


def check_protected(study):
    hashes = read_json(study/'historical_hashes.json')
    if any(not (ROOT/p).is_file() or sha(ROOT/p) != value for p, value in hashes.items()):
        raise ValueError('Protected historical artifact/data bytes changed')
    return len(hashes)


def run_config(plan, variant, seed, pilot=False):
    return {**plan['training'], 'seed': seed,
            'observation_encoding': plan['encodings'][variant],
            'interactions': plan['pilot_interactions'] if pilot else plan['training']['interactions'],
            'run_id': f"{plan['run_prefix']}-{variant}-dqn-{'pilot-' if pilot else ''}seed{seed}"}


def execute(plan, variant, seed, pilot=False):
    config = run_config(plan, variant, seed, pilot)
    print(f"Starting {config['run_id']}", flush=True)
    started = perf_counter()
    directory, summaries, diagnostics = train_part_a(config)
    manifest = read_json(directory/'manifest.json')
    result = {'run_id': directory.name, 'algorithm': 'dqn', 'variant': variant, 'seed': seed,
              'encoding': config['observation_encoding'], 'validation': summaries['dqn'],
              'pilot': diagnostics, 'selected_interactions': manifest['selected_interactions'],
              'wall_seconds': perf_counter()-started, 'monitoring_seconds': manifest['monitoring_seconds']}
    print(f"Finished {directory.name}: {summaries['dqn']['counts']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('pilot', 'run'))
    parser.add_argument('--config', type=Path, default=ROOT/'configs/representation_study.yaml')
    args = parser.parse_args()
    plan = yaml.safe_load(args.config.read_text('utf-8'))
    for key in ('suite_id', 'run_prefix'):
        if Path(plan[key]).name != plan[key] or plan[key] in ('.', '..'):
            raise ValueError('Study/run prefix must be a single directory name')
    if plan['encodings'] != {'control': COORDINATES, 'relative': ABSOLUTE_RELATIVE}:
        raise ValueError('The registered two representation treatments are required')
    if plan['seeds'] != [101,211,307,401,503] or plan['training']['interactions'] != 200000:
        raise ValueError('The registered paired seed matrix and budget are required')
    from robot_navigation.dqn import DQNPolicy  # Check the runtime before creating output.
    study = ROOT/'artifacts/studies'/plan['suite_id']
    if args.command == 'pilot':
        if study.exists():
            raise FileExistsError('Study already exists; do not overwrite/reselect')
        hashes = protected_files()
        study.mkdir(parents=True)
        write_json(study/'historical_hashes.json', hashes)
        write_json(study/'plan.json', plan)
        write_json(study/'pretraining.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
            'plan_hash': digest(plan), 'source_hash': source_hash(ROOT), 'signature': load_task().signature,
            'scope': 'exploratory development validation; no final-test execution', 'configuration_search': False})
        for source, name in ((ROOT/'docs/representation_protocol.md', 'protocol.md'),
                             (Path(__file__), 'runner.py'),
                             (ROOT/'scripts/analyze_representation_study.py', 'analyzer.py')):
            shutil.copy2(source, study/name)
        write_json(study/'pilot_status.json', {'status': 'running'})
        try:
            rows = [execute(plan, variant, plan['pilot_seed'], True) for variant in plan['encodings']]
            factor = plan['training']['interactions']/plan['pilot_interactions']
            estimate = len(plan['seeds'])*sum((r['pilot']['training_seconds']+r['monitoring_seconds'])*factor
                          + max(0, r['wall_seconds']-r['pilot']['training_seconds']-r['monitoring_seconds']) for r in rows)
            write_json(study/'pilot.json', {'runs': rows, 'estimated_main_wall_seconds': estimate,
                       'settings_changed': False, 'excluded_from_main_statistics': True})
            write_json(study/'pilot_status.json', {'status': 'complete'})
            check_protected(study)
            print(f'Pilot estimate for main matrix: {estimate:.2f} seconds', flush=True)
        except BaseException as exc:
            write_json(study/'pilot_status.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
            raise
        return
    if read_json(study/'plan.json') != plan or read_json(study/'pilot_status.json')['status'] != 'complete':
        raise ValueError('Pilot or unchanged registered configuration is missing')
    if read_json(study/'pilot.json')['estimated_main_wall_seconds'] > plan['limits']['suite_wall_seconds']:
        raise ValueError('Measured pilot estimate exceeds the predeclared main wall-time cap')
    if (study/'manifest.json').exists():
        raise FileExistsError('Matrix already started; do not silently overwrite/resume')
    check_protected(study)
    frozen_source = source_hash(ROOT)
    started = perf_counter()
    manifest = {'status': 'running', 'signature': load_task().signature, 'plan_hash': digest(plan),
                'source_hash': frozen_source, 'started_utc': datetime.now(timezone.utc).isoformat(),
                'completed_runs': [], 'final_test_evaluated': False, 'exploratory_post_release': True}
    write_json(study/'manifest.json', manifest)
    results = []
    try:
        for seed in plan['seeds']:
            for variant in plan['encodings']:
                if source_hash(ROOT) != frozen_source:
                    raise ValueError('Scientific/UI/config source changed during the registered matrix')
                if perf_counter()-started > plan['limits']['suite_wall_seconds']:
                    raise TimeoutError('Representation matrix wall-time limit reached')
                row = execute(plan, variant, seed)
                results.append(row); manifest['completed_runs'].append(row['run_id'])
                write_json(study/'run_results.json', results)
                write_json(study/'manifest.json', manifest)
                names = manifest['completed_runs']+[run_config(plan,v,plan['pilot_seed'],True)['run_id'] for v in plan['encodings']]
                size = sum(p.stat().st_size for directory in [study,*[ROOT/'artifacts/runs'/name for name in names]]
                           for p in directory.rglob('*') if p.is_file())
                if size > plan['limits']['new_artifact_bytes']:
                    raise RuntimeError('Representation artifact storage limit exceeded')
                if perf_counter()-started > plan['limits']['suite_wall_seconds']:
                    raise TimeoutError('Representation matrix wall-time limit exceeded')
        check_protected(study)
        manifest.update(status='complete', wall_seconds=perf_counter()-started,
                        training_interactions=sum(r['pilot']['training_interactions'] for r in results),
                        artifact_bytes=size, finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                        error=f'{type(exc).__name__}: {exc}', wall_seconds=perf_counter()-started)
        write_json(study/'manifest.json', manifest)
        raise
    write_json(study/'manifest.json', manifest)
    print('Representation matrix complete; run its audit/analyzer.', flush=True)


if __name__ == '__main__':
    main()
