"""Record fresh four-family demonstrations without changing existing evidence."""
import argparse
from pathlib import Path
from time import perf_counter

import yaml

from robot_navigation.core import ROOT, digest, read_json, write_json
from robot_navigation.learning_replays import audit_demo, demo_configs
from robot_navigation.part_a import train_part_a
from robot_navigation.release import sha
from robot_navigation.workflow import source_hash


def verify_existing(root=ROOT):
    """Repeat only integrity checks after a verifier repair; never retrain."""
    study = root/'artifacts/studies/learning-replays-v1'
    frozen = read_json(study/'plan.json')
    plan, configs = frozen['plan'], frozen['configs']
    if digest(plan) != frozen['plan_hash']:
        raise ValueError('Recorded demo plan hash changed')
    status = read_json(study/'manifest.json')
    started = perf_counter()
    destinations = [root/'artifacts/runs'/config['run_id'] for config in configs]
    audits = []
    for directory, config in zip(destinations, configs, strict=True):
        manifest = read_json(directory/'manifest.json')
        if (manifest['source_hash'] != frozen['source_hash']
                or read_json(directory/'configuration.json')['training'] != config):
            raise ValueError('Training configuration or recorded source identity changed')
        audits.append(audit_demo(directory, root))
    historical = read_json(study/'historical_hashes.json')
    if any(not (root/p).is_file() or sha(root/p) != value for p,value in historical.items()):
        raise ValueError('Pre-existing artifact/data files changed')
    artifact_bytes = sum(p.stat().st_size for folder in [study,*destinations]
                         for p in folder.rglob('*') if p.is_file())
    if artifact_bytes > plan['limits']['new_artifact_bytes']:
        raise ValueError('Recorded demo artifacts exceed their declared size cap')
    verification = {'status':'passed', 'recording_source_hash':frozen['source_hash'],
                    'auditor_sha256':sha(root/'src/robot_navigation/learning_replays.py'),
                    'verification_seconds':perf_counter()-started,
                    'historical_files_verified':len(historical), 'historical_inputs_unchanged':True,
                    'new_training_interactions_during_verification':0, 'new_final_test_policy_executions':0,
                    'training_suite_elapsed_seconds':status.get('elapsed_seconds'),
                    'return_comparison_absolute_tolerance':1e-12,
                    'raw_actions_and_transition_rewards_unchanged':True}
    if status['status']=='failed':
        verification['previous_audit_error'] = status.get('error')
    status.pop('error',None)
    status.update(status='verified', total_training_interactions=sum(a['training_interactions'] for a in audits),
                  recorded_episodes=sum(len(a['recordings']) for a in audits),
                  recorded_actions=sum(a['recorded_actions'] for a in audits),
                  historical_files_verified=len(historical), historical_inputs_unchanged=True,
                  source_hash=frozen['source_hash'], new_artifact_bytes=artifact_bytes)
    write_json(study/'run_results.json',audits)
    write_json(study/'verification.json',verification)
    write_json(study/'manifest.json',status)
    status['new_artifact_bytes'] = sum(p.stat().st_size for folder in [study,*destinations]
                                     for p in folder.rglob('*') if p.is_file())
    write_json(study/'manifest.json',status)
    print(status,flush=True)
    return status


def run(root=ROOT):
    plan = yaml.safe_load((root/'configs/learning_replays.yaml').read_text(encoding='utf-8'))
    configs = demo_configs(plan, root)
    study = root/'artifacts/studies'/plan['suite_id']
    destinations = [root/'artifacts/runs'/config['run_id'] for config in configs]
    if study.exists() or any(path.exists() for path in destinations):
        raise FileExistsError('Recorded demos are immutable; choose a new suite/run identity for a new attempt')
    historical = {p.relative_to(root).as_posix():sha(p) for folder in ('artifacts','data')
                  for p in (root/folder).rglob('*') if p.is_file()}
    study.mkdir(parents=True)
    source = source_hash(root)
    write_json(study/'historical_hashes.json', historical)
    write_json(study/'plan.json', {'plan':plan, 'configs':configs, 'plan_hash':digest(plan),
                                  'source_hash':source, 'runner_sha256':sha(Path(__file__)),
                                  'purpose':'record genuine new training examples; not a tuning or benchmark study'})
    started = perf_counter()
    status = {'status':'running', 'completed_runs':[], 'new_final_test_policy_executions':0}
    write_json(study/'manifest.json', status)
    try:
        for config in configs:
            if perf_counter()-started > plan['limits']['suite_wall_seconds']:
                raise TimeoutError('Recording suite exceeded its declared time cap')
            if source_hash(root) != source:
                raise ValueError('Source changed while recorded training was running')
            print(f"Starting {config['run_id']} — {config['interactions']:,} actual training steps", flush=True)
            directory, _, _ = train_part_a(config, root)
            status['completed_runs'].append(directory.name)
            write_json(study/'manifest.json', status)
            print(f'Completed {directory.name}', flush=True)
        audits = [audit_demo(path, root) for path in destinations]
        if any(not (root/p).is_file() or sha(root/p) != value for p,value in historical.items()):
            raise ValueError('Pre-existing artifact/data files changed')
        artifact_bytes = sum(p.stat().st_size for folder in [study, *destinations]
                             for p in folder.rglob('*') if p.is_file())
        if artifact_bytes > plan['limits']['new_artifact_bytes']:
            raise ValueError('Recorded demo artifacts exceed the declared size cap')
        status.update(status='verified', total_training_interactions=sum(a['training_interactions'] for a in audits),
                      recorded_episodes=sum(len(a['recordings']) for a in audits),
                      recorded_actions=sum(a['recorded_actions'] for a in audits),
                      historical_files_verified=len(historical), historical_inputs_unchanged=True,
                      source_hash=source, elapsed_seconds=perf_counter()-started, new_artifact_bytes=artifact_bytes)
        write_json(study/'run_results.json', audits)
        write_json(study/'manifest.json', status)
        print(status, flush=True)
    except BaseException as exc:
        status.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                      error=f'{type(exc).__name__}: {exc}', elapsed_seconds=perf_counter()-started)
        write_json(study/'manifest.json', status)
        raise
    return status


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-existing',action='store_true',help='Audit existing recordings without training or replacing them')
    args=parser.parse_args()
    verify_existing() if args.verify_existing else run()
