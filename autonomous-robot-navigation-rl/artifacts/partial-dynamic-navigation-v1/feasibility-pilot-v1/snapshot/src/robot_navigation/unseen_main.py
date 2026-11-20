"""Frozen, bounded two-worker shared-feature development experiment."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from time import monotonic, sleep

import numpy as np
import yaml

from .core import ROOT, read_json, write_json
from .release import sha
from .shared_features import FEATURE_VERSION
from .unseen_benchmark import MapBenchmark
from .unseen_study import STUDY as PILOT, train_run
from .workflow import source_hash

STUDY = 'unseen-map-main-v1'
METRICS = ('success_rate','collision_rate','timeout_rate','spl','mean_return',
           'mean_successful_route_length')


def run_settings(config, phase, algorithm, seed, attempt=1):
    settings = deepcopy(config)
    budget = config['pilot_interactions'] if phase=='pilot' else config['interactions']
    base = f'm11-shared-{phase}-{algorithm}-seed{seed}'
    settings.update(seed=seed,interactions=budget,
                    trace_windows=[0,budget//2,budget*19//20],
                    run_id=base if attempt==1 else f'{base}-attempt{attempt}')
    return settings


def jobs(config, phase):
    seeds = [config['pilot_seed']] if phase=='pilot' else config['seeds']
    return [(algorithm,seed) for seed in seeds for algorithm in config['algorithms']]


def validate_config(config):
    if (config['study_id'] != STUDY or config['representation'] != FEATURE_VERSION
            or config['algorithms'] != ['q-learning','sarsa','dqn','ppo']
            or len(set(config['seeds'])) != len(config['seeds'])
            or config['pilot_seed'] in config['seeds'] or not config['seeds']
            or config['workers'] not in (1,2)):
        raise ValueError('Invalid shared-feature study registration')
    for budget in (config['pilot_interactions'],config['interactions'],config['monitor_every']):
        if (budget <= 0 or budget % config['dqn']['train_freq']
                or budget % config['ppo']['n_steps']):
            raise ValueError('Budgets must align with every neural update boundary')
    if (config['pilot_interactions'] % config['monitor_every']
            or config['interactions'] % config['monitor_every']
            or config['ppo']['n_steps'] % config['ppo']['batch_size']):
        raise ValueError('Budget, monitoring and minibatch boundaries differ')


def prepare(root):
    config = yaml.safe_load((root/'configs/unseen_main.yaml').read_text(encoding='utf-8'))
    validate_config(config)
    benchmark = MapBenchmark(root)
    study = root/'artifacts/studies'/STUDY
    if not study.exists():
        study.mkdir(parents=True)
        for folder in ('src','app','configs'):
            shutil.copytree(root/folder,study/'source'/folder,
                            ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
        shutil.copy2(root/'docs/milestone11_protocol.md',study/'protocol.md')
        references = root/'artifacts/studies'/PILOT
        previous = read_json(references/'summary.json')
        if previous['status'] != 'verified' or previous['signature'] != benchmark.signature:
            raise ValueError('Verified reference study is required')
        record = {'status':'registered','signature':benchmark.signature,'settings':config,
                  'source_hash':source_hash(study/'source'),'protocol_hash':sha(study/'protocol.md'),
                  'python':platform.python_version(),
                  'dependencies':{p:version(p) for p in ('numpy','gymnasium','torch','stable-baselines3')},
                  'registered_utc':datetime.now(timezone.utc).isoformat(),
                  'references':{'study':PILOT,'summary_sha256':sha(references/'summary.json'),
                                'files':{n:sha(references/f'{n}.jsonl.gz') for n in ('bfs','random')}},
                  'attempts':[],'final_test_evaluated':False}
        write_json(study/'manifest.json',record)
    record = read_json(study/'manifest.json')
    if (record['settings'] != config or record['signature'] != benchmark.signature
            or record['source_hash'] != source_hash(study/'source')
            or record['protocol_hash'] != sha(study/'protocol.md')):
        raise ValueError('Frozen source/configuration/benchmark mismatch')
    return benchmark,study,record


def worker(root, run_id):
    benchmark = MapBenchmark(root)
    study = root/'artifacts/studies'/STUDY
    record = read_json(study/'manifest.json')
    if source_hash(study/'source') != record['source_hash']:
        raise ValueError('Archived worker source changed')
    if not Path(__file__).resolve().is_relative_to((study/'source/src').resolve()):
        raise ValueError('Workers must execute the archived source')
    matches = [(phase,algorithm,seed,attempt)
               for phase in ('pilot','main') for algorithm,seed in jobs(record['settings'],phase)
               for attempt in range(1,record['settings']['max_attempts_per_run']+1)
               if run_settings(record['settings'],phase,algorithm,seed,attempt)['run_id']==run_id]
    if len(matches) != 1 or not any(a['run_id']==run_id for a in record['attempts']):
        raise ValueError('Unregistered worker run')
    phase,algorithm,seed,attempt = matches[0]
    return train_run(benchmark,study,algorithm,record['source_hash'],
                     run_settings(record['settings'],phase,algorithm,seed,attempt))


def summarize(study, record):
    rows = []
    for phase in ('pilot','main'):
        for algorithm,seed in jobs(record['settings'],phase):
            attempts = [a for a in record['attempts'] if (a['phase'],a['algorithm'],a['seed'])==(phase,algorithm,seed)]
            complete = [a for a in attempts if a['status']=='complete']
            if len(complete) != 1:
                raise ValueError('Every registered logical run requires exactly one complete attempt')
            run_id = complete[0]['run_id']
            directory = study/'runs'/run_id
            run = read_json(directory/'manifest.json')
            rows.append({k:run[k] for k in ('algorithm','run_id','training_seed','training_interactions',
                'selected_interactions','training_seconds','total_seconds','checkpoint_bytes',
                'optimized_parameter_count','mean_decision_microseconds')}
                | {'phase':phase,'validation':read_json(directory/'validation.json'),
                   'familiar':read_json(directory/'familiar.json')})
    systems = []
    for algorithm in record['settings']['algorithms']:
        group = [r for r in rows if r['phase']=='main' and r['algorithm']==algorithm]
        results = {}
        for split in ('validation','familiar'):
            stats = {}
            for metric in METRICS:
                values = [r[split][metric] for r in group if r[split][metric] is not None]
                stats[metric] = {'mean':float(np.mean(values)) if values else None,
                                'sd':float(np.std(values,ddof=1)) if len(values)>1 else None,
                                'defined_seeds':len(values)}
            results[split] = {'statistics':stats,'episodes':sum(r[split]['episodes'] for r in group),
                'counts':{k:sum(r[split]['counts'][k] for r in group) for k in ('success','collision','timeout')},
                'by_map':{name:{metric:float(np.mean([r[split]['by_map'][name][metric] for r in group]))
                                for metric in ('success_rate','collision_rate','timeout_rate','spl')}
                          for name in group[0][split]['by_map']}}
        systems.append({'algorithm':algorithm,'seed_count':len(group),**results,
                        'training_interactions':sum(r['training_interactions'] for r in group),
                        'training_seconds':sum(r['training_seconds'] for r in group)})
    reference_dir = study.parents[0]/PILOT
    references = read_json(reference_dir/'summary.json')['references']
    return {'status':'complete; audit pending','signature':record['signature'],'systems':systems,
            'runs':rows,'references':references,'training_seed_count':len(record['settings']['seeds']),
            'total_training_interactions':sum(r['training_interactions'] for r in rows),
            'reserved_attempt_interactions':sum(a['budget'] for a in record['attempts']),
            'scope':'shared-feature multi-seed development validation; final layouts reserved',
            'representation':FEATURE_VERSION,'final_test_evaluated':False}


def execute(root=ROOT):
    _,study,record = prepare(root)
    if record['status'] in ('complete; audit pending','verified'):
        return read_json(study/'summary.json')
    config = record['settings']
    lock = study/'run.lock'
    with lock.open('x',encoding='utf-8') as file:
        file.write(str(os.getpid()))
    active = {}
    started = monotonic()
    try:
        if any(a['status']=='running' for a in record['attempts']):
            raise RuntimeError('An interrupted worker owner must be checked before recovery')
        for phase in ('pilot','main'):
            pending = list(jobs(config,phase))
            while pending or active:
                if monotonic()-started > config['suite_wall_seconds']:
                    raise TimeoutError('Study suite wall limit exceeded')
                for run_id,(process,log,when,attempt) in list(active.items()):
                    if monotonic()-when > config['run_wall_seconds']+30:
                        process.kill()
                    code = process.poll()
                    if code is None:
                        continue
                    process.wait()
                    log.close()
                    run_path = study/'runs'/run_id/'manifest.json'
                    run = read_json(run_path) if run_path.exists() else {}
                    attempt.update(status='complete' if code==0 and run.get('status')=='complete' else 'failed',
                                   exit_code=code,wall_seconds=monotonic()-when,
                                   observed_interactions=run.get('training_interactions',run.get('training_interactions_observed')))
                    if attempt['observed_interactions'] is None:
                        attempt['unobserved_steps_upper_bound'] = attempt['budget']
                    del active[run_id]
                    if attempt['status']=='failed':
                        pending.append((attempt['algorithm'],attempt['seed']))
                    print(f'{run_id} {attempt["status"]}',flush=True)
                    write_json(study/'manifest.json',record)
                while pending and len(active)<config['workers']:
                    algorithm,seed = pending.pop(0)
                    previous = [a for a in record['attempts'] if
                                (a['phase'],a['algorithm'],a['seed'])==(phase,algorithm,seed)]
                    if any(a['status']=='complete' for a in previous):
                        continue
                    attempt_number = len(previous)+1
                    settings = run_settings(config,phase,algorithm,seed,attempt_number)
                    if (attempt_number>config['max_attempts_per_run']
                            or len(record['attempts'])>=config['max_runs_including_attempts']
                            or sum(a['budget'] for a in record['attempts'])+settings['interactions']>
                                config['max_training_interactions_including_attempts']):
                        raise RuntimeError('Registered retry/interaction limit reached')
                    attempt = {'run_id':settings['run_id'],'phase':phase,'algorithm':algorithm,'seed':seed,
                               'attempt':attempt_number,'budget':settings['interactions'],'status':'running'}
                    record['attempts'].append(attempt)
                    record['status'] = 'running'
                    write_json(study/'manifest.json',record)
                    logs = study/'logs'
                    logs.mkdir(exist_ok=True)
                    log = (logs/f'{settings["run_id"]}.log').open('xb')
                    env = {**os.environ,'PYTHONPATH':str(study/'source/src'),
                           'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
                    process = subprocess.Popen([os.sys.executable,'-X','utf8','-m','robot_navigation.unseen_main',
                        'worker','--root',str(root),'--run-id',settings['run_id']],cwd=study/'source',
                        env=env,stdout=log,stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    active[settings['run_id']] = (process,log,monotonic(),attempt)
                if sum(p.stat().st_size for p in study.rglob('*') if p.is_file())>config['artifact_bytes']:
                    raise RuntimeError('Study storage limit exceeded')
                if active:
                    sleep(2)
        summary = summarize(study,record)
        write_json(study/'summary.json',summary)
        record.update(status=summary['status'],elapsed_this_invocation_seconds=monotonic()-started,
                      artifact_bytes=sum(p.stat().st_size for p in study.rglob('*') if p.is_file()))
        write_json(study/'manifest.json',record)
        return summary
    except BaseException as exc:
        record.update(status='interrupted',error=f'{type(exc).__name__}: {exc}')
        for process,log,_,attempt in active.values():
            process.terminate()
            process.wait()
            log.close()
            attempt.update(status='interrupted',unobserved_steps_upper_bound=attempt['budget'])
        write_json(study/'manifest.json',record)
        raise
    finally:
        lock.unlink(missing_ok=True)


def verify(root=ROOT):
    from .unseen_audit import audit_run
    benchmark,study,record = prepare(root)
    summary = read_json(study/'summary.json')
    expected = summarize(study,record)
    expected['status'] = summary['status']
    if summary != expected:
        raise ValueError('Aggregate results differ from registered run support')
    audits = []
    for system in summary['runs']:
        attempt = next(a for a in record['attempts'] if a['run_id']==system['run_id'])
        settings = run_settings(record['settings'],system['phase'],system['algorithm'],
                                system['training_seed'],attempt['attempt'])
        audits.append(audit_run(benchmark,study,system,settings,record['source_hash']))
    reference_dir = root/'artifacts/studies'/PILOT
    if (sha(reference_dir/'summary.json') != record['references']['summary_sha256']
            or any(sha(reference_dir/f'{name}.jsonl.gz') != value
                   for name,value in record['references']['files'].items())):
        raise ValueError('Reused reference evidence changed')
    # The M10 audit checked every saved reference action and declared case order.
    prior = read_json(reference_dir/'verification.json')
    if prior['status'] != 'passed' or prior['signature'] != benchmark.signature:
        raise ValueError('Reference action audit is unavailable')
    result = {'status':'passed','signature':benchmark.signature,'runs':audits,
              'training_source_hash':record['source_hash'],'verification_source_hash':source_hash(root),
              'reused_reference_actions':prior['reference_actions'],'final_policy_executions':0,
              'total_audited_actions':sum(r['validation_actions']+r['familiar_actions']+
                                          r['training_trace_actions'] for r in audits)}
    write_json(study/'verification.json',result)
    summary['status'] = record['status'] = 'verified'
    write_json(study/'summary.json',summary)
    write_json(study/'manifest.json',record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','worker','verify'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    result = {'run':execute,'verify':verify}.get(args.command)
    output = result(args.root) if result else worker(args.root,args.run_id)
    print(json.dumps({k:output[k] for k in ('status','total_training_interactions','total_audited_actions')
                      if k in output},indent=2))


if __name__=='__main__':
    main()
