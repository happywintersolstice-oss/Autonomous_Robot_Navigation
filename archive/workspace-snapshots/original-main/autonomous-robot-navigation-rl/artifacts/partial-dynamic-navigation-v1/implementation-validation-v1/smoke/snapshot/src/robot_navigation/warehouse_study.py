"""Registered, matched four-algorithm warehouse training and saved-result readers."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
from datetime import datetime,timezone
import gzip
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
from time import perf_counter

import numpy as np
import yaml

from .agents import QLearning,SARSA,load_policy
from .core import ROOT,NavigationEnv,read_json,write_json,digest
from .episodes import aggregate,validate_replay
from .release import sha
from .training import train_loop
from .training_traces import TrainingTraces
from .warehouse import load as load_floor
from .workflow import evaluate

STUDY = 'warehouse-four-algorithm-v1'
FAMILIES = {'q-learning':'Q-learning','sarsa':'SARSA','dqn':'DQN','ppo':'PPO'}
TITLE = 'Autonomous Warehouse Robot Navigation: A Comparative Study of Q-learning, SARSA, DQN and PPO'
ENGINE = ('warehouse_study.py','warehouse.py','core.py','agents.py','episodes.py','training.py',
          'training_traces.py','part_a.py','dqn.py','ppo.py','ppo_config.py','observation_encoding.py','workflow.py')


def plan(root=ROOT):
    p = yaml.safe_load((root/'configs/warehouse_comparison.yaml').read_text(encoding='utf-8'))
    if (p['study_id']!=STUDY or p['map_id']!='warehouse-delivery-v1'
            or p['algorithms']!=list(FAMILIES) or p['seeds']!=[19411,19421,19431]
            or p['common']['interactions']!=600000 or p['workers']!=2):
        raise ValueError('Warehouse comparison differs from its registered matrix')
    return p


def location(root=ROOT):
    return root/'artifacts/warehouse'/STUDY


def monitoring_routes(routes):
    selected = [job['scenario'] for job in routes['jobs']]
    ids = {s['id'] for s in selected}
    remaining = sorted((s for s in routes['validation'] if s['id'] not in ids),
                       key=lambda s:(s['shortest_length'],s['id']))
    # Evenly cover the sorted route-difficulty range, independent of outcomes.
    indices = np.linspace(0,len(remaining)-1,35-len(selected),dtype=int)
    return selected+[remaining[i] for i in indices]


def freeze(root=ROOT):
    p = plan(root)
    task,layout,routes = load_floor(root)
    folder = location(root)
    files = [root/'configs/warehouse_comparison.yaml',root/'configs/warehouse_demo.yaml',
             root/'docs/warehouse_comparison_protocol.md',
             root/'data/warehouse/warehouse-delivery-v1/map.json',
             root/'data/warehouse/warehouse-delivery-v1/routes.json',
             *(root/'src/robot_navigation'/name for name in ENGINE)]
    expected = {path.relative_to(root).as_posix():sha(path) for path in files}
    if folder.exists():
        record = read_json(folder/'freeze.json')
        if record['files']!=expected or record['plan_hash']!=digest(p):
            raise ValueError('Registered warehouse inputs changed; use a new study version')
        return record
    folder.mkdir(parents=True)
    for path in files:
        target = folder/'snapshot'/path.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    record = {'study_id':STUDY,'title':TITLE,'plan_hash':digest(p),'files':expected,
        'signature':task.signature,'created_utc':datetime.now(timezone.utc).isoformat(),
        'seeds':p['seeds'],'algorithms':p['algorithms'],'interactions_per_run':600000,
        'training_pairs':len(routes['train']),'assessment_pairs':len(routes['validation']),
        'monitoring':monitoring_routes(routes),'selection':'fixed final-budget checkpoint; no selection',
        'scope':'Inspected development routes on one familiar warehouse; not a new final or unseen-map evaluation',
        'python':platform.python_version(),'dependencies':{x:version(x) for x in ('numpy','gymnasium','torch','stable-baselines3')},
        'workers':2,'torch_threads_per_worker':1}
    write_json(folder/'freeze.json',record)
    write_json(folder/'seal.json',{'freeze_digest':digest(record)})
    write_json(folder/'status.json',{'status':'registered','completed_runs':0,'expected_runs':12})
    return record


def checked_freeze(root=ROOT):
    folder = location(root)
    record = read_json(folder/'freeze.json')
    if digest(record)!=read_json(folder/'seal.json')['freeze_digest']:
        raise ValueError('Warehouse comparison seal changed')
    for name,value in record['files'].items():
        if sha(root/name)!=value or sha(folder/'snapshot'/name)!=value:
            raise ValueError(f'Warehouse comparison input changed: {name}')
    return record


def configuration(algorithm,seed,p):
    section = 'tabular' if algorithm in ('q-learning','sarsa') else algorithm
    return {**p['common'],**p[section],'algorithm':algorithm,'seed':seed,
            'run_id':f'warehouse-{algorithm}-seed{seed}'}


def run_one(algorithm,seed,root=ROOT):
    root = Path(root)
    record = checked_freeze(root)
    p = plan(root)
    c = configuration(algorithm,seed,p)
    task,_,routes = load_floor(root)
    folder = location(root)/'runs'/c['run_id']
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'status.json',{'status':'training','algorithm':algorithm,'seed':seed})
    write_json(folder/'configuration.json',c)
    trace = TrainingTraces(task,folder,c)
    logs = []
    monitor_seconds = 0.
    started = perf_counter()
    policy = None
    def monitor(step,phase):
        nonlocal monitor_seconds
        before = perf_counter()
        result,_,_ = evaluate(task,record['monitoring'],policy,0,'fixed warehouse development monitoring')
        logs.append({'training_interactions':int(step),'phase':phase,**result})
        write_json(folder/'monitoring.json',logs)
        monitor_seconds += perf_counter()-before
        print(f"{c['run_id']}: {step:,}/{c['interactions']:,} actions; monitoring {result['success_rate']:.1%}",flush=True)
    try:
        with (folder/'training.csv').open('w',encoding='utf-8',newline='') as file:
            writer = csv.DictWriter(file,fieldnames=['episode','scenario_id','end_interaction','steps','return','outcome','epsilon'])
            writer.writeheader()
            if algorithm in ('q-learning','sarsa'):
                policy = (QLearning if algorithm=='q-learning' else SARSA)(task,f'{algorithm}/{c["run_id"]}')
                visitation,training = train_loop(task,routes['train'],policy,c,
                    on_monitor=lambda step,visits:monitor(step,'after tabular update'),
                    on_episode=writer.writerow,trace=trace)
            else:
                from stable_baselines3.common.callbacks import BaseCallback
                from .part_a import TrainingLedger
                ledger = TrainingLedger(NavigationEnv(task,routes['train']),writer.writerow,trace)
                if algorithm=='dqn':
                    from .dqn import DQNPolicy,build_model
                    model = build_model(ledger,c)
                    policy = DQNPolicy(task,model,f'dqn/{c["run_id"]}')
                else:
                    from .ppo import PPOPolicy,build_model
                    from .ppo_config import validate_config
                    validate_config(c)
                    model = build_model(ledger,c)
                    policy = PPOPolicy(task,model,f'ppo/{c["run_id"]}')
                class Monitor(BaseCallback):
                    def _on_step(self):
                        if perf_counter()-started-monitor_seconds>c['max_wall_seconds']:
                            raise TimeoutError('Warehouse training exceeded its registered run limit')
                        if algorithm=='dqn' and self.num_timesteps%c['monitor_every']==0 and self.num_timesteps<c['interactions']:
                            monitor(self.num_timesteps,'DQN callback before boundary replay/gradient update')
                            file.flush()
                        return True
                    def _on_rollout_start(self):
                        if algorithm=='ppo' and self.num_timesteps and self.num_timesteps%c['monitor_every']==0:
                            monitor(self.num_timesteps,'after preceding PPO rollout update')
                            file.flush()
                    def _on_training_end(self):
                        monitor(self.num_timesteps,'after final scheduled update')
                model.learn(total_timesteps=c['interactions'],callback=Monitor())
                if model.num_timesteps!=c['interactions'] or ledger.interactions!=c['interactions']:
                    raise ValueError('Warehouse neural training budget mismatch')
                visitation,training = ledger.visitation,ledger.report(perf_counter()-started-monitor_seconds)
                training['gradient_updates'] = model._n_updates
        traces = trace.finish()
        extension = '.zip' if algorithm in ('dqn','ppo') else '.npz'
        checkpoint = folder/f'final{extension}'
        policy.save(checkpoint,{'training_interactions':c['interactions'],'training_seed':seed,'study_id':STUDY})
        np.savez_compressed(folder/'visitation.npz',visitation=visitation)
        loaded = load_policy(checkpoint,task)
        assessment,metrics,_ = evaluate(task,routes['validation'],loaded,0,
            'Warehouse development assessment; familiar floor; no selection',visitation,
            folder/'assessment.jsonl.gz')
        audits = []
        with gzip.open(folder/'assessment.jsonl.gz','rt',encoding='utf-8') as file:
            for line in file:
                episode = json.loads(line)
                validate_replay(task,episode)
                audits.append(episode)
        if [e['scenario'] for e in audits]!=routes['validation'] or aggregate([e['metrics'] for e in audits])['counts']!=assessment['counts']:
            raise ValueError('Warehouse recorded assessment coverage/metrics mismatch')
        loops = sum(e['metrics']['outcome']=='timeout' and len({tuple(t['next_observation'][:2]) for t in e['transitions']})<len(e['transitions']) for e in audits)
        result = {'algorithm':algorithm,'label':FAMILIES[algorithm],'seed':seed,'run_id':c['run_id'],
            'signature':task.signature,'training':training,'monitoring_seconds':monitor_seconds,
            'assessment':assessment,'looping_timeouts':loops,'checkpoint':checkpoint.name,
            'checkpoint_bytes':checkpoint.stat().st_size,'training_traces':traces,
            'selection':'fixed final-budget checkpoint; no seed or checkpoint selection',
            'assessment_audit':{'episodes':len(audits),'actions':sum(len(e['transitions']) for e in audits),'status':'passed'}}
        write_json(folder/'summary.json',result)
        write_json(folder/'outputs.json',{f.relative_to(folder).as_posix():sha(f) for f in folder.rglob('*') if f.is_file() and f.name!='status.json'})
        write_json(folder/'status.json',{'status':'verified','algorithm':algorithm,'seed':seed})
        print(f"VERIFIED {c['run_id']}: {assessment['success_rate']:.1%} completion",flush=True)
        return result
    except BaseException as exc:
        trace.finish('interrupted')
        write_json(folder/'status.json',{'status':'failed','algorithm':algorithm,'seed':seed,'reason':f'{type(exc).__name__}: {exc}'})
        raise


def checked_run(algorithm,seed,root=ROOT):
    folder = location(root)/'runs'/f'warehouse-{algorithm}-seed{seed}'
    if read_json(folder/'status.json')['status']!='verified':
        raise ValueError(f'Warehouse run incomplete: {folder.name}')
    for name,value in read_json(folder/'outputs.json').items():
        if sha(folder/name)!=value:
            raise ValueError(f'Warehouse run output changed: {folder.name}/{name}')
    return read_json(folder/'summary.json'),folder


def stats(values):
    values = [float(v) for v in values if v is not None]
    return {'mean':float(np.mean(values)) if values else None,
            'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,'contributing_seeds':len(values)}


def publish(root=ROOT):
    record = checked_freeze(root)
    p = plan(root)
    systems = {}
    for algorithm,label in FAMILIES.items():
        rows = [checked_run(algorithm,seed,root)[0] for seed in p['seeds']]
        systems[algorithm] = {'label':label,'seeds':p['seeds'],'seed_count':len(rows),'seed_results':rows,
            'metrics':{m:stats([r['assessment'][m] for r in rows]) for m in
                ('success_rate','collision_rate','timeout_rate','spl','mean_return','mean_successful_route_length')},
            'costs':{'training_seconds':stats([r['training']['training_seconds'] for r in rows]),
                     'decision_microseconds':stats([r['assessment']['mean_decision_microseconds'] for r in rows]),
                     'checkpoint_bytes':stats([r['checkpoint_bytes'] for r in rows])},
            'looping_timeouts':sum(r['looping_timeouts'] for r in rows)}
    result = {'status':'verified','title':TITLE,'study_id':STUDY,'signature':record['signature'],
        'systems':systems,'training_interactions_per_run':600000,'total_training_interactions':7200000,
        'assessment_routes_per_run':188,'scope':record['scope'],
        'aggregation':'Equal training-seed means and sample SD; conditional route means omit undefined zero-success seed values',
        'timing_scope':'Descriptive wall time with up to two concurrent CPU workers; not an isolated hardware benchmark',
        'default_playback_seed':p['seeds'][0],'default_algorithm':'q-learning; display order only, not a selected winner'}
    folder = location(root)
    write_json(folder/'summary.json',result)
    with (folder/'comparison.csv').open('w',encoding='utf-8',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=['algorithm','seeds','success_mean','success_sd','collision_mean','timeout_mean','spl_mean','spl_sd'])
        writer.writeheader()
        for algorithm,system in systems.items():
            m = system['metrics']
            writer.writerow({'algorithm':algorithm,'seeds':3,'success_mean':m['success_rate']['mean'],
                'success_sd':m['success_rate']['sample_sd'],'collision_mean':m['collision_rate']['mean'],
                'timeout_mean':m['timeout_rate']['mean'],'spl_mean':m['spl']['mean'],'spl_sd':m['spl']['sample_sd']})
    write_json(folder/'published.json',{'summary_sha256':sha(folder/'summary.json'),'comparison_sha256':sha(folder/'comparison.csv')})
    write_json(folder/'status.json',{'status':'verified','completed_runs':12,'expected_runs':12})
    return result


def saved(root=ROOT):
    folder = location(root)
    checked_freeze(root)
    if read_json(folder/'status.json')['status']!='verified':
        raise ValueError('Warehouse comparison is still running or incomplete')
    pin = read_json(folder/'published.json')
    if sha(folder/'summary.json')!=pin['summary_sha256'] or sha(folder/'comparison.csv')!=pin['comparison_sha256']:
        raise ValueError('Warehouse comparison publication checksum mismatch')
    return read_json(folder/'summary.json')


def episodes(algorithm,seed,root=ROOT):
    _,folder = checked_run(algorithm,seed,root)
    with gzip.open(folder/'assessment.jsonl.gz','rt',encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def run_all(root=ROOT):
    freeze(root)
    p = plan(root)
    todo = []
    for seed in p['seeds']:
        for algorithm in p['algorithms']:
            folder = location(root)/'runs'/f'warehouse-{algorithm}-seed{seed}'
            if folder.exists():
                checked_run(algorithm,seed,root)
            else:
                todo.append((algorithm,seed))
    with ProcessPoolExecutor(max_workers=p['workers']) as pool:
        jobs = {pool.submit(run_one,algorithm,seed,root):(algorithm,seed) for algorithm,seed in todo}
        for future in as_completed(jobs):
            future.result()
    return publish(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','run','publish'))
    args = parser.parse_args()
    result = {'freeze':freeze,'run':run_all,'publish':publish}[args.command]()
    print(json.dumps({k:v for k,v in result.items() if k not in ('systems','files','monitoring')},indent=2))


if __name__=='__main__':
    main()
