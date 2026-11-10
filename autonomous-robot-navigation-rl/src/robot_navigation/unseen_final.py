"""Sealed, inference-only evaluation of the complete M11 roster on final maps."""
import argparse
from datetime import datetime, timezone
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from time import perf_counter

import numpy as np
import yaml

from .agents import BFSPolicy, RandomPolicy
from .core import ROOT, digest, read_json, write_json
from .episodes import aggregate
from .release import sha, verify_freeze
from .shared_features import FEATURE_VERSION
from .unseen_agents import load_map_policy, map_adapter
from .unseen_audit import _audit_validation, load_recording, replay_task
from .unseen_benchmark import BENCHMARK, MapBenchmark, map_observation
from .unseen_main import METRICS
from .unseen_study import evaluate
from .workflow import source_hash

STUDY = 'unseen-map-final-v1'
INTERVAL_METRICS = ('success_rate','collision_rate','timeout_rate','spl')


def sealed_record(study):
    study = Path(study)
    seal = read_json(study/'seal.json')['freeze_digest']
    record = verify_freeze(study,seal,root=study/'snapshot')
    if record['study_id'] != STUDY:
        raise ValueError('Unseen final release identity mismatch')
    if 'source_hash' in record and source_hash(study/'snapshot')!=record['source_hash']:
        raise ValueError('Sealed execution source hash mismatch')
    return record,seal


def publish_freeze(study, record, files):
    """Publish a complete checksum manifest last, before any final accessor exists."""
    study = Path(study)
    for relative,path in sorted(files.items()):
        target = study/'snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    record = {**record,'files':{name:sha(study/'snapshot'/name) for name in sorted(files)}}
    write_json(study/'freeze.json',record)
    write_json(study/'seal.json',{'freeze_digest':digest(record)})
    write_json(study/'status.json',{'status':'frozen','final_evaluation_complete':False})
    sealed_record(study)
    return record


def freeze(root=ROOT):
    config = yaml.safe_load((root/'configs/unseen_final.yaml').read_text(encoding='utf-8'))
    if (config['study_id'] != STUDY or config['algorithms'] != ['q-learning','sarsa','dqn','ppo']
            or config['training_seeds'] != [12101,12203,12301]
            or len(set(config['random_seeds'])) != 3 or config['bootstrap_samples']<1
            or not 0<config['confidence_level']<1):
        raise ValueError('Final evaluation matrix/configuration mismatch')
    study = root/'artifacts/studies'/STUDY
    if study.exists():
        record,_ = sealed_record(study)
        if record['configuration'] != config or sha(root/'docs/milestone12_protocol.md') != record['protocol_hash']:
            raise ValueError('Existing final release protocol/configuration changed')
        return study
    benchmark = MapBenchmark(root)
    development = root/'artifacts/studies'/config['development_study']
    main = read_json(development/'summary.json')
    manifest = read_json(development/'manifest.json')
    verification = read_json(development/'verification.json')
    if (main['status'] != manifest['status'] or main['status']!='verified'
            or verification['status']!='passed' or main['signature']!=benchmark.signature
            or source_hash(development/'source')!=manifest['source_hash']):
        raise ValueError('The final roster requires verified compatible development evidence')
    runs = [r for r in main['runs'] if r['phase']=='main']
    if (len(runs)!=12 or {(r['algorithm'],r['training_seed']) for r in runs} !=
            {(a,s) for a in config['algorithms'] for s in config['training_seeds']}):
        raise ValueError('The complete three-seed/four-family roster is required')
    files = {}
    for folder,pattern in (('src','*.py'),('app','*.py'),('configs','*.yaml')):
        files.update({p.relative_to(root).as_posix():p for p in (root/folder).rglob(pattern)})
    for name in ('requirements-lock.txt','pyproject.toml','docs/milestone12_protocol.md',
                 'data/maps/bottleneck-v1.json'):
        files[name] = root/name
    # Opaque copying/hashing is the only handling of final bytes before the seal.
    files.update({p.relative_to(root).as_posix():p for p in benchmark.directory.glob('*.json')})
    for name in ('summary.json','manifest.json','verification.json'):
        path = development/name
        files[path.relative_to(root).as_posix()] = path
    records = []
    for row in runs:
        directory = development/'runs'/row['run_id']
        run = read_json(directory/'manifest.json')
        checkpoint = directory/run['viewer_checkpoint']
        policy = load_map_policy(checkpoint,benchmark)
        if (run['status']!='complete' or run['algorithm']!=row['algorithm']
                or run['training_seed']!=row['training_seed'] or run['training_interactions']!=200000
                or run['source_hash']!=manifest['source_hash'] or policy.feature_version!=FEATURE_VERSION
                or policy.algorithm!=row['algorithm']):
            raise ValueError('Selected-policy provenance mismatch')
        for path in (checkpoint,checkpoint.with_suffix('.json'),directory/'manifest.json',directory/'monitoring.json'):
            files[path.relative_to(root).as_posix()] = path
        records.append({'run_id':row['run_id'],'kind':'learned','algorithm':row['algorithm'],
            'training_seed':row['training_seed'],'evaluation_seed':config['evaluation_seed'],
            'checkpoint':checkpoint.relative_to(root).as_posix(),'checkpoint_sha256':sha(checkpoint),
            'selected_interactions':row['selected_interactions'],'development':row['validation'],
            'training_seconds':row['training_seconds'],'training_interactions':row['training_interactions'],
            'checkpoint_bytes':row['checkpoint_bytes'],'optimized_parameter_count':row['optimized_parameter_count']})
    records += [{'run_id':'bfs','kind':'reference','algorithm':'bfs','evaluation_seed':0}]
    records += [{'run_id':f'random-seed{s}','kind':'reference','algorithm':'random','evaluation_seed':s}
                for s in config['random_seeds']]
    record = {'study_id':STUDY,'configuration':config,'signature':benchmark.signature,'records':records,
        'created_utc':datetime.now(timezone.utc).isoformat(),'source_hash':source_hash(root),
        'protocol_hash':sha(root/'docs/milestone12_protocol.md'),'representation':FEATURE_VERSION,
        'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),
        'logical_processors':os.cpu_count(),'device':'cpu','torch_threads':1,
        'dependencies':{p:version(p) for p in ('numpy','gymnasium','torch','stable-baselines3','streamlit')},
        'final_policy_executions_before_freeze':0}
    study.mkdir(parents=True,exist_ok=False)
    publish_freeze(study,record,files)
    return study


class FinalBenchmark(MapBenchmark):
    final_authorized = True

    def __init__(self,study):
        self.release, self.release_digest = sealed_record(study)
        super().__init__(Path(study)/'snapshot')
        if self.signature != self.release['signature']:
            raise ValueError('Sealed benchmark signature mismatch')
        self.final_layouts = self._read('test')
        if (len(self.final_layouts)!=self.config['layout_counts']['test']
                or any(len(m['scenarios'])!=3*self.config['scenarios_per_band'] for m in self.final_layouts)):
            raise ValueError('Final support differs from the frozen benchmark')

    def layouts(self,split):
        return self.final_layouts if split=='test' else super().layouts(split)


def validate_result(directory,record,release_digest):
    result = read_json(directory/'manifest.json')
    if (result['status']!='complete' or result['record']!=record
            or result['release_digest']!=release_digest):
        raise ValueError('Completed final result identity mismatch')
    for name,expected in result['files'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path)!=expected:
            raise ValueError('Final result file checksum mismatch')
    if result['final'] != read_json(directory/'final.json'):
        raise ValueError('Final result and summary differ')
    return result


def evaluate_record(study,benchmark,record,deadline=None):
    """Publish complete results atomically and retain interrupted attempts."""
    directory = study/'runs'/record['run_id']
    if directory.exists():
        return validate_result(directory,record,benchmark.release_digest)
    pending = study/'pending'/record['run_id']
    if pending.exists():
        retained = study/'incomplete'/f'{record["run_id"]}-{len(list((study/"incomplete").glob("*")))+1}'
        retained.parent.mkdir(parents=True,exist_ok=True)
        pending.rename(retained)
    pending.mkdir(parents=True)
    started = perf_counter()
    policy = (load_map_policy(study/'snapshot'/record['checkpoint'],benchmark)
              if record['kind']=='learned' else record['algorithm'])
    initial_fingerprint = policy.state_fingerprint() if record['kind']=='learned' else None
    summary,rows,examples = evaluate(benchmark,benchmark.cases('test'),policy,record['evaluation_seed'],
        split='test',output=pending/'trajectories.jsonl.gz',final=True,deadline=deadline)
    latency = None
    if record['kind']=='learned':
        layout,scenario = benchmark.cases('test')[0]
        probe = map_observation([*scenario['start'],*scenario['goal']],benchmark.task(layout).grid)
        before = policy.state_fingerprint()
        timer = perf_counter()
        for _ in range(100):
            policy.act(probe)
        latency = (perf_counter()-timer)*1e4
        if before!=policy.state_fingerprint():
            raise ValueError('Final latency probe changed learned state')
        if initial_fingerprint!=policy.state_fingerprint():
            raise ValueError('Final evaluation changed the loaded learned state')
    write_json(pending/'final.json',summary)
    write_json(pending/'episodes.json',rows)
    replays = []
    for index,episode in enumerate(examples):
        path = pending/f'replays/final-{index:03d}.json'
        write_json(path,episode)
        replays.append({'path':path.relative_to(pending).as_posix(),'sha256':sha(path),
                        'layout_id':episode['layout_id'],'outcome':episode['metrics']['outcome']})
    result = {'status':'complete','kind':'frozen-final-evaluation','run_id':record['run_id'],
        'signature':benchmark.signature,'release_digest':benchmark.release_digest,'record':record,
        'representation':FEATURE_VERSION if record['kind']=='learned' else None,
        'replays':replays,'final':summary,'evaluation_seconds':summary['evaluation_seconds'],
        'total_seconds':perf_counter()-started,'decision_microseconds':latency,
        'fingerprint_before':initial_fingerprint,'fingerprint_after':initial_fingerprint,
        'files':{p.relative_to(pending).as_posix():sha(p) for p in pending.rglob('*') if p.is_file()}}
    write_json(pending/'manifest.json',result)
    sealed_record(study)
    directory.parent.mkdir(exist_ok=True)
    pending.rename(directory)
    return result


def analyze(results,config):
    systems = []
    layout_names = sorted(results[0]['final']['by_map'])
    rng = np.random.default_rng(config['bootstrap_seed'])
    draws = rng.integers(len(layout_names),size=(config['bootstrap_samples'],len(layout_names)))
    tail = (1-config['confidence_level'])/2
    for algorithm in [*config['algorithms'],'bfs','random']:
        group = [r for r in results if r['record']['algorithm']==algorithm]
        if not group or any(sorted(r['final']['by_map'])!=layout_names for r in group):
            raise ValueError('Every method requires identical final layouts')
        stats = {}
        for metric in METRICS:
            values = [r['final'][metric] for r in group if r['final'][metric] is not None]
            stats[metric] = {'mean':float(np.mean(values)) if values else None,
                'sd':float(np.std(values,ddof=1)) if len(values)>1 else None,'defined_seeds':len(values)}
            if metric in INTERVAL_METRICS:
                maps = np.array([np.mean([r['final']['by_map'][name][metric] for r in group])
                                 for name in layout_names])
                stats[metric]['layout_ci95'] = np.quantile(maps[draws].mean(axis=1),[tail,1-tail]).tolist()
        by_map = {name:{'episodes':sum(r['final']['by_map'][name]['episodes'] for r in group),
            'counts':{k:sum(r['final']['by_map'][name]['counts'][k] for r in group)
                      for k in ('success','collision','timeout')},
            **{metric:float(np.mean([r['final']['by_map'][name][metric] for r in group]))
               for metric in INTERVAL_METRICS}} for name in layout_names}
        systems.append({'algorithm':algorithm,'seed_count':len(group),'statistics':stats,'by_map':by_map,
            'episodes':sum(r['final']['episodes'] for r in group),
            'counts':{k:sum(r['final']['counts'][k] for r in group) for k in ('success','collision','timeout')},
            'evaluation_seconds':sum(r['evaluation_seconds'] for r in group)})
    return systems


def summarize(study,release,seal):
    results = [validate_result(study/'runs'/r['run_id'],r,seal) for r in release['records']]
    rows = [{'run_id':r['run_id'],'algorithm':r['record']['algorithm'],'phase':'final',
             'training_seed':r['record'].get('training_seed'),'record':r['record'],'final':r['final'],
             'evaluation_seconds':r['evaluation_seconds'],'decision_microseconds':r['decision_microseconds']}
            for r in results]
    return {'status':'complete; audit pending','signature':release['signature'],'release_digest':seal,
        'systems':analyze(results,release['configuration']),
        'runs':[r for r in rows if r['record']['kind']=='learned'],
        'references':[r for r in rows if r['record']['kind']=='reference'],
        'total_episodes':sum(r['final']['episodes'] for r in results),
        'total_interactions':sum(r['final']['interactions'] for r in results),
        'new_training_interactions':0,'final_test_evaluated':True,
        'scope':'independent final layouts; frozen M11 controllers; observed holdout after this release',
        'incomplete_attempts':[p.name for p in (study/'incomplete').glob('*')]}


def execute(study):
    study = Path(study).resolve()
    if not Path(__file__).resolve().is_relative_to(study/'snapshot/src'):
        raise ValueError('Final execution requires the archived source')
    release,seal = sealed_record(study)
    lock = study/'run.lock'
    with lock.open('x',encoding='utf-8') as file:
        file.write(str(os.getpid()))
    started = perf_counter()
    try:
        benchmark = FinalBenchmark(study)
        for record in release['records']:
            if perf_counter()-started>release['configuration']['wall_seconds']:
                raise TimeoutError('Final evaluation exceeded its wall limit')
            evaluate_record(study,benchmark,record,started+release['configuration']['wall_seconds'])
            if sum(p.stat().st_size for p in study.rglob('*') if p.is_file())>release['configuration']['artifact_bytes']:
                raise RuntimeError('Final release exceeded its storage limit')
            write_json(study/'status.json',{'status':'evaluating','last_completed':record['run_id'],
                                          'final_evaluation_complete':False})
            print(f'{record["run_id"]} complete',flush=True)
        summary = summarize(study,release,seal)
        sealed_record(study)
        elapsed = perf_counter()-started
        if elapsed>release['configuration']['wall_seconds']:
            raise TimeoutError('Final evaluation exceeded its total wall limit')
        write_json(study/'summary.json',summary)
        write_json(study/'status.json',{'status':summary['status'],'evaluation_seconds':elapsed,
            'final_evaluation_complete':True,'incomplete_attempts':summary['incomplete_attempts']})
        return summary
    except BaseException as exc:
        write_json(study/'status.json',{'status':'interrupted','error':f'{type(exc).__name__}: {exc}',
                                      'final_evaluation_complete':False})
        raise
    finally:
        lock.unlink(missing_ok=True)


def run(root=ROOT):
    study = freeze(root)
    state = read_json(study/'status.json')
    if state['status']=='verified':
        release,seal = sealed_record(study)
        summarize(study,release,seal)
        return read_json(study/'summary.json')
    env = {**os.environ,'PYTHONPATH':str(study/'snapshot/src'),
           'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    subprocess.run([os.sys.executable,'-X','utf8','-m','robot_navigation.unseen_final','execute',
                    '--study',str(study)],cwd=study/'snapshot',env=env,check=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    return read_json(study/'summary.json')


def verify(study):
    study = Path(study)
    started = perf_counter()
    release,seal = sealed_record(study)
    benchmark = FinalBenchmark(study)
    saved = read_json(study/'summary.json')
    expected = summarize(study,release,seal)
    expected['status'] = saved['status']
    if expected!=saved:
        raise ValueError('Final aggregates or intervals do not reconcile')
    audits = []
    for record in release['records']:
        directory = study/'runs'/record['run_id']
        result = validate_result(directory,record,seal)
        if record['kind']=='learned':
            policy = load_map_policy(study/'snapshot'/record['checkpoint'],benchmark)
            if policy.state_fingerprint()!=result['fingerprint_before'] or result['fingerprint_before']!=result['fingerprint_after']:
                raise ValueError('Final evaluation fingerprint differs from the frozen policy')
            actions = _audit_validation(benchmark,directory/'trajectories.jsonl.gz',policy,'test',result['final'],final=True)
        else:
            random = RandomPolicy(record['evaluation_seed'])
            rows,actions = [],0
            with gzip.open(directory/'trajectories.jsonl.gz','rt',encoding='utf-8') as file:
                for (layout,scenario),line in zip(benchmark.cases('test'),file,strict=True):
                    episode = json.loads(line)
                    task = replay_task(benchmark,episode,allow_final=True)
                    if episode['scenario']!=scenario or episode['layout_id']!=layout['map_id']:
                        raise ValueError('Final reference case order mismatch')
                    policy = BFSPolicy(task.grid) if record['algorithm']=='bfs' else random
                    if any(policy.act(t['observation'])!=t['action'] for t in episode['transitions']):
                        raise ValueError('Final reference action mismatch')
                    rows.append(episode['metrics'])
                    actions += len(episode['transitions'])
            observed = aggregate(rows)
            if any(observed[k]!=result['final'][k] for k in observed):
                raise ValueError('Final reference summary mismatch')
        for item in result['replays']:
            load_recording(benchmark,directory,item)
        audits.append({'run_id':record['run_id'],'actions':actions,'episodes':result['final']['episodes'],
                       'inference_unchanged':record['kind']=='learned'})
    bfs = next(s for s in saved['systems'] if s['algorithm']=='bfs')
    if bfs['statistics']['success_rate']['mean']!=1 or bfs['statistics']['spl']['mean']!=1:
        raise ValueError('Final BFS reference must be optimal on every reachable case')
    sealed_record(study)
    result = {'status':'passed','signature':benchmark.signature,'release_digest':seal,'runs':audits,
        'total_audited_actions':sum(r['actions'] for r in audits),'episodes':sum(r['episodes'] for r in audits),
        'audit_seconds':perf_counter()-started,'execution_source_hash':release['source_hash'],
        'verification_source_hash':source_hash(ROOT),'new_training_interactions':0}
    saved['status'] = 'verified'
    write_json(study/'summary.json',saved)
    result['summary_sha256'] = sha(study/'summary.json')
    write_json(study/'verification.json',result)
    status = read_json(study/'status.json')
    status.update(status='verified',final_evaluation_complete=True)
    write_json(study/'status.json',status)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','run','execute','verify'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--study',type=Path)
    args = parser.parse_args()
    study = args.study or args.root/'artifacts/studies'/STUDY
    if args.command=='freeze':
        print(freeze(args.root))
        return
    result = {'run':lambda:run(args.root),'execute':lambda:execute(study),'verify':lambda:verify(study)}[args.command]()
    print(json.dumps({k:result[k] for k in ('status','total_episodes','total_interactions','total_audited_actions')
                      if k in result},indent=2))


if __name__=='__main__':
    main()
