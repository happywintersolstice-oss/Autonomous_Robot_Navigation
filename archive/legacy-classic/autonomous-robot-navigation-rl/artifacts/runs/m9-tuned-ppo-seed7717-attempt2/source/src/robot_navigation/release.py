"""Explicit final-evaluation freeze; generic training/UI split loaders stay sealed."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import shutil

from .agents import BFSPolicy, RandomPolicy, load_policy, viewer_checkpoint
from .core import ROOT, digest, load_task, read_json, write_json
from .dashboard import PART_A_SEEDS, PART_B_SEEDS, system_runs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registry():
    records=[]
    for algorithm,name in (('q-learning','Q-learning'),('sarsa','SARSA'),('dqn','DQN')):
        records += [{'id':f'm3-{algorithm}-seed{s}','group':f'Part A {name}','kind':'learned',
                     'seed':s,'evaluation_seed':0} for s in PART_A_SEEDS]
    for name,runs in system_runs().items():
        if name=='DQN Part A':
            continue
        for run in runs:
            records.append({'id':run,'group':name,'kind':'voting' if 'Voting' in name else 'learned',
                            'seed':None if 'Voting' in name else int(run.rsplit('seed',1)[1]),'evaluation_seed':0})
    records += [{'id':f'random-seed{s}','group':'Random reference','kind':'random',
                 'seed':s,'evaluation_seed':s} for s in PART_A_SEEDS]
    records.append({'id':'bfs','group':'BFS reference','kind':'bfs','seed':None,'evaluation_seed':0})
    return records


def freeze_release(study, root=ROOT):
    """Copy and seal all inputs before parsing test scenarios or executing policies."""
    study=Path(study)
    study.mkdir(parents=True,exist_ok=False)
    task=load_task(root)
    files=set()
    for folder,pattern in (('src','*.py'),('app','*.py'),('scripts','*.py'),('configs','*.yaml'),('data','*.json')):
        files.update((root/folder).rglob(pattern))
    files.update(root / name for name in ('requirements-lock.txt','pyproject.toml','docs/milestone6_protocol.md'))
    records=registry()
    estimated_seconds=0.;estimated_trace_bytes=0
    for record in records:
        if record['kind'] not in ('learned','voting'):
            continue
        directory=root / 'artifacts/runs' / record['id']
        manifest=read_json(directory/'manifest.json')
        if manifest['status']!='complete':
            raise ValueError('Cannot freeze incomplete model run')
        task.check(manifest['signature'])
        checkpoint=viewer_checkpoint(directory)
        load_policy(checkpoint,task)
        record['checkpoint']=checkpoint.relative_to(root).as_posix()
        record['selected_interactions']=manifest.get('selected_interactions')
        files.update((checkpoint,checkpoint.with_suffix('.json'),directory/'configuration.json',directory/'manifest.json'))
        validation=next(iter(read_json(directory/'validation.json').values()))
        estimated_seconds+=validation['evaluation_seconds']
        estimated_trace_bytes+=sum(p.stat().st_size for p in directory.glob('*_trajectories.jsonl.gz'))
    # Voting references retain their original relative layout; all three members
    # already occur in the tuned confirmation matrix.
    frozen={p.relative_to(root).as_posix():sha(p) for p in sorted(files)}
    for relative in frozen:
        target=study/'snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/relative,target)
    report=read_json(root/'data/scenarios/benchmark_report.json')
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'signature':task.signature,
        'selected_system':'m4-tuned-q-learning-seed101','selected_control':'m4-control-q-learning-seed101',
        'selection_basis':'Predeclared Part B family selection and seed101; unchanged after validation robustness',
        'test_scenario_hash':report['splits']['test']['scenario_hash'],'test_scenario_count':report['splits']['test']['count'],
        'records':records,'files':frozen,'source_base_revision':'f963f37 plus exact frozen local source files',
        'dependencies':{p:version(p) for p in ('numpy','gymnasium','matplotlib','streamlit','stable-baselines3','torch','pytest')},
        'python':platform.python_version(),'platform':platform.platform(),
        'targets':{'success_rate_at_least':.8,'collision_rate_at_most':.02,'spl_at_least':.75},
        'limits':{'wall_seconds':900,'evaluation_storage_bytes':500*1024**2},
        'estimate':{'learned_validation_evaluation_seconds':estimated_seconds,
                    'conservative_evaluation_seconds':estimated_seconds*2+30,
                    'conservative_trace_bytes':int(estimated_trace_bytes*2+10*1024**2),
                    'method':'Saved full-validation wall time/trace bytes, doubled plus reference allowance; audit separate'},
        'test_policy_executions_before_freeze':0}
    write_json(study/'freeze.json',record)
    write_json(study/'seal.json',{'freeze_digest':digest(record)})
    write_json(study/'status.json',{'status':'frozen','final_evaluation_complete':False})
    return digest(record)


def verify_freeze(study, expected_digest, root=ROOT):
    record=read_json(Path(study)/'freeze.json')
    if digest(record)!=expected_digest or read_json(Path(study)/'seal.json')['freeze_digest']!=expected_digest:
        raise ValueError('Final freeze digest mismatch')
    for relative,expected in record['files'].items():
        if sha(root/relative)!=expected or sha(Path(study)/'snapshot'/relative)!=expected:
            raise ValueError(f'Frozen input changed: {relative}')
    return record


def frozen_policy(study, record, task):
    if record['kind']=='bfs':
        return BFSPolicy(task.grid)
    if record['kind']=='random':
        return RandomPolicy(record['seed'])
    return load_policy(Path(study)/'snapshot'/record['checkpoint'],task)


def load_final_scenarios(study, freeze):
    snapshot=Path(study)/'snapshot'
    task=load_task(snapshot)
    task.check(freeze['signature'])
    record=read_json(snapshot/'data/scenarios/test.json')
    task.check(record['signature'])
    if (record['split']!='test' or digest(record['scenarios'])!=freeze['test_scenario_hash']
            or len(record['scenarios'])!=freeze['test_scenario_count']):
        raise ValueError('Frozen final scenario identity mismatch')
    return task,record['scenarios']


def final_summary(root=ROOT):
    study=root/'artifacts/studies/final-v1'
    try:
        if not read_json(study/'status.json').get('final_evaluation_complete'):
            return None
        return read_json(study/'summary.json')
    except FileNotFoundError:
        return None
