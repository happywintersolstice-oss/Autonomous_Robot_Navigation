"""Registered Phase 5B only: one seed, two feature arms, fixed A-only ceilings."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .core import ROOT, digest, read_json, write_json
from .release import sha
from .partial_agents import ALGORITHMS
from .partial_baselines import OnlinePlanner
from .partial_commands import read_records, require_gates
from .partial_contract import configuration, signature, MOVES
from .partial_evaluation import validate_replay
from .partial_feasibility import rollout, EpisodeAbort, audit_episode, _resource_usage
from .partial_feasibility_fixtures import cases
from .partial_features_scaled import ARMS, build_policy, load_scaled_policy, encode_scaled, parameter_norm
from .partial_pilot_diagnostics import evaluation_summary, training_summary, decision_values, preserve_random_state, model_diagnostics
from .partial_pilot_ledger import PilotLimit, PilotRecorder, ContinuousLinear, ActionStream
from .partial_training import validate_training_trace
from .partial_phase5b_fixtures import expanded_cases, audit_cases, strata, GoalDirectionScript
from .partial_phase5b_diagnostics import (historical_diagnosis, decision_diagnostic, feature_statistics,
    optimizer_state, UpdateRecorder, LinearUpdateRecorder, continuity_state, continuity_hash, save_continuation, distribution)

STUDY = 'phase5b-a-only-v1'
BRANCH = 'feature/phase5b-basic-navigation-diagnosis-v1'
BASE = 'd3822423ffd6fa3c83f9a26fc15808fa9103e36c'


def location(root=ROOT):
    return root/'artifacts/partial-dynamic-navigation-v1'/STUDY


def settings(root=ROOT):
    c = yaml.safe_load((root/'configs/partial_phase5b_v1.yaml').read_text(encoding='utf-8'))
    expected = dict(study_id=STUDY, protocol_version='partial-a-learning-diagnosis-v1',
        algorithms=list(ALGORITHMS), arms=list(ARMS), primary_seed=23021, fixture_seed=23022,
        evaluation_seed=0, checkpoints=[0,2000,6000], per_arm_actions=6000,
        total_training_actions=48000, evaluation_action_ceiling=100000,
        suite_wall_seconds=1800, per_arm_wall_seconds=300, artifact_byte_ceiling=512*1024**2,
        linear_epsilon_horizon=10000, model_config='configs/partial_feasibility_v1.yaml',
        screen=dict(original_successes=3,gain_over_b1_initial=1,gain_over_b0_final=1,
                    expanded_successes=12,expanded_gain_over_b0=4))
    if c != expected:
        raise ValueError('Phase 5B registered envelope changed')
    c['model'] = read_baseline_settings(root)['model']
    task = configuration(root)
    if any(task[k] != v for k,v in dict(max_steps=200,step_reward=-.01,success_bonus=10.,
                                      collision_penalty=-5.,extra_timeout_penalty=0.).items()):
        raise ValueError('A task/reward invariant changed')
    return c


def read_baseline_settings(root=ROOT):
    return yaml.safe_load((root/'configs/partial_feasibility_v1.yaml').read_text(encoding='utf-8'))


def source_inputs(root=ROOT):
    paths = list((root/'src/robot_navigation').glob('partial_*.py'))
    paths += list((root/'developer/tests').glob('test_partial_*.py'))
    paths += [root/p for p in ('configs/partial_phase5b_v1.yaml','configs/partial_feasibility_v1.yaml',
        'configs/partial_dynamic.yaml','docs/partial_phase5b_protocol.md','pyproject.toml','requirements-lock.txt')]
    paths += [root.parent/'CODEX_PHASE5B_BASIC_NAVIGATION_DIAGNOSIS.md']
    return {p.relative_to(root.parent).as_posix():sha(p) for p in sorted(paths)}


def junit(path):
    tree = ET.parse(path).getroot()
    suites = [tree] if tree.tag=='testsuite' else list(tree.findall('testsuite'))
    result = {k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if not result['tests'] or any(result[k] for k in ('failures','errors','skipped')):
        raise ValueError('Regression gate failed or skipped')
    return result


def historical_preservation(root=ROOT):
    stored=location(root)/'base-inventory.json'
    inventory = read_json(stored if stored.exists() else root/'.runtime/phase5b-base-inventory.json')
    changed = [name for name,h in inventory['files'].items() if sha(root.parent/name)!=h]
    if changed != ['.gitattributes'] and changed:
        raise ValueError(f'Pre-existing file changed: {changed}')
    # The sole permitted edit is an additive raw-byte preservation rule for the new brief.
    old = subprocess.check_output(['git','show',BASE+':.gitattributes'],cwd=root)
    new = (root.parent/'.gitattributes').read_bytes()
    if new.replace(b'\r\n',b'\n') != old.replace(b'\r\n',b'\n')+b'CODEX_PHASE5B_BASIC_NAVIGATION_DIAGNOSIS.md -text whitespace=cr-at-eol\n':
        raise ValueError('Unexpected existing Git attribute edit')
    return {'checked':len(inventory['files']),'historical_files_changed':[],
            'only_existing_file_edit':'.gitattributes: byte preservation for new Phase 5B brief'}


def diagnosis_path(root=ROOT):
    return root/'.runtime'/('phase5b-zero-learning-'+sha(root/'src/robot_navigation/partial_phase5b_diagnostics.py')[:12])


def diagnose(root=ROOT):
    target=diagnosis_path(root)
    if target.exists():
        pins=read_json(target/'hashes.json')
        if any(sha(target/p)!=h for p,h in pins.items()): raise ValueError('Zero-learning audit changed')
        result=read_json(target/'zero-learning-audit.json')
        if result.get('diagnostic_source_sha256')!=sha(root/'src/robot_navigation/partial_phase5b_diagnostics.py'):
            raise ValueError('Zero-learning diagnostic engine changed; preserve old output and use a fresh audit version')
        return result
    return historical_diagnosis(target,root)


def register(root=ROOT):
    c=settings(root); inputs=source_inputs(root); historical=historical_preservation(root)
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()
    if branch!=BRANCH: raise ValueError('Use the specified new Phase 5B feature branch')
    zero=diagnose(root)
    if zero['status']!='passed': raise ValueError('Zero-learning diagnosis blocked training')
    baseline=junit(root/'.runtime/phase5b-baseline-tests.xml')
    full=junit(root/'.runtime/phase5b-implementation-tests.xml')
    focused=require_gates(root)
    if read_json(root/'.runtime/phase5b-validated-inputs.json') != inputs:
        raise ValueError('Full regressions/source boundary are stale')
    from .partial_phase5b_report import preflight_report
    preflight_report(root)
    expanded=expanded_cases(); case_audit=audit_cases(expanded)
    folder=location(root); folder.mkdir(parents=True,exist_ok=False)
    snapshot=folder/'source-snapshot'
    for name in inputs:
        target=snapshot/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root.parent/name,target)
    shutil.copytree(diagnosis_path(root),folder/'zero-learning-diagnosis')
    shutil.copy2(root/'docs/partial_phase5b_diagnosis.md',folder/'preflight-diagnosis.md')
    for before,after in [('phase5b-baseline-tests.xml','baseline-regressions.xml'),
                         ('phase5b-implementation-tests.xml','pretraining-regressions.xml'),
                         ('partial-validation.log','partial-validation.log'),
                         ('phase5b-base-inventory.json','base-inventory.json')]:
        shutil.copy2(root/'.runtime'/before,folder/after)
    write_json(folder/'expanded-A-fixtures.json',{'cases':expanded,'audit':case_audit})
    write_json(folder/'pretraining-gates.json',{'baseline':baseline,'full':full,'focused':focused,
        'historical':historical,'zero_learning_status':zero['status'],'inputs':inputs})
    manifest={'study_id':STUDY,'registered_utc':datetime.now(timezone.utc).isoformat(),
        'settings':c,'inputs':inputs,'case_audit':case_audit,'training_cases':cases('A','train'),
        'original_cases':cases('A'),'expanded_cases':expanded,'feature_versions':ARMS,
        'run_order':[f'{a}/{arm}' for a in ALGORITHMS for arm in ARMS],
        'task_signature':signature(configuration(root)),
        'historical_protocol_sha256':sha(root/'artifacts/partial-dynamic-navigation-v1/feasibility-pilot-v1/phase5a_pilot_protocol.json'),
        'zero_learning_sha256':sha(folder/'zero-learning-diagnosis/zero-learning-audit.json'),
        'git':{'branch':branch,'base':BASE,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
               'status':subprocess.check_output(['git','status','--porcelain=v1','-uall'],cwd=root,text=True)},
        'runtime':{'python':platform.python_version(),'platform':platform.platform(),'cpu':platform.processor(),
            'logical_cpus':os.cpu_count(),'torch_threads':1,'interpreter':sys.executable,
            'working_directory':str(root),'PYTHONPATH':os.environ.get('PYTHONPATH'),
            'dependencies':{n:version(n) for n in ('numpy','torch','gymnasium','stable-baselines3','pytest')}},
        'commands':[f'{sys.executable} -m robot_navigation.partial_phase5b {v}' for v in ('diagnose','register','run','audit','report')],
        'split_scope':'Training pairs and fixed familiar-geometry development pairs only; no final/test reads',
        'sampling':'Original environment uniform episode sampling with independent reset seed 23021 per arm',
        'linear_epsilon':'Phase 5A horizon 10000, decay over 8000; epsilon at 6000 = .325',
        'dqn_epsilon':'Original SB3 public 2000 then 4000 learn calls with reset_num_timesteps=False; no schedule override',
        'timing_scope':'Primary trial includes controls, recording, training, checkpoint/evaluation; post-run audits/tests separate'}
    write_json(folder/'manifest.json',manifest)
    write_json(folder/'protocol-seal.json',{'sha256':sha(folder/'manifest.json')})
    write_json(folder/'status.json',{'status':'registered','training_actions':0})
    return {'status':'registered','manifest_sha256':sha(folder/'manifest.json')}


def checked_manifest(root=ROOT):
    folder=location(root); record=read_json(folder/'manifest.json')
    if sha(folder/'manifest.json')!=read_json(folder/'protocol-seal.json')['sha256']:
        raise ValueError('Corrupt registered manifest')
    if record['inputs']!=source_inputs(root): raise ValueError('Stale source/config/test/protocol hash')
    for name,h in record['inputs'].items():
        if sha(folder/'source-snapshot'/name)!=h: raise ValueError('Source snapshot changed')
    if digest(record['expanded_cases'])!=audit_cases(record['expanded_cases'])['expanded_hash']:
        raise ValueError('Corrupt expanded manifest')
    if read_json(folder/'expanded-A-fixtures.json')['cases']!=record['expanded_cases']:
        raise ValueError('Expanded fixture file changed')
    return record


class ArtifactQuota:
    def __init__(self,folder,ceiling):
        self.folder,self.ceiling,self.failed=Path(folder),ceiling,False
        self.synchronize()
    def synchronize(self):
        self.used=sum(p.stat().st_size for p in self.folder.rglob('*') if p.is_file())
    def check(self,extra=0):
        if self.used+extra+(0 if self.failed else 1048576)>self.ceiling:
            self.failed=True
            raise PilotLimit('Artifact ceiling: reserve buffered writes and failure metadata; preserve prefix, no retry')
        return self.used
    def open_binary(self,path,mode='xb'):
        return QuotaFile(Path(path).open(mode,buffering=0),self)
    def open_gzip(self,path):
        raw=self.open_binary(path)
        return ClosingTextGzip(raw)
    def write_bytes(self,path,payload):
        with self.open_binary(path) as sink: sink.write(payload)


class QuotaFile:
    def __init__(self,raw,quota): self.raw,self.quota=raw,quota
    def write(self,data):
        self.quota.check(len(data)); n=self.raw.write(data); self.quota.used+=n; return n
    def __getattr__(self,name): return getattr(self.raw,name)
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
    def close(self): self.raw.close()


class ClosingTextGzip(io.TextIOWrapper):
    def __init__(self,raw):
        self.raw_file=raw
        super().__init__(gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0),encoding='utf-8')
    def close(self):
        try: super().close()
        finally: self.raw_file.close()


class TrialBudget:
    def __init__(self,c,folder=None,clock=perf_counter):
        self.c,self.clock,self.started=c,clock,clock()
        self.training={f'{a}/{b}':0 for a in c['algorithms'] for b in c['arms']}
        self.evaluation_claims,self.starts=0,{}
        self.quota=ArtifactQuota(folder,c['artifact_byte_ceiling']) if folder else None
        self.file=io.TextIOWrapper(self.quota.open_binary(Path(folder)/'training-budget-ledger.jsonl'),
                                 encoding='utf-8',line_buffering=True) if folder else None
    def check(self,key=None):
        now=self.clock()
        if now-self.started>=self.c['suite_wall_seconds']: raise PilotLimit('Complete-trial wall ceiling')
        if key in self.starts and now-self.starts[key]>=self.c['per_arm_wall_seconds']:
            raise PilotLimit('Per-arm wall ceiling')
        if self.quota:
            if self.quota.failed: raise PilotLimit('Artifact budget already aborted; cleanup only')
            self.quota.check()
    def start(self,key):
        self.check()
        if key not in self.training or key in self.starts: raise PilotLimit('Unknown arm or free retry')
        self.starts[key]=self.clock()
    def reserve(self,kind,key=None):
        self.check(key)
        if self.quota: self.quota.check(extra=200000)  # reserve a complete action record before execution
        if kind=='training':
            if key not in self.starts: raise PilotLimit('Arm must be started once')
            if self.training[key]>=self.c['per_arm_actions'] or sum(self.training.values())>=self.c['total_training_actions']:
                raise PilotLimit('Training action ceiling')
            self.training[key]+=1; count=self.training[key]
        elif kind=='evaluation':
            if self.evaluation_claims>=self.c['evaluation_action_ceiling']: raise PilotLimit('Evaluation action ceiling')
            self.evaluation_claims+=1; count=self.evaluation_claims
        else: raise ValueError('Unknown reservation kind')
        if self.file:
            self.file.write(json.dumps({'kind':kind,'run_id':key,'count':count,'elapsed_seconds':self.clock()-self.started})+'\n')
    def summary(self):
        return {'training_reserved':self.training.copy(),'training_total':sum(self.training.values()),
                'evaluation_reserved':self.evaluation_claims,'elapsed_seconds':self.clock()-self.started}
    def close(self):
        if self.file: self.file.close()


class ArmBudget:
    def __init__(self,budget,algorithm,arm=None):
        self.budget,self.algorithm,self.key=budget,algorithm,f'{algorithm}/{arm}' if arm else None
    @property
    def training(self): return {self.algorithm:self.budget.training[self.key]}
    def training_action(self,algorithm):
        if algorithm!=self.algorithm: raise ValueError('Wrong family counter')
        self.budget.reserve('training',self.key)
    def evaluation_action(self,algorithm=None): self.budget.reserve('evaluation',self.key)


class QuotaStream(ActionStream):
    def __init__(self,path,quota): super().__init__(path); self.quota=quota; self.raw=None
    def begin(self):
        if self.file is not None: raise RuntimeError('Member already open')
        self.raw=self.quota.open_binary(self.path,'ab' if self.path.exists() else 'xb')
        self.file=io.TextIOWrapper(gzip.GzipFile(filename='',fileobj=self.raw,mode='wb',mtime=0),encoding='utf-8')
    def write(self,record):
        self.quota.check(extra=len(json.dumps(record,allow_nan=False).encode())+65536)
        super().write(record)
    def close(self):
        super().close()
        if self.raw: self.raw.close(); self.raw=None


def save_json(path,value,quota):
    quota.check(extra=len(json.dumps(value,allow_nan=False).encode())*2+65536)
    write_json(path,value)
    quota.synchronize()


def panel_summary(records,policy):
    result=evaluation_summary(records)
    result['median_steps']=float(np.median([len(r['transitions']) for r in records]))
    result['first_decisions']=[{'case_id':r['scenario']['id'],**decision_diagnostic(policy,r['initial_observation'])} for r in records]
    result['goal_directed_initial_count']=sum(r['first_step_goal_directed'] for r in result['first_decisions'])
    result['decision_distributions']=recorded_values([s for r in records for s in r['transitions']])
    if all(r['scenario']['id'].startswith('phase5b-open') for r in records):
        result['strata']={name:{str(value):evaluation_summary([r for r in records if strata(r['scenario'])[name]==value])
            for value in sorted({strata(r['scenario'])[name] for r in records},key=str)}
            for name in ('direction','distance','quadrant','boundary_band')}
    return result


def recorded_values(steps,training=False):
    margins=[]; entropy=[]; scores=[]
    for step in steps:
        decision=step.get('behavior_context',{}).get('decision') if training else step.get('decision')
        if not decision: continue
        values=np.asarray(decision.get('action_values',decision.get('action_probabilities')))
        scores.append(values)
        delta=np.asarray(step['observation'][2:4])-step['observation'][:2]
        good=[i for i,(dr,dc) in enumerate(MOVES[:4])
              if dr*delta[0]+dc*delta[1]>0]
        other=[i for i in range(5) if i not in good]
        if good: margins.append(float(values[good].max()-values[other].max()))
        if 'action_probabilities' in decision:
            p=np.maximum(values,1e-30); entropy.append(float(-np.sum(p*np.log(p))))
    matrix=np.asarray(scores)
    return {'goal_direction_margin':distribution(margins),'entropy':distribution(entropy),
            'per_action_score_or_probability':[distribution(matrix[:,i]) for i in range(5)] if scores else [],
            'scope':'Recorded pre-action values; training exploratory choices may differ from argmax'}


def assess(cases_to_run,policy,task,budget,path,quota):
    records=[]
    # Each complete gzip member is independently readable, including after an abort.
    with quota.open_binary(path) as sink:
        for case in cases_to_run:
            quota.check(extra=32*1024**2)  # preserve a whole 200-action assessment before starting it
            try: record=rollout(case,policy,task,budget)
            except EpisodeAbort as exc:
                payload=gzip.compress((json.dumps(exc.record,allow_nan=False)+'\n').encode(),mtime=0)
                quota.check(len(payload)); sink.write(payload); sink.flush(); raise
            record['feature_signature']=getattr(policy,'signature',None)
            payload=gzip.compress((json.dumps(record,allow_nan=False)+'\n').encode(),mtime=0)
            quota.check(len(payload)); sink.write(payload); sink.flush()
            validate_replay(record); records.append(record)
    return records


def checkpoint(folder,policy,trainer,env,stream,manifest,budget,block_seconds):
    algorithm,arm=folder.parts[-2:]; key=f'{algorithm}/{arm}'; count=budget.training[key]
    budget.check(key); quota=budget.quota
    target=folder/f'{count:05}'; target.mkdir()
    path=target/('policy.npz' if trainer else 'policy.zip')
    quota.check(2*1024**2)
    policy.save(path,{'study_id':STUDY,'arm':arm,'training_interactions':count,
        'manifest_sha256':sha(folder.parents[2]/'manifest.json'),'checkpoint_role':'Fixed boundary, not best selection'})
    quota.synchronize()
    state=continuity_state(policy,trainer,env); before=continuity_hash(state)
    # Full trusted continuation/buffer state is compressed, separate from inference inputs.
    import pickle
    payload=gzip.compress(pickle.dumps(state,protocol=pickle.HIGHEST_PROTOCOL),compresslevel=3,mtime=0)
    quota.check(len(payload)); quota.write_bytes(target/'continuation-state.pkl.gz',payload)
    with preserve_random_state():
        loaded=load_scaled_policy(path,env.unwrapped.config,arm)
        if loaded.state_fingerprint()!=policy.state_fingerprint(): raise ValueError('Checkpoint reload mismatch')
        records=assess(manifest['original_cases']+manifest['expanded_cases'],loaded,env.unwrapped.config,
            ArmBudget(budget,algorithm,arm),target/'evaluation.jsonl.gz',quota)
        panels={name:panel_summary(part,loaded) for name,part in [('original',records[:4]),('expanded',records[4:])]}
    after=continuity_hash(continuity_state(policy,trainer,env))
    if before!=after: raise ValueError('Assessment contaminated trainer/RNG/optimizer/buffer state')
    trace=read_records(stream.path) if stream.path.exists() else []
    if trace and validate_training_trace(trace,env.unwrapped.config)!=count: raise ValueError('Wrong actual training count')
    summary={'algorithm':algorithm,'arm':arm,'actions':count,'panels':panels,
        'parameter_fingerprint':policy.state_fingerprint(),'parameter_norm':parameter_norm(policy),
        'checkpoint':path.name,'checkpoint_sha256':sha(path),
        'continuity_before':before,'continuity_after':after,'continuation_sha256':sha(target/'continuation-state.pkl.gz'),
        'training_prefix_bytes':stream.path.stat().st_size if stream.path.exists() else 0,
        'training_prefix_sha256':sha(stream.path) if stream.path.exists() else None,
        'training':training_summary(trace) if trace else None,'block_training_seconds':block_seconds,
        'numerical':optimizer_state(policy),'linear_updates':trainer.updates if trainer else None,
        'model':model_diagnostics(policy) if count else {'parameters_finite':True},'resources':_resource_usage()}
    if not summary['numerical']['parameters_finite']: raise FloatingPointError('Nonfinite checkpoint')
    save_json(target/'summary.json',summary,quota)
    print(f'{key} {count}: original={panels["original"]["counts"]}, expanded={panels["expanded"]["counts"]}',flush=True)
    budget.check(key)
    return summary


def train_arm(algorithm,arm,folder,manifest,budget,root=ROOT):
    folder.mkdir(parents=True); key=f'{algorithm}/{arm}'; c=manifest['settings']; quota=budget.quota
    budget.start(key); adapter=ArmBudget(budget,algorithm,arm)
    import random
    import torch
    random.seed(c['primary_seed']); np.random.seed(c['primary_seed']); torch.manual_seed(c['primary_seed'])
    stream=QuotaStream(folder/'training-actions.jsonl.gz',quota)
    env=PilotRecorder(manifest['training_cases'],configuration(root),adapter,algorithm,stream,'A',c['primary_seed'])
    policy=build_policy(env,algorithm,c['model'],c['primary_seed'],arm)
    env.unwrapped.np_random=np.random.default_rng(c['primary_seed'])
    env.action_space.seed(c['primary_seed'])
    policy.policy_id=f'{algorithm}/{STUDY}/{arm}/seed{c["primary_seed"]}'
    trainer=None
    if algorithm in ('q-learning','sarsa'):
        # Keep Phase 5A's 10k exploration horizon even though this arm stops at 6k.
        linear_settings={**read_baseline_settings(root),'primary_seed':c['primary_seed'],
                         'per_algorithm_actions':c['linear_epsilon_horizon']}
        trainer=ContinuousLinear(policy,adapter,linear_settings); trainer.set_env(env)
        observer=LinearUpdateRecorder(policy,folder/'linear-updates.jsonl.gz',ARMS[arm],quota.open_gzip)
        context=env.context
        env.context=lambda:{**context(),'decision':decision_values(policy,env.previous.copy()),
                            'feature_norm':float(np.linalg.norm(encode_scaled(env.previous,ARMS[arm])))}
    else:
        observer=UpdateRecorder(policy,folder/'optimizer-updates.jsonl.gz',lambda:budget.training[key],quota.open_gzip)
        env.context=lambda:{'decision':decision_values(policy,env.previous.copy()),
            'epsilon_used':float(policy.model.exploration_rate) if algorithm=='dqn' else None,
            'training_selection':'DQN epsilon/warmup' if algorithm=='dqn' else 'PPO stochastic categorical',
            'warmup_uniform':algorithm=='dqn' and policy.model.num_timesteps<policy.model.learning_starts,
            'entropy_coefficient':float(policy.model.ent_coef) if algorithm=='ppo' else None,
            'updates_before_action':int(policy.model._n_updates),
            'feature_norm':float(np.linalg.norm(encode_scaled(env.previous,ARMS[arm])))}
    summaries=[]; training_seconds=0.; result={}
    try:
        summaries.append(checkpoint(folder,policy,trainer,env,stream,manifest,budget,0.))
        for boundary in c['checkpoints'][1:]:
            actions=boundary-budget.training[key]; budget.check(key)
            stream.begin(); started=perf_counter()
            try:
                if trainer: trainer.train(actions)
                else:
                    policy.model.learn(total_timesteps=actions,reset_num_timesteps=False)
                    if policy.model.num_timesteps!=boundary: raise ValueError('Wrong SB3 interaction count')
            finally:
                seconds=perf_counter()-started; training_seconds+=seconds; stream.close()
            summaries.append(checkpoint(folder,policy,trainer,env,stream,manifest,budget,seconds))
        records=read_records(stream.path)
        stats=feature_statistics([r['observation'] for r in records if r['kind']=='step'],ARMS[arm])
        save_json(folder/'training-feature-statistics.json',stats,quota)
        resets=[{'episode':r['episode'],'case_id':r['scenario']['id'],'seed':r['seed']} for r in records if r['kind']=='reset']
        save_json(folder/'training-case-stream.json',resets,quota)
        result={'algorithm':algorithm,'arm':arm,'status':'completed','actual_training_actions':budget.training[key],
            'checkpoints':summaries,'training_seconds_including_recording':training_seconds,
            'arm_wall_seconds':budget.clock()-budget.starts[key],
            'observed_updates':observer.steps,'resources':_resource_usage()}
        steps=[r for r in records if r['kind']=='step']
        result['training_reward_distribution']=dict(Counter(str(s['reward']) for s in steps))
        result['training_value_distributions']=recorded_values(steps,training=True)
        budget.check(key)
    except BaseException as exc:
        stream.close()
        # Never relabel a prefix or hide a consumed reservation; no retry is exposed.
        if isinstance(exc,EpisodeAbort):
            save_json(folder/'interrupted-evaluation.json',exc.record,quota)
        result={'algorithm':algorithm,'arm':arm,'status':'interrupted','actual_training_actions':budget.training[key],
            'checkpoints':summaries,'training_seconds_including_recording':training_seconds,
            'exception':f'{type(exc).__name__}: {exc}',
            'limit_abort':isinstance(exc,PilotLimit) or (isinstance(exc,EpisodeAbort) and isinstance(exc.cause,PilotLimit)),
            'traceback':traceback.format_exc()}
    finally:
        stream.close(); observer.close()
    save_json(folder/'run-summary.json',result,quota)
    return result


def screening(systems,c):
    result={}
    for algorithm in ALGORITHMS:
        if algorithm not in systems or any(arm not in systems[algorithm] or systems[algorithm][arm]['status']!='completed' for arm in ARMS):
            result[algorithm]={'passed':False,'assessed':False}; continue
        b0=systems[algorithm]['B0-original']['checkpoints'][-1]['panels']
        b1runs=systems[algorithm]['B1-relative-x4']['checkpoints']
        initial=b1runs[0]['panels']; b1=b1runs[-1]['panels']
        n=lambda panels,panel:panels[panel]['counts']['success']
        checks={'original_minimum':n(b1,'original')>=c['screen']['original_successes'],
            'gain_over_b1_initial':n(b1,'original')-n(initial,'original')>=c['screen']['gain_over_b1_initial'],
            'original_gain_over_b0':n(b1,'original')-n(b0,'original')>=c['screen']['gain_over_b0_final'],
            'expanded_minimum':n(b1,'expanded')>=c['screen']['expanded_successes'],
            'expanded_gain_over_b0':n(b1,'expanded')-n(b0,'expanded')>=c['screen']['expanded_gain_over_b0']}
        result[algorithm]={'assessed':True,'passed':all(checks.values()),'checks':checks,
            'original_difference':n(b1,'original')-n(b0,'original'),
            'expanded_difference':n(b1,'expanded')-n(b0,'expanded')}
    return result


def run(root=ROOT):
    manifest=checked_manifest(root); folder=location(root); c=manifest['settings']
    if read_json(folder/'status.json')['status']!='registered': raise ValueError('Attempt already spent; no free retry')
    with (folder/'execution.lock').open('x') as lock: lock.write(datetime.now(timezone.utc).isoformat())
    budget=TrialBudget(c,folder); quota=budget.quota
    save_json(folder/'status.json',{'status':'running','started_utc':datetime.now(timezone.utc).isoformat()},quota)
    result={'systems':{},'controls':{}}
    try:
        controls=folder/'controls'; controls.mkdir()
        for name,cls in [('online-planner',OnlinePlanner),('goal-direction-script',GoalDirectionScript)]:
            records=assess(manifest['original_cases']+manifest['expanded_cases'],cls(),configuration(root),
                ArmBudget(budget,name),controls/f'{name}.jsonl.gz',quota)
            result['controls'][name]={'original':evaluation_summary(records[:4]),'expanded':evaluation_summary(records[4:])}
            if any(r['metrics']['outcome']!='success' for r in records): raise ValueError('A mechanics reference failed')
        for algorithm in ALGORITHMS:
            result['systems'][algorithm]={}
            for arm in ARMS:
                value=train_arm(algorithm,arm,folder/'training-and-evaluation-evidence'/algorithm/arm,manifest,budget,root)
                result['systems'][algorithm][arm]=value
                if value['status']!='completed': raise RuntimeError(f'{algorithm}/{arm} interrupted; no remaining arm training')
        checked_manifest(root); historical_preservation(root); budget.check()
        result.update(status='completed',screen=screening(result['systems'],c),
                      verdict='PENDING_G4_AUDIT')
    except BaseException as exc:
        if isinstance(exc,EpisodeAbort): save_json(folder/'interrupted-control.json',exc.record,quota)
        result.update(status='interrupted',exception=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),
                      verdict='INCOMPLETE_BUDGET_OR_EXECUTION')
    finally:
        result.update(budget=budget.summary(),ended_utc=datetime.now(timezone.utc).isoformat(),resources=_resource_usage())
        budget.close()
    save_json(folder/'summary.json',result,quota)
    save_json(folder/'checkpoint-hashes.json',{p.relative_to(folder).as_posix():sha(p)
        for p in folder.rglob('policy.*') if p.suffix in ('.zip','.npz')},quota)
    save_json(folder/'status.json',{'status':result['status'],'verdict':result['verdict']},quota)
    write_json(folder/'outputs.json',{p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*')
        if p.is_file() and p.name not in ('status.json','outputs.json')})
    quota.synchronize(); quota.check()
    return {'status':result['status'],'budget':result['budget'],'screen':result.get('screen'),
            'verdict':result['verdict'],'exception':result.get('exception')}


def reconcile_ledger(rows,training,evaluations,completed):
    actual=dict.fromkeys(training,0); evaluated=0
    for row in rows:
        if row['kind']=='training':
            key=row['run_id']
            if key not in actual: raise ValueError('Unknown ledger arm')
            actual[key]+=1
            if row['count']!=actual[key] or actual[key]>6000: raise ValueError('Training count/ceiling mismatch')
        elif row['kind']=='evaluation':
            evaluated+=1
            if row['count']!=evaluated or evaluated>100000: raise ValueError('Evaluation count/ceiling mismatch')
        else: raise ValueError('Unknown ledger entry')
    if sum(actual.values())>48000: raise ValueError('Global training ceiling')
    if any(training[k]>actual[k] for k in actual) or evaluations>evaluated:
        raise ValueError('Executed actions exceed reservations')
    if completed and (training!=actual or evaluations!=evaluated): raise ValueError('Wrong actual interaction counts')
    return {'training_actions':training,'training_total':sum(training.values()),'evaluation_actions':evaluations,
            'reserved_training':actual,'reserved_evaluation':evaluated}


def audit(root=ROOT):
    manifest=checked_manifest(root); folder=location(root); summary=read_json(folder/'summary.json')
    for name,h in read_json(folder/'outputs.json').items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path)!=h: raise ValueError('Output checksum mismatch: '+name)
    actual=dict.fromkeys(manifest['run_order'],0); evaluated=0; episodes=0; checks=0
    expected_ids=[c['id'] for c in manifest['original_cases']+manifest['expanded_cases']]
    for path in (folder/'controls').glob('*.jsonl.gz'):
        records=read_records(path)
        if [r['scenario']['id'] for r in records]!=expected_ids: raise ValueError('Control case matrix mismatch')
        for record in records: evaluated+=audit_episode(record); episodes+=1
    sampled={}
    for algorithm,arms in summary['systems'].items():
        for arm,run_result in arms.items():
            arm_folder=folder/'training-and-evaluation-evidence'/algorithm/arm; key=f'{algorithm}/{arm}'
            trace=read_records(arm_folder/'training-actions.jsonl.gz') if (arm_folder/'training-actions.jsonl.gz').exists() else []
            actual[key]=validate_training_trace(trace,configuration(root)) if trace else 0
            if actual[key]!=run_result['actual_training_actions']: raise ValueError('Actual training count mismatch')
            steps=[r for r in trace if r['kind']=='step']
            if [s['global_training_interaction'] for s in steps]!=list(range(1,actual[key]+1)):
                raise ValueError('Non-continuous training action counter')
            for reset in (r for r in trace if r['kind']=='reset'):
                if reset['scenario'] not in manifest['training_cases']: raise ValueError('Training case leakage')
            sampled[key]=[r['scenario']['id'] for r in trace if r['kind']=='reset']
            if run_result['status']=='completed' and [x['actions'] for x in run_result['checkpoints']]!=[0,2000,6000]:
                raise ValueError('Missing fixed checkpoint')
            for checkpoint_result in run_result['checkpoints']:
                target=arm_folder/f'{checkpoint_result["actions"]:05}'
                prefix=(arm_folder/'training-actions.jsonl.gz').read_bytes()[:checkpoint_result['training_prefix_bytes']] if trace else b''
                if prefix and hashlib.sha256(prefix).hexdigest()!=checkpoint_result['training_prefix_sha256']:
                    raise ValueError('Earlier training prefix rewritten')
                with preserve_random_state():
                    policy=load_scaled_policy(target/checkpoint_result['checkpoint'],configuration(root),arm)
                    records=read_records(target/'evaluation.jsonl.gz')
                    if [r['scenario']['id'] for r in records]!=expected_ids: raise ValueError('Unpaired assessment cases')
                    before=policy.state_fingerprint()
                    for record in records:
                        if record['feature_signature']!=policy.signature: raise ValueError('Evaluation feature signature mismatch')
                        evaluated+=audit_episode(record); episodes+=1
                        for step in record['transitions']:
                            obs=np.asarray(step['observation'],dtype=np.float32)
                            if policy.act(obs.copy())!=step['action']: raise ValueError('Checkpoint action mismatch')
                            decision=decision_values(policy,obs)
                            if any(not np.allclose(v,step['decision'][k],atol=1e-7,rtol=1e-6) for k,v in decision.items()):
                                raise ValueError('Checkpoint value/probability mismatch')
                    if before!=policy.state_fingerprint() or before!=checkpoint_result['parameter_fingerprint']:
                        raise ValueError('Inference changed model state')
                    panels={name:panel_summary(part,policy) for name,part in [('original',records[:4]),('expanded',records[4:])]}
                    if digest(panels)!=digest(checkpoint_result['panels']): raise ValueError('Panel summary mismatch')
                if checkpoint_result['continuity_before']!=checkpoint_result['continuity_after']:
                    raise ValueError('Checkpoint contaminated trainer')
                if prefix:
                    with gzip.open(io.BytesIO(prefix),'rt',encoding='utf-8') as f: prefix_records=[json.loads(line) for line in f]
                    if digest(training_summary(prefix_records))!=digest(checkpoint_result['training']):
                        raise ValueError('Training prefix summary mismatch')
                checks+=1
    stream_checks={}
    for algorithm in ALGORITHMS:
        a=sampled.get(f'{algorithm}/B0-original',[]); b=sampled.get(f'{algorithm}/B1-relative-x4',[])
        shared=min(len(a),len(b)); match=a[:shared]==b[:shared]
        if not match: raise ValueError('Arm case-sampling streams differ by episode index')
        stream_checks[algorithm]={'shared_episode_prefix':shared,'identical':match,'B0_episodes':len(a),'B1_episodes':len(b)}
    rows=[json.loads(line) for line in (folder/'training-budget-ledger.jsonl').read_text().splitlines()]
    counts=reconcile_ledger(rows,actual,evaluated,summary['status']=='completed')
    if counts['reserved_training']!=summary['budget']['training_reserved'] or counts['reserved_evaluation']!=summary['budget']['evaluation_reserved']:
        raise ValueError('Summary and durable budget journal differ')
    historical=historical_preservation(root)
    final_tests=junit(root/'.runtime/phase5b-postrun-tests.xml')
    if source_inputs(root)!=read_json(root/'.runtime/phase5b-postrun-validated-inputs.json'):
        raise ValueError('Post-run regressions/source are stale')
    screens=screening(summary['systems'],manifest['settings'])
    if summary['status']=='completed' and (counts['training_total']!=48000 or episodes!=520):
        raise ValueError('Completed study missing registered actions/assessment episodes')
    passed=sum(v['passed'] for v in screens.values())
    verdict=('INCOMPLETE_BUDGET_OR_EXECUTION' if summary['status']!='completed' else
             'SCALING_HYPOTHESIS_SUPPORTED_IN_ONE_SEED' if passed else 'SCALING_HYPOTHESIS_NOT_SUPPORTED')
    result={'status':'passed' if summary['status']=='completed' else 'partial_prefixes_audited',
        **counts,'evaluation_episodes':episodes,'checkpoint_decision_audits':checks,
        'case_stream_prefixes':stream_checks,'historical_preservation':historical,
        'final_regressions':final_tests,'manifest_sha256':sha(folder/'manifest.json'),
        'source_hashes_checked':len(manifest['inputs']),
        'replay_scope':'Exact environment/action/observation/reward/snapshot/flag/value/metric/prefix verification; no new learning'}
    write_json(folder/'replay-audit.json',result)
    write_json(folder/'final-verdict.json',{'verdict':verdict,'families_passed':passed,'families_assessed':4,
        'screens':screens,'scope':'Single seed, familiar geometry development only; no further training permission'})
    write_json(folder/'gate-results.json',{'G0_SOURCE_AND_EVIDENCE':'passed','G1_ZERO_LEARNING_DIAGNOSIS':'passed',
        'G2_FAIR_REGISTERED_COMPARISON':'passed','G3_MEASURED_LEARNING':screens,
        'G4_AUDIT_AND_REPRODUCIBILITY':result['status']})
    shutil.copy2(root/'.runtime/phase5b-postrun-tests.xml',folder/'postrun-regressions.xml')
    return {'audit':result,'verdict':verdict,'families_passed':passed}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('diagnose','validate','validate-postrun','register','run','audit','report'))
    action=parser.parse_args().command
    if action=='report':
        from .partial_phase5b_report import report
        result=report()
    elif action=='diagnose':
        r=diagnose(); result={'status':r['status'],'new_training_actions':0,'new_policy_environment_actions':0}
    elif action in ('validate','validate-postrun'):
        result=validate(postrun=action=='validate-postrun')
    else: result={'register':register,'run':run,'audit':audit}[action]()
    print(json.dumps(result,indent=2,allow_nan=False))


def validate(root=ROOT,postrun=False):
    before=source_inputs(root)
    prefix='phase5b-postrun' if postrun else 'phase5b-implementation'
    path=root/'.runtime'/f'{prefix}-tests.xml'; path.parent.mkdir(exist_ok=True)
    result=subprocess.run([sys.executable,'-m','pytest','-q',f'--junitxml={path}'],cwd=root,
                          capture_output=True,text=True)
    (root/'.runtime'/f'{prefix}-tests.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    print(result.stdout+result.stderr,end='')
    if result.returncode or source_inputs(root)!=before:
        raise ValueError('Full regression/source-boundary gate failed')
    counts=junit(path)
    write_json(root/'.runtime'/('phase5b-postrun-validated-inputs.json' if postrun else 'phase5b-validated-inputs.json'),before)
    return {'status':'passed',**counts,'source_boundary':'unchanged throughout full regressions'}


if __name__=='__main__': main()
