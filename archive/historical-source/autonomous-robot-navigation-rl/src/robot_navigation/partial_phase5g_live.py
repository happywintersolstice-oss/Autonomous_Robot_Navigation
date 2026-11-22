"""Exactly two fresh 2000-action runs; checkpoints after actual train returns."""
import gc,random,sys
from time import perf_counter
from unittest.mock import patch
import numpy as np
import torch

from .core import write_json
from .partial_contract import configuration
from .partial_dynamic_env import PartialDynamicEnv
from .partial_features_scaled import build_policy,SCALED
from .partial_agents import metadata
from .partial_training_instrumentation import observe,network_versions
from .partial_instrumentation_trace import TraceWriter,checksum,quota_bytes
from .partial_phase5b_diagnostics import continuity_hash
from .partial_phase5f_live import Calls,ActionRecorder,Sink,rng_signature
from .partial_phase5g_io import STUDY,ALGORITHMS,CHECKPOINTS,UPDATES,Journal,Budget


def snapshot(policy,base,target,budget,journal,manifest):
    m=policy.model;n=int(m.num_timesteps)
    if n not in CHECKPOINTS or n!=budget.counts[policy.algorithm]:raise ValueError('Not an exact checkpoint')
    if quota_bytes([target.parents[2]])+8*1024**2>=536870912:raise RuntimeError('Quota before checkpoint reservation')
    target.mkdir(exist_ok=False);buffer=m.replay_buffer if policy.algorithm=='dqn' else m.rollout_buffer
    before={'network':network_versions(m),'optimizer':continuity_hash(m.policy.optimizer.state_dict()),
        'buffer':continuity_hash(buffer),'rng':rng_signature(base)}
    # Only the instance orchestration wrapper is excluded, never learner state.
    # Otherwise SB3 would pickle its closure (budget/environment/file handles).
    m.save(target/'policy.zip',exclude=['train'])
    record=metadata(policy,target/'policy.zip',{'study_id':STUDY,'classification':'live_actual','training_actions':n,
        'manifest_sha256':manifest['manifest_sha256'],'checkpoint_role':'Fixed development measurement; no winner selection'})
    record['feature_version']=SCALED;write_json(target/'policy.json',record)
    arrays={k:v[:n].copy() for k,v in vars(buffer).items() if isinstance(v,np.ndarray)} if policy.algorithm=='dqn' else {
        k:v.copy() for k,v in vars(buffer).items() if isinstance(v,np.ndarray)}
    np.savez_compressed(target/'buffer.npz',**arrays)
    after={'network':network_versions(m),'optimizer':continuity_hash(m.policy.optimizer.state_dict()),
        'buffer':continuity_hash(buffer),'rng':rng_signature(base)}
    if before!=after:raise RuntimeError('Checkpoint save mutated training state/RNG')
    receipt={'training_actions':n,'optimizer_steps':budget.updates[policy.algorithm],'library_updates':int(m._n_updates),
        'model_num_timesteps':int(m.num_timesteps),'total_timesteps':int(m._total_timesteps),
        'network':before['network'],'optimizer_sha256':before['optimizer'],'buffer_sha256':before['buffer'],
        'rng_before':before['rng'],'rng_after':after['rng'],'state_unchanged':True,'buffer_pos':int(buffer.pos),'buffer_full':bool(buffer.full),
        'checkpoint_sha256':record['checkpoint_sha256'],'buffer_file_sha256':__import__('hashlib').sha256((target/'buffer.npz').read_bytes()).hexdigest(),
        'capture_boundary':'initialized, before first reset' if n==0 else 'after original model.train returned, with all boundary optimizer steps complete',
        'excluded_save_parameters':['train (orchestration callable only)'],'feature_version':SCALED}
    write_json(target/'receipt.json',receipt)
    journal.append('checkpoint_saved',algorithm=policy.algorithm,training_actions=n,optimizer_steps=budget.updates[policy.algorithm],
        checkpoint_sha256=record['checkpoint_sha256'],network=before['network'],optimizer_sha256=before['optimizer'],
        rng_sha256=checksum(before['rng']))


def run_models(folder,manifest,training_cases):
    journal=Journal(folder/'training-ledger.jsonl',quota_root=folder);budget=Budget(journal);results=[]
    try:
        for algorithm in ALGORITHMS:
            budget.start(algorithm);start=perf_counter();arm=folder/algorithm;arm.mkdir(exist_ok=False);(arm/'checkpoints').mkdir()
            random.seed(23121);np.random.seed(23121);torch.manual_seed(23121)
            base=PartialDynamicEnv(training_cases,configuration())
            writer=TraceWriter(arm/'trace',{'study_id':STUDY,'run_id':algorithm+'-B1-seed23121-horizon2000','algorithm':algorithm,
                'arm':'B1-relative-x4','seed':23121,'source_kind':'live_actual','source_hash':manifest['source_digest'],
                'config_hash':checksum(manifest['model_config']),'dependency_hash':checksum(manifest['library_contract']),'split':'train'},
                byte_cap=536870912,quota_roots=[folder])
            try:
                sink=Sink(writer,budget,algorithm,[c['id'] for c in training_cases]);env=ActionRecorder(base,budget,algorithm,journal,sink)
                policy=build_policy(env,algorithm,manifest['model_config'],23121,'B1-relative-x4');env.model=policy.model
                base.np_random=np.random.default_rng(23121);base.action_space.seed(23121)
                policy.policy_id=f'{algorithm}/{STUDY}/B1-relative-x4/seed23121';m=policy.model
                initial_rng=rng_signature(base);initial_network=network_versions(m)
                snapshot(policy,base,arm/'checkpoints/0000',budget,journal,manifest)
                old_train=m.train;old_optimizer=m.policy.optimizer.step;old_mean=torch.Tensor.mean;old_std=torch.Tensor.std
                def train(*args,**kwargs):
                    result=old_train(*args,**kwargs)  # One unchanged original training operation.
                    if m.num_timesteps in CHECKPOINTS:snapshot(policy,base,arm/'checkpoints'/f'{m.num_timesteps:04d}',budget,journal,manifest)
                    return result
                with patch.object(m,'train',train):
                    with Calls(policy,env,budget,journal,sink) as calls:
                        with observe(policy,sink=sink,environment=base):m.learn(total_timesteps=2000,reset_num_timesteps=False)
                budget.check()
                if budget.counts[algorithm]!=2000 or budget.updates[algorithm]!=UPDATES[algorithm] or m.num_timesteps!=2000:raise ValueError('Incomplete exact arm')
                restored=(sys.gettrace() is None and sys.getprofile() is None and torch.Tensor.mean is old_mean and torch.Tensor.std is old_std
                    and m.policy.optimizer.step==old_optimizer and m.train==old_train)
                if not restored:raise ValueError('Hooks not restored')
                result={'algorithm':algorithm,'training_actions':2000,'optimizer_steps':budget.updates[algorithm],
                    'library_update_counter':int(m._n_updates),'profile_counts':calls.counts,'network_before':initial_network,'network_after':network_versions(m),
                    'optimizer_final_sha256':continuity_hash(m.policy.optimizer.state_dict()),'rng_before':initial_rng,'rng_after':rng_signature(base),
                    'observer_hooks_restored':restored,'wall_seconds':perf_counter()-start,'trace_uncompressed_bytes':writer.reserved_bytes,
                    'research_assessment_actions_during_training':0,'final_unseen_evaluations':0}
                write_json(arm/'run-receipt.json',result);results.append(result)
                journal.append('arm_finished',algorithm=algorithm,training_actions=2000,optimizer_steps=budget.updates[algorithm])
            finally:writer.close()
            del m,policy,base,env;gc.collect()
        budget.check();journal.append('training_finished',training_actions=4000,optimizer_steps=sum(budget.updates.values()))
        return {'status':'training_completed','arms':results,'scientific_training_actions':4000,'optimizer_steps':sum(budget.updates.values()),
            'wall_seconds':perf_counter()-budget.started,'assessments_during_training':0,'evaluation_isolation':'Training models destroyed before child assessments; OS process memory/RNG isolated'}
    except BaseException as exc:
        try:journal.append('training_failed',error=repr(exc),counts=budget.counts,updates=budget.updates)
        except BaseException:pass
        raise
    finally:journal.close()
