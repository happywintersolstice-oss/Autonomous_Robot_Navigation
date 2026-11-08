"""Vanilla PPO baseline, probability diagnostics and exact rollout budgets."""
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .core import NavigationEnv,read_json,write_json
from .dqn import CoordinateObservation,ENCODING,normalize,_hash_value


from .ppo_config import validate_config

def build_model(env,config):
    torch.set_num_threads(config['torch_threads'])
    return PPO('MlpPolicy',CoordinateObservation(env),seed=config['seed'],device='cpu',
        learning_rate=config['learning_rate'],gamma=config['discount'],n_steps=config['n_steps'],
        batch_size=config['batch_size'],n_epochs=config['n_epochs'],gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],ent_coef=config['ent_coef'],vf_coef=config['vf_coef'],
        max_grad_norm=config['max_grad_norm'],normalize_advantage=True,clip_range_vf=None,target_kl=None,
        policy_kwargs={'net_arch':dict(pi=config['net_arch'],vf=config['net_arch'])},verbose=0)


class PPOPolicy:
    algorithm='ppo'

    def __init__(self,task,model,policy_id):
        self.task,self.model,self.policy_id=task,model,policy_id

    def decision(self,observation):
        tensor=torch.as_tensor(normalize(observation,self.task.grid.shape),device=self.model.device).unsqueeze(0)
        with torch.inference_mode():
            probabilities=self.model.policy.get_distribution(tensor).distribution.probs[0].cpu().numpy().copy()
            value=float(self.model.policy.predict_values(tensor).item())
        return {'action':int(np.argmax(probabilities)),'action_probabilities':probabilities.tolist(),
                'state_value':value}

    def act(self,observation):
        return self.decision(observation)['action']

    def state_fingerprint(self):
        hasher=hashlib.sha256()
        _hash_value(hasher,self.model.get_parameters())
        _hash_value(hasher,(self.model.num_timesteps,self.model._n_updates))
        buffer=self.model.rollout_buffer
        if buffer is not None:
            _hash_value(hasher,(buffer.pos,buffer.full))
            for name in ('observations','actions','rewards','returns','episode_starts','values','log_probs','advantages'):
                _hash_value(hasher,getattr(buffer,name))
        return hasher.hexdigest()

    def save(self,path,metadata):
        path=Path(path)
        if path.suffix!='.zip':
            raise ValueError('PPO checkpoints must use .zip')
        path.parent.mkdir(parents=True,exist_ok=True)
        self.model.save(path)
        write_json(path.with_suffix('.json'),{**metadata,'signature':self.task.signature,
            'algorithm':self.algorithm,'policy_id':self.policy_id,'encoding':ENCODING,
            'kind':'inference-only; rollout/RNG/environment state not saved; exact resume unsupported',
            'diagnostic_type':'categorical action probabilities and state value; not Q-values',
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})

    @classmethod
    def load(cls,path,task):
        path=Path(path);metadata=read_json(path.with_suffix('.json'))
        task.check(metadata['signature'])
        if metadata.get('algorithm')!='ppo' or metadata.get('encoding')!=ENCODING:
            raise ValueError('PPO algorithm/encoding mismatch')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['sha256']:
            raise ValueError('Checkpoint checksum mismatch')
        torch.set_num_threads(1)
        model=PPO.load(path,device='cpu')
        if model.observation_space.shape!=(4,) or model.action_space.n!=4:
            raise ValueError('PPO observation/action shape mismatch')
        if any(not torch.isfinite(v).all() for v in model.policy.state_dict().values()):
            raise ValueError('Nonfinite PPO parameters')
        return cls(task,model,metadata['policy_id'])


def train(task,scenarios,config,record,file,writer):
    from .part_a import TrainingLedger
    ledger=TrainingLedger(NavigationEnv(task,scenarios),writer.writerow,record.trace)
    model=build_model(ledger,config)
    policy=PPOPolicy(task,model,f"ppo/{config['run_id']}")
    started=perf_counter()

    class Monitor(BaseCallback):
        def _on_step(self):
            if perf_counter()-started>config.get('max_wall_seconds',1200):
                raise TimeoutError('PPO run exceeded its declared wall-time limit')
            return True

        def _on_rollout_start(self):
            if self.num_timesteps and self.num_timesteps%config['monitor_every']==0:
                record.monitor(policy,self.num_timesteps,ledger.visitation,
                               'after preceding PPO gradient update; before next rollout')
                file.flush()
                print(f"{config['run_id']}: {self.num_timesteps:,} training interactions",flush=True)

        def _on_training_end(self):
            record.monitor(policy,self.num_timesteps,ledger.visitation,'after final PPO gradient update')

    model.learn(total_timesteps=config['interactions'],callback=Monitor())
    seconds=perf_counter()-started-record.monitor_seconds
    if model.num_timesteps!=config['interactions'] or ledger.interactions!=config['interactions']:
        raise AssertionError('PPO training budget mismatch')
    pilot=ledger.report(seconds)
    steps=[int(v['step'].item()) for v in model.policy.optimizer.state.values() if 'step' in v]
    pilot.update(ppo_rollout_batches=model.num_timesteps//config['n_steps'],
                 ppo_epoch_passes=model._n_updates,optimizer_steps=max(steps,default=0),
                 observation_encoding=ENCODING)
    return ledger.visitation,pilot
