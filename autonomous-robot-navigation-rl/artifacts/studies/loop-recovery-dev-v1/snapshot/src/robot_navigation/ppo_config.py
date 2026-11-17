"""PPO configuration checks without importing its optional neural runtime."""
from pathlib import Path
import numpy as np

def validate_config(config):
    integer_fields=('interactions','monitor_every','n_steps','batch_size','n_epochs','torch_threads')
    if any(type(config[k]) is not int or config[k]<=0 for k in integer_fields):
        raise ValueError('PPO rollout, budget, batch and thread settings must be positive integers')
    if (config['n_steps']<2 or config['batch_size']<2 or
            config['interactions']%config['n_steps'] or config['monitor_every']%config['n_steps'] or
            config['interactions']%config['monitor_every'] or config['n_steps']%config['batch_size']):
        raise ValueError('PPO budget/monitor must divide into complete rollouts and minibatches')
    if Path(config['run_id']).name!=config['run_id'] or config['run_id'] in ('.','..'):
        raise ValueError('run_id must be one directory name')
    for key in ('learning_rate','clip_range','max_grad_norm'):
        if not np.isfinite(config[key]) or config[key]<=0:
            raise ValueError(f'Invalid PPO {key}')
    if not 0<=config['discount']<=1 or not 0<=config['gae_lambda']<=1:
        raise ValueError('Invalid PPO discount or GAE lambda')
    if any(not np.isfinite(config[k]) or config[k]<0 for k in ('ent_coef','vf_coef')):
        raise ValueError('Invalid PPO loss coefficient')
    if not config['net_arch'] or any(type(n) is not int or n<=0 for n in config['net_arch']):
        raise ValueError('Invalid PPO network architecture')
