"""Explicit bounded CPU training for four native five-action RL algorithms."""
import argparse,csv,json,os,random,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from .core import ROOT,write_json,digest
from .partial_contract import configuration
from .partial_dynamic_env import PartialDynamicEnv
from .partial_agents import ALGORITHMS
from .partial_features_scaled import build_policy
from .app_registry import RUNS,registry
from .app_scenarios import training_cases

def model_settings(preset):
    return {'discount':.99,'linear_learning_rate':.1,'neural_learning_rate':.0003,'net_arch':[32,32],
        'epsilon_start':1.,'epsilon_end':.1,'buffer_size':4096 if preset=='quick' else 10000,
        'learning_starts':32 if preset=='quick' else 100,'batch_size':32 if preset=='quick' else 50,'train_freq':4,
        'target_update_interval':128 if preset=='quick' else 250,'n_steps':32 if preset=='quick' else 100,'n_epochs':2 if preset=='quick' else 5}

def train(preset='quick',models=ALGORITHMS,actions=None,seed=74001,output=None,job_dir=None):
    if preset not in ('quick','standard'):raise ValueError('Unknown preset')
    models=list(models);actions=int((256 if preset=='quick' else 10000) if actions is None else actions);seed=int(seed)
    if not models or len(set(models))!=len(models) or any(a not in ALGORITHMS for a in models):raise ValueError('Choose native RL algorithms')
    if not 1<=actions<=50000 or not 0<=seed<=2**31-1:raise ValueError('Actions must be 1–50000 per model and seed a nonnegative 31-bit integer')
    cfg=model_settings(preset)
    if 'ppo' in models and actions%cfg['n_steps']:raise ValueError(f"PPO needs an exact multiple of {cfg['n_steps']} actions for this preset; no automatic budget overrun.")
    if 'dqn' in models and actions%cfg['train_freq']:raise ValueError('DQN actions must be a multiple of 4')
    base=Path(output or RUNS).resolve()
    forbidden=[ROOT/'artifacts',ROOT/'data',ROOT.parent/'archive',ROOT.parent/'.git']
    if any(base==p.resolve() or base.is_relative_to(p.resolve()) for p in forbidden):raise ValueError('Training output must be separate from archived evidence and inputs')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    folder=base/'training'/run_id;folder.mkdir(parents=True,exist_ok=False);cases=training_cases();started=perf_counter()
    manifest={'schema':'dynamic-app-training-v1','run_id':run_id,'preset':preset,'seed':seed,'requested_actions_per_model':actions,
        'representation':'B1-relative-x4','scenario_split':'train','case_ids':[c['id'] for c in cases],'case_hash':digest(cases),
        'settings':cfg,'models':{},'status':'running','purpose':'Product development demonstration, not a new Phase 5 scientific study',
        'checkpoint_kind':'inference-only; exact resume is not supported'}
    job=Path(job_dir) if job_dir else folder
    job.mkdir(parents=True,exist_ok=True)
    progress={'status':'running','run_id':run_id,'folder':str(folder),'algorithm':None,'completed_actions':0,'total_actions':actions*len(models),'fraction':0.}
    def publish():
        write_json(folder/'manifest.json',manifest);write_json(job/'progress.json',progress)
    def cancel():
        if (job/'cancel.request').exists():raise InterruptedError('Training cancelled by user; partial run preserved')
        if perf_counter()-started>3600:raise TimeoutError('One-hour product training safety cap')
    publish();history=[]
    try:
        for index,algorithm in enumerate(models):
            cancel();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            env=PartialDynamicEnv(cases,configuration());policy=build_policy(env,algorithm,cfg,seed,'B1-relative-x4')
            env.np_random=np.random.default_rng(seed);env.action_space.seed(seed)
            progress['algorithm']=algorithm;count=0;episodes=0;goals=0;armstart=perf_counter()
            if algorithm in ('q-learning','sarsa'):
                rng=np.random.default_rng(seed);obs,_=env.reset(seed=seed);action=None
                for step in range(actions):
                    cancel();epsilon=1.-.9*min(step/(max(1,actions)*.8),1.)
                    if action is None:action=policy.explore(obs,epsilon,rng)
                    nxt,reward,term,trunc,info=env.step(action);count+=1
                    next_epsilon=1.-.9*min((step+1)/(actions*.8),1.)
                    sampled_next=policy.explore(nxt,next_epsilon,rng) if algorithm=='sarsa' and not term else None
                    policy.update(obs,action,reward,nxt,term,.1,.99,sampled_next)
                    history.append({'algorithm':algorithm,'action_index':count,'reward':reward,'outcome':info['outcome']})
                    if term or trunc:
                        episodes+=1;goals+=info['outcome']=='success';obs,_=env.reset();action=None
                    else:obs=nxt;action=sampled_next if algorithm=='sarsa' else None
                    if count%32==0 or count==actions:
                        progress.update(completed_actions=index*actions+count,fraction=(index*actions+count)/(actions*len(models)));publish()
            else:
                class Progress(BaseCallback):
                    def _on_step(self):
                        nonlocal count,episodes,goals
                        cancel();count=int(self.model.num_timesteps)
                        for info,done,reward in zip(self.locals['infos'],self.locals['dones'],self.locals['rewards']):
                            history.append({'algorithm':algorithm,'action_index':count,'reward':float(reward),'outcome':info.get('outcome','running')})
                            if done:episodes+=1;goals+=info.get('outcome')=='success'
                        if count%32==0 or count==actions:
                            progress.update(completed_actions=index*actions+count,fraction=(index*actions+count)/(actions*len(models)));publish()
                        return True
                policy.model.learn(total_timesteps=actions,callback=Progress(),reset_num_timesteps=False)
                if count!=actions:raise ValueError('Learner did not stop at the exact requested action boundary')
            target=folder/algorithm;target.mkdir();checkpoint=target/('policy.npz' if algorithm in ('q-learning','sarsa') else 'policy.zip')
            policy.policy_id=f'{algorithm}/dynamic-app/{run_id}/seed{seed}'
            policy.save(checkpoint,{'run_id':run_id,'training_actions':count,'training_case_hash':digest(cases),'purpose':manifest['purpose']})
            manifest['models'][algorithm]={'status':'complete','checkpoint':checkpoint.relative_to(folder).as_posix(),'actions':count,
                'training_goals':goals,'completed_episodes':episodes,'wall_seconds':perf_counter()-armstart}
            publish()
        manifest['status']='complete';progress.update(status='complete',fraction=1.,completed_actions=actions*len(models));publish()
    except Exception as exc:
        manifest['status']='cancelled' if isinstance(exc,InterruptedError) else 'failed';manifest['error']=repr(exc)
        progress.update(status=manifest['status'],error=str(exc));publish();raise
    finally:
        with (folder/'training.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['algorithm','action_index','reward','outcome']);w.writeheader();w.writerows(history)
    return {'run_id':run_id,'folder':str(folder),'status':manifest['status'],'actual_actions':sum(m['actions'] for m in manifest['models'].values())}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preset',choices=['quick','standard'],default='quick');p.add_argument('--models',nargs='+',default=['all']);p.add_argument('--actions',type=int);p.add_argument('--seed',type=int,default=74001);p.add_argument('--output',type=Path);p.add_argument('--job-dir',type=Path);p.add_argument('--list-models',action='store_true');a=p.parse_args()
    if a.list_models:r=registry(a.output)
    else:
        selected=[s for part in a.models for s in part.split(',')];selected=list(ALGORITHMS) if selected==['all'] else selected
        r=train(a.preset,selected,a.actions,a.seed,a.output,a.job_dir)
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
