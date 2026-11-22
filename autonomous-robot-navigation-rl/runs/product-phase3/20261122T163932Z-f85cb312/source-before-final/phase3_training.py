"""Fresh bounded Phase3 learning on the train file only; never reads holdout cases."""
import csv,random
from pathlib import Path
from time import perf_counter
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from .core import ROOT,write_json,digest
from .partial_dynamic_env import PartialDynamicEnv
from .partial_contract import configuration
from .phase3_data import train_dev,TRAIN_PATH
from .phase3_features import PublicHistory,STATE,VERSION,build_policy
from .app_service import identity
from .app_registry import RUNS
from .release import sha

class HistoryCurriculumEnv(gym.Wrapper):
    def __init__(self,budget,seed,objective):
        self.pool=train_dev()['training'];super().__init__(PartialDynamicEnv(self.pool,configuration()));self.observation_space=gym.spaces.Box(-1.,1.,(STATE,),np.float32);self.budget=budget;self.actions=0;self.rng=np.random.default_rng(seed);self.objective=objective;self.history=PublicHistory();self.previous=None
    def reset(self,*,seed=None,options=None):
        if seed is not None:self.rng=np.random.default_rng(seed)
        fraction=self.actions/max(1,self.budget);stage=0 if fraction<.10 else 1 if fraction<.30 else 2;pool=[c for c in self.pool if c['curriculum_stage']<=stage]
        if not pool:raise ValueError('Curriculum stage has no cases')
        case=pool[int(self.rng.integers(len(pool)))];obs,info=self.env.reset(seed=seed,options={'scenario':case});self.history=PublicHistory();self.previous=obs.copy();return self.history.state(obs),info
    def step(self,action):
        action=int(action);old=self.previous.copy();was_wait=self.history.waits;alternative=self.history.observed_alternative(old);recent=[tuple(p) for p in list(self.history.positions)[-4:]]
        obs,reward,term,trunc,info=self.env.step(action);extra=0.
        if self.objective=='liveness-v1':
            progress=float((np.abs(old[2:4]-old[:2]).sum()-np.abs(obs[2:4]-obs[:2]).sum())*9)
            extra=.10*progress-.04*int(action!=4 and tuple(obs[:2]) in recent)-.02*int(action==4 and was_wait>=3 and alternative)
        self.history.remember(old,action);self.previous=obs.copy();self.actions+=1
        return self.history.state(obs),float(reward+extra),term,trunc,{**info,'official_reward':reward,'training_shaping':extra}

def train(preset='standard',models=('q-learning','sarsa','dqn','ppo'),actions=None,seed=93511,output=None,job_dir=None,objective='none',study_id=None,candidate=None):
    if preset not in ('quick','standard') or objective not in ('none','liveness-v1'):raise ValueError('Invalid Phase3 training configuration')
    actions=(256 if preset=='quick' else 10000) if actions is None else int(actions);models=list(models);rollout=32 if preset=='quick' else 200
    if not models or len(models)!=len(set(models)) or any(a not in ('q-learning','sarsa','dqn','ppo') for a in models):raise ValueError('Choose genuine independent RL families')
    if not 1<=actions<=50000 or not 0<=int(seed)<=2**31-1:raise ValueError('Actions 1..50000/model; nonnegative31-bit seed required')
    if 'ppo' in models and actions%rollout:raise ValueError('PPO needs exact rollout multiple '+str(rollout))
    if 'dqn' in models and actions%4:raise ValueError('DQN needs multiple of4')
    base=Path(output or RUNS).resolve()
    for p in (ROOT/'artifacts',ROOT/'data',ROOT.parent/'archive'):
        if base.is_relative_to(p.resolve()):raise ValueError('Training output cannot overwrite history or inputs')
    run_id=identity();folder=base/'training'/run_id;folder.mkdir(parents=True,exist_ok=False);job=Path(job_dir) if job_dir else folder;job.mkdir(parents=True,exist_ok=True);data=train_dev();started=perf_counter();history=[];env=None;active=False
    manifest={'schema':'phase3-training-v1','product_phase':3,'run_id':run_id,'preset':preset,'seed':int(seed),'requested_actions_per_model':actions,'representation':VERSION,'agent_state_values':STATE,'shaping':objective,'scenario_split':'train','case_hash':digest(data['training']),'case_ids':[c['id'] for c in data['training']],'phase3_study_id':study_id,'candidate':candidate,'status':'running','models':{},'checkpoint_kind':'inference-only, not exact optimizer/buffer/RNG resume','training_source_sha256':{n:sha(ROOT/'src/robot_navigation'/n) for n in ('phase3_features.py','phase3_training.py','phase3_data.py')},'train_file_sha256':sha(TRAIN_PATH)}
    progress={'status':'running','run_id':run_id,'folder':str(folder),'algorithm':None,'completed_actions':0,'total_actions':actions*len(models),'fraction':0.}
    def publish():write_json(folder/'manifest.json',manifest);write_json(job/'progress.json',progress)
    def check():
        if (job/'cancel.request').exists():raise InterruptedError('Cancelled; no models from this run are registered')
        if perf_counter()-started>3600:raise TimeoutError('One-hour worker limit')
    publish()
    try:
        for index,algorithm in enumerate(models):
            check();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);env=HistoryCurriculumEnv(actions,seed,objective);policy=build_policy(env,algorithm,seed,preset=='quick');active=True
            before=policy.state_fingerprint();initial=policy.weights.copy() if hasattr(policy,'weights') else torch.cat([p.detach().cpu().flatten() for p in policy.model.policy.parameters()]).numpy().copy();count=goals=episodes=optimizer_steps=0;progress['algorithm']=algorithm
            def log(action_count,reward,info):
                nonlocal goals,episodes
                history.append({'algorithm':algorithm,'action_index':action_count,'training_reward':float(reward),'official_reward':float(info['official_reward']),'shaping':float(info['training_shaping']),'outcome':info['outcome']})
                if info['outcome']!='running':episodes+=1;goals+=info['outcome']=='success'
                if action_count%100==0 or action_count==actions:progress.update(completed_actions=index*actions+action_count,fraction=(index*actions+action_count)/(len(models)*actions));publish()
            if algorithm in ('q-learning','sarsa'):
                rng=np.random.default_rng(seed);obs,_=env.reset(seed=seed);action=None
                for step in range(actions):
                    check();epsilon=1.-.92*min(step/max(1,actions*.8),1.)
                    if action is None:action=policy.explore(obs,epsilon,rng)
                    nxt,reward,term,trunc,info=env.step(action);count+=1;next_epsilon=1.-.92*min((step+1)/max(1,actions*.8),1.);sampled=policy.explore(nxt,next_epsilon,rng) if algorithm=='sarsa' and not term else None
                    policy.update(obs,action,reward,nxt,term,.1,.99,sampled);log(count,reward,info)
                    if term or trunc:obs,_=env.reset();action=None
                    else:obs=nxt;action=sampled if algorithm=='sarsa' else None
            else:
                original=policy.model.policy.optimizer.step
                def observed(*a,**kw):
                    nonlocal optimizer_steps
                    result=original(*a,**kw);optimizer_steps+=1;return result
                policy.model.policy.optimizer.step=observed
                class Callback(BaseCallback):
                    def _on_step(self):
                        nonlocal count
                        count=int(self.model.num_timesteps);check()
                        for reward,info in zip(self.locals['rewards'],self.locals['infos']):log(count,reward,info)
                        return True
                policy.model.learn(total_timesteps=actions,callback=Callback(),reset_num_timesteps=False);policy.model.policy.optimizer.step=original
            if count!=actions or env.actions!=actions:raise ValueError('Exact requested action boundary violated')
            final=policy.weights if hasattr(policy,'weights') else torch.cat([p.detach().cpu().flatten() for p in policy.model.policy.parameters()]).numpy();change=float(np.linalg.norm(final-initial))
            if change<=0 or before==policy.state_fingerprint():raise ValueError('No actual learning parameter change')
            evidence={'td_updates':count if hasattr(policy,'weights') else 0,'optimizer_steps':optimizer_steps,'sb3_update_counter':int(policy.model._n_updates) if hasattr(policy,'model') else None,'target_weight_refreshes':getattr(policy,'target_updates',None),'parameter_l2_change':change,'initial_parameter_sha256':before,'final_parameter_sha256':policy.state_fingerprint()}
            target=folder/algorithm;target.mkdir();path=target/('policy.npz' if hasattr(policy,'weights') else 'policy.zip');policy.policy_id=f'{algorithm}/phase3/{run_id}/seed{seed}';policy.save(path,{'run_id':run_id,'training_actions':count,'learning_evidence':evidence,'training_case_hash':manifest['case_hash'],'objective':objective,'study_id':study_id,'candidate':candidate})
            effective={'alpha':.1,'gamma':.99,'td_clip':5,'weight_norm_bound':100,'target_interval':100 if algorithm=='q-learning' else None} if hasattr(policy,'weights') else {'gamma':policy.model.gamma,'learning_rate':float(policy.model.learning_rate),'batch_size':policy.model.batch_size,'parameter_shapes':{n:list(v.shape) for n,v in policy.model.policy.state_dict().items()}}
            manifest['models'][algorithm]={'status':'complete','checkpoint':path.relative_to(folder).as_posix(),'actions':count,'training_goals':goals,'completed_episodes':episodes,'learning_evidence':evidence,'effective_settings':effective,'wall_seconds':perf_counter()-started};active=False;publish()
        manifest['status']='complete';progress.update(status='complete',completed_actions=actions*len(models),fraction=1.);publish()
    except BaseException as exc:
        manifest.update(status='cancelled' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'failed',error=repr(exc),actual_actions=sum(v['actions'] for v in manifest['models'].values())+(env.actions if active and env is not None else 0));progress.update(status=manifest['status'],completed_actions=manifest['actual_actions'],error=str(exc));publish();raise
    finally:
        with (folder/'training.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['algorithm','action_index','training_reward','official_reward','shaping','outcome']);w.writeheader();w.writerows(history)
    return {'run_id':run_id,'folder':str(folder),'status':'complete','actual_actions':sum(m['actions'] for m in manifest['models'].values())}
