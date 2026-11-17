from dataclasses import replace
import random

import numpy as np
import pytest
import torch
from robot_navigation.agents import load_policy
from robot_navigation.core import GridMap,Task,NavigationEnv,load_task,read_json,write_json
from robot_navigation.dqn import normalize
from robot_navigation.ppo import PPOPolicy,build_model,validate_config
from robot_navigation.workflow import evaluate


def fixture():
    task=Task(GridMap('ppo-fixture',('...',)),{**load_task().config,'max_steps':1})
    scenario={'id':'fixture','start':[0,0],'goal':[0,1],'shortest_length':1}
    config=dict(algorithm='ppo',run_id='fixture-ppo',seed=7,interactions=1000,monitor_every=1000,
        learning_rate=.003,discount=.99,n_steps=20,batch_size=10,n_epochs=4,gae_lambda=.95,
        clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,net_arch=[16,16],torch_threads=1)
    return task,scenario,config


def test_ppo_learns_probabilities_read_only_inference_and_checkpoint_integrity(tmp_path):
    task,scenario,config=fixture()
    validate_config(config)
    model=build_model(NavigationEnv(task,[scenario]),config)
    model.learn(total_timesteps=config['interactions'])
    policy=PPOPolicy(task,model,'ppo/fixture')
    before=policy.state_fingerprint()
    numpy_rng,torch_rng,python_rng=np.random.get_state(),torch.get_rng_state(),random.getstate()
    decision=policy.decision([0,0,0,1])
    assert decision['action']==3 and sum(decision['action_probabilities'])==pytest.approx(1.)
    assert all(0<=p<=1 for p in decision['action_probabilities'])
    assert int(model.predict(normalize([0,0,0,1],task.grid.shape),deterministic=True)[0])==3
    summary,_,records=evaluate(task,[scenario],policy,7,'fixture')
    assert summary['success_rate']==1 and policy.state_fingerprint()==before
    assert records[0]['transitions'][0]['scores'] is None
    assert records[0]['transitions'][0]['decision']==decision
    after=np.random.get_state()
    assert numpy_rng[0]==after[0] and np.array_equal(numpy_rng[1],after[1]) and numpy_rng[2:]==after[2:]
    assert torch.equal(torch_rng,torch.get_rng_state()) and python_rng==random.getstate()
    path=tmp_path/'model.zip';policy.save(path,{'training_interactions':1000})
    loaded=load_policy(path,task)
    assert loaded.algorithm=='ppo' and loaded.decision([0,0,0,1])==decision
    with pytest.raises(ValueError,match='mismatch'):
        load_policy(path,replace(task,config={**task.config,'step_reward':-.1}))
    metadata=read_json(path.with_suffix('.json'))
    write_json(path.with_suffix('.json'),{**metadata,'encoding':'wrong'})
    with pytest.raises(ValueError,match='encoding mismatch'):
        load_policy(path,task)
    write_json(path.with_suffix('.json'),metadata)
    path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='checksum'):
        load_policy(path,task)


@pytest.mark.parametrize('goal,expected',[([0,2],1.97),([0,1],9.99)])
def test_ppo_bootstraps_only_timeouts_using_sb3_terminal_observation(goal,expected):
    task,scenario,config=fixture()
    scenario={**scenario,'goal':goal,'shortest_length':goal[1]}
    model=build_model(NavigationEnv(task,[scenario]),{**config,'n_steps':2,'batch_size':2,'n_epochs':1})
    def right(observation,deterministic=False):
        n=len(observation)
        return torch.full((n,),3,dtype=torch.long),torch.zeros((n,1)),torch.zeros(n)
    observed=[]
    model.policy.forward=right
    def value(observation):
        observed.append(observation.cpu().numpy().copy())
        return torch.full((len(observation),1),2.)
    model.policy.predict_values=value
    model.learn(total_timesteps=2)
    assert np.allclose(model.rollout_buffer.rewards,expected)
    if goal==[0,2]:
        assert np.array_equal(observed[0][0],normalize([0,1,*goal],task.grid.shape))


def test_ppo_training_monitor_follows_updates_and_budget_is_exact(tmp_path,monkeypatch):
    import robot_navigation.part_a as part_a
    task,scenario,config=fixture()
    for name in ('src','app','configs'):
        (tmp_path/name).mkdir()
    monkeypatch.setattr(part_a,'load_task',lambda root:task)
    monkeypatch.setattr(part_a,'load_scenarios',lambda task,split,root:[scenario])
    config={**config,'interactions':400,'monitor_every':200}
    directory,summary,pilot=part_a.train_part_a(config,tmp_path)
    assert pilot['training_interactions']==400
    assert pilot['ppo_epoch_passes']==80 and pilot['optimizer_steps']==160
    monitor=read_json(directory/'monitoring.json')
    assert [m['training_interactions'] for m in monitor]==[200,400]
    for interaction,updates in ((200,40),(400,80)):
        checkpoint=load_policy(directory/f'checkpoints/step-{interaction:07d}.zip',task)
        assert checkpoint.model._n_updates==updates
    assert summary['ppo']['episodes']==1
    with pytest.raises(FileExistsError):
        part_a.train_part_a(config,tmp_path)
