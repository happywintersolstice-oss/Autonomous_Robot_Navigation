"""Topology, backtracking and potential rewards are executable contracts."""
import numpy as np
import pytest

from robot_navigation.core import ROOT,GridMap,Task,NavigationEnv
from robot_navigation.navigation_lab import (RewardEnv,load_layouts,topology,configuration,
    make_task,validate_record,route_splits)
from robot_navigation.agents import BFSPolicy
from robot_navigation.episodes import rollout


def fixture_task(scale=.2,cap=20):
    return Task(GridMap('fixture',('...','.#.','...')),dict(map_id='fixture',
        environment_version='grid-v1',observation_version='coordinates-v1',
        action_version='up-down-left-right-v1',reward_version='fixture-potential',
        max_steps=cap,step_reward=-.05,goal_reward=10.,collision_reward=-5.,discount=.99,shaping_scale=scale))


def test_graph_topology_and_routes_are_reproducible_and_nontrivial():
    config = configuration()
    layouts = load_layouts()
    assert len(layouts)==4
    for layout in layouts:
        grid = GridMap(layout['map_id'],tuple(layout['rows']))
        stats = topology(grid)
        assert stats==layout['topology'] and stats['connected']
        assert stats['dead_ends']>=3 and stats['junctions']>=3 and stats['corridor_cells']>=8
        assert stats['deepest_dead_end']>=3 and stats['diameter']>=20
        assert stats['ordered_pairs_requiring_detour']>0
        assert route_splits(grid,config)==layout['splits']
        splits = [{s['id'] for s in layout['splits'][name]} for name in ('train','validation','reserved')]
        assert not splits[0]&splits[1] and not splits[0]&splits[2] and not splits[1]&splits[2]
        assert {s['id'] for s in layout['splits']['monitoring']}<=splits[1]
    assert layouts[0]['topology']['cycle_rank']==0
    assert layouts[1]['topology']['cycle_rank']>0
    assert layouts[3]['topology']['deepest_dead_end']>=8


def test_discounted_shaping_telescopes_on_return_cycle():
    task = fixture_task()
    scenario = {'id':'cycle','start':[0,0],'goal':[2,2],'shortest_length':4}
    env = RewardEnv(task,[scenario]);obs,_ = env.reset()
    initial = RewardEnv.potential(obs)
    rewards = []
    for action in (3,2):
        obs,reward,end,cut,info = env.step(action)
        rewards.append(info['reward_components']['potential'])
    assert not end and not cut
    assert np.isclose(rewards[0]+.99*rewards[1],.2*(.99**2*initial-initial))


def test_goal_and_collision_use_absorbing_potential_but_timeout_bootstraps():
    task = fixture_task(cap=1)
    scenario = {'id':'one','start':[0,0],'goal':[0,1],'shortest_length':1}
    env = RewardEnv(task,[scenario]);env.reset()
    _,reward,end,cut,info = env.step(3)
    assert end and not cut and np.isclose(reward,9.95+.2)
    env.reset();_,reward,end,cut,info = env.step(0)
    assert end and not cut and np.isclose(reward,-5.05+.2)
    env.reset();obs,reward,end,cut,info = env.step(1)
    assert cut and not end
    assert np.isclose(info['reward_components']['potential'],.2*(.99*-2-(-1)))


def test_backtracking_is_legal_and_has_no_hidden_revisit_penalty():
    task = fixture_task()
    scenario = {'id':'backtrack','start':[0,0],'goal':[2,2],'shortest_length':4}
    env = RewardEnv(task,[scenario]);env.reset()
    env.step(3)
    _,reward,end,cut,info = env.step(2)
    assert not end and not cut
    assert set(info['reward_components'])=={'base','potential'}
    assert info['reward_components']['base']==-.05


def test_lab_replay_revalidates_actual_reward_and_rejects_tampering():
    task = fixture_task()
    scenario = {'id':'path','start':[0,0],'goal':[2,2],'shortest_length':4}
    record = rollout(RewardEnv(task,[scenario]),BFSPolicy(task.grid),scenario)
    assert validate_record(task,record)['metrics']['spl']==1
    record['transitions'][0]['reward'] += 1
    with pytest.raises(ValueError,match='reward differs'):
        validate_record(task,record)
