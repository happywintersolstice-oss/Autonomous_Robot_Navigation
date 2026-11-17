"""History-aware recovery keeps the simulator contract and can add risk."""
import numpy as np
import pytest

from robot_navigation.core import GridMap,Task,NavigationEnv
from robot_navigation.episodes import rollout
from robot_navigation.loop_intervention import RecoveryPolicy,suffix_cycle


class Base:
    policy_id = 'fixture-sarsa'
    def __init__(self,escape=1):
        self.escape = escape
    def scores(self,observation):
        r,c = observation[:2]
        values = np.zeros(4)
        action = 3 if r==1 or c==0 else 2
        values[action] = 10
        if (r,c)==(0,0):
            values[self.escape] = 9.9
        return values


def episode(controller,start=(0,0)):
    task = Task(GridMap('fixture',('...','...')),dict(environment_version='grid-v1',
        observation_version='coordinates-v1',action_version='up-down-left-right-v1',
        reward_version='step-goal-collision-v1',max_steps=20,step_reward=-.01,goal_reward=10.,collision_reward=-5.))
    scenario = {'id':'route','start':list(start),'goal':[1,2],
                'shortest_length':abs(1-start[0])+abs(2-start[1])}
    return rollout(NavigationEnv(task,[scenario]),controller,scenario)


def test_two_cell_recovery_reaches_goal_and_does_not_change_scores():
    base = Base()
    before = base.scores([0,0,1,2]).copy()
    controller = RecoveryPolicy(base,7)
    record = episode(controller)
    assert record['metrics']['outcome']=='success'
    assert record['transitions'][4]['decision']['recovery']
    np.testing.assert_array_equal(base.scores([0,0,1,2]),before)


def test_unaffected_success_path_and_reset():
    controller = RecoveryPolicy(Base(),7)
    record = episode(controller,(1,0))
    assert record['metrics']['steps']==2 and controller.recoveries==0
    controller.reset_episode()
    assert controller.positions==[] and controller.recoveries==0


def test_recovery_can_collide_and_is_not_a_safety_mask():
    record = episode(RecoveryPolicy(Base(escape=2),7))
    assert record['metrics']['outcome']=='collision'
    assert record['transitions'][-1]['decision']['recovery']
    assert record['transitions'][-1]['action']==2


def test_absent_near_ties_keeps_timeout_and_new_suffix_diagnosis():
    record = episode(RecoveryPolicy(Base(),7,margin=.01))
    assert record['metrics']['outcome']=='timeout' and suffix_cycle(record)
    assert not any(t['decision']['recovery'] for t in record['transitions'])


def test_seeded_choices_repeat_and_recovery_budget_is_bounded():
    class Tied:
        policy_id='tied'
        def scores(self,observation):
            return np.zeros(4)
    def choices():
        policy = RecoveryPolicy(Tied(),17011)
        values = [policy.decision([0,i%2,1,2]) for i in range(40)]
        assert sum(v['recovery'] for v in values)==4
        return values
    assert choices()==choices()
