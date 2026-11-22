"""Synchronized reward ablations retain their distinct task signatures."""
from .episodes import ComparisonPlayback


class RewardComparisonPlayback(ComparisonPlayback):
    def __init__(self,episodes,tasks):
        if len(episodes)!=2 or len(tasks)!=2:
            raise ValueError('Two reward-comparison episodes and tasks are required')
        ignored = {'reward_version','step_reward','goal_reward','collision_reward','shaping_scale'}
        dynamics = lambda task:{k:v for k,v in task.config.items() if k not in ignored}
        if tasks[0].grid!=tasks[1].grid or dynamics(tasks[0])!=dynamics(tasks[1]):
            raise ValueError('Reward comparison requires matching geometry and dynamics')
        if any(episodes[0][key]!=episodes[1][key] for key in ('scenario','mode','seed')):
            raise ValueError('Reward comparison requires the same scenario, mode and replay seed')
        for task,episode in zip(tasks,episodes,strict=True):
            task.check(episode['signature'])
        self.episodes = tuple(episodes)
        self.index,self.playing = 0,False
        self.length = max(len(e['transitions']) for e in episodes)
