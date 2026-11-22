"""Bounded observations of actual training actions; never calls a policy or RNG."""
from copy import deepcopy
from bisect import bisect_right
import hashlib
from pathlib import Path

from .core import write_json
from .episodes import episode_metrics, validate_replay


def trace_limit(config):
    value = config.get('training_trace_episodes', 0)
    if type(value) is not int or not 0 <= value <= 10:
        raise ValueError('training_trace_episodes must be an integer from 0 to 10')
    starts = config.get('training_trace_starts')
    if starts is not None:
        if (not isinstance(starts, list) or not starts or not value
                or any(type(n) is not int or n < 0 for n in starts)
                or starts != sorted(set(starts)) or starts[0] != 0
                or value % len(starts) or starts[-1] >= config.get('interactions', 0)):
            raise ValueError('Trace windows must start at zero, be unique/increasing within the budget and divide the episode limit')
    return value


class TrainingTraces:
    """Keep bounded chronological samples, never selecting by observed outcome."""
    def __init__(self, task, directory, config):
        self.task, self.directory = task, Path(directory)
        self.limit = trace_limit(config)
        self.starts = config.get('training_trace_starts')
        self.window_counts = [0] * len(self.starts) if self.starts else None
        self.seed = config['seed']
        self.policy_id = f"{config['algorithm']}/{config['run_id']}/training-behavior"
        self.pending, self.records, self.closed = None, [], False

    def start(self, observation, scenario, interaction):
        if self.closed or len(self.records) >= self.limit:
            return
        if self.pending is not None and self.pending['transitions']:
            raise RuntimeError('Training trace reset before the previous episode ended')
        window = bisect_right(self.starts, interaction)-1 if self.starts else None
        if self.starts and (window < 0 or self.window_counts[window] >= self.limit // len(self.starts)):
            return
        self.pending = {'signature': self.task.signature, 'scenario': deepcopy(scenario),
                        'seed': self.seed, 'policy_id': self.policy_id, 'mode': 'training',
                        'initial_observation': observation.tolist(), 'transitions': [],
                        'training_start_interaction': interaction,
                        'behavior': 'actual exploratory training actions; policy may change within an episode; no frozen-policy scores recorded'}
        if self.starts:
            self.pending.update(recording_window_index=window, recording_window_start_interaction=self.starts[window])

    def transition(self, observation, action, reward, next_observation, terminated, truncated, info, interaction):
        if self.pending is None:
            return
        self.pending['transitions'].append({'observation': observation.tolist(), 'action': int(action),
            'reward': float(reward), 'next_observation': next_observation.tolist(),
            'terminated': bool(terminated), 'truncated': bool(truncated), 'info': deepcopy(info), 'scores': None})
        self.pending['training_end_interaction'] = interaction
        if terminated or truncated:
            self.pending['complete'] = True
            self.pending['metrics'] = episode_metrics(self.pending)
            validate_replay(self.task, self.pending)
            self._save()

    def _save(self):
        path = self.directory/'training_traces'/f'episode-{len(self.records)+1:03d}.json'
        if path.exists():
            raise FileExistsError('Training traces must not overwrite an existing record')
        write_json(path, self.pending)
        item = {'path': path.relative_to(self.directory).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'complete': self.pending['complete'], 'actions': len(self.pending['transitions'])}
        if self.starts:
            window = self.pending['recording_window_index']
            self.window_counts[window] += 1
            item.update(recording_window_index=window,
                        training_start_interaction=self.pending['training_start_interaction'],
                        training_end_interaction=self.pending['training_end_interaction'])
        self.records.append(item)
        self.pending = None

    def finish(self, reason='budget_cutoff'):
        if not self.closed and self.pending is not None and self.pending['transitions']:
            self.pending.update(complete=False, cutoff_reason=reason)
            # A budget cutoff is not a task timeout or a completed-episode metric.
            self._save()
        self.closed = True
        result = {'requested_episodes': self.limit, 'selection': 'first episodes in chronological order; no outcome selection',
                'records': self.records, 'extra_environment_interactions': 0,
                'historical_training_reconstructed': False}
        if self.starts:
            result.update(selection='first episodes starting at/after each declared training-step boundary; no outcome selection',
                          window_starts=self.starts, episodes_per_window=self.limit // len(self.starts))
        return result
