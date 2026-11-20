"""Phase 5A diagnostics. True geometry is used only by trusted retrospective analysis."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import random

import numpy as np

from .partial_contract import ACTION_NAMES, MOVES
from .partial_dynamic_env import PartialDynamicEnv, make_scenario
from .partial_features import encode_features
from .partial_evaluation import episode_metrics


@contextmanager
def preserve_random_state():
    """Loading/evaluating checkpoints must not reseed an ongoing learner."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    import torch
    torch_state = torch.get_rng_state()
    try: yield
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state); torch.set_rng_state(torch_state)


def decision_values(policy, observation):
    if policy.algorithm in ('q-learning', 'sarsa'):
        return {'action_values': policy.scores(observation).tolist()}
    if policy.algorithm in ('dqn', 'ppo'):
        import torch
        value = torch.as_tensor(observation, dtype=torch.float32, device='cpu').unsqueeze(0)
        with torch.inference_mode():
            if policy.algorithm == 'dqn':
                return {'action_values': policy.model.q_net(value)[0].cpu().numpy().tolist()}
            distribution = policy.model.policy.get_distribution(value).distribution.probs[0]
            return {'action_probabilities': distribution.cpu().numpy().tolist(),
                    'state_value': float(policy.model.policy.predict_values(value).item())}
    return {}


def prefix_metrics(initial, transitions):
    """Real incomplete prefixes remain incomplete; no fabricated terminal flags."""
    if not transitions:
        return {'actions': 0, 'return': 0., 'action_counts': dict.fromkeys(ACTION_NAMES, 0),
                'wait_count': 0, 'longest_wait_streak': 0, 'moves': 0, 'repeated_visits': 0,
                'oscillation_windows': 0, 'discovered_cells': int(np.count_nonzero(np.array(initial['memory']['occupancy']) != -1))}
    positions = [tuple(initial['position'])]+[tuple(t['snapshot']['position']) for t in transitions]
    counts = Counter(int(t['action']) for t in transitions)
    seen, repeated, moves, longest, streak = {positions[0]}, 0, 0, 0, 0
    for i, transition in enumerate(transitions):
        if positions[i+1] != positions[i]:
            repeated += positions[i+1] in seen; seen.add(positions[i+1]); moves += 1
        streak = streak+1 if transition['action'] == 4 else 0
        longest = max(longest, streak)
    return {'actions': len(transitions), 'return': float(sum(t['reward'] for t in transitions)),
        'action_counts': {name: counts[i] for i, name in enumerate(ACTION_NAMES)},
        'wait_count': counts[4], 'longest_wait_streak': longest, 'moves': moves,
        'repeated_visits': int(repeated),
        'oscillation_windows': sum(positions[i] == positions[i+2] and positions[i+1] == positions[i+3]
            and positions[i] != positions[i+1] for i in range(len(positions)-3)),
        'discovered_cells': int(np.count_nonzero(np.array(transitions[-1]['snapshot']['memory']['occupancy']) != -1))}


def _summarize(items):
    if not items: raise ValueError('No diagnostic observations')
    n = len(items)
    counts = {outcome: sum(i.get('outcome') == outcome for i in items) for outcome in ('success', 'collision', 'timeout')}
    complete = sum(counts.values())
    steps = sum(i['actions'] for i in items)
    opportunities = sum(i.get('recovery_opportunities', 0) for i in items)
    recoveries = sum(i.get('recoveries', 0) for i in items)
    delays = [d for i in items for d in i.get('observed_to_recovery_delays', [])]
    return {'episodes_or_prefixes': n, 'completed_episodes': complete, 'counts': counts,
        'rates': {o: counts[o]/complete if complete else None for o in counts},
        'incomplete_prefixes': n-complete, 'actions': steps,
        'mean_steps': float(np.mean([i['actions'] for i in items])),
        'length_distribution': {str(k): v for k, v in sorted(Counter(i['actions'] for i in items).items())},
        'mean_completed_return': float(np.mean([i['return'] for i in items if i.get('outcome') in counts])) if complete else None,
        'all_executed_return': float(sum(i['return'] for i in items)),
        'action_counts': {name: sum(i['action_counts'][name] for i in items) for name in ACTION_NAMES},
        'wait_count': sum(i['wait_count'] for i in items),
        'wait_fraction': sum(i['wait_count'] for i in items)/steps if steps else None,
        'longest_wait_streak': max(i['longest_wait_streak'] for i in items),
        'moves': sum(i['moves'] for i in items), 'repeated_visits': sum(i['repeated_visits'] for i in items),
        'oscillation_windows': sum(i['oscillation_windows'] for i in items),
        'mean_discovered_cells': float(np.mean([i['discovered_cells'] for i in items])),
        'recovery_opportunities': opportunities, 'recoveries': recoveries,
        'recovery_rate': recoveries/opportunities if opportunities else None,
        'mean_recovery_delay': float(np.mean(delays)) if delays else None,
        **{key: sum(i.get(key, 0) for i in items) for key in ('nominal_event_count', 'applied_event_count',
            'detected_event_count', 'events_never_observed', 'applied_event_ticks', 'suppressed_activation_ticks',
            'sensor_cell_refreshes', 'occupancy_corrections')}}


def evaluation_summary(records):
    items = []
    for record in records:
        basic = prefix_metrics(record['initial_snapshot'], record['transitions'])
        items.append({**basic, **record['metrics']})
    return _summarize(items)


def training_summary(records):
    """Completed-episode rates exclude curriculum/budget-cutoff prefixes."""
    items, case, initial, transitions, phase = [], None, None, [], None
    def finish():
        if case is None or not transitions: return
        item = prefix_metrics(initial, transitions)
        item['phase'] = phase; item['category'] = case['fixture_category']
        if transitions[-1]['terminated'] or transitions[-1]['truncated']:
            item.update(episode_metrics({'scenario': case, 'initial_snapshot': initial, 'transitions': transitions}))
        else: item['outcome'] = 'incomplete_prefix'
        items.append(item)
    for record in records:
        if record['kind'] == 'reset':
            finish(); case, initial, transitions = record['scenario'], record['snapshot'], []
            phase = record['phase']
        elif record['kind'] == 'step': transitions.append(record)
        else: raise ValueError('Unknown training record')
    finish()
    if not items: return {'actions': 0, 'completed_episodes': 0, 'items': []}
    return {'overall': _summarize(items),
            'by_category': {c: _summarize([i for i in items if i['category'] == c]) for c in sorted({i['category'] for i in items})},
            'episode_ledger': items}


def failure_examples(record):
    transitions = record['transitions']; result = {}; streak = 0
    initial = record['initial_snapshot']
    for i, transition in enumerate(transitions):
        before = initial if i == 0 else transitions[i-1]['snapshot']
        known = np.asarray(transition['observation'][8:108]).reshape(10, 10)
        blocked = np.argwhere(known == 1)
        def save(name, index=i):
            if name not in result:
                t = transitions[index]
                result[name] = {'action_index': index+1, 'observation_before_action': t['observation'],
                    'chosen_action': int(t['action']), 'decision': t.get('decision', {}),
                    'outcome_after_action': t['info']['outcome'],
                    'note': 'Only the recorded 409-value knowledge is shown here; no hidden obstacle label.'}
        if len(blocked): save('first_observed_obstacle')
        if transition['info']['outcome'] == 'collision': save('first_attempted_collision')
        streak = streak+1 if transition['action'] == 4 else 0
        if streak >= 5: save('first_sustained_wait', i-4)
        if (record['scenario']['fixture_category'] == 'B' and len(blocked)
                and sum(abs(a-b) for a, b in zip(transition['snapshot']['position'], before['goal']))
                > sum(abs(a-b) for a, b in zip(before['position'], before['goal']))):
            save('movement_away_from_observed_wall_not_an_intent_label')
        if record['metrics']['recoveries'] and transition['info']['outcome'] == 'success':
            save('successful_after_detected_route_obstruction')
    return result


def reward_sanity(gamma=.99):
    values = {'collision_immediately': [-5.01], 'wait_200': [-.01]*200}
    for n in (1, 2, 7, 10): values[f'success_in_{n}_actions'] = [-.01]*(n-1)+[9.99]
    return {name: {'actions': len(rs), 'undiscounted_return': float(sum(rs)),
                   'discounted_return': float(sum(gamma**i*r for i, r in enumerate(rs)))} for name, rs in values.items()}


def representation_probe():
    observations = []
    for goal in ([5, 4], [5, 6]):
        for blocked in (False, True):
            rows = ['.'*10]*10
            if blocked: rows[5] = '.......#..'
            env = PartialDynamicEnv([make_scenario(rows, [5, 5], goal)])
            observations.append(env.reset(seed=0)[0])
    features = [encode_features(o) for o in observations]
    # These are independent, permitted goal and occupancy factors, not desired
    # navigation labels. Their XOR is a controlled capacity example only.
    error = float(np.max(np.abs(features[0]+features[3]-features[1]-features[2])))
    near_rows = ['.'*10]*10; near_rows[5] = '......#...'
    near = PartialDynamicEnv([make_scenario(near_rows, [5, 5], [5, 4])]).reset()[0]
    return {'feature_count': len(features[0]), 'raw_information_preserved': True,
        'right_ray_no_obstacle': float(observations[0][6]), 'right_ray_distance_two': float(observations[1][6]),
        'right_ray_distance_one': float(near[6]),
        'immediate_and_far_right_obstacle_distinguishable': not np.array_equal(encode_features(near), features[1]),
        'independent_goal_occupancy_parallelogram_error': error,
        'parity_label_example': [1, -1, -1, 1],
        'interpretation': 'An affine action-score difference cannot strictly separate XOR labels on this parallelogram. '
            'These labels are synthetic, not optimal navigation actions; this does not prove A/B unlearnable. '
            'The transform is injective on the 409 input, so this is a capacity issue, not information deletion.'}


def model_diagnostics(policy):
    values = {'fingerprint': policy.state_fingerprint(), 'parameters_finite': True}
    if policy.algorithm in ('q-learning', 'sarsa'):
        values.update(weight_norm=float(np.linalg.norm(policy.weights)), parameters_finite=bool(np.isfinite(policy.weights).all()))
        return values
    import torch
    model = policy.model
    values.update(model_timesteps=int(model.num_timesteps),
        parameters_finite=all(bool(torch.isfinite(v).all()) for v in model.policy.state_dict().values()),
        sb3_version_specific_updates_counter=int(getattr(model, '_n_updates', 0)),
        updates_counter_meaning='DQN gradient updates; PPO epochs (SB3 2.9.2), not equal compute')
    if policy.algorithm == 'dqn':
        buffer = model.replay_buffer; n = buffer.buffer_size if buffer.full else buffer.pos
        values['replay'] = {'stored_transitions': int(n), 'capacity': int(buffer.buffer_size),
            'positive_goal_rewards_stored': int(np.count_nonzero(buffer.rewards[:n] > 0)),
            'timeouts_stored': int(np.count_nonzero(buffer.timeouts[:n])),
            'learning_starts': int(model.learning_starts), 'exploration_rate': float(model.exploration_rate)}
    else:
        values['completed_rollouts'] = int(model.num_timesteps//model.n_steps)
        values['last_rollout_returns_finite'] = bool(np.isfinite(model.rollout_buffer.returns).all())
    return values
