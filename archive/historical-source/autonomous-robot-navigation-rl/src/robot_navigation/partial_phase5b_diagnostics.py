"""Read-only historical diagnosis and shared Phase 5B numerical instrumentation."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import pickle
import random

import numpy as np
import torch

from .core import ROOT, digest, read_json, write_json
from .release import sha
from .partial_agents import LinearPolicy, load_policy
from .partial_commands import read_records
from .partial_contract import ACTION_NAMES, MOVES, configuration
from .partial_dynamic_env import PartialDynamicEnv
from .partial_feasibility_fixtures import cases
from .partial_features import encode_features
from .partial_features_scaled import encode_scaled, ARMS, parameter_norm
from .partial_pilot_diagnostics import decision_values, preserve_random_state
from .partial_pilot_ledger import ContinuousLinear, epsilon_at

GROUPS = {'bias': (0, 1), 'position': (1, 3), 'goal': (3, 5), 'rays': (5, 9),
          'occupancy': (9, 109), 'age': (109, 209), 'visits': (209, 309),
          'streak': (309, 409), 'elapsed': (409, 410), 'relative_goal': (410, 412)}


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return None
    if not np.isfinite(a).all():
        raise FloatingPointError('Nonfinite diagnostic')
    return {'n': int(a.size), 'min': float(a.min()), 'median': float(np.median(a)),
            'mean': float(a.mean()), 'max': float(a.max()), 'p95': float(np.quantile(a, .95))}


def feature_statistics(observations, feature_version=ARMS['B0-original']):
    matrix = np.stack([encode_scaled(np.asarray(o, dtype=np.float32), feature_version) for o in observations])
    return {'training_observations_only': True, 'count': len(matrix),
        'feature_norm': distribution(np.linalg.norm(matrix, axis=1)),
        'relative_goal_energy_fraction': distribution(np.sum(matrix[:, -2:]**2, axis=1)/np.sum(matrix**2, axis=1)),
        'per_feature': {'mean': matrix.mean(axis=0).tolist(), 'mean_absolute': np.abs(matrix).mean(axis=0).tolist(),
            'min': matrix.min(axis=0).tolist(), 'max': matrix.max(axis=0).tolist(),
            'nonzero_frequency': np.mean(matrix != 0, axis=0).tolist(),
            'absolute_one_frequency': np.mean(np.abs(matrix) == 1, axis=0).tolist()},
        'groups': {name: {'absolute': distribution(np.abs(matrix[:, lo:hi]).ravel()),
                         'nonzero_frequency': float(np.mean(matrix[:, lo:hi] != 0))}
                   for name, (lo, hi) in GROUPS.items()},
        'clipping_note': 'Unit saturation is measured, not assumed clipping. Memory age/visit/streak/time use their unchanged encoder caps.'}


def components(policy, observation, feature_version=ARMS['B0-original']):
    phi = encode_scaled(np.asarray(observation, dtype=np.float32), feature_version)
    return {name: (policy.weights[:, lo:hi] @ phi[lo:hi]).tolist() for name, (lo, hi) in GROUPS.items()}


def decision_diagnostic(policy, observation):
    observation = np.asarray(observation, dtype=np.float32)
    values = decision_values(policy, observation)
    scores = np.asarray(values.get('action_values', values.get('action_probabilities')))
    delta = observation[2:4]-observation[:2]
    correct = [i for i, (dr, dc) in enumerate(MOVES[:4]) if dr*delta[0]+dc*delta[1] > 0]
    other = [i for i in range(5) if i not in correct]
    chosen = policy.act(observation.copy())
    position = np.rint(observation[:2]*9).astype(int)
    attempted = position+MOVES[chosen]
    inside = bool(np.all((attempted >= 0) & (attempted < 10)))
    known = observation[8:108].reshape(10, 10)
    invalid = chosen != 4 and (not inside or known[tuple(attempted)] == 1)
    result = {'observation_before_action': observation.tolist(), 'relative_goal': delta.tolist(),
        'decision': values, 'chosen_action': chosen, 'goal_actions': correct,
        'goal_direction_margin': float(scores[correct].max()-scores[other].max()) if correct else None,
        'chosen_margin': float(np.sort(scores)[-1]-np.sort(scores)[-2]),
        'first_step_goal_directed': chosen in correct, 'known_invalid_attempt': bool(invalid),
        'argmax_wait': chosen == 4}
    if 'action_probabilities' in values:
        probs = np.maximum(scores, 1e-30)
        result['entropy'] = float(-np.sum(probs*np.log(probs)))
    if hasattr(policy, 'weights'):
        result['weighted_components'] = components(policy, observation,
            getattr(policy, 'signature', {}).get('feature_version', ARMS['B0-original']))
        c = result['weighted_components']; relative = np.asarray(c['relative_goal'])
        others = sum(np.abs(c[n]) for n in c if n != 'relative_goal')
        result['relative_to_other_absolute_component_ratio'] = np.divide(np.abs(relative), others,
            out=np.zeros(5), where=others != 0).tolist()
        result['zero_other_component_denominator'] = (others == 0).tolist()
    return result


def optimizer_state(policy):
    if not hasattr(policy, 'model'):
        return {'parameters_finite': bool(np.isfinite(policy.weights).all()), 'parameter_norm': parameter_norm(policy)}
    model = policy.model
    optimizer = model.policy.optimizer if policy.algorithm == 'ppo' else model.policy.optimizer
    tensors = [v for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor)]
    grads = [p.grad for p in model.policy.parameters() if p.grad is not None]
    return {'parameter_norm': parameter_norm(policy),
        'parameters_finite': all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()),
        'optimizer_tensors_finite': all(bool(torch.isfinite(p).all()) for p in tensors),
        'saved_gradients_available': bool(grads),
        'saved_gradients_finite': all(bool(torch.isfinite(p).all()) for p in grads) if grads else None,
        'gradient_limitation': 'Saved inference checkpoints omit live gradients; Adam moments/steps are inspected. Live Phase 5B optimizer steps are instrumented.',
        'updates': int(model._n_updates), 'learning_rate': float(model.lr_schedule(model._current_progress_remaining)),
        'epsilon': float(model.exploration_rate) if policy.algorithm == 'dqn' else None,
        'entropy_coefficient': float(model.ent_coef) if policy.algorithm == 'ppo' else None,
        'optimizer_steps': sorted({int(state['step'].item()) for state in optimizer.state.values() if 'step' in state})}


class UpdateRecorder:
    """Observe real clipped gradients/optimizer steps, with no update replacement."""
    def __init__(self, policy, path, action_count, opener=None):
        self.policy, self.steps, self.action_count = policy, 0, action_count
        self.file = opener(path) if opener else gzip.open(path, 'xt', encoding='utf-8')
        self.optimizer = policy.model.policy.optimizer
        self.original = self.optimizer.step
        self.optimizer.step = self.step

    def step(self, *args, **kwargs):
        params = list(self.policy.model.policy.parameters())
        old = [p.detach().clone() for p in params]
        gradients = [p.grad.detach() for p in params if p.grad is not None]
        finite = all(bool(torch.isfinite(g).all()) for g in gradients)
        norm = float(torch.sqrt(sum(torch.sum(g*g) for g in gradients)).item()) if gradients else 0.
        if not finite:
            raise FloatingPointError('Nonfinite gradient before optimizer update')
        result = self.original(*args, **kwargs)
        delta = float(torch.sqrt(sum(torch.sum((p.detach()-o)**2) for p, o in zip(params, old))).item())
        if not all(bool(torch.isfinite(p).all()) for p in params) or not np.isfinite(delta):
            raise FloatingPointError('Nonfinite optimizer update')
        self.steps += 1
        self.file.write(json.dumps({'optimizer_step': self.steps, 'training_actions': self.action_count(),
            'gradient_norm_after_library_clipping': norm, 'parameter_update_norm': delta,
            'learning_rate': self.optimizer.param_groups[0]['lr'], 'finite': True})+'\n')
        return result

    def close(self):
        self.optimizer.step = self.original
        self.file.close()


class LinearUpdateRecorder:
    def __init__(self,policy,path,feature_version,opener=None):
        self.policy,self.feature_version,self.steps=policy,feature_version,0
        self.original=policy.update
        self.file=opener(path) if opener else gzip.open(path,'xt',encoding='utf-8')
        policy.update=self.update
    def update(self,obs,action,reward,nxt,terminated,alpha,gamma,next_action=None):
        phi=encode_scaled(obs,self.feature_version); old=self.policy.weights.copy()
        target=float(reward)
        if not terminated:
            values=self.policy.scores(nxt)
            target+=gamma*(values[next_action] if self.policy.algorithm=='sarsa' else values.max())
        error=float(target-self.policy.scores(obs)[action])
        result=self.original(obs,action,reward,nxt,terminated,alpha,gamma,next_action)
        self.steps+=1
        self.file.write(json.dumps({'update':self.steps,'target':float(target),'td_error':error,
            'feature_norm':float(np.linalg.norm(phi)),
            'update_norm':float(np.linalg.norm(self.policy.weights-old)),
            'next_behavior_action':next_action,'terminated':bool(terminated),'finite':True})+'\n')
        return result
    def close(self):
        self.policy.update=self.original; self.file.close()


def bellman_checks():
    """Worked synthetic targets compared with the real unchanged linear update."""
    obs,_=PartialDynamicEnv(cases('A')).reset(seed=0)
    rows=[]
    for algorithm in ('q-learning','sarsa'):
        for outcome,reward,terminal in [('nonterminal',-.01,False),('goal',9.99,True),
                                        ('collision',-5.01,True),('external_truncation',-.01,False)]:
            p=LinearPolicy(algorithm,configuration()); p.weights[1,0]=2.; p.weights[3,0]=3.
            bootstrap=2. if algorithm=='sarsa' else 3.
            target=reward if terminal else reward+.99*bootstrap
            p.update(obs,2,reward,obs,terminal,1.,.99,1 if algorithm=='sarsa' and not terminal else None)
            actual=float(p.scores(obs)[2])
            if not np.isclose(actual,target,atol=1e-12): raise ValueError('Actual Bellman update failed')
            rows.append({'algorithm':algorithm,'case':outcome,'reward':reward,'terminated':terminal,
                         'sampled_next_action':1 if algorithm=='sarsa' and not terminal else None,
                         'bootstrap_value':None if terminal else bootstrap,'target':target,'actual_updated_score':actual})
        p=LinearPolicy(algorithm,configuration())
        p.update(obs,3,9.99,obs,True,1.,.99)
        target=-.01+.99*9.99
        p.update(obs,2,-.01,obs,False,1.,.99,3 if algorithm=='sarsa' else None)
        if not np.isclose(p.scores(obs)[2],target,atol=1e-12): raise ValueError('Two-step target failed')
        rows.append({'algorithm':algorithm,'case':'two-step-backup-to-goal',
                     'formula':'-0.01 + 0.99 * 9.99','target':target,'actual_updated_score':float(p.scores(obs)[2])})
    return rows


def two_step_checks():
    """Real backward TD updates on three linearly independent synthetic inputs."""
    observations=[np.zeros(409,dtype=np.float32) for _ in range(3)]
    for o in observations: o[:4]=np.array([5,5,5,6],dtype=np.float32)/9
    observations[1][8]=-1.; observations[2][9]=-1.
    matrix=np.stack([encode_features(o) for o in observations])
    rows=[]
    for algorithm in ('q-learning','sarsa'):
        for outcome,reward,terminal in [('nonterminal',-.01,False),('goal',9.99,True),
                                        ('collision',-5.01,True),('external_truncation',-.01,False)]:
            p=LinearPolicy(algorithm,configuration())
            p.weights[4]=np.linalg.lstsq(matrix,np.array([-100.,-100.,3.]),rcond=None)[0]
            target2=reward if terminal else reward+.99*3.
            p.update(observations[1],3,reward,observations[2],terminal,1.,.99,
                     4 if algorithm=='sarsa' and not terminal else None)
            actual2=float(p.scores(observations[1])[3])
            bootstrap=target2 if algorithm=='sarsa' else max(0.,target2)
            target1=-.01+.99*bootstrap
            p.update(observations[0],2,-.01,observations[1],False,1.,.99,
                     3 if algorithm=='sarsa' else None)
            actual1=float(p.scores(observations[0])[2])
            if not np.allclose([actual2,actual1],[target2,target1],atol=1e-10,rtol=0):
                raise ValueError('Two-step backward TD arithmetic failed')
            rows.append({'algorithm':algorithm,'second_transition':outcome,
                'second_target':target2,'actual_second_updated_score':actual2,
                'first_target':target1,'actual_first_updated_score':actual1,
                'note':'Q-learning can choose the zero-valued alternative after a collision backup; SARSA uses its declared action 3. External truncation bootstraps.'})
    return rows


def continuity_state(policy, trainer, env):
    """Full trusted state for detecting assessment contamination, not learner input."""
    result = {'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(), 'env_rng': deepcopy(env.unwrapped.np_random.bit_generator.state),
        'action_rng': deepcopy(env.action_space.np_random.bit_generator.state),
        'parameters': policy.state_fingerprint()}
    if trainer:
        result.update(linear_rng=deepcopy(trainer.rng.bit_generator.state),
            observation=trainer.observation, action=trainer.action, updates=trainer.updates,
            used_epsilon=trainer.used_epsilon)
    else:
        model = policy.model
        result.update(optimizer=model.policy.optimizer.state_dict(),
            timesteps=model.num_timesteps, updates=model._n_updates,
            last_observation=model._last_obs, last_episode_starts=model._last_episode_starts,
            replay=model.replay_buffer if policy.algorithm == 'dqn' else model.rollout_buffer)
    return result


def continuity_hash(state):
    """Use stable tensor/array content, rather than pickle storage object IDs."""
    h = hashlib.sha256()
    def feed(v):
        if isinstance(v, torch.Tensor):
            feed(v.detach().cpu().numpy())
        elif isinstance(v, np.ndarray):
            h.update(str((v.shape, v.dtype)).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=str): h.update(str(key).encode()); feed(v[key])
        elif isinstance(v, (list, tuple)):
            for item in v: feed(item)
        elif hasattr(v, '__dict__'):
            # Replay/rollout buffers: exclude only spaces/device metadata.
            feed({k: x for k, x in vars(v).items() if isinstance(x, (np.ndarray, int, float, bool))})
        else:
            h.update(repr(v).encode())
    feed(state)
    return h.hexdigest()


def save_continuation(path, state):
    with gzip.open(path, 'xb', compresslevel=3) as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def replay_linear_updates(records, algorithm, settings, checkpoint_weights, output):
    """Reconstruct old linear optimization using saved transitions; no env.step."""
    policy = LinearPolicy(algorithm, configuration(), settings['primary_seed'])
    rng = np.random.default_rng(settings['primary_seed'])
    count, action, need_reset, td, updates, norms, wrong, waits, matches = 0, None, True, [], [], [], 0, 0, []
    with gzip.open(output, 'xt', encoding='utf-8') as file:
        for record in records:
            if record['kind'] == 'reset':
                if not need_reset: raise ValueError('Unexpected old training reset')
                obs = np.asarray(record['observation'], dtype=np.float32)
                action = policy.explore(obs, epsilon_at(count, settings), rng); need_reset = False
                continue
            obs = np.asarray(record['observation'], dtype=np.float32)
            nxt = np.asarray(record['next_observation'], dtype=np.float32)
            if action != record['action']: raise ValueError('Historical behavior/RNG reconstruction mismatch')
            phi, values = encode_features(obs), policy.scores(obs)
            chosen = int(np.argmax(values)); waits += chosen == 4
            pos = np.rint(obs[:2]*9).astype(int); attempted = pos+MOVES[chosen]
            wrong += chosen != 4 and (np.any(attempted < 0) or np.any(attempted >= 10))
            count += 1
            next_action = policy.explore(nxt, epsilon_at(count, settings), rng) if algorithm == 'sarsa' and not record['terminated'] else None
            target = record['reward']
            if not record['terminated']:
                q = policy.scores(nxt)
                target += settings['model']['discount']*(q[next_action] if algorithm == 'sarsa' else q.max())
            error = float(target-values[action]); before = policy.weights.copy()
            comp = components(policy, obs)
            policy.update(obs, action, record['reward'], nxt, record['terminated'],
                settings['model']['linear_learning_rate'], settings['model']['discount'], next_action)
            magnitude = float(np.linalg.norm(policy.weights-before))
            td.append(error); updates.append(magnitude); norms.append(float(np.linalg.norm(phi)))
            file.write(json.dumps({'step': count, 'action': action, 'reward': record['reward'],
                'target': float(target), 'td_error': error, 'update_norm': magnitude,
                'feature_norm': norms[-1], 'weighted_components_before_update': comp,
                'greedy_margin': float(np.sort(values)[-1]-np.sort(values)[-2])})+'\n')
            if count in checkpoint_weights:
                if not np.array_equal(policy.weights, checkpoint_weights[count]):
                    raise ValueError('Historical update reconstruction differs from archived weights')
                matches.append(count)
            if record['terminated'] or record['truncated']:
                need_reset = True; action = None
            else:
                action = next_action if algorithm == 'sarsa' else policy.explore(nxt, epsilon_at(count, settings), rng)
    return {'actions': count, 'checkpoint_weights_exact': matches, 'td_errors': distribution(td),
        'update_norms': distribution(updates), 'feature_norms': distribution(norms),
        'greedy_wait_count_over_training_states': int(waits), 'greedy_out_of_bounds_count': int(wrong),
        'method': 'Transition/RNG arithmetic replay, zero new environment or training interactions'}


def historical_diagnosis(output, root=ROOT):
    """Read-only Phase 5A inputs. Write only a fresh Phase 5B preflight directory."""
    output.mkdir(parents=True, exist_ok=False)
    old = root/'artifacts/partial-dynamic-navigation-v1/feasibility-pilot-v1'
    pins = read_json(old/'delivery-manifest.json')
    for section, base in [('artifact_files', old), ('active_files', root)]:
        for name, value in pins[section].items():
            if sha(base/name) != value: raise ValueError(f'Phase 5A historical file changed: {name}')
    protocol = read_json(old/'phase5a_pilot_protocol.json')
    if sha(old/'phase5a_pilot_protocol.json') != read_json(old/'protocol-seal.json')['protocol_sha256']:
        raise ValueError('Phase 5A protocol seal mismatch')
    for name, value in protocol['inputs']['code'].items():
        if sha(root/'src/robot_navigation'/name) != value:
            raise ValueError('Recorded baseline source changed')
    summaries = {}
    for algorithm in ('q-learning', 'sarsa', 'dqn', 'ppo'):
        records = read_records(old/algorithm/'training-actions.jsonl.gz')
        steps = [r for r in records if r['kind'] == 'step']
        if len(steps) != 10000: raise ValueError('Missing historical training evidence')
        weights, probes, examples = {}, [], []
        schedules = [s.get('behavior_context', {}) for s in steps]
        for name, count in [('untrained-00000', 0), ('A-02000', 2000), ('A-06000', 6000), ('A-10000', 10000)]:
            folder = old/algorithm/name
            with preserve_random_state():
                policy = load_policy(folder/('policy.npz' if algorithm in ('q-learning','sarsa') else 'policy.zip'), configuration(root))
                if hasattr(policy, 'weights'): weights[count] = policy.weights.copy()
                evaluated = read_records(folder/'evaluation.jsonl.gz')
                for record in evaluated:
                    obs = record['initial_observation']
                    probes.append({'checkpoint': count, 'case_id': record['scenario']['id'],
                                   **decision_diagnostic(policy, obs)})
                if count == 10000:
                    for outcome in ('success', 'collision', 'timeout'):
                        selected = next((r for r in evaluated if r['metrics']['outcome'] == outcome), None)
                        if selected:
                            examples.append({'outcome': outcome, 'source': str((folder/'evaluation.jsonl.gz').relative_to(root)),
                                'case_id': selected['scenario']['id'], 'actions': len(selected['transitions']),
                                'first_decisions': [decision_diagnostic(policy, t['observation']) for t in selected['transitions'][:6]],
                                'terminal_decision': decision_diagnostic(policy, selected['transitions'][-1]['observation']),
                                'note': 'Pre-action observation/values only; outcome is trusted evaluator output.'})
                    numerical = optimizer_state(policy)
                    if (not numerical['parameters_finite']
                            or numerical.get('optimizer_tensors_finite') is False):
                        raise FloatingPointError('Historical numerical state is nonfinite')
        stats = feature_statistics([s['observation'] for s in steps])
        write_json(output/f'{algorithm}-training-feature-statistics.json', stats)
        write_json(output/f'{algorithm}-initial-decisions.json', probes)
        write_json(output/f'{algorithm}-trajectory-examples.json', examples)
        result = {'training_actions_read': len(steps), 'training_goals': sum(s['reward'] > 0 for s in steps),
            'reset_seed_counts': dict(Counter(str(r['seed']) for r in records if r['kind']=='reset')),
            'behavior_context_distribution': dict(Counter(json.dumps(c,sort_keys=True) for c in schedules)),
            'final_numerical_state': numerical,
            'feature_norm': stats['feature_norm'], 'relative_goal_energy_fraction': stats['relative_goal_energy_fraction']}
        if algorithm in ('q-learning', 'sarsa'):
            result['exact_update_reconstruction'] = replay_linear_updates(records, algorithm, protocol['settings'], weights,
                output/f'{algorithm}-historical-updates.jsonl.gz')
        else:
            run = read_json(old/algorithm/'run-summary.json')
            result['recorded_model_diagnostics'] = [c['model'] for c in run['checkpoints']]
            result['limitation'] = 'Historical checkpoints omit live gradients and replay/rollout buffers; read saved optimizer moments and the original recorded buffer diagnostics. No gradients are invented.'
        # If a deterministic evaluation has no success, retain a real training
        # success, explicitly exploratory and not attributed to final weights.
        successful_episode = next(s['episode'] for s in steps if s['reward'] > 0)
        success_steps=[s for s in steps if s['episode']==successful_episode]
        selected=sorted({0,1,len(success_steps)-1}&set(range(len(success_steps))))
        result['exploratory_training_success_example'] = {
            'episode':successful_episode,'source':str((old/algorithm/'training-actions.jsonl.gz').relative_to(root)),
            'scope':'Actual historical exploratory training actions, not final deterministic policy decisions. Hidden snapshots excluded.',
            'pre_action_examples':[{'index':i,'observation_before_action':success_steps[i]['observation'],
                'chosen_action':success_steps[i]['action'],'behavior_context':success_steps[i]['behavior_context']}
                for i in selected],
            'evaluator_only_after_action':{'outcome':success_steps[-1]['info']['outcome'],
                                         'reward':success_steps[-1]['reward'],'actions':len(success_steps)}}
        summaries[algorithm] = result
    witness=[]
    for case in cases('A'):
        obs,_ = PartialDynamicEnv([case], configuration(root)).reset(seed=0)
        delta=obs[2:4]-obs[:2]; scores=[-delta[0],delta[0],-delta[1],delta[1],0.]
        action=int(np.argmax(scores)); target=tuple(np.rint(obs[:2]*9).astype(int)+MOVES[action])
        if target != tuple(case['goal']): raise ValueError('Analytic capacity witness failed')
        witness.append({'case_id':case['id'],'relative_goal':delta.tolist(),'argmax':action,'scores':[float(v) for v in scores]})
    result={'status':'passed','phase5a_protocol_sha256':sha(old/'phase5a_pilot_protocol.json'),
        'diagnostic_source_sha256':sha(root/'src/robot_navigation/partial_phase5b_diagnostics.py'),
        'worked_bellman_targets':bellman_checks(),
        'worked_two_step_targets':two_step_checks(),
        'historical_pins_checked':len(pins['artifact_files'])+len(pins['active_files']),
        'new_training_actions':0,'new_policy_environment_actions':0,'systems':summaries,
        'capacity_witness':{'label':'capacity witness, no learning, no executed policy','cases':witness},
        'tie_policy':'Linear greedy first argmax, exploratory uniform among equal maxima; SB3 deterministic categorical/Q argmax in action order.',
        'source_extension_note':'Verify the recorded source list; new additive modules do not rewrite the archived Phase 5A provenance guard.'}
    write_json(output/'zero-learning-audit.json',result)
    write_json(output/'hashes.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    return result
