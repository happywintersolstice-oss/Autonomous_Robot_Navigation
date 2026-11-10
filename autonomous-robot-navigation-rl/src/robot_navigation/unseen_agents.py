"""Map-aware policy adapters. Legacy fixed-map checkpoint loading stays strict."""
import hashlib
from pathlib import Path

import numpy as np

from .agents import QLearning, SARSA
from .core import read_json, write_json
from .unseen_benchmark import DIMENSIONS, OBSERVATION, map_observation, table_key


class MapTablePolicy:
    def __init__(self, benchmark, algorithm, policy_id):
        if algorithm not in ('q-learning','sarsa'):
            raise ValueError('A tabular family is required')
        self.benchmark, self.algorithm, self.policy_id = benchmark, algorithm, policy_id
        self.layouts = benchmark.layouts('train')
        implementation = SARSA if algorithm == 'sarsa' else QLearning
        self.members = {}
        for layout in self.layouts:
            task = benchmark.task(layout)
            key, _ = table_key(map_observation([0,0,0,0], task.grid))
            self.members[key] = implementation(task, policy_id)

    def scores(self, observation):
        key, coordinates = table_key(observation)
        member = self.members.get(key)
        # Reading a new layout must not create a table or change learned state.
        return member.scores(coordinates).copy() if member is not None else np.zeros(4)

    def act(self, observation):
        return int(np.argmax(self.scores(observation)))

    def explore(self, observation, epsilon, rng):
        key, coordinates = table_key(observation)
        return self.members[key].explore(coordinates, epsilon, rng)

    def update(self, observation, action, reward, next_observation, terminated, alpha, gamma,
               next_action=None):
        key, coordinates = table_key(observation)
        next_key, next_coordinates = table_key(next_observation)
        if key != next_key:
            raise ValueError('An update cannot cross a layout reset')
        member = self.members[key]
        args = (coordinates, action, reward, next_coordinates, terminated, alpha, gamma)
        if self.algorithm == 'sarsa':
            member.update(*args, next_action=next_action)
        else:
            member.update(*args)

    def state_fingerprint(self):
        hasher = hashlib.sha256()
        for key, member in self.members.items():
            hasher.update(key)
            hasher.update(member.q.tobytes())
        return hasher.hexdigest()

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, q=np.stack([member.q for member in self.members.values()]))
        write_json(path.with_suffix('.json'), {
            **metadata, 'signature':self.benchmark.signature, 'algorithm':self.algorithm,
            'policy_id':self.policy_id, 'encoding':OBSERVATION,
            'layout_ids':[m['map_id'] for m in self.layouts],
            'kind':'inference-only; exact training resume unsupported',
            'unseen_layout_rule':'unvisited exact states have zero Q values; Up wins deterministic ties',
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})


def _save_neural(policy, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    policy.model.save(path)
    if getattr(policy,'feature_version',None):
        from .shared_features import FEATURE_COUNT
        metadata = {**metadata,'feature_version':policy.feature_version,'feature_count':FEATURE_COUNT}
    write_json(path.with_suffix('.json'), {
        **metadata, 'signature':policy.task.signature, 'algorithm':policy.algorithm,
        'policy_id':policy.policy_id, 'encoding':OBSERVATION,
        'kind':'inference-only; buffers/RNG/environment not saved; exact resume unsupported',
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})


def neural_policy(benchmark, algorithm, model, policy_id):
    # Import Torch only for the neural families and preserve the existing
    # exhaustive parameter/buffer fingerprints from the fixed-map adapters.
    import torch
    from .dqn import DQNPolicy
    from .ppo import PPOPolicy

    class MapDQNPolicy(DQNPolicy):
        def scores(self, observation):
            table_key(observation)
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.model.device).unsqueeze(0)
            with torch.inference_mode():
                return self.model.q_net(tensor)[0].cpu().numpy().copy()

        def save(self, path, metadata):
            _save_neural(self, path, metadata)

    class MapPPOPolicy(PPOPolicy):
        def decision(self, observation):
            table_key(observation)
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.model.device).unsqueeze(0)
            with torch.inference_mode():
                probabilities = self.model.policy.get_distribution(tensor).distribution.probs[0].cpu().numpy().copy()
                value = float(self.model.policy.predict_values(tensor).item())
            return {'action':int(np.argmax(probabilities)), 'action_probabilities':probabilities.tolist(),
                    'state_value':value}

        def save(self, path, metadata):
            _save_neural(self, path, metadata)

    implementation = {'dqn':MapDQNPolicy, 'ppo':MapPPOPolicy}[algorithm]
    policy = implementation(benchmark, model, policy_id)
    policy.encoding = OBSERVATION
    policy.feature_version = getattr(model,'navigation_feature_version',None)
    return policy


def build_neural(benchmark, env, algorithm, config, seed, policy_id, *, features_extractor=None, feature_version=None):
    import torch
    from stable_baselines3 import DQN, PPO
    torch.set_num_threads(1)
    common = dict(seed=seed, device='cpu', learning_rate=config['learning_rate'],
                  gamma=config['discount'], verbose=0)
    extractor_options = {'features_extractor_class':features_extractor} if features_extractor is not None else {}
    if algorithm == 'dqn':
        model = DQN('MlpPolicy', env, **common, buffer_size=config['buffer_size'],
                    learning_starts=config['learning_starts'], batch_size=config['batch_size'],
                    train_freq=config['train_freq'], gradient_steps=config['gradient_steps'],
                    target_update_interval=config['target_update_interval'],
                    exploration_initial_eps=config['epsilon_start'], exploration_final_eps=config['epsilon_end'],
                    exploration_fraction=config['epsilon_decay_fraction'],
                    policy_kwargs={'net_arch':config['net_arch'],**extractor_options})
    elif algorithm == 'ppo':
        model = PPO('MlpPolicy', env, **common, n_steps=config['n_steps'], batch_size=config['batch_size'],
                    n_epochs=config['n_epochs'], gae_lambda=config['gae_lambda'], clip_range=config['clip_range'],
                    ent_coef=config['ent_coef'], vf_coef=config['vf_coef'], max_grad_norm=config['max_grad_norm'],
                    policy_kwargs={'net_arch':{'pi':config['net_arch'],'vf':config['net_arch']},**extractor_options})
    else:
        raise ValueError('Unsupported neural family')
    if feature_version is not None:
        model.navigation_feature_version = feature_version
    return neural_policy(benchmark, algorithm, model, policy_id)


def load_map_policy(path, benchmark):
    path = Path(path)
    metadata = read_json(path.with_suffix('.json'))
    if metadata.get('signature') != benchmark.signature or metadata.get('encoding') != OBSERVATION:
        raise ValueError('Map-aware checkpoint/benchmark/input mismatch')
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
        raise ValueError('Map-aware checkpoint checksum mismatch')
    algorithm = metadata['algorithm']
    feature_version = metadata.get('feature_version')
    from .shared_features import FEATURE_VERSION, FEATURE_COUNT
    if feature_version not in (None,FEATURE_VERSION):
        raise ValueError('Unsupported map-aware feature version')
    if feature_version and metadata.get('feature_count') != FEATURE_COUNT:
        raise ValueError('Map-aware feature dimension mismatch')
    if feature_version and algorithm in ('q-learning','sarsa'):
        from .shared_agents import LinearMapPolicy
        policy = LinearMapPolicy(benchmark,algorithm,metadata['policy_id'])
        with np.load(path,allow_pickle=False) as data:
            weights = data['weights']
            if weights.shape != policy.weights.shape or not np.isfinite(weights).all():
                raise ValueError('Invalid shared linear weights')
            policy.weights[:] = weights
        return policy
    if algorithm in ('q-learning','sarsa'):
        policy = MapTablePolicy(benchmark, algorithm, metadata['policy_id'])
        if metadata['layout_ids'] != [m['map_id'] for m in policy.layouts]:
            raise ValueError('Tabular training layout identity mismatch')
        with np.load(path, allow_pickle=False) as data:
            tables = data['q']
            expected = (len(policy.members),10,10,10,10,4)
            if tables.shape != expected or not np.isfinite(tables).all():
                raise ValueError('Invalid map-aware table shape or values')
            for member, values in zip(policy.members.values(), tables, strict=True):
                member.q[:] = values
        return policy
    if algorithm not in ('dqn','ppo'):
        raise ValueError('Unsupported map-aware checkpoint algorithm')
    import torch
    from stable_baselines3 import DQN, PPO
    torch.set_num_threads(1)
    model = {'dqn':DQN,'ppo':PPO}[algorithm].load(path, device='cpu')
    if getattr(model,'navigation_feature_version',None) != feature_version:
        raise ValueError('Neural checkpoint feature version mismatch')
    if model.observation_space.shape != (DIMENSIONS,) or model.action_space.n != 4:
        raise ValueError('Map-aware neural input/action shape mismatch')
    if any(not torch.isfinite(value).all() for value in model.policy.state_dict().values()):
        raise ValueError('Nonfinite map-aware neural parameters')
    return neural_policy(benchmark, algorithm, model, metadata['policy_id'])


def map_adapter(policy, grid):
    """Supply the known map while retaining compact coordinate replay records."""
    class Adapter:
        policy_id = policy.policy_id

        def act(self, coordinates):
            return policy.act(map_observation(coordinates, grid))

    class ScoreAdapter(Adapter):
        def action_and_scores(self, coordinates):
            scores = policy.scores(map_observation(coordinates, grid))
            return int(np.argmax(scores)), scores

    class ProbabilityAdapter(Adapter):
        def decision(self, coordinates):
            return policy.decision(map_observation(coordinates, grid))

    return ProbabilityAdapter() if policy.algorithm == 'ppo' else ScoreAdapter()
