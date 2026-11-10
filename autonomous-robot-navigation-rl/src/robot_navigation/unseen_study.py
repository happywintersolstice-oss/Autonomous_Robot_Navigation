"""Bounded four-family development pilot. Final-map policy access is absent."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
from time import perf_counter

import gymnasium as gym
import numpy as np

from .agents import BFSPolicy, RandomPolicy
from .core import ROOT, NavigationEnv, read_json, write_json
from .episodes import aggregate, episode_metrics, rollout, validate_replay
from .part_a import checkpoint_rank
from .release import sha
from .training import epsilon_at
from .unseen_agents import MapTablePolicy, build_neural, load_map_policy, map_adapter
from .unseen_benchmark import MapBenchmark, MapNavigationEnv, OBSERVATION, map_observation, prepare
from .workflow import source_hash

STUDY = 'unseen-map-pilot-v1'


def episode_context(episode, benchmark, layout, split):
    episode.update(benchmark_signature=benchmark.signature, layout_id=layout['map_id'],
                   layout_split=split, policy_observation_version=OBSERVATION)
    return episode


class TrainingLedger(gym.Wrapper):
    def __init__(self, env, directory, writer, algorithm, seed, windows, feature_version=None):
        super().__init__(env)
        self.directory, self.writer, self.algorithm, self.seed = directory, writer, algorithm, seed
        self.windows, self.window_index, self.traces = list(windows), 0, []
        self.interactions, self.episodes = 0, 0
        self.map_steps, self.outcomes = Counter(), Counter()
        self.record, self.capture = None, False
        self.feature_version = feature_version

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        raw_env = self.env.env
        self.capture = (self.window_index < len(self.windows)
                        and self.interactions >= self.windows[self.window_index])
        self.record = {'signature':raw_env.task.signature, 'scenario':raw_env.scenario, 'seed':self.seed,
                       'policy_id':f'{self.algorithm}/{self.directory.name}/actual-training', 'mode':'training',
                       'initial_observation':raw_env._observation().tolist(), 'transitions':[],
                       'training_start_interaction':self.interactions}
        episode_context(self.record, self.env.benchmark, self.env.layout, 'train')
        if self.feature_version:
            self.record['policy_feature_version'] = self.feature_version
        return observation, info

    def step(self, action):
        before = self.env.env._observation().tolist()
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.interactions += 1
        self.map_steps[self.env.layout['map_id']] += 1
        self.record['transitions'].append({'observation':before, 'action':int(action), 'reward':reward,
            'next_observation':self.env.env._observation().tolist(), 'terminated':terminated,
            'truncated':truncated, 'info':dict(info), 'scores':None})
        if terminated or truncated:
            self.episodes += 1
            metrics = episode_metrics(self.record)
            self.outcomes[metrics['outcome']] += 1
            self.writer({'episode':self.episodes, 'layout_id':self.env.layout['map_id'],
                         'scenario_id':self.record['scenario']['id'],
                         'start_interaction':self.record['training_start_interaction'],
                         'end_interaction':self.interactions, 'steps':metrics['steps'],
                         'return':metrics['return'], 'outcome':metrics['outcome']})
            if self.capture:
                self._save_trace(True)
        return observation, reward, terminated, truncated, info

    def _save_trace(self, complete):
        self.record.update(complete=complete, training_end_interaction=self.interactions,
                           recording_window_start=self.windows[self.window_index])
        if complete:
            self.record['metrics'] = episode_metrics(self.record)
        else:
            self.record['cutoff_reason'] = 'budget_cutoff'
        filename = f'training_traces/window-{self.window_index:02d}.json'
        write_json(self.directory/filename, self.record)
        self.traces.append({'path':filename, 'sha256':sha(self.directory/filename),
                            'layout_id':self.env.layout['map_id'],
                            'window_start':self.windows[self.window_index], 'complete':complete})
        self.window_index += 1
        self.capture = False

    def finish(self):
        if self.capture and self.record['transitions']:
            self._save_trace(False)
        ended = self.record['transitions'][-1] if self.record['transitions'] else None
        incomplete = bool(ended and not (ended['terminated'] or ended['truncated']))
        return {'training_interactions':self.interactions, 'completed_episodes':self.episodes,
                'outcomes':dict(self.outcomes), 'map_interactions':dict(self.map_steps),
                'cutoff_steps':len(self.record['transitions']) if incomplete else 0,
                'training_traces':self.traces}


def evaluate(benchmark, cases, policy, seed, split='validation', output=None, *, final=False, deadline=None):
    if final and (split != 'test' or not getattr(benchmark,'final_authorized',False)):
        raise ValueError('Final evaluation requires a sealed final benchmark')
    if split not in ('train','validation') and not final:
        raise ValueError('Final layout evaluation is sealed in this milestone')
    declared = {(layout['map_id'], scenario['id']):(layout,scenario) for layout, scenario in benchmark.cases(split)}
    if not cases or any(declared.get((layout['map_id'], scenario['id'])) != (layout,scenario)
                        for layout, scenario in cases):
        raise ValueError('Evaluation requires declared development cases')
    fingerprint = policy.state_fingerprint() if hasattr(policy, 'state_fingerprint') else None
    metrics, examples, groups = [], {}, defaultdict(list)
    started = perf_counter()
    file = gzip.open(output, 'xt', encoding='utf-8') if output else None
    reference = RandomPolicy(seed) if policy == 'random' else None
    try:
        for layout, scenario in cases:
            if deadline is not None and perf_counter()>deadline:
                raise TimeoutError('Evaluation exceeded its registered wall limit')
            task = benchmark.task(layout)
            current = BFSPolicy(task.grid) if policy == 'bfs' else reference
            if current is None:
                current = map_adapter(policy, task.grid)
            mode = 'final evaluation' if final else 'validation'
            episode = rollout(NavigationEnv(task, [scenario]), current, scenario, seed, mode, allow_final=final)
            episode_context(episode, benchmark, layout, split)
            if getattr(policy,'feature_version',None):
                episode['policy_feature_version'] = policy.feature_version
            if final:
                episode['release_digest'] = benchmark.release_digest
            validate_replay(task, episode, allow_final=final)
            row = {'layout_id':layout['map_id'], 'scenario_id':scenario['id'], **episode['metrics']}
            metrics.append(row)
            groups[layout['map_id']].append(row)
            key = (layout['map_id'], row['outcome'])
            if key not in examples:
                examples[key] = episode
            if file:
                file.write(json.dumps(episode, allow_nan=False)+'\n')
    finally:
        if file:
            file.close()
    if fingerprint is not None and policy.state_fingerprint() != fingerprint:
        raise AssertionError('Map-aware evaluation changed learned state')
    per_map = {name:aggregate(rows) for name, rows in groups.items()}
    summary = {**aggregate(metrics), 'layout_count':len(per_map), 'by_map':per_map,
               'map_mean':{metric:float(np.mean([m[metric] for m in per_map.values()]))
                           for metric in ('success_rate','collision_rate','timeout_rate','spl','mean_return')},
               'evaluation_seconds':perf_counter()-started, 'split':split,
               'aggregation':'equal-support routes; equal-weight layout means; one trained seed',
               'inference_unchanged':fingerprint is not None, 'final_test_evaluated':final}
    return summary, metrics, list(examples.values())


def _tabular_training(ledger, policy, config, monitor):
    rng = np.random.default_rng(config['seed'])
    observation, _ = ledger.reset(seed=config['seed'])
    action = policy.explore(observation, epsilon_at(config, 0), rng)
    started = perf_counter()
    for index in range(config['interactions']):
        if perf_counter()-started > config['run_wall_seconds']:
            raise TimeoutError('Tabular pilot exceeded its wall limit')
        next_observation, reward, terminated, truncated, _ = ledger.step(action)
        next_action = None
        if policy.algorithm == 'sarsa' and not terminated:
            next_action = policy.explore(next_observation, epsilon_at(config, index+1), rng)
        policy.update(observation, action, reward, next_observation, terminated,
                      config['learning_rate'], config['discount'], next_action=next_action)
        if (index+1) % config['monitor_every'] == 0:
            monitor(policy, index+1)
        if index+1 == config['interactions']:
            break
        if terminated or truncated:
            observation, _ = ledger.reset()
            action = policy.explore(observation, epsilon_at(config, index+1), rng)
        else:
            observation = next_observation
            action = next_action if policy.algorithm == 'sarsa' else policy.explore(
                observation, epsilon_at(config, index+1), rng)


def train_run(benchmark, study, algorithm, source_digest, settings=None):
    settings = benchmark.config['pilot'] if settings is None else settings
    algorithm_config = settings['tabular' if algorithm in ('q-learning','sarsa') else algorithm]
    config = {**algorithm_config, 'seed':settings['seed'], 'algorithm':algorithm,
              'interactions':settings['interactions'], 'monitor_every':settings['monitor_every'],
              'run_wall_seconds':settings['run_wall_seconds']}
    if settings.get('representation'):
        config['representation'] = settings['representation']
    run_id = settings.get('run_id',f'm10-map-aware-{algorithm}-seed{settings["seed"]}')
    if Path(run_id).name != run_id or run_id in ('.','..'):
        raise ValueError('Run ID must be one directory name')
    directory = study/'runs'/run_id
    if directory.exists():
        existing = read_json(directory/'manifest.json')
        if (existing['status'] != 'complete' or existing['configuration'] != config
                or existing['signature'] != benchmark.signature or existing['source_hash'] != source_digest):
            raise ValueError(f'Existing run is incomplete or incompatible; retained at {directory}')
        return existing
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {'status':'running', 'run_id':run_id, 'algorithm':algorithm, 'training_seed':settings['seed'],
                'signature':benchmark.signature, 'source_hash':source_digest, 'configuration':config,
                'started_utc':datetime.now(timezone.utc).isoformat(),
                'information':OBSERVATION, 'final_test_evaluated':False,
                'selection_rule':'monitor success, then SPL, then earliest interaction',
                'monitoring_phase':'after preceding parameter update; before further collection',
                'exact_resume_supported':False}
    if settings.get('representation'):
        manifest['representation'] = settings['representation']
    write_json(directory/'manifest.json', manifest)
    monitoring, selected_rank, monitor_seconds = [], None, 0.
    extension = '.npz' if algorithm in ('q-learning','sarsa') else '.zip'
    monitor_cases = benchmark.cases('validation', monitoring=True)

    def monitor(policy, interaction):
        nonlocal selected_rank, monitor_seconds
        before = perf_counter()
        policy.policy_id = f'{algorithm}/{run_id}/step-{interaction}'
        summary, _, _ = evaluate(benchmark, monitor_cases, policy, settings['seed'])
        monitoring.append({'training_interactions':interaction, **summary})
        checkpoint = directory/f'checkpoints/step-{interaction:07d}{extension}'
        policy.save(checkpoint, {'training_interactions':interaction, 'configuration':config})
        rank = checkpoint_rank(summary, interaction)
        if selected_rank is None or rank > selected_rank:
            selected_rank = rank
            selected = directory/f'selected{extension}'
            shutil.copy2(checkpoint, selected)
            shutil.copy2(checkpoint.with_suffix('.json'), selected.with_suffix('.json'))
            manifest.update(selected_interactions=interaction, viewer_checkpoint=selected.name)
        write_json(directory/'monitoring.json', monitoring)
        write_json(directory/'manifest.json', manifest)
        monitor_seconds += perf_counter()-before
        print(f'{run_id}: {interaction:,} steps; monitor success {summary["success_rate"]:.3f}', flush=True)

    run_started = perf_counter()
    ledger = None
    try:
        with (directory/'training.csv').open('w', newline='', encoding='utf-8') as file:
            fields = ['episode','layout_id','scenario_id','start_interaction','end_interaction','steps','return','outcome']
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            ledger = TrainingLedger(MapNavigationEnv(benchmark, benchmark.cases('train')), directory,
                                    writer.writerow, algorithm, settings['seed'], settings['trace_windows'],
                                    settings.get('representation'))
            construction = perf_counter()
            if settings.get('representation'):
                from .shared_agents import build_shared_policy
                policy = build_shared_policy(benchmark,ledger,algorithm,algorithm_config,
                                             settings['seed'],f'{algorithm}/{run_id}')
            elif algorithm in ('q-learning','sarsa'):
                policy = MapTablePolicy(benchmark, algorithm, f'{algorithm}/{run_id}')
            else:
                policy = build_neural(benchmark, ledger, algorithm, algorithm_config,
                                      settings['seed'], f'{algorithm}/{run_id}')
            construction_seconds = perf_counter()-construction
            started = perf_counter()
            if algorithm in ('q-learning','sarsa'):
                _tabular_training(ledger, policy, config, monitor)
            else:
                from stable_baselines3.common.callbacks import BaseCallback

                class Monitor(BaseCallback):
                    def _on_step(self):
                        if perf_counter()-started > settings['run_wall_seconds']:
                            raise TimeoutError('Neural pilot exceeded its wall limit')
                        return True

                    def _on_rollout_start(self):
                        if self.num_timesteps and self.num_timesteps % settings['monitor_every'] == 0:
                            monitor(policy, self.num_timesteps)
                            file.flush()

                    def _on_training_end(self):
                        monitor(policy, self.num_timesteps)

                policy.model.learn(total_timesteps=settings['interactions'], callback=Monitor())
                if policy.model.num_timesteps != settings['interactions']:
                    raise AssertionError('Neural training budget mismatch')
            training_seconds = perf_counter()-started-monitor_seconds
            training = ledger.finish()
        if training['training_interactions'] != settings['interactions']:
            raise AssertionError('Actual training interactions do not match the registered budget')
        selected_path = directory/manifest['viewer_checkpoint']
        selected = load_map_policy(selected_path, benchmark)
        validation, rows, examples = evaluate(benchmark, benchmark.cases('validation'), selected,
            settings['seed'], output=directory/'validation.jsonl.gz')
        familiar, familiar_rows, _ = evaluate(benchmark, benchmark.cases('train', monitoring=True), selected,
            settings['seed'], split='train', output=directory/'familiar.jsonl.gz')
        write_json(directory/'validation.json', validation)
        write_json(directory/'episodes.json', rows)
        write_json(directory/'familiar.json', familiar)
        write_json(directory/'familiar_episodes.json', familiar_rows)
        replays = []
        for index, episode in enumerate(examples):
            filename = f'replays/validation-{index:03d}.json'
            write_json(directory/filename, episode)
            replays.append({'path':filename, 'sha256':sha(directory/filename),
                            'layout_id':episode['layout_id'], 'outcome':episode['metrics']['outcome']})
        probe_layout, probe_scenario = benchmark.cases('validation')[0]
        probe = map_observation([*probe_scenario['start'],*probe_scenario['goal']], benchmark.task(probe_layout).grid)
        latency = perf_counter()
        for _ in range(100):
            selected.act(probe)
        microseconds = (perf_counter()-latency)*1e4
        manifest.update(status='complete', training=training, replays=replays,
            construction_seconds=construction_seconds, training_seconds=training_seconds,
            monitoring_seconds=monitor_seconds, validation_seconds=validation['evaluation_seconds'],
            familiar_probe_seconds=familiar['evaluation_seconds'],
            training_interactions=training['training_interactions'],
            monitoring_interactions=sum(item['interactions'] for item in monitoring),
            validation_interactions=validation['interactions'], familiar_interactions=familiar['interactions'],
            mean_decision_microseconds=microseconds, latency_queries=100,
            checkpoint_bytes=selected_path.stat().st_size,
            total_seconds=perf_counter()-run_started, finished_utc=datetime.now(timezone.utc).isoformat())
        if hasattr(selected,'weights'):
            manifest['optimized_parameter_count'] = int(selected.weights.size)
        elif settings.get('representation'):
            parameters = {id(p):p for group in selected.model.policy.optimizer.param_groups for p in group['params']}
            manifest['optimized_parameter_count'] = sum(p.numel() for p in parameters.values())
        if manifest['total_seconds'] > settings['run_wall_seconds']:
            raise TimeoutError('Pilot run exceeded its total wall limit')
        write_json(directory/'manifest.json', manifest)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                        total_seconds=perf_counter()-run_started)
        manifest['training_interactions_observed'] = ledger.interactions if ledger is not None else 0
        write_json(directory/'manifest.json', manifest)
        raise


def pilot(root=ROOT):
    benchmark = MapBenchmark(root)
    settings = benchmark.config['pilot']
    if settings['study_id'] != STUDY:
        raise ValueError('Unsupported pilot identity')
    if (settings['algorithms'] != ['q-learning','sarsa','dqn','ppo']
            or settings['interactions'] <= 0 or settings['monitor_every'] <= 0
            or settings['interactions'] % settings['monitor_every']
            or settings['interactions'] % settings['dqn']['train_freq']
            or settings['monitor_every'] % settings['dqn']['train_freq']
            or settings['interactions'] % settings['ppo']['n_steps']
            or settings['monitor_every'] % settings['ppo']['n_steps']
            or settings['ppo']['n_steps'] % settings['ppo']['batch_size']):
        raise ValueError('Pilot families/budgets must match exact update and monitoring boundaries')
    study = root/'artifacts/studies'/STUDY
    if not study.exists():
        study.mkdir(parents=True, exist_ok=False)
        for folder in ('src','app','configs'):
            shutil.copytree(root/folder, study/'source'/folder,
                            ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
        write_json(study/'manifest.json', {'status':'running', 'signature':benchmark.signature,
            'protocol_hash':sha(root/'docs/milestone10_protocol.md'), 'settings':settings,
            'source_hash':source_hash(study/'source'), 'python':platform.python_version(),
            'dependencies':{p:version(p) for p in ('numpy','gymnasium','torch','stable-baselines3','streamlit')},
            'final_test_evaluated':False})
        shutil.copy2(root/'docs/milestone10_protocol.md', study/'protocol.md')
    manifest = read_json(study/'manifest.json')
    if (manifest['signature'] != benchmark.signature or manifest['settings'] != settings
            or source_hash(study/'source') != manifest['source_hash']):
        raise ValueError('Pilot source/configuration snapshot mismatch')
    if manifest['status'] in ('complete; audit pending','verified'):
        return read_json(study/'summary.json')
    lock = study/'run.lock'
    try:
        with lock.open('x', encoding='utf-8') as file:
            file.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError('Pilot execution lock exists; another run or an interrupted owner must be checked') from exc
    try:
        started = perf_counter()
        rows = []
        try:
            for algorithm in settings['algorithms']:
                if perf_counter()-started > settings['suite_wall_seconds']:
                    raise TimeoutError('Pilot suite exceeded its wall limit')
                run = train_run(benchmark, study, algorithm, manifest['source_hash'])
                directory = study/'runs'/run['run_id']
                rows.append({'algorithm':algorithm, 'run_id':run['run_id'],
                             'validation':read_json(directory/'validation.json'),
                             'familiar':read_json(directory/'familiar.json'),
                             'training_interactions':run['training_interactions'],
                             'training_seconds':run['training_seconds'], 'total_seconds':run['total_seconds'],
                             'selected_interactions':run['selected_interactions'],
                             'mean_decision_microseconds':run['mean_decision_microseconds'],
                             'checkpoint_bytes':run['checkpoint_bytes']})
                if sum(p.stat().st_size for p in study.rglob('*') if p.is_file()) > settings['artifact_bytes']:
                    raise RuntimeError('Pilot artifacts exceeded the registered storage limit')
            references = {}
            for name in ('bfs','random'):
                summary, _, _ = evaluate(benchmark, benchmark.cases('validation'), name,
                                          settings['seed'], output=study/f'{name}.jsonl.gz')
                references[name] = summary
            summary = {'status':'complete; audit pending', 'signature':benchmark.signature,
                       'scope':'one-seed development pilot on unseen validation layouts', 'systems':rows,
                       'references':references, 'total_training_interactions':sum(r['training_interactions'] for r in rows),
                       'training_seed_count':1, 'final_test_evaluated':False,
                       'tabular_transfer':'unseen occupancy keys retain zero Q values and deterministic Up ties',
                       'main_study_status':'budget/seed freeze and independent final evaluation are next-stage work'}
            write_json(study/'summary.json', summary)
            manifest.update(status='complete; audit pending', elapsed_this_invocation_seconds=perf_counter()-started,
                            artifact_bytes=sum(p.stat().st_size for p in study.rglob('*') if p.is_file()))
            if (manifest['artifact_bytes'] > settings['artifact_bytes']
                    or manifest['elapsed_this_invocation_seconds'] > settings['suite_wall_seconds']):
                raise RuntimeError('Completed pilot exceeded its registered suite resource limit')
            write_json(study/'manifest.json', manifest)
            return summary
        except BaseException as exc:
            manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            write_json(study/'manifest.json', manifest)
            raise

    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Separate unseen-map pilot; final policy access is sealed')
    parser.add_argument('command', choices=('prepare','pilot','verify'))
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare()
    elif args.command == 'pilot':
        summary = pilot()
        result = {key:summary[key] for key in ('status','scope','total_training_interactions','final_test_evaluated')}
        result['systems'] = [{'algorithm':row['algorithm'],
                              'validation_success_rate':row['validation']['success_rate'],
                              'run_id':row['run_id']} for row in summary['systems']]
    else:
        from .unseen_audit import verify
        result = verify()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
