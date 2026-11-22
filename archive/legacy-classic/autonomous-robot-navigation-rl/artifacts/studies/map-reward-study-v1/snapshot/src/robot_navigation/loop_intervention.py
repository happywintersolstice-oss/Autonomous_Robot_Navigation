"""Registered development-only, history-aware SARSA recovery; no retraining."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from .core import ROOT, NavigationEnv, read_json, write_json, digest
from .episodes import rollout, aggregate, validate_replay
from .release import sha
from .unseen_benchmark import MapBenchmark, map_observation
from .unseen_agents import load_map_policy, map_adapter

STUDY = 'loop-recovery-dev-v1'
SETTINGS = {'training_seeds':[12101,12203,12301],'rng_seeds':[17011,17021,17033],
            'margin':.25,'max_recoveries':4,'wall_seconds':180,'artifact_bytes':256*1024**2}


class RecoveryPolicy:
    """All four actions remain eligible; history is explicit controller state."""
    def __init__(self,base,seed,margin=.25,max_recoveries=4):
        self.base,self.rng = base,np.random.default_rng(seed)
        self.margin,self.max_recoveries = margin,max_recoveries
        self.policy_id = f'history-recovery/{base.policy_id}/rng{seed}'
        self.reset_episode()

    def reset_episode(self):
        self.positions,self.recoveries = [],0

    def decision(self,observation):
        values = np.asarray(self.base.scores(observation))
        action = int(np.argmax(values))
        self.positions.append(tuple(np.asarray(observation)[:2]))
        self.positions = self.positions[-5:]
        p = self.positions
        trigger = len(p)==5 and p[0]==p[2]==p[4] and p[1]==p[3] and p[0]!=p[1]
        candidates = [i for i,v in enumerate(values) if i!=action and v>=values.max()-self.margin]
        recovery = bool(trigger and candidates and self.recoveries<self.max_recoveries)
        if recovery:
            action = int(self.rng.choice(candidates))
            self.recoveries += 1
        return {'action':action,'recovery':recovery,'base_scores':values.tolist(),
                'candidates':candidates if trigger else [],'recoveries':self.recoveries,
                'controller':'history-aware near-tie recovery v1'}


def suffix_cycle(episode):
    """Trace-only classification; permits different actions on repeat visits."""
    if episode['metrics']['outcome']!='timeout':
        return False
    states = [tuple(t['observation']) for t in episode['transitions']]
    actions = [t['action'] for t in episode['transitions']]
    n = len(actions)
    for start in range(n):
        for period in range(1,(n-start)//2+1):
            if all(states[k]==states[start+(k-start)%period] and actions[k]==actions[start+(k-start)%period]
                   for k in range(start,n)):
                return True
    return False


def paired_result(base,wrapped):
    if len(base)!=len(wrapped):
        raise ValueError('Paired route support differs')
    changes = Counter()
    for left,right in zip(base,wrapped,strict=True):
        if left['scenario']!=right['scenario'] or left['signature']!=right['signature']:
            raise ValueError('Paired route identity differs')
        changes[left['metrics']['outcome']+'->'+right['metrics']['outcome']] += 1
    return dict(changes)


def summarize(rows):
    if len(rows)!=9 or {(r['training_seed'],r['rng_seed']) for r in rows} != {
            (s,r) for s in SETTINGS['training_seeds'] for r in SETTINGS['rng_seeds']}:
        raise ValueError('Incomplete registered paired matrix')
    mean = lambda side,metric:float(np.mean([r[side][metric] for r in rows]))
    before,after = mean('baseline','success_rate'),mean('wrapper','success_rate')
    return {'scope':'exploratory development layouts only; no independent final claim',
        'baseline_success':before,'wrapper_success':after,'success_change':after-before,
        'baseline_spl':mean('baseline','spl'),'wrapper_spl':mean('wrapper','spl'),
        'wrapper_collision':mean('wrapper','collision_rate'),
        'promotion_passed':after-before>=.05 and mean('wrapper','spl')>=mean('baseline','spl')
                           and mean('wrapper','collision_rate')<=.02,
        'paired_changes':dict(sum((Counter(r['paired_changes']) for r in rows),Counter())),
        'rows':rows,'training_interactions':0,'final_layout_evaluations':0,
        'replication':'3 trained policies; 3 intervention streams per policy; deterministic baseline reused, not 9 independent policies'}


def write_episodes(path,episodes):
    with gzip.open(path,'xt',encoding='utf-8') as file:
        for episode in episodes:
            file.write(json.dumps(episode,allow_nan=False)+'\n')


def figure(summary,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,metric in zip(axes,('success_rate','spl')):
        for index,row in enumerate(summary['rows']):
            ax.plot([0,1],[row['baseline'][metric],row['wrapper'][metric]],alpha=.6,marker='o')
        ax.set(xticks=[0,1],xticklabels=['Base SARSA','History recovery'],ylim=(0,1),title=metric)
    fig.supxlabel('Development only: 4 layouts x 48 routes; 3 trained seeds x 3 RNG streams; shared map/goal features',fontsize=8)
    fig.savefig(path,dpi=140)
    plt.close(fig)


def run(root=ROOT):
    study = root/'artifacts/studies'/STUDY
    if study.exists():
        return verify(root)
    benchmark = MapBenchmark(root)
    source_study = root/'artifacts/studies/unseen-map-main-v1'
    settings = dict(SETTINGS)
    inputs = [root/'docs/loop_intervention_protocol.md',
              *benchmark.directory.glob('manifest.json'),*benchmark.directory.glob('train.json'),
              *benchmark.directory.glob('validation.json'),
              *sorted((root/'src/robot_navigation').glob('*.py'))]
    for seed in SETTINGS['training_seeds']:
        directory = source_study/'runs'/f'm11-shared-main-sarsa-seed{seed}'
        inputs.extend(directory/name for name in ('selected.npz','selected.json','manifest.json'))
    # Seal and copy all declared scientific inputs before evaluating any route.
    study.mkdir(parents=True)
    frozen = {p.relative_to(root).as_posix():sha(p) for p in inputs}
    for name in frozen:
        target = study/'snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
    release = {'settings':settings,'inputs':frozen,'signature':benchmark.signature,
               'protocol':'development only; controller selected before scoring; no weights changed'}
    write_json(study/'freeze.json',release)
    write_json(study/'seal.json',{'digest':digest(release)})
    write_json(study/'status.json',{'status':'running'})
    started,rows = perf_counter(),[]
    try:
        for seed in SETTINGS['training_seeds']:
            checkpoint = study/'snapshot/artifacts/studies/unseen-map-main-v1/runs'/f'm11-shared-main-sarsa-seed{seed}'/'selected.npz'
            policy = load_map_policy(checkpoint,benchmark)
            fingerprint = policy.state_fingerprint()
            baseline = []
            for layout,scenario in benchmark.cases('validation'):
                task = benchmark.task(layout)
                baseline.append(rollout(NavigationEnv(task,[scenario]),map_adapter(policy,task.grid),scenario))
            write_episodes(study/f'baseline-{seed}.jsonl.gz',baseline)
            for rng_seed in SETTINGS['rng_seeds']:
                controller = RecoveryPolicy(policy,rng_seed,settings['margin'],settings['max_recoveries'])
                wrapped = []
                for layout,scenario in benchmark.cases('validation'):
                    if perf_counter()-started>settings['wall_seconds']:
                        raise TimeoutError('Registered development wall limit exceeded')
                    if sum(p.stat().st_size for p in study.rglob('*') if p.is_file())>settings['artifact_bytes']:
                        raise RuntimeError('Registered artifact limit exceeded')
                    task = benchmark.task(layout)
                    controller.reset_episode()
                    class Adapter:
                        policy_id = controller.policy_id
                        def decision(self,coordinates):
                            return controller.decision(map_observation(coordinates,task.grid))
                    wrapped.append(rollout(NavigationEnv(task,[scenario]),Adapter(),scenario))
                path = study/f'wrapper-{seed}-{rng_seed}.jsonl.gz'
                write_episodes(path,wrapped)
                # Latency probes use a separate controller/RNG, outside recorded streams.
                layout,scenario = benchmark.cases('validation')[0]
                probe = map_observation([*scenario['start'],*scenario['goal']],benchmark.task(layout).grid)
                latency = {}
                for name,query in (('base',policy.act),('wrapper',RecoveryPolicy(policy,rng_seed).decision)):
                    tick = perf_counter()
                    for _ in range(100):
                        query(probe)
                    latency[name] = (perf_counter()-tick)*1e4
                rows.append({'training_seed':seed,'rng_seed':rng_seed,'baseline':aggregate([e['metrics'] for e in baseline]),
                    'wrapper':aggregate([e['metrics'] for e in wrapped]),'paired_changes':paired_result(baseline,wrapped),
                    'persistent_timeouts':sum(suffix_cycle(e) for e in wrapped),
                    'interventions':sum(t['decision']['recovery'] for e in wrapped for t in e['transitions']),
                    'decision_microseconds':latency})
                examples = {}
                for episode in wrapped:
                    outcome = episode['metrics']['outcome']
                    if outcome not in examples:
                        examples[outcome] = episode
                        write_json(study/f'replays/{seed}-{rng_seed}-{outcome}.json',episode)
                print(f'SARSA seed {seed}, RNG {rng_seed}: {rows[-1]["wrapper"]["counts"]}',flush=True)
            if policy.state_fingerprint()!=fingerprint:
                raise ValueError('Intervention changed pretrained weights')
        summary = summarize(rows)
        summary['wall_seconds'] = perf_counter()-started
        write_json(study/'summary.json',summary)
        with (study/'paired.csv').open('w',newline='',encoding='utf-8') as file:
            writer = csv.DictWriter(file,fieldnames=['training_seed','rng_seed','baseline_success','wrapper_success','baseline_spl','wrapper_spl','wrapper_collision','timeout_to_success','timeout_to_collision','interventions','persistent_timeouts','base_microseconds','wrapper_microseconds'])
            writer.writeheader()
            for r in rows:
                writer.writerow({'training_seed':r['training_seed'],'rng_seed':r['rng_seed'],
                    'baseline_success':r['baseline']['success_rate'],'wrapper_success':r['wrapper']['success_rate'],
                    'baseline_spl':r['baseline']['spl'],'wrapper_spl':r['wrapper']['spl'],
                    'wrapper_collision':r['wrapper']['collision_rate'],
                    'timeout_to_success':r['paired_changes'].get('timeout->success',0),
                    'timeout_to_collision':r['paired_changes'].get('timeout->collision',0),
                    'interventions':r['interventions'],'persistent_timeouts':r['persistent_timeouts'],
                    'base_microseconds':r['decision_microseconds']['base'],'wrapper_microseconds':r['decision_microseconds']['wrapper']})
        figure(summary,study/'paired.png')
        outputs = {p.relative_to(study).as_posix():sha(p) for p in study.rglob('*')
                   if p.is_file() and not p.is_relative_to(study/'snapshot') and p.name not in ('status.json','verification.json','outputs.json')}
        write_json(study/'outputs.json',outputs)
        write_json(study/'status.json',{'status':'complete; audit pending'})
        return verify(root)
    except BaseException as exc:
        write_json(study/'status.json',{'status':'incomplete','completed_runs':len(rows),'error':f'{type(exc).__name__}: {exc}'})
        write_json(study/'partial_rows.json',rows)
        raise


def verify(root=ROOT):
    study = root/'artifacts/studies'/STUDY
    release = read_json(study/'freeze.json')
    if release['settings']!=SETTINGS or digest(release)!=read_json(study/'seal.json')['digest']:
        raise ValueError('Registered intervention settings/seal differ')
    for name,expected in release['inputs'].items():
        if sha(study/'snapshot'/name)!=expected:
            raise ValueError(f'Frozen intervention input changed: {name}')
    for name,expected in read_json(study/'outputs.json').items():
        if sha(study/name)!=expected:
            raise ValueError(f'Intervention output changed: {name}')
    benchmark = MapBenchmark(root)
    saved = read_json(study/'summary.json')
    actions = 0
    for row in saved['rows']:
        records = []
        seed,rng = row['training_seed'],row['rng_seed']
        policy = load_map_policy(study/'snapshot/artifacts/studies/unseen-map-main-v1/runs'/f'm11-shared-main-sarsa-seed{seed}'/'selected.npz',benchmark)
        controller = RecoveryPolicy(policy,rng)
        for kind,filename in (('baseline',f'baseline-{seed}.jsonl.gz'),('wrapper',f'wrapper-{seed}-{rng}.jsonl.gz')):
            episodes = []
            with gzip.open(study/filename,'rt',encoding='utf-8') as file:
                for (layout,scenario),line in zip(benchmark.cases('validation'),file,strict=True):
                    episode = json.loads(line)
                    task = benchmark.task(layout)
                    if episode['scenario']!=scenario:
                        raise ValueError('Intervention case order differs')
                    validate_replay(task,episode)
                    controller.reset_episode()
                    for t in episode['transitions']:
                        observation = map_observation(t['observation'],task.grid)
                        if kind=='wrapper':
                            if controller.decision(observation)!=t['decision']:
                                raise ValueError('Recovery stream or diagnostics differ')
                        elif policy.act(observation)!=t['action']:
                            raise ValueError('Baseline action differs')
                    episodes.append(episode)
                    actions += len(episode['transitions'])
            if aggregate([e['metrics'] for e in episodes])!=row[kind]:
                raise ValueError('Intervention aggregates differ')
            records.append(episodes)
        if paired_result(*records)!=row['paired_changes'] or sum(suffix_cycle(e) for e in records[1])!=row['persistent_timeouts']:
            raise ValueError('Intervention paired outcomes or cycle counts differ')
    expected = summarize(saved['rows'])
    if any(saved[k]!=v for k,v in expected.items()):
        raise ValueError('Intervention summary differs')
    result = {'status':'passed','actions_audited_including_reused_baselines':actions,
              'wrapper_episodes':1728,'unique_baseline_episodes':576,
              'new_training_interactions':0,'final_layout_evaluations':0,
              'artifact_bytes':sum(p.stat().st_size for p in study.rglob('*') if p.is_file())}
    write_json(study/'verification.json',result)
    write_json(study/'status.json',{'status':'verified'})
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','verify'))
    args = parser.parse_args()
    result = run() if args.command=='run' else verify()
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
