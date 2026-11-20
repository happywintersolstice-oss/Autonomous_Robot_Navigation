"""Structured fixed-layout benchmark and matched reward study, separately versioned."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import random
import shutil
from time import perf_counter

import numpy as np
import yaml

from .agents import BFSPolicy,QLearning
from .core import ROOT,MOVES,GridMap,Task,NavigationEnv,digest,read_json,write_json,validate_scenario
from .episodes import rollout,aggregate,episode_metrics
from .release import sha
from .training import epsilon_at

BENCHMARK = 'structured-navigation-v2'
STUDY = 'map-reward-study-v1'


def configuration(root=ROOT):
    value = yaml.safe_load((root/'configs/navigation_lab.yaml').read_text(encoding='utf-8'))
    if value['benchmark_id']!=BENCHMARK or value['study_id']!=STUDY or value['shape']!=[10,10]:
        raise ValueError('Unsupported navigation lab contract')
    if value['interactions']%value['monitor_every'] or value['discount']!=.99:
        raise ValueError('Invalid reward study monitoring/discount contract')
    return value


def neighbors(grid,cell):
    return [(cell[0]+dr,cell[1]+dc) for dr,dc in MOVES if grid.is_free((cell[0]+dr,cell[1]+dc))]


def topology(grid):
    degrees = {cell:len(neighbors(grid,cell)) for cell in grid.free_cells}
    ends = [cell for cell,n in degrees.items() if n==1]
    depths = []
    for end in ends:
        previous,current,depth = None,end,0
        while True:
            onward = [c for c in neighbors(grid,current) if c!=previous]
            if not onward or (depth and degrees[current]!=2):
                break
            previous,current = current,onward[0]
            depth += 1
        depths.append(depth)
    distances = {cell:grid.distances(cell) for cell in grid.free_cells}
    detours = sum(length>abs(start[0]-goal[0])+abs(start[1]-goal[1])
                  for start,values in distances.items() for goal,length in values.items() if start!=goal)
    return {'free_cells':len(degrees),'obstacles':100-len(degrees),
        'connected':len(distances[grid.free_cells[0]])==len(degrees),
        'dead_ends':len(ends),'dead_end_cells':[list(c) for c in ends],
        'deepest_dead_end':max(depths,default=0),'junctions':sum(n>=3 for n in degrees.values()),
        'corridor_cells':sum(n==2 for n in degrees.values()),
        'cycle_rank':sum(degrees.values())//2-len(degrees)+1,
        'diameter':max(max(v.values()) for v in distances.values()),'ordered_pairs_requiring_detour':detours}


def generate_maze(seed,family):
    rng = random.Random(seed)
    free,visited,stack = {(1,1)},{(1,1)},[(1,1)]
    while stack:
        r,c = stack[-1]
        candidates = [(r+2*dr,c+2*dc) for dr,dc in MOVES
                      if 1<=r+2*dr<=9 and 1<=c+2*dc<=9 and (r+2*dr,c+2*dc) not in visited]
        if not candidates:
            stack.pop()
            continue
        cell = rng.choice(candidates)
        free.update((cell,((r+cell[0])//2,(c+cell[1])//2)))
        visited.add(cell)
        stack.append(cell)
    if family=='looped-corridors':
        for _ in range(3):
            candidates = [(r,c) for r in range(1,9) for c in range(1,9) if (r,c) not in free
                          and sum((r+dr,c+dc) in free for dr,dc in MOVES)>=2]
            free.add(rng.choice(candidates))
    if family=='rooms-and-doorways':
        free.update((r,c) for r in range(1,4) for c in range(1,4))
        free.update((r,c) for r in range(6,9) for c in range(6,9))
    rows = tuple(''.join('.' if (r,c) in free else '#' for c in range(10)) for r in range(10))
    return GridMap(f'{BENCHMARK}-{family}',rows)


def route_splits(grid,config):
    rng = random.Random(config['split_seed'])
    groups = defaultdict(list)
    ends = {tuple(c) for c in topology(grid)['dead_end_cells']}
    for goal in grid.free_cells:
        for start,length in sorted(grid.distances(goal).items()):
            if start==goal:
                continue
            manhattan = abs(start[0]-goal[0])+abs(start[1]-goal[1])
            band = 'short' if length<=5 else 'medium' if length<=10 else 'long'
            groups[(goal,band)].append({'id':f'r{start[0]}c{start[1]}-r{goal[0]}c{goal[1]}',
                'start':list(start),'goal':list(goal),'shortest_length':length,'distance_band':band,
                'requires_detour':length>manhattan,'starts_in_dead_end':start in ends,'goal_in_dead_end':goal in ends})
    splits = {name:[] for name in ('train','validation','reserved')}
    for values in groups.values():
        rng.shuffle(values)
        n = max(1,round(len(values)*.15)) if len(values)>=4 else 0
        splits['train'].extend(values[:len(values)-2*n])
        splits['validation'].extend(values[len(values)-2*n:len(values)-n] if n else [])
        splits['reserved'].extend(values[len(values)-n:] if n else [])
    for values in splits.values():
        values.sort(key=lambda v:v['id'])
    # Monitor tasks are fixed by distance band, never learned outcomes.
    buckets = [[s for s in splits['validation'] if s['distance_band']==band] for band in ('short','medium','long')]
    for bucket in buckets:
        rng.shuffle(bucket)
    monitor = []
    while len(monitor)<min(config['monitor_count'],len(splits['validation'])):
        for bucket in buckets:
            if bucket and len(monitor)<config['monitor_count']:
                monitor.append(bucket.pop())
    splits['monitoring'] = monitor
    return splits


def prepare(root=ROOT):
    config = configuration(root)
    directory = root/'data/navigation-lab'/BENCHMARK
    if (directory/'manifest.json').exists():
        record = read_json(directory/'manifest.json')
        if record['configuration_hash']!=digest(config):
            raise ValueError('Existing structured benchmark changed; use a new version')
        for name,expected in record['files'].items():
            if sha(directory/name)!=expected:
                raise ValueError(f'Structured benchmark checksum mismatch: {name}')
        return record
    if directory.exists():
        raise ValueError('Incomplete benchmark publication retained; use a new version')
    layouts = []
    identities = set()
    from .unseen_benchmark import geometry_hash
    for index,family in enumerate(config['layout_families']):
        for attempt in range(500):
            grid = generate_maze(config['generation_seed']+index*1000+attempt,family)
            stats = topology(grid)
            if (stats['connected'] and stats['dead_ends']>=config['minimum_dead_ends']
                    and stats['junctions']>=config['minimum_junctions']
                    and stats['corridor_cells']>=config['minimum_corridor_cells']
                    and stats['deepest_dead_end']>=config['minimum_dead_end_depth']
                    and stats['diameter']>=config['minimum_diameter']
                    and (family!='deep-cul-de-sacs' or stats['deepest_dead_end']>=8)
                    and geometry_hash(grid) not in identities):
                break
        else:
            raise ValueError(f'No layout meets declared topology for {family}')
        identities.add(geometry_hash(grid))
        layouts.append({'map_id':grid.map_id,'family':family,'rows':list(grid.rows),'map_hash':grid.hash,
                        'generation_attempt':attempt,'topology':stats,'splits':route_splits(grid,config)})
    directory.mkdir(parents=True)
    for layout in layouts:
        write_json(directory/f'{layout["family"]}.json',layout)
    record = {'benchmark_id':BENCHMARK,'configuration_hash':digest(config),
              'layouts':[m['family'] for m in layouts],
              'files':{f'{m["family"]}.json':sha(directory/f'{m["family"]}.json') for m in layouts},
              'scope':'Four familiar structured maps; disjoint initial pairs, not unseen-layout evaluation',
              'reserved_access':'Not used in this development study'}
    write_json(directory/'manifest.json',record)
    gallery(layouts,directory/'gallery.png')
    return record


def load_layouts(root=ROOT):
    record = prepare(root)
    directory = root/'data/navigation-lab'/BENCHMARK
    return [read_json(directory/f'{name}.json') for name in record['layouts']]


def make_task(layout,condition,config):
    rewards = config['reward_conditions'][condition]
    return Task(GridMap(layout['map_id'],tuple(layout['rows'])),{
        'map_id':layout['map_id'],'environment_version':'grid-v1','observation_version':'coordinates-v1',
        'action_version':'up-down-left-right-v1','reward_version':f'structured-{condition}-v1',
        'max_steps':config['max_steps'],'discount':config['discount'],**rewards})


class RewardEnv(NavigationEnv):
    """Potential from observed coordinates; no planner or visitation penalty."""
    @staticmethod
    def potential(observation):
        r,c,gr,gc = map(int,observation)
        return -float(abs(r-gr)+abs(c-gc))

    def step(self,action):
        previous = self._observation()
        observation,base,terminated,truncated,info = super().step(action)
        scale = self.task.config['shaping_scale']
        next_potential = 0. if terminated else self.potential(observation)
        shaping = scale*(self.task.config['discount']*next_potential-self.potential(previous))
        return observation,float(base+shaping),terminated,truncated,{
            **info,'reward_components':{'base':float(base),'potential':float(shaping)}}


def validate_record(task,episode):
    task.check(episode['signature'])
    scenario = episode['scenario']
    env = RewardEnv(task,[scenario])
    obs,_ = env.reset(seed=episode['seed'],options={'scenario':scenario})
    if obs.tolist()!=episode['initial_observation']:
        raise ValueError('Lab initial observation differs')
    for transition in episode['transitions']:
        if transition['observation']!=obs.tolist():
            raise ValueError('Lab before-state differs')
        obs,reward,end,cut,info = env.step(transition['action'])
        if (transition['next_observation']!=obs.tolist() or transition['reward']!=reward
                or transition['terminated']!=end or transition['truncated']!=cut or transition['info']!=info):
            raise ValueError('Lab transition/reward differs')
    if episode_metrics(episode)!=episode['metrics']:
        raise ValueError('Lab metrics differ')
    return episode


def evaluate(task,scenarios,policy):
    env = RewardEnv(task,scenarios)
    episodes = [rollout(env,policy,scenario) for scenario in scenarios]
    return aggregate([e['metrics'] for e in episodes]),episodes


def extra_metrics(episodes):
    repeated = 0
    timeout_loops = 0
    for episode in episodes:
        states = [tuple(episode['initial_observation'][:2])]+[tuple(t['next_observation'][:2]) for t in episode['transitions']]
        # Exclude the collision's unchanged position from movement revisits.
        if episode['metrics']['outcome']=='collision':
            states = states[:-1]
        repeated += len(states)-len(set(states))
        timeout_loops += episode['metrics']['outcome']=='timeout' and len(states)>len(set(states))
    return {'revisited_movement_cells':repeated,'looping_timeouts':timeout_loops}


def gallery(layouts,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig,axes = plt.subplots(2,2,figsize=(10,10),layout='constrained')
    for ax,layout in zip(axes.flat,layouts):
        grid = GridMap(layout['map_id'],tuple(layout['rows']))
        ax.imshow([[int(c=='#') for c in row] for row in grid.rows],cmap=ListedColormap(['#e7f2f2','#243e50']),vmin=0,vmax=1)
        for r,c in layout['topology']['dead_end_cells']:
            ax.scatter(c,r,c='#e17633',s=35)
        ax.set(xticks=range(10),yticks=range(10),xlabel='column',ylabel='row',title=layout['family'])
        stats = layout['topology']
        ax.text(.5,-.15,f"{stats['dead_ends']} dead ends; depth {stats['deepest_dead_end']}; {stats['junctions']} junctions; diameter {stats['diameter']}",transform=ax.transAxes,ha='center',fontsize=9)
    fig.suptitle('Structured navigation v2 - orange marks degree-one dead ends',fontsize=13)
    fig.savefig(path,dpi=130)
    plt.close(fig)


def train_one(layout,condition,seed,config,directory,deadline):
    task = make_task(layout,condition,config)
    policy = QLearning(task,f'q-learning/{STUDY}/{layout["family"]}/{condition}/seed{seed}')
    rng = np.random.default_rng(seed)
    env = RewardEnv(task,layout['splits']['train'])
    obs,_ = env.reset(seed=seed)
    schedule = {**config,'interactions':config['interactions']}
    best,monitor = None,[]
    started = perf_counter()
    directory.mkdir(parents=True,exist_ok=False)
    for step in range(config['interactions']):
        if step%1000==0 and perf_counter()>deadline:
            raise TimeoutError('Registered map/reward study wall limit exceeded')
        action = policy.explore(obs,epsilon_at(schedule,step),rng)
        next_obs,reward,end,cut,_ = env.step(action)
        policy.update(obs,action,reward,next_obs,end,config['learning_rate'],config['discount'])
        obs = env.reset()[0] if end or cut else next_obs
        if (step+1)%config['monitor_every']==0:
            metrics,_ = evaluate(task,layout['splits']['monitoring'],policy)
            monitor.append({'step':step+1,**metrics})
            rank = (metrics['success_rate'],metrics['spl'],-(step+1))
            if best is None or rank>best:
                best = rank
                policy.save(directory/'selected.npz',{'training_interactions':step+1,'training_seed':seed})
                selected_step = step+1
    training_seconds = perf_counter()-started
    policy = QLearning.load(directory/'selected.npz',task)
    metrics,episodes = evaluate(task,layout['splits']['validation'],policy)
    with gzip.open(directory/'episodes.jsonl.gz','xt',encoding='utf-8') as file:
        for episode in episodes:
            file.write(json.dumps(episode,allow_nan=False)+'\n')
    row = {'family':layout['family'],'condition':condition,'training_seed':seed,
           'training_interactions':config['interactions'],'selected_interactions':selected_step,
           'training_monitor_seconds':training_seconds,'metrics':metrics,'repetition':extra_metrics(episodes),
           'signature':task.signature,'episode_sha256':sha(directory/'episodes.jsonl.gz')}
    write_json(directory/'monitoring.json',monitor)
    write_json(directory/'manifest.json',row)
    print(f"{layout['family']} / {condition} / seed{seed}: {metrics['success_rate']:.1%}",flush=True)
    return row


def summarize(rows,config):
    expected = {(f,c,s) for f in config['layout_families'] for c in config['reward_conditions'] for s in config['training_seeds']}
    if len(rows)!=len(expected) or {(r['family'],r['condition'],r['training_seed']) for r in rows}!=expected:
        raise ValueError('Map/reward study matrix incomplete')
    systems = {}
    for condition in config['reward_conditions']:
        per_seed = []
        for seed in config['training_seeds']:
            selected = [r for r in rows if r['condition']==condition and r['training_seed']==seed]
            per_seed.append({'seed':seed,**{m:float(np.mean([r['metrics'][m] for r in selected]))
                            for m in ('success_rate','collision_rate','timeout_rate','spl')},
                            'looping_timeouts':sum(r['repetition']['looping_timeouts'] for r in selected)})
        systems[condition] = {'seed_results':per_seed,
            'metrics':{m:{'mean':float(np.mean([s[m] for s in per_seed])),
                          'sample_sd':float(np.std([s[m] for s in per_seed],ddof=1))}
                       for m in ('success_rate','collision_rate','timeout_rate','spl')},
            'looping_timeouts':sum(s['looping_timeouts'] for s in per_seed)}
    return {'study_id':STUDY,'scope':'Development pairs on four familiar structured maps; not unseen-map or independent final evidence',
            'systems':systems,'runs':rows,'new_training_interactions':sum(r['training_interactions'] for r in rows),
            'reserved_evaluation_episodes':0,'selection':'monitor success, then SPL, then earliest step',
            'aggregation':'equal map means within each training seed, then equal seed means/sample SD'}


def run(root=ROOT):
    config = configuration(root)
    layouts = load_layouts(root)
    study = root/'artifacts/studies'/STUDY
    if study.exists():
        return verify(root)
    study.mkdir(parents=True)
    directory = root/'data/navigation-lab'/BENCHMARK
    inputs = [root/'configs/navigation_lab.yaml',root/'docs/navigation_lab_protocol.md',
              *sorted(directory.glob('*.json')),*sorted((root/'src/robot_navigation').glob('*.py'))]
    files = {p.relative_to(root).as_posix():sha(p) for p in inputs}
    for name in files:
        target = study/'snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
    release = {'configuration':config,'files':files,'benchmark':BENCHMARK}
    write_json(study/'freeze.json',release)
    write_json(study/'seal.json',{'digest':digest(release)})
    write_json(study/'status.json',{'status':'running'})
    start,rows = perf_counter(),[]
    try:
        for layout in layouts:
            for seed in config['training_seeds']:
                for condition in config['reward_conditions']:
                    if sum(p.stat().st_size for p in study.rglob('*') if p.is_file())>config['artifact_bytes']:
                        raise RuntimeError('Registered map/reward artifact limit exceeded')
                    directory = study/'runs'/f'{layout["family"]}-{condition}-seed{seed}'
                    rows.append(train_one(layout,condition,seed,config,directory,start+config['wall_seconds']))
        summary = summarize(rows,config)
        summary['wall_seconds'] = perf_counter()-start
        write_json(study/'summary.json',summary)
        hashes = {p.relative_to(study).as_posix():sha(p) for p in study.rglob('*') if p.is_file()
                  and not p.is_relative_to(study/'snapshot') and p.name not in ('status.json','verification.json')}
        write_json(study/'outputs.json',hashes)
        return verify(root)
    except BaseException as exc:
        write_json(study/'status.json',{'status':'incomplete','completed_runs':len(rows),'error':f'{type(exc).__name__}: {exc}'})
        write_json(study/'partial_rows.json',rows)
        raise


def verify(root=ROOT):
    study = root/'artifacts/studies'/STUDY
    release = read_json(study/'freeze.json')
    if digest(release)!=read_json(study/'seal.json')['digest'] or release['configuration']!=configuration(root):
        raise ValueError('Map/reward freeze or configuration mismatch')
    for name,expected in release['files'].items():
        if sha(study/'snapshot'/name)!=expected:
            raise ValueError(f'Frozen map/reward input changed: {name}')
    for name,expected in read_json(study/'outputs.json').items():
        if sha(study/name)!=expected:
            raise ValueError(f'Map/reward output changed: {name}')
    config = release['configuration']
    layouts = {m['family']:m for m in load_layouts(root)}
    summary = read_json(study/'summary.json')
    expected = summarize(summary['runs'],config)
    if any(expected[k]!=summary[k] for k in expected):
        raise ValueError('Map/reward summary does not reconcile')
    actions = episodes = 0
    for row in summary['runs']:
        layout = layouts[row['family']]
        task = make_task(layout,row['condition'],config)
        directory = study/'runs'/f'{row["family"]}-{row["condition"]}-seed{row["training_seed"]}'
        policy = QLearning.load(directory/'selected.npz',task)
        before = policy.q.copy()
        records = []
        with gzip.open(directory/'episodes.jsonl.gz','rt',encoding='utf-8') as file:
            for scenario,line in zip(layout['splits']['validation'],file,strict=True):
                episode = validate_record(task,json.loads(line))
                if episode['scenario']!=scenario:
                    raise ValueError('Map/reward validation scenario differs')
                for t in episode['transitions']:
                    if t['action']!=policy.act(t['observation']) or t['scores']!=policy.scores(t['observation']).tolist():
                        raise ValueError('Map/reward saved action differs from checkpoint')
                records.append(episode)
                episodes += 1
                actions += len(episode['transitions'])
        if aggregate([e['metrics'] for e in records])!=row['metrics'] or extra_metrics(records)!=row['repetition'] or not np.array_equal(before,policy.q):
            raise ValueError('Map/reward metrics, repetitions or inference state differ')
        monitor = read_json(directory/'monitoring.json')
        if max(monitor,key=lambda r:(r['success_rate'],r['spl'],-r['step']))['step']!=row['selected_interactions']:
            raise ValueError('Map/reward checkpoint selection differs')
    result = {'status':'passed','runs':len(summary['runs']),'episodes':episodes,'actions':actions,
              'inference_unchanged':True,'reserved_evaluation_episodes':0}
    write_json(study/'verification.json',result)
    write_json(study/'status.json',{'status':'verified'})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','verify'))
    args = parser.parse_args()
    result = {'prepare':prepare,'run':run,'verify':verify}[args.command]()
    print(json.dumps({k:v for k,v in result.items() if k in ('benchmark_id','scope','new_training_interactions','wall_seconds')},indent=2))


if __name__=='__main__':
    main()
