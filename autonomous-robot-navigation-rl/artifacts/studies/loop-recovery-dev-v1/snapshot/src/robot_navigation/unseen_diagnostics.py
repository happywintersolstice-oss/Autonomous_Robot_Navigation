"""Post hoc diagnostics over immutable M11/M12 trajectories, without policy calls."""
import argparse
from collections import Counter
import csv
import gzip
import json
from time import perf_counter

import numpy as np

from .core import ROOT, read_json, write_json
from .episodes import aggregate
from .four_way import FAMILIES
from .release import sha
from .unseen_audit import load_recording, replay_task
from .unseen_benchmark import MapBenchmark
from .unseen_final import FinalBenchmark, STUDY as FINAL, sealed_record, validate_result
from .unseen_main import STUDY as MAIN
from .workflow import source_hash

STUDY = 'unseen-map-diagnostics-v1'
CATEGORIES = ('optimal success','detour success','wall collision','boundary collision',
              'stationary timeout','two-cell timeout','longer-cycle timeout','non-periodic timeout')
LOOPS = ('stationary timeout','two-cell timeout','longer-cycle timeout')
COMPOSITION = {'Success':('optimal success','detour success'),'Two-cell cycle':('two-cell timeout',),
    'Other cycle':('stationary timeout','longer-cycle timeout'),
    'Collision':('wall collision','boundary collision'),'Other timeout':('non-periodic timeout',)}
COLORS = ('#14b8a6','#f59e0b','#f97316','#ef4444','#94a3b8')


def diagnose(task,episode):
    """Describe an already validated complete episode using its actual trace."""
    transitions = episode['transitions']
    states = [tuple(episode['initial_observation'])]+[tuple(t['next_observation']) for t in transitions]
    actions = [t['action'] for t in transitions]
    n = len(actions)
    metrics = episode['metrics']
    if not n or n!=metrics['steps']:
        raise ValueError('Diagnosis requires a complete, reconciled episode')
    mapping = {}
    for state,action in zip(states,actions):
        if state in mapping and mapping[state]!=action:
            raise ValueError('Frozen learned trace contains inconsistent state actions')
        mapping[state]=action
    first_return,cycle_start,period = None,None,None
    seen = {}
    outcome = metrics['outcome']
    if outcome=='timeout':
        for index,state in enumerate(states):
            if state in seen:
                start = seen[state]
                candidate = index-start
                persistent = (n-start>=2*candidate
                    and all(states[k]==states[start+(k-start)%candidate] for k in range(start,n+1))
                    and all(actions[k]==actions[start+(k-start)%candidate] for k in range(start,n)))
                if persistent:
                    cycle_start,first_return,period = start,index,candidate
                break
            seen[state]=index
    if outcome=='success':
        category = 'optimal success' if n==episode['scenario']['shortest_length'] else 'detour success'
    elif outcome=='collision':
        r,c = transitions[-1]['info']['attempted_position']
        category = 'boundary collision' if not (0<=r<task.grid.shape[0] and 0<=c<task.grid.shape[1]) else 'wall collision'
    elif outcome=='timeout':
        category = {1:'stationary timeout',2:'two-cell timeout'}.get(period,
                    'longer-cycle timeout' if period else 'non-periodic timeout')
    else:
        raise ValueError('Unknown complete episode outcome')
    distances = task.grid.distances(tuple(episode['scenario']['goal']))
    goal_distances = [distances[s[:2]] for s in states]
    event = first_return if period else n
    failure_cell = list(states[cycle_start if period else n][:2]) if outcome!='success' else None
    return {'category':category,'cycle_start':cycle_start,'first_return':first_return,'period':period,
        'actions_after_first_return':n-first_return if period else 0,
        'distinct_cells':len({s[:2] for s in states}),
        'minimum_goal_distance':min(goal_distances),'final_goal_distance':goal_distances[-1],
        'closest_approach_action':goal_distances.index(min(goal_distances)),
        'failure_cell':failure_cell,'event_action':event}


def source_jobs(root,track):
    """Pin the two verified learned-policy cohorts and their full support."""
    benchmark = MapBenchmark(root)
    jobs = []
    for dataset,name in (('development',MAIN),('final',FINAL)):
        study = root/'artifacts/studies'/name
        summary = track(study/'summary.json')
        verification = track(study/'verification.json')
        if summary['status']!='verified' or verification['status']!='passed' or summary['signature']!=benchmark.signature:
            raise ValueError('Diagnostics require verified compatible source studies')
        if dataset=='final':
            release,seal = sealed_record(study)
            track(study/'freeze.json')
            track(study/'seal.json')
            if verification['summary_sha256']!=sha(study/'summary.json') or summary['release_digest']!=seal:
                raise ValueError('Final summary/seal mismatch')
            current = FinalBenchmark(study)
        else:
            track(study/'manifest.json')
            current = benchmark
        for run in summary['runs']:
            if run['phase'] not in ('main','final'):
                continue
            directory = study/'runs'/run['run_id']
            manifest = track(directory/'manifest.json')
            filename = 'trajectories.jsonl.gz' if dataset=='final' else 'validation.jsonl.gz'
            expected = run['final'] if dataset=='final' else run['validation']
            if dataset=='final':
                if run['record'] not in release['records']:
                    raise ValueError('Diagnostic policy is outside the final roster')
                validate_result(directory,run['record'],seal)
            elif manifest['status']!='complete' or manifest['signature']!=current.signature:
                raise ValueError('Development run identity mismatch')
            path = directory/filename
            track(path,json_file=False)
            metadata = (study/'snapshot'/run['record']['checkpoint']).with_suffix('.json') if dataset=='final' else (directory/manifest['viewer_checkpoint']).with_suffix('.json')
            policy_id = track(metadata)['policy_id']
            for item in manifest['replays']:
                track(directory/item['path'])
            jobs.append({'dataset':dataset,'study':study,'benchmark':current,'run':run,
                         'manifest':manifest,'path':path,'expected':expected,'policy_id':policy_id})
    expected_roster = {(a,s) for a in FAMILIES for s in (12101,12203,12301)}
    for dataset in ('development','final'):
        group = [j for j in jobs if j['dataset']==dataset]
        if len(group)!=12 or {(j['run']['algorithm'],j['run']['training_seed']) for j in group}!=expected_roster:
            raise ValueError('Diagnostic roster must contain all twelve learned policies per cohort')
    return jobs


def grouped(rows):
    results = []
    keys = sorted({(r['dataset'],r['algorithm']) for r in rows},
                  key=lambda key:(key[0],list(FAMILIES).index(key[1])))
    for dataset,algorithm in keys:
        selected = [r for r in rows if (r['dataset'],r['algorithm'])==(dataset,algorithm)]
        seeds = sorted({r['training_seed'] for r in selected})
        seed_counts = {seed:Counter(r['category'] for r in selected if r['training_seed']==seed) for seed in seeds}
        totals = dict(Counter(r['category'] for r in selected))
        rates = {}
        for category in CATEGORIES:
            values = [seed_counts[seed][category]/seed_counts[seed].total() for seed in seeds]
            rates[category] = {'mean':float(np.mean(values)),
                               'sd':float(np.std(values,ddof=1)) if len(values)>1 else None}
        timeouts = [r for r in selected if r['outcome']=='timeout']
        success_rates = [(c['optimal success']+c['detour success'])/c.total() for c in seed_counts.values()]
        results.append({'dataset':dataset,'algorithm':algorithm,'episodes':len(selected),'seed_count':len(seeds),
            'success_mean':float(np.mean(success_rates)),
            'success_sd':float(np.std(success_rates,ddof=1)) if len(seeds)>1 else None,
            'counts':{category:totals.get(category,0) for category in CATEGORIES},'rates':rates,
            'timeout_episodes':len(timeouts),'periodic_timeouts':sum(r['category'] in LOOPS for r in timeouts),
            'timeout_actions':sum(r['steps'] for r in timeouts),
            'actions_after_first_return':sum(r['actions_after_first_return'] for r in timeouts),
            'mean_cycle_entry_action':float(np.mean([r['cycle_start'] for r in timeouts if r['period']]))
                                      if any(r['period'] for r in timeouts) else None})
    return results


def calculate(root,deadline=None):
    inputs = {}
    def track(path,json_file=True):
        relative = path.relative_to(root).as_posix()
        inputs[relative] = sha(path)
        return read_json(path) if json_file else None
    jobs = source_jobs(root,track)
    rows,examples,audits = [],[],[]
    for job in jobs:
        benchmark,run = job['benchmark'],job['run']
        split = 'test' if job['dataset']=='final' else 'validation'
        cases = benchmark.cases(split)
        records = []
        example_cases = {}
        directory = job['study']/'runs'/run['run_id']
        for item in job['manifest']['replays']:
            _,episode = load_recording(benchmark,directory,item)
            example_cases[(episode['layout_id'],episode['scenario']['id'])] = item
        example_categories = set()
        with gzip.open(job['path'],'rt',encoding='utf-8') as file:
            for (layout,scenario),line in zip(cases,file,strict=True):
                if deadline is not None and perf_counter()>deadline:
                    raise TimeoutError('Diagnostic calculation exceeded its registered wall limit')
                episode = json.loads(line)
                if (episode['layout_id']!=layout['map_id'] or episode['scenario']!=scenario
                        or episode['layout_split']!=split or episode['policy_id']!=job['policy_id']):
                    raise ValueError('Diagnostic case order/policy mismatch')
                task = replay_task(benchmark,episode,allow_final=split=='test')
                diagnosis = diagnose(task,episode)
                row = {'dataset':job['dataset'],'algorithm':run['algorithm'],'training_seed':run['training_seed'],
                    'run_id':run['run_id'],'layout_id':layout['map_id'],'scenario_id':scenario['id'],
                    'distance_band':scenario['distance_band'],'shortest_length':scenario['shortest_length'],
                    'outcome':episode['metrics']['outcome'],'steps':episode['metrics']['steps'],
                    'return':episode['metrics']['return'],'spl':episode['metrics']['spl'],**diagnosis}
                rows.append(row)
                records.append(episode['metrics'])
                item = example_cases.get((layout['map_id'],scenario['id']))
                if item and row['outcome']!='success' and row['category'] not in example_categories:
                    examples.append({**{k:row[k] for k in ('dataset','algorithm','training_seed','run_id','category',
                        'layout_id','scenario_id','event_action','cycle_start','first_return','period')},'item':item})
                    example_categories.add(row['category'])
        observed = aggregate(records)
        if any(observed[k]!=job['expected'][k] for k in observed):
            raise ValueError('Diagnostic outcomes differ from verified source metrics')
        audits.append({'dataset':job['dataset'],'run_id':run['run_id'],
                       'episodes':len(records),'actions':observed['interactions']})
    for name,expected in inputs.items():
        if sha(root/name)!=expected:
            raise ValueError('Diagnostic input changed during calculation')
    return {'inputs':inputs,'rows':rows,'groups':grouped(rows),'examples':examples,'runs':audits}


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)


def plot_summary(summary,path):
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes = plt.subplots(1,2,figsize=(12,4))
        for ax,dataset in zip(axes,('development','final'),strict=True):
            groups = [next(g for g in summary['groups'] if (g['dataset'],g['algorithm'])==(dataset,a))
                      for a in FAMILIES]
            bottom = np.zeros(len(groups))
            for (label,members),color in zip(COMPOSITION.items(),COLORS,strict=True):
                values = np.array([100*sum(g['counts'][k] for k in members)/g['episodes'] for g in groups])
                ax.bar(range(len(groups)),values,bottom=bottom,label=label,color=color,width=.65)
                bottom += values
            ax.set_xticks(range(len(groups)),list(FAMILIES.values()))
            ax.set_ylim(0,100)
            ax.set_ylabel('Share of policy–route cases (%)')
            name = 'Development validation' if dataset=='development' else 'Inspected final maps'
            ax.set_title(f'{name} · {sum(g["episodes"] for g in groups):,} cases')
        handles,labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(.5,1.02))
        fig.tight_layout(rect=(0,0,1,.92))
        fig.savefig(path,dpi=160,bbox_inches='tight')
        plt.close(fig)


def build(root=ROOT):
    study = root/'artifacts/studies'/STUDY
    if study.exists():
        return saved(root)
    pending = study.with_name(STUDY+'.pending')
    pending.mkdir(parents=True,exist_ok=False)
    started = perf_counter()
    result = calculate(root,deadline=started+900)
    summary = {'status':'complete; audit pending','groups':result['groups'],'examples':result['examples'],
        'difficulty':[{**g,'distance_band':band} for band in ('short','medium','long')
                      for g in grouped([r for r in result['rows'] if r['distance_band']==band])],
        'runs':result['runs'],'episodes':len(result['rows']),'actions':sum(r['actions'] for r in result['runs']),
        'new_training_interactions':0,'new_evaluation_episodes':0,'policy_queries':0,
        'scope':'post hoc saved-trajectory diagnosis; development and inspected-final cohorts separate'}
    write_json(pending/'summary.json',summary)
    write_json(pending/'episodes.json',result['rows'])
    write_csv(pending/'episodes.csv',result['rows'])
    per_seed = []
    for dataset,algorithm,seed in sorted({(r['dataset'],r['algorithm'],r['training_seed']) for r in result['rows']}):
        group = [r for r in result['rows'] if (r['dataset'],r['algorithm'],r['training_seed'])==(dataset,algorithm,seed)]
        per_seed.append({'dataset':dataset,'algorithm':algorithm,'training_seed':seed,'episodes':len(group),
                         **{category:sum(r['category']==category for r in group) for category in CATEGORIES}})
    write_csv(pending/'seeds.csv',per_seed)
    plot_summary(summary,pending/'failure_patterns.png')
    protocol = root/'docs/milestone13_protocol.md'
    (pending/'protocol.md').write_bytes(protocol.read_bytes())
    manifest = {'status':summary['status'],'inputs':result['inputs'],'protocol_hash':sha(protocol),
        'source_hash':source_hash(root),'seconds':perf_counter()-started,
        'outputs':{p.name:sha(p) for p in pending.iterdir() if p.is_file()}}
    write_json(pending/'manifest.json',manifest)
    if sum(p.stat().st_size for p in pending.rglob('*') if p.is_file())>64*1024**2:
        raise RuntimeError('Diagnostic derived artifacts exceeded 64 MiB')
    pending.rename(study)
    return summary


def saved(root=ROOT):
    study = root/'artifacts/studies'/STUDY
    manifest = read_json(study/'manifest.json')
    if sha(root/'docs/milestone13_protocol.md')!=manifest['protocol_hash']:
        raise ValueError('Diagnostic protocol changed; use a new study version')
    for base,paths in ((root,manifest['inputs']),(study,manifest['outputs'])):
        for name,expected in paths.items():
            if sha(base/name)!=expected:
                raise ValueError('Diagnostic source/output checksum mismatch')
    return read_json(study/'summary.json')


def verify(root=ROOT):
    started = perf_counter()
    study = root/'artifacts/studies'/STUDY
    summary = saved(root)
    result = calculate(root,deadline=perf_counter()+900)
    if (read_json(study/'episodes.json')!=result['rows'] or summary['groups']!=result['groups']
            or summary['examples']!=result['examples'] or summary['runs']!=result['runs']):
        raise ValueError('Recomputed diagnostic records differ')
    with (study/'episodes.csv').open(newline='',encoding='utf-8') as file:
        for row,stored in zip(result['rows'],csv.DictReader(file),strict=True):
            expected = {k:json.dumps(v) if isinstance(v,(dict,list)) else '' if v is None else str(v)
                        for k,v in row.items()}
            if stored!=expected:
                raise ValueError('Diagnostic CSV differs from its JSON records')
    difficulty = [{**g,'distance_band':band} for band in ('short','medium','long')
                  for g in grouped([r for r in result['rows'] if r['distance_band']==band])]
    if summary['difficulty']!=difficulty:
        raise ValueError('Route-difficulty groups differ')
    manifest = read_json(study/'manifest.json')
    if manifest['inputs']!=result['inputs']:
        raise ValueError('Diagnostic input roster differs')
    summary['status'] = manifest['status'] = 'verified'
    write_json(study/'summary.json',summary)
    manifest['outputs']['summary.json'] = sha(study/'summary.json')
    write_json(study/'manifest.json',manifest)
    audit = {'status':'passed','episodes':len(result['rows']),'actions':sum(r['actions'] for r in result['runs']),
             'audit_seconds':perf_counter()-started,
             'source_hash':source_hash(root),'input_files':len(result['inputs']),
             'summary_sha256':sha(study/'summary.json'),
             'new_training_interactions':0,'new_evaluation_episodes':0,'policy_queries':0}
    write_json(study/'verification.json',audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('build','verify'))
    args = parser.parse_args()
    result = build() if args.command=='build' else verify()
    print(json.dumps({k:result[k] for k in ('status','episodes','actions','policy_queries') if k in result},indent=2))


if __name__=='__main__':
    main()
