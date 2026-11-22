"""Versioned, reference-based project asset inventory; archives stay in place."""
import argparse

from .core import ROOT,read_json,write_json
from .release import sha,verify_freeze

MANIFEST = 'docs/evidence/warehouse-project-v1.json'


def build(root=ROOT):
    from .warehouse_study import FAMILIES,saved,checked_run
    study = saved(root)
    for algorithm in FAMILIES:
        for seed in study['systems'][algorithm]['seeds']:
            checked_run(algorithm,seed,root)
    groups = {}
    def add(group,path):
        path = path.resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f'Project dependency unavailable: {path}')
        groups.setdefault(group,set()).add(path.relative_to(root).as_posix())
    for folder,pattern in (('src','*.py'),('app','*.py')):
        for path in (root/folder).rglob(pattern):
            add('application',path)
    for name in ('pyproject.toml','requirements-lock.txt','.streamlit/config.toml','configs/environment.yaml',
                 'data/maps/bottleneck-v1.json','data/scenarios/validation.json',
                 'docs/evidence/fixed-final-v1.json','docs/evidence/study-summaries-v1.json'):
        add('historical-fixed',root/name)
    for run in ('m4-tuned-q-learning-seed101','m4-control-q-learning-seed101'):
        directory = root/'artifacts/runs'/run
        for name in ('manifest.json','configuration.json','selected.npz','selected.json'):
            add('historical-fixed',directory/name)
        for path in (directory/'replays').glob('*.json'):
            add('historical-fixed',path)
    # Library discovery reads manifests even when a model is not selected.
    for path in (root/'artifacts/runs').glob('*/manifest.json'):
        add('historical-fixed',path)
    for study in ('milestone5-v1','four-rl-v1','four-rl-analysis-v1','final-v1',
                  'unseen-map-pilot-v1','unseen-map-main-v1','unseen-map-final-v1'):
        directory = root/'artifacts/studies'/study
        for name in ('summary.json','verification.json','status.json','manifest.json'):
            if (directory/name).exists():
                add('results',directory/name)
        for pattern in ('*.csv','*.png','figures/*.png'):
            for path in directory.glob(pattern):
                add('results',path)
        if (directory/'verification.json').exists():
            for name in read_json(directory/'verification.json').get('output_hashes',{}):
                add('results',directory/name)
    for name in ('freeze.json','seal.json'):
        add('results',root/'artifacts/studies/final-v1'/name)
    for path in (root/'artifacts/studies/loop-recovery-dev-v1').rglob('*'):
        if path.is_file():
            add('recovery-development',path)
    for folder in ('data/navigation-lab/structured-navigation-v2','artifacts/studies/map-reward-study-v1'):
        for path in (root/folder).rglob('*'):
            if path.is_file():
                add('map-reward-lab',path)
    for name in ('configs/navigation_lab.yaml','docs/navigation_lab_protocol.md'):
        add('map-reward-lab',root/name)
    for folder in ('data/warehouse/warehouse-delivery-v1','artifacts/warehouse/warehouse-delivery-q-learning-v1'):
        for path in (root/folder).rglob('*'):
            if path.is_file():
                add('warehouse-delivery',path)
    for name in ('configs/warehouse_demo.yaml','docs/evidence/warehouse-protocol-v1.json'):
        add('warehouse-delivery',root/name)
    for path in (root/'artifacts/warehouse/warehouse-four-algorithm-v1').rglob('*'):
        if path.is_file():
            add('warehouse-comparison',path)
    for name in ('configs/warehouse_comparison.yaml','docs/warehouse_comparison_protocol.md','docs/warehouse_comparison_results.md','docs/evidence/warehouse-comparison-overview.png'):
        add('warehouse-comparison',root/name)
    for name in ('data/unseen/unseen-maps-v1/manifest.json','docs/milestone13_protocol.md'):
        add('historical-unseen',root/name)
    final = root/'artifacts/studies/unseen-map-final-v1'
    freeze = read_json(final/'freeze.json')
    for name in ('freeze.json','seal.json'):
        add('historical-unseen',final/name)
    for name in freeze['files']:
        add('historical-unseen',final/'snapshot'/name)
    for record in freeze['records']:
        directory = final/'runs'/record['run_id']
        add('historical-unseen',directory/'manifest.json')
        for name in read_json(directory/'manifest.json')['files']:
            add('historical-unseen',directory/name)
    diagnostic = root/'artifacts/studies/unseen-map-diagnostics-v1'
    record = read_json(diagnostic/'manifest.json')
    for name in record['inputs']:
        add('diagnostics',root/name)
    for name in (*record['outputs'],'manifest.json','verification.json'):
        add('diagnostics',diagnostic/name)
    # Both historical releases remain untouched. Report shared content identities.
    old = root/'artifacts/studies/final-v1'
    old_freeze = verify_freeze(old,read_json(old/'seal.json')['freeze_digest'],root=old/'snapshot')
    old_hashes,new_hashes = set(old_freeze['files'].values()),set(freeze['files'].values())
    names = sorted(set().union(*groups.values()))
    files = {name:{'sha256':sha(root/name),'bytes':(root/name).stat().st_size} for name in names}
    result = {'schema_version':1,'scope':'Current warehouse comparison and supporting historical evidence; optional historical checkpoints excluded',
        'entry':'app/streamlit_app.py','study':'warehouse-four-algorithm-v1',
        'title':'Autonomous Warehouse Robot Navigation: A Comparative Study of Q-learning, SARSA, DQN and PPO',
        'historical_selected_policy':'m4-tuned-q-learning-seed101',
        'cohorts':['warehouse four-algorithm familiar-floor development comparison','historical fixed validation','fixed inspected final aggregates','unseen inspected final recordings','post hoc diagnostics'],
        'feature_versions':['coordinates-v1','egocentric-map-goal-v1'],
        'training_seeds':[19411,19421,19431],
        'groups':{k:sorted(v) for k,v in groups.items()},'files':files,
        'total_bytes':sum(v['bytes'] for v in files.values()),
        'historical_releases':{'fixed_files':len(old_freeze['files']),'unseen_files':len(freeze['files']),
                               'shared_content_hashes':len(old_hashes & new_hashes)},
        'excluded':'Bulk historical periodic checkpoints and unrelated experiment snapshots; no archive files moved.'}
    write_json(root/MANIFEST,result)
    return result


def verify(root=ROOT,groups=None):
    manifest = read_json(root/MANIFEST)
    if manifest.get('schema_version')!=1:
        raise ValueError('Unsupported project manifest')
    names = manifest['files'] if groups is None else set().union(*(manifest['groups'][g] for g in groups))
    errors = []
    for name in names:
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.append(f'Missing project asset: {name}')
        elif sha(path)!=manifest['files'][name]['sha256']:
            errors.append(f'Changed project asset: {name}')
    if errors:
        raise ValueError('\n'.join(errors))
    return {'status':'passed','files':len(names),'total_bytes':sum(manifest['files'][n]['bytes'] for n in names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('build','verify'))
    args = parser.parse_args()
    result = build() if args.command=='build' else verify()
    print({k:v for k,v in result.items() if k in ('status','total_bytes','historical_releases')})


if __name__=='__main__':
    main()
