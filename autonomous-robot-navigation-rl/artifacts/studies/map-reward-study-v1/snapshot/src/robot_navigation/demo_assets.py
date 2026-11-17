"""Versioned, reference-based saved-demo asset inventory; archives stay in place."""
import argparse

from .core import ROOT,read_json,write_json
from .release import sha,verify_freeze

MANIFEST = 'docs/evidence/demo-v1.json'


def build(root=ROOT):
    groups = {}
    def add(group,path):
        path = path.resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f'Demo dependency unavailable: {path}')
        groups.setdefault(group,set()).add(path.relative_to(root).as_posix())
    for folder,pattern in (('src','*.py'),('app','*.py')):
        for path in (root/folder).rglob(pattern):
            add('application',path)
    for name in ('pyproject.toml','requirements-lock.txt','configs/environment.yaml',
                 'data/maps/bottleneck-v1.json','data/scenarios/validation.json',
                 'docs/evidence/fixed-final-v1.json','docs/evidence/study-summaries-v1.json'):
        add('fixed-demo',root/name)
    for run in ('m4-tuned-q-learning-seed101','m4-control-q-learning-seed101'):
        directory = root/'artifacts/runs'/run
        for name in ('manifest.json','configuration.json','selected.npz','selected.json'):
            add('fixed-demo',directory/name)
        for path in (directory/'replays').glob('*.json'):
            add('fixed-demo',path)
    # Library discovery reads manifests even when a model is not selected.
    for path in (root/'artifacts/runs').glob('*/manifest.json'):
        add('fixed-demo',path)
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
    for name in ('data/unseen/unseen-maps-v1/manifest.json','docs/milestone13_protocol.md'):
        add('unseen-demo',root/name)
    final = root/'artifacts/studies/unseen-map-final-v1'
    freeze = read_json(final/'freeze.json')
    for name in ('freeze.json','seal.json'):
        add('unseen-demo',final/name)
    for name in freeze['files']:
        add('unseen-demo',final/'snapshot'/name)
    for record in freeze['records']:
        directory = final/'runs'/record['run_id']
        add('unseen-demo',directory/'manifest.json')
        for name in read_json(directory/'manifest.json')['files']:
            add('unseen-demo',directory/name)
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
    result = {'schema_version':1,'scope':'Assessed saved-data rehearsal, not every optional training/checkpoint asset',
        'entry':'app/streamlit_app.py','selected_policy':'m4-tuned-q-learning-seed101',
        'cohorts':['fixed validation demo','fixed inspected final aggregates','unseen inspected final recordings','post hoc diagnostics'],
        'feature_versions':['coordinates-v1','egocentric-map-goal-v1'],
        'training_seeds':[101,12101,12203,12301],
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
        raise ValueError('Unsupported saved-demo manifest')
    names = manifest['files'] if groups is None else set().union(*(manifest['groups'][g] for g in groups))
    errors = []
    for name in names:
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.append(f'Missing saved-demo asset: {name}')
        elif sha(path)!=manifest['files'][name]['sha256']:
            errors.append(f'Changed saved-demo asset: {name}')
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
