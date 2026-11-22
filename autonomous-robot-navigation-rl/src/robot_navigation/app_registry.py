"""Registry exclusively for native 409-observation/five-action app models."""
from pathlib import Path
from .core import ROOT,read_json
from .partial_agents import ALGORITHMS
from .partial_contract import configuration
from .partial_features_scaled import ARMS,load_scaled_policy

RUNS=ROOT/'runs'

def registry(root=None):
    root=Path(root or RUNS);items=[]
    for file in sorted(root.glob('training/*/manifest.json'),reverse=True):
        try:
            run=read_json(file)
            for name,record in run.get('models',{}).items():
                if name not in ALGORITHMS or record.get('status')!='complete':continue
                path=file.parent/record['checkpoint'];meta=read_json(path.with_suffix('.json'))
                if meta.get('actions')!=5 or meta.get('observation_values')!=409 or meta.get('feature_count')!=412:continue
                arm=next((a for a,v in ARMS.items() if v==meta.get('feature_version')),None)
                if arm is None:continue
                items.append({'id':run['run_id']+'/'+name,'algorithm':name,'path':str(path),'arm':arm,'seed':run['seed'],
                    'actions':record['actions'],'preset':run['preset'],'label':f"{name.upper()} · {record['actions']} actions · seed {run['seed']} · {run['run_id']}"})
        except (OSError,ValueError,KeyError):continue
    return items

def load_model(item):
    path=Path(item['path'])
    if not path.is_file():raise ValueError('The checkpoint is missing. Refresh the registry or train a new model.')
    try:return load_scaled_policy(path,configuration(),item['arm'])
    except Exception as exc:raise ValueError('This checkpoint is incompatible or corrupt. Select a native five-action model or train a new one. Details: '+str(exc)) from exc

def newest_by_algorithm(items=None):
    result={}
    for item in items if items is not None else registry():result.setdefault(item['algorithm'],item)
    return result
