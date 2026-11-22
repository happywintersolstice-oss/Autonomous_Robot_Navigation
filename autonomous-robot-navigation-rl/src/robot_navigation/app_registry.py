"""Registry exclusively for native 409-observation/five-action app models."""
from pathlib import Path
from .core import ROOT,read_json
from .partial_agents import ALGORITHMS
from .partial_contract import configuration
from .partial_features_scaled import ARMS,load_scaled_policy
from . import product_features,phase3_features

RUNS=ROOT/'runs'

def registry(root=None):
    root=Path(root or RUNS);items=[]
    selection=read_json(root/'selection.json') if (root/'selection.json').is_file() else {}
    best=selection.get('selected',{})
    phase3=read_json(root/'phase3-selection.json') if (root/'phase3-selection.json').is_file() else {};phase3_best=phase3.get('selected',{})
    for file in sorted(root.glob('training/*/manifest.json'),reverse=True):
        try:
            run=read_json(file)
            if run.get('status')!='complete':continue
            for name,record in run.get('models',{}).items():
                if name not in ALGORITHMS or record.get('status')!='complete':continue
                path=file.parent/record['checkpoint'];meta=read_json(path.with_suffix('.json'))
                if meta.get('actions')!=5 or meta.get('observation_values')!=409:continue
                arm=next((a for a,v in ARMS.items() if v==meta.get('feature_version')),None)
                if meta.get('feature_version')==phase3_features.VERSION and meta.get('feature_count')==phase3_features.FEATURE_COUNT and meta.get('agent_state_values')==phase3_features.STATE:arm='product-v3'
                elif meta.get('feature_version')==product_features.VERSION and meta.get('feature_count')==product_features.FEATURE_COUNT:arm='product-v2'
                elif meta.get('feature_count')!=412:continue
                if arm is None:continue
                item_id=run['run_id']+'/'+name;is_best=best.get(name,{}).get('id')==item_id and best[name].get('checkpoint_sha256')==meta['checkpoint_sha256']
                is_phase3=phase3_best.get(name,{}).get('id')==item_id and phase3_best[name].get('checkpoint_sha256')==meta['checkpoint_sha256']
                scope='Phase 3 selected' if is_phase3 and not is_best else 'Phase 2 retained by Phase 3' if is_phase3 else 'Phase 2 baseline' if is_best else 'Phase 3 candidate' if run.get('product_phase')==3 else 'Phase 2 candidate'
                state='Best on development' if is_best else 'Trained' if record['actions']>=10000 else 'Starter'
                items.append({'id':run['run_id']+'/'+name,'algorithm':name,'path':str(path),'arm':arm,'seed':run['seed'],
                    'product_phase':run.get('product_phase',2),'phase3_selected':is_phase3,'phase2_baseline':is_best,'selection_scope':scope,'agent_state_values':meta.get('agent_state_values',409),'phase3_study_id':run.get('phase3_study_id'),'candidate':run.get('candidate'),'actions':record['actions'],'preset':run['preset'],'status':state,'checkpoint_sha256':meta['checkpoint_sha256'],'feature_version':meta['feature_version'],'shaping':run.get('shaping','none'),'q_stability':run.get('q_stability','off') if name=='q-learning' else None,'training_case_hash':run.get('case_hash'),'learning_evidence':record.get('learning_evidence',{}),'label':f"{scope} | {state} | {name.upper()} · {record['actions']} actions · seed {run['seed']} · {run['run_id']}"})
        except (OSError,ValueError,KeyError):continue
    return sorted(items,key=lambda i:i['status']!='Best on development')

def load_model(item):
    path=Path(item['path'])
    if not path.is_file():raise ValueError('The checkpoint is missing. Refresh the registry or train a new model.')
    try:return phase3_features.load_policy(path,configuration()) if item['arm']=='product-v3' else product_features.load_policy(path,configuration()) if item['arm']=='product-v2' else load_scaled_policy(path,configuration(),item['arm'])
    except Exception as exc:raise ValueError('This checkpoint is incompatible or corrupt. Select a native five-action model or train a new one. Details: '+str(exc)) from exc

def newest_by_algorithm(items=None):
    result={}
    for item in items if items is not None else registry():result.setdefault(item['algorithm'],item)
    return result


def preferred_by_algorithm(items=None):
    """Development-selected checkpoints first; newest is only the unvalidated fallback."""
    return newest_by_algorithm(items)


def phase3_selected(root=None):
    root=Path(root or RUNS);file=root/'phase3-selection.json'
    if not file.is_file():return {}
    chosen=read_json(file)['selected'];available={i['id']:i for i in registry(root)};result={}
    for a,record in chosen.items():
        item=available.get(record['id'])
        if item is None or item['checkpoint_sha256']!=record['checkpoint_sha256']:raise ValueError('Phase3 selection points to missing or incompatible checkpoint')
        result[a]=item
    return result
