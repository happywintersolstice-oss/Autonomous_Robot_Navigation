"""Staged five-seed extension using frozen M9 settings and two added seeds."""
from copy import deepcopy
import ast

from .core import read_json
from .neural_tuning import saved_summary as original_summary, summarize
from .release import sha

STUDY = 'neural-confirmation-five-v2'
ENGINE_FILES = ('agents.py', 'benchmark.py', 'core.py', 'dqn.py', 'episodes.py',
                'observation_encoding.py', 'part_a.py', 'ppo.py', 'ppo_config.py',
                'refinement.py', 'training_traces.py', 'workflow.py')


def core_learning_structure(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    tree.body=[node for node in tree.body
               if not (isinstance(node,ast.FunctionDef) and node.name in ('read_json','write_json'))
               and not (isinstance(node,ast.Import) and {a.name for a in node.names}<={'errno','os','tempfile','time'})]
    return ast.dump(tree,include_attributes=False)


def prepare(plan, root):
    if (plan['suite_id'] != STUDY or plan['source_study'] != 'neural-tuning-v2'
            or plan['original_seeds'] != [7307, 7411, 7523]
            or plan['additional_seeds'] != [7607, 7717]
            or plan['algorithms'] != ['dqn', 'ppo'] or plan['interactions'] != 200000
            or plan['workers'] != 2 or plan['selected_candidates'] != {'dqn':'c01', 'ppo':'c01'}
            or plan['recovery_source_study']!='neural-confirmation-five-v1'
            or set(plan['reused_completed_runs'])!={f'm9-{stage}-dqn-seed{seed}' for stage in ('control','tuned') for seed in (7607,7717)}
            or plan['limits'] != {'max_new_runs':4,'max_new_training_interactions':800000,
                                  'run_wall_seconds':900,'suite_wall_seconds':1800,'new_artifact_bytes':1073741824}):
        raise ValueError('Extension differs from the declared frozen-settings matrix')
    prior = original_summary(root)
    if prior is None:
        raise ValueError('Original verified study is required')
    source_study = root/'artifacts/studies'/plan['source_study']
    selected = read_json(source_study/'selected_configurations.json')['selected']
    if selected != plan['selected_candidates']:
        raise ValueError('Selected settings differ from the original frozen choices')
    previous = []
    jobs = []
    engine_hashes = {}
    for family in plan['algorithms']:
        system = prior['systems'][family]
        if system['seeds'] != plan['original_seeds'] or system['baseline_retained']:
            raise ValueError('Original paired seed matrix differs')
        for stage in ('control', 'tuned'):
            records = system['conditions'][stage]['seed_results']
            previous.extend(records)
            reference = root/'artifacts/runs'/records[0]['run_id']
            for name in ENGINE_FILES:
                relative = f'src/robot_navigation/{name}'
                value = sha(reference/'source'/relative)
                current=sha(root/relative)
                io_only=(name=='core.py' and core_learning_structure(reference/'source'/relative)==core_learning_structure(root/relative))
                if current!=value and not io_only:
                    raise ValueError(f'Training/evaluation engine changed since M9. {name}')
                engine_hashes[relative] = {'original_sha256':value,'current_sha256':current,
                                           'json_io_only_change':current!=value}
            config = read_json(reference/'configuration.json')['training']
            for seed in plan['additional_seeds']:
                addition = deepcopy(config)
                identity=f'm9-{stage}-{family}-seed{seed}'+('-attempt2' if family=='ppo' else '')
                addition.update(seed=seed, run_id=identity)
                jobs.append({'config':addition, 'candidate':'c00' if stage=='control' else selected[family]})
    aggregate_plan = read_json(source_study/'plan.json')
    aggregate_plan['confirmation_seeds'] = sorted(plan['original_seeds']+plan['additional_seeds'])
    aggregate_plan['scope'] = 'five-seed staged development-validation extension requested after the original three-seed results'
    return previous, jobs, aggregate_plan, engine_hashes


def combined_summary(previous, additions, plan, aggregate_plan, signature):
    expected = {(family, stage, seed) for family in plan['algorithms']
                for stage in ('control', 'tuned') for seed in plan['additional_seeds']}
    actual = [(row['algorithm'], row['stage'], row['seed']) for row in additions]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Additional-seed matrix is incomplete or duplicated')
    for row in additions:
        candidate = 'c00' if row['stage']=='control' else plan['selected_candidates'][row['algorithm']]
        if row['candidate'] != candidate:
            raise ValueError('Additional seed used a different selected candidate')
    summary = summarize(previous+additions, plan['selected_candidates'], aggregate_plan, signature)
    added_plan = {**aggregate_plan, 'confirmation_seeds':plan['additional_seeds']}
    added = summarize(additions, plan['selected_candidates'], added_plan, signature)
    summary.update(original_study=plan['source_study'], original_seeds=plan['original_seeds'],
                   additional_seeds=plan['additional_seeds'], added_seed_systems=added['systems'],
                   expansion_declared_after_original_results=True,
                   new_training_interactions=sum(row['pilot']['training_interactions'] for row in additions),
                   combined_confirmation_training_interactions=sum(row['pilot']['training_interactions'] for row in previous+additions))
    return summary


def saved_summary(root):
    study = root/'artifacts/studies'/STUDY
    if not (study/'verification.json').is_file():
        return None
    verification = read_json(study/'verification.json')
    expected = {'summary.json', 'paired_neural_tuning.png', 'neural_tuning_learning.png', 'plan.json'}
    if (verification['status'] != 'passed' or set(verification['output_hashes']) != expected
            or any(sha(study/name) != value for name, value in verification['output_hashes'].items())):
        raise ValueError('Five-seed publication verification failed')
    return read_json(study/'summary.json')
