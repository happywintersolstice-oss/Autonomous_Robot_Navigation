"""Audit all paired input-study evidence before publishing dashboard results."""
import csv
from time import perf_counter

from robot_navigation.core import ROOT,load_task,read_json,write_json
from robot_navigation.release import sha
from robot_navigation.representation import STUDY,METRICS,summarize,audit,add_groups,figures


def main():
    started=perf_counter();task=load_task();study=ROOT/'artifacts/studies'/STUDY
    manifest=read_json(study/'manifest.json');plan=read_json(study/'plan.json')
    if manifest['status']!='complete':
        raise ValueError('All ten representation runs must complete first')
    if (study/'verification.json').exists():
        raise FileExistsError('Representation analysis already published; no silent overwrite')
    hashes=read_json(study/'historical_hashes.json')
    if any(sha(ROOT/path)!=value for path,value in hashes.items()):
        raise ValueError('Historical artifacts/data changed')
    rows=read_json(study/'run_results.json')
    summary=summarize(rows,task.signature)
    audits=[audit(task,row,manifest,plan) for row in rows]
    add_groups(summary,task)
    for variant,system in summary['systems'].items():
        system['online_parameter_count']=sorted({r['pilot']['online_parameter_count'] for r in system['seed_results']})
        system['input_dimensions']=sorted({r['pilot']['input_dimensions'] for r in system['seed_results']})
    figures(study,summary)
    with (study/'paired_differences.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(summary['paired_seed_differences'][0]));writer.writeheader();writer.writerows(summary['paired_seed_differences'])
    write_json(study/'summary.json',summary)
    outputs={name:sha(study/name) for name in ('summary.json','paired_differences.csv','paired_representation.png','representation_learning.png')}
    write_json(study/'verification.json',{'status':'passed','audits':audits,'historical_files_verified':len(hashes),
        'historical_inputs_unchanged':True,'policy_actions_replayed':sum(r['policy_actions_replayed'] for r in audits),
        'training_trace_actions_replayed':sum(r['training_traces']['actions'] for r in audits),
        'new_final_test_policy_executions':0,'analysis_seconds':perf_counter()-started,'output_hashes':outputs})
    print({k:system['metrics']['success_rate'] for k,system in summary['systems'].items()},flush=True)


if __name__=='__main__':
    main()
