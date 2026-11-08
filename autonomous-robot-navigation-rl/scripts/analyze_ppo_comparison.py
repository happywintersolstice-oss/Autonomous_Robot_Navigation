"""Verify and publish a four-RL development comparison in its new study only."""
import csv
from datetime import datetime,timezone

from robot_navigation.benchmark import load_scenarios
from robot_navigation.core import ROOT,load_task,read_json,write_json
from robot_navigation.four_way import STUDY,FAMILIES,METRICS,summarize,audit_run,figures
from robot_navigation.release import sha


def main():
    task=load_task();study=ROOT/'artifacts/studies'/STUDY
    if read_json(study/'manifest.json')['status']!='complete':
        raise ValueError('All declared PPO runs must complete before comparison')
    if (study/'verification.json').exists():
        raise FileExistsError('This comparison was already analyzed; do not silently overwrite')
    hashes=read_json(study/'historical_hashes.json')
    if any(sha(ROOT/p)!=h for p,h in hashes.items()):
        raise ValueError('Historical comparison inputs changed')
    rows=read_json(ROOT/'artifacts/studies/part-a-v1/run_results.json')+read_json(study/'run_results.json')
    summary=summarize(rows,task.signature,[s['id'] for s in load_scenarios(task,'validation')])
    audits=[audit_run(task,row,ppo_inference=row['algorithm']=='ppo') for row in rows]
    figures(study,summary)
    with (study/'comparison.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['algorithm','seed','run_id','selected_interactions',*METRICS])
        writer.writeheader()
        writer.writerows({'algorithm':r['algorithm'],'seed':r['seed'],'run_id':r['run_id'],
                          'selected_interactions':r['selected_interactions'],**{m:r['validation'][m] for m in METRICS}}
                         for r in rows)
    write_json(study/'summary.json',summary)
    write_json(study/'verification.json',{'status':'passed','audits':audits,
        'historical_files_verified':len(hashes),'historical_inputs_unchanged':True,
        'ppo_training_interactions':1000000,'new_final_test_policy_executions':0,
        'verified_utc':datetime.now(timezone.utc).isoformat(),
        'comparison_scope':'exploratory post-release validation, not a newly blinded final evaluation'})
    print({a:s['metrics']['success_rate'] for a,s in summary['systems'].items()})


if __name__=='__main__':
    main()
