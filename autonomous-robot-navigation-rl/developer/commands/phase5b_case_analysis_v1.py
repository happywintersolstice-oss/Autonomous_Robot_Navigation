"""Post-hoc report arithmetic only: no env.step, model loading or learning."""
import argparse
from collections import Counter
from pathlib import Path

from robot_navigation.core import ROOT, read_json, write_json
from robot_navigation.release import sha
from robot_navigation.partial_commands import read_records
from robot_navigation.partial_phase5b_fixtures import strata
from robot_navigation.partial_phase5b_report import paired_rows

VERSION = 'phase5b-posthoc-case-analysis-v1'


def calculate(folder):
    summary=read_json(folder/'summary.json')
    paired=paired_rows(folder,summary)
    result={'version':VERSION,'new_training_actions':0,'new_assessment_actions':0,
        'scope':'Post-hoc case and path arithmetic; not a new screening criterion or a learned-policy run',
        'families':{},'arms':{}}
    for algorithm,arms in summary['systems'].items():
        result['families'][algorithm]={}
        for panel in ('original','expanded'):
            selected=[r for r in paired if r['algorithm']==algorithm and r['checkpoint']==6000 and r['panel']==panel]
            result['families'][algorithm][panel]=dict(Counter(
                ('goal' if r['B0_outcome']=='success' else 'failure')+' -> '+
                ('goal' if r['B1_outcome']=='success' else 'failure') for r in selected))
        for arm,run in arms.items():
            cp=run['checkpoints'][-1]
            records=read_records(folder/'training-and-evaluation-evidence'/algorithm/arm/'06000/evaluation.jsonl.gz')
            panels={}
            for name,part in [('original',records[:4]),('expanded',records[4:])]:
                good=[r for r in part if r['metrics']['outcome']=='success']
                minimum=[sum(abs(x-y) for x,y in zip(r['scenario']['start'],r['scenario']['goal'])) for r in good]
                lengths=[len(r['transitions']) for r in good]
                diagnostics=cp['panels'][name]['first_decisions']
                failures=[]
                for record,d in zip(part,diagnostics):
                    if record['metrics']['outcome']=='success': continue
                    failures.append({'case_id':record['scenario']['id'],**strata(record['scenario']),
                        'start':record['scenario']['start'],'goal':record['scenario']['goal'],
                        'outcome':record['metrics']['outcome'],'actions':len(record['transitions']),
                        'waits':record['metrics']['wait_count'],
                        'first_goal_directed':d['first_step_goal_directed'],
                        'first_goal_margin':d['goal_direction_margin']})
                panels[name]={'success_count':len(good),'successful_action_lengths':lengths,
                    'successful_minimum_axis_steps':minimum,
                    'success_only_total_action_excess':sum(lengths)-sum(minimum),
                    'reference_scope':'Evaluator-only open-geometry shortest-action arithmetic, never supplied to a learner',
                    'failures':failures}
            result['arms'][algorithm+'/'+arm]=panels
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    folder=ROOT/'artifacts/partial-dynamic-navigation-v1/phase5b-a-only-v1'
    target=folder/'derived-analysis'; path=target/'case-outcomes-v1.json'
    data=calculate(folder)
    inputs={p:sha(folder/p) for p in ('manifest.json','summary.json','outputs.json','replay-audit.json')}
    manifest={'version':VERSION,'source_sha256':sha(Path(__file__)),'inputs':inputs}
    if args.verify:
        if read_json(path)!=data or read_json(target/'case-outcomes-manifest-v1.json')!=manifest:
            raise ValueError('Post-hoc case analysis changed')
        print('Verified post-hoc arithmetic; zero new training/assessment actions.')
    else:
        if path.exists(): raise FileExistsError('Preserve existing derived analysis; no overwrite')
        target.mkdir(exist_ok=True)
        write_json(path,data); write_json(target/'case-outcomes-manifest-v1.json',manifest)
        print('Saved post-hoc case arithmetic; zero new training/assessment actions.')


if __name__=='__main__': main()
