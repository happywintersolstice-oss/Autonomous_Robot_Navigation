"""Measured C0 reports, explicitly separate from new learning experiments."""
import csv
import json

from .core import ROOT,read_json,write_json
from .partial_phase5b_report import table
from .partial_phase5c_coverage import DIRECTIONS


def csv_write(path,rows):
    if not rows: return
    with path.open('w',encoding='utf-8',newline='') as f:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def report(root=ROOT):
    from .partial_phase5c import location,checked_plan
    folder=location(root); plan=checked_plan(root); audit=read_json(folder/'coverage-audit.json')
    if audit['status']!='completed':
        text='# Phase 5C blocked\n\nIMPLEMENTATION_BLOCKED\n\n'+audit.get('exception','Missing evidence')+'\n'
        (root/'docs/partial_phase5c_results.md').write_text(text,encoding='utf-8',newline='\n')
        return {'status':'failed','verdict':'IMPLEMENTATION_BLOCKED'}
    verified=read_json(folder/'replay-audit.json'); overview=[]; reset_rows=[]; direction_rows=[]; near_rows=[]; quality_rows=[]; probe_rows=[]; right_rows=[]; paired=[]
    final_b1=[]; probe_summary=[]; gates=[]; pairs={}
    for family,arms in audit['systems'].items():
        for arm,cps in arms.items():
            for cp in cps:
                node=read_json(folder/cp['coverage_file']); cv=node['coverage']; count=cp['checkpoint']
                for cid,n in cv['reset_cases'].items(): reset_rows.append({'family':family,'arm':arm,'checkpoint':count,'case_id':cid,'episode_starts':n,'denominator':cv['reset_count']})
                for d,b in cv['marginals']['direction'].items():
                    direction_rows.append({'family':family,'arm':arm,'checkpoint':count,'current_direction':d,'decisions':b['decisions'],'denominator':count,
                        'distinct_states':b['distinct_states'],'selected_reducing':b['selected_reducing'],'greedy_reducing':b['greedy_reducing'],
                        'Up':b['actions']['Up'],'Down':b['actions']['Down'],'Left':b['actions']['Left'],'Right':b['actions']['Right'],'Wait':b['actions']['Wait'],
                        'reduced':b['actual_distance_change']['reduced'],'preserved':b['actual_distance_change']['preserved'],'increased':b['actual_distance_change']['increased'],'collisions':b['collisions']})
                for distance,groups in cv['near_goal'].items():
                    for d,b in groups.items(): near_rows.append({'family':family,'arm':arm,'checkpoint':count,'distance':distance,'current_direction':d,
                        'decisions':b['decisions'],'selected_reducing':b['selected_reducing'],'greedy_reducing':b['greedy_reducing'],
                        'wrong_movement_attempts':b['decisions']-b['waits']-b['selected_reducing'],'Wait':b['waits'],'collisions':b['collisions'],
                        'selected_first_argmax':b['selected_first_argmax'],'selected_in_max_set':b['selected_in_max_set']})
                decisions=read_json(folder/cp['decision_file']); pairs[family,arm,count]=decisions
                for q in decisions:
                    m=q['metrics']; d=q['first_decision']
                    row={'family':family,'arm':arm,'checkpoint':count,'panel':q['panel'],'case_id':q['case_id'],'direction':q['direction'],
                        'distance':q['distance'],'outcome':q['outcome'],'actions':m['actions'],'wait_fraction':m['wait_count']/m['actions'],
                        'longest_wait':m['longest_wait_streak'],'revisits':m['repeated_movement_visits'],'oscillations':m['two_cell_oscillation_windows'],
                        'first_argmax':d['first_argmax'],'goal_margin':d['goal_margin'],'scores':json.dumps(d['scores'])}
                    quality_rows.append(row)
                    if q['direction']=='Right': right_rows.append(row)
                ps=read_json(folder/cp['probe_file'])
                for p in ps: probe_rows.append({'family':family,'arm':arm,'checkpoint':count,**{k:v for k,v in p.items() if k not in ('maximizers','reducing_actions')},'scores':json.dumps(p['scores'])})
                if count==6000:
                    near=cv['near_goal']['1']['Right']; b=cv['marginals']['direction']['Right']
                    overview.append([family,arm,cv['reset_count'],*[cv['reset_directions'][d] for d in DIRECTIONS[:4]],
                        b['decisions'],near['decisions'],near['actions']['Right'],cv['goal_rewards_by_arrival_action']['Right'],node['expanded_direction_goals']['Right']])
                    original=sum(q['outcome']=='success' for q in decisions[:4]); expanded=sum(q['outcome']=='success' for q in decisions[4:])
                    final_b1.append([family,arm,original,expanded,*[node['expanded_direction_goals'][d] for d in DIRECTIONS[:4]]])
                    resets=[p for p in ps if p['probe_id'].startswith('reset-')]
                    transformed=[p for p in ps if 'D4_argmax_equivariant' in p and not p['probe_id'].endswith('-0-0')]
                    probe_summary.append([family,arm,sum(p['first_argmax'] in p['reducing_actions'] for p in resets),len(resets),
                        sum(p['D4_argmax_equivariant'] for p in transformed),len(transformed),round(max(p['D4_score_linf'] for p in transformed),6)])
    for (family,arm,count),left in pairs.items():
        if arm!='B0-original': continue
        for a,b in zip(left,pairs[family,'B1-relative-x4',count]):
            if a['case_id']!=b['case_id']: raise ValueError('Historical pairing changed')
            paired.append({'family':family,'checkpoint':count,'case_id':a['case_id'],'panel':a['panel'],
                'B0_outcome':a['outcome'],'B1_outcome':b['outcome'],
                'success_difference':int(b['outcome']=='success')-int(a['outcome']=='success'),
                'actions_difference':b['metrics']['actions']-a['metrics']['actions']})
    for name,rows in [('reset-exposure.csv',reset_rows),('direction-exposure.csv',direction_rows),('near-goal-exposure.csv',near_rows),
                      ('historical-decision-quality.csv',quality_rows),('historical-paired-cases.csv',paired),('counterfactual-scores.csv',probe_rows),('right-goal-cases.csv',right_rows)]: csv_write(folder/name,rows)
    for family,record in audit['sampling_gate']['families'].items():
        for d,r in record['directions'].items():
            gates.append([family,d,r['reset_count'],round(100*r['reset_share'],2),r['near1_decisions'],r['goal_arrivals'],r['expanded_goals'],r['checks']['reset_deficit'],r['checks']['scarce_near_goal_or_arrival_signal'],r['candidate']])
    prefix='../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/'
    text=f'''# Phase 5C: directional coverage diagnosis

**{audit['verdict']}**. Sampling intervention justified:
**{audit['coverage_intervention_justified']}**. New Phase 5C training actions:
**0**. New executed model/environment assessment actions: **0**.
No conditional training protocol, trial ledger, new development panel or model
checkpoint was created. Historical checkpoint scoring is not a new learning run.

## Scope, provenance and inherited facts

Base `{plan['baseline_commit']}`, tree `{plan['baseline_tree']}`; working branch
`{plan['branch']}`. Historical [Phase 5B results](partial_phase5b_results.md) and
[Phase 5A diagnosis](partial_feasibility_diagnosis_v1.md) remain unchanged.
Phase 5B's single-seed B1 DQN expanded direction outcomes were 3/4 Up, 4/4 Down,
3/4 Left and 0/4 Right. Those are inherited outcomes, not Phase 5C performance.
The full historical comparison is retained in the CSVs and decision records.

The [analysis plan](partial_phase5c_protocol.md) fixed numerical gate criteria
and all 40 legal probes before computing coverage. Snapshot only new files;
old evidence is referenced by path, Git tree and SHA256. The
[source/evidence manifest]({prefix}source-and-evidence-manifest.json) lists every
historical file read. No final maps, new reward or feature transformation was used.

## Exact historical denominators

Unique historical training actions reconciled: **{audit['historical_unique_training_actions']}**.
All 24 checkpoint prefixes are reported (0/2k/6k x eight arms). The 2k prefix is
contained in 6k; do not sum overlapping prefixes as independent samples.
Historical assessment replay: **{audit['historical_assessment_actions']} actions /
{audit['historical_assessment_episodes']} episodes**. These are recorded actions
replayed mechanically, not newly selected model actions. Score-only synthetic
probe calls: **{audit['score_probe_calls']}**, with no env.step.

Reset counts include empty autoresets, reported separately from completed
episodes and nonterminal prefixes with actions. States hash the entire 409-value
pre-action float32 vector; position-goal pairs and episode-start case IDs are
different denominators. Marginal tables reuse the same cohort; diagonals occupy
their own disjoint category. An unobserved bin has null rates, not zero probability.
See the [data dictionary]({prefix}data-dictionary.json).

## Final-prefix exposure, both feature arms

'''+table(['Family','Arm','Resets','Start U','Start D','Start L','Start R','Current R decisions','Near1 R decisions','Near1 selected Right','Goal arrivals Right','Expanded Right /4'],overview)+f'''

Training cases already have two examples per direction. Starts are episode
counts; current Right decisions exclude diagonal goals and can follow wandering
from a differently directed reset. Goal arrivals are classified by the final
approach action. Episode-start directions, near-goal choices and rewards therefore
need not have matching proportions. None alone identifies a causal mechanism.

Full reset, action, near-goal and joint coverage for each family/arm/checkpoint is
in [coverage-audit.json]({prefix}coverage-audit.json),
[reset exposure]({prefix}reset-exposure.csv),
[direction exposure]({prefix}direction-exposure.csv), and
[near-goal exposure]({prefix}near-goal-exposure.csv). Joint bins also condition on
quadrant, edge/interior, distance, visits and discovered-map history.

## Preregistered sampling gate, B1 only at 6k

'''+table(['Family','Direction','Resets','Share %','Near1 decisions','Goal arrivals','Expanded goals /4','Reset deficit','Sparse signal','Candidate'],gates)+f'''

The exact criteria and seeded schedule simulations are in the plan and audit.
A sampling candidate needs a reset deficit, failure in that direction, sparse
useful signal and a material planned schedule change simultaneously. The planned
balanced blocks guarantee equal *starts*, not equal transitions, rewards or
visitation. A no-learning schedule simulation cannot guarantee learned coverage.
The gate is engineering triage, not a test of statistical significance.

## Permitted score probes and goal conditioning

'''+table(['Family','Arm','Goal-directed reset argmax','Reset probes','D4 argmax agreement','Nonidentity D4 probes','Maximum D4 score discrepancy'],probe_summary)+f'''

Probe scores are Q values or PPO probabilities, not comparable numerical scales
across families. D4 agreement is descriptive learned symmetry, not a functional
pass requirement. Coordinates, rays, all memory/history grids and action mappings
were transformed consistently; tests compare with real scripted mechanical
histories. Translation probes change absolute location and the legally translated
memory together. They do not isolate a causal location variable.
See [all counterfactual scores]({prefix}counterfactual-scores.csv),
[probe observations]({prefix}probe-manifest.json) and
[Right success/failure comparisons]({prefix}right-goal-cases.csv).

## Historical decision quality and paired feature conditions

'''+table(['Family','Arm','Original goals /4','Expanded goals /16','Up /4','Down /4','Left /4','Right /4'],final_b1)+f'''

All 0/2k/6k original and expanded cases retain first Q/probabilities, margins,
outcomes, actions, Wait, revisits and oscillations in
[historical decision quality]({prefix}historical-decision-quality.csv).
[Exact paired cases]({prefix}historical-paired-cases.csv) compare B0/B1 within a
family on identical cases, not identical training trajectories. Their distributions
differ after actions diverge. No family superiority or new generalization result
is claimed. The old four original and sixteen expanded pairs are already inspected.

## Learning signals, integrity and limitations

Each checkpoint's coverage JSON includes exact goal arrivals by reset case and
pre-action position/goal pair, optimizer counts, recorded epsilon schedules and
stored DQN replay / PPO last-rollout checks. Only the final saved PPO rollout is
available per checkpoint, not its entire rollout history. Positive timeout
bootstrap rewards are not labelled goal rewards. Actual gradient/update records
were read and checked; PPO epochs are not DQN or linear update counts.

Selected actions versus current greedy preference do not reveal the realized
epsilon coin. PPO training is stochastic. SARSA carries an action sampled before
the previous update. Probe sensitivity and coverage associations do not prove
that scaling, reward, sampling or representation caused a failure.

Audit wall: {audit['elapsed_seconds']:.3f} seconds; process resource scope:
`{audit['resources']}`. Independent [replay audit]({prefix}replay-audit.json)
passed, with zero new learning. Historical files remained byte-identical.
Full test counts and original dirty-workspace preservation are recorded in final
delivery verification. Required inherited unit tests can perform tiny mechanical
optimizer checks; they are separate from the zero-action Phase 5C study.

## Decision and one next proposal

'''+('No family/direction meets the complete predeclared case-sampling justification. Stop at diagnosis: do not train the conditional C0/C1 arms. This does not prove that ordering never matters or that every state has adequate coverage. It means the specified episode-balanced scheduler is not a defensible first intervention for this evidence. Propose a separately reviewed A-only goal-conditioning/credit-assignment diagnostic; do not execute another optimization or training experiment under this brief.' if not audit['coverage_intervention_justified'] else
       'A candidate exists; a separate fully tested conditional protocol must be frozen before any training. The C0 audit itself grants no additional experiment.')+f'''

## Reproduce

From the app directory with the recorded installed Python and PYTHONPATH=src:
`python -m robot_navigation.partial_phase5c verify` checks the completed audit
without training; `report` regenerates these tables. A fresh `prepare` refuses an
existing namespace, and `analyze` refuses an existing lock. Do not delete outputs
to obtain a retry. Windows historical paths may require
`git -c core.longpaths=true`. No commit, push, UI change or main merge is performed
without the separate authorization required by the brief.
'''
    (root/'docs/partial_phase5c_results.md').write_text(text,encoding='utf-8',newline='\n')
    diagnosis=f'''# Phase 5C: diagnosis-only decision

**{audit['verdict']}**. New Phase 5C training actions: **0**.
Sampling intervention justified: **{audit['coverage_intervention_justified']}**.

The eight training cases already contain two cases per direction. The predeclared
gate requires a material reset deficit, same-direction failure, sparse useful
near-goal/arrival signal, and a meaningful planned schedule correction together.
It does not assume that equal episode starts produce equal state-action coverage.

'''+table(['Family','Direction','Resets','Share %','Near1 decisions','Goal arrivals','Expanded /4','Reset deficit','Sparse signal','Candidate'],gates)+f'''

See [complete results](partial_phase5c_results.md), the
[frozen analysis plan](partial_phase5c_protocol.md),
[raw coverage audit]({prefix}coverage-audit.json),
[schedule simulation]({prefix}schedule-simulations.json), and
[replay verification]({prefix}replay-audit.json).

This is a single-seed historical association and score-probe audit, not a causal
exclusion of sampling effects. Unobserved states have undefined rates. A
direction-balanced reset schedule cannot directly enforce near-goal actions,
useful updates or transfer across absolute positions/history.

No conditional experiment is run when the gate is false. The one next proposal
is a separately reviewed A-only goal-conditioning/credit-assignment diagnostic;
no optimization or training experiment is executed under this diagnosis.
'''
    (root/'docs/partial_phase5c_diagnosis.md').write_text(diagnosis,encoding='utf-8',newline='\n')
    (folder/'diagnosis-only.md').write_text(diagnosis.replace('../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/','./')
        .replace('(partial_phase5c_results.md)','(../../../docs/partial_phase5c_results.md)')
        .replace('(partial_phase5c_protocol.md)','(../../../docs/partial_phase5c_protocol.md)')
        .replace('(partial_phase5b_results.md)','(../../../docs/partial_phase5b_results.md)')
        .replace('(partial_feasibility_diagnosis_v1.md)','(../../../docs/partial_feasibility_diagnosis_v1.md)'),encoding='utf-8',newline='\n')
    return {'status':'written','verdict':audit['verdict'],'historical_case_rows':len(quality_rows),'paired_rows':len(paired),'probe_rows':len(probe_rows)}
