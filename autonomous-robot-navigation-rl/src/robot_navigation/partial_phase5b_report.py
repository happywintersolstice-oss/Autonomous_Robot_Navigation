"""Reports derived from recorded measurements; missing runs are never filled in."""
import csv
import json

import numpy as np

from .core import ROOT, read_json, write_json
from .release import sha
from .partial_commands import read_records
from .partial_agents import ALGORITHMS
from .partial_contract import configuration
from .partial_features_scaled import ARMS, load_scaled_policy
from .partial_pilot_diagnostics import preserve_random_state
from .partial_phase5b_diagnostics import decision_diagnostic
from .partial_phase5b_fixtures import strata


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def preflight_report(root=ROOT):
    from .partial_phase5b import junit, diagnosis_path
    z=read_json(diagnosis_path(root)/'zero-learning-audit.json')
    tests=junit(root/'.runtime/phase5b-implementation-tests.xml')
    rows=[]
    for name,s in z['systems'].items():
        rows.append([name,round(s['feature_norm']['mean'],4),
            f"{100*s['relative_goal_energy_fraction']['mean']:.4f}%",s['training_goals'],
            s.get('exact_update_reconstruction',{}).get('checkpoint_weights_exact','not applicable')])
    text='''# Phase 5B: zero-learning preflight diagnosis

This report was written before the registered Phase 5B trial. It reads existing
[Phase 5A results](partial_feasibility_results.md) and
[Phase 5A diagnosis](partial_feasibility_diagnosis_v1.md) without modifying them.
It is not a Phase 5B learning outcome or evidence of generalization.

## Observed historical evidence

'''+table(['Family','Mean 412-feature norm','Relative-goal energy fraction','Old training goals','Exact linear checkpoint reconstruction'],rows)+f'''

The historical source/checkpoint pins passed. Linear arithmetic/RNG replay
exactly reproduced the archived 2k, 6k and 10k weights. All five training actions
were sampled. All four old A reset decisions at 0/2k/6k/10k have saved values,
margins and component breakdowns. Historical optimizer moments are finite;
inference checkpoint files do not retain live gradients, so no historical gradient
measurement is invented. Live Phase 5B optimizer steps will be instrumented.

The full implementation gate passed **{tests['tests']} tests**, with zero failures,
errors or skips. Tests cover cardinal rotations, action/ray ordering, worked
Bellman targets, terminal/truncated bootstrapping and carried SARSA actions,
numpy/Torch parity, source boundaries, hidden-history equivalence, strict arm
loading, paired cases, RNG/optimizer continuity and injected failures.

## Worked targets and action alignment

Action IDs remain 0 Up, 1 Down, 2 Left, 3 Right, 4 Wait. Sensor rays remain
N/S/E/W; their third component is East, not action 2. Greedy ties choose the
first argmax; linear exploratory ties are sampled uniformly. The analytic
relative-goal witness chooses the four correct initial actions. It is a capacity
witness with no learning and no executed witness policy, never an initializer.

'''+table(['Family','Synthetic transition','Analytic target','Real updated score'],
         [[r['algorithm'],r['case'],round(r['target'],6),round(r['actual_updated_score'],6)] for r in z['worked_bellman_targets']])+'''

Synthetic updates use alpha=1 to expose targets directly. A true goal/collision
does not bootstrap. External truncation keeps its real continuing observation
and bootstraps. The two-step backup is -0.01 + 0.99 * 9.99 = 9.8801. These tests
demonstrate implementation semantics, not that a learner acquires a good policy.

'''+table(['Family','Second transition','Second target','First target'],
         [[r['algorithm'],r['second_transition'],round(r['second_target'],6),round(r['first_target'],6)]
          for r in z['worked_two_step_targets']])+'''

The full two-step synthetic check uses independent permitted feature vectors.
Q-learning takes a zero-valued alternative after the collision backup; SARSA
uses the declared sampled action. Both match the actual update function.

## Hypotheses and preflight decision

No deterministic test has established a baseline interface or optimization bug.
The original A goal directions are representable. Relative-goal energy is small
compared with dense memory channels, so a predeclared factor-four scale is a
plausible signal/conditioning probe. Feature energy alone is not a causal
explanation: normalized updates, state coverage, optimizer dynamics, the finite
budget and start/goal distribution remain alternatives. Rewards are unchanged;
dynamic events cannot explain A failures because A has none.

Proceed only to the frozen two-arm A-only protocol after G2 registration. Do not
enlarge features, change rewards/learning rates, train obstacles or select a new
multiplier. The successful and unsuccessful permitted-information examples,
training-only feature distributions, reset/exploration schedules and exact update
records are archived under the new study's `zero-learning-diagnosis/` directory.
'''
    (root/'docs/partial_phase5b_diagnosis.md').write_text(text,encoding='utf-8',newline='\n')
    return {'status':'written','tests':tests['tests']}


def write_csv(path,rows):
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def paired_rows(folder,summary):
    rows=[]
    for algorithm,arms in summary['systems'].items():
        for count in (0,2000,6000):
            paths=[folder/'training-and-evaluation-evidence'/algorithm/arm/f'{count:05}/evaluation.jsonl.gz' for arm in ARMS]
            if not all(p.exists() for p in paths): continue
            a,b=map(read_records,paths)
            if [r['scenario']['id'] for r in a]!=[r['scenario']['id'] for r in b]:
                raise ValueError('Unpaired cases in report')
            for index,(left,right) in enumerate(zip(a,b)):
                lm,rm=left['metrics'],right['metrics']
                rows.append({'algorithm':algorithm,'checkpoint':count,'panel':'original' if index<4 else 'expanded',
                    'case_id':left['scenario']['id'],**strata(left['scenario']),
                    'B0_outcome':lm['outcome'],'B1_outcome':rm['outcome'],
                    'success_difference':int(rm['outcome']=='success')-int(lm['outcome']=='success'),
                    'B0_steps':lm['actions'],'B1_steps':rm['actions'],
                    'B0_waits':lm['wait_count'],'B1_waits':rm['wait_count'],
                    'B0_revisits':lm['repeated_movement_visits'],'B1_revisits':rm['repeated_movement_visits'],
                    'B0_oscillations':lm['two_cell_oscillation_windows'],'B1_oscillations':rm['two_cell_oscillation_windows']})
    return rows


def report(root=ROOT):
    from .partial_phase5b import location
    folder=location(root); summary=read_json(folder/'summary.json'); manifest=read_json(folder/'manifest.json')
    verdict=read_json(folder/'final-verdict.json') if (folder/'final-verdict.json').exists() else {
        'verdict':summary.get('verdict','INCOMPLETE_BUDGET_OR_EXECUTION'),'families_passed':None,'screens':{}}
    audit=read_json(folder/'replay-audit.json') if (folder/'replay-audit.json').exists() else None
    rows=[]; train_rows=[]; example_rows=[]; examples={}
    for algorithm,arms in summary['systems'].items():
        for arm,run_result in arms.items():
            for cp in run_result['checkpoints']:
                for panel,metrics in cp['panels'].items():
                    rows.append({'algorithm':algorithm,'arm':arm,'actions':cp['actions'],'panel':panel,
                        'success':metrics['counts']['success'],'collision':metrics['counts']['collision'],
                        'timeout':metrics['counts']['timeout'],'denominator':metrics['completed_episodes'],
                        'mean_steps':round(metrics['mean_steps'],3),'median_steps':metrics['median_steps'],
                        'wait_percent':round(100*metrics['wait_fraction'],3),
                        'longest_wait':metrics['longest_wait_streak'],'revisits':metrics['repeated_visits'],
                        'oscillations':metrics['oscillation_windows'],'first_goal_directed':metrics['goal_directed_initial_count']})
            final=run_result['checkpoints'][-1] if run_result['checkpoints'] else None
            if not final: continue
            train=final['training']['overall'] if final['training'] else None
            if train:
                train_rows.append([algorithm,arm,run_result['actual_training_actions'],train['counts']['success'],
                    train['counts']['collision'],train['counts']['timeout'],train['incomplete_prefixes'],
                    round(100*train['wait_fraction'],2),run_result.get('observed_updates','unfinished'),
                    round(run_result['training_seconds_including_recording'],3)])
            target=folder/'training-and-evaluation-evidence'/algorithm/arm/f'{final["actions"]:05}'
            records=read_records(target/'evaluation.jsonl.gz')
            with preserve_random_state():
                policy=load_scaled_policy(target/final['checkpoint'],configuration(root),arm)
                chosen=[]
                for outcome in ('success','collision','timeout'):
                    item=next((r for r in records if r['metrics']['outcome']==outcome),None)
                    if item:
                        # Include a sustained Wait/oscillation decision as well as first/last.
                        indices={0,len(item['transitions'])-1}
                        indices.update(i for i,t in enumerate(item['transitions']) if t['action']==4 and i<12)
                        detail={'case_id':item['scenario']['id'],'outcome':outcome,'checkpoint':final['actions'],
                            'actions':len(item['transitions']),'diagnostics':[{'index':i,**decision_diagnostic(policy,item['transitions'][i]['observation'])} for i in sorted(indices)],
                            'source':(target/'evaluation.jsonl.gz').relative_to(folder).as_posix(),
                            'scope':'Only pre-action permitted observations/scores; outcome is retrospective evaluator output.'}
                        chosen.append(detail)
                        d=detail['diagnostics'][0]; obs=np.array(d['observation_before_action'])
                        example_rows.append([algorithm,arm,outcome,detail['actions'],np.rint(obs[:4]*9).astype(int).tolist(),
                                             d['chosen_action'],round(d['goal_direction_margin'],5)])
                examples[f'{algorithm}/{arm}']=chosen
    pairs=paired_rows(folder,summary)
    write_csv(folder/'learning-curves.csv',rows); write_csv(folder/'paired-results.csv',pairs)
    write_json(folder/'trajectory-examples.json',examples)
    path_prefix='../artifacts/partial-dynamic-navigation-v1/phase5b-a-only-v1/'
    text=f'''# Phase 5B: A-only navigation learning diagnosis

## 1. Executive verdict and scope

**{verdict['verdict']}**. Families passing the complete predeclared screen:
**{verdict['families_passed']} / 4**. Trial status: **{summary['status']}**.
This is one seed (23021), two feature arms and fixed development pairs on familiar
open geometry. It is not full Phase 5, unseen-layout generalization, statistical
evidence across seeds or a ranking of algorithm families. No obstacle/dynamic
training, new final/test evaluation, UI work or main merge was performed.

## 2. Phase 5A facts and new measurements

The archived [Phase 5A results](partial_feasibility_results.md) and
[diagnosis](partial_feasibility_diagnosis_v1.md) remain unchanged: final original
A successes were Q-learning 1/4, SARSA 0/4, DQN 1/4 and PPO 1/4, all failing that
study's improvement gate. These old seed-23001 runs are not matched B0 controls.
The tables below are new, separately registered seed-23021 measurements.

## 3. Zero-learning audit

See the [preflight diagnosis](partial_phase5b_diagnosis.md),
[raw zero-learning audit]({path_prefix}zero-learning-diagnosis/zero-learning-audit.json)
and [pretraining gates]({path_prefix}pretraining-gates.json). Old linear TD/RNG
reconstruction exactly matched all archived weights. Real terminal/truncated
targets, rotations, feature ordering, information boundaries and parity passed.
Relative-goal energy is small; that is a hypothesis, not a causal finding.

## 4. Registered design

The [protocol](partial_phase5b_protocol.md) and
[sealed manifest]({path_prefix}manifest.json) fix B0 original 412 features versus
B1 with only the final two coordinates multiplied by four. Raw observations,
task/rewards/model settings and eight original training pairs are unchanged.
Original A has four inspected development pairs; expanded A has 16 registered
additional pairs: balanced directions, one/two-step distances and quadrants,
with no duplicate training/evaluation pairs. No outcome-dependent case selection.
Linear methods are semi-gradient approximations, not tabular. The old 10k linear
epsilon horizon remains fixed; DQN keeps its declared continuation schedule.

## 5. Every registered checkpoint and panel

'''+table(list(rows[0]) if rows else ['status'],[list(r.values()) for r in rows] if rows else [['No completed assessment']])+'''

Denominators are exact. No earlier checkpoint replaces the final 6k boundary.
Full per-direction/distance/quadrant/edge-band breakdowns, return and episode
length distributions and initial action values are in each checkpoint summary.

### Training, kept separate from deterministic assessment

'''+table(['Family','Arm','Actions','Goals','Collisions','Timeouts','Incomplete prefixes','Wait %','Observed updates','Training seconds'],train_rows)+'''

Linear update counts, DQN gradient steps and PPO optimizer steps/epochs/rollouts
are distinct compute units. Training seconds include action recording and live
instrumentation; they exclude assessment and checkpoint serialization.

## 6. Permitted-information success and failure examples

'''+table(['Family','Arm','Outcome','Actions','Robot/goal coordinates','Initial action','Goal-direction margin'],example_rows)+f'''

See [trajectory examples]({path_prefix}trajectory-examples.json) and raw traces
for the actual 409-value pre-action observations, scores/probabilities, rays,
memory and retained failures. Margins are Q differences or PPO probability
differences, not comparable scales across families. No true-map, scenario label,
remaining event duration or privileged optimal action is a learner input.

## 7. Within-family paired differences and fixed screen

'''+table(['Family','Original B1-B0','Expanded B1-B0','Passed complete screen','Individual criteria'],
        [[a,v.get('original_difference'),v.get('expanded_difference'),v.get('passed'),v.get('checks')]
         for a,v in verdict.get('screens',{}).items()])+f'''

The screen requires B1 original >=3/4, +1 over its own untrained checkpoint and
matched B0 final; expanded >=12/16 and +4 over B0 final; all integrity gates pass.
The [exact per-case paired table]({path_prefix}paired-results.csv) retains
success-to-failure and failure-to-success changes, collisions, Wait and loops.
One paired seed cannot establish a cross-seed effect. Finite coverage, feature
conditioning, normalized update magnitudes, optimization and familiar-pair
transfer remain alternatives. Dynamic composition is irrelevant to open A.

## 8. Runtime, resources and independent audit

Training reservations: **{summary['budget']['training_total']} / 48,000**.
Assessment/control reservations: **{summary['budget']['evaluation_reserved']} / 100,000**.
Primary wall: **{summary['budget']['elapsed_seconds']:.3f} / 1,800 seconds**.
Each arm cap: 300 seconds. Resources: `{summary['resources']}`.
These process measurements include recording/evaluation, not model-only costs.
Post-run audits/regressions are outside the primary clock and add no learning.

Audit: `{audit}`.

The [budget journal]({path_prefix}training-budget-ledger.jsonl),
[checkpoint hashes]({path_prefix}checkpoint-hashes.json),
[replay audit]({path_prefix}replay-audit.json),
[raw summary]({path_prefix}summary.json) and
[final verdict]({path_prefix}final-verdict.json) preserve exact accounting.
Compressed continuation state includes trainer RNG/optimizer/buffers; inference
checkpoints load strictly by feature version. Registration/execution refuse a
second attempt. All historical pins were rechecked after execution.

## 9. Gate decision and one next step

See [gate results]({path_prefix}gate-results.json). Engineering success is not
learning success. '''+('Propose independent seed replication of the same A-only protocol, requiring new authorization before any further training.' if verdict['verdict']=='SCALING_HYPOTHESIS_SUPPORTED_IN_ONE_SEED' else
                         'Remain at A and propose one separately registered optimization/coverage hypothesis for review before any further training.')+f'''
No B/C/D/E, full Phase 5 or Phase 6 progression is authorized.

## Provenance and reproduction

Registered branch: `{manifest['git']['branch']}`; reviewed base/head at registration:
`{manifest['git']['head']}`. Source/config/case/dependency hashes and exact interpreter
are in the manifest; protocol SHA256 `{sha(folder/'manifest.json')}`.
From the application root with the recorded interpreter and PYTHONPATH=src:

```powershell
python -m robot_navigation.partial_phase5b diagnose
python -m robot_navigation.partial_phase5b validate
python -m robot_navigation.partial_commands validate
python -m robot_navigation.partial_phase5b register
python -m robot_navigation.partial_phase5b run
python -m robot_navigation.partial_phase5b validate-postrun
python -m robot_navigation.partial_phase5b audit
python -m robot_navigation.partial_phase5b report
```

Do not rerun training into this existing namespace. Reproduction requires a fresh
separately authorized attempt/version and the same declared ceilings. Final commit
and push status are recorded separately in publication metadata, not retroactively
inserted into the frozen pre-run manifest.
'''
    (root/'docs/partial_phase5b_results.md').write_text(text,encoding='utf-8',newline='\n')
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,4,figsize=(15,6),sharex=True)
        for col,algorithm in enumerate(ALGORITHMS):
            for row,panel in enumerate(('original','expanded')):
                ax=axes[row,col]
                for arm,color in zip(ARMS,('#263645','#d96b23')):
                    points=[r for r in rows if r['algorithm']==algorithm and r['arm']==arm and r['panel']==panel]
                    ax.plot([r['actions'] for r in points],[r['success'] for r in points],marker='o',label=arm,color=color)
                denominator=4 if panel=='original' else 16
                ax.axhline(3 if panel=='original' else 12,color='#999999',linestyle=':',linewidth=1)
                ax.set(title=f'{algorithm} - {panel}',ylim=(-.3,denominator+.3),xticks=[0,2000,6000])
                if col==0: ax.set_ylabel(f'Goals / {denominator}')
                if row==1: ax.set_xlabel('Training actions')
                ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        fig.suptitle('Phase 5B: one paired seed, familiar-geometry development cases; no uncertainty estimate')
        fig.tight_layout(); fig.savefig(folder/'learning-curves.png',dpi=140); plt.close(fig)
        with (root/'docs/partial_phase5b_results.md').open('a',encoding='utf-8') as f:
            f.write(f'\n![All retained checkpoint success counts]({path_prefix}learning-curves.png)\n')
    return {'status':'written','checkpoint_panel_rows':len(rows),'paired_case_rows':len(pairs),'verdict':verdict['verdict']}
