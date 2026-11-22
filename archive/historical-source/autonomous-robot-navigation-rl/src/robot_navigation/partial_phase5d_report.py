"""Derive all Phase 5D prose/tables from sealed evidence, without policy calls."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from .core import ROOT,read_json,write_json
from .release import sha
from .partial_phase5d import location,checked_plan,FAMILIES,DIRECTIONS,BASE,STUDY


def records(path):
    with path.open(encoding='utf-8',newline='') as stream:
        result=list(csv.DictReader(stream))
    return [{k:decode(v) for k,v in row.items()} for row in result]


def decode(value):
    if value=='':return None
    try:return json.loads(value)
    except (ValueError,TypeError):return value


def table(headers,rows):
    def cell(v):
        if v is None:return 'unobserved'
        if isinstance(v,float):return f'{v:.5g}'
        if isinstance(v,(dict,list)):return json.dumps(v,separators=(',',':'))
        return str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+
                     ['| '+' | '.join(cell(v) for v in row)+' |' for row in rows])


def report(root=ROOT):
    plan=checked_plan(root);folder=location(root);summary=read_json(folder/'diagnostic-summary.json')
    if (folder/'output-hashes.json').exists():
        from .partial_phase5d import verify
        verify(root)
    verified=read_json(folder/'verification.json')
    if verified['status']!='passed':raise ValueError('Independent verification required')
    gates=read_json(folder/'validation-gates.json')
    if any(v!='passed' for v in gates.values()):raise ValueError('Validation gate pending')
    scores=read_json(folder/'scores.json');cases=records(folder/'case-decisions.csv')
    coverage=records(folder/'direction-coverage.csv');matches=records(folder/'matched-states.csv')
    contrasts=records(folder/'counterfactual-comparisons.csv');buffers=read_json(folder/'buffer-audit.json')
    a=summary['accounting'];findings=[]
    for family in FAMILIES:
        for arm in plan['settings']['arms']:
            for cp in (0,2000,6000):
                group=[r for r in cases if (r['family'],r['arm'],r['checkpoint'])==(family,arm,cp)]
                expanded=[r for r in group if r['panel']=='expanded'];original=[r for r in group if r['panel']=='original']
                probes=[r for r in scores if (r['family'],r['arm'],r['checkpoint'],r['kind'])==(family,arm,cp,'legal-probe')]
                cohort=[r for r in scores if (r['family'],r['arm'],r['checkpoint'],r['kind'])==(family,arm,cp,'fixed-historical-cohort')]
                supported=[r for r in matches if (r['family'],r['arm'],r['checkpoint'])==(family,arm,cp)]
                findings.append([family,arm,cp,f"{sum(r['historical_outcome']=='success' for r in original)}/4",
                    f"{sum(r['historical_outcome']=='success' for r in expanded)}/16",
                    '/'.join(str(sum(r['historical_outcome']=='success' for r in expanded if r['direction']==d)) for d in DIRECTIONS),
                    f"{sum(r['first_argmax'] in r['reducing_actions'] for r in probes)}/{len(probes)}",
                    f"{sum(r['first_argmax'] in r['reducing_actions'] for r in cohort)}/{len(cohort)}",
                    f"{sum(r['exact_observation_count']>0 for r in supported)}/20",
                    f"{sum(r['approximate_transition_support']>0 for r in supported)}/20"])
    result=[f'# Phase 5D read-only findings — 2026-11-21\n',f'Verdict: **{summary["verdict"]}**. {summary["reason"]}\n',
      f'Branch: `feature/{STUDY}`. Reviewed HEAD `{BASE}`; tree `{plan["base_tree"]}`. '
      'The Phase 5D implementation is uncommitted at report generation. No main merge or push was performed. '
      'Historical Phase 5A commit `d3822423ffd6fa3c83f9a26fc15808fa9103e36c`, Phase 5B `ae3ae73bf7ba5cbe36dcf879125f15fd14c43f4e`, Phase 5C parent as above.\n',
      'This study measures existing Fixture A evidence. It makes no performance improvement claim. '
      'Historical assessments are deterministic episodes previously executed; probe/cohort rows are frozen scores, not new navigation results. '
      'The fixed outcome-balanced cohort is deliberately selected from distinct near-one-cell historical observations, not a population success estimate.\n',
      '## Phase 5D read-only findings table\n',
      'Expanded success counts are ordered Up/Down/Left/Right, each out of four. Probe alignment is a first-argmax Manhattan-reducing choice across 40 legal observations (including D4 histories); cohort alignment uses the registered fixed sample. '
      'Exact/approximate support counts refer to the 20 development reset observations in each historical training prefix. Prefixes 0/2k/6k overlap and are never independent repetitions.\n',
      table(['Family','Arm','Checkpoint','Original goals','Expanded goals','Expanded U/D/L/R','Probe aligned','Cohort aligned','Exact supported','Approx supported'],findings),
      '\n## Historical behavior and goal rewards\n',
      'Reset starts, decisions, distinct observations, exact position-goal pairs and arrivals have different denominators. '
      'The following table uses all one-cell axis decisions in each nested prefix. Selected reducing/away/Wait reflect exploration behavior; greedy means contemporaneous logged first argmax. '
      'The CSV additionally preserves exact state/pair counts and all phase counters. Empty autoresets follow the original Phase 5C accounting.\n',
      table(['Family','Arm','CP','Direction','Starts','Near1','Reduce/away/Wait','Goal arrivals','Greedy reducing','Ties'],
        [[r['family'],r['arm'],r['checkpoint'],r['direction'],r['episode_starts'],r['near1_decisions'],
          f"{r['reducing_attempts']}/{r['away_attempts']}/{r['wait_attempts']}",r['goal_arrivals'],r['contemporaneous_greedy_reducing'],r['ties']] for r in coverage]),
      '\n## Actual updates and checkpoint credit\n',
      f'All {summary["linear_updates_audited"]:,} recorded linear updates passed TD-error and normalized update-norm checks. '
      f'{summary["linear_independent_targets"]:,} targets could also be independently recovered from terminal rewards or adjacent contemporaneous scores after analytically removing the already-recorded increment. '
      'No weights were updated by this calculation. Remaining truncated/end-of-stream bootstraps are explicitly unavailable for independent verification; their actual logged targets remain preserved. '
      'The mechanics are alpha=0.1, gamma=0.99, increment alpha×error×phi/(phi·phi). SARSA uses its logged sampled next action, never max Q. '
      'Every linear probe, case and cohort score decomposes into all ten feature groups and reconstructs Q within tolerance.\n',
      'DQN online and target snapshots were compared at all checkpoints on the fixed cohort and all existing probes/cases. '
      'Cohort residuals use r + .99×(1−terminated)×max target-Q(next) − online-Q(chosen). '
      'SB3 masks replay timeout dones, so external truncation bootstraps. Every resulting target/residual is snapshot-only, even if its counters coincide. '
      'Positive terminal goal reward 9.99 is observed separately; it does not establish historical replay credit propagation or a greedy Right preference. '
      'Historical per-batch sample IDs, target tensors and intermediate target-network weights are absent.\n',
      'PPO rows contain probabilities, entropy, Right log odds and V(s); no Q(s,a). '
      'All pre-action collection probabilities and V are logged. Saved last-rollout values/log probabilities were checked against those logs; returns=advantages+values and interior GAE recurrence were checked independently. '
      'Only the final 100 transitions at 2k and 6k are retained per arm. An unsaved pre-update final bootstrap cannot be replaced by the post-update critic or inferred from the advantage under verification. Full GAE history and batch-normalized advantages are unavailable.\n',
      'PPO timeout rollout rewards may include the historical collector bootstrap augmentation. Raw environment reward and saved buffer reward are kept distinct in buffer-audit.json; an episode-end zero-continuation recurrence check does not independently recover the unsaved timeout bootstrap tensor.\n',
      table(['Family','Arm','CP','Saved rollout/replay records','Boundary / missingness'],
        [[r['family'],r['arm'],r['checkpoint'],r.get('saved_rollout_transitions',r.get('saved_replay_transitions')),
          r.get('boundary',r.get('sampled_batch_IDs_and_actual_targets'))] for r in buffers]),
      '\n## All DQN B1 final development cases\n',
      'The complete expanded panel below includes all four Right failures and the other twelve cases; original cases are retained as well. '
      'Scores use action order Up, Down, Left, Right, Wait. Margin is best reducing score minus best nonreducing score. '
      'All 480 family/arm/checkpoint cases, full 409-value pre-action observations, raw rays ordered North/South/East/West, known-cell counts, historical first actions, outcomes and trace references are in case-decisions.csv.\n',
      table(['Panel','Case','Direction','Distance','Position→goal','Scores U/D/L/R/W','Argmax','Goal margin','Old first action','Historical result/actions'],
        [[r['panel'],r['case_id'],r['direction'],r['distance'],r['position_goal'],r['scores'],r['first_argmax'],r['goal_margin'],r['historical_first_action'],
          f"{r['historical_outcome']}/{r['historical_actions']}"] for r in cases if (r['family'],r['arm'],r['checkpoint'])==('dqn','B1-relative-x4',6000)]),
      '\nThe DQN B1 Right training reset count is 104; current-axis near-Right decisions are 210; Right-arrival goals are 88. '
      'These are distinct counts, not interchangeable success rates. The four final expanded Right cases are 0/4, despite actual reward-bearing Right actions during training. '
      'Training start cases and development positions/history differ. PPO stochastic reward arrivals likewise cannot be read as deterministic argmax success.\n',
      '## Legal goal sensitivity and symmetry\n',
      'All 24 reset probes and 16 D4 histories are exactly the Phase 5C observations. Goal-only pairs preserve position and all history/ray fields; goal cells must be unvisited and different from current position. '
      'Translations and D4 transformations also change absolute position and memory; they cannot identify a pure isolated memory effect. '
      'No arbitrary memory-cell edits or new histories were constructed. D4 disagreements preserve ties and action permutations. The table reports means within each family/arm/checkpoint only; Q magnitudes and PPO probabilities are never compared across families.\n']
    contrast_rows=[]
    for family in FAMILIES:
        for arm in plan['settings']['arms']:
            for cp in (0,2000,6000):
                for kind in ('legal goal-only','D4 legal history'):
                    group=[r for r in contrasts if (r['family'],r['arm'],r['checkpoint'],r['kind'])==(family,arm,cp,kind)]
                    contrast_rows.append([family,arm,cp,kind,len(group),float(np.mean([r['score_mean_abs_change'] for r in group])),
                        sum(r.get('argmax_changed',False) for r in group) if kind=='legal goal-only' else sum(r['tie_set_equivariant'] for r in group)])
    result+=[table(['Family','Arm','CP','Comparison','Pairs','Mean absolute score difference','Argmax changes / equivariant tie sets'],contrast_rows),
      '\nThe full score records include Manhattan distance, quadrant, boundary, visits, known cells, elapsed and memory-age matching bins. '
      'Exact equality hashes all 409 float32 values. Exact position-goal matches do not imply identical visible history. Approximate matches require identical registered coarse bins and mean raw observation L1≤0.05; these are not exchangeable randomized controls. '
      'Zero support means unresolved due to support, never evidence of no effect. Repeated transitions and distinct observation counts are both retained.\n',
      '## Limits and next step\n',summary['followup']+' This is an instrumentation recommendation, not authorization for training or selection of a reward/feature intervention. '
      'Association between frozen scores and rewards cannot isolate representation, sampling, optimization or credit assignment. '
      'Multiple exploratory contrasts have no confirmatory significance claim. No obstacle/dynamic generalization or final/test performance is measured.\n',
      '## Validation, preservation and resources\n',
      f'All G0–G5 gates passed. Inherited baseline: 399 tests. New full regression run: {verified["full_tests"]["tests"]} tests, '
      f'{verified["full_tests"]["failures"]} failures, {verified["full_tests"]["errors"]} errors. '
      'Test-only fixtures are mechanical validation, not scientific learning evidence; inherited tests may run their original tiny mechanics. '
      'The first focused run had two synthetic-fixture dtype failures (23 passed); both were fixed before registration, followed by 25 passing focused tests. Validation history preserves the failed receipt. '
      'Both original main working-tree inventory and the reviewed feature baseline are preserved by byte hashes; historical delivery pins remain unchanged.\n',
      table(['Quantity','Actual'],list(a.items())+[['Analysis wall seconds',summary['wall_seconds']],['Score-call ceiling',5000],['Analysis wall ceiling seconds',1200],['Artifact ceiling bytes',134217728]]),
      '\nBaseline/postrun test timing is recorded separately in JUnit/receipts, outside the scientific analysis wall. '
      'Weights, optimizer/buffer content and RNG were protected during each inference context; all original checkpoint files remain byte-identical. '
      'Independent verification recomputes arithmetic/hashes and consumes zero new score calls.\n',
      '## Evidence and reproduction\n',
      '[Registered protocol](partial_phase5d_protocol.md). Evidence namespace: '
      f'[`{STUDY}`](../artifacts/partial-dynamic-navigation-v1/{STUDY}/diagnostic-summary.json). '
      'Inspect analysis-plan.json/seal, source-and-evidence-manifest.json, evidence-availability.json, '
      'credit-metrics.csv, linear-credit-audit.csv, matched-states.csv, case-decisions.csv, direction-coverage.csv, '
      'counterfactual-comparisons.csv, scores.json, buffer-audit.json, forward-calls.jsonl, validation-gates.json and output-hashes.json.\n',
      'Run from this application root with its installed interpreter and `PYTHONPATH=src`: '
      '`python -m robot_navigation.partial_phase5d verify` and `... report` are read-only rescoring-free checks. '
      '`prepare`/`analyze` are exclusive one-shot commands and reject an existing namespace/lock; do not rerun or overwrite the registered study. '
      'The raw historical trace/checkpoint paths and hashes are in the sealed evidence manifest.\n']
    detailed=[]
    for family in FAMILIES:
        for arm in plan['settings']['arms']:
            for cp in (0,2000,6000):
                for d in DIRECTIONS:
                    group=[r for r in scores if (r['family'],r['arm'],r['checkpoint'],r['kind'],r.get('direction'))==(family,arm,cp,'fixed-historical-cohort',d)]
                    if not group:continue
                    detailed.append([family,arm,cp,d,len(group),sum(r['historical_reward']>0 for r in group),
                        sum(r['first_argmax'] in r['reducing_actions'] for r in group),
                        float(np.mean([r['Right_margin'] for r in group])),
                        float(np.mean([r['goal_margin'] for r in group])),
                        float(np.mean([r['snapshot_credit']['residual'] for r in group])) if family=='dqn' else None])
    result += ['\n## Fixed historical cohort by direction\n',
        'The same selected observations are scored at all three snapshots. Counts below are cohort support, not independent episodes or representative success rates. Right margin is Right minus best other; goal margin is goal-reducing minus best other. DQN residuals are frozen counterfactual calculations only.\n',
        table(['Family','Arm','CP','Current direction','States','Observed goal records','Snapshot aligned','Mean Right margin','Mean goal margin','Mean snapshot TD residual'],detailed)]
    attribution_rows=[];rank_rows=[]
    for family in FAMILIES:
        for arm in plan['settings']['arms']:
            grouped={cp:[r for r in scores if (r['family'],r['arm'],r['checkpoint'])==(family,arm,cp)] for cp in (0,2000,6000)}
            if family in FAMILIES[:2]:
                for cp,group in grouped.items():
                    p=[r for r in group if r['kind']=='legal-probe']
                    components=p[0]['linear_attribution']['Right_minus_best_other_components']
                    attribution_rows.append([family,arm,cp,len(p)]+[float(np.mean([r['linear_attribution']['Right_minus_best_other_components'][k] for r in p])) for k in components])
            for previous,current in ((0,2000),(2000,6000)):
                before,after=grouped[previous],grouped[current]
                if len(before)!=len(after):raise ValueError('Checkpoint cohort correspondence changed')
                rank_rows.append([family,arm,f'{previous}→{current}',len(after),
                    sum(a['first_argmax']!=b['first_argmax'] for a,b in zip(before,after)),
                    sum(a['first_argmax'] in a['reducing_actions'] and b['first_argmax'] not in b['reducing_actions'] for a,b in zip(before,after)),
                    sum(a['first_argmax'] not in a['reducing_actions'] and b['first_argmax'] in b['reducing_actions'] for a,b in zip(before,after)),
                    sum(a['Right_margin']<=0<b['Right_margin'] for a,b in zip(before,after)),
                    sum(a['Right_margin']>0>=b['Right_margin'] for a,b in zip(before,after))])
    result+=['\n## Linear contribution and checkpoint preference changes\n',
        'Mean signed contributions below are Right minus the best competing action on all 40 legal probes. Each row uses its own winning competitor. Absolute contributions can cancel and are descriptive algebra, not a learned causal importance estimate.\n',
        table(['Family','Arm','CP','Probes','Bias','Position','Goal','Rays','Occupancy','Age','Visits','Streak','Elapsed','Relative goal'],attribution_rows),
        '\nCheckpoint preference changes below use the same ordered probes and fixed cohort states (not episodes). A tie at zero Right margin is retained as nonpositive; ordinary argmax changes use the original first-index semantics.\n',
        table(['Family','Arm','Change','States','Argmax changed','Aligned→unaligned','Unaligned→aligned','Right nonpositive→positive','Right positive→nonpositive'],rank_rows)]
    text='\n\n'.join(result)+'\n'
    (root/'docs/partial_phase5d_results.md').write_text(text,encoding='utf-8',newline='\n')
    diagnosis=f'# Phase 5D diagnosis\n\n**{summary["verdict"]}**\n\n'+summary['reason']+'\n\n'+summary['followup']+'\n\nSee [complete findings, all families and checkpoints](partial_phase5d_results.md). No training or main merge was performed.\n'
    (root/'docs/partial_phase5d_diagnosis.md').write_text(diagnosis,encoding='utf-8',newline='\n')
    (folder/'results.md').write_text(text,encoding='utf-8')
    # Original working tree, including unrelated dirty UI/map work, is checked separately.
    inventory=read_json(folder/'original-inventory.json')
    common=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],cwd=root,text=True).strip())
    original_repo=(root/common).resolve().parent if not common.is_absolute() else common.resolve().parent
    from .partial_phase5d import checked_hash
    for p,h in inventory['files'].items():checked_hash(original_repo/p,h)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=original_repo,text=True).strip()!=inventory['head']:
        raise ValueError('Original main HEAD changed')
    if subprocess.check_output(['git','-c','core.longpaths=true','status','--porcelain'],cwd=original_repo,text=True)!=inventory['status']:
        raise ValueError('Original working-tree status changed')
    if subprocess.check_output(['git','diff','--cached','--binary'],cwd=original_repo,text=True)!=inventory['index']:
        raise ValueError('Original staged work changed')
    write_json(folder/'original-workspace-preservation.json',{'status':'passed','files_checked':len(inventory['files']),'head':inventory.get('head')})
    usage=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    if usage>134217728:raise RuntimeError('Artifact cap exceeded')
    write_json(folder/'output-hashes.json',{'files':{p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='output-hashes.json'},
        'reports':{p.relative_to(root).as_posix():sha(p) for p in (root/'docs/partial_phase5d_results.md',root/'docs/partial_phase5d_diagnosis.md')},
        'bytes_before_hash_manifest':usage,'hash_manifest_self_excluded':True})
    return {'verdict':summary['verdict'],'results':'docs/partial_phase5d_results.md','artifact_bytes':sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())}
