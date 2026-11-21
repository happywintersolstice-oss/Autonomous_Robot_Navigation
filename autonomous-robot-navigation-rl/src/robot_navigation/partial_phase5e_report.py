"""Render Phase 5E evidence already captured; never executes a learner."""
from pathlib import Path

from .core import ROOT,read_json
from .partial_phase5e import location,BASE,STUDY


def table(headers,rows):
    def cell(v):return f'{v:.6g}' if isinstance(v,float) else str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+
        ['| '+' | '.join(cell(x) for x in row)+' |' for row in rows])


def render(summary,pairs,numerical,root=ROOT):
    folder=location(root);p=read_json(folder/'attempt-plan.json');totals=summary['new_synthetic_cumulative']
    preservation=read_json(folder/'preservation.json');legacy=summary['legacy_activity'];timing=[]
    for row in pairs:
        path=folder/'pairs'/(row['algorithm']+'-'+row['arm'])
        off,on=[read_json(path/mode/'receipt.json') for mode in ('off','on')]
        timing.append([row['algorithm'],row['arm'],row['transitions_per_run'],row['optimizer_steps_per_run'],
            'bitwise / tolerance 0',off['wall_seconds'],on['wall_seconds'],on['wall_seconds']/off['wall_seconds']])
    history=[]
    for attempt in sorted((folder/'development-history').glob('*')) if (folder/'development-history').exists() else []:
        f=attempt/'validation-failure.json';c=attempt/'validation-complete.json'
        if f.exists():
            failure=read_json(f);history.append([attempt.name,'failed development attempt',failure.get('counts'),
                'Preserved genuine receipt; any source fix occurred before final seal; operations remain cumulatively charged'])
        elif c.exists():history.append([attempt.name,'passed development attempt',read_json(c)['focused'],'Preserved; counted in cumulative mechanical ledger'])
    text=[f'# Phase 5E training instrumentation results\n\n**{summary["verdict"]}**\n',
      f'Branch `feature/{STUDY}`, reviewed parent `{BASE}`, tree `{p["base_tree"]}`. '
      'This report is generated before the Phase 5E publication commit; the final response/Git branch identifies that commit. '
      'Instrumentation and synthetic mechanics only. No scientific learning, research navigation-assessment episode, sealed-final/unseen evaluation or Phase 6. '
      'No model superiority, improved navigation or resolved Right-goal cause is claimed.\n',
      '## Scope and source provenance\n',
      '[Protocol](partial_phase5e_protocol.md), [diagnosis](partial_phase5e_diagnosis.md), '
      f'[evidence summary](../artifacts/partial-dynamic-navigation-v1/{STUDY}/summary.json). '
      'The hash-sealed attempt plan pins the new source, tests, configuration, minimal flat source snapshot and actual installed library implementations. '
      'The library audit contains exact call-path source and dependency SHA256 values; no assumption about another SB3 release is made.\n',
      table(['Package','Installed version'],list(p['library']['versions'].items())),
      '\nThe optional observer is disabled by default. A missing sink installs no hooks. Enabled observers capture actual live computations on the pinned CPU/single-environment path. '
      'Original learner methods are called once, with no replacement optimizer, second sampler, extra permutation, prediction or environment action for logging. '
      'The independent profile witness is present in both off/on members and verifies the optional line/optimizer/tensor observers introduce no behavioral difference. '
      'All hooks and reduction wrappers are restored, including under injected exceptions.\n',
      '## Paired on/off equivalence and measured overhead\n',
      table(['Model','Arm','Transitions per member','Updates per member','Equality','Off seconds','On seconds','On/off ratio'],timing),
      '\nEach member runs in a fresh matched subprocess using seed 23101 and original model hyperparameters. CPU comparison tolerance is zero. '
      'Compared records include all executed actions and probabilities, resets, raw observations/rewards/flags, global/env/action RNG, actual replay samples and PPO minibatches, '
      'losses, clipped gradients, optimizer state, model/target weights, complete buffer fingerprints and forward counts. '
      'Timing is mechanical-fixture overhead including capture and serialization, not a representative production benchmark or statistical estimate.\n',
      'The fixed synthetic stream deliberately repeats one legal 409-value observation while emitting controlled continuing/goal/collision/timeout flags and rewards. '
      'Actions cannot change that artificial stream. Its positive labels are not research navigation successes. '
      'DQN collects four transitions below normal warmup and then executes one explicitly isolated real train-method unit update. '
      'The target-sync callback is exercised at its actual 250th callback invocation without more environment steps. '
      'PPO collects its original 100-step rollout and normal five-epoch/two-minibatch training, then both disposable test instances receive the declared target_kl=1e-12 '
      'early-stop fault fixture on that same existing rollout. No historical model, config or production hyperparameter is changed.\n',
      '## Actual signals captured and independent numerical checks\n',
      '| Learner | Captured actual fields | Independent check / limitation |\n'
      '| --- | --- | --- |\n'
      '| Q-learning / SARSA | Actual target/error, phi, prior/live weights, chosen action, actual next sampled SARSA action, alpha/gamma, normalized increment and post-update weights | Existing unchanged normalized semi-gradient rule; no max substitution for SARSA; prior Q is algebra on actual pre-update inputs |\n'
      '| DQN | Real batch indices in order including duplicates; env index, slot/generation/transition IDs; consumed replay tensors and masked dones; actual pre-update chosen Q, target max, discounts, target tensor and Huber loss; clipped gradients; update IDs/LR/fingerprints; target sync and epsilon/counters | Actual r+gamma×(1-done)×target max; terminal targets equal raw reward; external timeout bootstrap retained; never reconstructed with a later network |\n'
      '| PPO | Actual sampled actions/probabilities/log-probs/V; raw and collector-adjusted rewards; timeout terminal observations/values; exact final pre-update bootstrap/dones; rollout rows/GAE/returns; original shuffled batches; actual normalization mean/std; ratios/objectives/value/entropy/total losses/KL; clip/early-stop events and optimizer IDs | Backward GAE with actual bootstrap, returns identity, sample-std normalization and loss arithmetic; PPO Q fields explicitly not_applicable |\n',
      table(['Pair','DQN loss batches','PPO loss batches','GAE rollouts','Linear updates','Neural optimizer steps'],
        [[r['pair'],r['counts']['DQN'],r['counts']['PPO'],r['counts']['GAE'],r['counts']['linear'],r['optimizer_steps']] for r in numerical]),
      '\nFloat32 numerical reconstruction uses preregistered atol=2e-5, rtol=2e-6; linear algebra uses atol=1e-10. '
      'These numerical tolerances never relax bitwise on/off equality, identities, permutations or sampled operations. '
      'Actual raw terminal/truncation flags, replay timeouts, consumed masks, collector reward augmentation and final rollout continuation are kept separate. '
      'Versioned records carry provenance, shape/dtype, finite checks and hashes. Missing values remain not_observed; absent architecture fields are not_applicable.\n',
      '## Supported path and unsupported conditions\n',
      'The validated path is the current project CPU, one synchronous environment, 409 inputs/412 transformed features, Discrete(5), standard one-step replay and default production architecture. '
      'Unsupported modes fail closed: '+', '.join(summary['unsupported'])+'. '
      'This is explicit compatibility scope, not a claim that every SB3 mode or release has been tested. '
      'Case identifiers must be provided by the caller/public metadata; missing IDs are recorded as not_observed rather than reading hidden geometry. '
      'The instrumentation does not recover missing historical Phase 5B/DQN minibatches or old PPO GAE; it records future authorized execution only.\n',
      '## Tests, injections and failure history\n',
      table(['Test process','Tests','Failures','Errors','Skipped'],
        [['New final Phase 5E focused suite']+list(summary['final_focused_tests'].values()),
         ['Complete unchanged parent regression suite']+list(summary['legacy_tests'].values())]),
      '\nNegative fixtures require rejection of malformed replay indices, duplicate transition IDs, wrong generations/rollout IDs, wrong network versions, nonfinite tensors, '
      'corrupt payloads/checkpoints, missing bootstrap, interrupted logs, provenance spoofing, exhausted quotas and forbidden final/unseen paths. '
      'These expected rejections are passing tests, not discarded failed experiments. Genuine unexpected development failures, if any, are listed below and their receipts retained.\n',
      table(['Attempt','Status','Actual test counts','Handling'],history) if history else 'No development attempt was run; final validation is the only new mechanical attempt.\n',
      '\n## Resource accounting\n',
      table(['Category','Actual','Hard cap'],[
        ['New Phase 5E transitions, cumulative across development and final',totals['transitions'],1000],
        ['New Phase 5E optimizer steps, cumulative',totals['optimizer_steps'],500],
        ['New active synthetic validation wall seconds, cumulative',totals['wall_seconds'],1200],
        ['Cumulative artifact bytes before final report metadata (includes retained local development)',summary['artifact_bytes_before_final_report_metadata'],134217728],
        ['New scientific training actions',0,0],['New research assessment actions',0,0],['Sealed-final/unseen evaluations',0,0]]),
      '\nThe user explicitly authorized unchanged legacy regressions to be separately ledgered. They were not excluded, modified or charged to the new Phase 5E fixture cap. '
      'Their actual operations/time are reported below, not hidden inside a scientific count. Failed-call attempts are distinguished from completed environment transitions.\n',
      table(['Legacy quantity','Actual'],list(legacy['operations'].items())+[['wall_seconds',legacy['wall_seconds']]]),
      '\nThe new validation namespace is checked against 128 MiB; final delivered file sizes can be summed from the files named in output-hashes.json. '
      'Source/tests are small new additive files; old archives are referenced by hashes and commit rather than copied. The cumulative ledger includes failed development operations and provides no free retry.\n',
      '## Preservation and verification\n',
      table(['Preservation field','Value'],list(preservation.items())),
      '\nAll previous tracked files and experiment evidence are byte-identical to the parent inventory. Main HEAD, the original dirty file contents, '
      'expanded untracked status, staged index and historical branch refs match the pre-work snapshot. '
      'The old Phase 5D renderer and its separately documented delivery limitations are untouched.\n',
      'Run from the application root using the installed interpreter and `PYTHONPATH=src`: '
      '`python -m robot_navigation.partial_phase5e verify` reads hashes/captured arrays only and performs no training, environment stepping or predictions. '
      '`validate` is exclusive and must not be rerun on an existing final namespace; all mechanical operations are cumulatively charged. '
      'Evidence includes attempt-plan/seal, source/library hashes, schema-bearing trace shards, independent witnesses, paired receipts, numeric audit, '
      'operation ledger, both JUnit/log processes, development failure history, preservation and output hashes.\n',
      '## Exactly one proposed next study — not authorized or started\n',summary['future_proposal']+' '
      'Its objective would be checking actual reward-to-replay/rollout-to-update evidence, not claiming that instrumentation cures navigation or proves generalization. '
      'No learning pilot or Phase 6 is started by this validation.\n']
    if (folder/'development-fixes.json').exists():
        text+=['\n## Development corrections retained before final sealing\n',
            table(['Attempt','Actual result','Cause','Correction','Threshold changed'],
                [[r['attempt'],r['actual_result'],r['root_cause'],r['fix'],r['threshold_changed']] for r in read_json(folder/'development-fixes.json')])]
    (root/'docs/partial_phase5e_results.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8',newline='\n')
    diagnosis=('# Phase 5E diagnosis\n\n**'+summary['verdict']+'**\n\n'
        'Future explicitly authorized learning can record the actual replay sample/slot/generation, network version and TD target consumed by each DQN optimizer step. '
        'PPO can retain raw/adjusted rewards, pre-update boundary values, complete actual GAE/returns, shuffled batches, normalized advantages, losses and optimizer outcomes. '
        'These observers passed synthetic bitwise noninterference tests on the pinned supported path. They do not recover absent old history or explain the old Right failures causally.\n\n'
        'Actual future logging can test whether Right reward transitions are sampled, which target/network states were used, and which PPO advantage/minibatch signals accompanied chosen actions. '
        'It cannot by itself distinguish all representation, optimization, exploration and state-support effects or establish a beneficial intervention. '
        'The synthetic controlled stream is not navigation performance evidence.\n\n'+summary['future_proposal']+'\n\n'
        'See [complete validation results](partial_phase5e_results.md) and [registered protocol](partial_phase5e_protocol.md). '
        'Historical experiments and the original dirty workspace remain unchanged. No scientific training, final evaluation or Phase 6 occurred.\n')
    (root/'docs/partial_phase5e_diagnosis.md').write_text(diagnosis,encoding='utf-8',newline='\n')
