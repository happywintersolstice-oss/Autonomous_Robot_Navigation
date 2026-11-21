"""Measured Phase 5G report rendering from closed verification receipts only."""
import json
from .core import read_json
from .partial_phase5g_io import location,CHECKPOINTS
from .partial_instrumentation_trace import quota_bytes


def render(root,result):
    folder=location(root);m=read_json(folder/'manifest.json');verdict=result['verdict'];training=result.get('training',{});assess=result.get('assessment',{}).get('arms',{})
    legacy=read_json(folder/'legacy-regressions/receipt.json');run=read_json(folder/'run.json') if (folder/'run.json').exists() else {}
    lines=['# Phase 5G measured longer-horizon learning diagnosis','',f'Engineering verdict: **{verdict}**.','',
        f"Parent `{m['parent_commit']}`; branch `{m['branch']}`. Fresh B1 DQN then PPO, seed23121, one original 2000-action learning call each. Checkpoints 0/500/1000/2000. Only the existing eight training, four original development and sixteen expanded development cases were used.",
        '', 'All numerical, identity, source and measurement gates must pass independently of navigation readiness. These familiar static open-grid development panels are not held-out generalization evidence. No full training, tuning, extra seed, dynamic-map learning, final/unseen evaluation or Phase6 was authorized or run.',
        '', '## Fixed checkpoint results','',
        '| Model | Training actions | Original success | Expanded success | U / D / L / R expanded successes | First aligned combined | Original / expanded alignment | Right first margin (5 Right cases) | Wait actions / actual actions | Collisions / timeouts |',
        '|---|---:|---:|---:|---|---:|---|---:|---|---|']
    for a,arm in assess.items():
        for cp in CHECKPOINTS:
            r=arm['checkpoints'][str(cp)];directions=' / '.join(str(r['expanded_by_direction'][d]['successes'])+'/4' for d in ('Up','Down','Left','Right'))
            lines.append(f"| {a.upper()} | {cp} | {r['original']['successes']}/4 | {r['expanded']['successes']}/16 | {directions} | {r['combined']['first_aligned']}/20 | {r['original']['first_aligned']}/4 ; {r['expanded']['first_aligned']}/16 | {r['first_right_margin']:.7f} | {r['wait_count']}/{r['assessment_actions']} | {r['collisions']} / {r['timeouts']} |")
    if not training:lines.extend(['','No complete measurement pass is claimed. Preserved failure:','', '```json',json.dumps(result,indent=2),'```'])
    lines.extend(['','DQN margins are Q units. PPO margins are action-probability differences; its separate Right log odds are preserved below and in verification.json. First alignment means a legal strict Manhattan-distance reduction, not eventual success. Wait, invalid first moves and exact maximal-score ties are reported separately. Neither panels nor earlier checkpoints select a new training regime.'])
    for a,arm in assess.items():
        screen=arm['screen'];lines.extend(['',f'## {a.upper()} readiness and first-action detail','',f"**{screen['verdict']}**. Expanded success gain from its own initialized checkpoint: {screen['expanded_gain']}/16. Frozen checks: `{json.dumps(screen['checks'],sort_keys=True)}`.",
            '', '| Actions | Combined U/D/L/R alignment (each /5) | Expanded U/D/L/R alignment (each /4) | First Wait / invalid / ties | Mean goal margin | Right log odds | Moving revisits / ABAB loops |',
            '|---:|---|---|---|---:|---|---|'])
        for cp in CHECKPOINTS:
            r=arm['checkpoints'][str(cp)];comb=' / '.join(str(r['combined_by_direction'][d]['first_aligned'])+'/5' for d in ('Up','Down','Left','Right'))
            exp=' / '.join(str(r['expanded_by_direction'][d]['first_aligned'])+'/4' for d in ('Up','Down','Left','Right'))
            lines.append(f"| {cp} | {comb} | {exp} | {r['first_wait']} / {r['first_invalid']} / {r['first_ties']} | {r['mean_goal_directed_margin']:.7f} | {r['first_right_log_odds']} | {r['moving_revisits']} / {r['oscillation_windows']} |")
        lines.extend(['',f'## {a.upper()} all case-level measured outcomes','',
            'Full trajectories, initial public observation and first-action score vectors are preserved in the linked development files. Case rows below are recomputed from those actual paths; moving revisits exclude stationary Wait/collision steps. Shortest-route inefficiency is steps/Manhattan distance for successful cases only.',
            '', '| Training actions | Panel | Case ID | Direction | Outcome | Steps | First action | Aligned | Wait | Revisit | ABAB | Successful step inefficiency |',
            '|---:|---|---|---|---|---:|---:|---|---:|---:|---:|---|'])
        for cp in CHECKPOINTS:
            for r in arm['checkpoints'][str(cp)]['case_metrics']:
                lines.append(f"| {cp} | {r['panel']} | {r['case_id']} | {r['direction']} | {r['outcome']} | {r['steps']} | {r['first_action']} | {r['aligned']} | {r['wait_count']} | {r['moving_revisits']} | {r['oscillation_windows']} | {r['success_step_inefficiency']} |")
        for cp in CHECKPOINTS:lines.extend(['',f"[{a.upper()} development records at {cp}](../artifacts/partial-dynamic-navigation-v1/{m['study_id']}/{a}/development/{cp:04d}.json); checkpoint SHA256 `{arm['checkpoints'][str(cp)]['checkpoint_sha256']}`."])
    lines.extend(['','## Actual optimizer reward use','',
        '| Algorithm | Training actions | Actual optimizer steps | Goal experiences | Distinct goals sampled | Goal sample draws | Actual updates with goals | All sample draws / distinct experiences |',
        '|---|---:|---:|---:|---:|---:|---:|---|'])
    for a,r in training.items():
        s=r['signals'];cov=r['goal_coverage']['cumulative']['2000']
        lines.append(f"| {a.upper()} | {r['training_actions']} | {r['optimizer_steps']} | {s['recorded_goal_experiences']} | {s['distinct_sampled_goals']} | {s['goal_sample_draws']} | {cov['actual_updates_using_goals']} | {s['all_sample_draws']} / {s['distinct_sampled_experiences']} |")
    for a,r in training.items():
        s=r['signals'];lines.extend(['',f'## {a.upper()} captured arithmetic and directional exposure','',
            f"Independent actual-tensor arithmetic: `{json.dumps(r['arithmetic']['counts'])}`. Required missing fields: `{s['required_missing_fields']}`. Completed training episodes {r['completed_episodes']}; trailing incomplete prefix {r['incomplete_episode_actions']} actions; raw outcomes `{json.dumps(r['terminal_outcomes'],sort_keys=True)}`.",
            '', '```json',json.dumps({k:s[k] for k in (('actual_target','residual_snapshot_reconstructed','target_syncs','profile_counts') if a=='dqn' else ('advantages','returns','timeout_bootstraps','rollouts','epochs','Q_values','profile_counts'))},indent=2),'```',
            '', '| Cumulative training actions | Direction | Goal experiences | Distinct sampled goals | Goal sample draws | Actual updates using goals |',
            '|---:|---|---:|---:|---:|---:|'])
        for cp in CHECKPOINTS:
            for d,cov in r['goal_coverage']['cumulative'][str(cp)]['by_direction'].items():lines.append(f"| {cp} | {d} | {cov['goal_experiences']} | {cov['distinct_goals_sampled']} | {cov['goal_sample_draws']} | {cov['actual_updates_using_goals']} |")
        lines.extend(['','Actual use within each update-time interval (experiences may have occurred in an earlier interval):','',
            '| Interval | New goal experiences | Distinct goals consumed | Goal draws | Updates using goals |','|---|---:|---:|---:|---:|'])
        for cov in r['goal_coverage']['intervals']:lines.append(f"| {cov['training_action_interval']} | {cov['goal_experiences']} | {cov['distinct_goals_sampled']} | {cov['goal_sample_draws']} | {cov['actual_updates_using_goals']} |")
        lines.extend(['','Earliest directional goal-to-actual-update examples, including explicit absent events:','', '```json',
            json.dumps({d:v['earliest_link'] for d,v in r['goal_coverage']['cumulative']['2000']['by_direction'].items()},indent=2),'```',
            '',f"[Actual compressed {a.upper()} learning traces](../artifacts/partial-dynamic-navigation-v1/{m['study_id']}/{a}/trace/). Sequence/update/transition IDs in the examples resolve to those immutable shards. Every checkpoint's network/optimizer hashes are joined to its completed update boundary in verification.json; no saved model is scored by verification."])
    lines.extend(['','## Resource, isolation and validation evidence','',
        f"Measured run accounting: `{json.dumps(run,sort_keys=True)}`. New synthetic physical transitions and optimizer operations: **0 / 0**, preregistered caps0/0. Legacy regression mechanics are separate: `{json.dumps(legacy,sort_keys=True)}`.",
        '',f"Legacy tests `{json.dumps(m['legacy_counts'])}`; targeted post tests `{json.dumps(result.get('post_tests','not_observed'))}`. All pre/post attempts and genuine errors are preserved. A local Git ownership setup failure was corrected with process-local safe-directory configuration; no shared/global Git setting or scientific source/evidence was changed. The first plain F verifier also identified only the newly supplied G instruction file in main's untracked set. Its scoped pass excludes only that file for the old status comparison; G's own inventory includes it. Full receipts are in the evidence namespace.",
        '',f"Padded preregistered aggregate storage estimate {m['storage_plan']['aggregate_padded_estimate']} bytes, cap536870912. Namespace bytes before final report/inventory write: {quota_bytes([folder])}. Training wall time includes logging and checkpoint saves, capped1800 per arm/3600 combined; actual per-arm seconds: `{json.dumps({a:r['wall_seconds'] for a,r in training.items()})}`. Assessment and full legacy times are separate. Extra instrumentation overhead and live bitwise on/off equivalence are **not_observed**, because an uninstrumented matched research run is unauthorized; E's synthetic equivalence remains inherited evidence only.",
        '', 'All 160 fixed case episodes used detached checkpoint copies. Native step counts equal assessment ledger debits, native optimizer calls are zero, each child checkpoint model/optimizer and child global RNG stayed unchanged during assessment, and parent global RNG/training ledger/receipt/checkpoint bytes were unchanged across children. The full before/after isolation receipt is assessment-parent-isolation.json. Checkpoint saves also preserve training network/optimizer/buffer/RNG fingerprints exactly.',
        '', 'DQN actual TD targets, consumed masks and Huber losses are measured optimizer tensors; residual summaries are snapshot_reconstructed algebra on those same pre-update tensors. PPO reports actual collected probabilities/log probabilities/V, GAE, returns and batch advantages/losses; Q-values are not_applicable. Goal sample draws include repeated draws/epochs; neither draws nor training goals are development successes. Timeout bootstraps are reported only when actually observed, and never counted as goal rewards.',
        '', 'Zero final/unseen research accesses. Original dirty main, historical branch refs and all 10,901 parent files are byte-checked. Output hashes, frozen source/library/case pins, raw ledgers, checkpoint copies and JUnit/log receipts are retained. Read-only reproduction: PYTHONPATH=src python -m robot_navigation.partial_phase5g verify. register/run are one-shot; do not remove locks to rerun.',
        '', 'F and G are separate fresh studies: G uses a different DQN schedule horizon. The four G checkpoint rows describe one G trajectory only. No association establishes causality, fixes credit assignment or ranks algorithms. Stop after Phase5G.'])
    (root/'docs/partial_phase5g_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    screens={a:r['screen'] for a,r in assess.items()};poor=any(r['verdict']!='BASIC_NAVIGATION_SCREEN_MET' for r in screens.values())
    diagnosis=['# Phase 5G diagnosis','',f'Engineering verdict: **{verdict}**.','', 'Measured readiness:', '', '```json',json.dumps(screens,indent=2),'```','',
        'Reward encounters, distinct sampled experiences, repeated goal draws, updates consuming them and deterministic development successes have separate denominators. Complete traces establish that measured signals reached actual optimization; they do not establish that a particular signal caused goal-directed behavior. All checkpoint outcomes, including reversals and directional failures, remain in the results.','',
        'The unchanged representation, reward, uniform sampler and learner settings were held fixed. There was no hyperparameter search, action masking, planner fallback, auxiliary objective, checkpoint selection, dynamic training or generalization evaluation. A fresh 2000-action DQN schedule is not a continuation/control-equivalent extension of F.','']
    if poor:diagnosis.extend(['One specific future intervention, **not implemented or authorized**: preregister a bounded comparison adding an auxiliary four-way goal-direction prediction loss from the already public relative-goal vector, while fixing reward, sampler and training budget. Its purpose would be to test whether explicitly representing goal direction changes first-action alignment after optimizer reward exposure; these associations do not establish that weak goal conditioning is the cause or that the intervention will work. Require separate authorization and a sealed control/intervention design before execution.',''])
    else:diagnosis.extend(['Both frozen readiness screens were met on familiar development cases. This provides no permission to extend training or claim held-out/generalization success. No follow-up intervention was implemented.',''])
    diagnosis.append('Stopped after Phase5G. No full Phase5, final/unseen evaluation or Phase6 was started.')
    (root/'docs/partial_phase5g_diagnosis.md').write_text('\n'.join(diagnosis)+'\n',encoding='utf-8',newline='\n')
