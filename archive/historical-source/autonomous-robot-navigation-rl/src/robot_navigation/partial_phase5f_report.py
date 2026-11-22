"""Data-only rendering of Phase 5F evidence; never constructs an agent."""
import json
from .core import read_json
from .partial_phase5f_io import location
from .partial_instrumentation_trace import quota_bytes


def render(root,result):
    folder=location(root);manifest=read_json(folder/'manifest.json');legacy=read_json(folder/'legacy-regressions/receipt.json')
    verdict=result['verdict'];arms=result.get('arms',{})
    rows=['# Phase 5F actual instrumented learning signals','',f'Verdict: **{verdict}**.','',
        'This is a measurement-validity pilot on real Fixture A training, not a navigation-performance comparison. No evaluation episodes, checkpoint selection, additional seeds, full Phase 5 or Phase 6 were run.',
        '',f"Parent: `{manifest['parent_commit']}`. Branch: `{manifest['branch']}`. Frozen seed 23121; B1-relative-x4; original Phase 5B configuration and eight training cases.",
        '', '## Actual runs', '', '| Algorithm | Training actions | Actual optimizer steps | Recorded goal experiences | Goal sample draws | Distinct sampled goals | Wall seconds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,a in arms.items():
        s=a['signals'];rows.append(f"| {name.upper()} | {a['training_actions']} | {a['optimizer_steps']} | {s['recorded_goal_experiences']} | {s['goal_sample_draws']} | {s['distinct_sampled_goals']} | {a['wall_seconds']:.3f} |")
    if not arms:rows.extend(['', 'No successful learning outcome is claimed. Failure: `'+str(result.get('error',result.get('verdict')))+'`. Inspect the preserved failure/verification receipt and ledger for the actual prefix.'])
    for name,a in arms.items():
        s=a['signals'];rows.extend(['',f'## {name.upper()} actual signal summary','',
            f"{s['all_sample_draws']} sample draws represent {s['distinct_sampled_experiences']} distinct experiences. Repeated sample use is optimizer exposure, not additional environment actions.",
            '',f"Completed episodes: {a['completed_episodes']}; resets: {a['resets']}; trailing incomplete-episode actions: {a['incomplete_episode_actions']}; empty final auto-reset: {a['empty_final_reset']}. Raw outcomes: `{json.dumps(a['terminal_outcomes'],sort_keys=True)}`.",
            '', 'Observed quantities:', '', '```json',json.dumps({k:s[k] for k in (('actual_target','residual_snapshot_reconstructed','goal_draw_frequencies','profile_counts') if name=='dqn' else ('advantages','returns','timeout_bootstraps','profile_counts','Q_values'))},indent=2),'```',
            '', 'Earliest observed goal-reward-to-update link (IDs and sequence point directly into the closed trace):','', '```json',json.dumps(s['reward_to_update_example'],indent=2),'```',
            '',f"Evidence: [{name} trace shards](../artifacts/partial-dynamic-navigation-v1/{manifest['study_id']}/{name}/trace/), [run receipt](../artifacts/partial-dynamic-navigation-v1/{manifest['study_id']}/{name}/run-receipt.json). Final provenance checkpoint SHA256: `{a['checkpoint_sha256']}`. Trace-chain tail: `{a['trace_chain_sha256']}`.",
            '',f"Independent checks: `{json.dumps(a['arithmetic']['counts'])}`; every actual optimizer operation has one ledger debit, completion and observer record. Required missing fields: `{s['required_missing_fields']}`."])
    rows.extend(['','## Validation and limits','',
        f"Unchanged parent regression tests: `{json.dumps(manifest['legacy_counts'])}`; elapsed {legacy['wall_seconds']:.3f} seconds. Parent process: `{json.dumps(legacy['parent_process']['operations'])}`. Child E regression mechanics: `{json.dumps(legacy['child_fixture_mechanics'])}`. Combined legacy environment transitions: {legacy['combined_environment_transitions']}; combined optimizer and partial-linear steps: {legacy['combined_optimizer_and_partial_linear_steps']}. These are legacy fixtures, not F scientific learning or new F synthetic validation.",
        '',f"Targeted post-pilot tests: `{json.dumps(result.get('post_tests','not_observed'))}`. Every test attempt, stdout/stderr and JUnit receipt is retained under tests/. New F synthetic physical transitions/optimizer steps: **0 / 0**, against preregistered 0 / 0 caps. Pretests use fake journal reservations; posttests read actual pilot traces.",
        '', 'The original E read-only verifier initially rejected the newly added untracked F instruction file in the original workspace. Its scoped pass used a process-local Git exclude for that one file, without changing old code, settings or evidence. The initial failure, exact diff and scoped pass are retained in parent-preflight.json. The separate F preservation inventory includes the full current original workspace, including that file.',
        '',f"Current evidence bytes before this report/output inventory: {quota_bytes([folder])}; hard cap 536870912. Conservative preregistered uncompressed upper estimate: {manifest['expected_uncompressed_upper_bytes']}. Actual pilot runtime: {read_json(folder/'run.json')['wall_seconds'] if (folder/'run.json').exists() else 'not_observed'} seconds; hard caps 900 per arm /1800 combined. No uninstrumented matched research run exists, so additional live instrumentation overhead and live bitwise on/off equivalence are **not_observed**. E's synthetic on/off equivalence remains inherited evidence only.",
        '', 'DQN targets and losses are actual consumed optimizer tensors; reported residuals are snapshot_reconstructed algebra on those same pre-update tensors, not later checkpoint inference. PPO reports probabilities, log-probabilities, V, GAE and returns, with Q-values not_applicable. Raw environment goal rewards are separated from timeout bootstrap additions. Observed reward use does not prove causal credit assignment or navigation improvement. A 500-action arm may contain no goal event; missing examples are reported as not_observed, never synthesized.',
        '', 'All historical source/evidence and original dirty-main files are byte-checked by preservation.json. No historical trial was rewritten. Source/config/library provenance, closed trace shard hashes, checkpoints and tests are indexed in output-hashes.json. Run `python -m robot_navigation.partial_phase5f verify` from this application with PYTHONPATH=src for read-only verification. The one-shot run command cannot be retried after run.lock exists.',
        '', 'Stopped after Phase 5F. No further learning/evaluation is authorized.'])
    (root/'docs/partial_phase5f_results.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    diagnosis=['# Phase 5F diagnosis','',f'Actual verdict: **{verdict}**.','',
        'The question is whether the inherited passive observer captures actual learning signals and whether closed evidence supports independent arithmetic and lineage checks. The results report and verification.json contain the observed counts and every required-gate result. Historical Phase 5B outcomes are contextual evidence only; Phase 5E mechanical equivalence is synthetic evidence; neither substitutes for these F live traces.','',
        'Even validated traces do not show that either agent learned to navigate, resolve directional imbalance, or improve task performance. The experiment contains no deterministic evaluation, matched uninstrumented arm, causal intervention, alternate reward, altered sampler or winner selection. DQN and PPO have equal environment budgets but different optimizer computation and sample reuse.','',
        'One next proposal: subject to separate authorization and a newly registered bounded protocol, extend the same instrumented Fixture A training measurement to test whether the observed reward-to-update coverage persists over a longer fixed horizon, keeping model, reward and sampler unchanged. Require an explicit action budget and independent trace gates before starting. This proposal is not executed.','',
        'No full Phase 5, new evaluation panel, additional seed or Phase 6 was started.']
    (root/'docs/partial_phase5f_diagnosis.md').write_text('\n'.join(diagnosis)+'\n',encoding='utf-8')
