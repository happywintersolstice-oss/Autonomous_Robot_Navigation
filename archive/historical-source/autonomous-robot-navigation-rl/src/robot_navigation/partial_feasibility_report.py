"""Source-backed Phase 5A report/figures; never trains or changes measured data."""
import csv
from pathlib import Path

from .core import ROOT, read_json
from .partial_commands import read_records
from .partial_feasibility import location


def _row(name, category, checkpoint, stats):
    counts = stats['counts']
    return {'Algorithm': name, 'Fixture': category, 'Checkpoint actions': checkpoint,
        'Success': counts['success'], 'Collision': counts['collision'], 'Timeout': counts['timeout'],
        'Mean steps': round(stats['mean_steps'], 2), 'Wait %': round(100*stats['wait_fraction'], 2),
        'Longest Wait': stats['longest_wait_streak'], 'Mean discovered cells': round(stats['mean_discovered_cells'], 2)}


def table(rows):
    if not rows: return ['No completed assessments available.']
    columns = list(rows[0])
    return ['| '+' | '.join(columns)+' |', '| '+' | '.join('---' for _ in columns)+' |']+[
        '| '+' | '.join(str(row[column]) for column in columns)+' |' for row in rows]


def report(root=ROOT):
    folder = location(root); summary = read_json(folder/'summary.json'); protocol = read_json(folder/'phase5a_pilot_protocol.json')
    latest, curves = [], []
    for algorithm, result in summary['systems'].items():
        current = {}
        for checkpoint in result['checkpoints']:
            for category, stats in checkpoint['evaluation'].items():
                current[category] = _row(algorithm, category, checkpoint['training_actions'], stats)
                curves.append({'algorithm': algorithm, 'phase': checkpoint['phase'], 'category': category,
                    'training_actions': checkpoint['training_actions'], 'successes': stats['counts']['success'],
                    'episodes': stats['completed_episodes'], 'wait_fraction': stats['wait_fraction']})
        latest.extend(current.values())
    controls = [_row(name, category, 'control', stats) for name, groups in summary.get('controls', {}).items()
                for category, stats in groups.items()]
    for name, rows in (('latest-fixture-results.csv', latest), ('controls.csv', controls), ('learning-curves.csv', curves)):
        with (folder/name).open('w', encoding='utf-8', newline='') as file:
            if rows:
                writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# Phase 5A: bounded learning-feasibility results', '', 'Measured 20 November 2026.', '',
        f"**Verdict: `{summary['verdict']}`**. This is one diagnostic seed, not a main comparison or evidence of unseen-layout generalization.", '',
        '## Branch, preservation and executed scope', '',
        f"Started from `{protocol['git']['head']}` on `{protocol['git']['branch']}`. The original dirty main workspace was kept separate. Pre-existing tracked source, configs, studies, models, reports and replay files were not edited. New work is in the Phase 5A fixture, ledger, diagnostics, runner, report, config, test and documentation files.", '',
        f"Training reservations: **{summary['budget']['total_training_reserved']:,} / 40,000**; evaluation/control/composition reservations: **{summary['budget']['evaluation_reserved']:,} / 100,000**. Elapsed pilot wall time: **{summary['budget']['elapsed_seconds']:.2f} / 1,200 seconds**. The machine-readable protocol, per-algorithm settings and raw budget journal preserve exact actual accounting.", '',
        'No confirmation seed, feature/reward change, full Phase 5 study, new final set, final evaluation or UI work was performed. Stage gates and unused budgets were respected.', '',
        '## Latest assessed checkpoint per fixture', '',
        'Four held-out start-goal cases per category on familiar fixture geometry. Entries use the latest actually assessed checkpoint for each category, never a selected best result. A missing category means its prerequisite gate or remaining budget prevented progression; it is not a measured failure. Later checkpoint states can differ across categories.', '',
        *table(latest), '', '## Functional controls', '',
        'The same 20 diagnostic cases were used. Scripts are privileged mechanics demonstrations, not learned or fair operational policies. The online planner sees only the observation and can fail. Random is seeded and exploratory.', '',
        *table(controls), '', '## Stage gates and failures', '']
    for algorithm, result in summary['systems'].items():
        lines.append(f"- **{algorithm}**: {result['training_actions']:,} training actions; basic gate {result['basic_gate']}; status {result['status']}; {result.get('halt') or result.get('exception') or 'registered curriculum completed' }.")
    lines += ['', 'Every training/evaluation action distribution, length histogram, incomplete prefix, failure case, sensor/memory state, recovery denominator, model fingerprint and checkpoint is retained. Completed-episode training rates exclude incomplete curriculum/budget prefixes; overall executed returns and action counts include them. Evaluation and training behavior are never pooled.', '',
        '## Actual training and update evidence', '']
    learning_rows = []
    for algorithm, result in summary['systems'].items():
        if not result['checkpoints']: continue
        last = result['checkpoints'][-1]
        training = last.get('training')
        if not training: continue
        stats, model = training['overall'], last['model']
        learning_rows.append({'Algorithm': algorithm, 'Actions': last['training_actions'],
            'Training goals': stats['counts']['success'], 'Training collisions': stats['counts']['collision'],
            'Training timeouts': stats['counts']['timeout'], 'Incomplete prefixes': stats['incomplete_prefixes'],
            'Training Wait %': round(100*stats['wait_fraction'], 2),
            'Up/Down/Left/Right/Wait': '/'.join(str(stats['action_counts'][k]) for k in ('Up', 'Down', 'Left', 'Right', 'Wait')),
            'Updates (family-specific)': last.get('linear_updates') or model.get('sb3_version_specific_updates_counter', 0),
            'DQN replay goals': model.get('replay', {}).get('positive_goal_rewards_stored', 'n/a')})
    lines += [*table(learning_rows), '',
        '## D: privileged hindsight diagnostics', '']
    hindsight_rows = []
    for path in (folder/'controls').glob('*.jsonl.gz'):
        for record in read_records(path):
            if record['scenario']['fixture_category'] == 'D' and 'hindsight' in record:
                hindsight_rows.append({'Method': path.stem.split('.')[0], 'Case': record['scenario']['id'],
                    'Outcome': record['metrics']['outcome'], 'Actions': record['metrics']['actions'],
                    'Clairvoyant actions': record['hindsight']['clairvoyant_minimum_actions'],
                    'Excess if success': record['hindsight']['success_action_excess']})
    lines += [*table(hindsight_rows), '',
        'The reference knows the complete future schedule and is evaluator-only. Excess is undefined for failures. These values do not prove that an observation-only choice was irrational.', '',
        '## What the robot knew in selected failures', '']
    knowledge_rows = []
    for algorithm, result in summary['systems'].items():
        if not result['checkpoints']: continue
        last = result['checkpoints'][-1]
        example_path = folder/algorithm/f"{last['phase']}-{last['training_actions']:05}"/'examples.json'
        if not example_path.exists(): continue
        for case_id, examples in read_json(example_path).items():
            name = next((n for n in ('first_sustained_wait', 'first_attempted_collision', 'first_observed_obstacle') if n in examples), None)
            if name is None: continue
            example = examples[name]; obs = example['observation_before_action']; values = example['decision']
            scores = values.get('action_values', values.get('action_probabilities', []))
            knowledge_rows.append({'Algorithm': algorithm, 'Example': name, 'Action index': example['action_index'],
                'Robot/goal': str([round(v*9) for v in obs[:4]]), 'N/S/E/W rays': str(obs[4:8]),
                'Known cells': sum(v != -1 for v in obs[8:108]), 'Chosen action': example['chosen_action'],
                'U/D/L/R/Wait scores or probabilities': '/'.join(f'{v:.3f}' for v in scores)})
            break
    lines += [*table(knowledge_rows), '',
        'These are bounded illustrative failure links, not a representative sample or a causal diagnosis. They display the actual pre-action observation and checkpoint decision, without an obstacle-type label or future schedule.', '',
        '## Learning diagnostics and competing explanations', '',
        'The fixed reward schedule gives -2 undiscounted return for 200 Wait actions, -5.01 for an immediate collision, and +9.99 for a one-action arrival. Discounted examples at gamma .99 are in `reward-sanity.json`. This shows plausible incentives, not a causal explanation for a learned stall. No reward ablation was run.', '',
        'The engineering gates demonstrate adjacent-goal success/reward, five valid action IDs, unchanged Wait execution, sensor/memory/leakage rules, SARSA next-action/reset semantics, SB3 terminal/truncation bootstrapping, model reload and inference immutability. The zero-linear-weight greedy tie selects Up, while exploratory ties and epsilon exploration sample all five actions. See the validation logs and tests.', '',
        'Successful goal rewards, actual exploratory directions, DQN buffer goal counts, finite updates and state-specific values/probabilities are recorded at each checkpoint. Interpret failures in this order: lack of successful exploration; weak state-dependent value separation or affine interaction capacity; finite training budget; and reward/cap hypotheses. The data can support or weaken these explanations but this single-seed design does not isolate any one cause.', '',
        'The shared feature transform retains all 409 inputs. Immediate versus farther right obstacles are distinguishable. A controlled independent goal/occupancy parallelogram cannot support a strict XOR score difference with an affine model; those synthetic labels are not asserted optimal navigation actions. Therefore this probe illustrates a capacity limitation, not proof that these tasks are unlearnable. Neural capacity and compute differ.', '',
        'DQN continuation uses the declared SB3 cumulative-endpoint exploration schedule, which can rise at a continued learn call; linear exploration uses its declared fixed global schedule. Actual epsilon is logged. Python/NumPy/Torch RNG states are restored around checkpoint reload and assessment so monitoring cannot reseed training.', '',
        '## Scenario composition of the existing generated dataset', '']
    composition = summary.get('composition')
    if composition:
        rows = [{'Measure': key, 'Count': value} for key, value in composition['totals'].items()]
        lines += [composition['scope'], '', *table(rows), '',
                  f"Duration counts: `{composition['duration_counts']}`. Activation-step counts: `{composition['activation_step_counts']}`.", '',
                  'The 16/540 single-cell impact figure counts hypothetical event removals, not episodes requiring Wait. Realized reference trajectories expose different windows. Detour and disconnection flags can both occur in one episode. Routes ending before nominal activation and suppressed ticks remain visible. The unimpeded near-ray counterfactual is geometric proximity, not an observed sensor detection or hidden map supplied to a learner.']
    lines += ['', '## Proposal only: a controlled representation follow-up', '',
        'If weak behavior remains, propose `partial-visible-relative-v2`, retaining raw 409 plus bias (1), relative goal (2), four neighboring occupancy one-hots (12), their ages/visits/streaks (12), and goal-sector (9) by neighbor occupancy (12) conjunctions (108): **644 features** total. Bounds are known; unknown remains distinct; all values derive only from the existing observation. No hidden lifetime/type or true-map feature is permitted.', '',
        'The 644 transform would be identical for all four families, with a matching vectorized neural extractor, versioned checkpoint metadata and leakage/parity tests. Linear matrices become 5 by 644; each 32-unit first neural layer gains 7,424 weights. Feature computation uses four bounded gathers and 108 categorical conjunctions. Register matched interactions, seeds, curriculum, optimizer settings, evaluation cases and capacity/cost reporting before execution. This is a proposal; no such extractor or ablation was trained.', '',
        'A future reward ablation, if warranted, must receive its own reward/task identity and matched controls; this pilot provides no causal reward conclusion. Do not combine a feature and reward change into a claimed single-factor improvement.', '',
        '## Proposal only: future benchmark stratification', '',
        'A new `partial-dynamic-navigation-v2` could pre-register equal scenario quotas for unknown static, permanent-wall detours, initially mandatory Wait, short-detour/long-block, and long-detour/short-block cases. Cross these with familiar versus held-out layout conditions; unseen-dynamic is a condition, not a substitute for mechanism balance. Anchor nominal windows to independently seeded expected arrival ranges, never agent actions, and retain suppression and realized-exposure diagnostics.', '',
        'Reject D4 overlap with all inspected layouts and diagnostic geometry; reserve/seal new final geometry and event manifests before substantive selection. Existing v1 manifests must remain intact. Counts, seeds, horizon and duration distributions require approval and a new registered study; no new dataset was generated here.', '',
        '## Reproduce and inspect', '',
        'From the application directory with its installed environment:', '', '```powershell',
        '.\\.venv\\Scripts\\python.exe -m robot_navigation.partial_commands validate',
        '.\\.venv\\Scripts\\python.exe -m robot_navigation.partial_feasibility register',
        '.\\.venv\\Scripts\\python.exe -m robot_navigation.partial_feasibility run',
        '.\\.venv\\Scripts\\python.exe -m robot_navigation.partial_feasibility audit',
        '.\\.venv\\Scripts\\python.exe -m robot_navigation.partial_feasibility report', '```', '',
        'Registration/execution refuse an existing attempt; reproduce training only in a fresh authorized workspace, not by overwriting evidence or treating retries as free. `audit` is read-only over measured records, apart from a new verification file. It checks hashes, exact replays, checkpoints, saved decisions, immutable training prefixes, summaries and budgets.', '',
        '[Registered machine-readable protocol](../artifacts/partial-dynamic-navigation-v1/feasibility-pilot-v1/phase5a_pilot_protocol.json), '
        '[raw summary](../artifacts/partial-dynamic-navigation-v1/feasibility-pilot-v1/summary.json), '
        '[output hashes](../artifacts/partial-dynamic-navigation-v1/feasibility-pilot-v1/outputs.json), '
        '[replay verification](../artifacts/partial-dynamic-navigation-v1/feasibility-pilot-v1/replay-verification.json), '
        '[per-fixture CSV](../artifacts/partial-dynamic-navigation-v1/feasibility-pilot-v1/latest-fixture-results.csv).', '',
        'The next step is a narrowly registered diagnosis or protocol review according to the verdict. Further training, final reserve creation/evaluation and Phase 6 still need authorization.', '']
    (root/'docs/partial_feasibility_results.md').write_bytes(('\n'.join(lines)).encode('utf-8'))
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for algorithm in summary['systems']:
            values = [v for v in curves if v['algorithm'] == algorithm and v['category'] == 'A']
            axes[0].plot([v['training_actions'] for v in values], [v['successes']/v['episodes'] for v in values], 'o-', label=algorithm)
            axes[1].plot([v['training_actions'] for v in values], [v['wait_fraction'] for v in values], 'o-', label=algorithm)
        for ax in axes: ax.set_xlabel('Actual training actions'); ax.set_ylim(-.05, 1.05); ax.grid(alpha=.2); ax.legend()
        axes[0].set_ylabel('Fixed A success fraction (4 cases)'); axes[1].set_ylabel('Fixed A Wait fraction')
        fig.suptitle('Phase 5A: one diagnostic seed; no uncertainty/generalization claim')
        fig.savefig(folder/'A-learning-diagnostics.png', dpi=180); plt.close(fig)
    return {'status': 'report_written', 'verdict': summary['verdict'], 'rows': len(latest), 'control_rows': len(controls)}
