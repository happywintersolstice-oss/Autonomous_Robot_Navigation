"""Generate the current warehouse report from verified saved results; never retrain."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from robot_navigation.core import ROOT
from robot_navigation.warehouse_study import FAMILIES,TITLE,checked_run,saved


def main():
    result=saved()
    lines=[f'# {TITLE}', '', 'Measured results, 19 November 2026.', '',
        'The registered twelve-run study completed **7.2 million training actions**. Every final checkpoint was reloaded and assessed on the same 188 development routes. All 2,256 recorded assessment episodes passed action/dynamics/metric audits. Three independent training seeds contribute equally to each method average. No checkpoint or seed was selected for a favorable result.', '',
        '## Completion and route efficiency', '',
        '| Algorithm | Completion, mean +/- seed SD | Collision, mean +/- seed SD | Timeout, mean +/- seed SD | SPL, mean +/- seed SD |',
        '| --- | --- | --- | --- | --- |']
    for algorithm,system in result['systems'].items():
        m=system['metrics']
        cells=[f"{100*m[k]['mean']:.2f}% +/- {100*m[k]['sample_sd']:.2f} pp" for k in ('success_rate','collision_rate','timeout_rate')]
        cells.append(f"{m['spl']['mean']:.4f} +/- {m['spl']['sample_sd']:.4f}")
        lines.append('| '+system['label']+' | '+' | '.join(cells)+' |')
    lines+=['', '![Completion and SPL across independent training seeds](evidence/warehouse-comparison-overview.png)', '',
        'Bars show equal-weight seed means; error bars show sample SD, and dots show individual seeds. SD is not a confidence interval. The same route cases recur for each run; they are not independent training replicates.', '',
        '## Every training seed', '',
        '| Algorithm | Seed | Completed / 188 | Collision / 188 | Timeout / 188 | SPL | Repeating-position timeouts |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for algorithm,system in result['systems'].items():
        for seed in system['seeds']:
            run,_=checked_run(algorithm,seed)
            m=run['assessment'];c=m['counts']
            lines.append(f"| {system['label']} | {seed} | {c['success']} | {c['collision']} | {c['timeout']} | {m['spl']:.4f} | {run['looping_timeouts']} |")
    ranking=sorted(result['systems'].values(),key=lambda s:s['metrics']['success_rate']['mean'],reverse=True)
    lines+=['', '## Interpretation', '',
        f"**{ranking[0]['label']} has the highest measured mean completion under this registered setup.** This describes the reported floor, coordinate representation, action budget and settings; it does not establish that the algorithm is universally best. No post-result tuning was performed.", '',
        'The floor is small and finite, so an explicit state-action table can represent its robot/goal combinations directly. Neural methods must approximate useful values or actions from scaled coordinates without an obstacle channel. Different exploration/update rules and terminal collisions can produce different learning progress. These are plausible explanations, rather than effects isolated by this experiment.', '',
        'The common -0.05 step cost discourages unnecessary movement, but deterministic policies can still repeat states and time out. The timeout counter records any repeated position within a timed-out episode; it does not classify a specific cycle or prove that reward design caused it. A causal reward claim would require matched reward ablations with independently registered settings.', '',
        'The 35-case monitoring curves describe training progress. A later checkpoint can perform worse than an earlier one; the fixed 600,000-action checkpoint remains in the comparison. All failures can be inspected through Simulator > Browse every assessed route.', '',
        '## Compute and reproducibility', '',
        '| Algorithm | Mean recorded training seconds | Mean final checkpoint bytes | Mean recorded decision microseconds |',
        '| --- | --- | --- | --- |']
    for algorithm,system in result['systems'].items():
        cost=system['costs']
        lines.append(f"| {system['label']} | {cost['training_seconds']['mean']:.2f} | {cost['checkpoint_bytes']['mean']:.0f} | {cost['decision_microseconds']['mean']:.2f} |")
    lines+=['',
        'At most two CPU workers ran concurrently, with one Torch thread per neural worker. Equal environment-action budgets do not imply equal gradient updates, compute or hyperparameter tuning. Timings are descriptive, not isolated hardware benchmarks. Tabular timing excludes setup; neural recorded elapsed time includes setup and excludes measured monitoring. The reused neural ledger metadata contains a generic construction-exclusion label; this report follows the actual warehouse runner timing scope.', '',
        'The [protocol](warehouse_comparison_protocol.md) and [configuration](../configs/warehouse_comparison.yaml) were frozen before the matrix ran. `freeze.json` and `seal.json` retain exact engine, environment and split hashes. Each run saves final weights, training ledgers, monitoring, six actual training episodes, all 188 assessment trajectories and output checksums. `published.json` pins the [summary](../artifacts/warehouse/warehouse-four-algorithm-v1/summary.json) and [comparison CSV](../artifacts/warehouse/warehouse-four-algorithm-v1/comparison.csv). The earlier seed-19301 warehouse pilot is excluded.', '',
        '## Evidence boundaries', '',
        'These are **previously inspected development routes on one familiar warehouse**. Initial training and assessment pairs are disjoint, but trajectories can visit overlapping states. Named jobs and the monitoring subset are part of development assessment. This is not a fresh final holdout, unseen-warehouse evaluation or physical-robot safety assessment. Three seeds provide limited variability evidence and no statistical significance claim.', '',
        'The point-robot simulation has four grid moves and static fixtures. It does not model loading parcels, robot footprint, sensors, continuous motion, people, moving obstacles or battery use. Temporary 3D preview changes rendering only. Historical final cohorts were neither rerun nor overwritten.', '',
        'For operation and implementation details, see the [user and code guide](../../docs/user-guide.md), [technical report](technical_report.md) and [verification guide](../../docs/developer-guide.md).', '']
    (ROOT/'docs/warehouse_comparison_results.md').write_bytes(('\n'.join(lines)).encode('utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    colors=['#147d92','#bb7c19','#6e5cba','#3c8b58']
    for ax,key,label,scale in zip(axes,('success_rate','spl'),('Route completion (%)','Route efficiency (SPL)'),(100,1)):
        systems=list(result['systems'].values());x=np.arange(4)
        means=[s['metrics'][key]['mean']*scale for s in systems]
        errors=[s['metrics'][key]['sample_sd']*scale for s in systems]
        ax.bar(x,means,yerr=errors,capsize=5,color=colors,alpha=.75)
        for i,s in enumerate(systems):
            ys=[r['assessment'][key]*scale for r in s['seed_results']]
            ax.scatter(i+np.linspace(-.12,.12,3),ys,color='#122835',s=25,zorder=4)
        ax.set_xticks(x,[s['label'] for s in systems]);ax.set_ylabel(label)
        ax.set_ylim(0,115 if scale==100 else 1.15);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Warehouse navigation: three seeds per algorithm, 600,000 actions per run')
    fig.savefig(ROOT/'docs/evidence/warehouse-comparison-overview.png',dpi=180)
    plt.close(fig)
    print('Updated docs/warehouse_comparison_results.md and comparison figure from verified publication.')


if __name__=='__main__':
    main()
