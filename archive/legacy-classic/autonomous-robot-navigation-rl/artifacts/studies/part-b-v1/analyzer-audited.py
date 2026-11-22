"""Audit the bounded Part B study, paired comparisons, reward control and voting."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

from analyze_part_a import audit_run, failure_type, format_axes
from robot_navigation.agents import load_policy
from robot_navigation.benchmark import load_scenarios
from robot_navigation.core import ROOT, load_task, read_json, write_json
from robot_navigation.episodes import aggregate, validate_replay
from robot_navigation.refinement import reward_task, search_candidates, selection_rank
from robot_navigation.workflow import source_hash

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ALGORITHMS = ("q-learning", "sarsa")
LABELS = {"q-learning": "Q-learning", "sarsa": "SARSA"}
COLORS = {"q-learning": "#2463a6", "sarsa": "#b87816"}
FIELDS = ("success_rate", "collision_rate", "timeout_rate", "spl", "mean_return")


def seed_summary(rows):
    return {"seeds": [r["seed"] for r in rows], "seed_results": [{"seed": r["seed"], "run_id": r["run_id"],
            "selected_interactions": r["selected_interactions"], "validation": r["validation"]} for r in rows],
        "metrics": {field: {"mean": float(np.mean([r["validation"][field] for r in rows])),
                            "sample_std": float(np.std([r["validation"][field] for r in rows], ddof=1))} for field in FIELDS},
        "training_interactions": sum(r["pilot"]["training_interactions"] for r in rows),
        "training_seconds": sum(r["pilot"]["training_seconds"] for r in rows),
        "aggregation_unit": "independent training seed; paired seeds for before/after; not episode-level confidence"}


def paired_changes(before, after, fields=FIELDS):
    lookup = {r["seed"]: r for r in before}
    assert set(lookup) == {r["seed"] for r in after}
    differences = [{"seed": row["seed"], **{field: row["validation"][field] - lookup[row["seed"]]["validation"][field]
                   for field in fields}} for row in after]
    return {"seed_differences": differences,
            "metrics": {field: {"mean_difference": float(np.mean([r[field] for r in differences])),
                                 "sample_std_of_paired_differences": float(np.std([r[field] for r in differences], ddof=1))}
                        for field in fields}}


def write_report(study, rows, summaries, selected, comparisons, reward_comparisons, targets, voting, verification):
    manifest, plan = read_json(study / "manifest.json"), read_json(study / "plan.json")
    lines = ["# Milestone 4 — Part B refinement results", "",
        "Milestones 1–3 were completed and committed before this work. Milestone 4 now implements the bounded search, paired confirmations, reward-only experiment and frozen voting. The next planned milestone is 5: integration and core robustness on the fixed map. The final test remains sealed.", "",
        "## Frozen protocol and actual coverage", "",
        "The search sampled eight distinct settings from the declared finite grid, using RNG seed 20261106. Both Q-learning and SARSA received the same eight settings and development seeds 7/19. Settings were selected by mean complete-validation success, then SPL, seed SD and candidate ID. Checkpoints were selected by the unchanged 30-pair monitor, then evaluated on all 928 validation scenarios.", "",
        "Confirmations and matched original-setting controls used fresh training seeds 101, 211, 307, 401, 503. Every run received 200,000 interactions. The selected reward experiment changed only ordinary step reward from -.01 to 0, using a separate reward version/hash; it did not rewrite maps or scenario manifests. No settings were retuned for that variant.", "",
        f"Completed **{manifest['training_runs']} training runs / {manifest['training_interactions']:,} interactions**: 32 search (6.4m), 10 matched controls (2m), 10 tuned confirmations (2m), and 10 reward ablations (2m). The run phase took **{manifest['wall_seconds']:.2f} seconds ({manifest['wall_seconds']/60:.2f} minutes)**. New training-run artifacts used **{manifest['run_storage_mib']:.2f} MiB**, below the declared 45-minute / 750-MiB limits. Voting reuses already trained members and adds zero training interactions.", "",
        "The full run budget is counted even when monitoring selects an earlier checkpoint. New records retain selected-prefix visitation separately from full-budget visitation. Fresh training seeds check stability on familiar validation tasks; they do not make these independently held-out final-test scores.", "",
        "## Selected settings", "",
        "| Family | Candidate | Learning rate | Discount | Initial epsilon | Final epsilon | Decay fraction |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for algorithm in ALGORITHMS:
        chosen, params = selected[algorithm], selected[algorithm]["parameters"]
        lines.append(f"| {LABELS[algorithm]} | {chosen['candidate_id']} | {params['learning_rate']} | {params['discount']} | {params['epsilon_start']} | {params['epsilon_end']} | {params['epsilon_decay_fraction']} |")
    lines += ["", "Original settings: learning rate .3, discount .99, epsilon 1 → .1 over 80% of budget. All candidates, both development seeds and poor trials remain available in `search_summary.json`; no best-seed selection or extra search followed.", "",
        "## Paired baseline versus tuned confirmations", "",
        "| Family | Baseline success | Tuned success ± seed SD | Paired success change ± SD | Baseline SPL | Tuned SPL |",
        "|---|---:|---:|---:|---:|---:|"]
    for algorithm in ALGORITHMS:
        control, tuned = summaries[algorithm]["control"]["metrics"], summaries[algorithm]["tuned"]["metrics"]
        delta = comparisons[algorithm]["metrics"]["success_rate"]
        lines.append(f"| {LABELS[algorithm]} | {100*control['success_rate']['mean']:.2f}% | {100*tuned['success_rate']['mean']:.2f}% ± {100*tuned['success_rate']['sample_std']:.2f} pp | {100*delta['mean_difference']:+.2f} ± {100*delta['sample_std_of_paired_differences']:.2f} pp | {control['spl']['mean']:.6f} | {tuned['spl']['mean']:.6f} |")
    lines += ["", "SD is sample SD over five independent training seeds, or over five paired seed differences as labeled. It is not an episode-level confidence interval or a significance claim. The matched controls use the same fresh seeds, budget, monitor, checkpoint rule and complete-validation cases as tuned runs; historical Part A means are kept separate.", "",
        "| Family | Seed | Baseline goals / 928 | Tuned goals / 928 | Baseline SPL | Tuned SPL |",
        "|---|---:|---:|---:|---:|---:|"]
    for algorithm in ALGORITHMS:
        base = {r["seed"]: r for r in summaries[algorithm]["control"]["seed_results"]}
        for row in summaries[algorithm]["tuned"]["seed_results"]:
            before = base[row["seed"]]["validation"]
            after = row["validation"]
            lines.append(f"| {LABELS[algorithm]} | {row['seed']} | {before['counts']['success']} | {after['counts']['success']} | {before['spl']:.6f} | {after['spl']:.6f} |")
    lines += ["", "### Predeclared targets", "",
        "Targets were success >=80%, collision <=2%, and SPL >=.75 for each tuned family's five-seed mean. They were saved before search and were not changed afterward.", ""]
    for algorithm in ALGORITHMS:
        missed = [name for name, met in targets[algorithm].items() if not met]
        metrics = summaries[algorithm]["tuned"]["metrics"]
        lines.append(f"- {LABELS[algorithm]}: success {100*metrics['success_rate']['mean']:.2f}%, collision {100*metrics['collision_rate']['mean']:.2f}%, SPL {metrics['spl']['mean']:.6f}. " + (f"Missed targets: {', '.join(missed)}." if missed else "All three mean targets met."))
    lines += ["", "Missed targets and poor individual seeds are retained. No model is described as ready for unseen maps or physical robots.", "",
        "## Reward-only ablation", "",
        "| Family | Ordinary step reward | Success | Collision | Timeout | SPL | Mean return (version-specific) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for algorithm in ALGORITHMS:
        for stage, reward in (("tuned", "-.01"), ("terminal", "0")):
            metrics = summaries[algorithm][stage]["metrics"]
            lines.append(f"| {LABELS[algorithm]} | {reward} | {100*metrics['success_rate']['mean']:.2f}% | {100*metrics['collision_rate']['mean']:.2f}% | {100*metrics['timeout_rate']['mean']:.2f}% | {metrics['spl']['mean']:.6f} | {metrics['mean_return']['mean']:.6f} |")
    lines += ["", "Goal remains +10 and collision -5; ordinary timeout has no extra penalty. Returns across these reward scales are not a navigation-quality ranking. `reward_ablation.json` reports paired success/collision/timeout/SPL differences and deliberately excludes return differences. The variant is diagnostic and was not automatically promoted or combined with another search.", "",
        "## Frozen voting experiment", "",
        f"The original-reward tuned confirmation means selected **{LABELS[voting['family']]}** before reward ablation/voting. Members are the predeclared seeds {voting['member_seeds']} in that fixed priority order. A majority wins; a three-way tie uses the first member. No action masking, fallback, Q-value averaging or voting training occurs. The selected single is seed {voting['selected_single_seed']}, not a cherry-picked best seed.", "",
        "| System | Success / 928 | Collision | Timeout | SPL |",
        "|---|---:|---:|---:|---:|"]
    for name, result in list(voting["members"].items()) + [("Voting", voting["voting"])]:
        lines.append(f"| {name} | {result['counts']['success']} | {result['counts']['collision']} | {result['counts']['timeout']} | {result['spl']:.6f} |")
    single = voting["selected_single"]
    delta = voting["voting"]["success_rate"] - single["success_rate"]
    lines += ["", f"Voting changed success by **{100*delta:+.2f} percentage points** and SPL by **{voting['voting']['spl']-single['spl']:+.6f}** versus the predeclared single. Comparisons against every member are in `voting_comparison.json`; this is one fixed ensemble, not an ensemble-seed uncertainty study. Component metrics are reused on the same 928 scenarios and are not counted as extra independent samples.", "",
        f"There were {voting['three_way_tie_actions']:,} three-way ties in {voting['total_voting_actions']:,} voting actions, affecting {voting['episodes_with_three_way_ties']} episodes. Replays retain all member proposals and tie decisions, with no invented Q scores.", "",
        f"Q-array memory is {voting['voting_table_memory_bytes']:,} bytes for voting versus {voting['single_table_memory_bytes']:,} for one policy (excluding Python overhead). Compressed member checkpoint files total {voting['voting_member_checkpoint_bytes']:,} bytes versus {voting['single_checkpoint_bytes']:,} for the selected single, excluding metadata/descriptors. Members consumed {voting['member_training_interactions']:,} interactions and {voting['member_training_seconds']:.2f} training-loop seconds; that cost is already included in this study and is not added again. The wrapper adds zero training.", "",
        "| Policy | Mean decision microseconds | Median decision microseconds |", "|---|---:|---:|"]
    for name, timing in voting["timing"].items():
        lines.append(f"| {name} | {timing['mean_microseconds']:.3f} | {timing['median_microseconds']:.3f} |")
    lines += ["", "Timing uses 1,000 round-robin validation initial-state queries per policy, with `perf_counter_ns`, excluding resets/rendering/learning. These are local measurements under concurrent machine activity, not a hardware-isolated latency guarantee.", "",
        "## Verification and preserved scope", "",
        f"Audited **{verification['learned_episode_records_verified']:,} training-run validation episodes / {verification['learned_actions_verified']:,} actions**, **{verification['periodic_checkpoints_verified']:,} periodic models**, and **{verification['voting_episodes_verified']:,} voting episodes / {verification['voting_actions_verified']:,} actions**. Checks reconcile source/task/model signatures, parent scenario relationships, training counts, checkpoint rules, selected-prefix visitation and greedy scores/actions. The voting audit recomputes every member proposal/tie and verifies no parameter changes.", "",
        "All benchmark inputs, original Part A baselines and earlier artifacts were preserved. Final-test policy executions: zero. Exact training resume remains unsupported; snapshots are inference-only, and failed/interrupted status is explicit. Targets, candidate count, seeds and scope were not adjusted after outcomes.", "",
        "The minimal viewer loads matching step-cost/terminal-only tasks and voting descriptors, labels reward versions/member order, and never trains. Tests and browser-specific observations are recorded separately in `milestone4_verification.json` and `milestone4_ui_check.json`. The retained practical playback target is 2 actions/second; no new quantitative browser timing benchmark is claimed.", "",
        "Evidence: [summary](../artifacts/studies/part-b-v1/summary.json), [all search candidates](../artifacts/studies/part-b-v1/search_summary.json), [paired changes](../artifacts/studies/part-b-v1/paired_improvements.json), [reward ablation](../artifacts/studies/part-b-v1/reward_ablation.json), [voting and cost](../artifacts/studies/part-b-v1/voting_comparison.json), [audits](../artifacts/studies/part-b-v1/verification.json), [figures](../artifacts/studies/part-b-v1/figures/), and [protocol](milestone4_protocol.md).", "",
        "## Remaining work", "",
        "Results remain selected development validation on a familiar map. Inspect residual timeouts, collisions, poor seeds and reward sensitivity before final freezing. Milestone 5 adds the integrated dashboard and predefined core robustness groups; milestone 6 freezes/evaluates the final system. This milestone does not execute final test, automatically publish/deploy, or add PPO, unseen maps, moving obstacles, continuous control, hardware or the full dashboard.", ""]
    report = "\n".join(lines)
    local_report = report.replace("../artifacts/studies/part-b-v1/", "./").replace(
        "[protocol](milestone4_protocol.md)", "[protocol](protocol.md)")
    (study / "report.md").write_text(local_report, encoding="utf-8")
    if study.name == "part-b-v1":
        (ROOT / "docs/milestone4_results.md").write_text(report, encoding="utf-8")


def plot_results(study, rows, summaries, selected, voting):
    folder = study / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, algorithm in zip(axes, ALGORITHMS):
        for index in range(8):
            candidate = f"c{index+1:02d}"
            values = [r["validation"]["success_rate"] for r in rows if r["algorithm"] == algorithm and r["candidate_id"] == candidate]
            ax.scatter([index-.04, index+.04], values, marker="x", color=COLORS[algorithm], s=30)
            ax.plot(index, np.mean(values), "o", color=COLORS[algorithm], markersize=7,
                    markerfacecolor="white" if candidate != selected[algorithm]["candidate_id"] else COLORS[algorithm])
        ax.set(title=f"{LABELS[algorithm]} · development search", xticks=range(8),
               xticklabels=[f"c{i:02d}" for i in range(1, 9)], ylim=(0, 1.02), ylabel="Validation success")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        format_axes(ax)
    fig.supxlabel("Eight frozen candidates, two development seeds each; crosses = seed values, circles = means; filled circle = selected setting", fontsize=9)
    fig.savefig(folder / "search.png", dpi=150)
    plt.close(fig)

    def paired_figure(stages, labels, filename, caption):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for ax, field, title in zip(axes, ("success_rate", "spl"), ("Validation success", "Validation SPL")):
            for index, algorithm in enumerate(ALGORITHMS):
                left = [r for r in rows if r["algorithm"] == algorithm and r["stage"] == stages[0]]
                right = {r["seed"]: r for r in rows if r["algorithm"] == algorithm and r["stage"] == stages[1]}
                for row in left:
                    ax.plot([2*index, 2*index+1], [row["validation"][field], right[row["seed"]]["validation"][field]],
                            color=COLORS[algorithm], linewidth=.8, alpha=.45, marker="x")
                for offset, stage in enumerate(stages):
                    group = [r["validation"][field] for r in rows if r["algorithm"] == algorithm and r["stage"] == stage]
                    ax.plot(2*index+offset, np.mean(group), "o", color=COLORS[algorithm], markersize=8,
                            markerfacecolor="white" if offset == 0 else COLORS[algorithm])
            ax.set(title=title, ylim=(0, 1.02), xticks=range(4),
                   xticklabels=[f"{LABELS[a]}\n{label}" for a in ALGORITHMS for label in labels])
            format_axes(ax)
        axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        fig.supxlabel(caption, fontsize=9)
        fig.savefig(folder / filename, dpi=150)
        plt.close(fig)
    paired_figure(("control", "tuned"), ("Baseline", "Tuned"), "paired_refinement.png",
                  "Five fresh paired training seeds; lines = seed pairs, circles = group means. Same task, budget and checkpoint-selection rule.")
    paired_figure(("tuned", "terminal"), ("Step -.01", "Step 0"), "reward_ablation.png",
                  "Only ordinary step reward changes; fixed tuned settings and five paired seeds. Compare success/SPL, not raw returns across reward scales.")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for i, algorithm in enumerate(ALGORITHMS):
        for stage, color, style, label in (("control", "#667085", "--", "Baseline control"),
                                          ("tuned", COLORS[algorithm], "-", "Tuned, step cost"),
                                          ("terminal", "#6c7d3d", ":", "Tuned, terminal only")):
            curves = [read_json(ROOT / "artifacts/runs" / r["run_id"] / "monitoring.json")
                      for r in rows if r["algorithm"] == algorithm and r["stage"] == stage]
            x = [m["training_interactions"] for m in curves[0]]
            for j, field in enumerate(("success_rate", "spl")):
                values = np.array([[m[field] for m in curve] for curve in curves])
                mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
                axes[i, j].plot(x, mean, style, color=color, label=label, linewidth=2)
                axes[i, j].fill_between(x, np.maximum(0, mean-sd), np.minimum(1, mean+sd), color=color, alpha=.1)
                axes[i, j].set(title=f"{LABELS[algorithm]} · {'Monitoring success' if j==0 else 'Monitoring SPL'}",
                               xlabel="Training interactions", ylim=(-.02, 1.02))
                axes[i, j].ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
                if j == 0:
                    axes[i, j].yaxis.set_major_formatter(PercentFormatter(1))
                format_axes(axes[i, j])
        axes[i, 1].legend(frameon=False, loc="upper left", fontsize=9)
    fig.supxlabel("Fixed 30-pair validation subset; five-seed means ±1 sample SD clipped to [0,1], not confidence intervals", fontsize=9)
    fig.savefig(folder / "learning_curves.png", dpi=150)
    plt.close(fig)

    names = list(voting["members"]) + ["voting"]
    values = list(voting["members"].values()) + [voting["voting"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, field, title in zip(axes, ("success_rate", "spl"), ("Frozen-policy validation success", "Frozen-policy validation SPL")):
        for i, (name, value) in enumerate(zip(names, values)):
            ax.bar(i, value[field], color=COLORS[voting["family"]] if name == "voting" else "#d7e0e8", edgecolor="#344054", width=.6)
            ax.text(i, value[field]+.025, f"{value[field]:.3f}", ha="center", fontsize=9)
        ticklabels = [f"Seed {seed}" + ("\nselected single" if seed == voting["selected_single_seed"] else "")
                      for seed in voting["member_seeds"]] + ["Voting"]
        ax.set(title=title, ylim=(0, 1.1), xticks=range(4), xticklabels=ticklabels)
        format_axes(ax)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    fig.supxlabel("Same 928 validation pairs; one predeclared three-policy vote, no ensemble uncertainty estimate. Member metrics are reused, not new samples.", fontsize=9)
    fig.savefig(folder / "voting.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Audit and compare a complete Part B refinement study")
    parser.add_argument("--suite-id", default="part-b-v1")
    args = parser.parse_args()
    if Path(args.suite_id).name != args.suite_id:
        parser.error("suite-id must be one directory name")
    study = ROOT / "artifacts/studies" / args.suite_id
    manifest, plan = read_json(study / "manifest.json"), read_json(study / "plan.json")
    assert manifest["status"] == "complete" and not manifest["final_test_evaluated"]
    rows = read_json(study / "run_results.json")
    assert len(rows) == plan["limits"]["training_runs"] == 62
    candidates = read_json(study / "candidates.json")
    assert candidates == search_candidates(plan)
    task, scenarios = load_task(), load_scenarios(load_task(), "validation")
    audits, taxonomy, metrics_lookup = [], defaultdict(Counter), {}
    for i, row in enumerate(rows, 1):
        derived_task = reward_task(task, row["reward_variant"])
        episodes, audit = audit_run(derived_task, row, scenarios)
        directory = ROOT / "artifacts/runs" / row["run_id"]
        with np.load(directory / "selected_visitation.npz", allow_pickle=False) as file:
            visits = file["state_action_counts"]
        assert int(visits.sum()) == row["selected_interactions"]
        overlap = sum(bool(visits[(*s["start"], *s["goal"])].sum()) for s in scenarios)
        assert overlap == row["validation"]["initial_states_visited_in_training"]
        audit["selected_prefix_visitation_verified"] = True
        audits.append(audit)
        if row["stage"] != "search":
            metrics_lookup[(row["algorithm"], row["seed"], row["stage"])] = [e["metrics"] for e in episodes]
            taxonomy[f"{row['algorithm']}/{row['stage']}"] += Counter(failure_type(e, derived_task.grid) for e in episodes)
        if i % 8 == 0 or i == len(rows):
            print(f"Audited {i}/{len(rows)} training runs", flush=True)
    selected = read_json(study / "selected_configurations.json")
    search_summary = {}
    for algorithm in ALGORITHMS:
        values = {}
        for candidate in candidates:
            group = [r for r in rows if r["stage"] == "search" and r["algorithm"] == algorithm and r["candidate_id"] == candidate["candidate_id"]]
            assert [r["seed"] for r in group] == plan["development_seeds"]
            values[candidate["candidate_id"]] = {"parameters": candidate["parameters"], **seed_summary(group)}
        chosen = min(candidates, key=lambda c: selection_rank(
            [r for r in rows if r["stage"] == "search" and r["algorithm"] == algorithm and r["candidate_id"] == c["candidate_id"]], c["candidate_id"]))
        assert chosen == selected[algorithm]
        search_summary[algorithm] = values
    summaries, comparisons, reward_comparisons, targets = {}, {}, {}, {}
    for algorithm in ALGORITHMS:
        groups = {stage: [r for r in rows if r["algorithm"] == algorithm and r["stage"] == stage]
                  for stage in ("control", "tuned", "terminal")}
        for group in groups.values():
            assert [r["seed"] for r in group] == plan["confirmation_seeds"]
        for row in groups["tuned"] + groups["terminal"]:
            assert row["parameters"] == selected[algorithm]["parameters"]
        for row in groups["control"]:
            assert row["parameters"] == plan["baseline"]
        summaries[algorithm] = {stage: seed_summary(group) for stage, group in groups.items()}
        comparisons[algorithm] = paired_changes(groups["control"], groups["tuned"])
        reward_comparisons[algorithm] = paired_changes(groups["tuned"], groups["terminal"], FIELDS[:-1])
        means = summaries[algorithm]["tuned"]["metrics"]
        targets[algorithm] = {"success": means["success_rate"]["mean"] >= plan["targets"]["success_rate_at_least"],
            "collision": means["collision_rate"]["mean"] <= plan["targets"]["collision_rate_at_most"],
            "spl": means["spl"]["mean"] >= plan["targets"]["spl_at_least"]}
    choice = read_json(study / "family_choice_before_voting.json")
    family = min(ALGORITHMS, key=lambda a: (-summaries[a]["tuned"]["metrics"]["success_rate"]["mean"],
        -summaries[a]["tuned"]["metrics"]["spl"]["mean"], summaries[a]["tuned"]["metrics"]["success_rate"]["sample_std"],
        summaries[a]["tuned"]["training_seconds"], a))
    assert choice["family"] == family
    voting = read_json(study / "voting_comparison.json")
    vote_dir = ROOT / "artifacts/runs" / voting["run_id"]
    assert source_hash(vote_dir / "source") == read_json(vote_dir / "manifest.json")["source_hash"]
    vote = load_policy(vote_dir / "voting.json", task)
    assert list(vote.seeds) == plan["voting_member_seeds"] and voting["family"] == family
    before, vote_metrics, vote_actions = vote.state_fingerprint(), [], 0
    with gzip.open(vote_dir / "voting_trajectories.jsonl.gz", "rt", encoding="utf-8") as file:
        for expected, line in zip(scenarios, file, strict=True):
            episode = json.loads(line)
            assert episode["scenario"] == expected
            validate_replay(task, episode)
            for transition in episode["transitions"]:
                decision = vote.decision(transition["observation"])
                assert transition["decision"] == decision and transition["action"] == decision["action"]
                assert transition["scores"] is None
            vote_metrics.append(episode["metrics"])
            vote_actions += len(episode["transitions"])
    assert vote.state_fingerprint() == before
    for field, value in aggregate(vote_metrics).items():
        assert voting["voting"][field] == value
    for seed in plan["voting_member_seeds"]:
        row = next(r for r in rows if r["algorithm"] == family and r["stage"] == "tuned" and r["seed"] == seed)
        assert voting["members"][f"member-seed{seed}"] == row["validation"]
    write_json(study / "summary.json", summaries)
    write_json(study / "search_summary.json", search_summary)
    write_json(study / "paired_improvements.json", comparisons)
    write_json(study / "reward_ablation.json", {"paired_changes_excluding_return": reward_comparisons,
        "returns_comparable_across_versions": False, "only_changed_factor": "step reward -.01 to 0; version/hash explicitly distinct"})
    write_json(study / "target_assessment.json", {"targets": plan["targets"], "tuned_family_mean_targets_met": targets})
    write_json(study / "failure_analysis.json", dict(taxonomy))
    with (study / "paired_scenarios.csv").open("w", encoding="utf-8", newline="") as file:
        fields = ["algorithm", "seed", "scenario_id", "distance_band"]
        fields += [f"{stage}_{field}" for stage in ("control", "tuned", "terminal") for field in ("outcome", "steps", "spl")]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for algorithm in ALGORITHMS:
            for seed in plan["confirmation_seeds"]:
                for index, scenario in enumerate(scenarios):
                    record = {"algorithm": algorithm, "seed": seed, "scenario_id": scenario["id"], "distance_band": scenario["distance_band"]}
                    for stage in ("control", "tuned", "terminal"):
                        metric = metrics_lookup[(algorithm, seed, stage)][index]
                        record.update({f"{stage}_{field}": metric[field] for field in ("outcome", "steps", "spl")})
                    writer.writerow(record)
    plot_results(study, rows, summaries, selected, voting)
    verification = {"audits": audits,
        "learned_episode_records_verified": sum(a["episodes_verified"] for a in audits),
        "learned_actions_verified": sum(a["actions_verified"] for a in audits),
        "periodic_checkpoints_verified": sum(a["checkpoint_count_verified"] for a in audits),
        "voting_episodes_verified": len(vote_metrics), "voting_actions_verified": vote_actions,
        "selected_prefix_visitation_verified": True, "candidate_and_family_selection_verified": True,
        "voting_inference_state_unchanged": True, "final_test_policy_executions": 0,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(study / "verification.json", verification)
    write_report(study, rows, summaries, selected, comparisons, reward_comparisons, targets, voting, verification)
    print(json.dumps({"selected_configurations": selected, "targets": targets,
        "confirmation_means": {a: {stage: value["metrics"] for stage, value in group.items()} for a, group in summaries.items()},
        "voting_counts": voting["voting"]["counts"]}, indent=2))


if __name__ == "__main__":
    main()
