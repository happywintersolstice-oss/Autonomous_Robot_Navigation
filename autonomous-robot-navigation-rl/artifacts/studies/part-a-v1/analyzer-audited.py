"""Audit saved Part A evidence, generate research figures and rank families."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

from robot_navigation.agents import load_policy, viewer_checkpoint
from robot_navigation.benchmark import load_scenarios
from robot_navigation.core import ROOT, load_task, read_json, write_json
from robot_navigation.episodes import aggregate, validate_replay
from robot_navigation.visualization import save_figure
from robot_navigation.workflow import source_hash

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

LABELS = {"q-learning": "Q-learning", "sarsa": "SARSA", "dqn": "DQN"}
COLORS = {"q-learning": "#2463a6", "sarsa": "#b87816", "dqn": "#ae4c79"}
STYLES = {"q-learning": "-", "sarsa": "--", "dqn": ":"}


def failure_type(episode, grid):
    outcome, steps = episode["metrics"]["outcome"], episode["metrics"]["steps"]
    if outcome == "success":
        return "success_optimal" if steps == episode["scenario"]["shortest_length"] else "success_detour"
    if outcome == "collision":
        r, c = episode["transitions"][-1]["info"]["attempted_position"]
        return "collision_boundary" if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) else "collision_obstacle"
    positions = [tuple(episode["initial_observation"][:2])] + [tuple(t["next_observation"][:2]) for t in episode["transitions"]]
    return "timeout_repeated_state" if len(set(positions)) < len(positions) else "timeout_without_revisit"


def family_summary(rows):
    fields = ("success_rate", "collision_rate", "timeout_rate", "spl", "mean_return")
    return {"seeds": [r["seed"] for r in rows], "seed_results": rows,
        "metrics": {field: {"mean": float(np.mean([r["validation"][field] for r in rows])),
                            "sample_std": float(np.std([r["validation"][field] for r in rows], ddof=1))} for field in fields},
        "total_training_seconds": sum(r["pilot"]["training_seconds"] for r in rows),
        "aggregation_unit": "independent training seed; five seeds, not episode-level uncertainty"}


def rank_families(summaries):
    return sorted(summaries, key=lambda a: (-summaries[a]["metrics"]["success_rate"]["mean"],
        -summaries[a]["metrics"]["spl"]["mean"], summaries[a]["metrics"]["success_rate"]["sample_std"],
        summaries[a]["total_training_seconds"], a))


def audit_run(task, row, expected_scenarios):
    run = ROOT / "artifacts/runs" / row["run_id"]
    manifest = read_json(run / "manifest.json")
    if manifest["status"] != "complete" or manifest["final_test_evaluated"]:
        raise AssertionError("Only complete validation-only runs can enter comparison")
    if source_hash(run / "source") != manifest["source_hash"]:
        raise AssertionError("Source snapshot mismatch")
    policy = load_policy(viewer_checkpoint(run), task)
    fingerprint = policy.state_fingerprint() if hasattr(policy, "state_fingerprint") else policy.q.copy()
    key = row["algorithm"].replace("-", "_")
    score_cache, metrics, episodes = {}, [], []
    with gzip.open(run / f"{key}_trajectories.jsonl.gz", "rt", encoding="utf-8") as file:
        for expected, line in zip(expected_scenarios, file, strict=True):
            episode = json.loads(line)
            assert episode["scenario"] == expected and episode["mode"] == "validation"
            validate_replay(task, episode)
            for transition in episode["transitions"]:
                state = tuple(transition["observation"])
                if state not in score_cache:
                    score_cache[state] = policy.scores(state).tolist()
                assert transition["scores"] == score_cache[state]
                assert transition["action"] == int(np.argmax(score_cache[state]))
            metrics.append(episode["metrics"])
            episodes.append(episode)
    expected_metrics = read_json(run / f"{key}_episodes.json")
    assert metrics == [{k: v for k, v in m.items() if k != "scenario_id"} for m in expected_metrics]
    for field, value in aggregate(metrics).items():
        assert row["validation"][field] == value
    if hasattr(policy, "state_fingerprint"):
        assert policy.state_fingerprint() == fingerprint
    else:
        assert np.array_equal(policy.q, fingerprint)
    with (run / "training.csv").open(newline="", encoding="utf-8") as file:
        logged = sum(int(r["steps"]) for r in csv.DictReader(file))
    partial = row["pilot"]["incomplete_budget_cutoff_episode"]
    with np.load(run / "visitation.npz", allow_pickle=False) as file:
        visits = int(file["state_action_counts"].sum())
    assert visits == logged + (partial["steps"] if partial else 0) == 200000
    monitored = read_json(run / "monitoring.json")
    best = max(monitored, key=lambda m: (m["success_rate"], m["spl"], -m["training_interactions"]))
    assert best["training_interactions"] == row["selected_interactions"]
    checkpoints = list((run / "checkpoints").glob("*.npz")) + list((run / "checkpoints").glob("*.zip"))
    assert len(checkpoints) == 20
    for path in checkpoints:
        load_policy(path, task)
    return episodes, {"run_id": run.name, "episodes_verified": len(episodes),
        "actions_verified": sum(m["steps"] for m in metrics), "training_interactions_reconciled": visits,
        "checkpoint_count_verified": len(checkpoints), "source_snapshot_matches": True,
        "selected_checkpoint_rule_verified": True, "inference_state_unchanged": True}


def format_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9dee5", linewidth=.6)
    ax.set_axisbelow(True)


def figures(study, rows, summaries, taxonomy, heatmaps, lengths, common_episodes):
    folder = study / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    curve_data = {}
    for algorithm in LABELS:
        curves = [read_json(ROOT / "artifacts/runs" / r["run_id"] / "monitoring.json") for r in rows if r["algorithm"] == algorithm]
        x = [m["training_interactions"] for m in curves[0]]
        assert all([m["training_interactions"] for m in curve] == x for curve in curves)
        curve_data[algorithm] = {"training_interactions": x}
        for ax, field, label in zip(axes, ("success_rate", "spl"), ("Monitoring success", "Monitoring SPL")):
            values = np.array([[m[field] for m in curve] for curve in curves])
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            curve_data[algorithm][field] = {"seed_values": values.tolist(), "mean": mean.tolist(), "sample_std": sd.tolist()}
            ax.plot(x, mean, STYLES[algorithm], color=COLORS[algorithm], label=LABELS[algorithm], linewidth=2)
            ax.fill_between(x, np.maximum(0, mean - sd), np.minimum(1, mean + sd), color=COLORS[algorithm], alpha=.12)
            ax.set(title=label, xlabel="Training environment interactions", ylim=(-.02, 1.02))
            ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
            format_axes(ax)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(frameon=False, loc="upper left")
    fig.supxlabel("Fixed 30-scenario validation subset; lines = five-seed means, bands = ±1 sample SD (clipped to [0,1]); not confidence intervals", fontsize=9)
    fig.savefig(folder / "learning_curves.png", dpi=150)
    plt.close(fig)
    write_json(study / "curve_data.json", curve_data)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, field, title in zip(axes, ("success_rate", "spl"), ("Full-validation success", "Full-validation SPL")):
        for index, algorithm in enumerate(LABELS):
            data = [r["validation"][field] for r in rows if r["algorithm"] == algorithm]
            ax.errorbar(index, np.mean(data), yerr=np.std(data, ddof=1), fmt="o", color=COLORS[algorithm], capsize=6, markersize=8)
            ax.scatter(index + np.linspace(-.08, .08, len(data)), data, color=COLORS[algorithm], marker="x", s=28)
        ax.axhline(1, color="#4b5563", linestyle="--", linewidth=1, label="BFS reference")
        ax.set(xticks=range(3), xticklabels=list(LABELS.values()), title=title, ylim=(-.02, 1.06))
        format_axes(ax)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    fig.supxlabel("928 scenarios per seed; validation-selected checkpoints; dots ± sample SD over five training seeds, crosses = individual seeds", fontsize=9)
    fig.savefig(folder / "validation_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    for col, algorithm in enumerate(LABELS):
        for row, field, title in ((0, "collision_robot_cell", "Collisions by robot cell"), (1, "timeout_start", "Timeouts by start cell")):
            data = np.array(heatmaps[algorithm][field])
            image = axes[row, col].imshow(data, cmap="Blues", vmin=0, vmax=max(1, int(data.max())))
            axes[row, col].set(title=f"{LABELS[algorithm]} · {title}", xlabel="Column", ylabel="Row", xticks=range(0, 10, 2), yticks=range(0, 10, 2))
            fig.colorbar(image, ax=axes[row, col], label="Episodes (all five seeds)", shrink=.8)
    fig.supxlabel("Actual count per cell; each panel has its own labeled count scale; collisions retain the robot's actual cell", fontsize=9)
    fig.savefig(folder / "failure_heatmaps.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    max_length = max((max(v) for v in lengths.values() if v), default=1)
    bins = np.arange(.5, max_length + 1.5, 1)
    for ax, algorithm in zip(axes, LABELS):
        ax.hist(lengths[algorithm], bins=bins, color=COLORS[algorithm], edgecolor="white")
        ax.set(title=f"{LABELS[algorithm]} · {len(lengths[algorithm])} successes", xlabel="Actions to goal", ylabel="Successful episodes")
        format_axes(ax)
    fig.supxlabel("Conditional on success; failure counts are reported separately. 4,640 evaluated episodes per algorithm.", fontsize=9)
    fig.savefig(folder / "successful_route_lengths.png", dpi=150)
    plt.close(fig)
    for algorithm, episode in common_episodes.items():
        save_figure(folder / f"common_scenario_{algorithm}.png", load_task().grid, episode)


def main():
    parser = argparse.ArgumentParser(description="Audit, analyze and shortlist a complete Part A study")
    parser.add_argument("--suite-id", default="part-a-v1")
    args = parser.parse_args()
    if Path(args.suite_id).name != args.suite_id:
        parser.error("suite-id must be one directory name")
    study = ROOT / "artifacts/studies" / args.suite_id
    manifest = read_json(study / "manifest.json")
    if manifest["status"] != "complete" or manifest["final_test_evaluated"]:
        raise ValueError("Analysis requires a complete validation-only matrix")
    task, rows = load_task(), read_json(study / "run_results.json")
    scenarios = load_scenarios(task, "validation")
    summaries, audits = {}, []
    taxonomy, heatmaps, lengths = {}, {}, {}
    common_episodes, per_scenario, examples = {}, defaultdict(dict), {}
    for algorithm in LABELS:
        family = [r for r in rows if r["algorithm"] == algorithm]
        assert [r["seed"] for r in family] == [7, 19, 43, 67, 91]
        summaries[algorithm] = family_summary(family)
        taxonomy[algorithm], lengths[algorithm], examples[algorithm] = Counter(), [], {}
        heatmaps[algorithm] = {"collision_robot_cell": np.zeros(task.grid.shape, dtype=int), "timeout_start": np.zeros(task.grid.shape, dtype=int)}
        for row in family:
            episodes, audit = audit_run(task, row, scenarios)
            audits.append(audit)
            for episode in episodes:
                kind = failure_type(episode, task.grid)
                taxonomy[algorithm][kind] += 1
                if kind not in examples[algorithm]:
                    examples[algorithm][kind] = {"run_id": row["run_id"], "scenario_id": episode["scenario"]["id"],
                        "metrics": episode["metrics"], "last_decision": episode["transitions"][-1]}
                    write_json(study / "examples" / f"{algorithm}_{kind}.json", episode)
                    if "timeout" in kind or kind == "success_detour":
                        save_figure(study / "figures" / f"example_{algorithm}_{kind}.png", task.grid, episode)
                per_scenario[episode["scenario"]["id"]].setdefault(algorithm, []).append(episode["metrics"])
                if episode["metrics"]["outcome"] == "success":
                    lengths[algorithm].append(episode["metrics"]["steps"])
                elif episode["metrics"]["outcome"] == "collision":
                    heatmaps[algorithm]["collision_robot_cell"][tuple(episode["transitions"][-1]["next_observation"][:2])] += 1
                else:
                    heatmaps[algorithm]["timeout_start"][tuple(episode["scenario"]["start"])] += 1
                if row["seed"] == 7 and episode["scenario"]["id"] == scenarios[0]["id"]:
                    common_episodes[algorithm] = episode
        assert sum(taxonomy[algorithm].values()) == 5 * len(scenarios)
        heatmaps[algorithm] = {k: v.tolist() for k, v in heatmaps[algorithm].items()}
    reference_records = 0
    for path in sorted(study.glob("*_trajectories.jsonl.gz")):
        metrics, common = [], None
        with gzip.open(path, "rt", encoding="utf-8") as file:
            for expected, line in zip(scenarios, file, strict=True):
                episode = json.loads(line)
                assert episode["scenario"] == expected
                validate_replay(task, episode)
                metrics.append(episode["metrics"])
                if expected["id"] == scenarios[0]["id"]:
                    common = episode
        reference_records += len(metrics)
        name = path.name.removesuffix("_trajectories.jsonl.gz")
        summary = read_json(study / "references.json")[name]
        for field, value in aggregate(metrics).items():
            assert summary[field] == value
        if name == "bfs":
            assert summary["success_rate"] == 1 and summary["spl"] == 1
            common_episodes["bfs"] = common
    ranked = rank_families(summaries)
    write_json(study / "summary.json", summaries)
    write_json(study / "shortlist.json", {"ranked_families": ranked, "shortlist": ranked[:2],
        "ranking_rule": "mean validation success, mean SPL, lower seed success SD, lower total training seconds, name",
        "selection_uses_final_test": False, "scope": "five-seed Part A development validation; final evaluation still sealed"})
    write_json(study / "failure_analysis.json", {"taxonomy": taxonomy, "examples": examples, "heatmaps": heatmaps,
        "successful_route_lengths": lengths, "common_scenario_id": scenarios[0]["id"]})
    with (study / "common_scenarios.csv").open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["scenario_id", "distance_band", "shortest_length", "bfs_successes"]
        fieldnames += [f"{a}_{suffix}" for a in LABELS for suffix in ("successes_of_5", "collisions_of_5", "timeouts_of_5", "mean_spl")]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for scenario in scenarios:
            row = {"scenario_id": scenario["id"], "distance_band": scenario["distance_band"],
                   "shortest_length": scenario["shortest_length"], "bfs_successes": 1}
            for algorithm in LABELS:
                result = aggregate(per_scenario[scenario["id"]][algorithm])
                row.update({f"{algorithm}_successes_of_5": result["counts"]["success"],
                            f"{algorithm}_collisions_of_5": result["counts"]["collision"],
                            f"{algorithm}_timeouts_of_5": result["counts"]["timeout"], f"{algorithm}_mean_spl": result["spl"]})
            writer.writerow(row)
    figures(study, rows, summaries, taxonomy, heatmaps, lengths, common_episodes)
    write_json(study / "verification.json", {"audits": audits, "reference_episode_records_verified": reference_records,
        "learned_episode_records_verified": sum(a["episodes_verified"] for a in audits),
        "learned_actions_verified": sum(a["actions_verified"] for a in audits),
        "periodic_checkpoints_verified": sum(a["checkpoint_count_verified"] for a in audits),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "final_test_policy_executions": 0})
    print(json.dumps({"shortlist": ranked[:2], "metrics": {a: s["metrics"] for a, s in summaries.items()}}, indent=2))


if __name__ == "__main__":
    main()
