"""Render inspectable figures from saved evidence only; never execute a policy."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robot_navigation.core import ROOT, read_json, write_json
from robot_navigation.release import sha
from robot_navigation.ensemble import LABEL


def main():
    destination = ROOT.parent / "docs/submission/figures"
    destination.mkdir(parents=True, exist_ok=True)
    pointer = read_json(ROOT / "runs/rubric-publication.json")
    current_path = ROOT / "runs" / pointer["comparison"] / "manifest.json"
    current = read_json(current_path)
    historic_pointer = read_json(ROOT / "runs/phase3-publication.json")
    historic_path = (
        ROOT / "runs" / historic_pointer["heldout_comparison"] / "manifest.json"
    )
    historic = read_json(historic_path)
    development_path = (
        ROOT / "runs" / historic_pointer["study"] / "development-scores.json"
    )
    development = read_json(development_path)
    families = ["q-learning", "sarsa", "dqn", "ppo"]
    names = ["Q-learning", "SARSA", "DQN", "PPO"]
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )

    def save(fig, name):
        fig.savefig(destination / (name + ".png"), dpi=220, bbox_inches="tight")
        fig.savefig(destination / (name + ".svg"), bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 3.8), layout="constrained")
    x = np.arange(4)
    for phase, offset, color in (
        ("Phase2", -0.18, "#7891a5"),
        ("Phase3", 0.18, "#117e9c"),
    ):
        values = [
            next(
                r["Successes"]
                for r in historic["summary"]
                if r["Policy"] == phase + " " + family and r["Subset"] == "all"
            )
            for family in families
        ]
        bars = ax.bar(x + offset, values, 0.34, label=phase, color=color)
        ax.bar_label(bars, labels=[f"{value}/64" for value in values], padding=3)
    ax.set(
        xticks=x,
        xticklabels=names,
        ylim=(0, 73),
        ylabel="Successful episodes",
        title="Original Phase3 matched product holdout\nSaved results only; the ensemble was never tested on this panel",
    )
    ax.legend(loc="lower right")
    save(fig, "phase3-saved-holdout")

    labels = [*families, LABEL]
    overview = {
        row["Policy"]: row for row in current["summary"] if row["Subset"] == "all"
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), layout="constrained")
    shown = [*names, "Vote ensemble"]
    bars = axes[0].bar(
        shown,
        [overview[label]["Successes"] for label in labels],
        color=["#7891a5"] * 4 + ["#117e9c"],
    )
    axes[0].bar_label(
        bars,
        labels=[f"{overview[label]['Successes']}/36" for label in labels],
        padding=3,
    )
    axes[0].set(
        ylim=(0, 42),
        ylabel="Successful episodes",
        title="Previously inspected development cases",
    )
    latency = [
        current["inference_timing"][label]["mean_inference_ms_per_action"]
        for label in labels
    ]
    bars = axes[1].bar(shown, latency, color=["#7891a5"] * 4 + ["#117e9c"])
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].set(
        ylim=(0, max(latency) * 1.25),
        ylabel="Milliseconds per decision",
        title="Single local run; timing includes history",
    )
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=35)
    fig.suptitle(
        "Ensemble comparison is development-only, not unseen accuracy", fontsize=11
    )
    save(fig, "ensemble-development-cost")

    fig, ax = plt.subplots(figsize=(8.2, 3.8), layout="constrained")
    for candidate, offset, color in (
        ("A-public-history-unshaped", -0.18, "#7891a5"),
        ("B-public-history-liveness", 0.18, "#117e9c"),
    ):
        values = [
            next(
                row["successes"]
                for row in development[family]
                if row.get("candidate") == candidate
            )
            for family in families
        ]
        bars = ax.bar(
            x + offset,
            values,
            0.34,
            label="Unshaped" if candidate.startswith("A") else "Liveness feedback",
            color=color,
        )
        ax.bar_label(bars, labels=[f"{value}/32" for value in values], padding=3)
    ax.set(
        xticks=x,
        xticklabels=names,
        ylim=(0, 38),
        ylabel="Successful episodes",
        title="Previously completed optimization evidence\n40,000 fresh actions per candidate; one shared training seed",
    )
    ax.legend(loc="lower right")
    save(fig, "registered-objective-comparison")
    write_json(
        destination / "provenance.json",
        {
            "policy_executions": 0,
            "sources": {
                str(path.relative_to(ROOT.parent)): sha(path)
                for path in (current_path, historic_path, development_path)
            },
            "figures": {
                path.name: sha(path)
                for path in destination.iterdir()
                if path.suffix in (".png", ".svg")
            },
            "note": "Separate development and held-out populations; no cross-population performance delta or ensemble unseen score",
        },
    )
    print(destination)


if __name__ == "__main__":
    main()
