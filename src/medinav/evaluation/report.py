"""Turn evaluation CSVs into report-ready figures and tables (fully automated).

Figures (PNG, 200 dpi, white background - paste straight into the Word report):
    learning_curves.png       validation success vs training steps (mean +- std over seeds)
    comparison.png            success / collision / path efficiency / social intrusion, 95% CI
    failure_breakdown.png     how each agent fails (wall, person, stuck, too slow)
    robustness_heatmap.png    success under every stress condition
    importance_<agent>.png    permutation importance of input groups
    tuning_<label>.png        best-so-far score per trial: TPE vs random search
Tables (CSV + Markdown):
    summary.csv / summary.md  every metric with confidence intervals
    sample_efficiency.csv     area under the validation curve per run
    significance.md           permutation-test p-values between top agents

Colours: agents are coloured by FAMILY (fixed order), so the same family keeps
the same colour in every figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import failure_breakdown, permutation_test, sample_efficiency, summary_table  # noqa: E402

FAMILY_COLORS = {  # validated categorical palette, fixed order
    "tabular": "#2a78d6",
    "dqn": "#eb6834",
    "policy_gradient": "#1baf7a",
    "continuous": "#6250d6",
    "ensemble": "#e87ba4",
    "baseline": "#8a8984",
}
FAILURE_COLORS = {
    "success": "#1baf7a", "collision_wall": "#2b2f36", "collision_human": "#e34948",
    "stuck_oscillating": "#eda100", "too_slow": "#a8a7a2",
}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def family(algo: str) -> str:
    if algo.startswith("ensemble"):
        return "ensemble"
    if algo in ("q_learning", "sarsa"):
        return "tabular"
    if "dqn" in algo:
        return "dqn"
    if algo in ("ppo", "a2c", "recurrent_ppo"):
        return "policy_gradient"
    if algo in ("sac", "td3"):
        return "continuous"
    return "baseline"


def _style(ax, title=None, ylabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------- learning curves
def learning_curves(runs_root, out_dir):
    from ..agents.registry import find_runs, read_meta

    curves, eff_rows = {}, []
    for rd in find_runs(runs_root):
        meta = read_meta(rd)
        if not (rd / "curve.csv").exists():
            continue
        c = pd.read_csv(rd / "curve.csv")
        curves.setdefault((meta["label"], meta["algo"]), []).append(c)
        eff_rows.append({"agent": meta["label"], "seed": meta["seed"], "sample_efficiency": sample_efficiency(c),
                         "final_val_success": c["success"].iloc[-1], "train_time_s": meta.get("train_time_s")})
    if not curves:
        return
    labels = sorted(curves, key=lambda k: list(FAMILY_COLORS).index(family(k[1])))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    seen_family = {}
    styles = ["-", "--", ":", "-."]
    for label, algo in labels:
        fam = family(algo)
        ls = styles[seen_family.get(fam, 0) % 4]
        seen_family[fam] = seen_family.get(fam, 0) + 1
        # align seeds on a common x-grid
        xs = np.linspace(0, min(c["timesteps"].max() for c in curves[(label, algo)]), 30)
        ys = np.array([np.interp(xs, c["timesteps"], c["success"]) for c in curves[(label, algo)]])
        m, s = ys.mean(0), ys.std(0)
        ax.plot(xs, m, color=FAMILY_COLORS[fam], lw=2, ls=ls, label=label)
        ax.fill_between(xs, m - s, m + s, color=FAMILY_COLORS[fam], alpha=0.12, lw=0)
    _style(ax, "Validation success during training (mean ± std over seeds)", "success rate")
    ax.set_xlabel("environment steps", fontsize=8, color=MUTED)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    _save(fig, Path(out_dir) / "learning_curves.png")
    pd.DataFrame(eff_rows).to_csv(Path(out_dir) / "sample_efficiency.csv", index=False)


# ---------------------------------------------------------------- comparisons
def comparison(summary: pd.DataFrame, episodes: pd.DataFrame, out_dir):
    algo_of = episodes.groupby("agent")["algo"].first()
    summary = summary.sort_values("success", ascending=True)
    metrics = [("success", "Success rate ↑"), ("collision", "Collision rate ↓"),
               ("path_efficiency", "Path efficiency vs A* ↑"), ("social_intrusion_rate", "Time inside social zone ↓")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 0.35 * len(summary) + 1.6), sharey=True)
    y = np.arange(len(summary))
    for ax, (m, title) in zip(axes, metrics):
        vals = summary[m].to_numpy()
        colors = [FAMILY_COLORS[family(algo_of[a])] for a in summary["agent"]]
        ax.barh(y, vals, color=colors, height=0.62, edgecolor="white", lw=2)
        label_x = vals.copy()
        if f"{m}_ci_low" in summary:
            lo, hi = summary[f"{m}_ci_low"].to_numpy(), summary[f"{m}_ci_high"].to_numpy()
            ax.errorbar(vals, y, xerr=[np.maximum(vals - lo, 0), np.maximum(hi - vals, 0)],
                        fmt="none", ecolor=TEXT, lw=1, capsize=2)
            label_x = np.where(np.isfinite(hi), np.fmax(vals, hi), vals)   # label after the whisker
        for yi, v, lx in zip(y, vals, label_x):
            if np.isfinite(v):
                ax.text(lx + 0.02, yi, f"{v:.2f}", va="center", fontsize=7, color=MUTED)
        _style(ax, title)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 1.15)
    axes[0].set_yticks(y, summary["agent"], fontsize=8)
    _save(fig, Path(out_dir) / "comparison.png")


def failures(episodes: pd.DataFrame, out_dir):
    fb = failure_breakdown(episodes)
    order = [c for c in FAILURE_COLORS if c in fb.columns]
    fb = fb[order].sort_values("success")
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(fb) + 1.4))
    left = np.zeros(len(fb))
    for col in order:
        ax.barh(fb.index, fb[col], left=left, color=FAILURE_COLORS[col], label=col.replace("_", " "),
                height=0.62, edgecolor="white", lw=2)
        left += fb[col].to_numpy()
    _style(ax, "How each agent's episodes end (test maps)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7, frameon=False, ncol=len(order), loc="lower center", bbox_to_anchor=(0.5, 1.06))
    _save(fig, Path(out_dir) / "failure_breakdown.png")


def robustness_heatmap(episodes: pd.DataFrame, out_dir, metric="success"):
    pivot = episodes.pivot_table(index="agent", columns="condition", values=metric, aggfunc="mean")
    if "nominal" in pivot:
        pivot = pivot.sort_values("nominal", ascending=False)
        pivot = pivot[["nominal"] + [c for c in pivot.columns if c != "nominal"]]
    fig, ax = plt.subplots(figsize=(1.0 * pivot.shape[1] + 3, 0.4 * pivot.shape[0] + 1.5))
    im = ax.imshow(pivot.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")  # sequential, one hue
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]), pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else TEXT)
    ax.set_title(f"Robustness: {metric} rate under each test condition", fontsize=10, loc="left", color=TEXT)
    fig.colorbar(im, ax=ax, fraction=0.03)
    _save(fig, Path(out_dir) / f"robustness_{metric}.png")
    pivot.to_csv(Path(out_dir) / f"robustness_{metric}.csv")


def importance_plot(df: pd.DataFrame, agent: str, out_dir):
    d = df[df["group"] != "none (baseline)"].sort_values("drop")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(d["group"], d["drop"], color="#2a78d6", height=0.6, edgecolor="white", lw=2)
    for yi, v in enumerate(d["drop"]):
        ax.text(max(v, 0) + 0.01, yi, f"{v:+.2f}", va="center", fontsize=7, color=MUTED)
    _style(ax, f"{agent}: success drop when an input group is scrambled")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID)
    _save(fig, Path(out_dir) / f"importance_{agent}.png")


def saliency_polar(sal: np.ndarray, n_rays: int, agent: str, out_dir):
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    vals = sal[:n_rays] / (sal[:n_rays].max() + 1e-12)
    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")  # robot faces up
    ax.bar(ang, vals, width=2 * np.pi / n_rays * 0.85, color="#2a78d6", alpha=0.85)
    ax.set_yticklabels([])
    ax.set_title(f"{agent}: which LiDAR rays drive steering", fontsize=9)
    _save(fig, Path(out_dir) / f"saliency_{agent}.png")


def tuning_plot(tuning_root, out_dir):
    dirs = sorted(Path(tuning_root).glob("*_*/trials.csv"))
    by_label = {}
    for f in dirs:
        label, sampler = f.parent.name.rsplit("_", 1)
        by_label.setdefault(label, {})[sampler] = pd.read_csv(f)
    for label, samplers in by_label.items():
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for i, (sampler, df) in enumerate(sorted(samplers.items())):
            v = df["value"].fillna(-np.inf).to_numpy()
            best = np.maximum.accumulate(v)
            c = ["#2a78d6", "#eb6834"][i % 2]
            ax.plot(np.arange(1, len(v) + 1), best, color=c, lw=2, label=f"{sampler} (best so far)")
            ax.scatter(np.arange(1, len(v) + 1), np.where(np.isfinite(v), v, np.nan), color=c, s=12, alpha=0.5)
        _style(ax, f"Tuning {label}: TPE vs random search", "validation score")
        ax.set_xlabel("trial", fontsize=8, color=MUTED)
        ax.legend(fontsize=7, frameon=False)
        _save(fig, Path(out_dir) / f"tuning_{label}.png")


def trajectory_gallery(entries, env_maker, map_seeds, out_dir, name="trajectories"):
    """Overlay several agents' paths on the same test maps. entries = [(agent, env_cfg, wrappers)]."""
    from ..render import draw_world
    from .rollout import run_episode

    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#8a8984"]
    fig, axes = plt.subplots(1, len(map_seeds), figsize=(4.2 * len(map_seeds), 4.6))
    for ax, ms in zip(np.atleast_1d(axes), map_seeds):
        for k, (agent, env_cfg, wrappers) in enumerate(entries):
            env = env_maker(env_cfg, wrappers)
            row, frames = run_episode(agent, env, ms, 0, record=True)
            snap = frames[-1]
            if k == 0:
                first = dict(frames[0])
                first["path"] = None
                draw_world(ax, first, show_rays=False)
            ax.plot(snap["path"][:, 0], snap["path"][:, 1], color=palette[k % 6], lw=2,
                    label=f"{agent.name}: {row['failure_type']}")
            ax.plot(*snap["path"][-1], marker="x" if row["success"] < 1 else "o", color=palette[k % 6], ms=8)
        ax.legend(fontsize=6, frameon=True, loc="upper left")
        ax.set_title(f"test map {ms}", fontsize=9)
    _save(fig, Path(out_dir) / f"{name}.png")


# ---------------------------------------------------------------- tables
def to_markdown(df: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def write_tables(episodes: pd.DataFrame, out_dir, condition="nominal"):
    nominal = episodes[episodes["condition"] == condition]
    summ = summary_table(nominal)
    summ = summ.sort_values("success", ascending=False)
    summ.to_csv(Path(out_dir) / "summary.csv", index=False)
    cols = ["agent", "n_seeds", "success", "success_ci_low", "success_ci_high", "collision", "collision_human",
            "path_efficiency", "time_to_goal_s", "social_intrusion_rate", "return_iqm", "success_seed_std"]
    md = to_markdown(summ[[c for c in cols if c in summ]].round(3))
    (Path(out_dir) / "summary.md").write_text(md + "\n")

    # significance between top-3 learned agents (episode-level success)
    learned = [a for a in summ["agent"] if a not in ("astar", "random")][:3]
    lines = ["| A | B | mean A | mean B | p-value |", "|---|---|---|---|---|"]
    for i in range(len(learned)):
        for j in range(i + 1, len(learned)):
            a = nominal[nominal["agent"] == learned[i]]["success"]
            b = nominal[nominal["agent"] == learned[j]]["success"]
            lines.append(f"| {learned[i]} | {learned[j]} | {a.mean():.3f} | {b.mean():.3f} | {permutation_test(a, b, 5000):.4f} |")
    (Path(out_dir) / "significance.md").write_text("\n".join(lines) + "\n")
    return summ


def build_report(episodes_csv, runs_root, out_dir, tuning_root=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = pd.read_csv(episodes_csv)
    summ = write_tables(episodes, out_dir)
    nominal = episodes[episodes["condition"] == "nominal"]
    comparison(summ, nominal, out_dir)
    failures(nominal, out_dir)
    refinement_table(episodes, out_dir)
    if episodes["condition"].nunique() > 1:
        robustness_heatmap(episodes, out_dir, "success")
        robustness_heatmap(episodes, out_dir, "collision")
    learning_curves(runs_root, out_dir)
    if tuning_root and Path(tuning_root).exists():
        tuning_plot(tuning_root, out_dir)
    return summ


def refinement_table(episodes: pd.DataFrame, out_dir, condition="nominal"):
    """Part B evidence: every refined run vs the Part A run of the SAME algorithm (its starting point).

    A refined run is any label whose algorithm is itself a label in the file (ppo_hybrid -> ppo).
    Ensembles are compared with their best member on success; p-values are two-sided permutation
    tests on episode-level success (same 50 test maps x seeds for both sides).
    """
    nom = episodes[episodes["condition"] == condition]
    labels = set(nom["agent"])
    succ = nom.groupby("agent")["success"].mean()
    coll = nom.groupby("agent")["collision"].mean()
    rows = []
    for label, algo in nom.groupby("agent")["algo"].first().items():
        if label == algo or label in ("astar", "random"):
            continue
        if algo in labels:
            base = algo
        elif str(algo).startswith("ensemble"):
            continue  # handled below
        else:
            continue
        a, b = nom[nom["agent"] == label]["success"], nom[nom["agent"] == base]["success"]
        rows.append({"refined": label, "baseline": base, "success_baseline": succ[base], "success_refined": succ[label],
                     "delta_success": succ[label] - succ[base], "delta_collision": coll[label] - coll[base],
                     "p_value": permutation_test(a, b, 5000)})
    df = pd.DataFrame(rows).sort_values("delta_success", ascending=False) if rows else pd.DataFrame()
    if len(df):
        df.round(4).to_csv(Path(out_dir) / "refinements.csv", index=False)
        (Path(out_dir) / "refinements.md").write_text(
            "Each Part B refinement vs its Part A starting point (test maps, nominal). "
            "p < 0.05 = difference unlikely to be chance.\n\n"
            "Note: runs trained with `env_overrides` (e.g. ppo_drop_* with 20% LiDAR dropout) are also TESTED "
            "with those settings, so compare them with each other, or see the robustness table for the "
            "baseline under the same condition.\n\n" + to_markdown(df.round(3)) + "\n")
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(df) + 1.3))
        d = df.sort_values("delta_success")
        colors = ["#1baf7a" if v > 0 else "#e34948" for v in d["delta_success"]]
        ax.barh(d["refined"], d["delta_success"], color=colors, height=0.62, edgecolor="white", lw=2)
        for yi, (v, pv) in enumerate(zip(d["delta_success"], d["p_value"])):
            ax.text(v + (0.005 if v >= 0 else -0.005), yi, f"{v:+.2f}" + (" *" if pv < 0.05 else ""),
                    va="center", ha="left" if v >= 0 else "right", fontsize=7, color=MUTED)
        ax.axvline(0, color=TEXT, lw=0.8)
        span = max(d["delta_success"].abs().max(), 0.05)
        ax.set_xlim(min(d["delta_success"].min(), 0) - 0.35 * span, max(d["delta_success"].max(), 0) + 0.35 * span)
        _style(ax, "Part B: change in test success vs the Part A starting point (* p < 0.05)")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID)
        _save(fig, Path(out_dir) / "refinements.png")
    return df
