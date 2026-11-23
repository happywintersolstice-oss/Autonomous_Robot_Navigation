"""Per-member fact sheets: the numbers and file locations each member needs for their own algorithms.

    python -m medinav factsheet --config configs/part_a.yaml
    -> runs/<experiment>/results/factsheets/<owner>.md

Deliberately FACTS ONLY (settings, measured metrics with confidence intervals, robustness, failure
types, where the code and figures are). The interpretation, justification and critical discussion
belong to the member writing the report: the assignment is "Yellow - Limited AI".
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..agents.registry import find_runs, read_meta

CODE_FILES = {
    "q_learning": "src/medinav/agents/tabular.py", "sarsa": "src/medinav/agents/tabular.py",
    "dqn": "src/medinav/agents/dqn_family.py", "double_dqn": "src/medinav/agents/dqn_family.py",
    "dueling_dqn": "src/medinav/agents/dqn_family.py", "dueling_double_dqn": "src/medinav/agents/dqn_family.py",
    "ppo": "src/medinav/agents/policy_gradient.py", "a2c": "src/medinav/agents/policy_gradient.py",
    "recurrent_ppo": "src/medinav/agents/policy_gradient.py",
    "sac": "src/medinav/agents/continuous_control.py", "td3": "src/medinav/agents/continuous_control.py",
}


def _fmt(v, digits=3):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def _ci(row, m):
    lo, hi = row.get(f"{m}_ci_low"), row.get(f"{m}_ci_high")
    if lo is None or pd.isna(lo):
        return _fmt(row.get(m))
    return f"{_fmt(row.get(m))} (95% CI {_fmt(lo)}–{_fmt(hi)})"


def build_factsheets(runs_root, results_dir) -> list[Path]:
    runs_root, results_dir = Path(runs_root), Path(results_dir)
    summary_csv = results_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"{summary_csv} not found - run `evaluate` and `report` first")
    summary = pd.read_csv(summary_csv).set_index("agent")
    episodes = pd.read_csv(results_dir / "episodes.csv") if (results_dir / "episodes.csv").exists() else None
    robust = pd.read_csv(results_dir / "robustness_success.csv", index_col=0) \
        if (results_dir / "robustness_success.csv").exists() else None
    eff = pd.read_csv(results_dir / "sample_efficiency.csv") if (results_dir / "sample_efficiency.csv").exists() else None
    ranking = list(summary.index)
    from ..config import ROOT

    try:  # show a project-relative path so the sheet makes sense on every member's laptop
        shown = results_dir.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = results_dir.as_posix()

    by_owner: dict[str, dict[str, list[dict]]] = {}
    for rd in find_runs(runs_root):
        meta = read_meta(rd)
        owner = meta.get("owner") or "Unassigned"
        by_owner.setdefault(owner, {}).setdefault(meta["label"], []).append(meta)

    out_dir = results_dir / "factsheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for owner, labels in sorted(by_owner.items()):
        lines = [f"# Fact sheet: {owner}", "",
                 "Measured facts for your algorithms. Interpretation and justification are yours to write.",
                 f"Source: `{shown}` (test maps, nominal condition unless stated).", ""]
        for label, metas in sorted(labels.items()):
            m0 = metas[0]
            lines += [f"## {label}", "",
                      f"- Algorithm: `{m0['algo']}`; action space: {m0['action_mode']}; code: `{CODE_FILES.get(m0['algo'], '?')}`",
                      f"- Seeds trained: {sorted(m['seed'] for m in metas)}",
                      f"- Budget: {m0.get('timesteps') or m0.get('episodes')} "
                      f"{'environment steps' if m0.get('timesteps') else 'episodes'} per seed; network {m0.get('net_arch') or 'n/a (table)'}",
                      f"- Training time per seed (s): {[m.get('train_time_s') for m in sorted(metas, key=lambda m: m['seed'])]}",
                      f"- Hyper-parameters: `{m0.get('params')}`",
                      f"- Env overrides / wrappers: `{ {k: v for k, v in m0['env'].items() if k != 'action_mode'} }` / `{m0.get('wrappers')}`",
                      ""]
            if label in summary.index:
                r = summary.loc[label].to_dict()
                lines += ["| Metric | Value |", "|---|---|",
                          f"| Rank (by success, of {len(ranking)} agents incl. baselines) | {ranking.index(label) + 1} |",
                          f"| Success rate | {_ci(r, 'success')} |",
                          f"| Collision rate | {_ci(r, 'collision')} |",
                          f"| Collision with a person | {_fmt(r.get('collision_human'))} |",
                          f"| Timeout rate | {_fmt(r.get('timeout'))} |",
                          f"| Path efficiency vs A* (successes only) | {_ci(r, 'path_efficiency')} |",
                          f"| Time to goal (s, successes only) | {_fmt(r.get('time_to_goal_s'), 1)} |",
                          f"| Share of steps inside a person's comfort zone | {_fmt(r.get('social_intrusion_rate'))} |",
                          f"| Return: mean / IQM | {_fmt(r.get('return'), 2)} / {_fmt(r.get('return_iqm'), 2)} |",
                          f"| Success std across seeds (stability) | {_fmt(r.get('success_seed_std'))} |",
                          f"| Decision time (ms) | {_fmt(r.get('decision_ms'), 2)} |", ""]
            if episodes is not None:
                nom = episodes[(episodes["agent"] == label) & (episodes["condition"] == "nominal")]
                if len(nom):
                    shares = nom["failure_type"].value_counts(normalize=True).round(3).to_dict()
                    lines += [f"- How episodes end: {shares}", ""]
            if robust is not None and label in robust.index:
                row = robust.loc[label].round(3).to_dict()
                lines += ["- Success under stress conditions: " + ", ".join(f"{k} {v}" for k, v in row.items()), ""]
            if eff is not None and label in set(eff["agent"]):
                e = eff[eff["agent"] == label]
                lines += [f"- Sample efficiency (area under validation curve, 0–1) per seed: "
                          f"{e['sample_efficiency'].round(3).tolist()}", ""]
            safe = re.escape(label)
            figs = sorted(p.name for p in results_dir.glob("*.png") if re.search(rf"_{safe}\.png$", p.name))
            lines += ["- Your figures: " + (", ".join(f"`{f}`" for f in figs) or "none yet (run `interpret`)"),
                      "- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, "
                      "`robustness_success.png`, `trajectories.png`", ""]
        lines += ["## Questions to prepare for the Q&A (answer them yourself)", "",
                  "- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?",
                  "- Which hyper-parameter would you change first, and what do you expect to happen?",
                  "- What do the failure types say about what the agent has not learned?",
                  "- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?", ""]
        path = out_dir / f"{re.sub(r'[^A-Za-z0-9]+', '_', owner).strip('_').lower()}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        written.append(path)
        print(f"  wrote {path}")
    return written
