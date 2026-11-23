"""Statistics for comparing agents (Part A "measure and compare performance").

RL has no k-fold cross-validation. Our equivalent:
  * several training seeds per algorithm       (variance from learning)
  * many held-out test maps per seed           (variance from the environment)
  * stratified bootstrap over seeds AND episodes -> 95% confidence intervals
  * interquartile mean (IQM) of returns: robust to a few extreme episodes
    (recommended by Agarwal et al., 2021, "Deep RL at the Edge of the Statistical Precipice")
  * stability = std of per-seed success rates (does the method work reliably?)
  * permutation test for "is A really better than B?"
  * sample efficiency = normalised area under the validation learning curve
All numpy - no extra dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ["success", "collision", "collision_human", "timeout", "return", "path_efficiency",
           "time_to_goal_s", "min_human_dist", "social_intrusion_rate", "decision_ms"]


def iqm(x) -> float:
    x = np.sort(np.asarray(x, dtype=float)[~np.isnan(np.asarray(x, dtype=float))])
    if len(x) == 0:
        return float("nan")
    lo, hi = int(np.floor(0.25 * len(x))), int(np.ceil(0.75 * len(x)))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def stratified_bootstrap(df: pd.DataFrame, metric: str, stat=np.nanmean, n_boot=2000, seed=0):
    """95% CI: resample seeds with replacement, then episodes within each chosen seed."""
    rng = np.random.default_rng(seed)
    if not np.isfinite(df[metric].to_numpy(dtype=float)).any():
        return float("nan"), float("nan")  # e.g. path efficiency of an agent that never succeeds
    groups = [g[metric].to_numpy(dtype=float) for _, g in df.groupby("seed")]
    groups = [g for g in groups if len(g)]
    if not groups:
        return float("nan"), float("nan")
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(len(groups), size=len(groups))
        sample = np.concatenate([groups[i][rng.integers(len(groups[i]), size=len(groups[i]))] for i in pick])
        with np.errstate(all="ignore"):
            stats[b] = stat(sample) if np.isfinite(sample).any() else np.nan
    return float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))


def summary_table(episodes: pd.DataFrame, group="agent") -> pd.DataFrame:
    """One row per agent: mean, 95% CI, IQM, stability for the key metrics."""
    rows = []
    for name, df in episodes.groupby(group, sort=False):
        row = {group: name, "n_seeds": df["seed"].nunique(), "n_episodes": len(df)}
        for m in METRICS:
            if m not in df:
                continue
            row[m] = float(np.nanmean(df[m])) if df[m].notna().any() else np.nan
            if m in ("success", "collision", "return", "path_efficiency"):
                row[f"{m}_ci_low"], row[f"{m}_ci_high"] = stratified_bootstrap(df, m)
        row["return_iqm"] = iqm(df["return"])
        per_seed = df.groupby("seed")["success"].mean()
        row["success_seed_std"] = float(per_seed.std(ddof=0)) if len(per_seed) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def permutation_test(a, b, n_perm=10_000, seed=0) -> float:
    """Two-sided p-value for difference in means between samples a and b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    rng = np.random.default_rng(seed)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        count += abs(pooled[: len(a)].mean() - pooled[len(a):].mean()) >= observed
    return (count + 1) / (n_perm + 1)


def sample_efficiency(curve: pd.DataFrame, metric="success") -> float:
    """Normalised area under the validation curve, in [0, 1]. 1 = perfect from step 0."""
    if curve is None or len(curve) < 2:
        return float("nan")
    x = curve["timesteps"].to_numpy(float)
    y = curve[metric].to_numpy(float)
    return float(np.trapezoid(y, x) / (x[-1] - x[0])) if x[-1] > x[0] else float(y.mean())


def failure_breakdown(episodes: pd.DataFrame, group="agent") -> pd.DataFrame:
    """Share of each failure type per agent (rows sum to 1)."""
    return pd.crosstab(episodes[group], episodes["failure_type"], normalize="index")
