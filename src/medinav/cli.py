"""One command for everything (spec: "automate these steps as much as possible").

    python -m medinav train     --config configs/part_a.yaml [--only ppo sac] [--seeds 0 1]
    python -m medinav evaluate  --config configs/part_a.yaml [--episodes 50] [--robustness]
    python -m medinav interpret --config configs/part_a.yaml [--only ppo]
    python -m medinav tune      --config configs/tuning.yaml --label ppo --sampler both
    python -m medinav report    --config configs/part_a.yaml
    python -m medinav factsheet --config configs/part_a.yaml      # per-member facts for the report
    python -m medinav pipeline  --config configs/part_a.yaml      # train -> evaluate -> interpret -> report

Outputs go to runs/<experiment>/ (models) and runs/<experiment>/results/ (CSV + PNG).
"""

from __future__ import annotations

import os

# Our networks are tiny (2 x 128 units): one training process gains nothing from 20 CPU threads,
# but several processes each grabbing all threads slow each other down badly. Cap per-process
# threads (override by setting OMP_NUM_THREADS yourself) so members can train in parallel.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from .config import ROOT, load_config, run_root, run_roots  # noqa: E402


def _results(cfg):
    return run_root(cfg) / "results"


def cmd_train(cfg, args):
    from .training.train import train_experiment

    train_experiment(cfg, run_root(cfg), only=args.only, seeds=args.seeds, skip_existing=not args.retrain)


def cmd_evaluate(cfg, args):
    from .evaluation.evaluate import evaluate_parallel

    ecfg = cfg.get("evaluation") or {}
    conditions = {"nominal": {}}
    if args.robustness and cfg.get("robustness"):
        conditions.update(cfg["robustness"]["conditions"])
    n = args.episodes or int(ecfg.get("episodes", 50))
    # one process per test condition (2 threads each, see top of file); --jobs 1 = sequential
    jobs = args.jobs or max(1, min(len(conditions), (os.cpu_count() or 2) // 2))
    evaluate_parallel(cfg, run_roots(cfg), conditions, split=ecfg.get("split", "test"), n_episodes=n,
                      labels=args.only, jobs=jobs, out_csv=_results(cfg) / "episodes.csv")


def cmd_interpret(cfg, args):
    from .agents.registry import TABULAR, baseline
    from .env import make_env
    from .evaluation import report
    from .evaluation.evaluate import eval_env_config, load_all
    from .evaluation.interpret import collect_observations, gradient_saliency, permutation_importance, tabular_policy_map

    out = _results(cfg)
    out.mkdir(parents=True, exist_ok=True)
    eval_env = (cfg.get("evaluation") or {}).get("env") or {}
    trained = load_all(run_roots(cfg), args.only)
    gallery = []
    for label, runs in trained.items():
        agent, meta = runs[0]  # first seed is representative
        env_cfg = eval_env_config(meta["env"], eval_env, {}, meta["action_mode"], "validation")
        print(f"== interpreting {label}")
        imp = permutation_importance(agent, env_cfg, meta["wrappers"], n_episodes=args.episodes or 20)
        imp.to_csv(out / f"importance_{label}.csv", index=False)
        report.importance_plot(imp, label, out)
        if meta["algo"] in TABULAR:
            tabular_policy_map(agent).to_csv(out / f"policy_map_{label}.csv", index=False)
        elif meta["algo"] != "recurrent_ppo" and not meta["wrappers"].get("frame_stack", 1) > 1:
            obs = collect_observations(agent, env_cfg, meta["wrappers"], 5)
            sal = gradient_saliency(agent.model, obs, meta["algo"])
            report.saliency_polar(sal, env_cfg.get("n_rays", 24), label, out)
        gallery.append((agent, {**env_cfg, "split": "test"}, meta["wrappers"]))
    # trajectories of up to 4 agents + A* on three test maps
    summary = out / "summary.csv"
    if summary.exists():
        top = [a for a in pd.read_csv(summary)["agent"] if a in trained][:4]
        gallery = [g for g in gallery if g[0].name in top]
    gallery = gallery[:4] + [(baseline("astar"), {**(cfg.get("env") or {}), **eval_env, "split": "test"}, {})]
    report.trajectory_gallery(gallery, make_env, [2_000_000, 2_000_001, 2_000_002], out)


def cmd_report(cfg, args):
    from .evaluation.report import build_report

    csv = _results(cfg) / "episodes.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} not found - run `evaluate` first")
    summ = build_report(csv, run_root(cfg), _results(cfg), ROOT / "runs" / "tuning")
    print(summ[["agent", "success", "collision", "path_efficiency"]].to_string(index=False))


def cmd_tune(cfg, args):
    from .tuning import tune

    samplers = ["tpe", "random"] if args.sampler == "both" else [args.sampler]
    for s in samplers:
        tune(cfg, args.label, s, ROOT / "runs" / "tuning", n_trials=args.trials)
    from .evaluation.report import tuning_plot

    tuning_plot(ROOT / "runs" / "tuning", ROOT / "runs" / "tuning")


def cmd_factsheet(cfg, args):
    from .evaluation.factsheet import build_factsheets

    build_factsheets(run_root(cfg), _results(cfg))


def cmd_pipeline(cfg, args):
    cmd_train(cfg, args)
    args.robustness = True
    cmd_evaluate(cfg, args)
    cmd_report(cfg, args)
    cmd_interpret(cfg, args)
    cmd_factsheet(cfg, args)


def main(argv=None):
    p = argparse.ArgumentParser(prog="medinav", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["train", "evaluate", "interpret", "tune", "report", "factsheet", "pipeline"])
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--only", nargs="*", help="run labels (or algorithm names) to include")
    p.add_argument("--seeds", nargs="*", type=int)
    p.add_argument("--episodes", type=int)
    p.add_argument("--robustness", action="store_true", help="also evaluate the stress-test conditions")
    p.add_argument("--retrain", action="store_true", help="retrain runs that already exist")
    p.add_argument("--label", help="(tune) which run label to tune")
    p.add_argument("--sampler", default="both", choices=["tpe", "random", "both"])
    p.add_argument("--trials", type=int)
    p.add_argument("--jobs", type=int, help="(evaluate) parallel processes, default = half the CPU threads")
    args = p.parse_args(argv)
    if args.command == "tune" and not args.label:
        p.error("tune needs --label (e.g. --label ppo)")
    try:
        cfg = load_config(args.config)
        {"train": cmd_train, "evaluate": cmd_evaluate, "interpret": cmd_interpret, "tune": cmd_tune,
         "report": cmd_report, "factsheet": cmd_factsheet, "pipeline": cmd_pipeline}[args.command](cfg, args)
    except KeyboardInterrupt:
        # finished runs keep their meta.json, so re-running the same command resumes
        raise SystemExit("\nInterrupted - finished runs are saved; run the same command again to resume.")
    except (FileNotFoundError, ValueError, KeyError) as exc:
        # configuration mistakes: show a one-line message instead of a long traceback
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main()
