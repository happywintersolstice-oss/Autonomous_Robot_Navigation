import argparse
import json

from gymnasium.utils.env_checker import check_env

from .benchmark import load_scenarios, prepare
from .core import ROOT, NavigationEnv, load_task


def main():
    parser = argparse.ArgumentParser(description="Local navigation milestone; final test is sealed")
    parser.add_argument("command", choices=("prepare", "check", "smoke", "train", "evaluate"))
    parser.add_argument("--config", help="Smoke YAML; use a new run_id for each run")
    parser.add_argument("--checkpoint", help="Frozen table for read-only evaluation")
    parser.add_argument("--split", choices=("validation", "monitoring"), default="validation")
    args = parser.parse_args()
    if args.command == "prepare":
        from .visualization import save_figure
        report = prepare()
        save_figure(ROOT / "data/maps/bottleneck-v1.png", load_task().grid)
        print(json.dumps({"free_cells": report["free_cells"], "component_sizes": report["component_sizes"],
                          "splits": {k: v["count"] for k, v in report["splits"].items()}}, indent=2))
    elif args.command == "check":
        task = load_task()
        check_env(NavigationEnv(task, load_scenarios(task, "train")[:10]), skip_render_check=True)
        print("Gymnasium environment checker passed (rendering is separate from the environment).")
    elif args.command == "evaluate":
        from .agents import load_policy
        from .workflow import evaluate
        if not args.checkpoint:
            parser.error("evaluate requires --checkpoint")
        task = load_task()
        policy = load_policy(args.checkpoint, task)
        summary, _, _ = evaluate(task, load_scenarios(task, args.split), policy, 0,
                                  "complete validation manifest" if args.split == "validation" else "fixed monitoring subset")
        print(json.dumps(summary, indent=2))
    else:
        from .workflow import train_smoke
        directory, summaries, pilot = train_smoke(args.config)
        print(json.dumps({"run_directory": str(directory), "validation": summaries, "pilot": pilot}, indent=2))


if __name__ == "__main__":
    main()
