import argparse
import json

from gymnasium.utils.env_checker import check_env

from .benchmark import load_scenarios, prepare
from .core import ROOT, NavigationEnv, load_task


def main():
    parser = argparse.ArgumentParser(description="Local navigation milestone; final test is sealed")
    parser.add_argument("command", choices=("prepare", "check", "smoke"))
    parser.add_argument("--config", help="Smoke YAML; use a new run_id for each run")
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
    else:
        from .workflow import train_smoke
        directory, summaries, pilot = train_smoke(args.config)
        print(json.dumps({"run_directory": str(directory), "validation": summaries, "pilot": pilot}, indent=2))


if __name__ == "__main__":
    main()
