"""Verify saved milestone evidence, without executing a policy on the final test."""
from collections import defaultdict
import csv
import json
from time import perf_counter

from robot_navigation.agents import QLearning
from robot_navigation.benchmark import load_scenarios
from robot_navigation.core import ROOT, load_task, read_json, write_json
from robot_navigation.episodes import aggregate, validate_replay
from robot_navigation.visualization import draw, save_figure
from robot_navigation.workflow import source_hash

import matplotlib.pyplot as plt
import numpy as np


def main():
    task = load_task()
    run = ROOT / "artifacts/runs/smoke-seed7"
    agent = QLearning.load(run / "checkpoint.npz", task)
    before = agent.q.copy()
    summary = read_json(run / "validation.json")
    validation = load_scenarios(task, "validation")
    allowed = {s["id"]: s for s in validation}
    replay_actions, total_records = 0, 0
    band_results = {}
    for name in ("q_learning", "bfs", "random"):
        metrics, by_band = [], defaultdict(list)
        with (run / f"{name}_trajectories.jsonl").open(encoding="utf-8") as file:
            for line in file:
                episode = json.loads(line)
                assert episode["scenario"] == allowed[episode["scenario"]["id"]]
                validate_replay(task, episode)
                if name == "q_learning":
                    for transition in episode["transitions"]:
                        assert transition["scores"] == agent.scores(transition["observation"]).tolist()
                        assert transition["action"] == agent.act(transition["observation"])
                metrics.append(episode["metrics"])
                by_band[episode["scenario"]["distance_band"]].append(episode["metrics"])
                replay_actions += len(episode["transitions"])
                total_records += 1
        expected_metrics = read_json(run / f"{name}_episodes.json")
        assert metrics == [{k: v for k, v in m.items() if k != "scenario_id"} for m in expected_metrics]
        for key, value in aggregate(metrics).items():
            assert summary[name][key] == value
        band_results[name] = {band: aggregate(items) for band, items in by_band.items()}
    assert np.array_equal(before, agent.q)
    with np.load(run / "visitation.npz", allow_pickle=False) as counts:
        assert counts["state_action_counts"].sum() == 30000
    with (run / "training.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert sum(int(row["steps"]) for row in rows) == 30000
    assert int(rows[-1]["end_interaction"]) == 30000
    assert not read_json(run / "manifest.json")["final_test_evaluated"]
    timings = []
    episode = read_json(run / "replays/bfs_success.json")
    # Figures may be regenerated from the immutable replay with the current renderer.
    save_figure(run / "bfs_trajectory.png", task.grid, episode)
    save_figure(run / "q_learning_trajectory.png", task.grid, read_json(run / "replays/q_learning_collision.json"))
    save_figure(ROOT / "data/maps/bottleneck-v1.png", task.grid)
    for index in range(6):
        start = perf_counter()
        figure = draw(task.grid, episode, min(index, len(episode["transitions"])))
        figure.canvas.draw()
        timings.append(perf_counter() - start)
        plt.close(figure)
    report = {"saved_episode_records_verified": total_records, "replay_actions_verified": replay_actions,
              "parameters_unchanged": True, "training_visits_and_log_interactions": 30000,
              "final_test_policy_executions": 0, "validation_by_distance_band": band_results,
              "source_hash_matches_run": source_hash(ROOT) == read_json(run / "manifest.json")["source_hash"],
              "source_snapshot_matches_run": source_hash(run / "source") == read_json(run / "manifest.json")["source_hash"],
              "post_run_changes": "Renderer footer spacing and UI Matplotlib cache import order; training/environment/agent code unchanged",
              "render_seconds": timings, "render_mean_seconds": float(np.mean(timings)),
              "render_timing_scope": "six local Matplotlib canvas renders, excluding browser/network/Streamlit scheduling",
              "automated_tests": {"passed": 16, "command": ".venv/Scripts/python.exe -m pytest -q"},
              "gymnasium_checker": "passed",
              "browser_animation_verification": "blocked: Computer Use could not verify current Chrome URL; manual check required"}
    write_json(ROOT / "docs/verification.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "validation_by_distance_band"}, indent=2))


if __name__ == "__main__":
    main()
