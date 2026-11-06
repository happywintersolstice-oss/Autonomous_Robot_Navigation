from dataclasses import replace
import importlib.util
import shutil

import pytest

from robot_navigation.core import ROOT, GridMap, NavigationEnv, load_task, read_json
from robot_navigation.episodes import rollout
from robot_navigation.part_a import train_part_a

spec = importlib.util.spec_from_file_location("part_a_analysis", ROOT / "scripts/analyze_part_a.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_failure_taxonomy_uses_actual_outcomes():
    task = replace(load_task(), grid=GridMap("fixture", ("...", ".#.", "...")),
                   config={**load_task().config, "max_steps": 4})
    scenario = {"id": "fixture", "start": [0, 0], "goal": [0, 2], "shortest_length": 2}
    class Scripted:
        policy_id = "fixture"
        def __init__(self, actions):
            self.actions = iter(actions)
        def act(self, observation):
            return next(self.actions)
    cases = [([3, 3], "success_optimal"), ([3, 2, 3, 3], "success_detour"),
             ([0], "collision_boundary"), ([3, 1], "collision_obstacle"),
             ([3, 2, 3, 2], "timeout_repeated_state")]
    for actions, expected in cases:
        episode = rollout(NavigationEnv(task, [scenario]), Scripted(actions), scenario)
        assert analysis.failure_type(episode, task.grid) == expected


def test_shortlist_uses_seed_means_and_declared_ties():
    def summary(success, spl, sd=.1, seconds=10):
        return {"metrics": {"success_rate": {"mean": success, "sample_std": sd}, "spl": {"mean": spl}},
                "total_training_seconds": seconds}
    assert analysis.rank_families({"a": summary(.5, .1), "b": summary(.4, .4)}) == ["a", "b"]
    assert analysis.rank_families({"a": summary(.5, .1), "b": summary(.5, .4)}) == ["b", "a"]
    assert analysis.rank_families({"a": summary(.5, .4, .2), "b": summary(.5, .4, .1)}) == ["b", "a"]
    assert analysis.rank_families({"a": summary(.5, .4, seconds=20), "b": summary(.5, .4)}) == ["b", "a"]


def fixture_root(tmp_path):
    for folder in ("src", "app", "configs", "data"):
        shutil.copytree(ROOT / folder, tmp_path / folder, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    return tmp_path


def test_part_a_writes_selected_and_final_without_overwriting(tmp_path):
    root = fixture_root(tmp_path)
    config = dict(algorithm="q-learning", seed=7, interactions=40, monitor_every=20,
        learning_rate=.3, discount=.99, epsilon_start=1., epsilon_end=.1,
        epsilon_decay_fraction=.8, run_id="pipeline-fixture")
    directory, summary, pilot = train_part_a(config, root)
    assert (directory / "selected.npz").exists() and (directory / "checkpoint.npz").exists()
    manifest = read_json(directory / "manifest.json")
    assert manifest["status"] == "complete" and manifest["training_interactions"] == 40
    assert manifest["selected_interactions"] in (20, 40) and not manifest["final_test_evaluated"]
    import numpy as np
    with np.load(directory / "selected_visitation.npz", allow_pickle=False) as data:
        assert data["state_action_counts"].sum() == manifest["selected_interactions"]
    assert summary["q_learning"]["initial_state_overlap_scope"] == "selected checkpoint training prefix"
    assert summary["q_learning"]["episodes"] == 928 and pilot["total_state_action_visits"] == 40
    with pytest.raises(FileExistsError):
        train_part_a(config, root)


def test_part_a_failure_is_recorded(tmp_path, monkeypatch):
    root = fixture_root(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("intentional evaluation fixture error")
    monkeypatch.setattr("robot_navigation.part_a.evaluate", fail)
    config = dict(algorithm="sarsa", seed=7, interactions=4, monitor_every=4,
        learning_rate=.3, discount=.99, epsilon_start=1., epsilon_end=.1,
        epsilon_decay_fraction=.8, run_id="failure-fixture")
    with pytest.raises(RuntimeError, match="intentional"):
        train_part_a(config, root)
    record = read_json(root / "artifacts/runs/failure-fixture/manifest.json")
    assert record["status"] == "failed" and not record["resume_supported"] and not record["final_test_evaluated"]
