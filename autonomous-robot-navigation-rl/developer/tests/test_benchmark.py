import shutil

import pytest

from robot_navigation.benchmark import load_scenarios, prepare
from robot_navigation.core import ROOT, load_task, read_json


def test_splits_deterministic_disjoint_goal_coverage(tmp_path):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    shutil.copytree(ROOT / "data/maps", tmp_path / "data/maps")
    report = prepare(tmp_path)
    task = load_task(tmp_path)
    # Structural inspection of holdout metadata only. No test policy rollout.
    splits = {name: read_json(tmp_path / f"data/scenarios/{name}.json")["scenarios"]
              for name in ("train", "validation", "test")}
    ids = {name: {s["id"] for s in rows} for name, rows in splits.items()}
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]
    assert sum(map(len, ids.values())) == report["free_cells"] * (report["free_cells"] - 1)
    goals = {name: {tuple(s["goal"]) for s in rows} for name, rows in splits.items()}
    assert goals["validation"] <= goals["train"] and goals["test"] <= goals["train"]
    assert min(report["splits"]["train"]["goal_frequencies"].values()) >= 20
    assert report["component_sizes"] == [report["free_cells"]]
    assert {s["distance_band"] for s in splits["validation"]} == {"short", "medium", "long"}
    original = (tmp_path / "data/scenarios/train.json").read_bytes()
    assert prepare(tmp_path) == report
    assert (tmp_path / "data/scenarios/train.json").read_bytes() == original
    monitoring = load_scenarios(task, "monitoring", tmp_path)
    assert len(monitoring) <= 50 and {s["id"] for s in monitoring} <= ids["validation"]
    with pytest.raises(ValueError, match="sealed"):
        load_scenarios(task, "test", tmp_path)
