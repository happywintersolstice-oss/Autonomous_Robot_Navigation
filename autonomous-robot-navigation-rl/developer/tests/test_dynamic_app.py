"""Focused product integration; no historical experiment/regression reruns."""

import csv, gzip, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from robot_navigation.core import ROOT
from robot_navigation.app_scenarios import (
    NAMES,
    scenario,
    training_cases,
    development_cases,
)
from robot_navigation.app_registry import registry, load_model, newest_by_algorithm
from robot_navigation.app_training import train
from robot_navigation.app_service import generate, frame, cursor, compare, read_replay
from robot_navigation.partial_baselines import OnlinePlanner
from robot_navigation.partial_dynamic_env import PartialDynamicEnv
from robot_navigation.partial_observation import validate
from robot_navigation.partial_contract import configuration


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("native-training")
    receipt = train("quick", actions=64, seed=74011, output=out)
    return out, receipt


def test_all_native_algorithms_train_load_and_infer(trained):
    out, receipt = trained
    items = registry(out)
    assert len(items) == 4 and receipt["actual_actions"] == 256
    env = PartialDynamicEnv(training_cases(), configuration())
    obs, _ = env.reset(seed=3)
    for item in items:
        p = load_model(item)
        assert p.act(obs) in range(5)
        meta = json.loads(Path(item["path"]).with_suffix(".json").read_text())
        assert (
            meta["feature_count"] == 480
            and meta["actions"] == 5
            and meta["observation_values"] == 409
        )
        if item["algorithm"] in ("dqn", "ppo"):
            assert p.model._n_updates > 0


def test_checkpoints_reject_old_four_actions_and_missing(trained, tmp_path):
    item = dict(registry(trained[0])[0])
    item["path"] = str(tmp_path / "missing.zip")
    with pytest.raises(ValueError, match="missing"):
        load_model(item)
    original = Path(registry(trained[0])[0]["path"])
    target = tmp_path / original.name
    target.write_bytes(original.read_bytes())
    metadata = json.loads(original.with_suffix(".json").read_text())
    metadata["actions"] = 4
    target.with_suffix(".json").write_text(json.dumps(metadata))
    item["path"] = str(target)
    with pytest.raises(ValueError, match="incompatible"):
        load_model(item)


def test_train_only_and_exact_budget_guards(tmp_path):
    assert all(c["split"] == "train" for c in training_cases())
    assert all(c["split"] == "manual" for c in development_cases())
    with pytest.raises(ValueError, match="multiple"):
        train("quick", ["ppo"], actions=65, output=tmp_path)
    with pytest.raises(ValueError):
        train("quick", ["dqn"], actions=63, output=tmp_path)


def test_static_permanent_and_temporary_real_sensor_contract():
    for name in NAMES:
        env = PartialDynamicEnv([scenario(name)], configuration())
        obs, info = env.reset(seed=0)
        validate(obs)
        assert (
            obs.shape == (409,)
            and env.action_space.n == 5
            and set(info) == {"outcome", "steps"}
        )
    env = PartialDynamicEnv([scenario(NAMES[2])], configuration())
    obs, _ = env.reset(seed=0)
    assert obs[6] == 0.5  # East is sensor index2, before West.
    for _ in range(6):
        obs, r, term, trunc, info = env.step(4)
        assert not term and r == -0.01
    assert obs[6] == 0 or obs[6] == 1
    obs, r, term, trunc, info = env.step(3)
    assert not term and info["outcome"] == "running"


def test_replanner_wait_and_detour_have_no_truth_argument(tmp_path):
    class Guarded:
        algorithm = "online-planner"
        policy_id = "guarded-observation-only"

        def act(self, observation):
            validate(observation)
            return OnlinePlanner().act(observation)

    from robot_navigation.partial_evaluation import record_episode

    waiting = record_episode(scenario(NAMES[2]), Guarded(), configuration())
    assert (
        waiting["metrics"]["outcome"] == "success"
        and waiting["metrics"]["wait_count"] >= 6
    )
    maze = record_episode(scenario(NAMES[1], "Static"), Guarded(), configuration())
    assert maze["metrics"]["outcome"] == "success"


def test_passive_playback_and_genuine_failure(tmp_path, monkeypatch):
    from robot_navigation.partial_baselines import AlwaysWait
    from robot_navigation.partial_evaluation import record_episode

    record = record_episode(scenario(NAMES[0], "Static"), AlwaysWait(), configuration())
    assert record["metrics"]["outcome"] == "timeout"
    monkeypatch.setattr(
        PartialDynamicEnv,
        "step",
        lambda *a, **k: pytest.fail("Playback stepped the environment"),
    )
    for n in [0, 1, 30, 200, 0]:
        assert frame(record, n)["step"] == n
    assert (
        cursor(0, "step", 200) == 1
        and cursor(10, "reset", 200) == 0
        and cursor(100, "seek", 200, 7) == 7
    )


def test_comparison_same_panels_csv_and_parameter_preservation(trained, tmp_path):
    items = newest_by_algorithm(registry(trained[0]))
    result = compare(
        items, cases=development_cases(), include_random=True, output=tmp_path
    )
    assert result["manifest"]["case_count"] == 24 and len(result["rows"]) == 24
    for policy in {r["policy"] for r in result["rows"]}:
        assert [
            r["scenario_hash"] for r in result["rows"] if r["policy"] == policy
        ] == result["manifest"]["case_hashes"]
    folder = Path(result["folder"])
    rows = list(csv.DictReader((folder / "case-results.csv").open()))
    assert len(rows) == 24
    for row in result["rows"]:
        replay = read_replay(folder / row["replay"])
        assert len(replay["transitions"]) == row["steps"]
    assert all(
        r["kind"] == "Non-RL reference"
        for r in result["rows"]
        if "reference" in r["policy"] or "Replanner" in r["policy"]
    )


def test_startup_does_not_import_retired_runtime():
    import robot_navigation.dynamic_demo_ui

    for name in (
        "agents",
        "dashboard",
        "navigation_demo",
        "training",
        "unseen_ui",
        "episodes",
    ):
        assert "robot_navigation." + name not in sys.modules
    assert not hasattr(
        __import__("robot_navigation.core", fromlist=["NavigationEnv"]), "NavigationEnv"
    )


def test_only_new_ui_and_playback_controls(tmp_path, monkeypatch):
    from robot_navigation.ui import simulator as simulator_page

    # A real simulator journey is a mechanical fixture, not a product result.
    original_generate = simulator_page.generate

    def temporary_journey(case, selection, seed=0):
        return original_generate(case, selection, seed, output=tmp_path)

    monkeypatch.setattr(simulator_page, "generate", temporary_journey)
    app = AppTest.from_file(
        str(ROOT / "app/streamlit_app.py"), default_timeout=30
    ).run()
    assert not app.exception
    assert next(r for r in app.radio if r.label == "Workspace").options == [
        "Simulator",
        "Train Models",
        "Compare & Export",
        "Guide",
        "Archives",
    ]
    next(b for b in app.button if b.label == "Run journey").click().run()
    assert not app.exception
    assert any(b.label == "Step" for b in app.button)
    next(b for b in app.button if b.label == "Step").click().run()
    assert not app.exception and app.session_state["frame_index"] == 1
    next(b for b in app.button if b.label == "Reset").click().run()
    assert app.session_state["frame_index"] == 0
    next(b for b in app.button if b.label == "End").click().run()
    assert app.session_state["frame_index"] == len(
        app.session_state["journey"]["transitions"]
    )
    next(b for b in app.button if b.label == "Back").click().run()
    assert not app.exception
    app.slider[0].set_value(3).run()
    assert app.session_state["frame_index"] == 3
    next(b for b in app.button if b.label == "Play").click().run()
    assert app.session_state["playing"]
    next(b for b in app.button if b.label == "Pause").click().run()
    assert not app.session_state["playing"]
    next(b for b in app.button if b.label == "Play").click().run()
    assert app.session_state["playing"]
    next(r for r in app.radio if r.label == "Workspace").set_value("Guide").run()
    assert not app.exception and not app.session_state["playing"]
    next(r for r in app.radio if r.label == "Workspace").set_value("Archives").run()
    assert not app.exception
    next(r for r in app.radio if r.label == "Workspace").set_value("Guide").run()
    assert not app.exception
    next(r for r in app.radio if r.label == "Workspace").set_value("Train Models").run()
    assert not app.exception
    next(r for r in app.radio if r.label == "Workspace").set_value(
        "Compare & Export"
    ).run()
    assert not app.exception


def test_archive_streamed_zip_fixture_and_tamper_detection(tmp_path):
    sys.path.insert(0, str(ROOT.parent / "scripts"))
    from archive_tools import export, verify_zip, verify_sources, safe_member

    root = tmp_path / "fixture"
    (root / "archive/manifests").mkdir(parents=True)
    (root / "archive/data").mkdir()
    p = root / "archive/data/old.txt"
    p.write_bytes(b"old negative result\n")
    import hashlib

    row = {
        "original_path": "reports/old.txt",
        "archive_path": "archive/data/old.txt",
        "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "category": "historical-source",
        "collection": "both",
        "source_version": "fixture-commit",
        "zip_path": "original/reports/old.txt",
    }
    (root / "archive/manifests/pre-migration.json").write_text(
        json.dumps({"files": [row], "notes": ["fixture"]})
    )
    assert verify_sources(root)["status"] == "passed"
    out = tmp_path / "export"
    export(root, out)
    assert verify_zip(out)["verified_members"] == 2
    out2 = tmp_path / "export2"
    export(root, out2)
    assert (out / "legacy-classic-history.zip").read_bytes() == (
        out2 / "legacy-classic-history.zip"
    ).read_bytes()
    with pytest.raises(ValueError):
        safe_member("../escape")
    p.write_text("changed")
    with pytest.raises(ValueError):
        verify_sources(root)
