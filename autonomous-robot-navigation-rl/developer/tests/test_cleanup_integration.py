"""Refactoring and UI integration checks; no training or fresh held-out episodes."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from robot_navigation.app_registry import registry, phase3_selected
from robot_navigation.core import ROOT, read_json

FIXTURES = Path(__file__).parents[1] / "fixtures"


def executable_syntax(source):
    """Normalize Python syntax while allowing comments, docstrings and formatting."""

    class WithoutDocstrings(ast.NodeTransformer):
        def visit_Module(self, node):
            self.generic_visit(node)
            return self.strip(node)

        visit_ClassDef = visit_Module
        visit_FunctionDef = visit_Module

        @staticmethod
        def strip(node):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
            return node

    def normalized(node):
        if node is Ellipsis:
            return {"literal": "Ellipsis"}
        if isinstance(node, bytes):
            return {"bytes": node.hex()}
        if isinstance(node, ast.AST):
            fields = {
                key: normalized(value)
                for key, value in ast.iter_fields(node)
                if value is not None and value != []
            }
            return {"node": type(node).__name__, **fields}
        if isinstance(node, list):
            return [normalized(child) for child in node]
        return node

    tree = WithoutDocstrings().visit(ast.parse(source))
    return hashlib.sha256(
        json.dumps(normalized(tree), sort_keys=True).encode()
    ).hexdigest()


def test_environment_learners_and_frozen_study_keep_executable_contracts():
    contracts = read_json(FIXTURES / "executable-contracts.json")
    for relative, expected in contracts["files"].items():
        source = (ROOT.parent / relative).read_text(encoding="utf-8-sig")
        assert executable_syntax(source) == expected, relative


def test_checkpoint_registry_refactor_preserves_every_existing_public_record():
    def portable(item):
        return {key: value for key, value in item.items() if key != "path"}

    expected = read_json(FIXTURES / "registry-before-cleanup.json")
    actual = {item["id"]: portable(item) for item in registry()}
    # New user-created runs may be added later; the published records must remain.
    assert {key: actual[key] for key in expected} == expected


def test_invalid_scope_and_wrong_algorithm_are_rejected_before_evaluation(monkeypatch):
    from robot_navigation import phase3_ui

    monkeypatch.setattr(
        phase3_ui,
        "export_comparison",
        lambda *a, **k: pytest.fail("Invalid request evaluated"),
    )
    with pytest.raises(ValueError, match="registered"):
        phase3_ui.saved("typo-is-not-a-development-scope")
    selected = phase3_selected()
    selected["dqn"] = selected["ppo"]
    with pytest.raises(ValueError, match="matching"):
        phase3_ui.development_compare(selected)


def test_custom_preview_uses_requested_positions_and_shows_initial_blockage():
    from robot_navigation.ui.simulator import preview_snapshot, scenario

    case = scenario("Temporary bridge - Wait")
    preview = preview_snapshot({**case, "start": [5, 7], "goal": [5, 1]})
    assert preview["position"] == [5, 7] and preview["goal"] == [5, 1]
    assert preview["occupancy"][5][2] == 1
    assert all(value == -1 for row in preview["memory"]["occupancy"] for value in row)


def test_packaged_theme_and_guide_are_present_and_navigation_is_shared():
    import tomllib
    from importlib.resources import files
    from robot_navigation.ui.components import PAGES
    from robot_navigation import dynamic_demo_ui
    from robot_navigation.ui.simulator import simulator

    package = files("robot_navigation.ui")
    assert "warehouse" in package.joinpath("styles.css").read_text(encoding="utf-8")
    guide = package.joinpath("guide.md").read_text(encoding="utf-8")
    assert all(
        heading in guide
        for heading in ("What the robot sees", "Q-learning", "SARSA", "DQN", "PPO")
    )
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert {"styles.css", "guide.md"} <= set(
        config["tool"]["setuptools"]["package-data"]["robot_navigation.ui"]
    )
    assert len(PAGES) == 5 and dynamic_demo_ui.simulator is simulator


def test_page_navigation_and_saved_replays_never_start_training_or_evaluation(
    monkeypatch,
):
    from streamlit.testing.v1 import AppTest
    from robot_navigation.partial_dynamic_env import PartialDynamicEnv
    from robot_navigation import app_service, phase3_ui

    def forbidden(*args, **kwargs):
        pytest.fail("Passive navigation executed a learner or an environment")

    monkeypatch.setattr(PartialDynamicEnv, "reset", forbidden)
    monkeypatch.setattr(PartialDynamicEnv, "step", forbidden)
    monkeypatch.setattr(app_service, "start_job", forbidden)
    app = AppTest.from_file(
        str(ROOT / "app/streamlit_app.py"), default_timeout=60
    ).run()
    assert not app.exception
    for page in ("Train Models", "Guide", "Archives", "Compare & Export"):
        next(r for r in app.radio if r.label == "Workspace").set_value(page).run()
        assert not app.exception
    scope = next(s for s in app.selectbox if s.label == "Evaluation scope")
    for name in phase3_ui.SCOPES:
        scope.set_value(name).run()
        assert not app.exception
        scope = next(s for s in app.selectbox if s.label == "Evaluation scope")
    scope.set_value(phase3_ui.SCOPES[0]).run()
    next(
        b for b in app.button if b.label == "Open recorded journey in Simulator"
    ).click().run()
    assert not app.exception
    next(b for b in app.button if b.label == "Step").click().run()
    assert not app.exception and app.session_state["frame_index"] == 1


def test_launcher_reuses_the_configured_environment_without_repointing_install(
    tmp_path, monkeypatch
):
    import importlib.util
    import os
    import sys

    spec = importlib.util.spec_from_file_location(
        "cleanup_launcher", ROOT.parent / "run_app.py"
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher, "APP", tmp_path)
    monkeypatch.delenv("ROBOT_NAV_PYTHON", raising=False)
    configured = tmp_path / "existing-python.exe"
    configured.touch()
    saved = tmp_path / ".runtime/python-path.txt"
    saved.parent.mkdir()
    saved.write_text(str(configured), encoding="utf-8-sig")
    assert launcher.resolve_interpreter() == str(configured)
    monkeypatch.setenv("ROBOT_NAV_PYTHON", "environment-python")
    assert launcher.resolve_interpreter() == "environment-python"
    assert launcher.resolve_interpreter("explicit-python") == "explicit-python"
    monkeypatch.delenv("ROBOT_NAV_PYTHON")
    local = (
        tmp_path / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    local.parent.mkdir(parents=True)
    local.touch()
    assert launcher.resolve_interpreter() == str(local)
