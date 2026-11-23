"""Public-only voting, executed-action history and new development evidence."""

from copy import deepcopy
from pathlib import Path
import gzip
import json
import zipfile

import numpy as np
import pytest

from robot_navigation.app_registry import RUNS, load_model, phase3_selected, registry
from robot_navigation.core import read_json
from robot_navigation.ensemble import VotingPolicy, descriptor, LABEL
from robot_navigation.partial_agents import ALGORITHMS
from robot_navigation.phase3_features import PublicHistory


def public_observation():
    observation = np.zeros(409, dtype=np.float32)
    observation[:4] = [5 / 9, 1 / 9, 5 / 9, 8 / 9]
    return observation


class Member:
    def __init__(self, action):
        self.action = action
        self.history = PublicHistory()
        self.seen = []

    def reset(self):
        self.history = PublicHistory()

    def state_fingerprint(self):
        return "fixed-weights"

    def act(self, observation):
        assert observation.shape == (409,)
        self.seen.append(self.history.last_action)
        self.history.remember(observation, self.action)
        return self.action


def fake_policy(actions):
    members = {name: Member(action) for name, action in zip(ALGORITHMS, actions)}
    return VotingPolicy(members, {"id": "mechanical-vote"})


def test_majority_and_ties_are_fixed_without_action_masks():
    observation = public_observation()
    for action in range(5):
        policy = fake_policy([action] * 4)
        assert policy.act(observation) == action
    assert fake_policy([4, 3, 4, 3]).act(observation) == 3
    assert fake_policy([3, 2, 1, 4]).act(observation) == 1


def test_every_member_remembers_the_executed_action_not_its_proposal():
    policy = fake_policy([0, 3, 3, 4])
    public = public_observation()
    before = public.copy()
    assert policy.act(public) == 3
    assert all(member.history.last_action == 3 for member in policy.members.values())
    assert policy.act(public) == 3
    assert all(member.seen == [None, 3] for member in policy.members.values())
    assert len(policy.vote_trace) == 2
    np.testing.assert_array_equal(public, before)
    policy.reset()
    assert not policy.vote_trace and policy.history.last_action is None


def test_invalid_member_vote_fails_instead_of_falling_back():
    policy = fake_policy([3, 3, 3, 7])
    with pytest.raises(ValueError):
        policy.act(public_observation())


def test_descriptor_does_not_invent_new_training_or_weights():
    members = phase3_selected()
    item = descriptor(members)
    assert item["actions"] == 0 and item["checkpoint_sha256"] is None
    assert item["constituent_training_actions"] == 160000
    assert item["member_provenance"].keys() == members.keys()
    assert len(registry()) == 24
    with pytest.raises(ValueError):
        descriptor({"q-learning": members["q-learning"]})


def test_real_native_models_load_reset_and_keep_weight_fingerprints():
    item = descriptor(phase3_selected())
    policy = load_model(item)
    before = policy.state_fingerprint()
    action = policy.act(public_observation())
    assert action in range(5) and policy.state_fingerprint() == before
    policy.reset()
    assert policy.act(public_observation()) == action
    assert all(
        member.history.last_action == action for member in policy.members.values()
    )
    bad = deepcopy(item)
    bad["member_set_sha256"] = "altered"
    with pytest.raises(ValueError):
        load_model(bad)


def test_study_cannot_silently_retry_or_enter_a_frozen_test(tmp_path):
    from robot_navigation.rubric_study import execute

    (tmp_path / "protocol.json").write_text("{}")
    (tmp_path / "started.json").write_text("{}")
    with pytest.raises(FileExistsError):
        execute(tmp_path)


def test_saved_ensemble_evidence_is_matched_and_votes_follow_real_transitions():
    from robot_navigation.phase3_ui import saved
    from robot_navigation.partial_evaluation import episode_metrics
    from robot_navigation.release import sha

    result = saved("Ensemble development")
    folder = Path(result["folder"])
    manifest = result["manifest"]
    assert manifest["panel_case_count"] == 36 and manifest["case_count"] == 252
    assert (
        manifest["training_actions"]
        == manifest["optimizer_steps"]
        == manifest["heldout_evaluations"]
        == 0
    )
    assert {row["policy"] for row in result["rows"]} == {
        *ALGORITHMS,
        LABEL,
        "Observed-Map Replanner",
        "Random reference",
    }
    for label in {row["policy"] for row in result["rows"]}:
        group = [row for row in result["rows"] if row["policy"] == label]
        assert len(group) == 36
        assert [row["case_hash"] for row in group] == manifest["case_hashes"]
    for row in result["rows"]:
        with gzip.open(folder / row["replay"], "rt", encoding="utf-8") as stream:
            replay = json.load(stream)
        assert replay["scenario"]["split"] in ("validation", "manual")
        assert episode_metrics(replay) == replay["metrics"]
        assert int(row["steps"]) == len(replay["transitions"]) <= 200
        if row["policy"] == LABEL:
            assert len(replay["ensemble_votes"]) == len(replay["transitions"])
            for trace, transition in zip(
                replay["ensemble_votes"], replay["transitions"]
            ):
                counts = np.bincount(list(trace["proposals"].values()), minlength=5)
                assert (
                    int(np.flatnonzero(counts == counts.max())[0])
                    == trace["chosen"]
                    == transition["action"]
                )
    for name, entry in read_json(folder / "export-inventory.json")["files"].items():
        assert sha(folder / name) == entry["sha256"]
    with zipfile.ZipFile(folder / "results-and-replays.zip") as archive:
        assert archive.testzip() is None
        assert (
            len([name for name in archive.namelist() if name.endswith(".json.gz")])
            == 252
        )
    for result in manifest["public_history_sensitivity"].values():
        assert (
            0 < result["probed_decisions"] <= 256 and result["environment_actions"] == 0
        )


def test_ensemble_ui_is_explicit_and_its_published_scope_is_read_only(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from robot_navigation.core import ROOT
    from robot_navigation.partial_dynamic_env import PartialDynamicEnv

    def forbidden(*args, **kwargs):
        pytest.fail("Viewing saved ensemble evidence executed an environment")

    monkeypatch.setattr(PartialDynamicEnv, "reset", forbidden)
    monkeypatch.setattr(PartialDynamicEnv, "step", forbidden)
    app = AppTest.from_file(
        str(ROOT / "app/streamlit_app.py"), default_timeout=60
    ).run()
    assert not app.exception
    select = next(
        widget for widget in app.selectbox if widget.label == "Controller / checkpoint"
    )
    select.set_value(LABEL).run()
    assert not app.exception
    next(widget for widget in app.radio if widget.label == "Workspace").set_value(
        "Compare & Export"
    ).run()
    next(
        widget for widget in app.selectbox if widget.label == "Evaluation scope"
    ).set_value("Ensemble development").run()
    assert not app.exception
    assert not any(button.label == "Run fair comparison" for button in app.button)
    assert len(app.get("download_button")) >= 5
