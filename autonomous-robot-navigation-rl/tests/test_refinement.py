from dataclasses import replace

import numpy as np
import pytest
import yaml

from robot_navigation.agents import QLearning, SARSA, load_policy
from robot_navigation.benchmark import load_scenarios
from robot_navigation.core import ROOT, GridMap, NavigationEnv, load_task, read_json, write_json
from robot_navigation.episodes import rollout
from robot_navigation.refinement import reward_task, search_candidates, selection_rank, task_for_run, validate_task_parent
from robot_navigation.voting import VotingPolicy
from robot_navigation.workflow import evaluate


def fixture():
    base = load_task()
    return replace(base, grid=GridMap("reward-fixture", ("...",)), config={**base.config, "max_steps": 1})


def test_search_is_frozen_distinct_and_within_domains():
    plan = yaml.safe_load((ROOT / "configs/part_b.yaml").read_text())
    candidates = search_candidates(plan)
    assert candidates == search_candidates(plan) and len(candidates) == 8
    assert len({tuple(c["parameters"].values()) for c in candidates}) == 8
    for candidate in candidates:
        assert all(v in plan["search_space"][k] for k, v in candidate["parameters"].items())
    assert not set(plan["confirmation_seeds"]) & set(plan["development_seeds"])


def test_reward_derivation_preserves_scenarios_and_other_rules():
    base = load_task()
    sparse = reward_task(base, "terminal-only")
    assert sparse.grid == base.grid
    changed = {k for k in base.config if base.config[k] != sparse.config[k]}
    assert changed == {"step_reward", "reward_version"}
    validate_task_parent(sparse, base)
    assert load_scenarios(base, "validation")
    with pytest.raises(ValueError, match="mismatch"):
        load_scenarios(sparse, "validation")
    with pytest.raises(ValueError, match="reward-only"):
        validate_task_parent(replace(sparse, config={**sparse.config, "goal_reward": 20}), base)
    with pytest.raises(ValueError, match="Unsupported"):
        reward_task(base, "planner-shaped")


@pytest.mark.parametrize("goal,action,reward,terminated,truncated", [([0, 2], 3, 0., False, True),
    ([0, 1], 3, 10., True, False), ([0, 2], 2, -5., True, False)])
def test_terminal_only_rewards(goal, action, reward, terminated, truncated):
    task = reward_task(fixture(), "terminal-only")
    scenario = {"id": "fixture", "start": [0, 0], "goal": goal, "shortest_length": goal[1]}
    env = NavigationEnv(task, [scenario])
    env.reset()
    _, actual, term, trunc, _ = env.step(action)
    assert (actual, term, trunc) == (reward, terminated, truncated)


def members(task, actions):
    policies = [QLearning(task, f"fixture-seed{s}") for s in (101, 211, 307)]
    for policy, action in zip(policies, actions):
        policy.q[0, 0, 0, 1, action] = 1
    return policies


def test_voting_majority_and_priority_without_value_averaging():
    task = fixture()
    policies = members(task, [0, 3, 3])
    policies[0].q[0, 0, 0, 1, 0] = 10000
    vote = VotingPolicy(task, policies, [101, 211, 307])
    assert vote.act([0, 0, 0, 1]) == 3
    assert vote.decision([0, 0, 0, 1])["vote_counts"] == [1, 0, 0, 2]
    tie = VotingPolicy(task, members(task, [2, 0, 3]), [101, 211, 307])
    assert tie.act([0, 0, 0, 1]) == 2 and tie.decision([0, 0, 0, 1])["three_way_tie"]
    with pytest.raises(ValueError, match="distinct"):
        VotingPolicy(task, policies, [101, 101, 307])
    with pytest.raises(ValueError, match="one shortlisted"):
        VotingPolicy(task, [policies[0], policies[1], SARSA(task)], [101, 211, 307])


def test_voting_save_load_diagnostics_and_immutability(tmp_path):
    task = fixture()
    policies = members(task, [3, 3, 0])
    paths = [tmp_path / f"member{s}.npz" for s in (101, 211, 307)]
    for policy, seed, path in zip(policies, (101, 211, 307), paths):
        policy.save(path, {"config": {"seed": seed}, "training_interactions": 1})
    vote = VotingPolicy(task, policies, [101, 211, 307])
    descriptor = tmp_path / "voting.json"
    vote.save(descriptor, paths)
    loaded = load_policy(descriptor, task)
    before = loaded.state_fingerprint()
    scenario = {"id": "fixture", "start": [0, 0], "goal": [0, 1], "shortest_length": 1}
    summary, _, records = evaluate(task, [scenario], loaded, 0, "fixture voting")
    assert summary["success_rate"] == 1 and loaded.state_fingerprint() == before
    transition = records[0]["transitions"][0]
    assert transition["scores"] is None and transition["decision"]["member_actions"] == [3, 3, 0]
    with pytest.raises(ValueError, match="mismatch"):
        load_policy(descriptor, reward_task(task, "terminal-only"))
    paths[0].write_bytes(paths[0].read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_policy(descriptor, task)


def test_candidate_ranking_uses_both_seeds():
    def row(value):
        return {"validation": {"success_rate": value, "spl": value}}
    assert selection_rank([row(.6), row(.6)], "b") < selection_rank([row(.9), row(.1)], "a")


def test_variant_run_loading_is_explicit_and_strict(tmp_path):
    base = load_task()
    sparse = reward_task(base, "terminal-only")
    write_json(tmp_path / "configuration.json", {"training": {"reward_variant": "terminal-only"}, "environment": sparse.config})
    write_json(tmp_path / "manifest.json", {"signature": sparse.signature})
    assert task_for_run(base, tmp_path).signature == sparse.signature
    write_json(tmp_path / "configuration.json", {"training": {"reward_variant": "step-cost"}, "environment": sparse.config})
    with pytest.raises(ValueError):
        task_for_run(base, tmp_path)


def test_paired_comparisons_align_seeds_and_exclude_reward_returns(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from analyze_part_b import paired_changes
    def row(seed, value):
        return {"seed": seed, "validation": {"success_rate": value, "spl": value, "mean_return": value}}
    before = [row(101, .2), row(211, .6)]
    after = [row(211, .5), row(101, .4)]
    result = paired_changes(before, after, ("success_rate", "spl"))
    assert result["metrics"]["success_rate"]["mean_difference"] == pytest.approx(.05)
    assert result["metrics"]["success_rate"]["sample_std_of_paired_differences"] == pytest.approx(.2121320344)
    assert "mean_return" not in result["metrics"]
    with pytest.raises(AssertionError):
        paired_changes(before, [row(101, .3)], ("success_rate",))
