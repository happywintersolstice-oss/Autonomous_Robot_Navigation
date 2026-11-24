"""Second test layer: reward correctness, error handling, every algorithm, ensembles, tuning.

Fast enough to run before every commit (deep-RL tests train for a few hundred steps only):
    python -m pytest -q
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medinav.agents import AStarAgent, RandomAgent, TabularAgent
from medinav.agents.base import Agent
from medinav.agents.ensemble import ActionAverageEnsemble, MajorityVoteEnsemble, SafetyShieldEnsemble
from medinav.agents.registry import load_agent
from medinav.config import deep_merge, load_config
from medinav.env import HospitalNavEnv, generate_layout, make_env, map_seed
from medinav.evaluation.evaluate import build_ensembles, evaluate_experiment
from medinav.evaluation.interpret import feature_groups, permutation_importance
from medinav.evaluation.metrics import failure_breakdown, sample_efficiency, summary_table
from medinav.evaluation.rollout import failure_type, run_episode
from medinav.training.train import train_experiment


# ------------------------------------------------------------------ environment / reward
def test_geodesic_is_finite_next_to_walls():
    """Regression: cells inside the inflated wall band used to fall back to straight-line distance."""
    lay = generate_layout(5, 2)
    band = np.argwhere(lay.inflated & ~lay.grid)          # legal-ish positions hugging a wall
    for r, c in band[:: max(1, len(band) // 50)]:
        xy = lay.centre((r, c))
        d = lay.geodesic_m(xy)
        assert np.isfinite(d)
        # never shorter than the straight line (a shortest route cannot beat a straight line)
        assert d >= np.linalg.norm(xy - lay.goal) - 0.5


@pytest.mark.parametrize("seed", range(4))
def test_progress_reward_never_jumps(seed):
    """Per step the robot moves <= v_max*dt = 0.12 m; progress must stay in that order of magnitude."""
    env = HospitalNavEnv({"difficulty": 3})
    env.reset(seed=seed)
    agent = RandomAgent(seed=seed)
    for _ in range(300):
        obs, r, term, trunc, info = env.step(agent.act(None, env))
        assert abs(info["reward_terms"]["progress"]) < 0.6
        if term or trunc:
            break


def test_unknown_env_key_rejected():
    with pytest.raises(KeyError):
        HospitalNavEnv({"not_a_setting": 1})


def test_step_after_done_raises():
    env = HospitalNavEnv({"max_steps": 1})
    env.reset(seed=0)
    env.step(0)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_frame_stack_shape():
    env = make_env({}, {"frame_stack": 3})
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3 * 29,) and env.observation_space.contains(obs)


def test_guidance_adds_waypoint_inputs():
    env = make_env({"guidance": True})
    obs, _ = env.reset(seed=0)
    assert obs.shape == (31,)


# ------------------------------------------------------------------ metrics / evaluation helpers
def test_failure_type_classification():
    still = np.zeros((150, 2))
    moving = np.c_[np.linspace(0, 5, 150), np.zeros(150)]
    assert failure_type("timeout", still) == "stuck_oscillating"
    assert failure_type("timeout", moving) == "too_slow"
    assert failure_type("collision_human", moving) == "collision_human"


def test_sample_efficiency_bounds():
    curve = pd.DataFrame({"timesteps": [0, 10, 20], "success": [0.0, 1.0, 1.0]})
    assert sample_efficiency(curve) == pytest.approx(0.75)
    assert np.isnan(sample_efficiency(curve.iloc[:1]))


def test_summary_and_failures_tables():
    df = pd.DataFrame({"agent": ["a"] * 4 + ["b"] * 4, "seed": [0, 0, 1, 1] * 2,
                       "success": [1, 0, 1, 1, 0, 0, 0, 1], "collision": [0, 1, 0, 0, 1, 1, 0, 0],
                       "return": np.arange(8.0), "path_efficiency": [0.9, np.nan, 0.8, 0.7, np.nan, np.nan, np.nan, 0.5],
                       "failure_type": ["success", "collision_wall", "success", "success",
                                        "collision_wall", "collision_human", "too_slow", "success"]})
    s = summary_table(df).set_index("agent")
    assert s.loc["a", "success"] == 0.75 and s.loc["b", "n_seeds"] == 2
    assert s.loc["a", "success_ci_low"] <= 0.75 <= s.loc["a", "success_ci_high"]
    fb = failure_breakdown(df)
    assert np.allclose(fb.sum(axis=1), 1.0)


# ------------------------------------------------------------------ config
def test_config_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text("env: {difficulty: 2, max_steps: 400}\ntraining: {seeds: [0, 1]}\n")
    (tmp_path / "child.yaml").write_text("base: base.yaml\nenv: {difficulty: 0}\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["env"] == {"difficulty": 0, "max_steps": 400} and cfg["training"]["seeds"] == [0, 1]
    assert deep_merge({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}


def test_config_errors_are_clear(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
    (tmp_path / "bad.yaml").write_text("- just\n- a list\n")
    with pytest.raises(ValueError):
        load_config(tmp_path / "bad.yaml")


def test_real_configs_parse():
    from medinav.config import ROOT, algorithm_specs

    for name in ("part_a", "part_b", "tuning", "smoke"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert algorithm_specs(cfg), name


# ------------------------------------------------------------------ training pipeline
def _tiny_cfg(algos):
    return {"experiment": "t", "env": {"difficulty": 0, "max_steps": 60},
            "training": {"seeds": [0], "n_envs": 1, "eval_freq": 200, "eval_episodes": 2},
            "algorithms": algos}


def test_train_save_load_tabular_roundtrip(tmp_path):
    cfg = _tiny_cfg({"q_learning": {"episodes": 20, "eval_every_episodes": 10}})
    (run_dir,) = train_experiment(cfg, tmp_path)
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["algo"] == "q_learning" and (run_dir / "q_table.npy").exists()
    agent, meta2 = load_agent(run_dir)
    obs, _ = make_env(meta2["env"]).reset(seed=0)
    assert agent.act(obs, None) in range(5)
    # second call skips the finished run instead of retraining it
    assert train_experiment(cfg, tmp_path) == [run_dir]


def test_one_failing_run_does_not_stop_the_rest(tmp_path):
    cfg = _tiny_cfg({"broken": {"algo": "no_such_algo", "timesteps": 10},
                     "sarsa": {"episodes": 10, "eval_every_episodes": 10}})
    done = train_experiment(cfg, tmp_path)
    assert [d.parent.name for d in done] == ["sarsa"]
    failed = json.loads((tmp_path / "failed_runs.json").read_text())
    assert failed[0]["label"] == "broken" and not (tmp_path / "broken" / "seed_0" / "meta.json").exists()


def test_only_filter_must_match(tmp_path):
    with pytest.raises(ValueError):
        train_experiment(_tiny_cfg({"sarsa": {"episodes": 5}}), tmp_path, only=["ppo"])


DEEP = {
    "dqn": {"timesteps": 300, "params": {"learning_starts": 50, "buffer_size": 1000}},
    "double_dqn": {"timesteps": 300, "params": {"learning_starts": 50, "buffer_size": 1000}},
    "dueling_dqn": {"timesteps": 300, "params": {"learning_starts": 50, "buffer_size": 1000}},
    "ppo": {"timesteps": 256, "params": {"n_steps": 128, "batch_size": 64}},
    "a2c": {"timesteps": 200},
    "sac": {"timesteps": 200, "params": {"learning_starts": 50, "buffer_size": 1000}},
    "td3": {"timesteps": 200, "params": {"learning_starts": 50, "buffer_size": 1000}},
}


@pytest.fixture(scope="module")
def deep_runs(tmp_path_factory):
    """Train every deep algorithm for a few hundred steps once, shared by the tests below."""
    root = tmp_path_factory.mktemp("deep")
    algos = {k: {**v, "net_arch": [16, 16]} for k, v in DEEP.items()}
    train_experiment(_tiny_cfg(algos), root)
    return root


@pytest.mark.parametrize("algo", list(DEEP))
def test_every_deep_algorithm_trains_saves_and_acts(deep_runs, algo):
    run_dir = deep_runs / algo / "seed_0"
    assert (run_dir / "meta.json").exists(), f"{algo} failed to train"
    agent, meta = load_agent(run_dir)
    env = make_env({**meta["env"], "split": "test"}, meta["wrappers"])
    row, _ = run_episode(agent, env, map_seed("test", 0))
    assert row["outcome"] in ("success", "collision_wall", "collision_human", "timeout")
    assert pd.read_csv(run_dir / "curve.csv")["timesteps"].iloc[-1] > 0


def test_ensembles_build_and_reject_mismatches(deep_runs):
    from medinav.evaluation.evaluate import load_all

    trained = load_all(deep_runs)
    specs = [
        {"name": "q_avg", "type": "q_average", "members": ["dqn", "double_dqn", "dueling_dqn"]},
        {"name": "avg", "type": "action_average", "members": ["ppo", "sac", "td3"]},
        {"name": "bad_vote", "type": "majority_vote", "members": ["dqn", "sac"]},       # mixed action modes
        {"name": "missing", "type": "majority_vote", "members": ["dqn", "nope"]},
    ]
    built = {meta["label"]: agent for agent, meta in build_ensembles(specs, trained)}
    assert set(built) == {"q_avg", "avg"}
    obs, _ = make_env({"difficulty": 0}).reset(seed=0)
    assert built["q_avg"].act(obs, None) in range(5)
    assert built["avg"].act(obs, None).shape == (2,)


def test_evaluate_experiment_end_to_end(deep_runs, tmp_path):
    cfg = {"env": {"difficulty": 0, "max_steps": 60},
           "ensembles": [{"name": "q_avg", "type": "q_average", "members": ["dqn", "double_dqn"]}]}
    df = evaluate_experiment(cfg, deep_runs, {"nominal": {}, "crowded": {"n_humans": 4}}, n_episodes=2,
                             labels=["dqn", "double_dqn"], out_csv=tmp_path / "e.csv")
    assert set(df["agent"]) == {"dqn", "double_dqn", "q_avg", "astar", "random"}
    assert set(df["condition"]) == {"nominal", "crowded"} and (tmp_path / "e.csv").exists()


# ------------------------------------------------------------------ ensembles (pure logic)
class _Fixed(Agent):
    def __init__(self, a, mode="discrete"):
        self.a, self.action_mode, self.name = a, mode, f"fixed{a}"

    def act(self, obs, env):
        return self.a


def test_majority_vote_and_tie_break():
    assert MajorityVoteEnsemble([_Fixed(1), _Fixed(2), _Fixed(2)]).act(None, None) == 2
    assert MajorityVoteEnsemble([_Fixed(3), _Fixed(4)]).act(None, None) == 3     # tie -> first agent


def test_action_average_weights():
    ens = ActionAverageEnsemble([_Fixed(np.array([1.0, 1.0]), "continuous"),
                                 _Fixed(np.array([-1.0, 0.0]), "continuous")], weights=[3, 1])
    assert np.allclose(ens.act(None, None), [0.5, 0.75])


def test_shield_overrides_only_when_blocked():
    obs = np.ones(29, dtype=np.float32)
    shield = SafetyShieldEnsemble(_Fixed(0))
    assert shield.act(obs, None) == 0 and shield.overrides == 0
    obs[0] = 0.05                                        # ray 0 points straight ahead: 0.15 m
    assert shield.act(obs, None) in (3, 4) and shield.overrides == 1


# ------------------------------------------------------------------ interpretability / tuning
def test_permutation_importance_covers_all_groups():
    imp = permutation_importance(AStarAgent(), {"difficulty": 0, "n_humans": 0}, {}, n_episodes=2)
    assert set(imp["group"]) == {"none (baseline)", *feature_groups()}
    # A* ignores observations entirely, so scrambling them must not change its success
    assert np.allclose(imp["drop"], 0.0)


def test_tuning_suggest_types():
    import optuna

    from medinav.tuning import _suggest

    trial = optuna.trial.FixedTrial({"a": 0.01, "b": 3, "c": "x"})
    assert _suggest(trial, "a", {"type": "loguniform", "low": 1e-3, "high": 1}) == 0.01
    assert _suggest(trial, "b", {"type": "int", "low": 1, "high": 5}) == 3
    assert _suggest(trial, "c", {"type": "categorical", "choices": ["x", "y"]}) == "x"
    with pytest.raises(ValueError):
        _suggest(trial, "a", {"type": "nope"})


def test_tabular_agent_learns_to_beat_random_on_easy_maps():
    """Not a benchmark - a sanity check that the update rule moves in the right direction."""
    env = make_env({"difficulty": 0, "n_humans": 0, "max_steps": 300})
    agent = TabularAgent("q_learning").train(env, 400, seed=0)
    from medinav.evaluation.rollout import evaluate

    # return, not success: 400 episodes is far too short to finish hospital-length routes reliably,
    # but the learned table must already make more progress than random driving
    learned = np.mean([r["return"] for r in evaluate(agent, env, "validation", 20)])
    rand = np.mean([r["return"] for r in evaluate(RandomAgent(), env, "validation", 20)])
    assert learned > rand


def test_parallel_evaluation_matches_sequential(deep_runs):
    from medinav.evaluation.evaluate import evaluate_parallel

    cfg = {"env": {"difficulty": 0, "max_steps": 60}}
    conds = {"nominal": {}, "crowded": {"n_humans": 4}}
    seq = evaluate_parallel(cfg, deep_runs, conds, n_episodes=2, labels=["dqn", "ppo"], jobs=1)
    par = evaluate_parallel(cfg, deep_runs, conds, n_episodes=2, labels=["dqn", "ppo"], jobs=2)
    cols = [c for c in seq.columns if c != "decision_ms"]          # wall-clock timing differs by nature
    pd.testing.assert_frame_equal(seq[cols], par[cols])


def test_guided_discretiser_uses_waypoint_bearing():
    from medinav.agents.tabular import Discretiser

    env = HospitalNavEnv({"guidance": True})
    obs, _ = env.reset(seed=0)
    plain, guided = Discretiser(), Discretiser(guidance=True)
    assert 0 <= guided(obs) < guided.n_states
    flipped = obs.copy()
    flipped[29:31] = -obs[29:31]                    # waypoint behind the robot instead
    assert guided(flipped) != guided(obs) and plain(flipped) == plain(obs)
    with pytest.raises(ValueError):
        guided(np.zeros(29, dtype=np.float32))      # env without guidance inputs


def test_people_never_walk_into_a_still_robot():
    """Env v2: with yielding on, a robot that only rotates in place can never be hit by a person."""
    hits = {True: 0, False: 0}
    for yield_ in (True, False):
        for seed in range(15):
            env = HospitalNavEnv({"difficulty": 3, "n_humans": 6, "humans_yield": yield_, "max_steps": 200})
            env.reset(seed=seed, options={"map_seed": map_seed("validation", seed)})
            done = False
            while not done:
                _, _, term, trunc, info = env.step(3)          # rotate left = never moves
                done = term or trunc
            hits[yield_] += info["outcome"] == "collision_human"
    assert hits[True] == 0
    assert hits[False] > 0          # the old behaviour really did produce unavoidable collisions


# ------------------------------------------------------------------ hospital floor
@pytest.mark.parametrize("difficulty", range(4))
def test_hospital_floor_structure(difficulty):
    for i in range(10):
        lay = generate_layout(map_seed("test", i), difficulty)
        kinds = [r.kind for r in lay.rooms]
        assert kinds.count("pharmacy") == 1 and kinds.count("ward") >= 2 and "nurses" in kinds
        assert lay.room_at(lay.start) == "Pharmacy"                       # deliveries start in the pharmacy
        assert lay.goal_room.startswith("Ward") and lay.room_at(lay.goal) == lay.goal_room
        assert lay.goal_label.startswith(lay.goal_room[-1] + "-")        # bed C-3 is in Ward C
        assert np.isfinite(lay.geodesic[lay.cell(lay.start)])             # the delivery is possible
        assert lay.width_m == 24.0 and lay.height_m == 16.0


def test_floors_differ_between_seeds_and_splits():
    a, b = generate_layout(map_seed("train", 0), 2), generate_layout(map_seed("test", 0), 2)
    assert not np.array_equal(a.grid, b.grid)


def test_harder_levels_mean_longer_routes_and_more_people():
    def stats(d):
        lays = [generate_layout(map_seed("validation", i), d) for i in range(20)]
        return np.mean([l.geodesic_m(l.start) for l in lays]), np.mean([len(l.human_spawns) for l in lays])
    (r0, h0), (r3, h3) = stats(0), stats(3)
    assert r3 > r0 and h3 > h0


def test_bigger_hospital_option():
    lay = generate_layout(map_seed("test", 0), 2, width_m=32.0, height_m=20.0)
    assert lay.grid.shape == (80, 128) and len(lay.rooms) >= 6


def test_app_figures_build():
    """The app's animated replay and comparison figure build from a real recorded episode."""
    from medinav.viz import episode_animation, paths_figure

    env = make_env({"difficulty": 1})
    row, frames = run_episode(AStarAgent(), env, map_seed("test", 1), record=True)
    fig = episode_animation(frames, max_frames=30)
    assert 1 <= len(fig.frames) <= 31 and len(fig.layout.images) == 1
    assert len(fig.data) == len(fig.frames[0].data)          # Plotly needs the same traces in every frame
    fig2 = paths_figure(frames[0], [("A*", frames[-1]["path"], row["failure_type"], row["success"])])
    assert len(fig2.data) == 2
    assert "Pharmacy" in frames[0]["mission"] and frames[0]["goal_room"].startswith("Ward")


def test_part_b_uses_part_a_base_env():
    """Part B refines Part A: same base world unless a run overrides it on purpose (e.g. *_noroute)."""
    import yaml
    root = Path(__file__).resolve().parents[1] / "configs"
    a = yaml.safe_load((root / "part_a.yaml").read_text(encoding="utf-8"))
    b = yaml.safe_load((root / "part_b.yaml").read_text(encoding="utf-8"))
    for key in ("difficulty", "max_steps", "guidance"):
        assert a["env"].get(key) == b["env"].get(key), key
    for label, spec in b["algorithms"].items():
        if "guidance" in spec.get("env_overrides", {}):
            assert label.endswith("_noroute"), label


def test_mission_overrides_keep_the_floor():
    """Picking a start room / bed changes only the mission, not the building or the people."""
    from medinav.env.maps import generate_layout
    base = generate_layout(2_000_000, difficulty=2)
    other = next(r.name for r in base.rooms if r.kind != "pharmacy")
    bed = next(b for r in base.rooms for b, _ in r.beds if b != base.goal_label)
    lay = generate_layout(2_000_000, difficulty=2, start_room=other, goal_bed=bed)
    assert (lay.grid == base.grid).all()
    assert lay.start_room == other and lay.goal_label == bed
    assert lay.room_at(lay.start) == other
    assert np.isfinite(lay.geodesic_m(lay.start))
    same = generate_layout(2_000_000, difficulty=2, start_room=None, goal_bed=None)
    assert same.mission == base.mission and np.allclose(same.goal, base.goal)
    with pytest.raises(ValueError):
        generate_layout(2_000_000, difficulty=2, goal_bed="Z-9")
    with pytest.raises(ValueError):
        generate_layout(2_000_000, difficulty=2, start_room="Cafeteria")


def test_clickable_floor_map_targets():
    """Every clickable point on the floor map maps back to a real room or bed."""
    from medinav.env import make_env
    from medinav.viz import pick_figure
    env = make_env({"difficulty": 2})
    env.reset(seed=0, options={"map_seed": 2_000_000})
    snap = env.unwrapped.state_snapshot()
    fig, targets = pick_figure(snap, height=300)
    rooms = {r.name for r in snap["rooms"]}
    beds = {(r.name, b) for r in snap["rooms"] for b, _ in r.beds}
    assert len(fig.data[0].x) == len(targets[0]) and len(fig.data[1].x) == len(targets[1])
    assert {t[1] for t in targets[0]} == rooms                  # every room is clickable
    assert {(t[1], t[2]) for t in targets[1]} == beds           # every bed is clickable
