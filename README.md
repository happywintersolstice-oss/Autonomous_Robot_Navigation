# MediNav-RL: hospital delivery robot navigation with reinforcement learning

BMDS2114 Machine Learning assignment. A differential-drive robot with a 24-ray LiDAR delivers medicine across procedurally generated hospital floors. It has to avoid walls, clutter and walking people, and keep a social distance. We compare 11 RL algorithms against A\* and random baselines, then refine the best ones with tuning, missing-data handling, curriculum learning, a hybrid A\*+RL agent and ensembles.

## Quick start (Windows)

1. Double-click **`Setup.cmd`**. It creates `.venv` and installs everything (needs internet, about 5 minutes).
2. Double-click **`Run Smoke Test.cmd`**. It trains every algorithm for a few seconds and produces all figures. It's a check that the install works, not real results.
3. Double-click **`Start App.cmd`** to open the dashboard in your browser.

The same steps from a terminal:

```bat
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m pytest -q                                          # 45 tests, about 30 s
python -m medinav pipeline --config configs/smoke.yaml       # tiny end-to-end run
streamlit run app/streamlit_app.py
```

## Running the real experiments

| Step | Command | Output |
|---|---|---|
| Part A: train, evaluate, stress-test, report | `python -m medinav pipeline --config configs/part_a.yaml` | `runs/part_a/results/` |
| Train one member's algorithms only | `python -m medinav train --config configs/part_a.yaml --only ppo a2c` | `runs/part_a/<algo>/seed_*/` |
| Fewer seeds while developing | `... train --config configs/part_a.yaml --seeds 0 1` | |
| Hyper-parameter tuning (TPE vs random) | `python -m medinav tune --config configs/tuning.yaml --label ppo` | `runs/tuning/ppo_tpe/`, `ppo_random/`, `tuning_ppo.png` |
| Part B refinements and ensembles | `python -m medinav pipeline --config configs/part_b.yaml` | `runs/part_b/results/` |
| Re-make figures only | `python -m medinav report --config configs/part_a.yaml` | |

Training skips runs that already exist, so members can train their own algorithms on their own laptops and copy the `runs/part_a/<algo>/` folders into one shared copy. After that, `evaluate` and `report` cover everyone's models.

**Compute budget (CPU laptop, rough):**

- Tabular: about 2 min per seed.
- PPO and A2C: about 5–10 min per seed.
- The DQN family: about 10–15 min per seed.
- SAC and TD3: about 20–30 min per seed.

All of Part A with 5 seeds takes several hours, so split it by member and start early.

## What is where

```
configs/          experiment definitions (YAML): every result is reproducible from these
src/medinav/
  env/            maps.py (procedural floors + train/val/test split), nav_env.py (robot, LiDAR,
                  people, reward), humans.py, planning.py (A*, geodesic distance), wrappers.py
                  (missing-LiDAR imputation, frame stacking)
  agents/         baselines.py (Random, A*), tabular.py (Q-learning, SARSA; Member 1),
                  dqn_family.py (DQN / Double / Dueling; Member 2),
                  policy_gradient.py (PPO / A2C / RecurrentPPO; Member 3),
                  continuous_control.py (SAC / TD3; Member 4), ensemble.py, registry.py
  training/       train.py (multi-seed training), callbacks.py (validation curves, best model, curriculum)
  evaluation/     rollout.py (episode metrics + failure types), metrics.py (bootstrap CI, IQM,
                  permutation test, sample efficiency), evaluate.py, interpret.py
                  (permutation importance, gradient saliency, Q-table policy map), report.py (figures)
  tuning.py       Optuna search
  cli.py          the `python -m medinav ...` commands
app/              Streamlit prototype (simulator, head-to-head, results browser)
tests/            unit tests
runs/             created by training: models, learning curves, results
```

## How the work maps to the assignment spec

| Spec item | Where it is done |
|---|---|
| Frame the problem / define the environment | `env/nav_env.py` docstring, `docs/how_it_works.md` |
| Diverse models | 11 algorithms across tabular, value-based, on-policy and off-policy families; continuous and discrete actions |
| Validation method | Disjoint train/validation/test map seeds; 5 seeds per algorithm; stratified bootstrap 95% CI (`metrics.py`) |
| Task-suited metrics | Success, collision (wall/person), path efficiency vs A\*, time to goal, social intrusion, min person distance, IQM return, seed stability, sample efficiency, decision latency |
| Robustness across conditions | 9 stress tests: noise, 20/40% dropout, crowds, fast people, clutter, bigger map, hardest level, non-stationary people (`robustness` in configs) |
| Patterns and influencing factors | Permutation importance per input group, LiDAR saliency polar plots, tabular policy map, reward-term breakdown |
| Error types | `failure_type`: wall, person, stuck/oscillating, too slow; failure breakdown figure; trajectory gallery |
| Refine representations / learning | Geodesic vs Euclidean reward ablation, curriculum learning, frame stacking, LSTM |
| Shortlist | `summary.md` + `significance.md` (permutation test between top agents) |
| Part B tuning | Optuna TPE vs random, median pruning (`tuning.py`) |
| Missing information | 5 imputation strategies + frame stack + RecurrentPPO, trained with 20% LiDAR dropout |
| Combine models | Q-value averaging, seed ensemble, majority vote, continuous action averaging, safety shield, hybrid A\*+RL |
| New data / adaptation | Unseen test maps, bigger maps, people who start walking mid-episode |
| Automation | One command per stage; YAML configs; figures and tables generated automatically |
| Prototype | `app/streamlit_app.py` |

## Robustness of the code itself

- **One failing run never stops an experiment.** The error is printed, recorded in `runs/<experiment>/failed_runs.json`, and that run is retried the next time you run the command. Finished runs are skipped, so an interrupted experiment (Ctrl+C, laptop sleep) resumes where it stopped.
- **Corrupt or half-copied model folders** are skipped by `evaluate`/`report`, with a warning. The app shows an error message instead of crashing.
- **Config mistakes** (missing file, bad YAML, unknown label, unknown env setting) give a one-line error.
- **Parallel training.** Each process is capped at 2 CPU threads (`OMP_NUM_THREADS`), so several `train --only <algo>` commands can run side by side on one laptop.
- **Tests.** `tests/test_core.py` covers the environment, planning and statistics. `tests/test_extended.py` covers reward continuity, every deep algorithm (train → save → load → act), ensembles, error handling, configs and tuning.

## Things to check yourselves (for Q&A)

- The default hyper-parameters are sensible starting points, not tuned results. Run the tuning, then paste `best_params.yaml` into `configs/part_b.yaml`.
- Choose the Part B shortlist (PPO/SAC/DQN in the config) from your actual Part A results.
- Every module starts with a comment explaining *why* it exists. Each member should be able to explain their algorithm file line by line and change a hyper-parameter live.
