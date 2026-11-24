# MediNav-RL: hospital delivery robot navigation with reinforcement learning

BMDS2114 Machine Learning assignment. A differential-drive robot with a 24-ray LiDAR delivers medicine from the **pharmacy** to a numbered **ward bed** in procedurally generated hospital wings: a 24 × 16 m floor with a main corridor and named rooms (pharmacy, wards, nurse station, lobby, supply). The robot knows the floor plan's route but not where people and trolleys are. It has to avoid them, keep a social distance and arrive quickly.

We compare 9 RL algorithms against A\* and random baselines on never-seen floors. We then refine the best ones with tuning (random vs TPE), a route-guidance ablation, missing-LiDAR handling, curriculum learning, reward shaping, longer training and ensembles. See [docs/how_it_works.md](docs/how_it_works.md) for the one-page explanation, including how the environment evolved.

## Quick start (Windows)

1. Double-click **`Setup.cmd`**. It creates `.venv` and installs everything (needs internet, about 5 minutes).
2. Double-click **`Run Smoke Test.cmd`**. It trains every algorithm for a few seconds and produces all figures. It's a check that the install works, not real results.
3. Double-click **`Start App.cmd`** to open the app in your browser. It has four tabs:
   - **Delivery:** pick a floor, a busy-ness level and an agent, then watch an animated replay.
   - **Compare agents:** run several agents on the same floor.
   - **Results:** a dashboard of every table and figure.
   - **How it works.**

The same steps from a terminal:

```bat
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m pytest -q                                          # 57 tests, about 2 min
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

On a 20-thread laptop, all of Part A (9 algorithms × 3 seeds) took about 40 minutes with one process per run (`Train Part A.cmd`).

## Results (hospital floor, v3)

All numbers are success rates on 50 never-seen test floors × 3 training seeds ("Busy day": 4 people, 3 trolleys).

| Agent | Success | Collision | Note |
|---|---|---|---|
| SAC, 2× training (Part B) | **95%** | 5% | best overall; 81% on "Visiting hours" |
| TD3, 2× training (Part B) | 91% | 7% | |
| TD3, Optuna-tuned (Part B) | 83% | 10% | TPE beat random search for TD3 |
| SAC (Part A) | 80% | 19% | significantly better than TD3 and Dueling DQN (p < 0.01) |
| Ensemble SAC + TD3 | 79% | 19% | |
| Dueling DQN (Part A) | 63% | 35% | best discrete-action agent |
| PPO (Part A) | 54% | 43% | |
| A\* route-follower (baseline) | 38% | 62% | knows the floor plan, not the people |
| Q-learning (Part A / tuned) | 15% / 27% | 18% / 19% | 1,944-state table is too coarse |
| Any agent **without** route guidance | ≤ 3% | | the floor plan matters most |

Weak spots for every agent: 40% LiDAR loss (≤ 34%) and "careless" people who walk into the robot (≤ 56%).
Full tables, CIs, significance tests and figures: `runs/part_a/results/` and `runs/part_b/results/`, or the app's **Results** tab.

**Known limitation.** People yield by stopping about 5 cm from the robot, so the median closest approach is about 0.07 m for every agent, including A\*. A wider yielding distance is a future-work item; changing it would mean retraining everything.

## What is where

```
configs/          experiment definitions (YAML): every result is reproducible from these
src/medinav/
  env/            maps.py (hospital floor generator + train/val/test split), nav_env.py (robot, LiDAR,
                  people, reward), humans.py, planning.py (A*, geodesic distance), wrappers.py
                  (missing-LiDAR imputation, frame stacking)
  agents/         baselines.py (Random, A*), tabular.py (Q-learning, SARSA; Member 1),
                  dqn_family.py (DQN / Double / Dueling; Member 2),
                  policy_gradient.py (PPO / A2C; Member 3),
                  continuous_control.py (SAC / TD3; Member 4), ensemble.py, registry.py
  training/       train.py (multi-seed training), callbacks.py (validation curves, best model, curriculum)
  evaluation/     rollout.py (episode metrics + failure types), metrics.py (bootstrap CI, IQM,
                  permutation test, sample efficiency), evaluate.py, interpret.py
                  (permutation importance, gradient saliency, Q-table policy map), report.py (figures)
  tuning.py       Optuna search
  cli.py          the `python -m medinav ...` commands
  render.py       static floor-plan drawing (report figures, app background)
  viz.py          animated / interactive floor plan for the app (Plotly)
app/              Streamlit prototype (delivery replay, agent comparison, results dashboard)
tests/            unit tests
runs/             created by training: models, learning curves, results
```

## How the work maps to the assignment spec

| Spec item | Where it is done |
|---|---|
| Frame the problem / define the environment | `env/nav_env.py` docstring, `docs/how_it_works.md` |
| Diverse models | 9 algorithms across tabular, value-based, on-policy and off-policy families; continuous and discrete actions |
| Validation method | Disjoint train/validation/test floor seeds; 3 seeds per algorithm; stratified bootstrap 95% CI (`metrics.py`) |
| Task-suited metrics | Success, collision (wall/person), path efficiency vs A\*, time to goal, social intrusion, min person distance, IQM return, seed stability, sample efficiency, decision latency |
| Robustness across conditions | 10 stress tests: noise, 20/40% LiDAR loss, crowds, fast people, clutter, bigger wing, visiting hours, careless people, non-stationary people (`robustness` in configs) |
| Patterns and influencing factors | Permutation importance per input group, LiDAR saliency polar plots, tabular policy map, reward-term breakdown |
| Error types | `failure_type`: wall, person, stuck/oscillating, too slow; failure breakdown figure; trajectory gallery |
| Refine representations / learning | Route-guidance ablation, geodesic vs Euclidean reward, curriculum learning, frame stacking, longer training; environment iterations v1 → v3 driven by error analysis |
| Shortlist | `summary.md` + `significance.md` (permutation test between top agents) |
| Part B tuning | Optuna TPE vs random, median pruning (`tuning.py`) |
| Missing information | Gap-filling strategies + frame stack, trained with 20% LiDAR dropout |
| Combine models | Q-value averaging, seed ensemble, majority vote, continuous action averaging, safety shield; A\* route + learned local control |
| New data / adaptation | Unseen test floors, bigger wing, people who start walking mid-delivery |
| Automation | One command per stage; YAML configs; figures and tables generated automatically |
| Prototype | `app/streamlit_app.py` |

## Robustness of the code itself

- **One failing run never stops an experiment.** The error is printed, recorded in `runs/<experiment>/failed_runs.json`, and that run is retried the next time you run the command. Finished runs are skipped, so an interrupted experiment (Ctrl+C, laptop sleep) resumes where it stopped.
- **Corrupt or half-copied model folders** are skipped by `evaluate`/`report`, with a warning. The app shows an error message instead of crashing.
- **Config mistakes** (missing file, bad YAML, unknown label, unknown env setting) give a one-line error.
- **Parallel training.** Each process is capped at 2 CPU threads (`OMP_NUM_THREADS`), so several `train --only <algo>` commands can run side by side on one laptop.
- **Tests.** `tests/test_core.py` covers the environment, planning and statistics. `tests/test_extended.py` covers reward continuity, every deep algorithm (train → save → load → act), ensembles, error handling, configs and tuning.

## Things to check yourselves (for Q&A)

- The Part A hyper-parameters are sensible starting points, not tuned results. The tuned settings in `configs/part_b.yaml` come from `runs/tuning/*/summary.json`.
- Your fact sheet is in `runs/part_b/results/factsheets/`. It has the numbers for your own algorithms; the interpretation is yours to write.
- Every module starts with a comment explaining *why* it exists. Each member should be able to explain their algorithm file line by line and change a hyper-parameter live.
