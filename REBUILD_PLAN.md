# Rebuild plan: Autonomous Robot Navigation (BMDS2114)

Status: **approved (MediNav base). In progress; see the progress log at the end.**
Written 23 Nov 2026 from `Assignment specification (v1).pdf`, `Assessment rubric (v1).docx` and the current repo.

---

## 1. What I found

### The repo on `main` (the "Codex" version)
- A warehouse grid robot with partial observation and temporary blockages. It has **4 learners** (Q-learning, SARSA, DQN, PPO), a 4-model voting ensemble, an A*-style replanner and a Streamlit app on port 8517.
- 45 commits and 10 feature branches on GitHub. Most of the volume is "phase" research records, hashes, receipts, protocols and archives (`archive/`, `docs/history/`, `.runtime/`, `artifacts/phase5*`). Most of that is process bookkeeping that a marker will never read.
- Its own status doc ([docs/rubric-completion-status.md](docs/rubric-completion-status.md)) says full compliance is **not established**.
- Weak points against the rubric:
  - **Too few, too small experiments.** Policies were trained for about 40k actions, on one panel with one seed. The rubric asks for "various performance metrics" and "extra metric or measurement", and the spec asks for cross-validation/bootstrapping and robustness across conditions.
  - **Little genuine tuning.** Part B (Model Optimization) needs a visible hyper-parameter search, with random search vs a smarter search, as the spec says.
  - **The ensemble does not help.** It tied the best single models and was slower.
  - **Tests were deleted** at your earlier request. The rubric's "exceptions handled well" and "no bugs during demo" both benefit from tests.
  - **No clean member-to-algorithm split.** 60% of the marks are individual, so each member needs their own algorithms to present.
  - **The navigation problem is a standard grid world.** That puts the "unique title, creative ideas" part of the Problem Statement row at risk.

### The untracked `medinav-rl/` folder (a second draft already on disk)
- "MediNav-RL": a hospital medicine-delivery robot with differential drive, 24-ray LiDAR, procedurally generated floors and **walking people with a social-distance penalty**.
- It has 11 RL algorithms plus Random/A* baselines, a train/val/test map split, multi-seed training, bootstrap CIs, IQM, permutation tests, 9 stress tests, Optuna tuning (TPE vs random), curriculum learning, a hybrid A*+RL agent, ensembles, interpretability (permutation importance, saliency), a Streamlit app and 12 unit tests.
- About 3,100 lines in total. It looks well structured, but **it has never been run**: there is no `runs/` folder and no results.
- `Replace Codex Version.cmd` was meant to swap it in. It created `_codex_old\`, but that folder is **empty**, so the swap stopped part-way and moved nothing. Running the script again would fail with "_codex_old already exists".
- Your Python is **3.14**, and torch, stable-baselines3 and gymnasium are **not installed**. 3.14 is too new for some of these wheels, so I plan to use a **Python 3.12 venv**. The machine has 20 CPU threads and no CUDA GPU.

## 2. Recommendation

**Rebuild on the MediNav-RL design, but treat the draft as a starting point, not finished work.** Review it, cut it down to a size four laptops can train before Week 6, run it for real, and fix what breaks. Retire the warehouse version cleanly through git, not through folder shuffling.

Why this design rather than continuing the warehouse one:
- A continuous-space LiDAR robot among moving people, with a social-comfort constraint, is a more distinctive problem. It addresses "unique title" and "creative ideas" in Problem Statement (10).
- It has room for diverse algorithm families: tabular, value-based, on-policy and off-policy. That gives each member their own models for the individual Programming (20), Model Optimization (10) and Results (10) rows.
- It produces the measurements the rubric rewards: multi-seed CIs, stress tests, failure types and interpretability.

## 3. Scope (trimmed to fit a CPU and the deadline)

The draft's 11 algorithms and 5 seeds would take many hours per member. Proposed scope:

| Member | Algorithms (individual work) | Their Part B optimization |
|---|---|---|
| 1 | Q-learning, SARSA (discretised LiDAR state) | Grid search vs random search over α, γ, ε-decay; state-binning ablation |
| 2 | DQN, Double DQN (Dueling optional) | Optuna TPE vs random search; network size and target-update ablation |
| 3 | PPO, A2C | Optuna TPE vs random search; frame-stacking / LSTM for missing LiDAR |
| 4 | SAC (TD3 optional) | Optuna tuning; reward-shaping ablation (geodesic vs Euclidean) |
| Team | Random, A*, **hybrid A*+RL**, **ensembles** (voting / Q-averaging, plus a learned gating "selector") | Curriculum learning; ensemble vs best single model |

With 3 members, Member 4's SAC moves to the team set (or is dropped), and the table becomes 3 owners.

- **Seeds:** 3 for development, 5 for final results only if time allows.
- **Budgets:** I will measure real wall-clock time on your machine with a smoke run, then fix the step budgets so that each member's full training fits in **≤ 2–3 hours on one laptop**.

## 4. Work phases and what each produces

| # | Phase | Output | Rubric rows it serves |
|---|---|---|---|
| 0 | **Repo reset (safe).** Tag current `main` as `warehouse-final`, push the tag, move the old tree to a `legacy/warehouse` branch, make MediNav the root of `main`. Delete the empty `_codex_old\` and the swap script. Nothing in git history is lost. | Clean root: `src/ configs/ app/ tests/ docs/` | Organization |
| 1 | **Environment setup.** Python 3.12 venv, pinned `requirements.txt`, one-click `Setup.cmd`. | Working install | Degree of completion |
| 2 | **Code review and hardening of the draft.** Read every module. Fix bugs. Add input validation and exception handling (bad config, missing model, corrupt checkpoint). Add clear comments that justify design choices, as the spec requires. Extend tests (env invariants, reward terms, map split leakage, each agent's train/predict, ensemble, metrics). | `pytest` green, coverage of every module | Programming (exceptions, comments), no-bug demo |
| 3 | **Smoke run and timing.** A tiny end-to-end pipeline that measures time per algorithm, then sets the real budgets. | `configs/part_a.yaml` with feasible budgets | Degree of completion |
| 4 | **Part A: train and shortlist.** Run every algorithm over several seeds on disjoint train/val/test maps. Evaluate on the test maps with metrics: success, collision (wall/person), timeout, path efficiency vs A*, time-to-goal, social intrusion, return IQM, seed stability, sample efficiency and latency, with bootstrap 95% CIs. Add a permutation test between the top agents, 9 stress conditions, failure taxonomy, permutation importance and LiDAR saliency. Produce an automated shortlist. | `runs/part_a/results/` tables + figures | Part A items 1–7, Results, Output |
| 5 | **Part B: refine.** Per-member tuning (random vs TPE, with search-efficiency plots). Missing-data handling (zero-fill, last-value, mask + frame stack, LSTM) under LiDAR dropout. Curriculum learning, reward ablation, hybrid A*+RL and ensembles compared with the best single model. Re-test on unseen maps and under non-stationary people. | `runs/part_b/results/` | Model Optimization, Part B 1–4 |
| 6 | **Prototype app.** Streamlit pages: live simulator (pick map, difficulty, agent; watch LiDAR and the trajectory), head-to-head race, results explorer with all figures and CIs, robustness dashboard and an "explain this decision" saliency view. Consistent theme and colours. | `Start App.cmd` | Output (visual design), System implementation, Presentation demo |
| 7 | **Report support (not the report itself).** A per-member *fact sheet*: what each of your algorithms does, the exact numbers, figures and where they live in code. Also a report skeleton that follows the spec's required sections and format (TNR 11pt, cover, about 40 pages), a task-allocation/project-plan table template, and a demo script per member. | `docs/report_kit/` | Organization, Presentation |

**Order:** 0 → 1 → 2 → 3, then 4 and 5 run in the background (long training) while 6 is built, and 7 comes last, once the numbers exist.

## 5. Important: the assignment's AI rule

> **Update, 23 Nov 2026:** the team reports that the AI category has been changed from Yellow to **Green**, which allows AI help with building the project. The rest of this section describes the original Yellow rule (spec v1) and is kept for reference. Under Green, two things still apply:
> - **Keep written proof of the change.** The spec PDF on file still says Yellow, so save the tutor's announcement or email.
> - **Still declare AI use.** Fill in the disclosure form; the git history records what was AI-assisted. Each member must still be able to explain, justify and modify their own code during the Q&A.

The spec classifies this assignment as **Yellow: Limited AI**. AI may help with "syntax support, data analysis… technical support". It must **not** "generate complete solutions, produce reports", core arguments or critical analysis. All AI use must be declared on the AI Usage Disclosure Form. Each member must be able to explain, justify and **modify** their code live.

Rebuilding the entire project with me is close to that line, so the plan is designed to keep you on the right side of it:
- **I will not write the report prose, critical analysis or justifications.** Phase 7 gives you facts, numbers and structure; your team writes the arguments.
- Each member should **own, read and run** their algorithm files, and ideally make their own tuning choices. I can walk each member through their code.
- Keep a log of what I did. The git commits record it, which makes the disclosure form easy and honest.
- If you are unsure whether this level of help is allowed, **ask your tutor before Phase 2.** Whether to proceed is your call; I am flagging it because undeclared or excessive AI use can cost marks.

## 6. Decisions I need from you before I start

1. **Base:** MediNav-RL hospital design (recommended), or keep and extend the warehouse version?
2. **Team:** 3 or 4 members? Which member takes which algorithm set? (Placeholders "Member 1–4" are fine for now.)
3. **Deadline:** which week is it now, and when is Week 6? This sets the training budgets.
4. **Repo reset (Phase 0):** OK to restructure `main` and push the tag and branch to GitHub? I will not force-push or delete remote branches.
5. **Old research branches** (`feature/phase5*` etc.): leave them on GitHub as they are (recommended), or archive them?

## 7. What "done" will mean

- `Setup.cmd` → `Run Smoke Test.cmd` → `Start App.cmd` all work on a clean clone.
- `pytest` passes.
- Real Part A and Part B results exist, with CIs, stress tests, tuning plots and an ensemble comparison. Every figure can be regenerated from a config with one command.
- Each member has a fact sheet and demo script for their own algorithms.
- Everything is committed and pushed. The old version remains recoverable from the `warehouse-final` tag.

---

## 8. Progress log

**23 Nov 2026**

- **Base chosen:** MediNav-RL.
- **Phase 0, part 1 (done).**
  - The old version is preserved as tag `warehouse-final` and branch `legacy/warehouse`. Both are local and not pushed yet.
  - Its files were moved to `Desktop\Autonomous_Robot_Navigation_warehouse_backup\`. Nothing was deleted.
  - Some permission-restricted `.runtime` test folders of the old app could not be moved. They remain in `autonomous-robot-navigation-rl/` and are git-ignored. You can delete them by hand.
- **Phase 1 (done).**
  - `.venv` runs on Python 3.14, which works: torch 2.15, SB3 2.9, gymnasium 1.4, optuna 5.1, streamlit 1.67. The 3.12 environment in the plan was not needed.
- **Phase 2 (done).**
  - **Reward bug fixed.** The progress reward jumped when the robot was near a wall.
  - **Error handling added.**
    - A failing run no longer stops the experiment.
    - Interrupted experiments resume where they stopped.
    - Corrupt models are skipped.
    - Config errors give clear messages, and the app shows errors instead of crashing.
  - **Ensembles check their members.** Members with mismatched action types are rejected.
  - **Reproducibility bug fixed.** The random baseline's scores depended on the order the test conditions ran in.
  - **Faster runs.** Evaluation runs one condition per process, and each process is capped at 2 threads.
  - **Tests.** 46 pass: the original 12 plus 34 new.
- **Phase 3 (done).**
  - The smoke pipeline runs end to end and writes 40 figures and tables.
  - Pilot at 60–120k steps, validation success: DQN 50%, PPO 45%, Q-learning about 25%, SAC 20%. The agents learn, and the Part A budgets were kept.
- **Phase 0, part 2 (done).**
  - MediNav now lives at the repo root. `.venv` was rebuilt there, and all tests pass.
- **Phase 4 (done), with one iteration.**
  - **Environment v1.** Best RL agent 49% success vs A* 66%.
  - **Error analysis.** About 95% of collisions with people were people walking *into* the robot, often while it stood still. No policy could avoid these.
  - **Environment v2.** People yield to the robot. The old behaviour is kept as the `careless_people` stress test, and the v1 results are archived in `runs/env_v1/`.
  - **Part A results in v2** (50 unseen test maps × 3 seeds):
    - SAC 73% and TD3 67% match A* (70%) with about a third of its collisions.
    - SAC beats Dueling DQN (55%) significantly (p = 0.002).
    - Weak spot for all RL agents: LiDAR dropout.
  - **Tuning, Optuna random vs TPE**, 20 trials each, best validation score:
    - Dueling DQN: 0.76 (random) vs 0.71 (TPE).
    - TD3: 0.76 with both.
    - PPO: 0.61 with both.
    - Q-learning: 0.55 (TPE) vs 0.50 (random).
- **Phase 5, first attempt (superseded).** Part B in the 10 m room (env v2): sac_hybrid 82%, sac_long 81%. Archived in `runs/env_v2_room/`.
- **Feedback: the map was too small** ("I don't know where the pharmacy or the wards are") and the UI wasn't liked.
  - **Environment v3: hospital wing.** 24 × 16 m with a corridor, Pharmacy, 2–6 Wards with numbered beds, Nurse station, Lobby/Supply, and trolleys. Missions read like "Pharmacy → Ward C, bed C-3".
  - **First result:** no agent learned without the floor plan (0% after 120k steps). Fixed by giving the robot the next waypoint on the planned route (`guidance: true`), as real hospital robots do. "No route guidance" became a Part B ablation.
  - **UI redesigned.** Polished Streamlit with Plotly animation, a named-room floor plan, KPI cards, Compare agents and Results tabs, and member fact sheets.
- **Phase 4 again (done).** Part A on v3 (50 test floors × 3 seeds):
  - SAC 80%, TD3 64%, Dueling DQN 63%, PPO 54%, Double DQN 51%, DQN 48%, A2C 48%, A* 38%, Q-learning 15%, SARSA 10%.
  - SAC is significantly better than TD3 and Dueling DQN (p < 0.01).
- **Tuning (done).** Optuna, 20 trials per sampler. Best validation score:
  - Dueling DQN: 0.87 random vs 0.82 TPE.
  - PPO: 0.66 random vs 0.61 TPE.
  - TD3: 0.71 TPE vs 0.61 random.
  - Q-learning: 0.41 both.
- **Phase 5 (done).** Part B on v3: 15 experiments × 3 seeds plus 7 ensembles/shields.
  - **Bug found and fixed.** Part B's config was missing `guidance: true`, so 12 experiments first trained without route guidance. They were retrained, and a test now checks that Part B keeps Part A's base environment.
  - **Results:**
    - sac_long 95% and td3_long 91% (both p < 0.05 vs Part A).
    - td3_tuned 83%, q_learning_tuned 27% (vs 15%).
    - ppo_tuned unchanged (54%); dueling_dqn_tuned and curriculum −6 pts.
    - LiDAR-gap filling: max-range 47% and last-valid 47%, vs gaps kept 31% and 3-frame stack 19%.
    - Straight-line reward 10% (vs walking-route 54%).
    - No route guidance ≤ 3% for every family.
    - Ensembles: DQN-family Q-average 69% beats every single DQN; SAC + TD3 79%. The safety shield does not help SAC (79% vs 80%).
- **Still open:**
  - Decisions 2, 3 and 5 from section 6: team names and ownership, and the deadline.
  - Opening the PR, since `gh` isn't installed.
