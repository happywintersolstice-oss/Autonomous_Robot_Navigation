# File and experiment naming guide

The next owner-authorized study is milestone 8, a DQN input comparison.
`m8-control-dqn-seed101` means the original four normalized absolute inputs;
`m8-relative-dqn-seed101` means those same four inputs plus two goal offsets.
The same seed is paired across treatments, and neither name means a tuned model.
Main seeds are 101, 211, 307, 401 and 503. Names containing `pilot-seed1701`
are excluded software/throughput pilots. See [current status](representation_status.md).
New runs include `training_traces/episode-001.json` through `003.json`: actual
chronological exploratory training actions. They are separate from frozen-policy
validation replays, and incomplete prefixes do not have terminal episode metrics.

The owner requested an additional PPO baseline on 8 November 2026. Its completed
run names are `m7-ppo-pilot-seed101` and `m7-ppo-seed7`, `seed19`, `seed43`,
`seed67`, `seed91`; its verified comparison study is `four-rl-v1`. These identify
the new extension, not a tuned PPO or a replacement for the frozen final study.
Each main PPO run used 200,000 interactions; the independent pilot used 10,000
and is excluded from comparison. The seed controls training randomness, not
algorithm quality or a tuning level. See [PPO status](ppo_extension_status.md)
and [the four-algorithm results](ppo_comparison_results.md).

This guide explains the project folders, experiment labels and saved files.
Most experiment folders contain different runs of the same algorithms, rather
than separate robot programs.

## Decode a run name

```text
m4-tuned-q-learning-seed101
│  │     │          │
│  │     │          └─ Random seed used for this training run
│  │     └──────────── Q-learning algorithm
│  └────────────────── Improved training settings
└───────────────────── Milestone 4
```

This means: **a Q-learning model trained during milestone 4 using tuned settings
and random seed 101**.

### What a seed means

A seed initializes a random-number generator. It controls training randomness,
including scenario sampling, exploratory actions and random training tie-breaking.
For DQN it also affects neural-network initialization. The tabular Q-tables start
at zero; their differences develop through different training experiences.

- `seed101` and `seed211` use the same method/settings but different random sequences.
- Several seeds help distinguish consistent performance from a fortunate run.
- A larger seed does not mean a better or newer model.
- The seed is not an episode number or training-step count.
- Matching the seed and setup supports repeatability on the tested software stack;
  exact results across different hardware/library versions are not guaranteed.

## Training-run names

These directories are under [artifacts/runs](../artifacts/runs/).

| Name or pattern | Meaning |
|---|---|
| `smoke-seed7` | Initial quick Q-learning check: 30,000 training actions. An early development model. |
| `m2-q-learning-seed19` | Milestone 2 Q-learning run, seed 19. |
| `m2-sarsa-seed43` | Milestone 2 SARSA run, seed 43. |
| `m2-…-seed7-repeat` | Another run with seed 7 to check repeatability. It is not an additional independent training seed. |
| `m3-dqn-pilot-seed7` | Small 10,000-action DQN pilot before the full comparison. |
| `m3-q-learning-seed67` | Milestone 3 Part A Q-learning run. |
| `m3-sarsa-seed91` | Milestone 3 Part A SARSA run. |
| `m3-dqn-seed19` | Milestone 3 Part A DQN run. |
| `m4-search-q-learning-c07-seed19` | Candidate settings `c07`, tested with Q-learning and seed 19. |
| `m4-control-q-learning-seed101` | Original settings, used as a matched baseline for the tuned run. |
| `m4-tuned-q-learning-seed101` | Selected improved settings, trained with seed 101. |
| `m4-terminal-q-learning-seed101` | Reward experiment: ordinary step cost changed from −0.01 to zero. |
| `m4-voting-q-learning` | Three saved Q-learning policies vote on each action. |

The milestone-4 search/control/tuned/terminal patterns also apply to SARSA.

`c01` through `c08` identify candidate configurations: different combinations of
learning rate, discount and exploration settings. They are not eight algorithms,
episodes or checkpoints. Both shortlisted families selected candidate `c07`.

`terminal` identifies the reward experiment. Goal and collision feedback remain
+10 and −5, while the ordinary step cost is zero. It does not mean “the final model.”

The vote uses independently trained members with seeds **101, 211 and 307**, so
the voting directory has no single seed. Majority action wins; a three-way tie
uses the first member's action.

### Seeds used in each stage

| Stage | Seeds |
|---|---|
| Milestone 2 preliminary comparison | 7, 19, 43; seed 7 also repeated |
| Milestone 3 main Part A comparison | 7, 19, 43, 67, 91 |
| Milestone 4 search | 7, 19 |
| Milestone 4 controls, tuned confirmations and reward experiments | 101, 211, 307, 401, 503 |
| Milestone 4 voting members | 101, 211, 307 |

The final single-policy seed **101** was predeclared rather than selected by
searching final-test scores.

### Control versus tuned

For example, `m4-control-q-learning-seed101` and
`m4-tuned-q-learning-seed101` share a seed and interaction budget but use different
training settings.

| Setting | Control | Tuned |
|---|---:|---:|
| Learning rate | 0.3 | 0.9 |
| Discount | 0.99 | 0.99 |
| Starting exploration probability | 1.0 | 0.7 |
| Ending exploration probability | 0.1 | 0.03 |
| Fraction of training over which exploration decays | 0.8 | 0.3 |

This compares refinement without quietly giving the tuned policy more training
actions. The main budget is 200,000 training actions across many episodes; each
episode has its own cap of 200 actions.

## Files inside a run directory

Example: [m4-tuned-q-learning-seed101](../artifacts/runs/m4-tuned-q-learning-seed101/).

| File or directory | Contents |
|---|---|
| `configuration.json` | Exact training settings and environment configuration. |
| `manifest.json` | Run identity, seed, status, training budget, timings and artifact information. |
| `checkpoint.npz` | Q-table at the end of the full training budget. |
| `checkpoint.json` | Metadata and checksum for that table. |
| `selected.npz` | Q-table selected using validation monitoring. |
| `selected.json` | Selected-model metadata, including its actual training age. |
| `checkpoints/` | Intermediate saved models from training. |
| `training.csv` | Completed training-episode statistics: steps, return, outcome and exploration level. |
| `monitoring.json` | Results on the small fixed validation-monitoring subset during training. |
| `validation.json` | Aggregate results on the complete validation set. |
| `pilot.json` | Training diagnostics such as goal discoveries and throughput. The filename is retained for full runs too. |
| `q_learning_episodes.json` | One metrics record per validation scenario. |
| `q_learning_trajectories.jsonl.gz` | Compressed complete validation episodes, including every transition. Each line contains one episode record. |
| `q_learning_trajectory.png` | A saved example route illustration. |
| `visitation.npz` | State/action visit counts across the full training budget. |
| `selected_visitation.npz` | Visit counts only through the selected checkpoint's training age, for newer runs that record this separately. |
| `replays/` | Representative success, collision and timeout episodes for inspection. |
| `source/` | A historical copy of the code/configuration used when the run was created. |

SARSA and DQN use corresponding prefixes such as `sarsa_episodes.json` and
`dqn_episodes.json`. DQN models use `.zip` instead of `.npz`. Voting uses a
`voting.json` descriptor referencing its three member checkpoints and fixed rule.

### Selected versus final-budget checkpoint

`checkpoint` and `selected` are not always the same model. An earlier checkpoint
can perform better on validation than the final-budget checkpoint. For tuned
Q-learning seed 101, the selected age happened to be 200,000 interactions.

```text
checkpoints/step-0010000.npz
```

This is a Q-table saved after **10,000 training actions**. Zero padding keeps the
filenames in numerical order. `step-200000` in a policy ID also describes training
age, not the current playback frame.

Snapshots support inference; they do not contain everything needed to resume
training exactly. `selected.json` describes a model; the actual learned Q-values
are inside `selected.npz`.

The training visitation files are different from the viewer's prefix visit-count
overlay. The overlay counts cells in the currently displayed episode prefix.

## Runs versus studies

A **run** records one trained model or evaluated policy. A **study** combines runs
into a comparison or analysis.

| Study directory | Purpose |
|---|---|
| `milestone2-tabular-v1` | Preliminary Q-learning/SARSA comparison. |
| `part-a-v1` | Main three-algorithm comparison. |
| `part-b-v1` | Search, refinement, reward experiment and voting. |
| `milestone5-v1` | Dashboard examples and fixed-map robustness analysis. |
| `milestone5-incomplete-render` | Preserved failed plotting attempt, not a failed learning model. |
| `final-v1` | Frozen final evaluation and release evidence. |

`v1` means version 1 of the study. It is unrelated to a training seed.
Milestones 5 and 6 mainly integrated/analyzed/evaluated existing models, so they
do not require new `m5-*` or `m6-*` training-run directories.

For example:

```text
artifacts/studies/final-v1/
  evaluation/
    runs/
      m3-dqn-seed19/
        summary.json
```

This means **final-test results for the DQN model originally trained during
milestone 3 with seed 19**. It does not mean DQN was retrained during final
evaluation. The parent directory determines what a file named `summary.json`
summarizes.

## Common study-result filenames

| Filename | Meaning |
|---|---|
| `summary.json` | Combined study results, or one policy's results when inside its evaluation directory. |
| `run_results.json` | Individual evaluated records collected by the study. |
| `candidates.json` | Settings tried during the search. |
| `search_summary.json` | Candidate performance. |
| `selected_configurations.json` | Settings chosen using validation. |
| `paired_improvements.json` | Baseline-versus-tuned changes matched by training seed. |
| `paired_scenarios.csv` | Scenario-level comparisons. |
| `reward_ablation.json` | Results of the controlled reward change. |
| `voting_comparison.json` | Voting versus components, including measured costs. |
| `failure_analysis.json` | Collisions, timeouts, loops and other outcomes. |
| `scenario_labels.json` | Difficulty/geometry labels assigned to scenarios. |
| `groups.csv` / `paired_groups.json` | Group results and seed-matched group changes. |
| `target_assessment.json` | Whether declared performance targets were met. |
| `verification.json` | Checks performed against saved evidence. |
| `inputs.json` | Input-file hashes used by an analysis. |
| `development_handoff.json` | Development reference and remaining work at that stage. |
| `early_checkpoints.json` | Location/identity of archived early models for comparisons. |
| `family_choice_before_voting.json` | Algorithm-family choice recorded before voting evaluation. |
| `plan.json` / `protocol.md` | Study settings and procedure. |
| `report.md` | Study findings in readable form. |
| `runner.py` / `analyzer.py` / `analyzer-audited.py` | Archived execution/analysis code. |
| `figure_qa.json` | Figure inspection or rendering-repair notes. |
| `figures/` | Saved charts. |

### Final-release files

These are under [final-v1](../artifacts/studies/final-v1/).

| File or directory | Meaning |
|---|---|
| `freeze.json` | Exact model choices, configurations, input hashes, versions and evaluation matrix fixed before test. |
| `seal.json` | Digest identifying that freeze. |
| `snapshot/` | Exact archived source, models and inputs. |
| `evaluation/` | Final trajectories, metrics, audits and figures. |
| `evaluation/run_results.json` | All 42 evaluated system/seed records, with their individual results. |
| `final_seed_results.csv` | Per-record final results in table form. |
| `development_run_matrix.csv` | Historical development runs, budgets and statuses. |
| `release_manifest.json` | Release identity and report references. |
| `status.json` | Frozen, running, failed, interrupted or complete status. |
| `.gitattributes` | Preserves exact archived bytes, including intentional line endings. |

## Main project directories

| Directory | Purpose |
|---|---|
| `src/robot_navigation/` | Working Python implementation. |
| `app/` | Streamlit application entry point. |
| `configs/` | Environment and experiment settings. |
| `data/` | Versioned map and scenario manifests. |
| `scripts/` | Study execution, analysis and reproduction commands. |
| `tests/` | Automated correctness checks. |
| `artifacts/` | Saved models, results, snapshots, plots and replays. |
| `docs/` | Protocols, explanations, verification and reports. |
| `.venv/` | Installed Python environment. |
| `.cache/` | Local working/reproduction directories. |
| `.pytest_cache/`, `__pycache__/` | Test/interpreter caches. |

## Configuration and data filenames

| Filename | Purpose |
|---|---|
| `configs/environment.yaml` | Rewards, action cap, versions and split settings. |
| `configs/smoke.yaml` | Initial short Q-learning run. |
| `configs/tabular_milestone.yaml` | Preliminary tabular comparison. |
| `configs/part_a.yaml` | Main Q-learning/SARSA/DQN comparison. |
| `configs/dqn_smoke.yaml` | DQN pilot. |
| `configs/part_b.yaml` | Search candidates, seeds, targets and limits. |
| `data/maps/bottleneck-v1.json` | Actual obstacle layout. |
| `data/maps/bottleneck-v1.png` | Image of the layout. |
| `data/scenarios/train.json` | Training start/goal pairs. |
| `data/scenarios/validation.json` | Development evaluation pairs. |
| `data/scenarios/monitoring.json` | Small validation subset used during training. |
| `data/scenarios/test.json` | Held-out final-evaluation pairs. |
| `data/scenarios/benchmark_report.json` | Counts, connectivity, distance distributions and hashes. |

The final test has now been evaluated. It cannot provide a fresh opportunity to
select or tune models; later changes need a new holdout or exploratory labeling.

## Working Python filenames

All files below, except the last row, are in `src/robot_navigation/`.

| File | Responsibility |
|---|---|
| `core.py` | Map, task identity and environment transitions. |
| `benchmark.py` | Scenario preparation and split loading. |
| `agents.py` | Random, BFS, Q-learning and SARSA policies. |
| `training.py` | Tabular training loop. |
| `part_a.py` | Main experiment recording and checkpoint selection. |
| `dqn.py` | Neural policy and Stable-Baselines3 integration. |
| `refinement.py` | Reward variants and search-selection helpers. |
| `voting.py` | Three-policy voting. |
| `episodes.py` | Episode records, metrics and playback state. |
| `workflow.py` | Shared evaluation workflow. |
| `visualization.py` | Map and trajectory drawing. |
| `dashboard.py` | Evidence and matched-comparison helpers. |
| `dashboard_ui.py` | Dashboard views and comparison panels. |
| `release.py` | Final freeze and integrity verification. |
| `cli.py` | Command-line entry point. |
| `__init__.py` | Identifies the Python package. |
| `app/streamlit_app.py` | Starts and wires together the interactive application. |

## Script filenames

| Script | Purpose |
|---|---|
| `run_tabular_milestone.py` | Runs the preliminary tabular comparison. |
| `run_part_a.py` | Runs the main Part A comparison. |
| `analyze_part_a.py` | Audits/analyzes Part A and produces its shortlist/figures. |
| `run_part_b.py` | Runs search, paired controls/confirmations, reward experiments and voting. |
| `analyze_part_b.py` | Audits/analyzes Part B. |
| `analyze_core_robustness.py` | Analyzes predefined fixed-map scenario groups. |
| `run_final_evaluation.py` | Freezes inputs and runs the digest-gated final evaluation/audit. |
| `verify_artifacts.py` | Audits earlier saved milestone evidence. |
| `verify_reproduction.py` | Reprepares/retrains in isolation and checks repeatability. |
| `replay_frozen_release.py` | Restores/verifies the exact frozen archive; optional unchanged evaluation replay. |

## Test filenames

| Test file | Checks |
|---|---|
| `test_core.py` | Environment rules, rewards, endings, metrics and small fixtures. |
| `test_benchmark.py` | Deterministic preparation, disjoint splits and goal coverage. |
| `test_tabular.py` | Tabular learning/training behavior. |
| `test_dqn.py` | DQN integration, encoding, replay/bootstrap and inference behavior. |
| `test_part_a.py` | Main-study recording and checkpoint-selection behavior. |
| `test_refinement.py` | Reward variants, candidate selection and voting behavior. |
| `test_viewer.py` | Replay/custom-demo controls and model compatibility. |
| `test_dashboard.py` | Dashboard pages, robustness groups and synchronized comparisons. |
| `test_release.py` | Final-mode guards, freeze integrity and synthetic release pipeline. |

## Documentation and setup filenames

| Filename or pattern | Meaning |
|---|---|
| `milestoneN_protocol.md` | What was planned before running that stage. |
| `milestoneN_results.md` | What actually happened. |
| `milestoneN_verification.json` | Checks and outcomes. |
| `milestoneN_preservation.json` | Evidence protected files stayed unchanged. |
| `milestoneN_ui_check.json` | UI observations and limitations. |
| `milestone6_reproduction.json` | Recorded isolated reproduction result. |
| `milestone_results.md`, `verification.json`, `preservation_check.json` | Earlier milestone-1 evidence using the original unnumbered naming. |
| `technical_report.md` | Complete research explanation. |
| `development_log.md` | Decisions and work performed. |
| `environment_specification.md` | Exact task rules. |
| `release_checklist.md` | Release acceptance evidence and limits. |
| `README.md` | Setup and usage instructions. |
| `pyproject.toml` | Python package and dependency declarations. |
| `requirements-lock.txt` | Exact tested dependency versions. |
| `.gitignore` | Generated/local files Git should leave out. |
| `.gitattributes` | File-handling and line-ending rules. |

## What to open or edit

For normal use, start with [the README](../README.md),
[the app](../app/streamlit_app.py), [working source](../src/robot_navigation/)
and [the selected Q-learning run](../artifacts/runs/m4-tuned-q-learning-seed101/).

Edit working code in `src/` and `app/` when making an authorized change. Keep
archived `source/` directories, frozen snapshots, benchmark manifests and prior
results as historical evidence. A changed configuration should create a new run
directory rather than overwrite an existing experiment.

Guide saved on 7 November 2026. For measured findings, see
[the technical report](technical_report.md).
