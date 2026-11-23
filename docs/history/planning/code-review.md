> Unified Demo update: choose the original map or any of the four structured maps under **Demo > Navigation map**. All maps share Trained replay, Learning replay, Compare agents and Try a route tabs; models are matched to their own map/reward identity. Structured maps show saved monitoring curves where training-action recordings were not retained.

> Navigation update, 17 November 2026: use **Demo**, **Results**, **Methods**. Results groups **Fixed layout** (with Results view) and **Unseen layouts**. The sidebar instructions and defect list below record the 16 November review; fixes and current verification are in [remediation status](autonomous-robot-navigation-rl/docs/remediation_status.md) and the [evidence index](autonomous-robot-navigation-rl/docs/evidence_index.md).

# Autonomous Robot Navigation: Code Verification and User Guide

Reviewed on **16 November 2026**, using the code, installed environment and saved artifacts in this workspace.

## 1. What this project does

This is a local Python reinforcement-learning research application. A simulated robot moves through a grid, avoids blocked cells and tries to reach a goal. The project trains and compares navigation algorithms, saves their decisions and results, and displays them in a Streamlit dashboard.

There are **two separate experiment tracks**:

| Track | Question it investigates | Where to explore it |
|---|---|---|
| Original fixed-map study | Can a policy learn routes between different starts and goals on one familiar obstacle map? | **Live map**, Overview, Learning analysis, Compare results, Optimization and robustness |
| Later unseen-map study | Can a policy use map information to navigate layouts excluded from training? | **Unseen maps**, with separate pilot, shared-feature and final-evaluation entries |

The default fixed-map model is **refined tabular Q-learning**, run `m4-tuned-q-learning-seed101`. The later unseen-map study evaluates **Q-learning, SARSA, DQN and PPO**. PPO has the highest observed unseen-map final mean success, but that does not replace the original demo's chosen controller.

This application uses simulated, discrete movements. It has no physical robot driver, camera, LiDAR, ROS integration or live sensor feed. The sidebar label **Live map** opens an interactive replay workspace. It does not mean a robot is learning or moving in the real world. No OpenAI API key or cloud service is needed.

## 2. Verification results

### Checks executed for this review

| Check | Actual result |
|---|---|
| Full regression suite: `.venv\Scripts\python.exe -m pytest -q` | **202 passed in 113.42 seconds**; no failures or skips reported |
| Gymnasium environment checker: `python -m robot_navigation.cli check` | Passed |
| Installed dependency consistency: `python -m pip check` | Passed: no broken requirements |
| Parse and compile current Python files in `src`, `app`, `scripts` and `tests` | **88 files passed** |
| Import package modules | **41 modules imported** successfully, in addition to the package itself |
| Parse current YAML configurations | **14 configurations passed** |
| Completed fixed-map run inventory | **140 complete runs**, with **270 selected/final checkpoint loads** passing integrity and compatibility checks |
| Saved fixed-map example episodes | **412 replays** validated against environment transitions and metrics |
| Default refined Q-learning policy | All **928 validation episodes / 23,835 actions** reconciled with the reloaded policy's actions, Q scores and saved aggregate metrics; its Q table stayed unchanged |
| Original fixed-map final archive | Seal and all **184 frozen input files** passed integrity checks |
| Unseen-map benchmark | **12 training, 4 validation and 10 final layouts** verified; connectivity, route lengths, split hashes and rotation/reflection uniqueness passed |
| Unseen-map final evaluation | All **16 evaluation runs / 7,680 episodes / 671,151 saved actions** passed re-audit; final aggregate statistics and layout intervals reconciled; all **114 sealed input files** passed |
| Saved failure diagnosis | Recomputed **8,064 learned-policy episodes / 908,298 actions** and checked **449 source inputs**; episode records, groups, difficulty summaries and illustrative replay references matched |

The final-evaluation audit checked learned actions and diagnostics against the reloaded frozen policies, verified unchanged inference fingerprints, and checked BFS optimality and the registered Random action streams. This reused stored episodes; it added no training and did not generate a new scientific test cohort.

The inventory also contains **one failed and three interrupted historical fixed-map runs**. These are retained experiment attempts, separate from the 140 completed runs. They were not counted as newly successful models.

The automated suite covers all eight dashboard pages, tab and page changes, play/pause/step/reset/seek behavior, synchronized comparisons, all four algorithms, timeout bootstrapping, checkpoint tampering, scenario isolation, training recordings and diagnostic-event navigation.

### Verification boundaries

- These checks provide project-wide executable verification. They are not a proof that every possible input or code branch is correct; line/branch coverage was not measured.
- Current source and entry points were inspected. Historical source copies were checked through archive seals and existing provenance checks; every old experiment was not retrained.
- The full selected/final checkpoint inventory above concerns `artifacts/runs`. It does not claim that every historical periodic checkpoint was loaded during this review.
- A local Streamlit server started successfully on `127.0.0.1:8527`. A fresh native browser inspection could not be completed: Computer Use stopped because it could not determine Chrome's current URL confidently. Current UI behavior was verified through Streamlit AppTest, not a new visual desktop/mobile inspection.
- This review used Windows and **Python 3.14.10**. Other Python versions, operating systems, GPU execution, physical robots and a fresh dependency installation were not tested here.
- Published experiment files and frozen archives were not rewritten. A read-only review helper and machine-readable result are stored locally under `autonomous-robot-navigation-rl/.cache/code-review-20261116/`; `.cache` is ignored by Git.

Installed versions checked: NumPy 2.5.4, Gymnasium 1.4.0, PyYAML 6.0.3, Matplotlib 3.11.2, Streamlit 1.67.0, Stable-Baselines3 2.9.2, PyTorch 2.15.0 and pytest 9.2.0.

### Findings and remaining limitations

**No failing regression, checkpoint-integrity or saved-transition checks were found in the executed scope.** The main issues to understand are:

| Finding | Why it matters | How to interpret it |
|---|---|---|
| Learned unseen-map performance misses the declared success/SPL targets | A functioning implementation can still learn an inadequate controller | Report the measured failures; do not describe these policies as deployment ready |
| Deterministic policies frequently repeat cycles | The UI warning does not alter actions or recover a route | A loop continues until the 200-action limit unless another terminal event occurs |
| The Methodology page contains historical fixed-map wording | Its footer still says unseen maps are deferred, despite the completed Unseen maps section | Read that page as the original fixed-map methodology; use the M10–M13 documents for the later track |
| The workspace-root `README.md` is an older copy | Some instructions refer to the former Mission setup layout, and links such as `docs/...` resolve from the wrong directory at the workspace root | Use this guide and the [application README](autonomous-robot-navigation-rl/README.md) |
| Saved final cohorts have already been inspected | Reusing them for tuning would weaken an independent performance claim | A future improvement needs a new holdout or explicit exploratory labeling |
| Checkpoints are intended for inference | Full RNG, replay/rollout buffers and environment state are not preserved for exact continuation | Loading a model supports demonstrations and evaluation, not exact training resume |

This review adds documentation; it does not change the algorithms or repair the historical wording listed above.

## 3. How the navigation environment works

### Map, state and actions

The original map is `bottleneck-v1`, a **10 × 10** grid with **79 free cells and 21 obstacles**. All free cells are connected. Coordinates use **[row, column]**, from 0 to 9. Row increases downward; column increases to the right.

The original observation is:

```text
[robot_row, robot_column, goal_row, goal_column]
```

| Action number | Name | Coordinate change |
|---|---|---|
| 0 | Up | row − 1 |
| 1 | Down | row + 1 |
| 2 | Left | column − 1 |
| 3 | Right | column + 1 |

There is no diagonal move or wait action. The original learned policies receive coordinates only. They can learn values associated with the familiar map, but do not receive its obstacle grid as an observation. BFS explicitly receives the map.

### A single environment step

`NavigationEnv.step()` in [core.py](autonomous-robot-navigation-rl/src/robot_navigation/core.py) performs this sequence:

1. Reject a step if the episode already ended, or if the action is invalid.
2. Calculate the attempted neighboring cell.
3. Move into it if it is free. For a wall/boundary collision, keep the robot in its previous valid cell.
4. Increment the action count and calculate the reward.
5. Return the next observation, reward, terminal flags and outcome information.

| Event | Reward for that action | Episode behavior |
|---|---:|---|
| Ordinary valid move | −0.01 | Continue |
| Reach the goal | +9.99, consisting of +10 − 0.01 | Terminate with success |
| Attempt a wall or out-of-bounds cell | −5.01, consisting of −5 − 0.01 | Terminate with collision; position stays unchanged |
| Reach 200 actions without success/collision | Ordinary −0.01 on the last action | Truncate with timeout; no extra timeout penalty |

Success and collision take priority if they occur on action 200. For example, a successful 10-action route earns `10 − 0.01 × 10 = 9.90`. A 200-action timeout earns approximately −2.00 under the original reward.

A separate **terminal-only reward ablation** sets the step cost to zero while retaining the goal/collision bonuses. Its task version is different, so its checkpoints and returns must be interpreted under that reward variant.

### Training and evaluation data

There are `79 × 78 = 6,162` distinct ordered start–goal pairs on the fixed map, excluding identical start and goal cells.

| Fixed-map manifest | Pairs | Purpose |
|---|---:|---|
| Training | 4,306 | Sample episode resets during learning |
| Validation | 928 | Development comparison and reporting |
| Monitoring | 30 | Fixed subset of validation used for periodic checkpoint selection |
| Final test | 928 | Original frozen final evaluation |

Pairs are generated deterministically and stratified by goal and BFS distance: short 1–5 actions, medium 6–10, long 11 or more. Monitoring is part of validation, not a fourth disjoint split.

Held-out **starting pairs on the same map** do not guarantee unseen states: a training trajectory can pass through a position/goal combination later used as an evaluation start. This is why the separate unseen-layout benchmark matters.

## 4. How the models learn, and why they were chosen

### Why reinforcement learning fits this task

Each movement changes the next state and the reward that can be collected later. The agent must learn a sequence of decisions, including detours around obstacles. Reinforcement learning directly models that interaction: **observe → choose an action → receive a reward and next state → update the policy/value estimate**.

The project compares value tables, value networks and a policy-gradient method to investigate their tradeoffs. For the narrow engineering problem of shortest paths on these fully known, static grids, **BFS is already a stronger practical reference**: it solves the registered routes optimally without training. RL is used here to study learned navigation, not because the experiments show it beating BFS.

### Q-learning: the original selected controller

Q-learning stores an estimated future discounted return for every state/action combination. On this grid its array has shape `(10, 10, 10, 10, 4)`: 40,000 float64 entries, or 320,000 bytes for the uncompressed array. Some allocated entries correspond to blocked or otherwise unused combinations.

During training it uses **epsilon-greedy exploration**: with probability epsilon choose a random action; otherwise choose a highest-valued action, breaking training ties randomly. During evaluation, `argmax` breaks ties in the fixed order Up, Down, Left, Right.

The update is:

```text
Q(state, action) ← Q(state, action)
                   + learning_rate × [target − Q(state, action)]

target = reward                                      on true termination
target = reward + discount × max Q(next_state, ·)     otherwise
```

Timeouts bootstrap from the last observation; they do not use a newly reset episode's observation. Goal/collision termination removes the future-value term.

Q-learning was selected for the original track because:

- A finite, small coordinate state space makes a table straightforward to store, inspect and query.
- The original five-seed Part A validation comparison measured **65.24%** mean success for Q-learning, versus **21.94% SARSA** and **5.32% DQN**, under the registered baseline configurations.
- Bounded tuning increased fresh-seed Q-learning validation success from **64.74% to 89.96%**, compared with **24.55% to 33.00%** for SARSA. The refined Q-learning mean met the declared success, collision and SPL targets.
- The selected demonstration seed **101 was predeclared**, rather than chosen retrospectively as the best final-test seed.

The chosen configuration is learning rate **0.9**, discount **0.99**, epsilon **0.7 → 0.03** over the first **30% of 200,000 training actions**, with monitoring every **10,000** actions. This run's selected checkpoint is at **200,000** actions.

Its validation outcome is **842/928 successes**, 2 collisions and 84 timeouts. Its frozen final outcome is **840/928 successes (90.52%)**, 7 collisions and 81 timeouts, with **SPL 0.867870**. The separately reported **90.43%** is the final mean across five refined Q-learning seeds; it is not the single default model's percentage.

Sources: [Part A results](autonomous-robot-navigation-rl/docs/milestone3_results.md), [refinement results](autonomous-robot-navigation-rl/docs/milestone4_results.md), [selected checkpoint metadata](autonomous-robot-navigation-rl/artifacts/runs/m4-tuned-q-learning-seed101/selected.json), [fixed-map final results](autonomous-robot-navigation-rl/docs/milestone6_results.md).

### Other implemented approaches

| Approach | How it works here | Why it is included / what to watch |
|---|---|---|
| SARSA | Uses the value of the next action actually selected by the exploration policy in its TD target | Compares on-policy learning with Q-learning's maximum-next-value target. Its theoretical behavior does not guarantee fewer collisions or better measured performance |
| DQN | An SB3 neural Q-value model with a replay buffer, target network and epsilon-greedy exploration | Tests function approximation. Original inputs are four normalized coordinates; a separate ablation adds two relative goal offsets. Poor baseline results do not establish that all DQN designs are unsuitable |
| PPO | An SB3 actor/value model trained on rollout batches with clipped policy updates | Adds a policy-gradient comparison. It outputs categorical action probabilities and a state-value estimate; evaluation chooses the highest-probability action |
| Frozen voting | Three tuned tabular policies propose actions; majority wins, or the first member wins a three-way tie | Tests combining independently trained policies. Uses seeds 101, 211 and 307, without Q-value averaging or additional voting training |
| BFS | Computes shortest-path distances on the known obstacle map | Deterministic planner/correctness reference; no learned checkpoint |
| Random | Chooses among four actions using a seeded RNG | Untrained reference showing task difficulty |

Voting was not promoted based on later final results. A higher score observed after freezing a model does not authorize changing the predeclared choice.

For ordinary monitored runs, checkpoints are selected by **highest monitoring success, then highest SPL, then earliest training step**. `selected.npz` / `selected.zip` hold that inference model; `checkpoint.*` can instead hold the final-budget model. They can differ. Legacy pilots have their own documented selection rules.

### The unseen-map representation is different

The later benchmark contains **12 training layouts, 4 validation layouts and 10 final layouts**, each with **48 routes**: 16 short, 16 medium and 16 long. Layouts are connected 10 × 10 grids with 21 obstacles. Duplicate layouts, including rotated/reflected equivalents and the original map, are excluded.

The raw map-aware observation has **104 values**: four coordinates normalized by 9, followed by 100 obstacle indicators. The initial M10 tabular pilot kept separate tables per training map. An unseen map has no learned table, so its all-zero scores select Up by the deterministic tie rule. That limitation is represented explicitly in the implementation.

M11 changes to **1,234 shared map-and-goal features** so behavior can transfer between layouts:

- Bias and two relative goal offsets: 3 features.
- Four directional clearances: 4 features.
- Relative goal location on a 19 × 19 window: 361 features.
- Goal direction sector combined with the four-neighbor blockage pattern: 144 features.
- Robot-centered obstacle and outside-grid channels: 361 + 361 features.

These are calculated directly from the supplied map and coordinates. They do not contain a BFS solution or mask the policy's actions. NumPy and Torch implementations are tested for feature parity.

For this study, **Q-learning and SARSA use shared linear value functions**, with 4 × 1,234 = 4,936 weights, rather than the original coordinate tables. DQN and PPO use the same fixed features before their neural layers. This shares input information, but does not make model capacity or compute equal.

### What the unseen-map final results mean

All twelve M11 policies—four families × three seeds—were frozen before evaluation on the ten final layouts. Each learned policy evaluated the same 480 routes.

| Method | Final mean success | Final mean SPL | Mean collision | Mean timeout |
|---|---:|---:|---:|---:|
| Shared linear Q-learning | 16.81% | 0.1645 | 0.07% | 83.12% |
| Shared linear SARSA | 62.08% | 0.6127 | 1.67% | 36.25% |
| Shared-feature DQN | 29.10% | 0.2834 | 0.00% | 70.90% |
| Shared-feature PPO | 64.10% | 0.6312 | 1.94% | 33.96% |
| BFS reference | 100.00% | 1.0000 | 0.00% | 0.00% |
| Random reference | 3.40% | 0.0285 | 96.60% | 0.00% |

PPO has the highest observed learned-family mean, with SARSA close behind. Their layout confidence intervals overlap. SARSA also has lower recorded inference/model costs in this study. These results do not establish a universal winner, and **every learned family misses the mean 80% success / 0.75 SPL targets**. The low collision rates can coexist with many failed routes because the agents repeatedly cycle until timeout.

Across the M11 validation and M12 final learned-policy traces, all **4,434 timeouts** meet the persistent-cycle definition, and **96.59%** are two-cell cycles. This describes saved behavior; a successful recovery method has not been demonstrated. The replay warning supplies no planner fallback or corrective action.

Sources: [shared-feature protocol](autonomous-robot-navigation-rl/docs/milestone11_protocol.md), [unseen-map final results](autonomous-robot-navigation-rl/docs/milestone12_results.md), [failure diagnosis](autonomous-robot-navigation-rl/docs/milestone13_results.md).

## 5. How the code is organized

The implementation is under `autonomous-robot-navigation-rl/`. The parent workspace contains the original project plan and this guide.

```text
configs + maps/scenario manifests
             |
             v
Gymnasium environment <---- training actions ---- agent
             |                                    ^
             +------ next state + reward ----------+
                                                  |
                                  monitoring selects a checkpoint
                                                  |
                                                  v
                           evaluation + saved trajectories/metrics
                                                  |
                                                  v
                          Streamlit dashboard + recorded playback
```

| File or group, relative to the application directory | Responsibility |
|---|---|
| `app/streamlit_app.py` | Configures Streamlit, sidebar navigation, task loading and the four fixed-map replay tabs |
| `src/robot_navigation/core.py` | Map representation, BFS geometry, task signatures, scenario validation, environment rules and atomic JSON writes |
| `benchmark.py`, `cli.py` | Fixed-map scenario generation/loading and command-line entry points |
| `agents.py`, `dqn.py`, `ppo.py`, `ppo_config.py`, `observation_encoding.py` | Agent implementations, neural input scaling, inference, configuration checks, checkpoint loading and integrity checks |
| `training.py`, `part_a.py`, `workflow.py` | Training loops, monitoring, checkpoint selection, evaluation, timing and run provenance |
| `refinement.py`, `voting.py` | Reward variants, bounded candidate generation/ranking and frozen policy voting |
| `training_traces.py`, `learning_replays.py` | Capture bounded actual training episodes and audit their chronological recording windows |
| `representation.py`, `neural_tuning.py`, `neural_confirmation.py` | Input ablation, neural search and staged confirmation evidence/statistics |
| `four_way.py`, `four_way_analysis.py` | Four-algorithm comparisons, route/cost summaries and geometry-group analysis |
| `episodes.py` | Rollouts, metrics, replay validation and single/comparison playback state |
| `dashboard.py`, `dashboard_ui.py`, `simulator_ui.py`, `interface.py`, `theme.py`, `visualization.py`, `neural_tuning_ui.py` | Evidence loading, controls, maps, telemetry, charts, styling and UI rendering |
| `release.py` | Original final-model roster, frozen inputs, archive verification and final-only access |
| `unseen_benchmark.py`, `unseen_agents.py`, `unseen_study.py`, `unseen_main.py` | Layout benchmark, map-aware policies and bounded pilot/shared-feature training studies |
| `shared_features.py`, `shared_neural.py`, `shared_agents.py` | Common feature engineering, Torch feature extractor and shared linear value models |
| `unseen_audit.py`, `unseen_final.py` | Map-aware saved-action audits and sealed independent final evaluation |
| `unseen_diagnostics.py`, `unseen_diagnostics_ui.py`, `unseen_ui.py` | Failure classification, diagnostic evidence, unseen-map results and replay navigation |
| `__init__.py` | Python package marker |
| `scripts/` | Experiment, analysis, reproduction and release-verification entry points |
| `configs/`, `data/` | YAML experiment settings, maps, versioned scenario manifests and benchmark metadata |
| `artifacts/runs/` | Fixed-map run configurations, checkpoints, logs and replays |
| `artifacts/studies/` | Study summaries, analyses, unseen-map runs and frozen releases |
| `tests/`, `docs/` | Automated checks, registered protocols, measured results and prior verification records |

### What happens when the app opens

1. The entry point applies the theme and builds sidebar navigation. The sidebar starts collapsed.
2. The cached fixed-map task and validation scenarios are loaded. Missing/incompatible benchmark data produces an error.
3. Selecting **Live map** opens the four replay tabs; other pages dispatch to their evidence renderers.
4. Run selection uses saved manifests and prioritizes the predeclared Q-learning controller.
5. **Trained replay** validates a stored episode. **Learning replay** loads a declared, checksummed training recording. **Try a route** and matched comparisons run a frozen policy once to create an in-memory demonstration.
6. Playback changes an action index through stored transitions. Maps, reward charts and telemetry derive from that displayed prefix. Ordinary playback ticks do not train the model.
7. Page/tab changes pause active players. Comparison agents share one action clock, with completed agents held at their terminal frame.

Task signatures include map identity/hash and environment, observation, action and reward versions. Checkpoint hashes, finite-value/shape checks and strict loaders catch common incompatibilities. Frozen releases pin source, data and model bytes so later UI changes do not silently alter the recorded experiment.

## 6. Start the application on Windows

From PowerShell at the workspace root:

```powershell
Set-Location .\autonomous-robot-navigation-rl
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8517 --server.headless true --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8517** in your browser. Keep the terminal running. Stop the server with **Ctrl+C** in that terminal. If the port is occupied, select another port and open its matching URL. The review used port 8527; 8517 is the normal documented example.

If the virtual environment is absent, run the following from the application directory before starting Streamlit:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation
```

The existing environment was verified for this review; these installation commands were not rerun. The project declares Python 3.11+, while this review ran on 3.14.10.

## 7. Navigate the UI

### Sidebar and research pages

Expand the sidebar using Streamlit's upper-left sidebar control if it is hidden. The main buttons are:

| Sidebar label | What opens | Use it for |
|---|---|---|
| **Live map** | Navigation simulator | Fixed-map trained/learning replays, comparisons and custom routes |
| **Unseen maps** | Unseen maps | Multi-layout studies, final results and failure analysis |
| **Overview** | Overview | Original fixed-map summary and frozen final results |
| **Learning analysis** | Learning analysis | Saved monitoring curves, learning systems and seed-level summaries |
| **Compare results** | Agent comparison | Aggregate results, route lengths, costs and the original frozen final table |
| **Neural tuning** | Neural tuning | Switch between original three-seed results and the five-seed extension; compare baseline and tuned neural runs |
| **Methodology** | Methodology | Original fixed-map rules, metric definitions and task signature; note the historical wording described above |

Expand **Research library** to use **Dashboard view**, which lists all eight pages. **Optimization and robustness** is available there. It contains input-ablation results, matched refinement, reward/voting comparisons and scenario-group tables. It has no dedicated sidebar shortcut.

### A. Watch the default trained policy

1. Open **Live map → Trained replay**.
2. Under **Trained model**, keep the Q-learning entry marked **Selected system**. Its run ID is `m4-tuned-q-learning-seed101`.
3. Under **Recorded episode**, choose the saved Q-learning success example. Other entries may show collisions, timeouts or reference methods.
4. Press **Next step** to inspect one action, or **Play** to animate the recording.
5. Use **Pause**, **Reset**, **Speed** and the **Replay position** slider as needed. Seeking pauses playback. At the terminal frame, reset before playing again.

The model selector's **Filter** popover narrows runs by **Experiment** and **Algorithm**. If no runs match, broaden those filters.

Common controls:

| Control | Effect |
|---|---|
| Play | Advance automatically through recorded actions |
| Pause | Stop automatic advancement |
| Next step | Advance by one action |
| Reset | Return to the initial frame and pause |
| Speed | Choose 0.5, 1, 2 or 4 steps/second; this affects presentation only |
| Replay position | Jump to a recorded action index |
| Step-by-step history → Seek displayed event | While paused, jump to an event already included in the displayed history |

Open **Map display options** for **Traveled trail**, **Prefix visit counts** and **BFS planner reference**. The BFS overlay is an explicitly labeled optimal-planner route, not a learned policy's future decisions.

Open **Model and recording details** for the scenario, replay seed, run ID, reward version, checkpoint and selected interaction. **Agent state → Action details** shows available recorded outputs for the last displayed action. Those values refer to the action's **before-state**.

### B. Understand the map and telemetry

- **R**, **S** and **G** identify robot, start and goal. Blocked cells are obstacles.
- **Actions** counts attempted actions, including a collision attempt.
- **Distance traveled** counts successful cell movements. A collision attempt adds no travel.
- **Goal distance** is Manhattan distance; walls can make the actual shortest route longer.
- **Reward so far** and the reward chart include only displayed actions.
- **Revisited cells** reflects repeated positions in the visible trail. A repeat alone does not end an episode.
- **Completed SPL** appears at a completed terminal frame. It is not a live confidence score.
- A repeating-pattern warning describes observed actions. Playback does not intervene.

Q-learning/SARSA/DQN show **Q scores**, which are not probabilities. PPO shows **action probabilities** and a **state value**. Voting shows member proposals/ties. Training recordings intentionally have no invented checkpoint decision scores.

### C. Watch recorded learning

1. Choose **Live map → Learning replay**.
2. Select a **Training run**, optionally filtering by algorithm/experiment.
3. Select a **Training episode**, checking its training-stage/start-step label.
4. Play or step through the recording.

These are actual actions captured during training, including exploration and policy updates. Pressing Play does not start training. Only bounded chronological samples were retained; older histories cannot be reconstructed from a final checkpoint. A recording that ends at a training-budget cutoff can be incomplete and has no completed success/collision/timeout metric.

### D. Compare agents on the same route

1. Choose **Live map → Compare agents**.
2. Under **What would you like to compare?**, select **Two agents** or **Four algorithms**.
3. For two agents, choose **First trained model**, **Different models**, a **Second trained model**, and the common **Start and goal**.
4. Alternatively select **Earlier vs trained model** to compare the same run's archived early checkpoint with its selected checkpoint. Use an available exemplar such as `m4-tuned-q-learning-seed101`.
5. For four algorithms, choose a **Training seed** and **Start and goal**. This lineup uses the original Q-learning, SARSA and DQN baselines plus the additional PPO baseline, not four tuned models.
6. Use the shared playback controls. Each panel represents the same action index; an agent that finishes early stays at its terminal frame.

Not every run has an archived early model. Incompatible task/reward versions are rejected. Individual demonstrations and visual comparisons do not change aggregate benchmark results.

### E. Try your own route

1. Choose **Live map → Try a route** and select a **Trained model**.
2. Enter **Start row**, **Start column**, **Goal row** and **Goal column**. For a simple valid example, use start **[0, 0]** and goal **[0, 9]** on the open top row.
3. Choose the **Frozen policy**: the selected learned algorithm, **BFS**, or **Random (untrained)**.
4. Set **Demonstration seed** if needed. It controls Random's choices; it does not retrain a learned policy.
5. Click **Generate demonstration**, then play or step through the generated episode.

Start and goal must be distinct, free and reachable cells. This form uses the original map; it is not a map editor or a route-entry form for the unseen-map studies. A generated route stays outside measured benchmark summaries.

### F. Explore unseen-map results and replays

1. Click **Unseen maps**.
2. Choose **Unseen-map study**:
   - **Final evaluation · ten held-out layouts**: frozen M12 final results; this is the default when verified evidence is available.
   - **Shared features · three seeds**: M11 development validation and recorded learning.
   - **Initial pilot · one seed**: M10 pilot results and recordings.
3. In **Results**, inspect outcomes, seed variation, per-layout tables, reference methods and recorded costs. Read the study label before comparing numbers.
4. In **Replay**, choose **Map-aware algorithm**, **Training seed** when available, and **Map recording**.
5. For the final study, **Recording type** is restricted to **Final policy**. For development studies, choose **Trained policy** or **Recorded learning**.
6. Use the usual playback controls and **Map display options**. **Run and policy information** identifies the layout, scenario and policy context.

Unseen-map playback reads registered saved actions. It does not query a policy to invent historical learning. The final viewer uses the sealed final roster and recordings; it does not run new final scenarios interactively.

### G. Inspect why unseen-map policies failed

1. In **Unseen maps**, select the final or shared-feature study, then **Failure analysis**. The initial pilot has no corresponding diagnosis view.
2. Read the failure-composition chart and counts. Expand **Route difficulty and seed variation** for distance-band outcomes.
3. Choose **Diagnostic algorithm** and **Failure map** to inspect the failure-location heatmap.
4. Choose an **Illustrative recording**, read its own map/seed label, and press **Replay this event**.
5. The UI opens Replay, selects that policy/recording and pauses at the first cycle return or terminal collision. Step around the event to inspect it.

The heatmap counts one location per failed case: cycle entry, last valid cell before collision, or final cell for another timeout. Illustrative recordings have their own source-map labels and need not match the currently selected heatmap. These are diagnostic examples, not a random sample or a tested recovery intervention.

## 8. Read results correctly

| Metric or label | Meaning |
|---|---|
| Success rate | Fraction of completed episodes that reached the goal |
| Collision / timeout rates | Mutually exclusive failure outcomes; together with success they sum to 1 |
| SPL | For success: shortest route length divided by `max(shortest route length, actions taken)`; failures contribute 0. Average across all episodes |
| Successful route length | Average actions among successful episodes only; read it with success counts |
| Seed SD | Sample standard deviation across training seeds; not a confidence interval |
| Layout interval | M12 bootstrap interval resampling layout means, conditional on the fixed trained-policy roster |
| Training seed | Identifies an independently trained run |
| Replay/evaluation seed | Controls an episode/evaluation RNG context; it is not another trained model |
| Selected interaction | Training age of the monitored checkpoint selected by the registered rule |

Some tables display rates as fractions: **0.90 means 90%**. Other tables explicitly show percentages. A high return under one reward version should not be ranked directly against a different reward scale. A short mean successful route can simply mean the policy only solved easy cases.

## 9. Common problems and repeatable checks

| Symptom | What to do |
|---|---|
| Browser cannot open the local URL | Check that Streamlit is still running and that the URL port matches the command |
| Sidebar is missing | Expand the upper-left sidebar control; initial state is collapsed |
| Model list is empty after filtering | Open Filter and choose broader Experiment/Algorithm values |
| No learning recording or early checkpoint | Choose a run with retained recordings/checkpoints; the application cannot reconstruct missing history |
| Playback will not restart at the end | Reset, then Play |
| Robot remains stuck in a cycle | Inspect a failure recording or choose another route/model; playback does not repair the controller |
| Missing evidence or checksum/task-version mismatch | Check the referenced artifacts and matching task/configuration; do not edit a checksum to make incompatible files load |
| Source changes are not reflected in the browser | Restart the Streamlit server so Python imports are refreshed |
| Deep audit fails after a fresh Git checkout | Some bulky trajectories and periodic checkpoints are ignored by Git and may exist only in this local workspace; inspect the missing-file error |

From the application directory, rerun the ordinary verification checks with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m robot_navigation.cli check
.\.venv\Scripts\python.exe -m pip check
```

The additional local read-only audit used for this review can be repeated while its helper and the full local artifacts are present:

```powershell
.\.venv\Scripts\python.exe .cache/code-review-20261116/verify_read_only.py
```

Experiment runners and some historical `verify` scripts can publish reports or regenerate figures. They are not necessary just to browse saved evidence. Retraining is separate from starting the dashboard.

For deeper context, start with the [technical report](autonomous-robot-navigation-rl/docs/technical_report.md), [file/experiment naming guide](autonomous-robot-navigation-rl/docs/file_and_experiment_guide.md), [original project plan](Autonomous_Robot_Navigation_Codex_Project_Plan.md), and [evaluation boundary handoff](autonomous-robot-navigation-rl/docs/evaluation_boundary_and_handoff.md).
