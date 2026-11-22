# Autonomous Robot Navigation Using Reinforcement Learning

## Private Project - Codex Development Brief

**Project title:** Autonomous Robot Navigation Using Reinforcement Learning: Comparative Evaluation and Optimization in a Simulated Environment  
**Owner:** Individual personal project  
**Status:** Six core milestones completed; fourth-algorithm PPO extension trained and verified. Current integration checks and remaining gaps are recorded in `autonomous-robot-navigation-rl/docs/plan_closure_status.md`.
**Planning date:** 6 November 2026  
**Development period:** Suggested six-week first release, adjustable to personal availability  
**Language:** Python  
**Application type:** Reinforcement learning research prototype with an interactive navigation demonstration

**Confirmed scope:** The core plan includes the animated navigation UI. PPO was subsequently authorized as the separate extension below. Unseen-map generalization, moving obstacles, continuous movement and physical robots remain deferred. Retain their descriptions as future reference only; do not implement or schedule them without a new request from the owner.

**Owner extension, 8 November 2026:** Add PPO as a fourth trained RL algorithm
for comparison. This supersedes PPO's deferred status for this extension only.
Use a separate post-release development-validation study and preserve the
original three-family results and frozen final evaluation. Other deferred
additions remain deferred. The new protocol is
`autonomous-robot-navigation-rl/docs/ppo_comparison_protocol.md`.

## 1. Project Information and Institutional Guidance

This project uses the same development structure as the Smart Energy project: define the problem, prepare reproducible inputs, compare learning approaches in Part A, refine the strongest approaches in Part B, and produce an inspectable prototype and technical report.

The institution's BMDS2114 assignment specification explicitly includes defining an environment, exploring its dynamics, preparing state representations, and training agents. This supports using reinforcement learning rather than forcing the project into supervised prediction.

### Requirements translated into this project

| Institution's guidance | Robot navigation implementation | Evidence to produce |
|---|---|---|
| Frame a problem and define data/environment | Specify simulated robot, map, actions, observations, rewards and episode endings | Environment specification and scenario manifest |
| Explore dynamics and prepare representations | Analyze reachability, shortest paths, obstacle layouts and state coverage | Environment analysis and visualizations |
| Train diverse suitable approaches | Compare Q-learning, SARSA and DQN | Configurations, checkpoints and learning curves |
| Compare and shortlist promising methods | Use common evaluation episodes and documented selection criteria | Part A comparison and justified shortlist |
| Interpret influential factors and errors | Investigate loops, collisions, detours, difficult goals and exploration | Failure taxonomy and trajectory examples |
| Refine models and learning mechanisms | Tune hyperparameters, exploration, rewards and representations | Controlled before-and-after experiments |
| Combine models | Test voting across independently trained policies | Ensemble-versus-single-policy comparison |
| Test reliability in different situations | Test held-out start/goal pairs and difficulty groups on the fixed map; defer transfer to new maps | Scenario-level and seed-level robustness results |
| Automate repetitive work | Configuration-driven training, evaluation and artifact generation | Reproduction commands and run manifests |
| Explain and present results | Technical report and working interactive simulation | Source, report, charts and demonstration |

The energy plan uses a **40% Part A / 40% Part B / 20% output and presentation** balance. Retain that as a planning guide. Those component weights were not independently confirmed from an assessment rubric here. The four-page assignment specification separately states a 40% team / 60% individual marking split; that is a different classification and does not apply to a solo private project.

### Private-project adaptations

- Replace the four-person allocation with individual workstreams and a development log.
- Treat six weeks as a suggested development cycle, not a new institutional deadline.
- Keep source, models and reports local. If a remote repository is later requested, make it private. Do not publish the dashboard or create public share links without an explicit request.
- Keep the existing energy project and synced reference files unchanged. Develop the robot project in its own directory or repository.
- No institutional submission, tutor approval, presentation date or AI disclosure is required by this private-project plan. If any work is later submitted for assessment, the institution's actual submission and AI-use rules must be reviewed then.

### Development principles

Build incrementally. Keep environment rules, agents, training, evaluation and the interface separate. Record assumptions and explain substantial choices. Save failed experiments as well as successful ones. Never invent performance figures or imply that training has occurred when it has not.

### Proposed architecture

```text
Versioned maps + start/goal scenario manifests
                    |
                    v
Gymnasium navigation environment
  Observations -> Actions -> Transitions -> Rewards -> Episode outcomes
                    ^                         |
                    |                         v
             RL training loop <--------- Experience
               Q-learning | SARSA | DQN
                    |
                    v
Part A: Common validation -> Failure analysis -> Shortlist
                    |
                    v
Part B: Tuning -> Controlled refinements -> Policy voting -> Robustness
                    |
                    v
Frozen final evaluation -> Saved metrics, trajectories and checkpoints
                    |
                    v
Local Streamlit interface + technical report

Random policy and BFS: reference baselines through the same environment
```

## 2. Project Overview

### Problem statement

A mobile robot must choose movements that take it from a starting position to a goal while avoiding obstacles and unnecessary travel. Reinforcement learning offers a way to learn this behavior through interaction and feedback.

This project investigates how different learning methods trade off navigation success, collision avoidance, route efficiency, training effort and consistency. It begins with an abstract grid-based robot so that behavior can be checked and experiments repeated before adding physical complexity.

### Objectives

1. Build a reproducible simulated navigation environment.
2. Implement classical reference baselines and several reinforcement learning agents.
3. Compare agents under the same task definition and training budget.
4. Explain failures and factors that affect learning.
5. Improve promising agents through controlled experiments.
6. Test whether combining policies improves reliability.
7. Evaluate held-out start/goal tasks on the fixed map; reserve unseen maps for a later phase.
8. Demonstrate learned navigation through an interactive local application.

### Research questions

- RQ1: Which learning approach offers the best balance of success, route efficiency and training cost on the defined navigation task?
- RQ2: How do exploration settings, reward design and state representations affect learning and failure modes?
- RQ3: Do tuned agents and policy voting improve performance over their Part A baselines?
- RQ4: How stable are results across training seeds and navigation scenarios?
- Deferred RQ5, future reference only: Can a policy trained across multiple maps navigate unseen layouts without retraining?

### Scope boundaries

The first release covers one robot, a fully observed 2D grid, static obstacles, discrete movements and a simulated point-sized robot. Real hardware, continuous motor control, camera perception, SLAM, multiple robots and moving obstacles belong to later releases. Grid navigation demonstrates decision learning; it does not establish readiness for a physical robot.

## 3. Environment, Experience Data and Evaluation Splits

### Initial task contract

| Item | Initial decision |
|---|---|
| Map | One versioned 10 x 10 grid with static obstacles |
| Connectivity | At least 30 connected free cells; all benchmark start/goal pairs reachable |
| Agent | One point-sized robot occupying one free cell |
| Actions | Up, down, left, right; no diagonal movement or wait action |
| Transition | Deterministic movement of one cell per action |
| Invalid movement | Attempt into wall or outside map; remain in place and end in collision failure |
| Goal | Reach the specified goal cell |
| Start/goal | Distinct cells sampled from the saved scenario manifest |
| Episode limit | 200 actions; reaching the limit without another terminal outcome is a timeout |
| Observation | Robot row/column and goal row/column; map remains fixed in the core task |
| Visibility | Exact position and goal; no sensor noise in the core task |
| Rendering | Optional; disabled during training and timed inference measurements |

On this fixed map, position and goal identify the navigation state because the obstacle layout and dynamics do not change. Encode the same information as a table index for tabular agents and a normalized vector for neural agents. Document encoding differences. Do not treat this observation as sufficient for changing maps.

The viewer displays the obstacle map to the human. The core learned agent receives only its position and goal, and learns the fixed obstacle layout through experience; BFS receives the map explicitly. Record this information difference. No learned agent may silently use shortest-path directions, invalid-action masking, planner fallback or UI collision prevention in the primary benchmark. Loop highlighting is diagnostic and does not end an episode before the agreed limit.

### Benchmark definition gate

Before the main experiments, save the actual 10 x 10 map, not just its dimensions. Include walls requiring detours, a corridor or bottleneck, and an open region. Record free/blocked cell counts, connected components, a map image and the shortest-path length distribution. Choose the layout for its stated navigation characteristics rather than favorable agent results. Unit-test fixtures may use smaller maps; these are debugging tools, not additional generalization benchmarks.

Every goal represented in validation/final-test scenarios must also occur as a goal in training scenarios. Freeze a split with adequate training starts per goal and representation of short/medium/long routes; save actual counts and declare distance-bin boundaries. This keeps the task about held-out initial conditions rather than accidentally testing goals absent from tabular training. Log goal frequencies, state-action visitation and successful goal discoveries during training. A table entry reached during training may correspond to a held-out starting pair; report that overlap and do not call this strict unseen-state generalization.

Run a learning pilot before freezing the benchmark. The combination of multiple goals, sparse successes and collision-ending episodes may make exploration difficult. Check updates on a hand-solvable fixture and inspect successful discoveries on the proposed map. If the pilot fails, diagnose encoding, exploration and rewards before launching the comparison. Any task adjustment creates a new version applied consistently to all agents and is recorded before main training.

### Initial reward

Use a transparent provisional reward:

```text
reward = -0.01 per attempted action
         +10.0 if the goal is reached
          -5.0 if an invalid movement causes collision failure
```

Goal and collision outcomes are mutually exclusive. A timeout receives the ordinary step reward and no additional terminal penalty. Version this reward before Part A. Numerical values are starting assumptions, not optimal settings or measured findings.

Use Gymnasium's separate episode flags: goal/collision are `terminated`; an external action limit is `truncated`. Preserve this distinction in training targets, logs and wrappers. Use a tested time-limit mechanism and verify that tabular and library agents bootstrap correctly through truncations. The exact API and seeding contract follow the [Gymnasium custom environment documentation](https://gymnasium.farama.org/introduction/create_custom_env/) and [time-limit explanation](https://farama.org/Gymnasium-Terminated-Truncated-Step-API).

### What replaces the energy dataset?

The simulator produces experience as the agent interacts with it. An external historical dataset is unnecessary for the initial online RL task.

Save maps, scenario manifests and per-episode outcomes. Retain complete trajectories for evaluation episodes and selected training episodes. Where transitions are stored, include observation, action, reward, next observation, termination, truncation, map ID, scenario ID and seed. Keep replay buffers training-only. Do not create demonstration labels from shortest paths for the RL agents unless imitation learning is introduced as a separately labeled extension.

This interaction-based data collection is a key distinction described in [Stable-Baselines3's RL guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).

### Scenario preparation

1. Validate grid dimensions and cell values.
2. Check that start and goal are distinct free cells.
3. Use BFS to confirm reachability and calculate reference shortest-path lengths.
4. Enumerate valid ordered start/goal pairs and assign stable IDs.
5. Split pairs once into 70% training, 15% validation and 15% final test, stratified by shortest-path distance bands where feasible.
6. Save exact manifests, split seed, map hash and generation code version.
7. Inspect distance distributions and scenario counts before training.

The split is by initial start/goal pair, not by individual transitions. Training paths can pass through states associated with held-out starting positions. Describe the core evaluation as **held-out initial conditions on a familiar map**, not unseen-environment generalization. No validation/test episodes may update tables, networks, replay buffers or fitted normalization statistics.

### Deferred future extension: unseen-map evaluation

This subsection is retained for future planning and is outside the current release.

For transfer experiments, generate a new versioned suite with 20 training, 5 validation and 5 test layouts as an initial target. Remove duplicate layouts across splits, retain only appropriately reachable tasks, and save actual counts. Within each layout, save fixed evaluation pairs and difficulty information.

Provide neural agents with the full obstacle grid plus robot and goal indicator channels, using an explicitly configured feature extractor. Retrain on training maps. Q-learning and SARSA remain core familiar-map baselines; a table trained on one map cannot be presented as a transferable navigation policy. Training a fresh table on each new map measures relearning, a different experiment.

Hold test maps back until the transfer method is frozen. Full visibility means that an unseen test map is visible at evaluation time; it means an unseen layout, not hidden obstacles.

## 4. Part A - Identifying and Selecting Promising Models

### Approaches

| Approach | Purpose | Priority |
|---|---|---|
| Random policy | Reference for navigation without learning | Essential reference |
| BFS shortest-path planner | Exact minimum-step reference for the static, unit-cost grid | Essential reference |
| Q-learning | Off-policy tabular learning baseline | Essential learned agent |
| SARSA | On-policy tabular comparison for exploration behavior | Essential learned agent |
| DQN | Neural value-based learning with replay and a target network | Essential learned agent |
| PPO | Additional policy-gradient comparison | Completed owner-authorized post-release baseline; separate exploratory validation |

BFS is a classical planner, not a trained ML model. It uses the known map and serves as a correctness and efficiency reference. RL is not expected to outperform an exact shortest-path planner on minimum route length in this setting. A* may be added to compare planner computation, but BFS is sufficient for the initial grid.

Use small, readable tabular implementations for Q-learning and SARSA. Prefer an established DQN implementation from Stable-Baselines3. Its [DQN documentation](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) supports the proposed discrete actions. Do not label the standard SB3 DQN implementation as Double DQN, Dueling DQN or prioritized replay. PPO is implemented under the separate owner-extension protocol; it is not part of the original frozen final evaluation.

### Common experiment protocol

- Begin with one smoke run, then compare each learned approach over five independent training seeds. Three seeds may be used for a clearly labeled preliminary study if compute is constrained.
- Proposed Part A budget: 200,000 environment interactions per seed. Validate runtime on a pilot before committing; if it changes, set and record the common budget before the main comparison.
- Evaluate every 10,000 training interactions on a fixed, distance-stratified monitoring subset of at most 50 validation scenarios, using a separate environment and no learning. Save the subset before training. Use the complete validation manifest for shortlisted checkpoint/configuration comparisons before final model selection. Validation interactions do not count toward the training budget but must be recorded separately. Record whether each result uses the monitoring subset or the complete validation manifest.
- Use greedy tabular actions and deterministic neural policy predictions for evaluation. Define tie-breaking explicitly. Exploration is for training.
- Keep map, action rules, rewards, start/goal distributions and episode limit consistent. Compare learning curves against environment interactions, and also report wall-clock time on the same machine.
- Select checkpoints using validation results only. Save all seeds, including runs that fail to learn.

Pilot both training and evaluation throughput. With 80 connected free cells, there are 80 * 79 = 6,320 ordered start/goal pairs; a 15% validation split has 948 scenarios. Evaluating all of them at 20 checkpoints with a 200-action cap could consume up to 3,792,000 evaluation interactions per run, compared with 200,000 training interactions. This is an upper-bound planning calculation, not a measured runtime. Limit routine monitoring as specified above and estimate the full study's wall-clock time before freezing the run matrix. At the proposed settings, Part A, two eight-candidate/two-seed searches, and two five-seed confirmations already total 57 training runs, before ablations or voting evaluations.

Multiple independent runs and evaluation in a separate environment follow [SB3's evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html). The specific seed counts and budgets above are project planning choices.

### Evaluation measures

| Measure | Definition and interpretation |
|---|---|
| Success rate | Goal-reaching episodes / all evaluated reachable episodes |
| Collision rate | Episodes ending in collision / all episodes |
| Timeout rate | Episodes truncated without goal/collision / all episodes |
| Successful route length | Number of actions to reach the goal, reported only for successes with the success count |
| SPL | Mean of success * shortest_length / max(shortest_length, actual_length); failed episodes contribute zero |
| Episode return | Sum of rewards; comparable only under the same reward specification |
| Learning efficiency | Validation success/SPL at fixed interaction counts and interactions to a predeclared target, if reached |
| Computational cost | Training runtime, decision latency and saved model size, with hardware and timing method |
| Stability | Distribution of results across independent training seeds |

Record mutually exclusive outcomes, giving priority to a goal/collision reached on the last permitted action over timeout. Success, collision and timeout fractions must total one for the core benchmark.

For reporting, average episode metrics within each training seed, then show mean, standard deviation and individual seed values. Do not treat hundreds of episodes from one trained policy as hundreds of independently trained agents. Any uncertainty estimates must state the resampling unit and the small number of independent runs.

### Analysis and shortlisting

Produce learning curves with seed variability, common-scenario comparison tables, trajectory overlays, collision/failure heatmaps, successful path-length distributions, and example loops or detours. Show Q-values/action preferences for tabular agents as decision evidence, without implying they establish causality.

Shortlist two learned approaches using validation success rate first, SPL second, and runtime/stability as documented tie-breakers. Keep BFS and random-policy references in the comparison. Do not pick the winner from the final test set.

**Part A deliverables:** Validated environment, saved splits, baseline references, three original trained RL approaches, run manifests, comparable validation results, failure analysis and a justified shortlist. The later PPO extension adds a fourth trained baseline for exploratory development-validation comparison while preserving Part A and its shortlist.

## 5. Part B - Refining the Model/System

### B1. Hyperparameter and exploration optimization

Tune the shortlisted approaches with a reproducible random search; add a smarter search method only if runtime justifies it. Proposed limit: eight candidate configurations per shortlisted approach, two development seeds per candidate, then five fresh training seeds for each selected configuration. Record total search cost in addition to the final training cost.

Candidate settings:

- Q-learning/SARSA: learning rate, discount factor, initial/final exploration probability and exploration decay.
- DQN: learning rate, replay capacity, batch size, warm-up steps, target update interval and exploration schedule.

Keep interaction budgets and search opportunities comparable. Select settings on validation scenarios and keep the final test sealed.

### B2. Reward and representation experiments

Compare sparse task feedback with the initial step-cost reward. Optionally test potential-based shaping with a fully documented potential, discount and boundary treatment. Avoid planner-derived dense supervision in the primary comparison.

For the fixed-map neural agent, compare normalized coordinates with a goal-relative vector that also retains enough absolute position information to identify obstacle context. A goal-relative vector alone can alias different obstacle situations. Tabular state-index renaming alone is not meaningful feature engineering.

Change one factor at a time before a combined configuration. Report success, collisions and SPL when rewards change; higher return under a new reward scale does not establish better navigation. Fit any learned normalization on training experience only and freeze it for evaluation.

### B3. Combining policies

Implement an explicit policy-voting experiment to preserve the institution's combination-method guidance:

1. Use three independently trained policies of the selected algorithm from predeclared training seeds; do not choose only the best seeds.
2. Each proposes one action from the same observation.
3. Choose the majority action. For ties, use a fixed priority among the member policies chosen before final evaluation.
4. Compare voting with every component policy and the selected single-policy system on identical validation scenarios.
5. Record the additional model memory, total training cost and decision latency.

Do not average raw Q-values from unrelated algorithms or reward scales. Voting may create loops and is not guaranteed to improve results. A well-supported negative finding still completes this experiment. A planner fallback is a separately labeled hybrid controller; it must report intervention rates and must not be described as autonomous RL success.

### B4. Robustness and generalization

Core tests include short versus long routes, narrow corridors, positions near walls, different start/goal regions, and independent training seeds. These use predefined scenario labels.

As a first extension, test blocked-action noise or a small probability of remaining in place. Keep the same observation contract, freeze the policy, record the noise seed and clearly label the shifted dynamics. Do not tune against the final robustness suite.

Unseen-map experiments are deferred. If requested in a later phase, use the separate map-aware training track in Section 3, distinguish zero-shot evaluation from adaptation with extra training, and disclose adaptation interactions. Missing sensor values, moving obstacles and partial observability require a new environment version and potentially memory; do not quietly add them to the core task.

### B5. Automation and final system selection

Automate experiment scheduling, configuration validation, metric aggregation and figure generation. Preserve Part A baselines. Compare original and refined agents using the same evaluation procedure and training budgets, and disclose the extra tuning effort.

Freeze algorithms, checkpoints, reward/observation versions, voting rules and analysis code before the final test. Run the saved final test manifest for the frozen baseline and refined systems without further selection. Any changes prompted by those results require a new holdout or must be reported as exploratory.

**Part B deliverables:** Search records, controlled reward/representation comparisons, policy-voting results, robustness analysis, baseline-versus-refined results and a final selected system supported by measured evidence.

## 6. Interactive Simulation and Dashboard

An animated start-to-goal navigation viewer is a required part of the project. Its purpose is to let the owner see how the robot moves, where it hesitates or fails, and how learning changes its behavior. Build a minimal viewer alongside the first environment and baseline agents, then expand it into the full dashboard.

Use Streamlit with local trajectory rendering, initially using Matplotlib. The viewer must visibly advance through individual robot movements; a static final route alone does not meet this requirement. Add another rendering approach only if needed for usable playback. Training should run through scripts; the interface loads trained policies and recorded episodes.

| Page | Required behavior |
|---|---|
| Overview | Explain the task, environment version, implemented agents and available evidence |
| Navigation simulator | Animate the robot from start to goal; play, pause, step, scrub, change speed and reset; inspect its trail and decisions |
| Learning analysis | Display saved validation learning curves and seed variation |
| Agent comparison | Show success, collisions, timeout, SPL, route lengths and measured runtimes |
| Optimization and robustness | Compare baseline/refined configurations, voting and scenario groups |
| Methodology | Explain observations, actions, rewards, splits, budgets and limitations |

### Navigation viewer layout

```text
Scenario: [Map / start / goal]    Agent: [Q-learning]    Checkpoint: [Saved run]

+-----------------------------------+--------------------------------+
|                                   | Episode status: Running        |
|   2D navigation map               | Step: 12 / 200                 |
|                                   | Last action: Right             |
|   S = start    G = goal           | Step reward: -0.01             |
|   R = robot    # = obstacle       | Cumulative reward: [measured]  |
|   Trail = cells already visited   | Position: [row, column]        |
|                                   | Revisited cells: [measured]    |
+-----------------------------------+--------------------------------+

[Play] [Pause] [Next step] [Reset]     Speed: [Slow ---- Fast]
Replay position: [Start ------------------------------- End]
Overlays: [Trail] [Visit counts] [BFS reference] [Action scores, if available]
```

This wireframe describes the intended layout. Its example step and reward are illustrative labels, not experimental results.

### Required visual behavior and controls

- Keep the start marker visible after the robot leaves it. Show a distinct goal, robot marker, obstacles, free cells and grid coordinates, with a clear legend. Use symbols as well as color.
- Animate one environment action at a time. A movement between adjacent cells represents one discrete action; any visual interpolation must not imply continuous robot dynamics.
- Default to showing the route traveled so far. Keep the future trajectory hidden during playback so the viewer can inspect the robot's behavior as it unfolds.
- Provide play, pause, next-step, reset-to-start, adjustable playback speed and a timeline slider for recorded episodes. Changing playback speed changes only the display pace.
- At each step, update the robot location, last attempted action, step reward, cumulative reward, action count and episode status together.
- Highlight a collision attempt at the blocked cell or boundary while retaining the robot in its actual cell. Stop playback at goal, collision or timeout and show the recorded outcome clearly.
- Allow selection of a saved validation scenario or a separate custom demonstration. Allow custom start/goal selection only on valid free cells; report unreachable custom goals before running.
- Keep an episode reproducible by recording the scenario, policy/checkpoint, seed and evaluation mode. Resetting a replay returns to its first frame rather than retraining or generating a different trajectory.
- Preserve playback position and loaded artifacts when display controls change. Distinguish running a frozen policy now from replaying a saved episode; neither mode trains the agent. Provide read-only training progress from saved logs rather than a second training implementation in the UI.

### Insight overlays and comparisons

| Feature | Insight it provides | Interpretation rule |
|---|---|---|
| Traveled trail and cell visit counts | Detours, repeated visits and loops | Derive counts from the displayed episode prefix |
| Optional BFS route overlay | Difference from the minimum-step reference | Label it as a planner reference; never feed it into the learned policy |
| Per-action scores | Which moves the current policy favors | Show actual Q-values or policy probabilities when available; Q-values are not probabilities |
| Step event log | Exact sequence of actions, rewards and outcomes | Selecting an event moves the replay to the matching frame |
| Early-versus-late checkpoint replay | How behavior changes during training | Use saved checkpoints on the same validation scenario with the same evaluation mode |
| Side-by-side agent replay | Differences in route choice and failure behavior | Match map/start/goal, synchronize by action index and hold finished agents at their terminal frame |

Keep overlays optional to avoid obscuring the robot. State/action displays describe recorded model outputs and do not claim a human-style explanation of its reasoning. If an agent does not expose a score, show it as unavailable.

Successful path efficiency is a completed-episode result. During playback, show traveled actions and reference shortest distance separately; do not display a partial route as completed SPL. All displayed rewards and counters must correspond to the current replay frame.

### Implementation and acceptance requirements

- Use the same environment transitions and policy interface as the experiments. The UI must not contain an alternative movement or reward implementation.
- A saved episode contains its initial state and each transition's before/after position, attempted action, reward, termination/truncation flags and available diagnostic scores. Rewinding reads these records without changing policy parameters or inventing new decisions.
- Verify pause/resume, stepping, timeline seeking and terminal frames against a hand-solvable route and a collision fixture. Check that controls preserve the current scenario and that counters match the replay position.
- First viewer milestone: animate random-policy and BFS episodes, then a saved Q-learning validation episode. The later dashboard adds comparisons, optimization views and richer diagnostics.
- UI completion requires an observable start-to-goal animation, an inspectable failed episode, working playback controls and synchronized step information. A route image or metrics table alone is insufficient.

Interface requirements:

- Display model/checkpoint ID, map ID, scenario ID and whether the episode is training, validation, final evaluation or a custom demonstration.
- Show actual step count, cumulative reward and the terminal outcome.
- Compare trajectories on the same scenario. Explain the information used by BFS and by each agent.
- Keep benchmark maps immutable. Custom map editing creates a separate demonstration artifact and never overwrites benchmark splits.
- Check the map hash and observation/action/reward versions before loading an agent or replay. A fixed-map policy is supported only on its matching map. If a custom layout changes, allow random/BFS demonstrations or show that a new matching policy is required; do not present the old coordinate-only policy as trained for the edited layout. Custom start/goal selection on the unchanged map remains supported.
- Custom demonstrations do not contribute to benchmark results. Do not enable interactive browsing of sealed test scenarios before the final evaluation is complete.
- Handle unavailable models, invalid scenarios and absent results with clear messages.
- Offer saved-run playback without retraining. Label an untrained/random demo clearly.
- Keep the application local by default; deployment is outside the initial scope.

## 7. Recommended Project Structure and Technology

```text
autonomous-robot-navigation-rl/
|-- README.md
|-- pyproject.toml
|-- .gitignore
|-- configs/
|   |-- environment.yaml
|   |-- agents.yaml
|   |-- experiments.yaml
|   `-- tuning.yaml
|-- data/
|   |-- maps/
|   |-- scenarios/          # versioned train/validation/test manifests
|   `-- README.md
|-- src/robot_navigation/
|   |-- envs/              # grid environment, maps, observations, rewards
|   |-- agents/            # common interface, random, BFS, Q-learning, SARSA, SB3 adapters
|   |-- training/          # loops, callbacks, checkpointing, seeds
|   |-- evaluation/        # metrics, scenarios, aggregation, failure analysis
|   |-- optimization/      # search, controlled experiments, policy voting
|   `-- visualization/     # shared map renderer, episode playback, overlays, learning curves
|-- scripts/
|   |-- prepare_scenarios.py
|   |-- train.py
|   |-- evaluate.py
|   |-- run_experiments.py
|   `-- summarize_results.py
|-- app/
|   |-- streamlit_app.py
|   `-- navigation_viewer.py
|-- tests/                 # environment, tabular updates, splits, metrics, checkpointing
|-- artifacts/
|   |-- runs/              # configuration + per-episode metrics per run
|   |-- checkpoints/
|   |-- trajectories/
|   `-- figures/
`-- docs/
    |-- project_plan.md
    |-- environment_specification.md
    |-- experiment_log.md
    |-- development_log.md
    `-- final_report.md
```

Create modules progressively. Use Python with NumPy, Gymnasium, pandas, Matplotlib, PyYAML and pytest initially. Add Stable-Baselines3/PyTorch for DQN, Streamlit for the interface, and optional tuning tools only when needed. Select a Python version supported by the installed RL packages and save exact dependency versions after a successful smoke run; do not assume the newest Python release is compatible.

A CPU-compatible small-grid configuration is the default. Measure pilot throughput on the actual machine. GPU acceleration is optional and should not be assumed faster for small environments. No paid API or cloud service is required for the proposed architecture.

Ignore caches, temporary files and bulky training artifacts in version control. Keep small maps, manifests, configurations and run summaries versioned. Store checkpoints locally with matching metadata and checksums.

## 8. Individual Workstreams

| Workstream | Responsibilities | Completion evidence |
|---|---|---|
| Environment and references | Task rules, maps, splitting, random agent, BFS | Environment tests and benchmark manifest |
| RL implementations | Q-learning, SARSA and DQN | Runnable agents and saved run configurations |
| Research and optimization | Fair evaluation, tuning, ablations, voting and robustness | Part A/B results and failure analysis |
| Application and documentation | Simulation interface, reproducibility instructions, report and demonstration | End-to-end local prototype and verified documentation |

One person owns all workstreams. Use milestone gates to control scope instead of maintaining fictional team contributions. Record decisions, time spent, external libraries and assistance in a development log so the work remains understandable and maintainable.

## 9. Six-Week Development Milestones

| Week | Work | Completion criterion |
|---|---|---|
| 1 - Environment and research setup | Finalize task, review RL approaches, implement grid rules, map checks, BFS, manifests and a minimal animated viewer | A deterministic, tested environment and visible random/BFS navigation with playback controls |
| 2 - First learned navigation | Implement Q-learning and SARSA; establish training/evaluation separation, artifacts and learned-agent replay | Reproducible tabular runs with measured validation outcomes and a visible Q-learning episode |
| 3 - Part A comparison | Add DQN; compare seeds, budgets and failure modes | Three-agent comparison and justified shortlist |
| 4 - Part B refinement | Tune shortlist, run controlled reward/representation experiments, evaluate voting | Traceable before-and-after and combination-method evidence |
| 5 - Integration and robustness | Expand the viewer into the full dashboard, add comparisons and insight overlays, analyze core robustness on the fixed map | Working navigation viewer with synchronized diagnostics, comparisons and robustness evidence |
| 6 - Final evaluation and documentation | Freeze system, run final tests, verify figures, reproduce from setup instructions, finish report | Reproducible private release with actual evidence and known limitations |

Unseen maps, moving obstacles, continuous movement and physical robots remain deferred even if time is available. PPO was authorized separately after these six core milestones. If behind schedule, reduce graphical polish while keeping the required animated viewer, model comparison, meaningful refinement, voting experiment and core robustness. If training fails, preserve the negative finding and reduce task difficulty only as a documented new benchmark version applied consistently to every approach.

## 10. Quality, Testing and Experimental Integrity

### Environment and algorithm checks

- Check seeded resets, valid observations/actions and deterministic transitions.
- Verify movement boundaries, obstacle collisions, goal detection and reward values.
- Verify terminal-versus-timeout handling, including a terminal event on action 200.
- Test hand-calculated Q-learning and SARSA updates, terminal targets and truncation bootstrapping.
- Check that episode reset does not corrupt SARSA's next-action handling or tabular state encoding.
- Confirm BFS distances and unreachable-map handling with small hand-solvable fixtures.
- Run the Gymnasium/SB3 environment checker before neural-agent training.
- Verify disjoint start/goal manifests for the core task. Duplicate-map rejection checks belong to the deferred unseen-map extension.
- Check metric calculations with known successful, failed and nonoptimal trajectories.
- Verify that saved/reloaded policies reproduce actions under the stated deterministic evaluation procedure.
- Confirm evaluation never changes learned parameters, buffers or normalization state.

### Experiment records

Each run must store run ID, source revision or source hash, dependencies, hardware, algorithm, configuration, training seed, scenario IDs, environment/reward/observation versions, interactions, runtime, checkpoint selection rule, per-episode outcomes and aggregate metrics.

### Run management and release acceptance

- Maintain a run matrix covering baselines, search trials, confirmations, ablations and voting evaluation. Estimate training time, validation time and artifact storage separately on the actual machine. Stop or reduce a search at predeclared limits, and report the reduced coverage.
- Save progress and recoverable checkpoints during long jobs. Distinguish an inference-only model from a training-resume checkpoint. A genuine resume must restore relevant optimizer, replay/exploration and random-state information, or disclose that exact resumption is unsupported. Interrupted runs are not complete-budget results.
- Before tuning, write down provisional learning targets for validation success, collision rate and route efficiency, and a practical playback responsiveness target based on the pilot. Report achieved values and missed targets; do not change targets after seeing the final test. These targets guide assessment and do not justify discarding poor seeds.
- Retain the existing correctness requirements independently of learning targets: BFS solves every reachable benchmark scenario optimally, outcome counts reconcile, and replay positions/rewards match saved transitions.
- Include a locally produced saved demonstration bundle so the viewer opens without retraining. Document setup, training, evaluation and UI launch commands for Windows, with the tested Python/dependency versions. Record the external sources used for methods or adapted code in the development log.

The core release has measured implementation, evaluation and reproduction evidence. Consult `docs/release_checklist.md` for the historical core acceptance record and `docs/plan_closure_status.md` for current checks and remaining limits. Passing implementation checks does not imply that every learned family met its performance targets.

Exact bitwise reproducibility may vary across hardware/libraries; document that limit and verify practical repeatability. Record runs interrupted by budget or software errors separately from complete learning runs.

### Interpretation requirements

- Explain when longer successful paths are preferable to a policy with occasional short successes and many failures.
- Report failure counts alongside successful-route statistics.
- Do not infer transfer to new maps from familiar-map held-out start positions.
- Do not present a more difficult environment, longer budget or changed observation as an isolated algorithm improvement.
- Describe collision avoidance in the simulator precisely; it is not physical safety certification.
- Report null results, instability and limitations in the final report.

### Technical report

Use the institution's report structure as a checklist: title, problem, relevant methods, proposed solution, environment/state design, experiment protocol, Part A results, Part B refinements, evaluation outcomes, limitations, development plan and reproducibility instructions.

The institution specifies a cover page, Times New Roman 11 pt, single spacing and around 40 pages below 50 for assessed reports. These are optional presentation targets here. Start with a concise technical report and expand only where experiments warrant it. Replace member allocation with the individual workstream/development log. Keep a separate rubric checklist if useful for self-assessment; there is no private-project submission requirement.

## 11. Development Priority and Definition of Done

### Priority 1 - First-release essentials

- Tested fixed-map environment and saved scenario splits.
- Random and BFS references; Q-learning, SARSA and DQN.
- Equal-budget multi-seed validation and a justified shortlist.
- Hyperparameter tuning and at least one controlled reward or representation experiment.
- Policy-voting comparison and core robustness analysis.
- Animated start-to-goal viewer with playback controls, synchronized step information and failure replay, introduced in the first milestone.
- Frozen final evaluation, traceable artifacts and an interactive local demonstration.
- README, experiment log and evidence-based technical report.

### Future backlog - Deferred learning and environment extensions

The items below are outside the current release and require a later scope decision before implementation.

- Further PPO tuning or a separately designed independent four-algorithm evaluation; the initial PPO validation comparison is complete.
- Map-aware policies trained across layouts and evaluated on held-out maps.
- Controlled action noise, deeper state/reward analysis and richer playback.

### Future backlog - Deferred robotics work

- Continuous 2D motion, robot footprint and realistic collision geometry.
- Partial observations, range sensors and memory-based policies.
- Moving obstacles with appropriate temporal state and new reference methods.
- ROS 2/Gazebo or another robotics simulator; physical hardware only after a separate hardware-specific plan.

### Definition of done

The first release is complete when another run from documented setup can prepare the saved task, train a chosen agent, evaluate without learning, load a checkpoint and watch navigation through the animated viewer with working playback controls and synchronized diagnostics; the full comparison/refinement study has real recorded evidence; the final test was kept out of selection; and the private report explains outcomes and limitations. Completion does not depend on every refinement improving performance or on RL beating BFS.

## 12. First Task for Codex

### Initial implementation steps

1. Inspect the workspace and preserve the energy plan and all synced references. Create an isolated `autonomous-robot-navigation-rl` working directory unless a different dedicated project location is supplied.
2. Add a minimal package, configuration, dependency specification and README. Do not create every optional module in advance.
3. Implement the Section 3 grid environment, with explicit observation/action spaces, seeded resets and separate termination/truncation handling.
4. Add one versioned connected map, map validation, BFS distances and saved train/validation/test start-goal manifests.
5. Implement random and BFS agents through the same action/episode interface.
6. Add Q-learning, initially with a small configurable smoke budget. Save its table, configuration, training log and validation trajectories.
7. Implement success/collision/timeout, SPL, return and route-length metrics. Validate these on hand-solvable scenarios.
8. Evaluate only training/validation fixtures in this milestone. Do not run the final test manifest.
9. Add focused tests for environment rules, updates, splits, metrics and save/load behavior; run environment checks and a command-line smoke workflow.
10. Build the minimal local navigation viewer from Section 6. Animate random/BFS episodes and a saved Q-learning validation episode, with play/pause, next-step, reset, speed control, replay seeking, visible start/goal/trail and synchronized action/reward/status information. Keep advanced comparisons and score overlays for later integration.
11. Save a trajectory image, replay records and actual measured validation outcomes. If the smoke run does not learn, report that accurately and identify the next debugging step.

### Acceptance criteria

- A documented command prepares the benchmark deterministically.
- Task rules match the saved configuration and environment specification.
- Reference routes have verified shortest lengths and no invalid movements.
- A real Q-learning run produces a saved table, interaction count and validation outcomes.
- Validation performs no learning; the final test is untouched.
- Seeded reset, episode endings, metric calculations and checkpoint loading pass focused checks.
- The map and split pass the benchmark definition gate; the pilot reports goal discovery and training/evaluation throughput, and the UI rejects mismatched fixed-map checkpoints.
- The local UI visibly animates a verified start-to-goal route and can replay an actual learned-agent validation episode, including failure if the baseline has not learned successfully.
- Playback controls work, diagnostic values match the current frame, and the UI does not alter benchmark rules or learn during replay.
- The existing energy project and `sources/` remain unchanged.
- No invented results or claims of completed experiments appear in the output.

### First handoff instruction

> Implement the first milestone of this private Autonomous Robot Navigation Using Reinforcement Learning project in its own directory. Build the configuration-driven fixed-map Gymnasium environment, versioned scenario splits, random/BFS references, a Q-learning baseline, evaluation metrics and focused tests. Include a minimal local UI that animates the robot moving from start toward the goal with play/pause, next-step, reset, speed control, replay seeking, a visible trail and synchronized action/reward/status information. Demonstrate a verified BFS success and replay an actual Q-learning validation episode, reporting failure honestly if it has not learned. Run a small real training-and-validation workflow, preserve the final test holdout, and save the actual configuration, table, metrics and replay records. Explain what was implemented and what the measured run shows. Complete this milestone before SARSA, DQN, PPO, tuning, ensembles, unseen maps or the full comparison dashboard. Keep all artifacts local and preserve the existing energy project and synced reference files.

---

## Sources and Planning Assumptions

- **Institutional source:** `C:/Users/User/Downloads/Assignment specification (v1).pdf`, BMDS2114 Machine Learning, four pages, inspected in full for this plan. No separate assessment rubric was available for weight verification.
- **Structure reference:** `BMDS2114_Smart_Energy_Codex_Project_Plan.md` in the current workspace. Its energy dataset/model details do not apply to this new navigation task.
- **Gymnasium environment contract:** [Create a Custom Environment](https://gymnasium.farama.org/introduction/create_custom_env/) and [Terminated/Truncated Step API](https://farama.org/Gymnasium-Terminated-Truncated-Step-API).
- **RL implementation and evaluation:** [Stable-Baselines3 RL Tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html), [DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html), and [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
- Technical documentation checked on 6 November 2026. Map size, reward values, six-week duration, scenario proportions, seed counts and training/search budgets are proposed project decisions. Pilot measurements may justify changes before the corresponding benchmark is frozen.
- **Online comparison and gap review, 6 November 2026:** See `Robot_Navigation_Project_Verification.md` alongside this plan for similar projects, supporting links, remaining decisions and the rationale for the benchmark, monitoring-budget and UI compatibility clarifications. Deferred additions remain outside the current release.

This document is a development plan. It contains no measured model performance and does not establish that a robot implementation, training run or final evaluation already exists.
