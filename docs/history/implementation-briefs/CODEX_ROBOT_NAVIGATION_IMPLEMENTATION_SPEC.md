# Codex Implementation Brief: Partial-Observation Dynamic Indoor Robot Navigation

> **Target repository:** https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation  
> **Active application directory:** `autonomous-robot-navigation-rl/`  
> **Purpose:** Implement a **new, separately versioned research benchmark** for a four-member Machine Learning assignment, while preserving every historical benchmark, model, report, replay, frozen final evaluation, scientific artifact, and working UI.  
> **Status of this document:** Agreed research requirements + explicit provisional engineering defaults. **Do not present unimplemented features or unmeasured performance as completed.**

## 0. Instructions to Codex — read before editing

You are working on an existing Python/Gymnasium/Stable-Baselines3/Streamlit repository containing completed fixed-map and unseen-static-map RL studies. **This is an extension, NOT a rewrite.**

1. First inspect the repository tree, `README.md`, `docs/developer-guide.md`, `docs/project-structure.md`, `autonomous-robot-navigation-rl/pyproject.toml`, active source code, existing tests, and evidence/versioning conventions. Check `git status` before modifying files. Respect pre-existing user changes.
2. In particular inspect:
   - `autonomous-robot-navigation-rl/src/robot_navigation/core.py`
   - `autonomous-robot-navigation-rl/src/robot_navigation/unseen_benchmark.py`
   - `autonomous-robot-navigation-rl/src/robot_navigation/shared_features.py`
   - `autonomous-robot-navigation-rl/src/robot_navigation/shared_agents.py`
   - `autonomous-robot-navigation-rl/src/robot_navigation/shared_neural.py`
   - `autonomous-robot-navigation-rl/src/robot_navigation/episodes.py`
   - `autonomous-robot-navigation-rl/configs/environment.yaml`
   - `autonomous-robot-navigation-rl/configs/unseen_maps.yaml`
   - `autonomous-robot-navigation-rl/docs/milestone12_results.md`
   - `autonomous-robot-navigation-rl/docs/milestone13_results.md`
   - `autonomous-robot-navigation-rl/developer/tests/`
3. **Never overwrite or alter** the original `grid-v1`, fixed-map, `unseen-maps-v1`, M11/M12/M13 protocols, sealed test manifests, old checkpoints, old metrics, frozen archives, or historical replays. Preserve original entry points and app behavior. Prefer separate modules, configuration, commands, model and artifact IDs. Do not pretend an old checkpoint works with the new observation/action spaces.
4. Make a concise implementation plan, then implement in the dependency order below. At every gate run targeted tests; on failure fix the root cause before advancing. Do not silently relax or remove tests.
5. The instructions labelled **MANDATORY** are agreed design constraints. Choices labelled **PROVISIONAL** are initial values to document and test, not scientifically validated claims. If a provisional choice creates a conflict, choose the simplest consistent alternative, record the rationale and use a new version identifier.
6. Do **not** run expensive multi-model training, release a final evaluation, push to GitHub, rewrite reports to claim results, or change dependencies unnecessarily without explicit user authorization. You **may** run short, bounded smoke tests/pilots as part of implementation. Implementation and automated tests should otherwise proceed without needless confirmation questions.
7. For each milestone, report files changed, tests executed, actual results, unresolved risks, and how the existing benchmark was kept reproducible. Distinguish *tested*, *implemented but untested*, and *planned*.

## 1. Research objective and scope

**Proposed study title:** *Comparative Evaluation of Reinforcement Learning Strategies for Autonomous Navigation in Unknown and Dynamic Indoor Grid Environments*.

The robot is a **simulated grid robot**. Its objective is to reach a known goal from a known starting position while exploring a previously unknown indoor floorplan, detecting obstacles only through local simulated sensing, remembering discoveries, and responding to temporary blockages by waiting, detouring, exploring or revisiting previously observed locations.

**MANDATORY assumptions**

- Robot knows its **exact grid position** and **goal coordinates**. Do not implement uncertain localisation, SLAM, real LiDAR, cameras, or continuous physics.
- Different episodes may use **unseen 10×10 static base layouts**; all structural walls are permanent within an episode.
- Temporary blockages may **appear and disappear during the episode** only on structurally traversable cells.
- Local sensor provides only **occupancy** (blocked/free), **not** permanent-vs-temporary classification, remaining blockage duration, future changes, or true hidden map.
- The robot **does not know the remaining duration** of any temporary blockage.
- The robot may move, **Wait**, explore unfamiliar corridors, preserve past observations, choose a detour, and revisit a previously blocked route after it may have reopened.
- All four RL families to compare: **Q-learning**, **SARSA**, **DQN**, **PPO**. Four team members: one model owner each. An online non-RL navigation reference is shared group work.
- This is a **comparative simulated navigation study**, **not** a validated physical deployment or proof RL is better than conventional path planning.

**Research questions:**

1. Which learning family navigates most successfully and efficiently under partial observability?
2. How do the policies generalize to entirely unseen base layouts?
3. How do they behave when blockages appear/disappear and the agent must decide whether to wait, explore, detour or revisit?
4. What failure modes (waiting forever, oscillation, bad exploration, collisions, timeouts) remain, and how do the learned policies compare with a conventional online planner under the same information restrictions?

## 2. Preserve the old work: mandatory version boundaries

The existing code has multiple distinct historical tasks:

- Original `NavigationEnv` is **4 actions** with coordinate-only observations on static geometry, reward `-0.01`/step, `+10` for reaching goal, `-5` for collision, cap `200` actions.
- Existing `unseen_benchmark.py` has **104 values**, the last 100 of which encode the **complete true obstacle map**.
- Existing `shared_features.py` derives **1,234 full-map features**; `shared_agents.py` has 4-action linear weights; `shared_neural.py` assumes full-map occupancy. **None of these are valid unchanged for the new agent-facing task.**
- Historical unseen-static final means: Q-learning **16.81%**, SARSA **62.08%**, DQN **29.10%**, PPO **64.10%**, full-information BFS **100%**, measured on the old task. These are **historical context only**, not dynamic benchmark results and not a controlled improvement comparison after changing task inputs.
- M13 found serious timeout/cycling behavior. Do not hide loops by quietly overriding learned policy decisions.

**IMPLEMENT** a new versioned namespace, such as `partial-dynamic-navigation-v1`, with distinct observation/action/reward/scenario/artifact identifiers, separate results and protocol. Reuse pure, validated utilities where useful, but prevent new changes from mutating historical behavior.

Suggested new modules (adjust to actual repository structure):

```text
src/robot_navigation/
  partial_dynamic_env.py      # Gymnasium env: true state, events, action dynamics
  partial_sensor.py           # 4 cardinal rays, range 2, occlusion
  partial_memory.py           # per-agent observed occupancy + history
  partial_observation.py      # versioned 409-value observation + validation
  partial_benchmark.py        # new disjoint layout/scenario splits and signatures
  partial_features.py         # agent-visible feature transforms only
  partial_agents.py           # Q-learning and SARSA adapters, 5 actions
  partial_neural.py           # DQN/PPO adapters/extractors, 5 actions
  partial_baselines.py        # online exploration/replanning reference
  partial_evaluation.py       # metrics, paired trials, replay verification

configs/
  partial_dynamic.yaml

developer/tests/
  test_partial_sensor.py
  test_partial_memory.py
  test_partial_observation.py
  test_partial_dynamic_env.py
  test_partial_leakage.py
  test_partial_benchmark.py
  test_partial_agents.py
  test_partial_evaluation.py

docs/
  partial_dynamic_environment_spec.md
  partial_dynamic_experiment_protocol.md
  partial_dynamic_implementation_status.md
```

New file names are suggestions, not an instruction to duplicate existing code unnecessarily. Follow current packaging and test patterns, and do not move archived files.

## 3. Environment, action, and time specification

### 3.1 True environment vs agent-visible knowledge

Maintain **two strictly separate objects**:

- `true_state` (environment-private): base wall map, temporary occupancy by current step, scheduled event times, actual robot/goal coordinates, collision detection, episode progress, and evaluator-only ground truth.
- `agent_memory` (agent-accessible): only sensor-derived occupancy and observation history, exact known position/goal, visits, and elapsed time.

**No helper used by the policy or feature extractor may read** `true_state` or ground-truth topology. Protect the boundary by API design and tests. Ground truth may be in trusted evaluator `info` or render/debug methods, **never in observations, model inputs, action masks, or online-planning fallback**. If evaluator-side info could be consumed by an agent wrapper, split the interfaces clearly.

### 3.2 Actions — MANDATORY

Use `gym.spaces.Discrete(5)` with explicit, fixed IDs:

| ID | Action | State effect |
|---:|---|---|
| 0 | Up | row − 1 |
| 1 | Down | row + 1 |
| 2 | Left | column − 1 |
| 3 | Right | column + 1 |
| 4 | Wait | remains at same position; time advances |

Wait must not count as a new cell visit. Invalid actions raise an error; attempting to move into a currently blocked cell or out of bounds is a collision (initially terminal, to match the historical event convention). Collision attempts keep the actual robot location unchanged. Goal arrival terminates successfully. An episode cap truncates if there was no terminal event. **Never silently transform a chosen RL action into a safer action.**

### 3.3 Simulation schedule — MANDATORY ordering

Pick a single, documented Gymnasium-compatible time convention and use it everywhere. Recommended:

- At `reset` (time `t=0`): select declared scenario and episode event schedule; apply events active at `t=0`; create empty agent memory and mark start visited once; scan four rays; update memory; return observation for time 0.
- At `step(action)` given observation at `t`: execute the chosen action against the environment occupancy already observed at `t`; apply goal/collision/reward; increment time to `t+1`; **if continuing**, activate/deactivate the predeclared events for `t+1`, perform a fresh sensor scan and memory update, then return the next observation. For ended episodes, make the terminal observation and bookkeeping deterministic and document any final sensor scan choice.
- Environmental changes must not happen between the scan used to decide and the execution of that action in the same time step.
- Every move/Wait consumes **one** step. The sensor scans automatically once per decision step, with no separate Scan action.

For temporary blockage scheduled on the robot's occupied cell, **PROVISIONAL safe rule**: do not activate the blockage while occupied; skip/defer occupancy for that scheduled active tick without shifting or extending the global event window, and log the suppressed activation. Never teleport the robot or place a blockage inside it. This keeps the nominal event schedule fixed across model evaluations; actual applied events may differ by agent trajectory. Measure and disclose these differences. Alternative safe policy needs explicit versioning/tests.

### 3.4 Temporary blockages — MANDATORY constraints

- Only in base-map free cells; never change permanent `#` walls.
- Events have predetermined activation and clearance times generated with seeds **independent of agent actions**. The policy must not see them.
- Temporary blockages always eventually clear; avoid overlapping simultaneous events at the same cell for the initial version.
- Allow temporary disconnection of routes; do not assume a detour always exists. Test both "only corridor blocked" and "detour available" scenarios.
- Prevent permanently unreachable start-goal routes based on base geometry (base map connected or route connectivity checked). Disallow start/goal obstacle activation in the initial version unless an explicit tested scenario requires it.
- **PROVISIONAL pilot duration:** sample blockages lasting **3–8 steps**, but include varied distributions and cases where short detours beat waiting. Do not claim the range is optimal; freeze it only after development analysis.
- Store nominal event schedules, actual applied/skipped events, and seeds in evaluator/replay artifacts; not in agent observations.

### 3.5 Episode limit and event semantics

**PROVISIONAL** `max_steps: 200`, subject to checking whether this is sufficient for unknown dynamic maps. Initially treat it as an **external experimental truncation**, with no extra timeout reward, not as a task-level terminal deadline. Use `terminated=True` for goal/collision, `truncated=True` for the external action cap if not terminated. Correctly handle Gymnasium/Stable-Baselines3 bootstrapping at truncation. If later redesigning a real finite-horizon deadline or timeout penalty, version the reward/task and include remaining-time semantics appropriately. Do not silently change historical semantics.

## 4. Sensor specification — MANDATORY

**Sensor:** four simultaneous absolute cardinal rays ordered **North, South, East, West**, maximum **2 cells**, no heading or rotation model, deterministic/perfect initially.

For each ray, step outward from the current robot position:

1. If the next in-bounds cell is free at the **current time**, mark it observed free and continue up to range 2.
2. If the next in-bounds cell is blocked (either permanent or temporary), mark it observed blocked and **stop the ray**. Cells behind remain unknown/stale.
3. If the ray reaches the map boundary, stop; no outside-map cell is added to memory. Because bounds are known with position/grid size, the boundary can be reported as a virtual obstacle at distance 1 or 2 (document this convention explicitly). No out-of-bounds indexing.
4. If all 2 in-bounds cells are free, return "no obstacle within range".

**PROVISIONAL scalar encoding per direction:** `0 = no obstacle within range`, `0.5 = first blocked cell or boundary at distance 1`, `1.0 = first blocked cell or boundary at distance 2`. Do not conflate `0` with "the first cell is unknown"; all cells on an unobstructed scanned ray up to sensor range become observed. The true map is **never** copied wholesale to memory. A previously observed free cell that is now blocked is updated **only if reobserved**; ditto a previously blocked cell that becomes free.

The sensor reports **occupancy only**, never permanent vs temporary class, predicted reopening, or a full-map raycast result beyond range 2. In a known-dimension grid, the robot may deduce boundary coordinates from its exact position; this is not equivalent to knowing interior obstacles.

## 5. Agent memory — MANDATORY

Represent each 10×10 cell with these logically distinct pieces of agent history:

- `occupancy`: **`-1 unknown`, `0 last observed free`, `1 last observed blocked`**. Value `1` is **not** a permanent-wall label.
- `last_seen_step`: initialized `-1`; set only when the cell is actually scanned.
- `visit_count`: initialized `0`, start cell `1`, increment on **entering** a cell; moving back counts another visit; **Wait does not increment**.
- `consecutive_blocked_observations`: increment only when the cell was seen blocked in the **immediately preceding decision-time scan** and again in the current scan. Reset to `0` on visible free or loss of uninterrupted visibility; do **not** increment while out of sensor range. This is not the true blockage age.
- Optional internal `currently_seen` mask derived from which cells were scanned at the current step, for testing/visualization. Do not expose privileged visibility or ground-truth labels.

At reset: clear every field between independent episodes. At each scan: update only sensed cells; undiscovered/stale cells retain their last known occupancy. Last-observed timestamps must not update for unseen cells. If a cell was free then becomes blocked, the history may suggest occupancy changed; do not provide a hidden permanence label.

**Critical scientific limitation:** A blocked cell first observed in its blocked state may be a permanent wall or a temporary blockage. These may be observationally indistinguishable. No implementation can guarantee correct classification of all such cases. The study should assess how policies handle that uncertainty, not falsely promise perfect classification.

## 6. Proposed common observation v2 — 409 values (PROVISIONAL, test before freezing)

Use a single `np.float32` flattened vector with deterministic row-major map order. Share this *agent-visible information contract* across all four algorithms. There is no requirement that all four architectures use the same hidden-layer capacity.

| Slice | Length | Data | Encoding |
|---|---:|---|---|
| `[0:4]` | 4 | robot `(r,c)`, goal `(r,c)` | each /9 |
| `[4:8]` | 4 | N, S, E, W first obstacle distance | `0`, `0.5`, or `1` per sensor spec |
| `[8:108]` | 100 | last observed cell occupancy | `-1` unknown, `0` free, `1` blocked |
| `[108:208]` | 100 | time since last actual observation | if ever seen, `min(t-last_seen,20)/20`; else `0`, distinguish with occupancy `-1` |
| `[208:308]` | 100 | visit count | `min(count,5)/5` |
| `[308:408]` | 100 | consecutive blocked observations | `min(streak,5)/5` |
| `[408:409]` | 1 | elapsed step | `min(t,200)/200` |
| **Total** | **409** |  | `Box(low=-1, high=1, shape=(409,), dtype=float32)` |

**Important consistency rules:**

- `t=0` observation reflects the initial scan. The robot knows position/goal and the 10×10 grid dimensions but not interior layout.
- Never derive local-clearance, frontier, shortest-path, map-graph or geometric features from the true map; any derived features may use **only these 409 values or the logically equivalent agent memory**.
- The observed-map encoding is a hypothesis, not a guarantee of learnability. Linear Q-learning/SARSA with 409 raw features may be inadequate. Allow a **versioned, deterministic shared feature transform** derived only from agent-visible input, document and ablate it. Keep underlying information access equivalent across algorithms.
- A clipped age of 20 means "observed at least 20 steps ago," not "known to be unchanged for 20 steps." A blockage streak is based on continuous observations, not true blockage age.
- Elapsed time is available, but the policy must not see the **remaining blockage lifetime** or a hidden event schedule. Explain whether elapsed time is used for history reasoning versus horizon awareness.
- Validate all vector lengths/ranges/finiteness and reset reproducibility.

**Potential improvement if pilot demonstrates serious feature confusion:** separate observed/unobserved masks or use a small tensor/dict representation; this constitutes a *new observation version*. Do not adjust the feature contract separately for one model.

## 7. Reward — PROVISIONAL conservative baseline, no hidden information

Begin with a **separately versioned reward configuration** closely aligned with historical semantics:

```yaml
step_reward: -0.01        # EVERY action, including Wait
success_bonus: 10.0      # reaching goal, plus step_reward
collision_penalty: -5.0  # collision, plus step_reward
max_steps: 200           # external truncation initially
extra_timeout_penalty: 0.0
```

- No automatic exploration bonus, coverage bonus, hidden shortest-path distance shaping, or artificial escalating Wait penalty in the initial benchmark.
- No forced "after three waits, move" or "after five visits, override action" logic inside the RL baseline. These would be **hybrid interventions**, not pure learned-policy behavior, and must be tested/labeled separately if introduced later.
- Log every Wait, time cost, collision and episode outcome. An always-Wait policy may avoid collisions but should fail by timeout; report that honestly.
- Evaluate whether −0.01/step makes waiting or oscillation attractive relative to collision, and whether 200 steps is enough for exploration; do **not** assert good reward design just because training runs.
- Reward/timeout changes require new signatures and matched development experiments. No retrospective changes to historical returns.

## 8. Four-model architecture and team division

Keep all four **separate new implementations/checkpoints**; use the common environment, actions, scenario split and information restrictions:

| Team member | Primary RL family | Proposed implementation | Shared responsibility |
|---|---|---|---|
| 1 | Q-learning | Semi-gradient linear action-value approximation with 5 actions; **not** exhaustive tabular map memory | Observation contract/validation |
| 2 | SARSA | On-policy semi-gradient linear action values with 5 actions; same allowed feature definition as Q-learning | Sensor/memory tests |
| 3 | DQN | SB3 DQN with 409-value (or shared, versioned agent-visible transformed) observation and 5 action outputs | Dynamic environment/event tests |
| 4 | PPO | SB3 PPO (actor-critic) with same underlying permitted observation and 5 action outputs | Evaluator/metrics/replay tests |

**Must not reuse** old 4-action weight matrices or neural policies as if action `Wait` already existed. Do not silently copy entire full-map feature extractor from `shared_features.py`/`shared_neural.py`: those read the actual 100-cell true wall layout. Maintain fresh checkpoint/version metadata and validate expected shapes (`5 × feature_count` for linear methods, five-output policy for neural agents).

Training fairness:

- Common layout and dynamic scenario generator, observation/sensor/reward rules, and evaluation runner.
- Predeclare training interaction budget, tuning/search budget, evaluation frequency, and seeds; use **at least three independent training seeds per family** for substantive reporting (more if feasible).
- Compare controlled **environment interactions**, not just wall-clock training hours; also report runtime and inference cost because model families have different computation.
- Do not claim that identical raw inputs make model capacity and hyperparameter tuning perfectly equal; explain architecture differences.
- Prevent testing-set leakage through tuning, feature design, early stopping, plots or manual example selection.
- Keep original Q-learning/SARSA semantics correct: Q-learning bootstraps with max next-action value; SARSA bootstraps with the **next behavior-policy action**. Separate genuine terminal transitions from external truncations according to the chosen Gymnasium semantics. Validate SB3 terminal observations/autoreset behavior.
- The trained agent may react to updated memory without updating model weights during evaluation; label this *adaptive decisions*, not necessarily *online learning*.

## 9. Fair non-RL references

Implement at least one **online exploration-and-replanning baseline** with **the same sensor and map-memory restrictions**, not a hidden full-map BFS. Define and document:

- How it treats unknown cells (e.g. optimistic exploration with controlled frontier selection).
- When it replans after new observations.
- Its rule for deciding to Wait versus explore/detour.
- Its handling of stale memory and repeated visits.
- That it cannot inspect future changes, true static walls behind occlusion, or obstacle type.

For interpretation, a **full-information BFS** (or future-aware oracle, if desired) may also be reported, **explicitly labeled an information-advantaged reference** and *never* placed as an information-equivalent competitor. Do not declare dynamic-route optimality using a static shortest-path length.

Add simple diagnostic policies: `always_wait`, `random`, and optionally `always_detour`/greedy toward goal to reveal degenerate rewards and scenario problems. These are references, not new student-assigned RL models.

## 10. Experimental scenarios, splits, and metrics

### 10.1 Required scenario classes

1. **Unknown static map:** permanent walls only; learn obstacle discovery/exploration.
2. **Previously unseen permanent wall:** confirm it does not induce unending Wait behavior without measure.
3. **Temporary obstacle in sole corridor:** waiting is necessary; corridor reopens.
4. **Temporary obstacle with alternate short detour:** rerouting may beat waiting.
5. **Temporary obstacle with long detour:** waiting may beat rerouting.
6. **A cell observed free later becomes blocked while out of range:** memory stays stale until rescanned.
7. **A blocked cell reopens while out of range:** memory stays blocked until revisited and rescanned.
8. **Unseen base map with multiple independently scheduled temporary changes:** combined stress test.
9. **Permanent obstacle adjacent to a tempting route** vs initially indistinguishable temporary blockage: expose observational ambiguity.
10. **Temporary blockage activation at occupied robot cell:** test safe suppression and logged actual event application.

Seed nominal schedules independently of agent actions; report realized differences caused by trajectory-specific suppression. Keep enough scenarios where an alternative route is useful; avoid a task where "always wait" is almost always best.

### 10.2 Training / validation / fresh final test

Create **new** layout/route/event manifests with different identities and seeds from old unseen-map M10–M12 data. Guard against identical layouts **and rotations/reflections (D4)** across splits and against all previously inspected historical final layouts if feasible. Different event seeds on the same old final layouts do **not** make them an independent new final map test.

**PROVISIONAL initial split proposal:** 24 training layouts, 6 development validation layouts, 10 *sealed* fresh final layouts; configurable counts if the generator, budget or difficulty stratification requires adjustment. Stratify by route length, dead ends, bottlenecks, dynamic obstruction impact, etc., and validate base connectedness. Freeze new final layouts and event schedules **before** final model selection. No final evaluation until the user approves the registered protocol and models/configurations are frozen.

Test and report all four conditions separately:

- Familiar layout / static obstacles.
- Unseen layout / static obstacles.
- Familiar layout / dynamic obstacles.
- Unseen layout / dynamic obstacles.

Avoid reporting training or development successes as independent final evidence. Use paired scenario IDs/seeds for each policy/seed where possible. Distinguish multiple obstacle RNG draws from independent **training seeds**; never treat repeated evaluations of one trained checkpoint as independent trained models.

### 10.3 Metrics and exact scope

Primary: success rate, collision rate, timeout rate, step/action count, mean episodic return; report by layout, difficulty, model, training seed, and relevant uncertainty/variation. Since Wait consumes an action, use **elapsed action count**, not just movement-path length.

Additional:

- Wait count and proportion, longest consecutive wait streak.
- Repeated visits, oscillation/loop frequency, and no-progress periods; distinguish useful revisits from pathological cycles.
- Discovered-cell coverage (diagnostic, **not** an objective to maximize).
- Recovery success after an obstacle obstructs a relevant route, and observed-to-recovery delay; **define the denominator and detection event** precisely before calculation.
- Replanning/detour behavior and *retrospective* unnecessary-wait estimates, explicitly labelled as post hoc (the agent does not see counterfactual outcomes).
- Sensor refresh/stale-memory correction counts, detected events vs actual scheduled events, events never observed by each trajectory.
- Inference latency, training time/steps, memory footprint, scenario count.

Preserve the old static SPL definition for historical studies. If computing dynamic efficiency, document a new metric clearly (e.g. elapsed-action ratio against an **online, same-information** reference under paired nominal scenarios) and disclose that different trajectories can encounter different applied obstacle changes. **Do not call a static-map BFS distance the dynamic optimum.**

### 10.4 Scientific integrity and result reporting

- A complete episode ledger including scenario ID, layout hash, event seed, training seed, observation version, reward/action versions, policy/checkpoint hash, code/config signatures and outcome.
- Record actual transitions, action IDs and rewards, sensor observations, known-map snapshots or reproducible deltas, nominal/applied events, and termination/truncation. Audit replay for exact reproduction.
- A model that times out, waits all episode, collides or fails to learn is still a valid experimental result; don't silently remove or "repair" failures.
- Do not report new success percentages until actually measured. Do not assert RL beats online planning without evidence.

## 11. Mandatory tests and information-leakage controls

Write deterministic unit/property/integration tests (use repository's `developer/tests/` conventions):

**Sensor**

- Obstacle at distance 1 detected; no cells behind revealed.
- Obstacle at distance 2 detected; first cell marked free; no cells beyond revealed.
- Obstacle at distance 3 **not** detected.
- Correct North/South/East/West order; walls occlude; boundary/corner rays safe.
- Temporary obstacle that is currently blocking has identical sensor output to a permanent wall at the same location.

**Memory**

- Unknown initialized to `-1`; only sensed cells change; current/start cell observed as free.
- Free→blocked and blocked→free changes only update after they enter sensor range again.
- Unobserved occupancy remains last seen; age increases; streak **does not** increment while unobserved.
- Visit count increments on movement into cells, not Wait; resets between episodes; repeat visits counted.
- If consecutive blocked observation is interrupted by loss of visibility, restart streak upon reobservation.

**Leakage / observational equivalence — HARD GATE**

- Construct two different true maps with **identical position/goal and identical observable history** but different unseen cells. The observation vector, memory state and every policy-facing feature must be exactly identical until the changed region becomes observable. Do this again for unseen dynamic event changes and future event schedules.
- Verify no agent-visible method, feature extractor, action mask, planner/helper or reward signal provides hidden map, hidden obstacle class, future schedule or hidden shortest-path distance.
- Sanity-check model wrappers and vectorized observations (training and inference) use the new versioned encoder, **not** `unseen_benchmark.map_observation()` or old `SharedMapExtractor`.

**Actions/time/environment**

- `action_space.n == 5`; Wait keeps position and advances time; scan/reward occurs exactly once under the declared transition order.
- Collision and goal events terminate; external max-step action limit truncates unless a terminal event has priority.
- Schedules deterministic from declared seeds; temporary blocked cell reopens on schedule; blocked permanent wall never clears.
- Temporary activation never overlaps the occupied robot cell; suppression logged without information leakage.
- With identical reset/scenario/agent actions, replay produces identical observations, events, rewards, positions, flags and metrics.
- Check terminal and truncation bootstrapping for linear and SB3 learners; test any vectorized autoreset wrappers explicitly.

**Observation**

- Float32, shape `(409,)`, finite and within declared bounds. Check 4 + 4 + 100×4 + 1 = 409; correct slice ordering, row-major memory and normalization/clipping.
- Position and goal coordinates correct; no corruption by flattening; unknown and freshly observed free are distinguishable.
- Action output for each of four algorithms is an integer 0–4; checkpoint metadata rejects incompatible task/observation/action/feature versions.

**Baselines/evaluator**

- Online planner has no full-map accessor.
- Known simple scenarios with "must wait," "detour preferable," and "wall remains permanent" behave according to declared logic.
- All metrics have unit tests for denominators, failures, Wait actions and no-obstacle episodes; all saved result counts reconcile.
- Verify historical tests and app still pass; do not weaken old regression assertions.

**No full RL training before the sensor/memory/leakage/environment gates pass.**

## 12. Implementation stages and stop/go checkpoints

### Phase 0 — Audit and baseline

- Record current Git revision/status, install/runtime assumptions and `pytest` baseline results.
- Confirm actual files and test commands. Read evidence/history docs. Document conflicts with this specification.
- Implement nothing destructive; add `docs/partial_dynamic_implementation_status.md`.

**Exit:** baseline health and compatibility risks recorded.

### Phase 1 — Sensor + memory + observation

- Implement versioned ray sensor, partial map memory, and 409-value encoder with strict unit tests, including observational-equivalence leakage test.
- Provide an easily inspectable deterministic demo trace for one hand-written 10×10 example. No RL needed.

**Exit:** all sensor/memory/leakage/encoding tests pass.

### Phase 2 — New Gymnasium dynamic environment

- Build separate `Discrete(5)` environment, event generator, reset/step ordering, safe occupied-cell handling, rewards, event logs.
- Add static/temporary wall scenarios and replay validations. Use manually chosen actions to prove Wait and reopening work.

**Exit:** Gymnasium API checks, deterministic replay and required scenario tests pass.

### Phase 3 — Benchmark splits + classical reference

- Generate/propose disjoint new map/schedule splits, validate uniqueness and difficulty, create only training/development manifests initially; reserve the sealed final set.
- Implement information-equivalent online exploration/replanning baseline and simple diagnostic policies.
- Implement metrics, event snapshots and development-only evaluator.

**Exit:** baseline can't see ground truth; metrics match handwritten expected outcomes; no final-set exposure.

### Phase 4 — Adapt all four RL approaches

- Q-learning and SARSA with five-action linear value functions from agent-visible features.
- DQN and PPO with new five-action compatible networks/wrappers and correctly versioned metadata.
- Tiny sanity training on small training cases to verify numerical stability, gradients/updates, replay buffers, checkpoint save/load, and valid actions. If any model fails, diagnose rather than claim it lost the comparison.

**Exit:** all four complete bounded training/validation smoke runs; old checkpoints remain untouched.

### Phase 5 — Controlled developmental study (requires user agreement on budgets)

- Register model configs, interaction budgets, map/event seeds, screening vs confirmation seeds, and selection rules **before** large runs.
- Pilot fixed scenarios first: unknown static, only-corridor blockade, useful detour, and combined unseen/dynamic.
- Tune on development data only, with fair declared budgets; do not force a model to learn by secretly injecting a pathfinding fallback.
- Diagnose loops and waiting before adding intervention or reward ablations; preserve failures.

**Exit:** reproducible development evaluation with complete records and an honest limitations section.

### Phase 6 — Final evaluation / UI (only after freeze authorization)

- Freeze model IDs/config/feature versions and the new independent held-out layout/event manifests.
- Run sealed final evaluation once under a registered protocol; audit saved replays and calculations.
- Extend Streamlit with a clearly labeled **new** experiment/view: true map vs *robot-known map*, sensor rays, timing of temporary events, Wait actions, and side-by-side policies. Never let UI visualizing ground truth become an agent input.
- Preserve current **Demo / Results / Guide**, saved study results and root `python run_app.py` startup.
- Document measured outcomes and limitations. Do not alter past reports or historical tables.

**Exit:** reproducible analysis, regression checks, visual verification and no historical breakage.

## 13. Commands / environment conventions

Repository states Python **3.11+** in `pyproject.toml`; existing documented Windows environment uses a locked installation. Follow `docs/developer-guide.md` before selecting tooling. From `autonomous-robot-navigation-rl/`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation --no-deps
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.runtime/regression-tests
.\.venv\Scripts\python.exe -m robot_navigation.cli check
.\.venv\Scripts\python.exe -m robot_navigation.demo_assets verify
```

On Linux/macOS, adapt activation/interpreter paths. Do not depend on private APIs or external services. The original root app launches with `python run_app.py`; verify it still starts if the UI is modified. Keep new runtime output in ignored temporary locations and write deliberate research artifacts to **new namespaced/versioned paths**.

## 14. Required deliverables from Codex

Provide:

1. **Code changes**: new versioned environment, sensing, memory, observation, experiment split tools, dynamic scenario generator, 4 RL adapters, online reference, evaluation/replay (phased as above).
2. **Tests**: evidence for all acceptance checks, especially no leakage and old regression preservation.
3. **Documentation**: new environment spec, experiment protocol, data-flow diagram, run commands, reward/action/observation versions, known limitations, team responsibilities.
4. **Reproducibility**: seeds, configs, hashes, checkpoint versions, manifest generation and audit steps; no new claims without measured runs.
5. **Status summary**: implemented vs not, exact tests and outcomes, smoke-run findings, remaining blockers and future protocol proposals.
6. **Demo**: optionally add a distinct view showing known-vs-true map and changing obstacles after core correctness. Existing app and historical evidence remain available.

**Definition of success for implementation:** the environment demonstrably lets a manually controlled robot discover unknown walls using only four two-cell sensor rays, accumulate and revise map knowledge, wait for a temporary blockage to clear, explore/detour/revisit, and expose exactly the same legal agent-facing observation contract to four independently trained five-action RL algorithms **without hidden-map or future-event leakage**. Successful *learning* and superior RL performance are **empirical questions**, not implementation acceptance assumptions.

## 15. Unresolved research choices — be explicit, do not conceal

These are **not** fully approved scientific constants:

- Whether the proposed **409-value** representation is learnable and efficient for all algorithms; engineered agent-visible features may be required and must be controlled/versioned.
- Whether 2-cell sensors should be ablated against 1 and 3 cells; use 2 as primary pilot and plan that ablation on validation data.
- Whether 3–8-step temporary blockages generate useful wait-vs-detour tradeoffs. Vary/test duration distributions before freezing.
- Whether the 200-action external cap is enough for exploration and waiting; test sensitivity without redefining reward unannounced.
- The final number of training/development/test layouts and route/event scenarios; pre-register, justify and ensure actual holdout independence.
- How to compare dynamic efficiency when agent trajectories alter which nominal blockages actually activate.
- How a constrained linear value approximation compares to neural architectures without implying representational capacities are identical.
- Whether additional recovery controllers, adaptive reward shaping, wait penalties, or memory encodings are worth an explicit follow-on ablation. **Never silently add these to baseline agents.**

**Final instruction:** Make the scientific boundaries visible in the implementation. Code that runs is not enough: if a policy sees hidden obstacles, if waiting is secretly overridden, if layouts are reused after inspection, or if old frozen results are rewritten, the new comparative study is invalid even if performance looks impressive.
