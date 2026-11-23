# Codex Task Brief — Gate 3 / Phase 5A: Bounded Learning-Feasibility Audit

> **Repository:** `https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation`  
> **Starting branch:** `feature/partial-dynamic-navigation-v1` (reported head at review: `45f3f389518226d3ab054eb3fd321625f7b2636a`; inspect live HEAD instead of assuming it remains unchanged).  
> **Working directory:** `autonomous-robot-navigation-rl/`  
> **Project:** Comparative Evaluation of Reinforcement Learning Strategies for Autonomous Navigation in Unknown and Dynamic Indoor Grid Environments  
> **Scope:** Investigate *whether* four freshly adapted RL families can learn basic navigation under the new observation/action contract. **This is a limited diagnostic pilot, NOT the main comparison, tuning study, sealed final evaluation, or UI redesign.**

## 0. Non-negotiable instructions to Codex

1. **Inspect before editing.** Read the actual current working tree, branch, `git status`, recent commits, developer guide, new partial/dynamic environment spec and protocol. Don't assume the PDF status report is more authoritative than the checked-out code. Detect untracked and uncommitted user changes and preserve them. If the starting branch does not exist locally, find it and switch safely without discarding work. Prefer a new working branch from the existing feature branch if practical. **Do not merge, force-push, reset, clean, or push without explicit permission.**
2. **Keep historical studies intact.** Do not edit or regenerate original fixed-grid, unseen-static, warehouse, frozen final, M11/M12/M13, historical models, sealed manifests, releases, replays, reports, archived artifacts or previously generated experimental outcomes. All new scripts, tests, logs and evidence must use clearly versioned, separate paths.
3. **Do not silently fix the all-Wait result.** Do not add an arbitrary Wait ban, automatic anti-loop policy override, hidden BFS fallback, leaked obstacle label, extra future information, post-hoc reward manipulation, automatic movement mask, or unpublished changed reward to make results look good. Failed learning is valid evidence.
4. **Execute only bounded pilots in this brief.** Explicitly track RL training interactions and compute/time limits; stop if limits or safety gates are exceeded. Never run unconstrained optimization or a full 4-model multi-seed comparison. No final/test layout generation, no final evaluation, no UI work.
5. **Source-of-truth boundary:** Every learned policy and online baseline may receive **only** the already-defined 409-value observation or deterministic transforms of it. Do not give a policy the environment object, ground-truth grid, nominal event schedule, future clear time, scenario metadata or trusted diagnostic snapshots. The evaluator may read ground truth only for metrics and reproducibility.
6. **Never modify measured evidence after the fact.** Keep raw failed trajectories, trial budgets, seeds, settings, artifact hashes, timestamps, exact execution commands and exceptions. Distinguish observed measurements from hypotheses. Never fabricate runs, metrics, learning curves, training completion, or improvements.
7. **Do not ask the user to make routine implementation choices.** Make safe decisions consistent with this brief, record them, and proceed. Stop and report a blocker rather than violate preservation or experiment-integrity constraints.

## 1. Current verified/reported context (do not treat it as new pilot evidence)

New partial-observation implementation exists under:

```text
src/robot_navigation/
  partial_contract.py
  partial_sensor.py
  partial_memory.py
  partial_observation.py
  partial_dynamic_env.py
  partial_fixtures.py
  partial_benchmark.py
  partial_baselines.py
  partial_evaluation.py
  partial_features.py
  partial_agents.py
  partial_neural.py
  partial_training.py
  partial_commands.py
configs/partial_dynamic.yaml
docs/partial_dynamic_environment_spec.md
docs/partial_dynamic_experiment_protocol.md
```

The Codex implementation status PDF reported 250 pre-existing tests and 42 new tests passing (292 total), with an intact historical workspace. **Independently rerun relevant tests in the current checkout** rather than asserting that the report is sufficient.

The existing *functional smoke*, seed `22001`, trained each of Q-learning, SARSA, DQN and PPO for **256 actions**, then evaluated each on four scenarios. All **16 assessment episodes timed out with 200 Wait actions**; the models completed no successful goals. This shows that the interface, updates, checkpoints and replay can function, **not** that learning succeeds or that rewards alone caused stalling.

Existing partial/dynamic task contract (preserve for baseline pilots):

| Item | Current setting |
|---|---|
| Grid | 10 × 10 |
| Position and destination | Exact, known |
| Sensor | Four cardinal rays; up to two cells; occlusion; occupancy only |
| Memory | Last sensed occupancy, age, visits, continuous blocked observation streak |
| Observation | 409 `float32` values, version `partial-memory-409-v2` |
| Shared transform | 412 affine features, version `partial-visible-affine-v1` |
| Actions | 0 Up, 1 Down, 2 Left, 3 Right, 4 Wait |
| Obstacles | Permanent walls and temporary blockages; remaining duration unknown to policy |
| Rewards | `step_reward=-0.01`, `success_bonus=10`, `collision_penalty=-5`, `extra_timeout_penalty=0` |
| Cap | 200 actions, as **external truncation**, not a task-terminal deadline |
| Model families | Linear semi-gradient Q-learning, linear semi-gradient SARSA, SB3 DQN, SB3 PPO |
| Existing dataset | 24 training layouts, 6 development layouts; 288/72 declared scenarios |
| Existing baseline | Online observed-map replanner, plus AlwaysWait and seeded Random |

Additional review concerns to investigate:

- The 412 affine features have limited robot-relative obstacle interactions for linear value methods. Do **not** assume the input is sufficient or change it silently.
- In the existing generated dataset, only **16 of 540** reported scheduled blockage impacts disconnect a route while active. This is *not* the fraction of all episodes requiring waiting, but suggests limited representation of mandatory-wait cases.
- With current rewards, 200 Wait actions yield **undiscounted** return `-2`, whereas immediate collision yields `-5.01`. This comparison **does not establish** that the reward function alone caused all-Wait behavior.
- The online reference is a documented heuristic and may fail. It is **not** an omniscient or mathematically optimal dynamic planner.

## 2. Target deliverable and research questions

Implement a **versioned, development-only diagnostic experiment runner** and its tests/evidence to answer:

1. Can all four methods learn to navigate a very simple, known-reachable task with the new five-action output, or do they still collapse to Wait?
2. Can they learn to discover an unknown permanent wall and navigate an alternative path?
3. Can they learn to Wait for a temporary blockage to reopen when no detour exists?
4. Can they distinguish wait-vs-detour cases in outcome and behavior, despite unknown remaining durations?
5. Are failures related to input-feature limitations, random exploration, insufficient learning interactions, episode truncation handling, model initialization/tie breaking, or reward structure? Provide **evidence**, not unsupported causal assertions.

**A successful pilot is a reproducible diagnosis, not necessarily a high success rate.** Report failures honestly.

## 3. Pre-flight audit and engineering checks — stop on failures

From `autonomous-robot-navigation-rl/`:

```powershell
.\.venv\Scripts\python.exe -m robot_navigation.partial_commands validate
.\.venv\Scripts\python.exe -m robot_navigation.partial_commands prepare
.\.venv\Scripts\python.exe -m robot_navigation.partial_commands trace --output .runtime/phase5a-manual-trace-NEW.json
```

Use unique fresh output names (or safe OS-appropriate Python commands), never overwrite earlier evidence. Also run the existing full regression test command documented in `docs/developer-guide.md` if feasible, without weakening existing assertions; otherwise state precisely which suite was not run.

Verify and test, specifically:

- Observation length and value bounds; occupancy unknown/free/blocked semantics; coordinate/goal and relative offset correctness; no out-of-range sensor rays.
- Hidden-map/future-event **observational-equivalence**: maps or schedules with the same observation history must yield identical policy-visible observations and engineered features.
- `Wait` consumes a step, does not count as movement/visit, and does not bypass occupancy updates.
- At a scan, permanent vs temporary blocked cells are **indistinguishable** until an actual observed occupancy change occurs.
- Correct `terminated` vs externally `truncated` handling for linear Q-learning/SARSA and SB3 DQN/PPO; correct next-state bootstrapping and terminal observations on vector autoreset.
- No forced action replacement; count chosen Wait actions exactly. With all-zero Q/SARSA weights, inspect deterministic tie behavior and whether the training exploration distribution covers all five actions.
- Checkpoint save/load and inference evaluation do not change policy parameters; record the real actions executed, not reconstructed actions.
- Trusted snapshots remain outside the policy input and any baseline helper.

**If a real implementation defect appears, repair it with a regression test and a versioned code/evidence record.** Re-run affected gates. Do not silently change the science settings to hide a defect.

## 4. Diagnostic fixture suite (new, distinct namespace)

Use deterministic, **hand-authored** 10×10 fixtures for unit checks and bounded learning pilots; keep them outside the generated benchmark train/validation/final data. Define each fixture explicitly and validate base-map connectivity and expected ground-truth outcome. Keep testing fixtures and learned-policy evaluation fixtures separately where practical; do not call training-fixture performance unseen-layout generalization.

| Fixture | Setup | What it tests |
|---|---|---|
| **A. Open-grid navigation** | No walls or temporary events; several start/goal pairs with varied directions and short distances, including adjacent goals | Basic goal-directed learning; detects Wait collapse and directional/tie bias |
| **B. Unknown permanent wall** | Hidden structural wall blocks direct movement; a valid detour exists | Sensing, memory and exploration rather than waiting forever |
| **C. Only-corridor temporary blockage** | Corridor has no alternate route, temporary obstacle clears within known *simulator* schedule but unknown to policy | Ability to Wait or avoid collisions and resume after reopening |
| **D. Temporary blockage with detour** | Both waiting and detouring can succeed; include one short-detour/longer-blockage and one longer-detour/shorter-blockage case | Behavioral preference and regret under unknown clearance times |
| **E. Reopened out-of-range cell** | Previously observed blocked/free cell changes while outside sensor range; later reobserved | Stale memory handling and legitimate revisiting |

**Critical design checks:**

- A/D should include several distinct cases, not a single memorized path. Avoid always putting the goal in one direction or always activating the same Wait duration.
- In C, the blockage must actually intersect the robot's necessary route during the intended evaluation window. Don't rely on a random event that might disappear before the robot reaches it.
- In D, compare waiting and detouring using **post-hoc, fully specified** hindsight diagnostics only; the learned policy must not be told the remaining blockage time or true complete map.
- Ensure blocked cells do not spawn inside the occupied robot cell; preserve existing event suppression rules.
- **Distinguish scripted environment verification from learning.** Successful manually scripted trajectories prove dynamics, not policy competence.
- Keep initial familiar-fixture training and evaluation distinct from truly unseen-layout generalization. The latter is **out of scope** for this bounded pilot.

## 5. Functional controls — required before learning conclusions

Evaluate the following diagnostic policies on the same fixtures/scenario versions:

- `AlwaysWait` (negative control; must time out unless start/goal invalid, which is forbidden).
- Seeded `RandomPolicy` (exploration and collision reference).
- Existing `OnlinePlanner` (same-information planning heuristic; report failures and limitations).
- A small **handwritten demonstration** controller for selected fixtures (test oracle *for environment mechanics only*, not a learner or a fair operational comparison).

An evaluator may compute post-hoc shortest paths with full true maps, but **never pass this information to policies**. Match scenario identity, event seed, initial state, and evaluation conditions across methods. Report actual realized event activations separately from nominal schedules where trajectory-dependent suppression occurs.

## 6. Registered learning-feasibility budget — finite and enforceable

Before each run, write a local machine-readable `phase5a_pilot_protocol.json` recording the selected training fixtures, action budgets, settings, random seeds, evaluation cases, checkpoint rule, time budget, code/config hashes and stop conditions. Do not change it after seeing results; amendments need a new version/ID and rationale.

**Default pilot envelope:**

- Four RL families: Q-learning, SARSA, DQN, PPO.
- **One primary diagnostic training seed per family** for the first pass. Treat this as an engineering pilot, **not a statistical comparison**. You may add **one separately registered confirmation seed per family** if learning is observed or results are inconsistent, but do not use it to make significance claims.
- **Training interaction budget:** at most **10,000 environment actions per family per seed**, inclusive of all curriculum stages and retries that perform training (do not count fresh reruns as free). Thus the first-pass ceiling is **40,000 actions total**, excluding scripted controls and evaluation. If the optional second registered seed is used, absolute ceiling **80,000 training actions**. Start smaller, such as 1,000–2,000 actions per family, and only progress toward the cap when needed and computationally viable.
- Include a **hard finite wall-clock budget**, logging the observed timing on the current machine; abort gracefully at the limit and preserve partial results. Do not extrapolate performance from unfinished budgets. Select a conservative cap based on existing smoke timings and runtime resources, then register the numeric cap before training.
- Limit pilot evaluation to a **small, predeclared fixture matrix** (e.g. 4–12 episodes per fixture family per checkpoint), with a strict cap and no access to reserved final/test scenarios. Do not repeatedly inspect hidden final data for debugging.
- Training should begin with **A**; if basic learning fails, stop progression to C/D for that algorithm and diagnose rather than burn the full budget on difficult tasks.
- If using a curriculum (A → B → C/D), it must be the same declared progression and total interaction accounting for all four families, though model-specific optimizer settings can differ as declared.
- Keep a small **training-versus-diagnostic evaluation split** within fixture families; disclose when the same layout geometry is reused and do not call such evaluation unseen generalization.
- The original baseline feature/reward/action/observation versions remain unchanged for this first diagnostic run. Keep the existing `max_steps=200` truncation convention. For easy-fixture diagnosis, report episode steps and excessive Wait separately; do not add a timeout penalty without its own future versioned ablation.

**Do not turn this into full Phase 5, a hyperparameter sweep, 3+-seed confirmation, or Phase 6.** All conclusions are exploratory until a separate main-study protocol is approved.

## 7. Training and evaluation diagnostics that MUST be collected

For each algorithm, fixture category, seed and training checkpoint, save:

1. Actual environment interactions, completed episodes, success/collision/timeout counts and rates, mean return, and episode length distribution.
2. Full **five-action selection distribution** (Up/Down/Left/Right/Wait) for training and evaluation **separately**; longest consecutive Wait streak, total Wait fraction, loops/revisits and newly discovered map cells.
3. Early-versus-later performance on **fixed development fixtures** with saved evaluation trajectories. Do not run only greedy evaluation if that makes training exploration invisible; record both training actions and deterministic evaluation separately.
4. Realized temporary events: nominal vs applied counts, observed events, suppressed activation ticks, recovery opportunities and successes under existing definitions; use null/undefined when denominator is zero, never invent a 0% recovery rate.
5. Model state fingerprints and checkpoint hashes; learner update counts; parameter finiteness; numerical errors; DQN replay-buffer/learning-start steps and PPO rollout/update counts where available.
6. Training budgets, seeds, exact config, code snapshot/hash, hardware/runtime metadata, resource usage and dependency versions.
7. Specific *failure examples*: first obstacle discovery, first attempted collision, first sustained Wait streak, permanent-wall abandonment or failure, and any successful post-blockage recovery. Show what the robot **knew** at the decision time rather than only the true map.

Also perform a **reward/value sanity analysis** using the unchanged reward schedule: show accumulated and, where appropriate, discounted returns of simple illustrative paths (e.g. immediate collision, successful short path, prolonged Wait). Explain precisely why these are *hypotheses* about incentives, not proof of what caused the learned policy behavior.

## 8. Diagnose failures without making unregistered changes

If an algorithm still selects Wait in nearly all evaluation steps, inspect in this order:

1. **Environment correctness:** does an adjacent-goal action succeed and return the expected observation/reward? Are action IDs matched across policies?
2. **Exploration:** does training actually sample directions as well as Wait? Does early collision terminate training episodes too quickly? Are starts/goals direction-balanced?
3. **Learning updates:** does the successful-goal reward ever reach the replay buffer/learner? Are transitions, targets, terminal/truncation and SARSA next-action logic correct?
4. **Feature representation:** can the agent distinguish a wall immediately to its right from a wall far away, relative to its current position, using the current features? For linear approximators, demonstrate expressiveness limits with controlled examples rather than assume universal failure.
5. **Value estimates:** inspect finite, state-specific Q values, action probabilities/greedy scores and the role of initial ties; do not secretly bias actions toward movement.
6. **Reward/cap hypotheses:** report whether unchanged penalties encourage stalling in illustrative cases, but do not attribute observed outcomes causally without a registered matched experiment.
7. **Data composition:** check that temporary blockages happen during relevant decision windows and that training encounters examples where waiting and detouring are each useful.

If it cannot learn easy Fixture A by the registered action cap, **report the failed basic-learning gate** and provide a ranked set of possible causes supported by logs. Avoid moving directly to a larger train/test study.

## 9. Representation improvement — DIAGNOSTIC PROPOSAL ONLY

The present common transform is `bias + raw 409 observation values + two relative goal coordinates` (412 total). The underlying 409 values include *absolute* occupancy positions and relative local rays. This can be restrictive for linear value models because many useful interactions depend on robot-relative position.

**Do not change the default feature extractor in the first run.** If diagnostic evidence warrants it, write a *proposal* for a separate versioned robot-relative feature ablation using only visible observations. Examples of permissible candidate features:

- occupancy/unknown/free status of adjacent robot-relative cells, respecting sensor visibility and stale memory;
- robot-relative goal direction and distance;
- local discovered-map topology and observed-frontier indicators, derived only from memory;
- observation age, visits and blocked-observation streaks near the robot.

The proposal must explain dimension, encoding, information source, feature-count budget, computational cost, comparison controls and exactly how neural and linear families receive the transformed information. **Do not silently train with a changed feature version or change the model mid-run.** Obtain a new experiment/version label and registered comparison protocol before executing the ablation.

## 10. Scenario composition — design critique required

Inspect `partial_benchmark.event_schedule`, generated development manifests and diagnostic metrics. Produce a short table with actual counts for:

- events active while a robot would be near the shortest route versus events that are never relevant;
- cases where a temporary blockage disconnects a currently relevant path;
- cases with a detour and cases where waiting is necessary;
- event duration and activation-step distribution;
- routes that end before their scheduled event can matter;
- trajectory-dependent activation suppression.

**Important:** A nominal event scheduled on an initial shortest path does not guarantee an agent encounters it while active. The figure **16 of 540** is an impact statistic, not proof that only 16 episodes require waiting. Define denominators carefully, and do not overclaim.

Offer a separately versioned **future benchmark stratification plan**, ensuring enough static, permanent-wall, mandatory-wait, short-detour and unseen-dynamic cases. **Do not replace or optimize current manifests in place after observing results**; do not silently cherry-pick favorable cases.

## 11. Minimal code changes expected

Prefer **additive** development-only files, e.g.:

```text
src/robot_navigation/partial_feasibility.py          # reproducible fixture/control/pilot runner
src/robot_navigation/partial_pilot_diagnostics.py     # action distributions and learning diagnostics
configs/partial_feasibility_v1.yaml                   # registered bounded pilot settings
developer/tests/test_partial_feasibility.py            # fixtures, ceilings, replay, no leakage
developer/tests/test_partial_pilot_diagnostics.py      # metrics, gating, zero-denominator behavior
docs/partial_feasibility_protocol.md                  # pre-run protocol and limitations
docs/partial_feasibility_results.md                   # measured results only
```

Adapt naming to repository conventions and avoid duplicate modules. Use new artifacts, e.g.:

```text
artifacts/partial-dynamic-navigation-v1/feasibility-pilot-v1/
.runtime/partial-feasibility-*/
```

Keep existing `partial_commands smoke` and its 4×256 functional proof intact. A **new** optional subcommand/CLI for the pilot is acceptable if it cannot access `final`/`test` data and enforces hard budgets, unique output paths, fresh validation gates, artifact hashing and replay audits.

Do not modify the existing public Streamlit `Simulator / Results / Guide` views for this task. If additional user-visible demo would help, propose it for later approval instead.

## 12. Go/no-go reporting rubric

Produce the following report, *even if all algorithms fail*:

| Gate | Required evidence | Result |
|---|---|---|
| G1 — Engineering | Sensor, memory, observation, action/Wait, leakage and regression tests | Pass/fail with commands |
| G2 — Basic learning | Performance on easy open-grid/adjacent-goal scenarios vs AlwaysWait/Random | Observed success, Wait distribution, curves |
| G3 — Obstacle reasoning | Permanent-wall and temporary-only-corridor cases | Actual observed behavior |
| G4 — Decision quality | Detour vs waiting cases | Actual results and post-hoc interpretation |
| G5 — Method integrity | Budget, train/eval separation, seeds, checkpoint, replay, hashes, no final peeking | Pass/fail |

**Suggested feasibility criterion (not grading target):** Demonstrate at least one reproducible successful learned trajectory on an easy fixture and a measurable change in goal success or useful action distribution over the untrained/early checkpoint. This is an evidence gate to consider more pilots, **not** proof of generalization or permission for the main study. If unmet, document why and recommend the smallest defensible follow-on diagnostic.

End with one of the following **honest verdicts**:

- `READY_FOR_PHASE5_PROTOCOL_REVIEW`: bounded pilot demonstrates meaningful basic learning and tests/method gates pass; a new multi-seed, disjoint-split protocol still requires approval.
- `REQUIRES_REPRESENTATION_OR_TRAINING_DIAGNOSIS`: pipeline correct but basic learning remains weak or all-Wait dominated.
- `IMPLEMENTATION_BLOCKED`: functional defects, leakage, reproducibility, or test failures remain.

No verdict means automatic permission to run a full 4-family training study or create the final/test reserve.

## 13. Required final response from Codex

Provide, in this order:

1. Checked branch/commit and pre-existing `git status`; exactly which files were created/changed, and why.
2. Commands executed, tests passed/failed, real training and evaluation interaction counts, and any halted or partial runs.
3. Compact per-algorithm/per-fixture table: success, collision, timeout, mean steps, Wait %, longest Wait streak, discovered cells, and controls.
4. Evidence artifact paths with hashes/checkpoint IDs, reproducible pilot command, and replay validation outcomes.
5. Failure analysis, including competing explanations and evidence that rules out simple interface defects where possible.
6. Scenario composition findings and an explicit *proposal only* for a future robot-relative feature or reward ablation (if justified).
7. One of the three go/no-go verdicts above and the narrowest recommended next step.

**Final rule:** The user wants a rigorous Machine Learning assignment comparison. Do not conflate "the code runs," "the policy learned something," "the policy generalizes," and "the policy is safe for real robots." Each requires separate evidence.
