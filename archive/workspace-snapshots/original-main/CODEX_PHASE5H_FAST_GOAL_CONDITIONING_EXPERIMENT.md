# Codex brief — Phase 5H: Fast, Controlled Goal-Conditioning Experiment

**Repository:** `happywintersolstice-oss/Autonomous_Robot_Navigation`  
**Starting branch:** `feature/phase5g-longer-horizon-diagnosis-v1`  
**Reviewed parent commit:** `33f5843aa6bf3a7188757b6b13cff361de697ecf` (verify HEAD once)  
**Project directory:** `autonomous-robot-navigation-rl/`  
**New branch:** `feature/phase5h-fast-goal-conditioning-v1`  
**Study ID:** `phase5h-fast-goal-conditioning-v1`  
**Status:** New, prospective experiment instructions. No results claimed.

## 0. User's key requirement: stop spending excessive time verifying

We need to **make research progress** while retaining validity. Previous stages repeatedly ran hundreds of inherited tests and enormous byte inventories. **Do not repeat that validation process in Phase 5H.** Follow this two-level test policy:

- **Every code change:** use only fast relevant unit tests (aim <= 20 focused test cases), syntax/import checks and a short one-arm synthetic smoke test of <= 20 mechanical transitions. Repeat focused tests only when changing code needed for them; retain failure receipts, but do not build massive append-only per-attempt archives.
- **After code is frozen, before the one research run:** run this focused suite once; verify expected parent SHA and read *existing* Phase 5G preservation/verification receipts. Check SHA256 of the specific source/config files directly needed and of each generated artifact. Do not rehash all historical 10k+ repository files or copy historical archives.
- **After the research run:** analyze results once, run read-only result validation and the same focused suite **at most once if the code has changed**. No full inherited 479-test suite in this phase. Mark `full_regression_suite: DEFERRED_TO_MAJOR_MILESTONE` honestly. **No claim that all legacy tests pass in Phase 5H**.
- Preserve historical files and earlier feature branches by modifying new files only. A `git diff --name-status <parent>...HEAD` check should show changes confined to new Phase 5H paths. This is a scope check, not a full bytewise audit. Do not touch original dirty main workspace.
- Establish a concise resource/time ledger. Avoid duplicate evaluations, Git tree exports, repeated verification, comprehensive optimizer tracing, and multi-gigabyte snapshots. If an essential gate fails, stop once and report it instead of extended self-repair loops.

**Important:** Speed does not authorize skipping checks of scientific correctness, exceeding budgets, accessing final/unseen data, changing registered thresholds after outcomes, or reporting unverified success.

## 1. Why this experiment

Phase 5G's *fresh, single-seed, 2,000-action* B1 tests gave:

| Condition | Expanded success at 0 | Expanded success at 2k | U/D/L/R at 2k | Original at 2k | Learning gate |
|---|---:|---:|---|---:|---|
| DQN | 5/16 | 10/16 | 2/4, 3/4, 1/4, 4/4 | 1/4 | failed |
| PPO | 3/16 | 4/16 | 4/4, 0/4, 0/4, 0/4 | 1/4 | failed |

DQN had many Wait and oscillation actions despite 27 goal experiences and 257 optimizer updates containing goal examples. PPO exhibited direction collapse. Both have validated reward-to-update instrumentation from E/F/G, so running yet another *diagnostic-only* stage is low value.

**Hypothesis H1 (prospective):** Adding a small, explicitly registered four-way goal-direction prediction auxiliary objective to DQN's shared learned representation can improve robust goal-conditioned action preferences relative to a **same-seed, same-budget DQN control**. This is a testable intervention, not a claim about the cause of Phase 5G's failures.

**Why DQN only now:** Test one intervention on one algorithm to limit runtime, source changes and confounds. Q-learning, SARSA and PPO remain required in the eventual four-algorithm comparison. **Do not infer a four-model ranking from this focused experiment**. Do not silently modify the project's final experimental protocol.

**Trivial-supervision warning:** The target direction is already deterministically encoded in the public goal-position difference. Predicting it could become a trivial auxiliary task and may not improve navigation. The experiment must prove that any improvement occurs in the *policy's real navigation behavior*, not just in auxiliary classification accuracy.

## 2. One permitted scientific change

Compare exactly two DQN arms:

- **H0 control:** baseline B1 DQN with unchanged original environment, 412-feature encoder, network, rewards, replay, exploration schedule, hyperparameters and training-case sampler.
- **H1 auxiliary:** same DQN, same initial common-trunk and Q-head weights as H0 and same learner settings, plus a four-logit goal-direction prediction head trained from the **current public observation**. Its loss is auxiliary to the original TD loss. Declare one fixed coefficient **lambda = 0.05** before the run. Train head and shared trunk jointly; ensure the additional head's RNG initialization does not disturb H0/H1 common network initial weights or baseline sampler/exploration RNG. The Q head still chooses the action and must retain five actions including Wait. Never replace or override chosen actions using the auxiliary head.

Interpret label as direction of the signed relative-goal vector (Up/Down/Left/Right). **Training Fixture A starts are axis-aligned;** for encountered diagonal states, define the deterministic label *before training*, using the coordinate component with greatest absolute magnitude and a fixed tie rule (vertical first). At exact goal cells there is no action; do not fabricate a direction label. Derive direction labels only from the current agent-visible encoded observation, not simulator state, hidden maps or target action oracle. Preserve distinct goal delta sign, axis, and action-index conventions by explicit tests. The label denotes coarse direction, **not** the optimal next navigation action in an obstructed environment.

Architectural constraints:

1. Preserve baseline DQN Q-head and shared network dimensions for the H0 control. Add H1 auxiliary projection only as a small head from the final shared latent vector. Explain exactly which shared parameters receive the auxiliary gradient.
2. Preserve TD target computation/target-network semantics, goal reward, sampling and optimizer type. H1 adds loss `TD_loss + 0.05 * direction_cross_entropy`, using **only the same replay minibatch already sampled by DQN**. No second replay draw, no new environment action, no separate policy optimizer step. Record both component losses, Q-head target and direction-prediction accuracy.
3. The H1 extra head adds parameters, so this is **not a parameter-count-controlled comparison**. State this as a limitation. A matched-capacity extra-head negative control is outside this pilot.
4. If the installed SB3 DQN implementation cannot support such a shared-head intervention without invasive rewriting, **STOP with `INTERVENTION_BLOCKED`** and describe the shortest safe implementation path. Do not silently substitute behavior cloning, action masking, reward shaping, supervised pretraining, hand-coded greedy movement, policy override, new feature input, or a different DQN implementation.
5. Any validation on the original/evaluated development pairs must be inference only; never backpropagate a development label.

## 3. Frozen run budget and study settings

| Item | Requirement |
|---|---|
| Environment | Original 10 x 10 open Fixture A, exactly the existing 8 training cases |
| Feature | Original Phase 5B B1 relative-goal-x4; 409 raw values, 412 encoded |
| Seed | **23141** for the paired study (distinct from Phase 5G) |
| Models | H0 baseline DQN; H1 DQN + auxiliary direction head |
| Research interactions | **2,000 training actions per arm; 4,000 total max** |
| Training checkpoints | **0 and 2,000 only**; do not add opportunistic checkpoints |
| Evaluation | Same original 4 + expanded 16 previously inspected development cases at both checkpoints, per arm |
| Development episodes | **80 maximum** (2 arms × 2 checkpoints × 20 cases) |
| Development action cap | **16,000 maximum**, 200 per case; stop if exceeded |
| New extra synthetic smoke interactions | **20 maximum**, isolated and ledgered; no extra research training |
| Full legacy regression suite | **Deferred**, do not run during this phase |
| Wall limits | Scientific training <= 1,200 sec/arm; assessments <= 1,800 sec total |
| Evidence size | <= 128 MiB total new artifacts; no full legacy/source snapshots |
| Final/unseen/dynamic environments | **Forbidden** |
| Extra seed, tuning, automatic continuation | **Forbidden** |

Train both arms from fresh, paired initialization and independent but equally seeded environments. Do not warm-start from Phase 5F/G checkpoints. Record/verify initial Q/common-trunk parameter equality. Auxiliary head initialization must be isolated from the action/sampler RNG; avoid assuming trajectories remain identical once actions diverge. Reserve action budget before each `env.step`. No automatic second study if one arm fails.

If identical RNG streams become impossible to maintain due to the extra head, explicitly record the mismatch and mark `PAIRING_INVALID`; do not characterize the comparison as controlled. RNG *state after divergent trajectories* is not expected to match, and that alone is not a pairing failure.

## 4. What to evaluate (registered before running)

For each arm at initialized and 2k checkpoints, assess exactly the same 20 development scenarios by deterministic argmax policy. **Do not use the 20-case panel for training, tuning, calibration or choosing a checkpoint.** Store complete per-case outcome and first action, with reason, initial Q scores, goal margin, goal direction, waits, collisions, moving revisits, ABAB loops, number of actions and shortest-path length (Manhattan, since these are open grids). The initial checkpoint is a reference baseline, not an extra learned model.

Primary metrics:

- Number of expanded goals succeeded out of 16 and by direction out of four each.
- **First-step alignment:** actual first chosen action strictly reduces Manhattan goal distance on the fixed expanded panel.
- **Failure modes:** fraction of Wait actions, timeouts, loops, collisions, path inefficiency among successes; report denominators rather than averaging undefined ratios.
- Difference-in-differences of 2k success gains from own initialization: `(H1_2k - H1_0) - (H0_2k - H0_0)`; descriptive only, one seed.
- Auxiliary classification accuracy on **training replay records only** (never development labels); show that accuracy alone is not evidence of better navigation.

### Fixed decision criteria (no post-hoc relaxation)

The auxiliary arm is **promising for further replication** only if *all* occur at 2k:

1. H1 expanded successes >= **12/16**;
2. H1 succeeds at least **2/4 in each of Up, Down, Left and Right**;
3. H1 expanded successes are at least **3 above** H0 expanded successes at 2k;
4. H1 first-step aligned cases on the 16 expanded panel exceed H0 by **>= 3**;
5. H1 is not worse than H0 by more than **10 percentage points** in global Wait fraction; H1 does not exhibit a new complete one-direction collapse;
6. all integrity and pairing gates passed.

If no change or mixed changes, verdict `INTERVENTION_NOT_SUPPORTED` (or `INCONCLUSIVE_SMALL_PILOT` when measurement is valid but thresholds are missed with partially favorable outcomes). If success gates pass, verdict `PROMISING_REPLICATION_REQUIRED`, **not** “problem solved” or “ready for final.” Never pick the best of multiple checkpoints or seeds, because none is authorized.

## 5. Minimal but sufficient validation

**Before research:**

- Verify exact parent HEAD, source references, and that prior scientific code/artifacts are unchanged by worktree diff.
- Static check: observation vector/raw-to-feature conversion and five-action mapping; tests on all four axis directions, diagonals/ties, near-border cases, and at-goal exception.
- Confirm Q-head initial weights, network/target bootstrapping, replay minibatch identity, optimizer call count, TD loss, auxiliary loss, lambda, gradients to trunk/head and reproducible seed. In one fixed synthetic minibatch, set lambda=0 and check baseline Q update matches H0 within the established float tolerance; this is a **test**, not a third research arm.
- Assert that H1 policy acts using Q-head only; hidden geometry is never read; reward and exploration sampler stay original.
- Run the focused suite once after source is fixed, at most 20 cases plus the <=20-transition smoke. No full parent test suite.

**After research:**

- Independently reconcile training/environment action counts; DQN optimizer steps and Q target math; source/config/model hashes; checkpoint/load equality; evaluation episode/action ledger; all 80 case outcomes and first-step vectors; reward-to-replay sampling counts (without duplicating all trace tensors).
- Use Phase 5E/F validated instrumentation selectively: record aggregate targets, TD loss, auxiliary loss, direction accuracy, and traceable reward-to-update examples; keep short fixed windows or compact sampled IDs. **Do not capture the enormous full tensor journal from prior stages** unless indispensable to an integrity gate. Explicitly report which complete-coverage assertions from Phase 5G are not claimed in this lean run.
- `verify` must read only and create no new model actions. Preserve errors and mark incomplete if a mandatory field was not observed. No repeated rendering/verification loops.
- Run no additional research or synthetic validation. Record `full_regression_suite: DEFERRED_TO_MAJOR_MILESTONE`.

**Integrity gates:**

`G0`: parent/source/split integrity and budget registration.  
`G1`: two-arm pairing and only one objective change.  
`G2`: full 2k + 2k action ledger, checkpoint/state integrity, no excess optimizer calls.  
`G3`: 80 fixed development case outcomes and denominators, no final/unseen access.  
`G4`: focused tests, output hashes, honest limitations and valid result math.  
Failure of any gate -> `INCOMPLETE_OR_INVALID`, not a fabricated success.

## 6. Deliverables — small and navigable

Create *new Phase 5H files* rather than modifying historical study files:

```text
configs/partial_phase5h_v1.yaml
src/robot_navigation/partial_phase5h.py
src/robot_navigation/partial_phase5h_auxiliary.py
src/robot_navigation/partial_phase5h_report.py
# optional: one additional small helper only if genuinely necessary
developer/tests/test_partial_phase5h.py
docs/partial_phase5h_protocol.md
docs/partial_phase5h_results.md
docs/partial_phase5h_diagnosis.md
artifacts/partial-dynamic-navigation-v1/phase5h-fast-goal-conditioning-v1/
    plan.json
    train-receipts.json
    checkpoint-metrics.csv
    case-results.csv
    optimizer-summary.json
    validation.json
    preservation.json
    hashes.json
    # a minimum set of H0/H1 checkpoints and compact evidence files
```

Avoid new nested copies of repository trees, 10k-file inventories, per-transition full observation dumps, multiple redundant JSON receipts, or historical source snapshots. Clear, small CSV/JSON summaries plus necessary frozen checkpoints suffice.

Report exactly what changed in the network, parameter counts H0 vs H1, number of actual training steps, reward encounters, goal-direction metrics, control-vs-intervention comparisons, failures and path efficiency, runtimes, method limitations, and a **single** follow-up recommendation. Explain that development sets have been repeatedly used over earlier phases; these are engineering diagnosis panels, not untouched generalization tests.

## 7. Branch, stop and communication

- Branch from the exact reviewed Phase 5G commit. Do not merge main, rewrite older studies, overwrite source checkpoints or alter the original dirty working tree.
- Make a single reviewed commit and push only `feature/phase5h-fast-goal-conditioning-v1` once done, if Git access permits.
- If blocked, commit a truthful `INTERVENTION_BLOCKED` report rather than quietly changing the scientific intervention or expanding the budget.
- **Stop after Phase 5H.** No Phase 5I, full four-model training, hidden/final evaluation, or Phase 6 without explicit user approval.

**Bottom line:** Measure one causal intervention versus a same-budget control now. Keep mandatory scientific gates, but stop paying the huge repeated full-regression and repository-preservation cost at every experimental phase.
