# Product Phase 2: measured RL performance

**Complete application phase.** Four genuine learned policies are integrated into the existing new-only app. Results below are on the same 24 fresh product development layouts, used for checkpoint selection. They are not scientific unseen accuracy or deployment-readiness claims.

## Same-panel starter versus selected results

| Family | Original 128-action starter success | Selected success | Selected collision / timeout | Selected training actions |
|---|---:|---:|---:|---:|
| q-learning | 0/24 | 23/24 (95.8%) | 0/24 / 1/24 | 30,000 |
| sarsa | 1/24 | 22/24 (91.7%) | 0/24 / 2/24 | 10,000 |
| dqn | 3/24 | 18/24 (75.0%) | 0/24 / 6/24 | 10,000 |
| ppo | 1/24 | 21/24 (87.5%) | 0/24 / 3/24 | 30,000 |

Non-RL references: Observed-Map Replanner **24/24** successes; Random **2/24**. Neither ever substitutes for an RL policy. Earlier delivery scores used four manual presets; those historical 0/4,1/4,0/4,0/4 values are not mixed into this same-24-case table.

## Cases, features and learning

- Curriculum `product-curriculum-v2`: 192 seeded train-only layouts (seed 82101), six families of 32: short open routes, turns, permanent racks, maze detours, temporary waits and temporary detours. The first 15% of actions uses short routes, then stages expand at 15%, 35%, 55%. Sampling remains deterministic under the learner seed.
- Development: 24 reachable layouts (seed 82901), all content/layout hashes distinct from training; 16 static and 8 temporary, with short (base distance <=5) and longer strata. Layout families are shared; hash separation alone does not demonstrate novel topology/generalization.
- Training case hash: `5d989b6d171175d50df212b1bed1ecc950b357e6160e716610b14088deaf9fab`. Development case hash: `846eb9dc1adb8a3568193103a60af8bb8bbb33d60fe471637003704c129d9020`. Exact IDs, rows, start/goal, events, seeds and case hashes are in the study datasets and comparison JSON.
- Official 10x10 environment, range-two occupancy/memory, 409 observations, five actions and 200-step cap are unchanged. Public sensor order is N/S/E/W; actions Up/Down/Left/Right/Wait. Official reward remains step -0.01, goal +10, collision -5, no extra timeout penalty.
- New feature version `partial-action-local-goal-v2`: 480 features, 96 per candidate action, derived only from the public 409-vector. Features include goal-offset/distance bins, candidate progress, occupancy, visits, age, repeated blocks and sensor rays. Boundary occupancy follows public coordinates. No shortest-path feature, map oracle, teacher, event schedule, clear-time input, action mask or scripted rescue is used.
- Q-learning/SARSA use independent semi-gradient TD action-value models. Weights are shared across candidate actions within each model, with different TD targets for the two families. DQN has its own shared-action neural scorer, replay and target network. PPO has its own learned shared-action actor and value model. Actor logits remain available for all five actions.
- Stabilized Q-learning selects the next greedy action online but evaluates it with weights frozen for 100 updates; TD error is clipped to [-5,5] and weight norm bounded at 100. This is a genuine Q-learning variant. Its final norm change was 9.151, with 300 target refreshes. No policy preference is initialized from a planner.
- B0/B1 native five-action starter loading remains strict and supported. Four-action models are rejected. F2 signatures validate task, feature count/version, architecture, finite weights and checkpoint SHA-256.

## Selected immutable checkpoints

| Family | Run | Feedback | Actual learning updates | Parameter L2 change |
|---|---|---|---:|---:|
| q-learning | `20261122T133047Z-befd7c15/q-learning` | none | 30,000 | 9.150876 |
| sarsa | `20261122T132241Z-144d4d48/sarsa` | progress-v1 | 10,000 | 9.354361 |
| dqn | `20261122T132156Z-f8d472ff/dqn` | none | 2,375 | 9.685571 |
| ppo | `20261122T132656Z-a8469b3e/ppo` | progress-v1 | 1,200 | 6.694591 |

DQN has 2,375 actual optimizer steps. PPO has 1,200 actual optimizer steps; its SB3 counter 600 counts epochs and is not reported as the optimizer-step total. These are inference checkpoints, not optimizer/buffer/RNG exact-resume snapshots.

- q-learning checkpoint SHA-256: `54a40d81afc9706fb2215e0904430f363e5803b0f95a318f6036581b1bb4ed82`.
- sarsa checkpoint SHA-256: `d0a60fd564567bcd79b20714583fd14594a282a6f09cbc6e808a8d3037219448`.
- dqn checkpoint SHA-256: `d71750fe816a0d450bbf9005628eb352c3dea18ee99f32405477b9ba12b05727`.
- ppo checkpoint SHA-256: `1fcec48acf883d12104a6a863a3ca854a7a6ffdce393d906b97b9776b5b8ba64`.

Selection is explicit in `runs/selection.json`: maximize development success count, then original mean return, fewer Wait actions, and lower budget; deterministic first candidate breaks an exact tie. Best is not an alias for newest: the selected DQN is the older unshaped18/24 model, rather than the newer shaped9/24 model.

## Matched 10,000-action shaping ablation

| Family | Unshaped | Progress-shaped |
|---|---:|---:|
| q-learning | 0/24 | 0/24 |
| sarsa | 22/24 | 22/24 |
| dqn | 18/24 | 9/24 |
| ppo | 16/24 | 13/24 |

All matched candidates used seed 82111, the same F2 representation, curriculum schedule, budget and neural settings. Shaping is train-only `official_reward +0.2*(old Manhattan distance-new Manhattan distance)`, computed only from observations. It is not claimed policy invariant under discount 0.99. Evaluation always uses official reward.

Progress shaping did not improve success in the matched 10k comparison and hurt the neural results. New training therefore defaults to unshaped feedback; optional shaping is visibly disclosed. The selected SARSA is shaped with the same success count but better official-return tie-break. PPO shaped 30k reached 21/24; the larger budget and shaping effects cannot be separated from that comparison.

## All attempts, budget and failures

| Family | Candidate actions | Feedback / rule | Development success |
|---|---:|---|---:|
| q-learning | 10,000 | none / original TD or neural | 0/24 |
| q-learning | 10,000 | progress-v1 / original TD or neural | 0/24 |
| q-learning | 30,000 | progress-v1 / original TD or neural | 0/24 |
| q-learning | 50,000 | progress-v1 / original TD or neural | 0/24 |
| q-learning | 30,000 | none / target-v1 | 23/24 |
| sarsa | 10,000 | none / original TD or neural | 22/24 |
| sarsa | 10,000 | progress-v1 / original TD or neural | 22/24 |
| dqn | 10,000 | none / original TD or neural | 18/24 |
| dqn | 10,000 | progress-v1 / original TD or neural | 9/24 |
| ppo | 10,000 | none / original TD or neural | 16/24 |
| ppo | 10,000 | progress-v1 / original TD or neural | 13/24 |
| ppo | 30,000 | progress-v1 / original TD or neural | 21/24 |

- **220,000 substantive training actions completed** across 12 genuine runs. One SARSA 30k attempt was interrupted: last saved progress 7,040 actions; exact stop count is unavailable. It has no usable checkpoint and is canonically cancelled. Its progress file is the last observed running snapshot; the manifest/interruption receipt records the final cancellation. The consumed 30k slot was not retrained. Conservative total including that entire requested slot is 250,000, below the registered 400,000 ceiling. Every individual run is <=50,000 actions and below the one-hour worker guard.
- Mechanical activity was separate: 256 actions before substantive training to validate four native save/load paths, plus 256 in the focused acceptance fixture. No scientific historical run or full legacy regression suite was executed.
- The first evaluation gate rejected `split=development` because the existing evaluator accepts `validation`/`manual` labels. All eight completed 10k checkpoints were preserved and reused. Only development split metadata was corrected; train hashes, layout geometry, goals, events and the authoritative evaluator stayed unchanged. The original failed-label dataset/protocol and the compatible dataset are both retained.
- A raw Q-learning 30k candidate started before that integration fault was noticed, and SARSA 30k was interrupted. This sequencing deviation is recorded rather than retroactively claimed compliant. Subsequent measurements confirmed the Q escalation criterion; it still scored 0/24 and was not selected. The interrupted SARSA slot was consumed.
- Raw Q-learning scored 0/24 at 10k, 30k and 50k despite training successes; its parameter-change norms grew roughly 9 ->155 ->893. A disclosed stability amendment was registered before one further 30k candidate. That candidate scored 23/24. Target lag, TD clipping and projection changed jointly; there is no isolated causal ablation of those mechanisms.
- After an intermediate 240-episode comparison had been written, selection publication hit a relative/absolute study-path error. The path was corrected, all intermediate artifacts retained, and the final comparison regenerated from existing checkpoints. No substantive training was repeated for that error. Completed study orchestration now refuses accidental repeated training.

## Subsets and remaining unsuccessful routes

| Family | Static success | Dynamic success | Short success | Longer success |
|---|---:|---:|---:|---:|
| q-learning | 15/16 | 8/8 | 13/13 | 10/11 |
| sarsa | 14/16 | 8/8 | 13/13 | 9/11 |
| dqn | 11/16 | 7/8 | 13/13 | 5/11 |
| ppo | 13/16 | 8/8 | 13/13 | 8/11 |

- q-learning: product-development-013 (maze-detour; timeout).
- sarsa: product-development-010 (permanent-racks; timeout), product-development-013 (maze-detour; timeout).
- dqn: product-development-004 (turns; timeout), product-development-012 (maze-detour; timeout), product-development-013 (maze-detour; timeout), product-development-014 (maze-detour; timeout), product-development-015 (maze-detour; timeout), product-development-016 (temporary-wait; timeout).
- ppo: product-development-011 (permanent-racks; timeout), product-development-012 (maze-detour; timeout), product-development-013 (maze-detour; timeout).

Every selected-policy failure is a real 200-step timeout, not hidden or replaced. These policies still loop or wait on some layouts. One seed and a 24-case panel used for model selection support a product demonstration, not statistical superiority or deployment safety. No frozen/final/unseen historical cohort was consulted.

## Exports, application and teammate demonstration

Saved final comparison: `runs/comparisons/20261122T133310Z-833d1797`. It contains 240 episodes and 15,142 executed actions, case CSV, summary CSV, JSON with hashes/provenance and replay ZIP. The interrupted/intermediate runs and all candidates remain preserved under their own IDs. CSV/JSON expose success/collision/timeout counts, successful-step means (blank/N/A with no successes), reward, Wait, revisits, loops, recovery and static/dynamic/distance subsets.

The existing pages/style and four simulator presets are retained. Best on development / Trained / Starter labels are truthful. Compare opens the saved comparison; new evaluations require an explicit rerun. Raw RL and the non-RL references have separate labels. All checkpoint choices remain accessible. See [team demonstration](docs/team-demo-script.md) for four fair roles and examples checked against the actual models.

## Effective architecture/settings provenance

The initial generic `settings` dictionary inherited starter defaults that did not govern the F2 builders (including hidden width 32, PPO epochs 5, replay capacity 10,000 and target interval 250). Completed run files remain immutable. The supplementary `effective-settings.json` reads actual settings and parameter shapes from all 12 serialized native checkpoints: shared DQN scorer 96->64->64->1, replay 20,000, batch 64, target interval 500 and epsilon 0.05; PPO actor 96->64->1, critic 480->64->64->1, rollout 200, batch 100, epochs 4 and entropy 0.02. Matched ablations used the same actual architectures/settings. Future manifests now record corrected F2 defaults and per-model effective settings. Source snapshots preserve the exact learner code used by the original and stabilized runs. This metadata correction does not change any learned parameters or measured outcome.

## Validation and exact commands

**16 focused tests passed in 14.54s**. A targeted follow-up passed **3 checks in 5.33s** with no new learning: the completed-study no-retraining guard, actual serialized neural-settings provenance and Guide/playback AppTest after the team document was added. A further targeted occupancy check passed (1 test, no learning) using an actual permanent wall and a temporary blocker with identical public observations; all five actions remain selectable. It emitted only a local pytest-cache warning. The primary suite covered actual updates/save/load, old native checkpoint compatibility, seeded separation, public-only inputs and no mask, untouched reward, bad/cancelled checkpoint handling, recorded cap and playback, fair comparison/export, all saved case counts, new-only pages and a small ZIP fixture. No multi-gigabyte archive was rehashed or retested. Git path diff reports no archive, old experiment, starter or authoritative partial-module changes.

An independent artifact pass verified all 240 replay case hashes/IDs, steps, reward sums and Wait counts against the CSV and original reward. AppTest verifies the UI; no manual browser visual review is claimed. The root launcher imported successfully and served HTTP 200/`ok` at 127.0.0.1:8517. The smoke server was stopped. An initial Windows PowerShell web-parser error was corrected with a basic Python HTTP client.

From the Python application directory, using its installed environment:

```powershell
.\.venv\Scripts\python.exe -m robot_navigation.product_phase2 --run
# Matched controls, if reproducing individually (fresh IDs):
.\.venv\Scripts\python.exe -m robot_navigation.app_training --preset standard --models all --actions 10000 --seed 82111 --shaping none --q-stability off
.\.venv\Scripts\python.exe -m robot_navigation.app_training --preset standard --models all --actions 10000 --seed 82111 --shaping progress-v1 --q-stability off
# Stabilized Q-learning candidate:
.\.venv\Scripts\python.exe -m robot_navigation.app_training --preset standard --models q-learning --actions 30000 --seed 82111 --shaping none --q-stability target-v1
# Focused acceptance:
.\.venv\Scripts\python.exe -m pytest developer/tests/test_dynamic_app.py developer/tests/test_product_phase2.py -q
```

The delivered study was continued using `product_phase2 --run --resume runs/product-phase2/20261122T132126Z-41f7725b` to evaluate already completed models after the metadata correction, then `--finalize-study` to publish saved results after the path correction. This continues orchestration only; every learner started fresh and no optimizer/buffer/RNG resume is claimed.

From repository root, launch **`py -3 run_app.py`** or **Start Robot Navigation.cmd**, then open **http://127.0.0.1:8517**. Follow the existing Windows installation guide first. The local isolated worktree used the original compatible environment through `run_app.py --python <absolute existing python.exe>`; `PYTHONPATH` was explicitly set to this checkout for all CLI jobs.

Branch: **feature/product-phase2-training-and-demo-v1**, based on `9c04f3a8a80c591e352ec752c6dde2c6cfdb8955`. [Delivery branch commits](https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/commits/feature/product-phase2-training-and-demo-v1). Implementation commit: [4c1dce6b562cded3e4823701b10b566eeacdc8f2](https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/commit/4c1dce6b562cded3e4823701b10b566eeacdc8f2), independently verified on GitHub. This following documentation/verification receipt names that immutable implementation commit; runtime and model payloads are unchanged. Main, earlier branches, the dirty original workspace and archive remain preserved. No main merge, force-push, full Phase 5 study or Phase 6 was performed.

## Exact local launch from the preserved isolated worktree

```powershell
Set-Location 'C:\Users\User\Desktop\Autonomous_Robot_Navigation\autonomous-robot-navigation-rl\.runtime\newapp'
& 'C:\Users\User\Desktop\Autonomous_Robot_Navigation\autonomous-robot-navigation-rl\.venv\Scripts\python.exe' run_app.py --python 'C:\Users\User\Desktop\Autonomous_Robot_Navigation\autonomous-robot-navigation-rl\.venv\Scripts\python.exe'
```

The original dirty main worktree stays separate. On a normally installed checkout, `py -3 run_app.py` remains the launch command.
