# Product Phase3: robustness, unfamiliar maps and checked final demo

Completed locally on **2026-11-23 (Asia/Kuala_Lumpur)**. Branch: **feature/product-phase3-robustness-and-final-demo-v1**. Base: `66e0b2126c33229bbf23622e96887b87a897f23b` (latest remote Phase2 head fetched and verified). Evaluated implementation commit: **`71d19a3deb5d42cd668868c317e96d8107aceb43`**. Suite/protocol freeze commit: **`4dc21aec5c095656934ce128b99a83cb1a3eeb1b`**, made before model work. Exact pushed results/code commit: **[`3b8ede785f4d833652ef9098e3430e138d1702d8`](https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/commit/3b8ede785f4d833652ef9098e3430e138d1702d8)**. The branch tip additionally records this documentation-only publication receipt and committed-blob checks; no model, evaluation or result was changed. No main merge or history rewrite.

Phase3 shipped four genuinely trained selected checkpoints, a new independent product test, public decision-history features, diverse curriculum, immutable development/final comparisons, failure replay access and four checked presentation routes. All eight predetermined training jobs completed; none was cancelled or failed. There was **one final held-out assessment**, **640 distinct executed episodes**, **26,160 environment actions**, and **zero training after final outcomes**. This is a product-only assessment, not the historic scientific final/unseen benchmark.

## Same held-out panel: Phase2 versus Phase3

Each row below uses the **same 64 ordered cases and environment evaluation seed 0**. Success, collision and timeout counts sum to 64. Wait fractions are weighted action totals, not averages of episode percentages. Successful-only Waits exclude every failed episode. These are new test outcomes; Phase2's older 24-case selection scores are not used as test baselines.

| Checkpoint cohort / family | Success | Collision | Timeout | Wait / all actions | Wait / successful actions |
|---|---|---|---|---|---|
| Phase2 q-learning | 48/64 (75.00%) | 0/64 | 16/64 | 2017/3803 (53.04%) | 110/603 (18.24%) |
| Phase3 q-learning | 59/64 (92.19%) | 0/64 | 5/64 | 209/2012 (10.39%) | 116/1012 (11.46%) |
| Phase2 sarsa | 52/64 (81.25%) | 0/64 | 12/64 | 67/3125 (2.14%) | 50/725 (6.90%) |
| Phase3 sarsa | 64/64 (100.00%) | 0/64 | 0/64 | 85/1066 (7.97%) | 85/1066 (7.97%) |
| Phase2 dqn | 38/64 (59.38%) | 0/64 | 26/64 | 4892/5684 (86.07%) | 138/484 (28.51%) |
| Phase3 dqn | 58/64 (90.62%) | 0/64 | 6/64 | 1161/2499 (46.46%) | 417/1299 (32.10%) |
| Phase2 ppo | 43/64 (67.19%) | 0/64 | 21/64 | 4262/4676 (91.15%) | 203/476 (42.65%) |
| Phase3 ppo | 64/64 (100.00%) | 0/64 | 0/64 | 113/1118 (10.11%) | 113/1118 (10.11%) |

Same-panel increases in successes are Q-learning **+11**, SARSA **+12**, DQN **+20**, PPO **+21** out of 64. These are descriptive paired-panel differences, not a statistical-superiority claim. In particular, SARSA's all-action Wait fraction rises from 2.14% to 7.97% while timeouts fall from 12 to 0 and two-cell loops fall from 2,079 to 16. Correct waiting and successful navigation matter more than minimizing Wait alone.

References are independently executed and remain explicitly non-RL:

| Reference | Success | Collision | Timeout | Wait / all actions | Wait / successful actions |
|---|---|---|---|---|---|
| Observed-Map Replanner | 58/64 | 0/64 | 6/64 | 93/1973 (4.71%) | 76/773 (9.83%) |
| Random reference | 1/64 | 63/64 | 0/64 | 48/204 (23.53%) | 0/2 (0.00%) |

The observed-map replanner also times out on six cases; it has no hidden complete-map access. It never substitutes for any RL model. Random uses a reproducible single RNG stream seeded 0 across the ordered panel; it is not restarted independently for every case. All environments use seed 0 and the same frozen per-case event definitions.

## Static, dynamic, distance and topology breakdown

There are **28 static**, **36 dynamic**, **32 short** (base path 2-7) and **32 long** (base path 8-40) cases. Dynamic/static strata are independent of short/long assignment within generators except the eight forced-bridge cases, which are all dynamic. The following triples are **success / collision / timeout**, with the stated denominator.

| Policy | Static | Dynamic | Short | Long |
|---|---|---|---|---|
| Phase3 q-learning | 27 / 0 / 1 (n=28) | 32 / 0 / 4 (n=36) | 32 / 0 / 0 (n=32) | 27 / 0 / 5 (n=32) |
| Phase3 sarsa | 28 / 0 / 0 (n=28) | 36 / 0 / 0 (n=36) | 32 / 0 / 0 (n=32) | 32 / 0 / 0 (n=32) |
| Phase3 dqn | 26 / 0 / 2 (n=28) | 32 / 0 / 4 (n=36) | 32 / 0 / 0 (n=32) | 26 / 0 / 6 (n=32) |
| Phase3 ppo | 28 / 0 / 0 (n=28) | 36 / 0 / 0 (n=36) | 32 / 0 / 0 (n=32) | 32 / 0 / 0 (n=32) |
| Phase2 q-learning | 20 / 0 / 8 (n=28) | 28 / 0 / 8 (n=36) | 29 / 0 / 3 (n=32) | 19 / 0 / 13 (n=32) |
| Phase2 sarsa | 23 / 0 / 5 (n=28) | 29 / 0 / 7 (n=36) | 31 / 0 / 1 (n=32) | 21 / 0 / 11 (n=32) |
| Phase2 dqn | 18 / 0 / 10 (n=28) | 20 / 0 / 16 (n=36) | 28 / 0 / 4 (n=32) | 10 / 0 / 22 (n=32) |
| Phase2 ppo | 19 / 0 / 9 (n=28) | 24 / 0 / 12 (n=36) | 30 / 0 / 2 (n=32) | 13 / 0 / 19 (n=32) |
| Observed-Map Replanner | 26 / 0 / 2 (n=28) | 32 / 0 / 4 (n=36) | 32 / 0 / 0 (n=32) | 26 / 0 / 6 (n=32) |
| Random reference | 1 / 27 / 0 (n=28) | 0 / 36 / 0 (n=36) | 1 / 31 / 0 (n=32) | 0 / 32 / 0 (n=32) |

Every Phase3 Q-learning and DQN failure was a long-route timeout. Q-learning failed 1/28 static and 4/36 dynamic cases; DQN failed 2/28 static and 4/36 dynamic. SARSA and PPO completed all strata in this one panel, which does not establish that either is universally reliable.

Topology results below use **eight cases per cell**, formatted **success / collision / timeout**. The eight names describe distinct generator designs, not rotations of one map.

| Topology (n=8) | Phase3 q-learning | Phase3 sarsa | Phase3 dqn | Phase3 ppo | Phase2 q-learning | Phase2 sarsa | Phase2 dqn | Phase2 ppo | Observed-Map Replanner | Random reference |
|---|---|---|---|---|---|---|---|---|---|---|
| irregular_room_chain | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 | 7/0/1 | 7/0/1 | 5/0/3 | 6/0/2 | 7/0/1 | 0/8/0 |
| serpentine_spurs | 6/0/2 | 8/0/0 | 6/0/2 | 8/0/0 | 3/0/5 | 3/0/5 | 3/0/5 | 3/0/5 | 5/0/3 | 0/8/0 |
| bilateral_deadends | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 | 7/0/1 | 8/0/0 | 6/0/2 | 4/0/4 | 8/0/0 | 0/8/0 |
| double_gate_ring | 8/0/0 | 8/0/0 | 7/0/1 | 8/0/0 | 7/0/1 | 7/0/1 | 4/0/4 | 5/0/3 | 8/0/0 | 0/8/0 |
| destination_enclosure | 7/0/1 | 8/0/0 | 8/0/0 | 8/0/0 | 5/0/3 | 7/0/1 | 5/0/3 | 7/0/1 | 8/0/0 | 1/7/0 |
| forced_bridge | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 | 4/0/4 | 8/0/0 | 8/0/0 | 0/8/0 |
| island_loops | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 | 7/0/1 | 7/0/1 | 8/0/0 | 0/8/0 |
| switchback_spine | 6/0/2 | 8/0/0 | 5/0/3 | 8/0/0 | 3/0/5 | 4/0/4 | 4/0/4 | 3/0/5 | 6/0/2 | 0/8/0 |

Full condition, distance, topology and outcome-stratified Wait/return/recovery/efficiency metrics are in the linked summary CSV/JSON; empty CSV fields and JSON null mean **N/A**, not zero.

## Steps, returns, efficiency and liveness diagnostics

Step statistics and mean path efficiency include **successful episodes only**, so differences partly reflect which routes each policy completed. Phase2 can have a higher successful-only efficiency because its harder routes failed. Efficiency is the per-success ratio of a post-hoc optimum action count to actual actions, averaged across successes; it is not a ratio of aggregate means.

| Policy | Successful n | Mean / median successful steps | Mean official return (all64) | Mean successful efficiency |
|---|---|---|---|---|
| Phase3 q-learning | 59 | 17.15 / 14.00 | 8.9044 | 0.8718 |
| Phase3 sarsa | 64 | 16.66 / 14.00 | 9.8334 | 0.8951 |
| Phase3 dqn | 58 | 22.40 / 15.00 | 8.6720 | 0.7759 |
| Phase3 ppo | 64 | 17.47 / 16.00 | 9.8253 | 0.8687 |
| Phase2 q-learning | 48 | 12.56 / 9.00 | 6.9058 | 0.9529 |
| Phase2 sarsa | 52 | 13.94 / 12.00 | 7.6367 | 0.9044 |
| Phase2 dqn | 38 | 12.74 / 12.00 | 5.0494 | 0.8511 |
| Phase2 ppo | 43 | 11.07 / 9.00 | 5.9881 | 0.9928 |
| Observed-Map Replanner | 58 | 13.33 / 11.50 | 8.7542 | 0.9289 |
| Random reference | 1 | 2.00 / 2.00 | -4.7975 | 1.0000 |

The optimum is a time-expanded BFS in true occupancy including Wait, computed **only during final post-hoc scoring**. Neither training, feature construction nor checkpoint ranking uses it. Event windows use occupancy at the action's current time, matching the authoritative environment.

| Policy | Wait at blocked approaching frontier | Repeated Wait with observed unvisited adjacent alternative | Wait with known-free memory route | Movement revisits | Two-cell loop windows | Recovery / observed opportunities |
|---|---|---|---|---|---|---|
| Phase3 q-learning | 162 | 0 | 0 | 1174 | 11 | 34/35 (97.14%) |
| Phase3 sarsa | 71 | 0 | 0 | 279 | 16 | 35/35 (100.00%) |
| Phase3 dqn | 364 | 193 | 6 | 728 | 48 | 34/35 (97.14%) |
| Phase3 ppo | 101 | 4 | 0 | 326 | 0 | 35/35 (100.00%) |
| Phase2 q-learning | 1064 | 1843 | 0 | 1303 | 1005 | 30/35 (85.71%) |
| Phase2 sarsa | 67 | 0 | 0 | 2527 | 2079 | 32/35 (91.43%) |
| Phase2 dqn | 2407 | 4708 | 0 | 431 | 289 | 23/34 (67.65%) |
| Phase2 ppo | 3309 | 4078 | 0 | 87 | 31 | 26/35 (74.29%) |
| Observed-Map Replanner | 79 | 0 | 0 | 1290 | 1 | 34/35 (97.14%) |
| Random reference | 10 | 0 | 4 | 30 | 0 | 0/33 (0.00%) |

These diagnostic categories overlap and are not an exhaustive partition of Waits. An observed safe adjacent cell does not guarantee a complete detour. A known-free route uses remembered occupancy that can be stale; it does not reveal blocker type or remaining duration. Recovery opportunities are event-specific observed opportunities, so denominators can differ between policies (a policy may finish or collide before detecting an event). A zero-opportunity recovery rate is N/A. No diagnostic BFS or privileged recovery data enters policy decisions or ranking.

DQN still uses **1,161 Waits / 2,499 actions**, including **417 / 1,299 successful actions**. Its static successful Wait fraction is **29.22%**: wasted waiting remains even when no temporary obstacles exist. Its 193 repeated Waits with an observed adjacent alternative are evidence of residual liveness problems. Q-learning's remaining failures also involve movement loops, not simply continuous Wait; zero flagged alternative-Waits does not mean optimal behavior.

## Selected checkpoints and actual learning

All Phase3 selected artifacts have public input **409**, internal agent state **431**, feature version **`partial-action-public-history8-v3`**, history version **`public-decision-history8-v1`**, and **520 features / 104 per action**. Each selected checkpoint has **40,000 actual fresh training actions**, seed **93511**. Checkpoints are inference artifacts; exact optimizer/buffer/RNG resume is not claimed.

| Family | Selected checkpoint ID | Candidate | Training objective | TD updates | Optimizer steps | Parameter L2 change | Checkpoint SHA256 |
|---|---|---|---|---|---|---|---|
| q-learning | 20261122T165606Z-ea2dd627/q-learning | B-public-history-liveness | liveness-v1 | 40000 | 0 | 9.349936 | 531a01f6dd09d27cfbaae7572be24568217e9c9685c2661ccffb6bb82bb04565 |
| sarsa | 20261122T165803Z-b0736b3d/sarsa | B-public-history-liveness | liveness-v1 | 40000 | 0 | 9.863243 | 8fda1dcd28fa6ef921899012a754fd40bb3eb089912a6f3b628276f109d4a8c1 |
| dqn | 20261122T165937Z-f7774052/dqn | B-public-history-liveness | liveness-v1 | 0 | 9875 | 11.186933 | 54b877b838f38e2acf44e5cccfa4747fd3fb8f53d5b6e49f814bd18d34a60a12 |
| ppo | 20261122T165456Z-9875f272/ppo | A-public-history-unshaped | none | 0 | 1600 | 6.882751 | 7c6d152a21f986fed96d823207e10eaef8134e466f2cd496e11086cc03aee4dd |

The first three families selected candidate B (liveness objective); PPO selected candidate A (unshaped), because its development unproductive-Wait count was lower. Every selected parameter fingerprint changed, and save/load, repeated inference reset and five-action signatures passed validation. The unchanged Phase2 selection still points to its original four F2 checkpoints (480 features), at 30k/10k/10k/30k actions respectively; original B0/B1 412-feature starters also remain loadable.

Why these algorithms and implementations: Q-learning provides a compact shared linear action-value learner with greedy bootstrapping; SARSA gives a genuine on-policy TD comparison using the sampled next behavior action; DQN adds neural approximation, replay and target updates; PPO adds an actor/critic learned through clipped rollout updates with entropy. All independently learn from experience. No policy uses a route teacher, a planner fallback, scripted paths or a hidden action mask. Every collision-causing action remains selectable.

F3 appends **eight past decision positions** (including repeats when waiting), **last-action one-hot**, and a clipped **consecutive-Wait counter** to public409. These 22 public-history values produce internal431. Inference constructs the same state as the training wrapper, resetting at episode boundaries. Action rows extend F2's observation-local/goal features with recent candidate visits, immediate reversal, last-action agreement, repeated Wait and public observed-alternative interactions. The existing range-two sensor, memory, 10x10 world, five actions, 200 cap and official rewards are unchanged. Occupied permanent walls and temporary blockers with the same history produce identical public/F3 inputs; no event IDs, future occupancy, clear schedule, diagnostic snapshot, true rows or omniscient path enters the learner.

The train-only objective **liveness-v1** is explicitly non-policy-invariant:

```
training reward = official reward
                + 0.10 * (old public Manhattan distance - new distance)
                - 0.04 * [movement revisits one of the previous4 decision positions]
                - 0.02 * [Wait streak >=3 and a currently observed free adjacent
                           cell unvisited among the previous8 positions exists]
```

Official evaluation remains -0.01/action, +10 on reaching the goal, -5 on a collision, and no extra timeout penalty. In particular, no hidden time-to-clear reward was added.

### Registered bounded jobs and development-only selection

The predetermined set was **two objectives x four families x 40,000 actions = 320,000 completed actions**. Both objectives use the same seed93511 and fresh initialization; these are not continued Phase2 runs. Individual jobs stayed below 50k actions and 3,600 seconds. Actual job wall time totals exclude later comparison/export time. A one-hour check and cancellation flag stop a job and exclude its incomplete checkpoints; zero-action mechanical cancellation/cap tests passed.

| Candidate | Family | Fresh run ID | Actions | Dev success | Dev long success | Dev alternative-Waits | TD updates | Optimizer steps | Seconds |
|---|---|---|---|---|---|---|---|---|---|
| A | q-learning | 20261122T164947Z-1db53f39 | 40000 | 32/32 | 16/16 | 0 | 40000 | 0 | 131.165 |
| A | sarsa | 20261122T165159Z-d1f1c34a | 40000 | 32/32 | 16/16 | 0 | 40000 | 0 | 94.077 |
| A | dqn | 20261122T165333Z-58804b7b | 40000 | 28/32 | 12/16 | 272 | 0 | 9875 | 79.907 |
| A | ppo | 20261122T165456Z-9875f272 | 40000 | 32/32 | 16/16 | 7 | 0 | 1600 | 69.169 |
| B | q-learning | 20261122T165606Z-ea2dd627 | 40000 | 32/32 | 16/16 | 0 | 40000 | 0 | 116.334 |
| B | sarsa | 20261122T165803Z-b0736b3d | 40000 | 32/32 | 16/16 | 0 | 40000 | 0 | 92.877 |
| B | dqn | 20261122T165937Z-f7774052 | 40000 | 32/32 | 16/16 | 14 | 0 | 9875 | 80.048 |
| B | ppo | 20261122T170058Z-57eb83d8 | 40000 | 32/32 | 16/16 | 9 | 0 | 1600 | 68.718 |

Total actual substantive learner wall time: **732.295 seconds**. All eight jobs completed, with no silent retry or unfinished attempt. Mechanical learner gate: **256 total actions** (64 per family), with eight DQN and eight PPO optimizer steps. Including its 20-action export gate and 227 nonlearning environment transitions in focused acceptance gives **503 mechanical transitions**, below the registered 512 cap. Manual demo checks and development/final inference are separately identifiable and are not model training. The corrected-assertion rerun made zero environment transitions.

Settings were frozen before model work: gamma .99; linear alpha .1, TD error clip5 and norm bound100; Q-learning evaluates online-greedy targets using a target weight copy refreshed every100 updates (400 refreshes/job); SARSA uses the actual sampled next action. Exploration decreases from1 to .08 over80% of the budget. DQN: learning starts500, replay20,000, batch64, frequency4, one gradient step, target interval250, learning rate .0003; actual9875 optimizer steps/job. PPO: rollout200, batch100, four epochs, entropy .015, learning rate .0003; actual1600 optimizer steps/job versus SB3's800 epoch counter. Neural shapes/effective serialized settings are in each manifest/checkpoint; these counters are not fabricated action proxies.

Curriculum: 288 new training layouts from comb yards, radial pods, parallel rails, three-room lines, tree carving and four-court stars. The first10% of actions uses paths<=4, the next20% paths<=9, and the remaining70% all stages. Goals/directions/distances and encounters vary; forced-Wait and alternate-route blockages occur without revealing their identity. Development:32 separate layouts from triangular yards, lollipop courts, pitchfork gates and diagonal rooms. Phase2 baselines scored Q29/32, SARSA30/32, DQN22/32, PPO26/32 on this new development panel.

The locked ranking is success count, long successes, fewer collisions, fewer repeated Waits with observed unvisited alternatives, higher mean official return, fewer successful steps, then smaller training budget. Only training/development evidence enters it. Candidate A development successes were32/32,32/32,28/32,32/32; candidate B achieved32/32 for each. Complete baseline and candidate case rows, not just winning summaries, are preserved.

## Holdout freeze, novelty and no-leakage record

- Version: **`product-phase3-holdout-v1`**.
- Suite content hash: **`b362a5f5ee845c1c672e2938ab0ff3c1af8784953b684360805ddcfdbc5349c3`**.
- Raw JSON SHA256: **`78d1338f9fba7ec708eeca9d83cde6d06e206f64deec2333c8975b13be92c9fb`** (sidecar and raw-byte Git attributes preserve it across checkouts).
- Selection lock SHA256: **`8b0eaf690f171ca1456453ccb26ad6d76f9c91401666789107fe5083180c6851`**.
- Selection locked at `2026-11-22T17:02:08.237817+00:00`, with **zero held-out evaluations**.
- Final started `2026-11-23T01:12:46.522791+08:00`; run count **1**, persisted before any test policy call in both study-local and holdout-version-global markers. Completed640 unique episodes /26,160 actions. Reuse of this holdout version is refused by the study runner.

The test was drawn first from eight distinct structural generators, in seed namespace93217; train/dev use93317/93417 with different topology families. Exact canonical occupancy overlap is zero against **all216 Phase2 train/dev layouts plus288 Phase3 training plus32 Phase3 development**. Canonicalization removes all eight rotations/reflections. Structural graph fingerprints (vertex/edge/degree census plus five-round Weisfeiler-Lehman histograms) also overlap zero across held-out versus every other split, and train/dev structure overlap is zero. Topology family overlap is zero. This is stronger than random tile hashes, but the fingerprint is not a complete graph-isomorphism proof.

Each base path is reachable and at most40 actions; initial temporary blockers clear within60, so a wait-until-clear plus base path is feasible below200. This is a construction/inclusion check, not a policy hint. All static walls remain permanent. Case IDs, layout/structural hashes, event seeds, exact events, directions and splits are persisted. Finite event timestamps are evaluator-only; elapsed public steps/Wait counters do not encode a known clearing clock. No historical frozen/final/unseen dataset was opened or evaluated, and no existing experiment influenced test difficulty.

Selection remained fixed before final; models, objectives and metrics were not changed after observing outcomes. Only two acceptance-test expectations were corrected afterward (short collision traces and clamped playback), plus documentation and saved-data audits. Any future model tuning needs a new independent final holdout version before claiming unseen performance. The interactive training/development tools cannot update the locked published result.

## Audited outcomes, failures and limits

- **Q-learning failure**: `Phase3 q-learning`, `p3-holdout-013` (serpentine_spurs); timeout, 200 actions, 12 Waits, official return -2.00. Replay: [episodes/00013.json.gz](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/heldout-comparison/episodes/00013.json.gz).
- **DQN failure**: `Phase3 dqn`, `p3-holdout-009` (serpentine_spurs); timeout, 200 actions, 115 Waits, official return -2.00. Replay: [episodes/00137.json.gz](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/heldout-comparison/episodes/00137.json.gz).
- **Real forced-Wait success**: `Phase3 sarsa`, `p3-holdout-009` (serpentine_spurs); success, 34 actions, 17 Waits, official return 9.66. Replay: [episodes/00073.json.gz](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/heldout-comparison/episodes/00073.json.gz).
- **Optional-detour success**: `Phase3 dqn`, `p3-holdout-004` (irregular_room_chain); success, 3 actions, 0 Waits, official return 9.97. Replay: [episodes/00132.json.gz](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/heldout-comparison/episodes/00132.json.gz).

Saved-transition checks confirm genuine 200-step truncation on failures, unchanged per-step official reward, Waits during a forced bottleneck, and a successful alternate route around an active optional blocker. The audit reads recorded truth only after evaluation, never re-executes held-out policies, and includes the last eight (or fewer for short episodes) decisions. Q-learning's serpentine failure ends in repeated left/right traversal; DQN's corresponding failure includes 115 Waits and movement between two nearby cells.

Only one training seed, two fixed candidate objectives and one64-case clustered generator panel were measured. No seed-replicated uncertainty or statistical superiority is claimed; 64/64 is not universal reliability. No full training/Phase6 was attempted. DQN still wastes time on static layouts and the manual permanent-detour scene takes199 actions/113 Waits. Q-learning also fails the manual winding maze and permanent-detour scene. Even a safe observed alternative can lead to another dead end, and the8-step decision-history window cannot solve arbitrary partial-observation planning. These weaknesses remain visible rather than being patched against the test.

## Application and checked four-person demonstration

The existing five pages and Streamlit design are retained; there is no Classic mode. Simulator explicitly lists Phase2 baselines, Phase3 selected and other verified native checkpoints, has the four new independent demo presets, viewer-only truth/memory, saved playback/seek/reset and a compact public-history readout. The Train page exposes representation, actual action counts, background progress/cancellation and integrity verification. Training is explicitly started; page loads never train.

Compare & Export selects **Phase3 product held-out**, **Phase3 development** or **Phase2 development**. Held-out scope is read-only; its UI has no Run comparison button. Saved CSV/JSON/ZIP and failure traces download directly, and a recorded failed route opens in Simulator without policy execution. Only development scopes can explicitly evaluate new selections. No old application is offered as a mode.

| Member | Family | Checked preset (Temporary, seed0) | Outcome | Actions | Waits |
|---|---|---|---|---|---|
| 1 | q-learning | Temporary bridge - Wait | success | 13 | 6 |
| 2 | sarsa | New static maze | success | 34 | 0 |
| 3 | dqn | Temporary gate - detour | success | 9 | 0 |
| 4 | ppo | Permanent dead end with detour | success | 28 | 0 |

The updated [four-person script](docs/team-demo-script.md) uses these measured trajectories and includes the honest Q-learning maze timeout (200 actions/19 Waits). The original Phase2 script was copied byte-for-byte to the new study before updating the active guide. Thirty-two completed demo checks plus the first saved replay from an interrupted reporting script are retained; that script failed while assembling its manifest, not while training. No model selection changed as a result.

## Focused acceptance and actual issues

No historic final/unseen tests, legacy scientific suite, mass regression or full archive rehash was run. The substantive focused suite covered structural split integrity, indistinguishable occupancy histories, all five actions, training/inference history alignment, actual updates and save-load hashes, old checkpoints, incomplete-run filtering, cancellation/hour guards, locked study reuse, Wait/timeout mechanics, all640 saved replays/CSV/ZIP, passive playback, Streamlit navigation/downloads/failure callback. Independent CSV recomputation validated every summary group.

Initial acceptance: **9 passed,2 failed in20.59s**. The failures were in the new tests: assuming every failure has eight decisions even if Random collided at step2; and expecting seeking to7 in a4-step replay to return step7 rather than correctly clamping to4. Corrected only those assertions. Targeted rerun: **2 passed,9 deselected in12.50s**; one non-fatal pytest cache-path warning was retained. Thus all11 focused checks passed across the substantive pass and targeted correction; this is not described as an11-pass single invocation. There are no unresolved functional test failures. Models, environment and final metrics were not edited or rerun.

From the application directory (an activated installed environment, or replace `python` with a compatible interpreter; set `PYTHONPATH=src` when necessary):

```powershell
python -m compileall -q src/robot_navigation developer/tests/test_product_phase3.py
python -m pytest developer/tests/test_product_phase3.py -q --basetemp=.runtime/phase3-acceptance --junitxml=.runtime/phase3-acceptance-junit.xml
python -m pytest developer/tests/test_product_phase3.py -q -k 'final_matched or saved_viewing' --basetemp=.runtime/phase3-corrected-assertions --junitxml=.runtime/phase3-corrected-assertions-junit.xml
```

Actual local interpreter was the existing sibling workspace `autonomous-robot-navigation-rl/.venv/Scripts/python.exe`, with PYTHONPATH pointing at this feature worktree's `src`; the shared editable install/environment was not repointed. Streamlit AppTest used the same interpreter with permission to operate its local runtime. AppTest visited Simulator, Train, Guide, Compare, checked all five navigation entries, saved held-out downloads and the failure-to-Simulator callback; it did not run old archived experiments. The root launcher served **HTTP200** both at `/` and `/_stcore/health` (body `ok`) on127.0.0.1:8517, then its own smoke-check process was stopped. **No manual visual browser validation was performed.**

The already-completed study commands, shown for provenance and deliberately guarded against reuse:

```powershell
python -m robot_navigation.phase3_study --study runs/product-phase3/20261122T163932Z-f85cb312 --run
python -m robot_navigation.phase3_study --study runs/product-phase3/20261122T163932Z-f85cb312 --final
```

## Evidence, preservation and launch

- [Registered protocol](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/protocol.json).
- [All candidate/development scores](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/development-scores.json).
- [Actual training ledger](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/training-ledger.json).
- [Locked selection](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/selection.json).
- [Selection lock](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/selection-lock.json).
- [Development comparison](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/development-comparison/manifest.json).
- [Final comparison JSON](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/heldout-comparison/manifest.json).
- [Case CSV](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/heldout-comparison/case-results.csv).
- [Summary CSV](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/heldout-comparison/summary.csv).
- [Replay ZIP](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/heldout-comparison/results-and-replays.zip).
- [Export SHA256 inventory](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/heldout-comparison/export-inventory.json).
- [Failures / public decision tails](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/heldout-comparison/failure-traces.json).
- [Post-hoc transition audit](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/transition-audit.json).
- [Independent summary recomputation](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/summary-recomputation.json).
- [Validation receipt](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/validation/acceptance.json).
- [Initial pytest output](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/validation/phase3-acceptance-output.txt).
- [Corrected assertions output](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/validation/phase3-corrected-assertions-output.txt).
- [HTTP smoke receipt](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/validation/phase3-http-health.json).
- [Path-level preservation receipt](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/validation/preservation.json).
- [Raw source snapshot / hashes](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/source-before-final/sha256.json).

Per-model training CSVs, policy metadata, weights, parameter fingerprints and update counts are under `autonomous-robot-navigation-rl/runs/training/<fresh-run-id>`. Publication pointer: `runs/phase3-publication.json`; Phase3 selection: `runs/phase3-selection.json`. Original Phase2 `runs/selection.json` is untouched. Source snapshots preserve actual Windows byte hashes separately from Git's LF-normalized `.py` source blobs, making raw training/evaluation hashes inspectable.

Path-level Git checks against the Phase2 base found **zero modifications** to archive/, existing artifacts, Phase2 runs/checkpoints/exports, Phase2 results report, original partial environment/sensor/memory/reward contract and F2 curriculum/features. No archives were rehashed. Historical Phase5A-H evidence remains at its existing preserved paths. The main checkout's unrelated uncommitted work was left alone. Updated active README/demo docs are accompanied by the preserved previous script; no old result was rewritten.

Normal Windows launch from repository root:

```powershell
py -3 run_app.py
```

Or double-click `Start Robot Navigation.cmd`, then open **http://127.0.0.1:8517**. Follow `docs/windows-install.md` on a fresh checkout. For this isolated feature worktree, the equivalent tested command is:

```powershell
& 'C:/Users/User/Desktop/Autonomous_Robot_Navigation/autonomous-robot-navigation-rl/.venv/Scripts/python.exe' run_app.py --python 'C:/Users/User/Desktop/Autonomous_Robot_Navigation/autonomous-robot-navigation-rl/.venv/Scripts/python.exe' --server.headless true
```

Phase3 is complete; no full training or Phase6 was started. Research limitations and model failures above remain unresolved by design after the frozen final assessment.

The [committed preservation receipt](autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/validation/committed-preservation.json) verifies protected paths and the selected checkpoint/holdout SHA256 values directly from committed Git blobs. The published results commit above includes every source/checkpoint/result/test/report; its following receipt commit adds only publication documentation and these blob checks.
