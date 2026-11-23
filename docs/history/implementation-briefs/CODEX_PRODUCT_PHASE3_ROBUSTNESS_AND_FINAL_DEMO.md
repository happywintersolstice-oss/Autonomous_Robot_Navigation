# CODEX — PRODUCT PHASE 3: ROBUST UNFAMILIAR-MAP NAVIGATION + FINAL USABLE DEMO

## Mission — deliver in one execution, not another series of diagnostics

Repository: `happywintersolstice-oss/Autonomous_Robot_Navigation`.

Base: latest remote `feature/product-phase2-training-and-demo-v1` (reviewed head `66e0b2126c33229bbf23622e96887b87a897f23b`; fetch and verify). Create and push **`feature/product-phase3-robustness-and-final-demo-v1`**. Do not merge to `main` or rewrite existing history without user direction.

**Implement, test, measure, document, commit and push the entire Phase 3.** The desired outcome is an app with four independently learned RL models that can navigate *unfamiliar* static and temporary-blockage layouts with fewer useless Wait actions, accompanied by an honest, reproducible held-out product evaluation and a ready-to-use four-member live demo. Prioritize working behavior and deliverables. Do not stop for micro-diagnostics or create another chain of Phase 3A/3B research gates.

## 0. Verified Phase 2 starting point

Phase 2 implemented four native five-action RL families, seeded 192-layout training curriculum, separate 24-layout development panel, selection registry, real checkpoints, 240 saved comparison episodes, Streamlit UI and a four-member script. Sources:

- `PRODUCT_PHASE2_RESULTS.md`
- `autonomous-robot-navigation-rl/runs/selection.json`
- `autonomous-robot-navigation-rl/runs/comparisons/20261122T133310Z-833d1797/manifest.json`
- `docs/team-demo-script.md`

Selected **development** successes (these layouts were used for checkpoint selection, so NOT unseen performance):

| RL family | Phase 2 actions for selected checkpoint | Development success | Wait share of ALL actions | Longer-route success |
|---|---:|---:|---:|---:|
| Q-learning | 30,000 | 23/24 | 52.7% | 10/11 |
| SARSA | 10,000 | 22/24 | 37.4% | 9/11 |
| DQN | 10,000 | 18/24 | 86.5% | 5/11 |
| PPO | 30,000 | 21/24 | 79.4% | 8/11 |

Non-RL observed-map replanner: 24/24 on the *development* panel. Its success **does not count as RL success**. Model failure was mostly timeouts, not collisions. DQN/PPO's high aggregate Wait rates are heavily influenced by their timed-out episodes; do not blindly optimize the average Wait rate without outcome/condition breakdowns.

## 1. Non-negotiable system and data constraints

1. **Only** the new partially observed dynamic navigation app. No Classic mode and no four-action inference paths. Keep all existing pages and normal Windows launcher (`Start Robot Navigation.cmd`, `py -3 run_app.py`, port `8517`).
2. Keep the original **10×10 grid, 409-value observation, cardinal range-two occupancy sensing, agent-local memory, five actions (Up/Down/Left/Right/Wait), 200-action cap, official reward/task contract, collision and goal termination**, unless an absolutely necessary and clearly versioned forward change is justified. Do not alter previously published experiment contracts retroactively.
3. The agent sees only permitted public observations. It **cannot distinguish a permanent wall from a temporary blockage** if they produce the same occupancy history and **cannot know a temporary blockage's remaining duration or `clear` schedule**. Never feed hidden `events`, true rows, `diagnostic_snapshot`, obstacle labels, future occupancy, omniscient paths or human-only scene into policy decisions, features, action masks, reward shaping, or model selection.
4. Preserve the actual **Q-learning, SARSA, DQN, PPO** algorithm identities. Policies must learn independently from training experience. No hidden observation-planner fallback, no scripted paths relabeled as RL, no selective substitution in comparisons. The observed-map replanner remains separately labeled **non-RL**.
5. Preserve every Phase 2 checkpoint, selection decision, result/export/replay, archived old experiment and sealed historic scientific cohort **byte-for-byte**. Do not open or evaluate historic frozen/final/unseen benchmark data. All new runs/checkpoints must have fresh, immutable IDs. No full archive rehash or legacy regressions.
6. Old native five-action checkpoints remain readable with their original feature signatures. If you add action-history or goal features, introduce a **new feature version** and strictly reject incompatible loads; never pretend new weights match the old schema. Checkpoints are **inference-only**, not exact optimizer resume snapshots.
7. Runs must be bounded and explicit: maximum 50,000 actions/model in one job, preserve one-hour worker cap, genuine update counters, cancel handling, seeds, parameter-change evidence, hashes and failure traces. No automatic training when a page loads.

## 2. Freeze a genuine NEW held-out product test BEFORE model work

Create a **new product-only held-out test suite**, never derived from historic sealed/final/unseen research assets. Use distinct deterministic generators/seed namespaces and **genuinely different layout structures**, not just changed coordinates or a random tile hash from the same template. For example: irregular room connections, winding corridors, dead ends, alternate gates, ring obstacles, obstacles surrounding a destination with a route around, and blockages on bottlenecks vs detour-available paths. Ensure 10×10, valid starts/goals and a physically reachable route under the declared static world/event protocol. Static permanent walls must not secretly disappear.

- At least **64 frozen test cases**, stratified by: static vs temporary, short vs long paths, permanent dead ends, forced Wait bottlenecks, optional detour under temporary blocking and different goal directions. Include cases where Wait is *correct* and cases where waiting forever at a permanent obstruction is *wrong*.
- Persist a versioned test manifest at e.g. `autonomous-robot-navigation-rl/data/product_phase3_holdout_v1.json` with case IDs, splits, seeds, shape/topology family, immutable hashes and a checksum. This is a **product held-out suite**, NOT an official scientific final or the historical sealed unseen benchmark. Allow evaluator to read truth only to score, not the policy.
- Add structural checks against all Phase 2 training/development cases and newly generated Phase 3 training/development cases. Record exact layout, topology family and hash overlaps (must be zero). Do not use these holdout cases to tune models, hyperparameters, case-selection difficulty or checkpoint ranking.
- **Evaluation protocol:** freeze suite and acceptance definitions first, then tune only with training + development data. Perform one final held-out evaluation of the selected checkpoint for each family. If final results expose weaknesses, **report them**, do not tune further against that suite and still call it unseen. If additional model work is truly indispensable, use new development cases and a **new versioned** holdout for a later final assessment.
- For deterministic block events, event timestamps are evaluator-only. Do not encode them into observation/agent features. Document that finite test events do not give the policy a clock to clearance.

## 3. Improve longer-horizon navigation and over-waiting using train-only evidence

Work primarily in the new product curriculum, feature/policy training and checkpoint selector, not authoritative environment/observation rules. Diagnose DQN/PPO long-route timeouts, Q-learning idle periods and SARSA two-cell loops on training/development only. Then implement one coherent improvement, not an endless hyperparameter hunt:

- Curriculum covering diverse distances, directions, structural topologies, revisiting known dead ends, U-turns, permanent bottlenecks with possible detours, temporary bottlenecks requiring Wait, and temporary obstacles where detouring is better. Vary start/goal placement. Balance successful navigation and obstacle encounters without revealing hidden obstacle type or clear time.
- Model only **observable history** relevant to liveness: latest step's action, consecutive Wait counter, recent position revisits, sensor observations and agent's memory if available or explicitly included in a new versioned agent state. Do not feed a hidden time-to-clear feature. Distinguish a necessary short Wait from 200-step paralysis using observation-history-based learning, not hard-coded ground-truth rules.
- Preserve policies' option to choose **all five actions**, even if a collision is possible; don't disguise hard action masks or planner fallbacks as improved learning. If adding learned exploration bonuses, auxiliary loss, timeout-aware penalties or permitted observation-only shaping, version/disclose the training objective; keep **official evaluation reward unchanged**. Do not use true shortest paths or future events in train-time shaping.
- Keep algorithm-specific improvements real: linear TD stability for Q-learning/SARSA; DQN replay/exploration/target updates; PPO entropy/rollout/optimization. Avoid a major rewrite unless necessary.
- Use at most a **small predetermined candidate set** per family. Compare Phase 2 checkpoints vs Phase 3 candidates on a *separate development panel*, select once with a documented tie-break rule emphasizing **success first**, then longer-route success, then reasonable Wait usage and official return. Train Phase 3 candidates for a meaningful bounded budget (up to **50k actions per model per run**; if a family needs separate jobs, ensure every job is freshly initialized or clearly labeled and do not claim exact resume). Keep compute within an explicit resource budget and report all unfinished attempts honestly.
- Do not promise all models will beat the observed-map replanner or obtain 90%+ unseen success. **Report real outcomes**, even if some Phase 2 model remains the selected candidate.

## 4. Evaluate the **right** behavior (not just an average success number)

For exactly identical held-out cases/seeds across all four selected RL checkpoints, Phase 2 reference checkpoints, and separately labeled observed-map replanner (optionally Random), report:

- Success / collision / timeout counts **with denominators** and success percentage; overall, static vs dynamic, short vs long, and each topology family.
- Mean and median steps **on successful episodes**, path efficiency relative to a post-hoc *evaluator-only* optimum if available (never policy input), mean official return, collisions and 200-step timeouts.
- Wait total and fraction on **all** episodes **AND** **successful-only** episodes, including Wait at known blocked-frontier vs Wait with an observable safe detour; diagnose useless waits post hoc without claiming the robot knew obstacle permanence.
- Revisit/two-cell-loop counts, successful recovery, blocked-frontier handling and last few state/action observations for failure replays.
- Show explicit **Phase 2 vs Phase 3** deltas on the same new held-out panel. Never compare Phase 3 test numbers against Phase 2's older 24-case *development* scores as if the test populations were interchangeable.
- Make test result metadata explicit: product holdout version/hash, code SHA, scenario IDs/seeds, feature version, selected checkpoint ID/hash, real training-action counts, inclusion criteria and run date. Provide CSV, JSON, replay ZIP, and enough detail for an independent reviewer to recompute rates. Undefined metrics = `N/A`, not 0.
- **Uncertainty**: no unwarranted statistical-superiority claims from a single 64-case panel. If useful, display uncertainty intervals or per-family variance/seed sensitivity; note what was and wasn't replicated.

## 5. Final application and 4-member presentation readiness

Keep the existing Streamlit design and pages, but complete the following workflows end to end:

- Simulator: choose **Phase 2 baseline**, **Phase 3 selected** (when truly new), or any verified checkpoint per algorithm; preserve playback, seek, step, rewind, memory vs human-only truth and failure transparency. Add compact presets demonstrating (a) new static maze, (b) permanent dead end with detour, (c) temporary blocked corridor requiring Wait, (d) temporary blockage where a detour is sensible. Presets are **demo/development**, not lifted from frozen Phase 3 holdout cases.
- Train Models: display selected family, real actions, feature/algorithm version, checkable checkpoint integrity and background status; no fake progress. Explicit standard/bounded training and safe cancellation continue to work.
- Compare & Export: select a saved **development** comparison or the published **Phase 3 product held-out** evaluation. Label test scope clearly. Show side-by-side Phase 2 vs Phase 3 and RL vs non-RL, downloadable evidence, and accessible failure replays. Do not automatically re-run the frozen held-out set at every page load.
- Guide: update a **3–5 minute four-person demo**: member 1 Q-learning, 2 SARSA, 3 DQN, 4 PPO, each showing a real verified trajectory and explaining the genuine family differences. Use scenes that actually work with the chosen checkpoints. Include one honestly failed route and explain why partial observation is hard. No invented demos or results.
- On standard Windows install double-click launcher should work; root `py -3 run_app.py` and `http://127.0.0.1:8517` remain valid. Keep README commands accurate.

## 6. One focused acceptance pass (do not loop through tiny phases)

After implementing, execute ONE substantive bounded end-to-end study, tests, smoke check and documentation update. Important checks:

1. Four families make actions from only the permitted 409-vector or a clearly versioned agent-local history built exclusively from that vector and past public action/reward. The exact same initial observation for a wall and a temporary blocker must not reveal their identities or clear times. No policy has truth/event access.
2. All five actions remain legal; waits, detours and 200-action timeouts are correctly recorded; no forced termination when the agent fails to progress.
3. Native model update counters, save-load stability, old Phase 2 checkpoint compatibility, feature/hash validation and completed/cancelled registry filtering work.
4. Data-split integrity: frozen Phase 3 test is distinct from all training and development by both hashes and topology design; zero test evaluation during selection; no historic sealed assets opened. Check selection logs and evaluation run count.
5. Same matched case IDs/seeds and true metrics/replays for all families. Audit at least one failure and one real Wait/detour success against recorded transitions. Verify 64+ held-out episodes per tested policy.
6. Streamlit AppTest of launch/main navigation/compare/export and simple HTTP health check. Avoid claiming manual visual browser validation if none was performed.
7. Preserve all existing `archive/`, Phase 2 runs, models, and research evidence unchanged; use path-level Git changes/checks instead of a multi-gigabyte rehash. Do **not** run historic scientific final/unseen/Phase 5 tests or mass regressions.

If a CPU/time/tool restriction prevents substantive new training, **still ship** the held-out suite, evaluate the existing four Phase 2 genuine models fairly once, improve honest demo presentation, and report the blocker. Never fabricate training, scores, tests or checkpoints.

## 7. Deliverables — all in this single Codex job

1. Committed, pushed **`feature/product-phase3-robustness-and-final-demo-v1`** with code + deterministic tests. Leave `main` untouched.
2. New versioned 64+-case **product-only held-out** suite and a clearly separate train/development split, with no historic sealed asset access.
3. Four verified selected RL checkpoints (new ones **only when actual training and selection warrant**; otherwise retain the genuine Phase 2 selection) and a bounded training/evaluation provenance record.
4. Immutable saved Phase 3 development and **single final held-out** evaluation: CSV, JSON manifest, actual replays and ZIP with hashes and counts.
5. `PRODUCT_PHASE3_RESULTS.md` at repo root containing a **same-panel Phase 2 vs Phase 3 held-out** table for all four models, static/dynamic/long-distance breakdown, success/timeout/collision/Wait metrics, exact run IDs/actions, failures, limitations, sources/commands and exact pushed commit.
6. Updated `docs/team-demo-script.md` with four real checked demos, plus README/Windows instructions only if necessary.
7. Exact focused tests and results, with any blocked items explicitly stated. No open-ended list of minor checks.

### Final Codex response — concise and verifiable

Return: **branch + pushed commit**; **what actually changed**; four selected checkpoint IDs/features/action budgets; **same-held-out-panel Phase 2 vs Phase 3 success/timeout/Wait table with denominators**; static/dynamic/long-route weaknesses; held-out version and no-leakage statement; passed/failed test commands; launch command; real limitations.

**Finish the product phase. Do not pause for a new approval cycle or start Phase 3B/3C unless a permission boundary genuinely requires user action.**
