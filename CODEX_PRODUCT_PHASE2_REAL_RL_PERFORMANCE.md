# CODEX — PRODUCT PHASE 2: REAL RL TRAINING, EVALUATION AND DEMO

## Mission: implement and ship, not just analyze

Repository: `happywintersolstice-oss/Autonomous_Robot_Navigation`.
Base: latest `feature/new-dynamic-app-archive-history-v1` (reviewed commit `9c04f3a8a80c591e352ec752c6dde2c6cfdb8955`). Create `feature/product-phase2-training-and-demo-v1`; commit and push the finished implementation. Preserve `main`, all prior feature branches, the dirty original workspace, existing starter checkpoints, and the entire archive. No force-push or main merge.

The new-only application already includes a 10x10 partially observed robot, five actions, four real RL families, simulator/replay, training, comparison, Windows launcher, and read-only archival history. **Do not rebuild the UI or reintroduce Classic.** Focus on making trained RL policies actually work and showing an honest four-model comparison in the existing app. Complete this as **one substantial Codex delivery**, not a sequence of micro-diagnostics.

## Starting evidence

`USABLE_APP_DELIVERY.md` reports just **128 actions/model** on the starter run; evaluation over **four manual development cases**, not unseen/generalization:

| Controller | Success | Collision | Timeout |
|---|---:|---:|---:|
| Q-learning | 0/4 | 0/4 | 4/4 |
| SARSA | 1/4 | 2/4 | 1/4 |
| DQN | 0/4 | 2/4 | 2/4 |
| PPO | 0/4 | 4/4 | 0/4 |
| Observed-Map Replanner (non-RL) | 4/4 | 0/4 | 0/4 |

The starting models are **not adequately trained**, and high performance must not be asserted without real measurement.

## Immutable product/science constraints

- Only active runtime is new partially observed/dynamic app: **10x10, range-two cardinal occupancy sensor, robot memory, 409-value observation, actions Up/Down/Left/Right/Wait, 200-step cap**.
- A controller sees occupancy only: it **cannot distinguish a permanent wall from a temporary blocker**, know the clear time/duration, or access future events, hidden map, privileged diagnostics, or evaluator snapshots. Human truth view is separate.
- Keep authoritative environment, sensor, memory, observation, action, and official reward contracts unchanged; do not access archived final/unseen studies. Never tamper with historical evidence or scores.
- Maintain **genuine** independent Q-learning, SARSA, DQN and PPO. The observed-map replanner and Random are explicitly **non-RL references**, never invisible rescue/fallback policies. No pre-scripted actions mislabeled learned RL.
- Feature changes must be versioned with strict checkpoint signatures and compatible loading of *existing* native starter checkpoints; do not fake compatibility with classic four-action models.
- Training output uses fresh immutable `runs/` IDs. Inference checkpoints are **not** exact-resume training states. Cancelled/failed runs must never register usable checkpoints.

## Workstream A — make the four RL models learn

1. Inspect active implementations in `autonomous-robot-navigation-rl/src/robot_navigation/`: `app_training.py`, `app_scenarios.py`, `app_service.py`, `app_registry.py`, `partial_agents.py`, `partial_neural.py`, `partial_features_scaled.py`, `partial_dynamic_env.py`, and relevant tests. Work from the existing new app, not archived code.
2. Replace dependence on only eight nearly fixed training route pairs with a **deterministic seeded train-only curriculum**. Begin with short traversable start-goal pairs, then longer goals, turns, permanent walls, maze detours, temporary blocked corridors and alternate-route cases. Vary goals/directions and map layouts. Keep events hidden from the agent and train-only cases distinguishable by layout hashes.
3. Build a separate **development evaluation suite of at least 24 reachable cases** with **non-overlapping layout identities/hashes**, spanning static and temporary scenarios and a range of path distances. Preserve exact case IDs, seeds, hashes, condition types. Do not use historical frozen/held-out tests. Name the output *product development comparison*, not scientific unseen accuracy.
4. Investigate sparse successes, oscillation, futile Wait, collisions with sensed walls, goal conditioning and unstable learning. Improve actual trainability of each algorithm. Any richer permitted sensor/memory/relative-goal features must be versioned and strictly derived from the 409-value observation. Avoid leakage of shortest paths or future blockage states.
5. Tune exploration and stable updates thoughtfully. Linear Q-learning and SARSA must remain genuine TD learning; DQN must genuinely update replay/target networks; PPO must genuinely update its policy from rollouts. Make update counts and parameter changes testable. Do not misrepresent a heuristic planner as an RL model.
6. Keep the **official evaluation reward** unchanged. If using train-only curriculum shaping or auxiliary objectives, version/disclose the formulas and evaluate against the original task reward. Prefer an ablation against unshaped training to substantiate gains; do not hide a planner teacher or action mask. Show raw RL separately from any disclosed assistance.
7. Make training **explicit, bounded and CPU-feasible**. Start with approximately 10,000 actions per model; if poor, allow a small number of deliberate 20,000–50,000 action/model candidates, constrained by the existing 50,000/model and one-hour worker safeguards. Respect PPO rollout multiples and DQN train-frequency multiples; no giant unbounded sweeps. Never call a separately started run an exact resume.
8. Save **one selected best legitimately validated checkpoint per RL family** (plus other verified immutable candidates) with run ID, true action budget, seed, cases/hashes, algorithm, feature/task signature, SHA-256, actual optimizer update metrics and transparent selection criterion. Selection must be based on development results, not always newest timestamp.
9. Target significant improvements over the tiny starters, **but report failures honestly**: if a family remains weak within bounded computation, still ship its real model and surface actual unsuccessful episodes and concrete limitations.

## Workstream B — fair side-by-side evaluation

- Extend existing `Compare & Export` rather than replacing it. Evaluate the same 24+ development cases and recorded seeds/stop rules for all four RL models; show the observation-only replanner and optional Random as separate non-RL references.
- Report, **with numerators and denominators**: success, collision, timeout, average successful steps (N/A if no successes), reward, Wait fraction, revisits, loops and blockage recovery. Break out static vs dynamic and short vs longer cases. Do not conceal unsuccessful runs.
- Clearly identify training/dev layout separation; compare old 128-action starters vs current checkpoints with explanatory labels, not ambiguous overall claims.
- Export summary CSV, per-case CSV, JSON manifest, checkpoint IDs/hashes, training provenance, case hashes, and recorded replay ZIP. All outputs reproducible under fresh run IDs.

## Workstream C — useful existing app demonstration

- Preserve existing pages: **Robot Simulator / Train Models / Compare & Export / Guide / Archives**. Do not introduce a Classic toggle, accounts, deployment dependency, or unrelated features.
- Simulator: retain playback controls, all true outcomes and `Robot knows` vs `Human sees` distinction. Make four clear presets easy to choose: open route, permanent-wall detour, temporary blockage where waiting can work, and temporary blockage with a detour. Let users select each family’s best verified checkpoint or a specific checkpoint.
- Train Models: show clear Quick/Standard distinction, bounded per-model budgets, progress/refresh/cancellation, and truthful `Starter` vs `Trained` vs `Best on development` status. Never auto-train on Streamlit rerun.
- Compare: one obvious workflow to view a saved four-model comparison; optional explicit rerun. A readable plot/table and real replay/export are enough—no redesign.
- Guide: add concise teammate division for four members: (1) Q-learning, (2) SARSA, (3) DQN, (4) PPO. Explain the occupancy-only uncertainty, Wait, and why the reference planner is not another RL family.

## One bounded acceptance/verification pass

Run *focused new-product tests*, not archived scientific suites. Verify:

1. All four algorithms do real training updates and save/load 5-action policies compatible with the new observation, strict signatures and registered provenance.
2. No privileged map/event/duration/clear-time leakage to action selection or learning; genuine train/dev layout separation.
3. Four-way identical-case comparison with clear metrics, static/dynamic reporting, genuine replay and full export.
4. A permanent wall does not force infinite evaluation (cap applies); waiting and detouring remain possible without knowing when blockers clear.
5. No silent replanner fallback, no fabricated scores, and no registration of a cancelled/incomplete checkpoint.
6. Windows launcher and Streamlit startup still work; no Classic runtime. Check focused AppTest or HTTP smoke; if browser visual review is unavailable, state that plainly rather than blocking the full delivery.
7. At least one **substantive bounded training attempt for each of four families** and a saved measured development comparison. If a runtime cap prevents all runs, report partial completion candidly—do not invent checkpoints.
8. Archive/history remains unchanged; **do not rehash or re-test the multi-gigabyte archive**.

## Deliver all of these in the same Codex run

- Working source changes on `feature/product-phase2-training-and-demo-v1`, committed and pushed.
- New genuine models/checkpoints for all completed RL trainings and honest measured comparison artifacts with replays.
- Root `PRODUCT_PHASE2_RESULTS.md`: baseline/new table, denominators, true training actions, task and feature versions, train/dev split, selected checkpoint names/hashes, training/test commands and actual results, remaining limitations, commit SHA.
- `docs/team-demo-script.md`: a 3–5-minute interactive demonstration and fair four-person role division, adapted to the *actual measured results*.
- Small README/app-guide adjustments only for real workflow changes, retaining launch command `py -3 run_app.py` and Streamlit port 8517.

**Final Codex reply:** branch, commit SHA, what was implemented, best checkpoint/action count for each of Q-learning/SARSA/DQN/PPO, baseline versus new outcome table (with denominator), tests and actual run commands, launch command, and blockers. Do not initiate another tiny diagnostic phase; stop after shipping the complete usable phase.
