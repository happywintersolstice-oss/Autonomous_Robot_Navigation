# CODEX — REPLACE THE CLASSIC APP WITH ONE NEW PARTIAL/DYNAMIC ROBOT NAVIGATION APP; ARCHIVE ALL HISTORY

**This brief supersedes `CODEX_ROBOT_NAVIGATION_COMPLETE_USABLE_APP.md` and its old backward-compatibility / two-mode requirements.** The user's final decision is: **remove the old/classic application from the active product, build the new partially observed dynamic-navigation application as the only app, and preserve ALL past training data and evidence in an accessible archive for reference.** Do not build a Classic/Dynamic toggle, legacy four-action compatibility layer, or old-model transfer mode.

## 0. One full delivery, ASAP

Repository: `happywintersolstice-oss/Autonomous_Robot_Navigation`.

Starting point (unless an even newer app branch is already pushed and contains work worth retaining): `feature/phase5h-fast-goal-conditioning-v1` at commit `0c0ba4b0583d7f03fabba901397a0f41f0aac452`. Create a new branch named `feature/new-dynamic-app-archive-history-v1`. Preserve `main` and all existing feature branches; **do not merge** without explicit user instruction.

**Deliver, not plan:** implement, archive, test, run a small working demo, document, commit and push in this single Codex task. Do not initiate another Phase 5I/5J research series, unbounded training, full legacy research regressions, or another permission cycle. Prioritize a usable end-to-end product over elaborate diagnostics.

### Acceptance summary

- The **only active UI** is a 10×10 **partial-observation navigation simulator** with static/permanent obstacles, temporary blockages, sensor/memory display, and action **Wait**.
- Four **real RL** families are usable: **Q-learning, SARSA, DQN, PPO**. Include a separately labeled observation-only replanner (non-RL reference) and optionally random policy.
- Users can **Run/Play/Pause/Step/Seek/Reset**, **Train Models**, **Compare**, and **Export CSV** in the same app.
- The **old 4-action coordinate-only application is gone from active UI/active code paths**. No Classic task mode or old four-action model option anywhere in the new app.
- **Every pre-existing historical training dataset/checkpoint/replay/report/experiment log and relevant reproduction metadata remains preserved**, located/indexed under an `archive/` system, with a straightforward command to export ZIP archive files. None are discarded simply because their old app is removed.
- The root Windows launcher still starts the **new** app at `http://127.0.0.1:8517` (reusing a launcher name is fine; retaining the old app is not).

## 1. Current implementation: reuse working components, retire the old UI

Inspect actual repository files briefly, then proceed:

**Keep/reuse for the new active app** (refactor shared utilities only as needed):

- `autonomous-robot-navigation-rl/src/robot_navigation/partial_dynamic_env.py`, `partial_contract.py`, `partial_sensor.py`, `partial_memory.py`, `partial_observation.py` — **authoritative new environment**.
- `partial_agents.py`, `partial_features.py`, `partial_features_scaled.py`, `partial_neural.py` — actual five-action Q-learning/SARSA/DQN/PPO and valid B0/B1 checkpoint loading.
- `partial_baselines.py` — observation-only `OnlinePlanner`, `AlwaysWait`, `RandomPolicy`.
- `partial_evaluation.py`, `partial_benchmark.py` and applicable `partial_training*` helpers — episode metrics, development data and training primitives.
- `core.GridMap`, `core.digest`, `core.read_json`, `core.write_json`, `core.ROOT` and other **shared utilities actually imported by partial modules**. Refactor to neutral modules if that helps; **do not delete a shared dependency just because it resides in `core.py`**.

**Retire from active application** after backing up historical assets and checking dependencies:

- Old 4-action `core.NavigationEnv`, old tabular `agents.py` / four-action checkpoint menus, `episodes.py` legacy replay views, `navigation_demo.py` classic saved-map UI, static-only `simulator_ui.py`/`dashboard.py` pages when not needed by the new interface, historic old `training.py` entrypoints, original benchmark-driven default screens.
- Classic Demo/Results/Guide pages, old voting/BFS classic comparisons, original-map selection, four-action transfer adapters, and obsolete old-root app instructions. **Do not leave them reachable as a secondary mode or hidden sidebar.**
- **Dependency audit first**: many partial modules import old `core`/`release`/utility symbols. Keep or relocate the required generic symbols instead of blindly deleting modules or breaking imports. Old code may remain **in a read-only archive for provenance**, but not imported by normal app startup/training.

The new `app/streamlit_app.py` is rewritten or simplified to exclusively render the partial/dynamic product. `Start Robot Navigation.cmd` and `run_app.py` launch that app, not the classic one.

## 2. Archive historical training/research content BEFORE deleting legacy application code

### 2.1 Scope: preserve ALL past data, not merely best models

Inventory **the actual Git tree, on-disk tracked files, Git LFS pointers/objects where available, and historical directories**. Include both historical **classic 4-action** research and all previous **partial/dynamic 5-action** work (Phase 5A through 5H), regardless of failed/inconclusive results:

- Original/static experiment input data, maps/scenarios, train/validation cohorts, original saved models (`.npz`, `.zip`, `.json`), training logs/curves/CSV, learning replays, final/frozen evaluation receipts and summaries.
- Partial/dynamic historical training cases, code/config/protocol versions needed to interpret them, original model checkpoints, source/test snapshots embedded in evidence, optimizer/replay/rollout traces, benchmark IDs, manifests, receipts, reports, and negative trials for **5A–5H**.
- Historic root `CODEX_PHASE*.md` briefs and other run-specific reference instructions, where these document provenance. Preserve exact origin commit/path and file hashes.
- Existing accepted/frozen scientific evidence, including inspected final/unseen cohorts, is **reference-only**, never relabeled as new unseen evaluation.
- Files already stored in historical evidence directories may be very large. **Preserve by move/reference instead of duplicating huge blob collections.** Never omit an old result because it failed a later gate.

### 2.2 Archive layout and safety

Use a clear `archive/` root, for example:

```
archive/
  README.md
  INDEX.csv
  manifests/
    legacy-classic.json
    partial-research.json
    integrity.json
    original-path-map.csv
  legacy-classic/
    ... historic legacy files grouped under their original relative paths ...
  partial-research/
    ... past Phase 5A–5H evidence or immutable references to its original retained paths ...
  historical-source/
    ... old application/reproduction source if needed for old-run interpretation ...
```

**The essential requirement is actual archive preservation.** Move obsolete **classic app assets and old training evidence** using `git mv` into `archive/legacy-classic/...` when practical; preserve their original inner relative hierarchy. Keep any old source needed for reproducibility in `archive/historical-source` without installing/running it as the new app. For existing large Phase 5A–5H **immutable research data** where moving paths would invalidate version-sealed evidence, it is acceptable to **leave byte-identical data in its existing versioned evidence location** and include it in `archive/partial-research` through a complete path/hash manifest plus the reproducible ZIP export below. Do NOT merely put a placeholder summary in the archive: every historical file must remain directly accessible at a recorded path.

- Before any change, generate a **pre-migration inventory** of original path, byte size, SHA-256, Git blob identity where applicable, category, and source commit/branch. Avoid hashing every large file repeatedly; perform a single manifest pass. Retain the manifest under `archive/manifests/`.
- After moves, verify every archived pre-existing training/evidence file exists and matches its original SHA-256; identify any LFS pointer vs missing LFS payload explicitly. **Any missing archive file fails delivery**. Do not invent missing data or silently omit too-large items.
- Old scientific results are immutable. **Never regenerate or overwrite an old training run, sealed receipt, checkpoint, replay, or final/test cohort** while archiving. Movement and index generation are allowed; changing content is not.
- Preserve a path map from original to archived/retained path. Update **active** docs/links/registries to the new path; do not mutate historical reports solely to repair their relative links. Explain historical relative links may need archived original directory context.
- Keep old history in Git and the `archive/` tree, not in the live model output directory. Do not delete past commits, rewrite Git history, force-push, squash away data, or run `git clean` on user files. If deletion of an obsolete app file is necessary, archive source first and verify its continued interpretability.

### 2.3 Provide REAL archive files on demand

Add a reproducible **ZIP export utility** (e.g. `python scripts/export_history_archives.py --output-dir <folder>`) which produces:

- `legacy-classic-history.zip`: original four-action study assets, training data, model checkpoints, logs/replays and source/protocol dependencies needed for reference.
- `partial-navigation-research-history.zip`: past five-action Phase 5A–5H training/evidence assets, including files retained at sealed original paths.
- `archive-inventory.json` (and/or CSV) with each included file path, size, SHA-256, archive category, source version, export status and ZIP checksums.

ZIP exports are **read-only copies**; no deletion/migration of originals during export. Ensure relative paths in ZIP are unambiguous and safe. Support streamed copying (do not load full archives into memory) and a `--verify` or equivalent command that validates inventories and ZIP contents. If ZIP exports are too large for the Git repository, **do not commit duplicate ZIP blobs**: retain indexed source files under Git and document the single-command local ZIP export. That still supplies actual user-downloadable archive files when requested. If archive export cannot complete because an LFS object is missing, fail explicitly with the missing file list and recovery instructions, not a false success.

### 2.4 Read-only reference access

Provide a compact `archive/README.md` or `ARCHIVE_GUIDE.md` explaining: the old app is retired; where original 4-action runs and Phase 5A–5H results live; what a training run contains; how to inspect metrics, checkpoints and logs; how to produce/open archive ZIPs; and that archived models may be **incompatible** with the new 409-observation/five-action task.

A **small read-only Archive/Research History page** or link in the new app may list historic runs and open local archived reports (no old simulation, no classic policy execution), but this should not delay working navigation. Archive files are reference material, **never loaded into the new scientific comparison unless a native five-action checkpoint passes strict compatibility validation**. In particular old four-action models are not converted or presented as new models.

## 3. Build the new partial/dynamic app as the ONLY runtime

Single application landing view: **Robot Simulator**, **Train Models**, **Compare & Export**, **Guide**. An optional simple **Archives** informational view is allowed, but **no Classic navigation mode**.

### A. Robot Simulator

- Select 10×10 preset map, custom legal start/goal, obstacle mode **Static** or **Temporary**, policy (Q-learning/SARSA/DQN/PPO/Observed-Map Replanner/optional Random), and training checkpoint.
- Provide >=3 deterministic scenarios: clear static path, maze/permanent-wall detour, and temporarily blocked route (ideally one that must wait and another with a detour).
- Generate an actual episode **once** via `PartialDynamicEnv`; save observations/actions/rewards/terminal flags/trusted human-only display frames. Animate via **Play / Pause / Step / Seek / Reset**, speed and visible robot/goal/walls/temporary blocker/trail. No repeated `env.step()` at each animation frame.
- Show Success/Collision/Timeout, reward, steps, Wait count, revisits/loops and robot-local memory. A viewer may toggle the **ground-truth scene** for explanation, clearly distinguished from what the RL agent perceives.
- Honest behavior: if an RL model fails or waits indefinitely up to the 200-step limit, show the failure. The reference replanner is a separate **non-RL** controller, not an invisible fallback.

### B. Train Models

- All **four actual native partial five-action RL algorithms** must be individually selectable and trainable on permitted **train-only** scenarios, and load/save genuine inference checkpoints with correct task/feature hashes. Reuse `partial_features_scaled.build_policy` and appropriate loaders (B0/B1 versioned), do not relabel the old classic Q tables.
- Provide `quick` (small bounded demo budget) and `standard` (user-initiated larger development budget) presets, seed/actions/model selectors, progress and honest resource requirements. **Do not launch standard training automatically or on a Streamlit rerun**. A CLI is mandatory, UI training is preferred if a safe background/subprocess lifecycle and progress can be implemented; otherwise provide copyable command from UI and refresh status.
- Commands, or equivalents documented precisely:

```powershell
cd autonomous-robot-navigation-rl
.\.venv\Scripts\python.exe -m robot_navigation.app_training --preset quick --models all
.\.venv\Scripts\python.exe -m robot_navigation.app_training --preset standard --models all
.\.venv\Scripts\python.exe -m robot_navigation.app_training --list-models
```

- Support bounded `--seed`, `--actions`, `--models` and `--output`. Fresh run IDs and immutable history; never overwrite archive data. Distinguish final inference-only save from exact-resume training snapshots; do not claim exact resume without optimizer/buffer/RNG restoration.
- Keep dependency installs workable on Windows; demonstrate CPU training. Newly trained models become immediately visible in the new app registry.

### C. Compare & Export

- Compare the **four native partial RL models** on identical registered development/manual scenarios/event seeds, with replanner and Random clearly marked as references.
- Report counts/denominators, success %, collision %, timeout %, steps, reward, Wait share, revisits, two-cell loops, obstacle recovery and static vs dynamic subsets. Preserve honest zero/NaN handling when there are no successes.
- Export genuine case-level CSV, compact comparison JSON manifest, and deterministic episode replays. No parameters updated during evaluation and no true-world geometry/events passed to policy.
- Never use existing final/unseen/held-out cases as new unseen evaluation. Archive historical results for reference only, separate from new development scores.

### D. Guide

- Explain sensor range two, four-direction occupancy-only sensing, the 409-value observation / 412-feature B1 transform, five actions, memory/stale walls, Wait vs detour, reward and the difference between the four RL models.
- Give a 3–5-minute teammate demonstration walkthrough and honest limitations. Explain why the old four-action app was replaced and how archived data can still be retrieved.

## 4. Immutable environment contracts (no demo cheats)

- 10×10 grid; **action IDs `0 Up, 1 Down, 2 Left, 3 Right, 4 Wait`**.
- Range-two cardinal-ray **sensor ordering N, S, E, W** (different from action IDs); policy sees only permitted current occupancy/memory and public goal information, not permanent-vs-temporary label, event schedules, unblock time, map truth or path oracle.
- 409-value observation; B1 feature extractor internally has 412 dimensions; maintain checkpoint feature/task version checks.
- Rewards: step `-0.01`, goal bonus `+10`, collision penalty `-5`, no extra timeout penalty, maximum 200 steps; a blocked attempted move is a terminal collision. Preserve terminated vs truncated and correct bootstrapping.
- Permanent walls and temporary obstructions are indistinguishable while occupied. Temporary blockers clear on their own externally scheduled event; agents do not know when. A planner can react only to observed occupancy and memory, never `diagnostic_snapshot()`.
- Evaluate in detached inference mode; diagnostic snapshots are **UI/evaluator only**, never passed to RL or replanner. Do not change action semantics, rewards or observations merely to inflate displayed scores.

## 5. Delivery architecture & effort control

Prefer a **thin active product layer** built around existing partial modules, e.g. `app_service.py`, `app_training.py`, `app_registry.py`, `dynamic_demo_ui.py`, and one new `app/streamlit_app.py`. Move only archived historical content; do not rewrite tested `PartialDynamicEnv` or neural algorithms without a concrete integration defect. Retain shared generic utility modules actively used by partial code. Avoid unnecessary abstraction, migration frameworks and huge telemetry.

**Implementation order:** (1) single archive inventory/move and integrity gate; (2) replace old UI/launcher active routes; (3) scenario+replay+reference demo; (4) native four-policy checkpoint selection/training; (5) comparison/CSV; (6) focused tests, documentation, commit/push. Do not stop halfway to issue another mini-brief.

For quick smoke/demo, use only a **small explicit training budget per algorithm**; production `standard` may be user-triggered later. Small training may yield poor performance and must not be represented as a strong research result. Reuse already existing **compatible five-action checkpoints** as read-only starting examples when the exact signature matches, including ones physically retained in the archive. Do not cherry-pick a winner based on final/unseen data.

## 6. Acceptance tests — targeted, not 479 historic regressions

Run focused checks for:

1. **Archive preservation:** inventory and verify all pre-existing historical training/evidence files after moves; construct/verify one small ZIP export fixture, then verify real export command can enumerate complete coverage. No missing hashes/archival omissions. The old app code and reference documents stay retrievable, but old routes are absent from runtime.
2. **Only the new UI:** Windows launcher/server starts; no Classic navigation, original coordinate-only model choices, old maps/results pages or legacy four-action transfer.
3. **Environment:** static, permanent obstacle and temporary blockage run with real occupancy-only observations and Wait; no hidden event schedule leakage.
4. **Replay controls:** Play/Pause/Step/Seek/Reset show actual stored transitions consistently; no frame-time environment stepping or model calls.
5. **Four policies:** tiny bounded native training smoke for Q-learning, SARSA, DQN, PPO; saved model loads, returns valid five-action inference and respects B0/B1 metadata. Missing/incompatible checkpoint produces a friendly error, never uses legacy Q tables.
6. **Non-RL replanner:** shows an observed detour/temporary clearing case without map oracle; collisions/failures are shown accurately.
7. **Fair comparison and CSV:** same scenario/event-seed set per model, exact case counts, stable export/schema, no training during evaluation.
8. **Old data not active:** archived historical results are visible/readable as references and not fed to fresh training or mixed with new results. Verify imports and app startup after retiring legacy modules.

Use a small `pytest` selection and Streamlit AppTest or a real local browser/server smoke. **Do not rerun 479+ legacy research tests or any old experiments**. If a common utility is intentionally changed, run only directly affected contract tests. Declare what was actually tested rather than claiming unperformed browser testing.

## 7. Documents / final response

Update root README, Windows installer guide, app Guide and launcher instructions to describe **only** the new partial/dynamic application. Replace old static app screenshots/menus/landing instructions, keeping archived historical guides in the archive. Clearly distinguish active code/docs, `archive/` historical reference, and new runs/checkpoints/results. Include a precise Windows PowerShell installation and launch sequence.

Create `USABLE_APP_DELIVERY.md`, concise and product-focused:

- Ready to launch / partially working (truthful).
- URL, exact double-click and CLI launch commands.
- How to use Simulator, Train Models, Compare & Export, Guide.
- Current available **native five-action** RL models, quick-run examples and known limitations; no model efficacy fabrication.
- `archive/` folders, pre/post integrity counts and any missing/LFS exceptions; ZIP export/verify command and where generated files land.
- Focused test outcomes, actual smoke actions and clear blockers/workarounds.
- Branch and commit SHA/link.

At the end **commit and push one new feature branch**, report usable commands and stop. Do not merge main automatically. Do not propose the next diagnosis; the user's goal is to use the application now.

## 8. Final instruction to Codex

**Build the new system as the ONLY app. Archive the past training/research data first. Remove old functionality from the active runtime. Keep all four RL methods working on the new task. Finish the playable demo, training, comparison, export, docs and targeted tests in one delivery. Push it and stop.**
