# Development log — 6 November 2026

Implemented Section 12 in its own project directory with Codex assistance.
The existing root README, plan, attributes and Git files were left in place.
No energy files or `sources/` folder were present in this checkout at inspection.
Original-file SHA-256 verification is recorded at handoff.

Scope: fixed grid, deterministic manifests, random/BFS, Q-learning smoke pilot,
metrics/tests, local animated replay. SARSA/DQN belong to later milestones. PPO,
new maps, moving obstacles, continuous control, robots, tuning, voting and the
comparison dashboard were not implemented. No external AI API/cloud is used.

Python 3.14.10 was the installed interpreter. The isolated environment successfully
installed Gymnasium, NumPy, Matplotlib, Streamlit, PyYAML and pytest; exact versions
are captured in `requirements-lock.txt`. Python 3.14 compatibility was tested for
these milestone dependencies only; later SB3/PyTorch compatibility is unverified.
The sandbox blocked the first PyPI connection; the approved dependency install
then succeeded. The editable package needed setuptools installed explicitly.

Implementation choices: use a small built-in tested time-limit counter rather than
stacking wrappers, keeping goal/collision priority on action 200. Use Matplotlib
prefix rendering inside a Streamlit fragment; playback state never calls a policy.
Custom demonstrations execute frozen inference through the same rollout, then
become read-only replays. Checkpoints are inference-only snapshots; exact training
resume is unsupported. The smoke saves the final-budget model without selection.
Each run uses a unique directory and refuses to overwrite prior runs.

Before pilot: 30,000 training interactions, seed 7, fixed 30-scenario validation
monitoring every 10,000, complete validation at the end; no final-test execution.
No navigation-success target is set for this diagnostic pilot, and it is not a
main Part A comparison. Provisional viewer target: comfortable 2 actions/second
playback on this machine, with manual pause/seek responsive within about one second.
Actual outcomes and limitations are recorded in `milestone_results.md`.

Final verification: 16 tests passed; Gymnasium checker passed; every saved validation
trajectory was replay-validated. The run's original source snapshot is retained and
hash-verified. Visual inspection found overlapping renderer footer text; spacing was
fixed and the original trajectories re-rendered. UI import order now selects the
project-local Matplotlib cache before importing pyplot. No learning code changed.
Computer Use ended at its URL-confidence policy gate, leaving browser-visible
animation verification open. Original-file hashes matched at handoff.

Sources used for API contracts (consulted 6 November 2026):
- [Gymnasium custom environment](https://gymnasium.farama.org/introduction/create_custom_env/)
- [Farama step API](https://farama.org/Gymnasium-Terminated-Truncated-Step-API)
- [Streamlit fragments](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment)

Code was written for this project; no external navigation repository was copied.

## Milestone 2 — 6 November 2026

Added SARSA, a shared Q-learning/SARSA training loop, independent evaluation CLI,
algorithm-tagged checkpoints with legacy loading, periodic inference snapshots,
failure/interruption status, single-pass compressed evaluation trajectory export,
and algorithm-aware replay/custom demos. Preserved the original smoke bundle,
benchmark inputs and root files. No energy project or `sources/` was present here.

Recorded the development protocol before running three seeds per algorithm at
200k interactions plus one seed-7 repeat each. All eight runs completed. Mean
validation success: Q-learning 66.45%, SARSA 25.68%; repeated seeds reproduced
tables, training logs and replays exactly. All saved validation actions and 160
periodic checkpoints were audited. Final test remains sealed. Full results and
limitations are in `milestone2_results.md`; this is not the full Part A study.

23 tests and the Gymnasium checker pass. The inherited browser blocker was resolved
with an isolated local Chrome profile; the existing user browser was left alone.
Restarted the owned Streamlit server after its old import cache caused an ImportError.
Observed BFS and saved Q-learning start-to-goal replay in-browser. Fixed dark-mode
metric contrast and narrow-window clipping after inspection. Run source snapshots
preserve the training revision; post-run changes affect UI styling/layout and runner
output-name options. No additional dependencies, cloud calls or public publication.

## Milestone 3 — 6 November 2026

Added optional SB3 2.9.2 / CPU PyTorch 2.15.0 dependencies, fixed coordinate scaling,
vanilla DQN training/inference, checksum/encoding-aware model loading, and generic
inference-state checks covering neural weights, optimizer, replay and counters.
New tests verify actual SB3 vector/autoreset timeout storage and bootstrap masking,
fixture learning, inference RNG isolation, DQN replay/custom demos, checkpoint
selection and ranking, failure taxonomy, overwrite protection and failed status.
33 tests pass; Gymnasium and SB3 environment checks pass.

Recorded protocol before a 10k DQN pilot, reported throughput and weak validation
results, then froze the unchanged benchmark and five-seed/200k matrix. Completed
all 15 runs (3m training interactions) in 648.37 seconds, with no tuning or final-test
use. Mean complete-validation success at monitoring-selected checkpoints: Q-learning
65.24%, SARSA 21.94%, DQN 5.32%. Declared ranking shortlists Q-learning and SARSA.
Both monitoring and complete validation scopes, selected age versus full budget,
per-seed outcomes, negative findings, costs and limitations are documented.

Audited 13,920 learned validation episodes / 1,622,440 actions and 5,568 reference
episodes, plus all 300 periodic models, training counts, checkpoint choices and
source hashes. Generated learning curves, comparison plots, failure heatmaps,
route distributions and actual common-scenario/loop/detour records. Added missing
reference-line labels after visual inspection; audited analyzer bytes are archived
separately from the rendering correction. No experimental measurements changed.

The viewer now chooses `selected.zip`/`selected.npz` through manifest metadata,
retains legacy checkpoint support, and exposes a matching frozen DQN custom demo.
Restarted the owned local server to load the new imports. Browser-specific results
are recorded separately. Preserved 256 existing root/benchmark/earlier-artifact
files by SHA-256; no energy files or `sources/` were present here. All work remains
local and uncommitted pending the owner's next instruction.

Method/API references consulted during implementation:
- https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
- https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
- https://pytorch.org/get-started/locally/
- Installed SB3 `collect_rollouts`, `ReplayBuffer._get_samples`, and DQN target code.

Part B tuning/voting, the full dashboard and all explicitly deferred additions
remain for later milestones. Poor DQN/long-route results were not repaired by
changing rules, selecting on final test, or introducing planner assistance.

## Milestone 4 — 6 November 2026

Completed Part B under the predeclared bounded protocol: eight shared candidates
per shortlisted family and two development seeds, five fresh paired control/tuned
seeds, and a five-seed reward-only ablation per family. All 62 runs completed their
200,000-interaction budgets: 12.4 million training actions in 858.57 seconds.
Training-run storage was 266.19 MiB. No follow-on search or final-test execution
was performed. Candidate c07 won for both Q-learning and SARSA.

Matched mean success changed from 64.74% to 89.96% for Q-learning and 24.55% to
33.00% for SARSA. Q-learning met all predeclared mean targets; SARSA missed all
three, including its 2.03% collision rate against a 2% cap. Removing the ordinary
step cost reduced success to 32.95% and 19.70% respectively; that diagnostic
variant was not promoted. Reward returns remain version-specific.

Frozen Q-learning voting used predeclared seeds 101/211/307, majority action and
first-member priority for three-way ties. Compared with reference seed101, it
lost five goals (837 versus 842 of 928) while improving SPL by .013117. Voting
requires three Q arrays and measured 7.51 versus 1.66 microseconds per decision
on this machine. Member training costs are counted once; the wrapper adds none.

Replayed and audited 57,536 trial-validation episodes / 6,543,058 actions, 1,240
periodic checkpoints, and 928 voting episodes / 24,810 actions. Verified selected
training-prefix visitation, original-scenario relationships for reward variants,
selection rules, immutable inference, source/model hashes and all voting proposals.
The runner and audited analyzer are archived; a later plot-only label repair is
recorded separately. All five generated figures were visually inspected.

The minimal animated viewer supports matching reward variants and frozen voting,
with reward/member labels. A new inventory regression exposed a playback reset;
stable saved-run and per-run episode widget keys fixed it. This viewer-only repair
postdates training; training/configuration sources and saved snapshots are unchanged.
All 46 tests pass. The local server health check and read-only voting CLI check
pass. A fresh browser rendering check remains unverified because the dedicated
Chrome window was minimized; AppTest replay/control checks passed.

Preserved 815 root/benchmark/earlier-artifact files by SHA-256 with zero changes.
No energy project or sources directory was present in this checkout. Full traces,
visitation and periodic models remain local under the existing storage policy;
selected models, configurations and measured summaries remain reviewable.
Milestone 5 (integration and core robustness) is next. Final freezing/evaluation,
the full comparison dashboard and deferred additions remain for their planned
milestones. This work is local and uncommitted.

## Milestone 5 — 6 November 2026

Expanded the local Streamlit viewer into six views using saved evidence: overview,
navigation, learning, agent comparison, optimization/robustness and methodology.
Added optional prefix visits/planner reference, recorded action scores and vote
diagnostics, prefix event seeking, synchronized two-agent and early/selected-model
replay. Both panels share task/scenario/mode and action index, holding completed
agents at their terminal state. No policy trains or writes benchmark results in
the UI. Archived five declared 10k models separately from original run directories.

Recorded the robustness protocol before assigning labels and aggregating outcomes.
The six dimensions partition 928 validation pairs by distance, canonical reference
corridor, start wall proximity, start/goal quadrant and quadrant crossing. Reused
all five seed metrics from paired Part B controls/refined Q/SARSA and historical
Part A DQN, plus the fixed vote. Reconciled 24,128 metric records and 442 group/seed
rows, verified model/task hashes and retained per-seed paired group differences.
No new training or benchmark evaluation was needed. Saved interactive examples
used 1,416 separately counted display actions; the analysis took 3.78 seconds.

Refined Q-learning success is 98.60% short / 97.77% medium / 74.44% long, versus
98.22% / 78.51% / 22.69% for matched controls. Its corridor group improves 30.63 pp;
the no-reference-corridor group declines .37 pp. Refined SARSA remains weak on
long routes (4.69%). Group support and confounding by geometry/distance are explicit;
no unseen-state/map or causal robustness claim is made. No extra tuning followed.

The first rendering attempt failed on the unavailable Tk/Tcl runtime; retained
that partial study with failed status. Added explicit Agg rendering and completed
a separate canonical analysis with archived source/input hashes. Inspected all
three figures and prefix/collision renderer QA. Fixed an event selector default
that prevented seeking to frame zero. All 58 tests pass (33.84 seconds after the live control fix), including
new dashboard/geometry/terminal synchronization/missing-evidence checks.

The owner restored the dedicated local Chrome window. Restarted the owned server
to clear a stale milestone-4 import. Native browser checks exposed live buttons
that did not advance frames inside the timer-refreshed display. Moved playback
buttons/speed outside that display for both modes; automated tests and live
checks then passed. Observed BFS three-action success/9.97, actual control-policy
one-action collision/-5.01, and matched control/refined seed101 at indices 0/1/2/3.
The control holds its terminal cell while the refined policy advances to success.
Q scores, rewards and counters match. Live visual gate is now passed; no browser
frame-rate claim or claim of manually replaying every model is made.

Preservation rechecks 3,041 root, benchmark and earlier-artifact files by SHA-256;
no protected changes. No energy project or sources directory exists in this checkout.
Final-test policy executions remain zero. Dynamics noise and all larger additions
are deferred; milestone 6 will freeze systems and conduct final evaluation.
No new dependencies, commit, push or deployment.

## Milestone 6 — 6 November 2026

Closed the milestone-5 live browser gate before final work. Added explicit final
rollout/replay authorization for a separate digest-gated runner; generic training,
CLI and simulator split loaders remain sealed. The UI displays aggregate final
results only after successful audit. No environment/learning rule changed.

Wrote protocol before opening final cases. Fixed Q-learning seed101 as the selected
system and original-setting seed101 as the single matched control; froze all 35
learned checkpoints, one fixed vote, five random streams and BFS. Snapshot contains
184 source/model/config/data/dependency/protocol inputs, with digest
f536d8ab40dc461f7dd2f94cb31cda2ce9a612f13f19c4bedd7c431bf91f8c28.
No scientific source/model/analysis changes occurred during final evaluation.

Before freezing, 64 tests passed (34.47 seconds), plus Gymnasium and dependency
checks. A synthetic final pipeline tested serialization, audit, publication and
overwrite refusal without real final cases. Isolated benchmark preparation and
200k Q-learning reproduction matched the selected Q array and every validation
metric exactly in 8.76 seconds. An initial preparation comparison stopped before
training because in-memory integer keys were compared with persisted JSON string
keys; corrected it to compare persisted reports and retained that local attempt.
No configuration or model selection changed.

Completed 42 final records / 38,976 episodes / 3,456,940 actions in 109.68 seconds.
Replayed every action, score, state, reward and flag against frozen policies and
the shared environment; audit/analysis took 87.87 seconds. Frozen inputs and learned
inference state remained unchanged. BFS solves all 928 cases optimally.

Matched five-seed Q-learning success is 65.52% baseline → 90.43% refined; paired
gain 24.91 pp (SD 3.03). Collision rises .32 pp to .54%, retained as a negative
finding. SARSA improves to 32.44% but misses all targets; Part A DQN reaches 5.78%.
The predeclared single Q seed101 reaches 840 goals, 7 collisions, 81 timeouts, SPL
.867870. Voting reaches 848 goals and SPL .893175, but is not promoted using test.
No tuning, checkpoint change, seed removal or rerun followed these final outcomes.

Inspected four final figures. Five dashboard page checks passed with final data,
and the isolated native Chrome Overview visibly shows the audited final table and
predeclared system. Finished the concise technical report, per-seed CSV, historical
run matrix, release checklist and Windows commands. Historical records contain
87 positive-budget runs / 17.04m interactions; the separate reproduction adds 200k.
Repeated seeds/streams/cases are not treated as extra independent training seeds.

Packaging review found newline normalization could invalidate raw frozen hashes
after checkout. Scoped new attributes preserve the final snapshot bytes. Added a
post-evaluation packaging helper that restores the exact archived source/model/data
into a fresh .cache workspace and verifies all 184 files/dependencies. Preparation
only was executed; no extra final policy pass or scientific code change. The
helper can optionally rerun the unchanged archived runner as repeatability evidence.
The frozen scientific source and original evaluation remain untouched.

Preserved 6,763 earlier root/data/artifact files, including ignored local periodic
models/traces. No energy project or sources directory was present. Known limits
remain familiar-map transfer, residual failures, poor SARSA/DQN, small seed counts,
one ensemble, inference-only snapshots and no clean network dependency installation
or cross-hardware verification. Deferred scope remains unchanged. No automatic
commit, push, share link or deployment. All six core milestones are complete.

## Owner-authorized PPO extension — 8 November 2026

The owner required four trained RL algorithms and selected PPO. Implemented a
categorical actor/value baseline with normalized coordinate inputs, fixed-rollout
budgets, post-update monitoring, checksum-validated ZIP models and probability/
state-value diagnostics. Windows Application Control initially blocked imports;
the owner changed the setting manually. The existing environment then loaded
PyTorch/pandas/SB3 successfully; no security setting was changed by the agent.

Registered a fixed single configuration before an independent seed-101 10k pilot.
The five main seeds [7,19,43,67,91] each completed exactly 200k interactions,
with no tuning or poor-seed replacement. Matrix wall time was 654.21 seconds.
Mean validation success was 15.22%, sample seed SD 13.46 percentage points.
Audited 241,569 validation actions, 100 monitoring checkpoints and all 128
registered historical comparison inputs. All 90 tests passed at that checkpoint.
PPO remains outside the old frozen final evaluation; its comparison is exploratory.
The owner requested a local commit, recorded as `3848049`.

## Plan reconciliation and saved-evidence integration — 8 November 2026

The owner authorized documentation reconciliation, complete PPO route/cost and
robustness views, fresh installation checks and another live UI check. Updated
the plan status and report addendum, retaining historical core results and dates.
Added `four-rl-analysis-v1` rather than changing old runs/studies. Reused the
unchanged six milestone-5 geometry labels before aggregating saved outcomes.
Reconciled 18,560 episode metrics and 340 group/seed rows across 20 baseline runs;
104 analysis inputs passed hash checks. The original success means are unchanged.
Analysis adds zero training, benchmark evaluations or final-test policy executions.

Dashboard additions show conditional successful route means, contributing seed/
success counts, recorded training/validation times, 100-query decision latency,
checkpoint file sizes, sample seed SD and CSV downloads. Four-family robustness
now covers distance, corridor, wall proximity, start/goal quadrant and region
crossing. Zero-success route means stay undefined. Costs are descriptive separate
run measurements; group effects are confounded with distance/geometry.

Created a fresh isolated Python 3.14.10 environment with no system site packages.
Initial restricted installation failed to resolve the first package; a retry with
approved explicit PyPI access installed all 58 unchanged lock entries using cached
wheels. Editable package installation, pip consistency, neural/UI imports and
Gymnasium checks passed. All 94 tests passed in 41.54 seconds in this environment.
Exact frozen archive preparation passed without final-policy execution. This is
a same-machine fresh-environment check, not cold-cache or cross-hardware evidence.

Selected historical training traces cannot be recovered from episode CSVs;
the optional neural representation comparison remains unperformed. Neither was
invented or silently added as new training. Current visual-check status and
remaining research are tracked in `docs/plan_closure_status.md`.

Preservation verified all 7,736 pre-existing artifact/data files unchanged and all
184 frozen inputs against their seal. The new local preview runs at port 8518
from the fresh environment. The renewed in-app browser attempt returned
`The browser access host is unavailable`; browser inventory exposed no connected
browser. Live verification therefore remains unconfirmed. No remote publication,
new model selection or additional research training was performed.

## Live verification and playback synchronization — 8 November 2026

The owner opened the local preview and left its tab active, resolving the earlier
native Chrome URL ambiguity. Four-family comparison/cost tables and PPO distance/
corridor groups were visually inspected. The live replay exposed a bug absent
from AppTest: delayed timeline events rewound index 1 to 0 and paused Play, then
old event-log events could reinforce the bounce. Temporary callback logging
confirmed that sequence and was removed after diagnosis.

Introduced frame-specific timeline keys and shared active-widget seek guards for
single/matched playback. Duplicate frame events leave Play running; obsolete
timeline/event callbacks are ignored. Paused viewers disable interval reruns,
and terminal frames trigger one timer-free rebuild. A stale server import was
cleared by restarting the owned port-8518 server. All 96 tests passed in 44.06
seconds in the fresh environment, including delayed-event regressions for both
playback modes.

Native Chrome observed PPO seed91 reach action 5 then goal `[9,0]` at action 9,
return 9.91/SPL 1.000. Seeking paused at action 2, `[2,0]`/-.02; expanding
probabilities preserved it, and Reset removed the previous decision table at
frame 0. Matched PPO7 versus control Q-learning101 finished at shared index 3:
PPO success `[0,3]`/9.97, baseline held at its one-action collision `[0,0]`/-5.01.
Narrow toolbar labels were clipped; two-row controls/full-width speed and stacked
single map/telemetry were fixed and inspected at approximately 854 pixels.
No quantitative browser cadence, phone/touch or all-run manual replay claim is
made. Historical training traces, representation experiments and an independent
four-family final evaluation remain outside this verification work.

## Milestone 8 — controlled DQN inputs, 8 November 2026

The owner selected the representation study over neural hyperparameter tuning.
Registered a fixed comparison of four absolute normalized coordinates versus
those coordinates plus two goal offsets. The canonical environment, reward,
actions and scenario manifests stay unchanged. Six inputs retain absolute
obstacle context and add no new information; their first layer has 128 extra
online parameters (4,868 total versus 4,740), disclosed as a capacity difference.

Added encoding-aware DQN save/load/inference with old four-feature compatibility.
Bounded optional training traces observe actual actions without policy queries
or RNG draws and retain completed episodes or explicitly incomplete prefixes.
Tests establish transition parity, six-feature timeout bootstrapping and capture
that leaves seeded neural/tabular training unchanged. All 108 initial tests
passed. Each new study run captures its first three chronological episodes,
including failed warm-up behavior; these do not reconstruct historical training.
Record-time replay checks use separate environments and add no learner training
or benchmark episodes.

Both 10k pilots (seed1701) passed, including chronological ledger/trace checks.
Their measured main estimate was 1,831.51 seconds. With settings unchanged, trained
both inputs on fresh paired seeds [101,211,307,401,503] at exactly 200k actions
each. All ten runs completed: 2m main plus 20k pilot training interactions. Main
wall time was 1,938.53 seconds, below 2,400; all individual runs were below 300
seconds and artifacts used 53.36 MiB, below 1 GiB. Source/configuration remained
fixed throughout; all 7,746 historical artifact/data files retained their hashes.

Validation success averaged 5.97% control versus 6.51% relative; paired change
+0.54 pp with sample paired-seed SD 2.88 pp. Two seeds improved, two worsened,
one tied. Collision mean fell from 36.38% to 24.40%, while timeout mean increased
from 57.65% to 69.09%. This is mixed evidence, with no consistent improvement and
both families still weak. No new controller or tuned DQN claim is promoted.

The full audit replayed 1,187,423 validation actions and 78 captured main training
actions, verified all 200 periodic models and all source/checkpoint/selection
rules, and confirmed inference state unchanged. Six existing geometry dimensions
are reported for both treatments. Inspected both scientific figures. After
publication, all 110 tests passed in 39.40 seconds, including the real six-feature
pilot UI and separate representation/four-family dashboard panels. The old frozen
archive's 184 inputs still match its seal; final-test policy executions are zero.
Live verification status is recorded separately in `docs/milestone8_verification.json`.

## 9 November 2026 preliminary neural tuning

The owner accepted the preceding dashboard visuals and authorized the next
milestone. Chose a bounded M9 DQN/PPO study with four configurations per family,
one development screening seed and three fresh matched confirmation seeds. This
uses the original plan's preliminary-study allowance to bound local CPU cost.
Retained absolute four-coordinate inputs, [64,64] networks and the original task.

An initial startup investigation interrupted two pilots after 10k training steps
each. Preserved them under neural-tuning-v1. A fresh attempt with pilot seed 7103
retained the predeclared screen candidates and main seeds. Its measured pilot
projection fit the limits. All 22 fresh runs completed 3.22m training steps in
1,073.28 seconds, including audit. The interrupted setup adds another 20k steps.

Both searches selected c01 before confirmation. Across seeds 7307/7411/7523,
success increased from 5.60% to 7.11% for DQN and 16.99% to 24.35% for PPO.
Every paired seed improved success, but both families missed all three target
means. DQN retains roughly 70% timeouts, and selected PPO exceeds 62% collisions.
Reported all poor outcomes, costs, selection ages and preliminary scope.

Audit replayed 1,751,597 validation actions and 3,464 actual recorded training
actions, checked all 322 periodic checkpoint checksums and all 22 source snapshots,
and verified 9,371 historical artifact/data files. Independent checks also
verified 8,816 original protected files and 184 sealed archived inputs. There were
zero new final-test policy executions. All 148 tests passed in 40.45 seconds after
the figure-label refinement. Added the separate Neural tuning page and matched
replay buttons, tested through the Streamlit harness. Prior dashboard visual
acceptance is owner-verified. The new research page has no fresh native-browser
inspection. Both saved scientific figures were visually inspected.

Codex assisted implementation, execution and analysis. The detailed protocol,
actual results and remaining limits are recorded in docs/milestone9_protocol.md
and docs/milestone9_results.md.

## Five-seed follow-up and native dashboard checks

The owner requested both extensions on 9 November 2026. Declared seeds 7607 and
7717 before training, with frozen c01 settings, paired baselines and exact 200k
budgets. Kept the original three-seed evidence and disclosed the staged design.

The first attempt completed four DQN additions. PPO baseline seed 7607 failed
on an in-place manifest write at 50k with Windows OSError 22. Its peer was stopped
after 40k. Preserved the failed attempt and reused completed DQN rows. Repaired
JSON I/O with atomic replacement and bounded transient-access retries, tested
format compatibility, permanent failure and concurrent readers, and verified
all other learning-engine core syntax/engine hashes. Restarted only the four
remaining PPO conditions from zero under recovery-cohort run IDs.

All eight successful additions total 1.6m training steps, plus at least 90k
interrupted PPO work. Recovery and combined audit took 474.10 seconds. Saved
parameters, optimizer state and counters match at shared failed/restarted PPO
prefixes. The five-seed success means are 6.88% versus 7.37% for DQN and 19.44%
versus 23.69% for PPO. The two additions alone averaged declines. Recorded all
regressions and missed targets without selecting a new release controller.

All 20 combined audits passed, covering 1,656,863 validation and 2,743 recorded
training actions, 400 periodic checkpoint checksums and 200 training recordings.
Preservation checks verified 11,563 pre-extension and 12,142 pre-recovery files,
and 184 sealed archived inputs. The older cross-session root README checksum
differs because of an earlier committed README change. It matches the starting
commit and was not edited by this task.

Native Chrome checks passed at 1536×912 and 854×722. Corrected and rechecked the
narrow model/route selector row. Verified the study switch, tables, images,
matched stepping/Play/seek/Reset and terminal holding. All 158 tests passed in
68.21 seconds. Codex assisted implementation, execution, diagnosis and reporting.
Detailed evidence is in docs/neural_confirmation_results.md and its verification.

## Report consolidation and evaluation handoff on 9 November 2026

The owner requested report consolidation, evaluation completion and a local
commit before unseen-map work. A coverage review confirmed 79 free cells and
all 6,162 possible distinct ordered start–goal pairs in the existing manifests.
The original test outcomes had already been inspected. After this finding, the
owner chose to defer independent four-family evaluation to the unseen-map study.

Rebuilt the main technical report from saved baseline, tabular, neural and final
summaries. Kept the original frozen final table separate from post-release
development validation. Integrated five-seed regressions, I/O failure/recovery
costs, actual learning recordings and native dashboard verification. Recorded
the next-stage evaluation requirements without creating new maps or models.

The closeout full suite passed all 158 tests in 76.39 seconds. This stage adds
zero canonical benchmark training or final-test policy executions. Isolated
regression fixtures exercise training and synthetic release logic. The requested
commit includes the verified five-seed extension, UI fix, JSON recovery,
consolidated report and explicit evaluation deferral.

The staging check exposed legacy text normalization and a shallow neural archive
attribute pattern. Added explicit CRLF checkout rules for Windows-generated
legacy JSON/CSV/log records and the historical Part B report, preserving the LF
map definition. Neural confirmation archives now use recursive raw-byte rules.
An isolated checkout matched all 7,104 indexed artifact/data files byte for byte.
The 12,603-file working artifact/data snapshot and original 184-file seal passed
preservation verification. These packaging rules retain the recorded data values
and model contents.

## Milestone 10 on 10 November 2026

The owner authorized new map-aware Q-learning, SARSA, DQN and PPO for unseen
layouts. Registered a fully observed 104-value input, the 12/4/10 layout split,
D4 duplicate rejection, distance-stratified routes and a one-seed 25,000-step
pilot per family before generation/training. Reused the core movement/reward
environment, tabular updates, metric aggregation, BFS, checkpoint ranking,
neural fingerprints and centered replay presentation.

All four runs completed exactly 100,000 training steps in total. The suite took
49.27 seconds. New-layout validation success was 2.60%, 2.60%, 4.17% and 5.21%
for Q-learning, SARSA, DQN and PPO. BFS was optimal on every validation route.
All learned families missed the targets. Tabular tables expose their exact-state
transfer limitation, and the neural pilot also learned weak controllers.

The first audit found SB3 timeout metadata appended to shared training-info
dictionaries. Core values matched. Added a compatibility reader that checks that
metadata against transition flags, fixed the future collector to copy info and
retained the original records/source snapshot. The pilot was trained once.
The audit checked 38,008 actions and unchanged inference fingerprints.

Added the Unseen maps results/replay page, including actual learning records for
all four families. Native browser checks verified DQN learning progression,
PPO start-to-goal replay and responsive layouts. All 174 tests passed in 51.35
seconds. All 12,605 protected files and 184 original sealed inputs passed
preservation checks. Final-layout policy execution count remains zero.

The main development budget, fresh seeds and any representation/architecture
comparison require registration in the next stage. Independent final evaluation
remains outstanding. Detailed costs, sources, results and limits are recorded in
docs/milestone10_protocol.md, docs/milestone10_results.md and
docs/milestone10_verification.json.

## Milestone 11 on 10 November 2026

The owner selected shared features for all four algorithms after the exact-state
pilot exposed its layout-transfer limitation. Registered 1,234 deterministic
map/goal features, normalized linear Q-learning/SARSA, fixed-extractor DQN/PPO,
four engineering pilots and a three-seed 200,000-step main matrix before training.
Workers executed frozen archived source with at most two CPU workers and one
Torch thread each. Exact budgets total 2.5 million transitions across sixteen
complete runs. There were no failed or retried attempts. Suite wall time was
1,255.86 seconds and study storage was approximately 298.13 MB.

Three-seed validation success averaged 18.92% Q-learning, 63.37% SARSA,
33.51% DQN and 73.96% PPO. All four mean collision rates were below 2%, while
every family missed the 80% success and .75 SPL targets. Timeouts dominate
failures. Cross-milestone differences also include budget, value function and
seeds, so this comparison does not isolate a representation effect.

The full audit checked 453,723 saved actions and 48 bounded actual learning
records. Models reload with matching actions/diagnostics, learned state remains
unchanged during inference and budgets/ledgers/checkpoint selection reconcile.
All 184 tests passed in 76.13 seconds. Dashboard tests covered all twelve main
runs in both replay modes and pause on study changes. Native checks verified
automatic learning playback, seeking/stepping, a selected-policy goal replay,
centered desktop presentation and responsive narrow controls without overflow.
The live environment heading was corrected after training. Separate executed
and verification source hashes retain that distinction.

All 12,772 protected pre-study files retain their hashes, and all 184 original
frozen inputs remain valid. Earlier evidence, the energy project and sources
remain preserved. Final-layout policy execution count is zero. Independent
evaluation on the ten reserved layouts is the next stage. See the registered
protocol, measured results and verification under docs/milestone11_*.
