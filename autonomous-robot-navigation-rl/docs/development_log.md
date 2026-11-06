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
