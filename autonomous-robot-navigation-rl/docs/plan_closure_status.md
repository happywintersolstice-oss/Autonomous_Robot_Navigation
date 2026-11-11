# Plan closure and remaining scope

Updated 11 November 2026 for milestone 13. The owner selected saved-trajectory
failure diagnosis. The calculation verifies 8,064 cases and 908,298 recorded
actions across M11 development and M12 inspected-final cohorts. All 4,434
timeouts contain persistent cycles, with 96.59% alternating between two cells.
The dashboard adds composition/difficulty views, failure-location maps and
41 bounded links to existing replay events. All 202 tests, native checks and
14,181-file preservation checks passed. This explains saved failure patterns.
Training causes and a performance-improving intervention remain untested.
See [milestone 13 measured results](milestone13_results.md).

Updated 10 November 2026 for milestone 12. The independent unseen-map evaluation
is complete. Twelve frozen policies and four reference runs evaluated 7,680
episodes on ten held-out layouts. Three-seed final success averaged 16.81%
Q-learning, 62.08% SARSA, 29.10% DQN and 64.10% PPO. BFS was optimal on every
route. All 671,151 saved actions, 189 tests and native dashboard checks passed.
Every family missed the mean success/SPL targets. The ten final layouts are
now inspected evidence. Further model selection requires a fresh holdout.
See [milestone 12 measured results](milestone12_results.md).

Updated 10 November 2026 for milestone 11. The owner selected shared features
for all four algorithms. Four engineering pilots and twelve main runs completed
2.5 million training steps. Three-seed new-layout validation success averaged
18.92% Q-learning, 63.37% SARSA, 33.51% DQN and 73.96% PPO. The action audit,
184-test regression suite and native dashboard checks passed. Every family
missed the mean success and SPL targets. The next milestone is a frozen
independent evaluation on the ten reserved layouts.
See [milestone 11 measured results](milestone11_results.md).

Updated 10 November 2026 for milestone 10. The separate unseen-map benchmark
and four-family 25,000-step pilot are implemented and audited. New-layout
validation success is 2.60% Q-learning, 2.60% SARSA, 4.17% DQN and 5.21% PPO.
The next stage is the main development study and budget/seed freeze, followed by
independent final evaluation. The 10 final layouts remain reserved.
See [milestone 10 results](milestone10_results.md).

Updated 9 November 2026 after five-seed neural confirmation, native dashboard
checks and report consolidation. The owner chose to defer independent evaluation
to the unseen-map stage. Historical milestone records retain their original scope.

## Current UI work and next milestone

On 9 November 2026 the owner requested a dashboard recreation using the supplied
NAVRL Navigation Control screenshots. The current work covers the dark navy and
cyan workspace, left navigation, replay controls, map, state and reward panels.
Displayed readings retain their actual saved-run provenance and grid units.
On 10 November the owner prioritized the first reference, with the map centered
between performance/agent state and distance/reward panels. Navigation is
collapsed initially to give this layout the available screen width.

The unseen-map foundation, shared-feature development study, frozen independent
four-family evaluation and saved-trajectory diagnosis are complete. The current
research stage has measured final outcomes, quantified cycles and documented
performance limitations. Further improvement work needs a registered design
and fresh independent holdout.
The original setup requirements remain in
[the evaluation handoff](evaluation_boundary_and_handoff.md). UI work introduces
no additional benchmark training, layouts or final-test executions.

## Implemented and measured

- Post hoc failure diagnosis with cycle/collision definitions, complete source
  reconciliation, per-seed and difficulty tables, event-location maps and replay
  jumps for both development and inspected-final evidence.
- Frozen independent unseen-map evaluation with all twelve selected controllers,
  ten held-out layouts, BFS/three Random seeds, layout-cluster intervals,
  complete action audit and separate final-results/replay dashboard entry.
- Shared-feature unseen-map comparison with three main seeds per algorithm,
  exact 200,000-step budgets, retained pilots, saved-action verification and
  results/learning/selected-policy replay with training-seed selection.
- All six core milestones: versioned environment, manifests, references, tabular
  and DQN baselines, bounded refinement, reward ablation, voting, fixed-map
  robustness, animated dashboard, original frozen final evaluation and report.
- Owner-authorized fourth algorithm: PPO, five independent seeds with exactly
  200,000 training interactions each. Its separate exploratory validation study
  passed verification. PPO tuning was added separately in M9 below.
- Four-family comparison includes success/collision/timeout/SPL/return, conditional
  successful route lengths and counts, recorded training/validation runtime,
  decision latency and checkpoint file size, with seed means, sample SD and
  downloadable per-seed records.
- All six predeclared geometry dimensions are available for the four baseline
  families in Optimization and robustness. The new analysis reconciles 18,560
  saved episodes and 340 group/seed rows, using 104 unchanged input files. It
  performs zero policy evaluations or training and leaves original studies intact.
- Plan, README, report addendum, development log and release/UI notes distinguish
  current PPO support from the historical three-family frozen release.
- The subsequent owner-authorized milestone 8 completed the controlled DQN
  representation study, with five fresh paired seeds, chronological training
  traces and a passed full replay audit. Mean success was 5.97% absolute versus
  6.51% absolute plus relative, without consistent seed-level improvement.
  All 110 tests passed at the M8 release. See [milestone 8 results](milestone8_results.md).

Analysis protocol, actual costs and PPO group outcomes are in
[four-family analysis results](four_way_analysis_results.md). Primary evidence:
[verification](../artifacts/studies/four-rl-analysis-v1/verification.json),
[cost records](../artifacts/studies/four-rl-analysis-v1/costs.csv) and
[group records](../artifacts/studies/four-rl-analysis-v1/groups.csv).

## Earlier automated and setup verification

All **96 tests passed in 44.06 seconds** in a fresh isolated Python 3.14.10
environment with system site packages disabled. All 58 pinned distributions
installed and match the unchanged lock file; the project installed in editable
mode with build isolation disabled. `pip check`, PyTorch/pandas/SB3/Streamlit
imports and the Gymnasium checker passed. Exact frozen-release preparation also
passed from this fresh environment, without policy execution.

The initial restricted install could not resolve the first pinned requirement.
Retrying with approved explicit PyPI access succeeded; pip reused cached wheels.
This verifies a fresh environment on this Windows machine, not a cold-cache
download test or cross-machine/hardware portability. The environment remains
local and ignored under `.cache/fresh-setup-v1`.

Reproduce setup verification using a fresh environment name:

```powershell
python -m venv .cache/fresh-setup-repeat
.\.cache\fresh-setup-repeat\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements-lock.txt
.\.cache\fresh-setup-repeat\Scripts\python.exe -m pip install -e . --no-build-isolation --no-deps
.\.cache\fresh-setup-repeat\Scripts\python.exe -m pip check
.\.cache\fresh-setup-repeat\Scripts\python.exe -m robot_navigation.cli check
.\.cache\fresh-setup-repeat\Scripts\python.exe -m pytest -q
```

Preservation checks verified all 7,736 pre-existing artifact/data files unchanged
and all 184 original frozen inputs against their seal. The current analysis is a
new study; no older evidence was rewritten. See the
[machine-readable verification](plan_closure_verification.json).

## Earlier live visual verification

Passed in native Chrome after the owner opened the local preview tab. The in-app
browser host remains unavailable, but direct observation of the active local tab
worked. Inspected the four-family route/cost tables and distance/corridor results,
PPO probabilities, automatic success playback, timeline seeking, Reset, and
matched PPO/baseline terminal holding. Desktop and approximately 854-pixel-wide
layouts were inspected; phone/touch and quantitative frame-rate/input-latency
benchmarks were not performed. The preview is `http://127.0.0.1:8518`.

The live check exposed delayed slider events that rewound and paused animation;
old event-log callbacks could reinforce the reset. Timeline widgets now have
frame-specific identities and reject stale events; redundant seeks do not pause
Play. Old event-log events are also ignored, paused timers are disabled, and
terminal frames rebuild once with the timer stopped. The new tests cover delayed
events for both single and comparison playback. Narrow controls use a two-by-two
button layout with a full-width speed slider; the single map and telemetry stack.

Observed PPO seed91 on `r0c0-r9c0`: automatic progression through action 5 to
9-action success at `[9,0]`, return 9.91, SPL 1.000. Seeking to action 2 produced
`[2,0]`, return -.02 and paused playback; opening probabilities preserved that
frame. Reset returned to action 0 and removed the old decision table. PPO seed7
versus Q-learning control101 on `r0c0-r0c3` reached shared index 3: PPO success at
`[0,3]`/9.97, control held at its one-action collision `[0,0]`/-5.01. These are
display checks excluded from benchmark summaries, not new final evaluations.

## Remaining research and limits

1. **Historical training trajectories:** historical runs retain episode CSVs,
   visitation and full evaluation traces, but selected training episodes were
   not saved transition by transition. Those historical traces cannot be
   reconstructed from summaries. Milestone 8 now implements bounded prospective
   capture and retains 36 actual early training records across its pilots/main
   runs. Subsequent four-family demos add 40 staged recordings, and the verified
   M9 suite adds 220 staged recordings. Historical actions remain unavailable.
2. **Representation scope:** milestone 8 tested normalized versus absolute-plus-
   relative DQN inputs. It addresses that portion of RQ2 under one fixed setup,
   with no consistent improvement. Other representations and capacity-matched
   neural comparisons remain untested.
3. **Staged neural confirmation:** M9 screened four configurations per algorithm.
   Its follow-up now combines five matched seeds for the same c01 settings.
   Success averages 6.88% baseline versus 7.37% selected settings for DQN and
   19.44% versus 23.69% for PPO. The two added seeds averaged declines for both.
   Both remain weak and miss all declared performance targets.
4. **Independent four-family final evaluation:** PPO was added after the original
   holdout had been inspected. The extension is exploratory development validation.
   All 6,162 possible fixed-map pairs are already covered by existing manifests.
   The owner chose to defer independent evaluation to the unseen-map study.
   Milestones 10 and 11 establish the separate benchmark and development study.
   Milestone 12 completes the frozen independent evaluation on ten held-out maps.
   The original fixed-map final record retains its historical scope. See
   [the final unseen-map results](milestone12_results.md).
5. **Scope and portability:** unseen-map evaluation is complete. Dynamics noise,
   moving obstacles, continuous movement, sensors, ROS and hardware remain deferred.
   Bitwise repeatability across machines and exact training resume are unsupported.
   Performance misses and residual collisions/timeouts are reported limitations.

The earlier integration did not initiate new research. Milestones 8 and 9 were
authorized separately afterward. Neural tuning remains separate from the input
study and the original final evaluation.

## Historical three-seed milestone 9 completion

The bounded suite completed 22 runs and 3.22 million training steps in 17.89
minutes, within its declared limits. One interrupted setup attempt added 20,000
training steps and was preserved separately. All 9,371 pre-study artifact/data
files retained their hashes. Independent checks verified the original 8,816
protected files and 184 sealed archived inputs.

The audit replayed 1,751,597 validation actions and 3,464 actual training actions.
All 148 tests passed after the final figure adjustment. The dashboard has a
separate Neural tuning page with matched model comparisons. Prior dashboard
visuals were owner-verified. At that checkpoint, the new research page had passed
the application harness. The subsequent native check is recorded below.

See [M9 measured results and limits](milestone9_results.md) and
[the registered protocol](milestone9_protocol.md).

## Five-seed follow-up and native visual completion

The owner requested both the five-seed expansion and native dashboard checks.
Eight successful additions total 1.6m training steps. A Windows progress-file
write failure interrupted two PPO attempts after at least 90k additional steps.
Four completed DQN additions were reused verbatim, and only the remaining PPO
conditions were trained again under recovery run names. Atomic JSON replacement
and transient-access retries were tested. Saved PPO parameters/optimizer state
match across failed/restarted prefixes at 50k baseline and 40k selected settings.

All 20 combined confirmation runs passed audit. The verifier replayed 1,656,863
validation actions and 2,743 actual training actions, and checked 400 periodic
checkpoint checksums. All 11,563 pre-extension and 12,142 pre-recovery artifact/data
files retained their hashes. The 184 sealed archived inputs remain valid.

All 158 tests passed in 68.21 seconds. Native Chrome checks passed at 1536×912
and 854×722, including study switching, five-seed/added-seed tables, curves,
costs, stepping, automatic playback, seeking, Reset and terminal holding.
The narrow second-model/route row now stacks, fixing the observed squeezed
selector. Independent four-family final evaluation remains outstanding.

See [the follow-up results and recovery record](neural_confirmation_results.md)
and [the native/regression verification](neural_confirmation_verification.json).

## Report consolidation and evaluation decision

The main technical report now integrates four-family baseline validation,
tabular refinement, representation results, five-seed neural outcomes,
the original frozen final evaluation, native UI checks and failed-run costs.
Historical final evidence and later exploratory evidence have separate tables
and scopes. The added-seed regressions and missed targets remain explicit.

The boundary review confirmed 79 free cells and all 6,162 distinct ordered pairs
covered by the existing splits. The owner chose independent evaluation on new
unseen layouts as the next stage. This report/commit stage performs no new
canonical training or final-test policy executions.

The closeout regression run passed all 158 tests in 76.39 seconds. The report
and handoff are ready for the requested local commit alongside the verified
five-seed implementation, native-layout fix and JSON I/O recovery work.

The commit packaging check restored 7,104 indexed artifact/data files into an
isolated checkout and matched every file's recorded working bytes. Legacy
Windows record line endings and recursive neural archive rules are explicit.
All 12,603 current artifact/data files and the 184 sealed inputs passed the
working-tree preservation checks. Detailed results are in
[the closeout verification](report_closeout_verification.json).
