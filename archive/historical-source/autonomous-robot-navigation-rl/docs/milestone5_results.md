# Milestone 5 — dashboard integration and core robustness

Milestone 4 was committed as `f963f37` before this work. Milestone 5 now implements
the six dashboard views and fixed-map robustness reanalysis. Automated checks and
the live browser success/failure/comparison gate pass. The owner restored the
dedicated local Chrome window, allowing native UI verification despite the browser
access host being unavailable. Final freeze/test evaluation remains milestone 6.

## Integrated dashboard

Run the existing local Streamlit command in the README and open
http://127.0.0.1:8517. The simulator remains the default, with all previous saved
replay/custom demonstration controls. The other views are Overview, Learning
analysis, Agent comparison, Optimization and robustness, and Methodology.
Missing evidence or incompatible checkpoints produce messages rather than
invented metrics. The interface never trains or opens the sealed final manifest.

New simulator features:

- Optional traveled trail, prefix cell-visit counts and labeled BFS planner route.
  BFS is a visual reference and is never fed to a learned policy.
- Actual recorded per-action Q scores, or voting member proposals/counts/tie rule.
  These describe the last displayed action's before-state; the initial frame has
  no displayed decision. Q scores are not probabilities. Unavailable scores remain
  unavailable rather than being fabricated.
- Prefix-only event log with frame seeking, preserving synchronized counters.
- Side-by-side frozen agents on the same canonical validation pair, task/reward
  signature and evaluation mode. One action-index clock advances both panels;
  completed agents stay at their actual terminal frame. Play/pause/step/reset,
  speed changes and seeking do not retrain or append benchmark results.
- Early (10k) versus monitoring-selected replay on the same validation scenario.
  Five declared exemplars cover Q-learning, SARSA, DQN and tuned tabular policies.
  Their early weights and metadata were copied into this new study; original
  run directories remain unchanged. Missing early models are explicitly reported.

Selected checkpoint age is shown separately from the full training budget.
Partial playback never displays completed SPL. Collision arrows retain the robot
in its actual cell. Training curves read saved logs; comparison tables show actual
success/collision/timeout/SPL, conditional successful route lengths, returns,
training/evaluation timing and seed variation. Voting memory/member training and
the shared 1,000-query latency measurements remain explicitly costed.

## Predefined robustness protocol and coverage

[The protocol](milestone5_protocol.md) was written before outcome aggregation.
The label file is saved before metrics are read. The six dimensions are distance,
canonical-reference corridor, start-wall proximity, start quadrant, goal quadrant
and same/different quadrant. Each dimension partitions all 928 validation pairs;
dimensions overlap and are not independent test sets. Corridor labels depend on
the existing deterministic BFS tie order and include endpoints. They do not assert
that the learned route or every shortest route traverses a corridor.

Coverage: five matched Part B control and refined seeds for Q-learning/SARSA,
five historical Part A DQN seeds, and one fixed voting ensemble. All **24,128 saved
episode metric records** reconcile with their saved run aggregates; this is 928
scenarios repeated across 26 system/seed records, not 24,128 independent tasks.
The output retains **442 group/seed rows** across 17 groups in six dimensions.
Every checkpoint passes task/checksum loading. Original-reward systems only are
included, keeping reward scales comparable. DQN is historical context, not a
seed-matched Part B control.

## Measured distance results

Mean complete-validation success across all declared seeds:

| System | Short (258 pairs) | Medium (350) | Long (320) |
|---|---:|---:|---:|
| Q-learning baseline | 98.22% | 78.51% | 22.69% |
| Q-learning refined | 98.60% | 97.77% | 74.44% |
| SARSA baseline | 68.99% | 13.54% | 0.75% |
| SARSA refined | 76.51% | 26.80% | 4.69% |
| DQN Part A | 14.19% | 3.03% | 0.69% |
| Q-learning voting (one ensemble) | 99.22% | 97.14% | 75.31% |

Q-learning long-route paired gain is **+51.75 percentage points**, sample SD
4.51 pp over five seed differences; medium gain is +19.26 pp (SD 3.83), short
+0.39 pp (SD 1.29). Gains are not uniform: in the 162-pair no-reference-corridor
group Q-learning declines **0.37 pp** (SD 1.61). That negative finding is retained.
Sample seed SD is not an episode confidence interval or significance test.

## Wall, corridor and region findings

| Group | Pairs per seed | Q baseline | Q refined | SARSA refined |
|---|---:|---:|---:|---:|
| Reference corridor | 766 | 57.68% | 88.30% | 22.85% |
| No reference corridor | 162 | 98.15% | 97.78% | 80.99% |
| Near-wall start (including boundary) | 813 | 64.33% | 89.35% | 32.15% |
| Open start | 115 | 67.65% | 94.26% | 38.96% |
| Different start/goal quadrants | 706 | 55.13% | 87.56% | 21.27% |
| Same quadrant | 222 | 95.32% | 97.57% | 70.27% |

The reference-corridor Q-learning paired gain is +30.63 pp, SD 2.96 pp.
Refined Q-learning start-quadrant success ranges 88.32–92.34%; goal-quadrant
success ranges 88.98–91.97%. SARSA start-quadrant success ranges 29.81–36.19% and
goal-quadrant success 31.12–35.73%. Support differs by quadrant and is saved for
each row. These descriptive associations are confounded with route length and
map geometry; they do not identify causal effects of walls or corridors.

Long routes, crossing regions and residual Q-learning timeouts remain development
weaknesses. SARSA and DQN remain poor on long routes. The single fixed vote does
not establish seed-robust ensemble improvement; milestone 4's matched comparison
still loses five goals versus predeclared single seed101, while improving SPL.
No new selection, reward tuning, fallback controller or extra training followed
these group findings. Unseen-map transfer and physical reliability are untested.

## Actual execution and verification

The successful analysis took **3.78 seconds** locally, with **zero new training
interactions and zero new benchmark evaluation interactions**. Frozen display
examples used 1,416 environment actions, separately excluded from the benchmark;
one existing one-action collision was also replay-verified. All display generation
uses the shared environment and checks that model inference state is unchanged.
The saved example suite includes early failures/timeouts and later outcomes.

The first analysis reached plot rendering but failed because Matplotlib selected
Tk and the local Tcl runtime was unavailable. Its partial evidence is preserved as
`milestone5-incomplete-render` with failed status. The successful run explicitly
uses Agg and remains a separate study. No experimental rule or outcome changed
to fix rendering.

**58 tests passed in 33.84 seconds**, after the live control fix. New tests cover geometry labels, group support
and counts, seed pairing, mismatched comparison rejection, terminal-frame holding,
early model loading, all five non-simulator views, absent evidence, overlay preservation, event
seeking, matched playback and pause/resume. Existing environment, tabular/DQN,
reward, voting and replay tests still pass. An event-default bug found during
testing was fixed: the event selector now defaults to the displayed frame, so
selecting frame zero actually seeks there.

All three analysis figures and prefix/collision renderer QA images were inspected.
They show readable labels, distinct planner/traveled paths, prefix counts and an
out-of-bounds collision arrow with the robot retained at its actual position.
The owner then restored the dedicated Chrome window. A server left running from
milestone 4 initially retained a stale module import; restarting the owned server
loaded the new dashboard. Live buttons inside the timer-refreshed display did not
advance playback. Keeping the Play/Pause/Next/Reset and speed controls outside
that refreshed display fixed the live interactions for both single and comparison
views. This browser issue was not reproduced by AppTest; the repaired app passed
the full suite and was checked again in the browser.

Observed a BFS success on `r0c0-r0c3`: position [0,0] to [0,3], three actions,
return 9.97, terminal step reward 9.99, SPL 1.0 and stopped playback. An actual
Q-learning control failure showed one Up attempt at [-1,0], robot retained at
[0,0], return -5.01, SPL 0 and a visible collision arrow. The matched control versus
refined seed101 comparison visibly advanced through shared indices 0/1/2/3:
the control held at its one-action collision while the refined policy moved
[0,0] → [0,1] → [0,2] → [0,3], ending at success/9.97. Recorded Q scores and panel
counters match the displayed states. These interactive rollouts remain excluded
from benchmark metrics. The visual gate is closed; no quantitative browser
frame-rate claim is made, and not every saved run was manually replayed.

Preservation checks cover **3,041 protected tracked files**: root files, canonical
benchmark data and every earlier artifact. No energy project or sources directory
was present in this checkout; none was created or changed. Final-test policy
executions remain **zero**. No public deployment, push or commit was performed.

Evidence: [summary](../artifacts/studies/milestone5-v1/summary.json),
[labels](../artifacts/studies/milestone5-v1/scenario_labels.json),
[per-seed groups](../artifacts/studies/milestone5-v1/groups.csv),
[paired group changes](../artifacts/studies/milestone5-v1/paired_groups.json),
[audit](../artifacts/studies/milestone5-v1/verification.json),
[input hashes](../artifacts/studies/milestone5-v1/inputs.json),
[early checkpoints](../artifacts/studies/milestone5-v1/early_checkpoints.json),
[UI checks](milestone5_ui_check.json), [verification](milestone5_verification.json)
and [preservation](milestone5_preservation.json).
