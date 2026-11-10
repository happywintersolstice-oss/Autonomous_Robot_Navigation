# Milestone 10 results. Unseen-map foundation and pilot

Completed 10 November 2026 under the
[registered protocol](milestone10_protocol.md). The owner chose new map-aware
versions of all four RL algorithms. This milestone establishes the benchmark,
trains a bounded pilot and adds saved results and replay to the dashboard.

## Benchmark and executed support

The separate `unseen-maps-v1` benchmark has 12 training, 4 validation and 10
reserved final layouts, with 48 distance-stratified routes per layout. Every
layout has 79 connected free cells. Duplicate rejection includes rotations,
reflections and the original fixed map. Training has 576 routes, validation 192
and reserved final 480. The preparation audit checks geometry and reachability.

All four learned families receive the same 104 inputs, consisting of scaled
robot/goal coordinates and the full binary obstacle map. Core movement, rewards,
collision termination and the 200-action cap are reused. Every algorithm ran
exactly 25,000 training transitions with seed 11003, giving 100,000 steps total.
Two monitoring opportunities used the same 24 development cases. All selected
checkpoints came from step 12,500 under the predeclared selection rule.

## Actual development validation

| Algorithm | Successes / 192 | Success | Collisions | Timeouts | SPL | Familiar-layout probe success |
|---|---:|---:|---:|---:|---:|---:|
| Q-learning | 5 | 2.60% | 187 | 0 | .0260 | 8.33% |
| SARSA | 5 | 2.60% | 187 | 0 | .0260 | 8.33% |
| DQN | 8 | 4.17% | 132 | 52 | .0417 | 2.78% |
| PPO | 10 | 5.21% | 175 | 7 | .0521 | 8.33% |

The familiar probe contains 72 training-layout routes. These are separate from
the 192 validation routes on new layouts. Validation support is equal across
the four maps, so pooled rates and equal-weight map means agree up to floating
point rounding. The dashboard also provides per-map results.

BFS solved all 192 validation routes optimally, with success 100% and SPL 1.000.
Random succeeded on 5 routes, with success 2.60% and SPL .0226. Conditional
successful route lengths were 1.8, 1.8, 2.625 and 2.0 actions for Q-learning,
SARSA, DQN and PPO respectively. These averages describe only their few successes.

Every learned family missed the aspirational 80% success, 2% maximum collision
and .75 SPL targets. One seed provides pilot measurements rather than a stable
algorithm ranking. The 25,000-step budget is short for this larger state space.

Tabular policies retain separate exact-state tables for training layouts. A new
occupancy map has zero Q values, and deterministic ties choose Up. Their five
validation successes reflect routes compatible with that initial rule. More
training on the existing layouts alone cannot fill unseen layout keys. DQN and
PPO share their learned functions, but their pilot transfer is also weak.

## Actual costs and audit

| Algorithm | Training seconds | Total run seconds | Selected checkpoint bytes | Mean decision µs |
|---|---:|---:|---:|---:|
| Q-learning | 5.09 | 5.96 | 29,084 | 32.24 |
| SARSA | 5.09 | 6.01 | 34,203 | 29.82 |
| DQN | 12.42 | 19.74 | 208,685 | 90.42 |
| PPO | 15.68 | 17.07 | 300,559 | 184.50 |

The suite took 49.27 seconds. Training time includes environment, learner and
ledger/capture work, with monitoring/checkpoint work subtracted. Total run time
also includes construction, monitoring, full validation, familiar probing,
loading and replay export. Decision timing uses 100 queries after validation.
The study occupied approximately 5.61 MB before the audit report, within the
512 MiB limit. Execution used CPU and one Torch thread.

The saved-action audit checked 38,008 actions, including 13,905 learned validation
actions, 21,837 familiar-probe actions, 80 actual training actions across 12
recordings and 2,186 reference actions. Reloaded actions and diagnostics match,
budgets and ledgers reconcile, and inference fingerprints remain unchanged.
All 174 regression tests passed in 51.35 seconds.

The first audit exposed SB3's vector wrapper appending `TimeLimit.truncated`
to some shared training-info dictionaries. All original core transition values
matched. The compatibility reader now verifies that extra flag against the
stored termination flags, then validates the original core fields in memory.
The original recordings and executed source snapshot are preserved. The
collector now copies the info dictionary for future captures. A regression test
checks both compatible older records and inconsistent timeout metadata. Training
was performed once. The audit records separate training and verification source
hashes.

## Dashboard and reproduction

Open **Unseen maps** in the sidebar or research library. Results includes the
four-family pilot table, per-map support and references. Replay offers each
algorithm's trained-policy examples and actual learning recordings, using the
same centered map, state, reward and playback presentation as the fixed-map UI.
Every recording identifies its layout, scenario, run and development split.
Browser checks at desktop, 854-pixel and 390-pixel widths passed. DQN's recorded
training window advanced to action 16 with return -.16 and position `[2,5]`.
A trained PPO example reached `[0,6]` in two actions with return 9.98 and SPL 1.
These are illustrative saved episodes. Aggregate performance is reported above.
Evidence is in [the verification record](milestone10_verification.json),
[results capture](unseen-map-pilot/results.jpg) and
[learning replay capture](unseen-map-pilot/learning-replay.jpg).

All 12,605 protected pre-study artifact/data and workspace README/plan files
retained their hashes, and all 184 original frozen snapshot inputs passed their
seal checks. The versioned artifact attributes preserve the new records' raw
bytes across checkout.

Run from the project directory in the existing environment.

```powershell
.venv\Scripts\python.exe -m robot_navigation.unseen_study prepare
.venv\Scripts\python.exe -m robot_navigation.unseen_study pilot
.venv\Scripts\python.exe -m robot_navigation.unseen_study verify
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Preparation verifies an existing compatible benchmark. Pilot returns the saved
completed study on repetition, and a per-study execution lock prevents concurrent
pilot writers. Completed runs can be reused after an interrupted suite. Incomplete
run directories are retained and rejected as complete results. Exact training
resume is unsupported.

Primary evidence is under
`artifacts/studies/unseen-map-pilot-v1/`, including `summary.json`,
`verification.json`, the executed source/protocol snapshot and four run folders.
New map manifests are under `data/unseen/unseen-maps-v1/`.

## Remaining stage

The next research milestone must register the main development budget, fresh
training seeds and any transfer-focused representation/architecture comparison.
It should address the pilot's weak learning before freezing an independent final
runner. Exact-state tabular transfer limitations require explicit interpretation.
Final evaluation remains outstanding. There have been zero policy executions
on the 10 reserved final layouts, and ordinary loaders/UI keep them sealed.

The historical fixed-map benchmark and selected controller retain their original
scope. Current results make no claim of successful general navigation or final
holdout performance.
