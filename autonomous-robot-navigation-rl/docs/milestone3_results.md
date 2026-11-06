# Milestone 3 — Part A comparison, 6 November 2026

Implemented vanilla Stable-Baselines3 DQN and completed the predeclared five-seed
comparison with Q-learning and SARSA. **Q-learning and SARSA are shortlisted for
Part B.** DQN performed poorly under this baseline configuration; all its runs and
failures are retained. No tuning, ensembles, PPO or environment extensions were
added. The final test remains sealed.

## Measured results

All learners received 200,000 training interactions per seed on the unchanged
fixed map, actions, rewards and training pairs. Monitoring used the saved 30-pair
validation subset every 10,000 interactions. Each seed's checkpoint was selected
by monitoring success, then SPL, then earliest checkpoint; it was then evaluated
on all 928 validation pairs. The full budget was completed even when the selected
checkpoint was earlier. These are development-validation results, not independent
final-test estimates.

| Agent | Mean success ± seed SD | Mean collision | Mean timeout | Mean SPL ± seed SD |
|---|---:|---:|---:|---:|
| Q-learning | 65.24% ± 1.64 pp | 0.17% | 34.59% | .619171 ± .015389 |
| SARSA | 21.94% ± 5.37 pp | 3.94% | 74.12% | .206985 ± .049524 |
| DQN | 5.32% ± 1.04 pp | 31.77% | 62.91% | .052956 ± .010513 |
| Random reference | 2.95% ± .41 pp | 97.05% | 0% | .023184 mean |
| BFS reference | 100% | 0% | 0% | 1.000000 |

Learned-agent SD is sample SD over five independent training seeds. Random SD is
over five policy-RNG seeds; BFS is deterministic and evaluated once. Each episode
uses the same environment rules; BFS explicitly receives the known map. Conditional
successful route distributions are reported alongside failures, not used to imply
that agents succeeding only on easy tasks are more efficient.

| Algorithm | Seed | Selected interaction | Success | Collision | Timeout |
|---|---:|---:|---:|---:|---:|
| Q-learning | 7 | 190,000 | 581 | 2 | 345 |
| Q-learning | 19 | 200,000 | 621 | 0 | 307 |
| Q-learning | 43 | 190,000 | 603 | 2 | 323 |
| Q-learning | 67 | 200,000 | 608 | 3 | 317 |
| Q-learning | 91 | 200,000 | 614 | 1 | 313 |
| SARSA | 7 | 200,000 | 255 | 15 | 658 |
| SARSA | 19 | 190,000 | 220 | 24 | 684 |
| SARSA | 43 | 200,000 | 238 | 18 | 672 |
| SARSA | 67 | 90,000 | 135 | 99 | 694 |
| SARSA | 91 | 130,000 | 170 | 27 | 731 |
| DQN | 7 | 40,000 | 49 | 237 | 642 |
| DQN | 19 | 20,000 | 57 | 340 | 531 |
| DQN | 43 | 70,000 | 61 | 177 | 690 |
| DQN | 67 | 40,000 | 39 | 329 | 560 |
| DQN | 91 | 20,000 | 41 | 391 | 496 |

Every row totals 928. Checkpoint selection on a small monitoring subset can select
an earlier model with worse complete-validation performance than a later model.
This protocol intentionally differs from milestone 2's final-budget checkpoints;
the changed seed set and selection rule mean the milestone means are not a
controlled before/after improvement comparison. No retrospective reselection was
performed against complete validation or final test.

## Shortlist and failure analysis

The frozen ranking uses mean complete-validation success first, mean SPL second,
then lower success SD, lower training time and name for exact ties. This ranks
**Q-learning, SARSA, DQN**. The shortlist is the first two algorithm families, not
two favorable individual seeds. Their five-seed evidence and poor runs remain
available. It does not establish that every tuned DQN or other benchmark would
have the same ranking.

Across 4,640 validation episodes per learned algorithm:

| Outcome taxonomy | Q-learning | SARSA | DQN |
|---|---:|---:|---:|
| Optimal success | 2,478 | 851 | 242 |
| Successful detour | 549 | 167 | 5 |
| Obstacle collision | 4 | 135 | 1,474 |
| Boundary collision | 4 | 48 | 0 |
| Timeout with repeated states | 1,605 | 3,439 | 2,919 |

On the 320 long-route scenarios, success counts across seeds 7/19/43/67/91 were
68/81/77/83/70 for Q-learning, 5/1/3/0/2 for SARSA, and 0/6/5/0/0 for DQN.
None is a reliable long-route navigator. Successful goal discovery during training
is insufficient: tabular runs discovered all 79 goals, while DQN discovered
77–79 by the full budget. The saved visitation overlap uses **all 200k interactions**,
not only the selected checkpoint's age. Exact selected-age visitation overlap was
not retained. These are held-out initial conditions on a familiar map, not strict
unseen-state or unseen-map generalization.

Actual decision evidence is stored in every replay. For example, the first
Q-learning timeout example (`r0c0-r4c9`, seed 7) ends at observation `[0,1,4,9]`
with scores Up -3.807099, Down -.040821, Left -.038832, Right -.039027. Greedy
selection chooses Left, consistent with the recorded repeated route. The first
DQN timeout (`r0c0-r0c3`, seed 7) ends at `[1,0,0,3]` with scores
Up 2.177706, Down 1.970388, Left -5.024187, Right 1.671855, choosing Up.
These are Q values, not probabilities or a causal explanation of training.

The [figures](../artifacts/studies/part-a-v1/figures/) include monitoring success/SPL
with five-seed variability, complete-validation seed values and reference lines,
collision/timeout heatmaps, conditional successful-route distributions, and
common-scenario trajectory panels plus actual loop/detour examples. The
[common-scenario CSV](../artifacts/studies/part-a-v1/common_scenarios.csv) aligns
all three learners across the same 928 pairs. Heatmap scales are labeled separately;
colors alone must not be compared as equal counts across panels. Figure values
come from saved records; legends were corrected after visual inspection without
changing underlying measurements.

## Implementation, cost and verification

DQN uses a four-coordinate vector scaled by dimension minus one, with the same
information as tabular state indexing. It gets no obstacle grid, planner directions,
invalid-action mask or fallback. Configuration: MLP [64,64], learning rate .0003,
gamma .99, replay capacity 50k, warm-up 1k, batch 64, one gradient step per four
interactions, hard target updates every 1k, epsilon 1 → .1 over 80%, CPU/one thread.
SB3 training ties use its greedy first maximum, whereas tabular training randomizes
ties. Evaluation uses first maximum in Up/Down/Left/Right order for every learner.

This is [SB3's vanilla DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html),
not Double DQN, Dueling DQN or prioritized replay. Separate-environment validation
and multiple training seeds follow [SB3's evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).
Installed versions tested: Python 3.14.10, Gymnasium 1.4.0, SB3 2.9.2 and
PyTorch 2.15.0+cpu; exact distributions are pinned in `requirements-lock.txt`.

The real 10k pilot trained in 3.63 seconds, discovered 36 goals, and reached 20/928
validation goals (358 collisions, 550 timeouts). Complete validation/export took
6.21 seconds. Fixture learning, normalization parity, environment checkers and
target tests passed, so the planned 200k comparison proceeded without task or
score-driven configuration changes. Estimates and the separate main-study freeze
are saved in `benchmark_freeze.json`; the older provisional data report was preserved.

The 15 main runs used **3,000,000 training interactions**, **1,245,286 monitoring
interactions**, **1,622,440 complete-validation actions** and **26,357 reference
actions**. Export wrote each evaluated trajectory once rather than repeating it.
The main runner took **648.37 seconds (10.81 minutes)**, excluding later analysis.
Total training-loop seconds were Q-learning 31.02, SARSA 27.76 and DQN 479.31,
excluding monitoring, model construction, exports and plotting. Other local work
was active; times describe this machine, not isolated hardware benchmarks. Main
and pilot run artifacts occupy about 76.29 MiB locally, including periodic models
and compressed trajectories; caches/dependencies are excluded.

- **33 tests pass** (12.05 seconds). Tests cover real SB3 timeout replay masking
  and final next-observation storage, true terminal targets, fixture learning,
  normalized/raw transition parity, map/encoding/checksum rejection, deterministic
  save/load, inference RNG/weight/optimizer/replay immutability, checkpoint selection,
  failure taxonomy, ranking, overwrite protection and failed-run status.
- Gymnasium and SB3 environment checks pass. A timeout retains bootstrap value;
  a true goal/collision does not. Goal/collision retain priority on the action cap.
- All **13,920 learned validation episodes / 1,622,440 saved actions** were checked
  against the environment and selected policy, including stored Q scores/actions.
  All **300 periodic checkpoints** load, source snapshots match, and checkpoint
  selections reproduce the declared monitoring rule.
- **5,568 reference episodes** were replay-checked; BFS solved all 928 validation
  pairs optimally. Training log steps plus each budget-cutoff episode reconcile to
  exactly 200k per run. No final-test policy executions occurred.
- Checkpoints remain inference-only. SB3's optimizer serialization does not provide
  exact resume without replay, RNG and environment state. Failed/interrupted jobs
  retain status and snapshots; no failed main run was discarded.
- Earlier benchmark data, runs and study evidence are preserved. Existing energy
  files and `sources/` were not modified; neither was present in this checkout.

## Handoff

The local viewer supports selected DQN `.zip` models and legacy tabular `.npz`
models. AppTest verifies DQN replay and custom frozen-policy selection, including
one recorded step; the earlier Q-learning/SARSA controls remain covered. It never
trains or exposes the sealed final-test manifest. Browser-specific observations
are recorded separately in `milestone3_ui_check.json`.
The dedicated browser visibly played the real pilot's `r0c2-r0c3` DQN replay from
`[0,2]` to `[0,3]`, ending in success after one action with return/step reward 9.99.
This verifies integration, not broad navigation capability. Main-study models
were all checkpoint/replay-audited; they were not all manually replayed in Chrome.

Part B can now refine the Q-learning/SARSA shortlist under a new predeclared search
protocol. Weak long-route reliability, selection-subset sensitivity and this DQN
configuration's failure remain unresolved research findings. No deployment,
final-test evaluation, tuning, voting, PPO, new maps, moving obstacles, continuous
control, physical robots or full comparison dashboard was added in this milestone.

Evidence: [summary](../artifacts/studies/part-a-v1/summary.json),
[shortlist](../artifacts/studies/part-a-v1/shortlist.json),
[audits](../artifacts/studies/part-a-v1/verification.json),
[failure records](../artifacts/studies/part-a-v1/failure_analysis.json),
[protocol](milestone3_protocol.md), and [freeze](../artifacts/studies/part-a-v1/benchmark_freeze.json).
