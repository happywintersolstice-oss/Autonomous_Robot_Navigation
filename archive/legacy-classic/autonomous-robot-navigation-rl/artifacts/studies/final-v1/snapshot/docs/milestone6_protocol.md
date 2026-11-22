# Milestone 6 — frozen final evaluation protocol

Written before any final-test policy execution. Preserve the canonical map and
all manifests, earlier experiment artifacts and milestone-5 outcomes. No new
selection or tuning on the holdout. A separately isolated training reproduction
may check the existing seed101 policy before freezing; it cannot change selection.

## Frozen choice and matrix

The final selected system is `m4-tuned-q-learning-seed101`, original step-cost
reward, monitoring-selected checkpoint, greedy argmax with Up/Down/Left/Right tie
priority. This choice comes from Part B family selection and predeclared seed101;
it remains selected regardless of final outcomes. Primary matched single control:
`m4-control-q-learning-seed101`. No reward/representation/version changes.

Evaluate these exact systems on every saved final-test pair, in stable manifest order:

- Part A Q-learning, SARSA and DQN: all seeds 7/19/43/67/91 (15 records).
- Part B original-setting Q-learning/SARSA controls, all seeds 101/211/307/401/503;
  and their tuned confirmations, all the same seeds (20 records).
- Existing Q-learning vote: ordered members 101/211/307, majority; three-way ties
  use member101 (one ensemble record, no ensemble SD).
- Random reference: policy RNG seeds 7/19/43/67/91 (five streams, not trained seeds).
- Known-map BFS reference (one record).

Total: 42 system/seed records. Deterministic learned policies and BFS use reset seed
0. Each random stream persists across manifest scenarios; environment reset seed
is its declared stream seed. No policy explores or updates. Validation-monitoring
selected checkpoints remain selected even if an earlier/later snapshot would score
better. All periodic model choices were made before test.

## Freeze and access gate

Before parsing test scenarios, save a freeze containing the matrix, exact model/
metadata/configuration hashes, map/task/split hashes, protocol, dependency lock,
installed versions, runner/analysis code and all src/app/config files. Archive that
source and selected models, retaining the relative voting descriptor/member links.
Record the freeze digest. The run command requires this digest; check every frozen
file before and after evaluation. Refuse an existing evaluation output directory.
The generic trainer, CLI evaluate and simulator continue accepting validation only;
only the dedicated frozen runner loads test and enables final-evaluation metadata.
No source or model changes may occur between freezing and completed analysis.

## Analysis fixed in advance

Report per-seed success/collision/timeout counts and rates, SPL, return and conditional
successful route length/count; means and sample seed SD, with no seed pooling into
episode-level confidence. Random SD spans RNG streams; voting/BFS have no seed SD.
Show original Part A separately from seed-matched Part B controls. Report paired
control→refined success/SPL/collision/timeout changes for Q/SARSA. Compare the vote
against each member and predeclared seed101; no test-based promotion.

Apply the existing milestone-5 geometry/distance labels, unchanged, to test pairs.
Retain supports and all per-seed groups; these overlapping dimensions are descriptive
and not unseen-map/causal evidence. Apply the unchanged 80% success / <=2% collision /
SPL >=.75 mean targets to tuned five-seed systems and report misses. Targets do not
override correctness gates or determine which seeds are retained.

Generate comparison, distance/group, outcome and conditional-route figures. Retain
full compressed final trajectories, per-episode metrics and first occurrence of each
outcome per record. Audit every saved action, score, reward, state and flag against
the shared environment and frozen policy. Audit replay is a repeatability pass on
the same cases, never extra independent evidence. Verify no parameter changes.
BFS must succeed optimally on every reachable final scenario.

## Limits, setup and release

Estimate final evaluation time and storage from recorded validation throughput and
trace sizes before execution. Cap the evaluation phase at 15 minutes / 500 MiB;
stop before the next record if exceeded and report incomplete coverage. Audit time
is separate. Do not adjust the matrix, retry poor seeds or overwrite interrupted
results. A reproduction run and setup checks use isolated local directories and
the installed locked dependencies; do not claim a clean network install if not done.

Record the complete development run matrix, costs, failures, environment/library
versions, actual results and known limits. Finish a concise Markdown technical
report and Windows instructions; assessed-report formatting is optional. The UI
may show frozen aggregate final results after completion, while its simulator still
uses validation/custom scenarios. No deployment, public share link, commit or push.
PPO, dynamics noise, unseen maps, moving obstacles, continuous control and hardware
remain deferred.
