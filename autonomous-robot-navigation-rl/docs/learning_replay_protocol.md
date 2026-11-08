# Recorded learning demonstrations

These are fresh training demonstrations for the Learning replay tab, not
reconstructed histories of archived models and not a new algorithm comparison.

## Declared scope

- Four families: Q-learning, SARSA, four-coordinate DQN and PPO.
- One fresh training seed, 1109, for each family.
- 200,000 actual training steps per family; 800,000 steps total.
- Baseline settings copied from the original Part A configurations and the PPO
  baseline. No hyperparameter search, reward change or algorithm modification.
- Independent `learning-replay-<algorithm>-seed1109` run directories.
- Bundle: `artifacts/studies/learning-replays-v1`.
- The original checkpoints, benchmark results and frozen final selection remain
  untouched. These new runs do not enter the original aggregate results.

## Recording schedule

Record the first two episodes that **start** at or after each boundary:

| Boundary | Training steps | Episodes recorded |
| --- | ---: | ---: |
| Start | 0 | 2 |
| 25% | 50,000 | 2 |
| 50% | 100,000 | 2 |
| 75% | 150,000 | 2 |
| 95% | 190,000 | 2 |

An episode already underway at a boundary is not replaced or restarted. The
recorder waits for the next ordinary training reset. Actual start/end interaction
numbers are saved. It does not select outcomes or scenarios, change actions,
consume the training RNG, or add training steps. Forty episode recordings are
expected across all four families. The existing default first-N recording mode
remains available for historical M8 runs.

These are real exploratory actions, not deterministic evaluations of frozen
checkpoints. A policy can change during an episode. Missing training-time action
scores and probabilities are not fabricated from a later checkpoint.

## Verification

The runner freezes the plan, source hash and baseline configurations before
training. It refuses to overwrite a run or bundle. Its audit checks:

- Training CSV and visitation totals reconcile to 200,000 steps per run.
- Each recording is the declared first/second episode for its training window,
  with matching ledger scenario, steps, return and outcome.
- Manifest checksums, canonical training scenarios, seed and task identity match.
- Recorded transitions replay correctly in an isolated environment.
- Source snapshots match their recorded hashes.
- All pre-existing artifacts and scenario data retain their original bytes.

Unit tests compare recording enabled/disabled for all four algorithms, checking
identical seeded parameters, visitation and episode logs. Invalid schedules fail
before training. Incomplete budget-cutoff recordings remain explicitly incomplete;
they are never relabeled as a successful, colliding or timed-out episode.

The bounded run plan allows 600 seconds per neural training run, 2,400 seconds for
the suite and 1 GiB of new artifacts. Execution status and actual audited results
are written to the new bundle. The runner is
`python scripts/record_learning_replays.py` from the project directory.

## Interpretation

Use these recordings to inspect behavior at different times during learning.
They are sparse samples, not a complete movie of 200,000 steps. Scenarios can
differ between windows, and one training seed cannot establish algorithm quality.
Later episodes may still fail. Use the existing research pages for the original
multi-seed results. There is no new final-test evaluation or model promotion.
