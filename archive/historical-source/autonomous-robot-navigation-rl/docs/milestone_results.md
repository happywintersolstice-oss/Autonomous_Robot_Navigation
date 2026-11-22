# Milestone 1 — actual results, 6 November 2026

The Section 12 implementation is present in this isolated directory. Environment,
scenario preparation, random/BFS policies, Q-learning, metrics, saved runs and
local Streamlit replay are implemented. Automated checks pass. **Browser-visible
animation acceptance remains unverified** because Computer Use stopped at its
current-URL policy check. No result below is a final-test result.

## Prepared benchmark

- Version: `bottleneck-v1`; 10 × 10, 79 free / 21 blocked cells, one component.
- 6,162 reachable ordered pairs, split into 4,306 training (69.88%), 928 validation
  (15.06%) and 928 final test (15.06%). All 79 goals occur in training, with 52–56
  training starts per goal. Validation/test each have 11–13 starts per goal.
- Distance bins: short 1–5, medium 6–10, long >=11. Validation has 258/350/320 pairs
  respectively. A fixed 30-pair distance-balanced monitoring subset was saved first.
- Preparation is deterministic and split/goal/coverage checks passed. The test
  manifest was generated and structurally checked, with **zero final-test policy
  executions**. Runtime loaders and the viewer exclude the test split.
- The layout/splits pass structural checks. Main-study benchmark freezing is
  deliberately pending learning-pilot review; the task was not made easier after
  seeing this result.

## Real Q-learning pilot

Seed 7; 30,000 interactions; alpha 0.3; gamma 0.99; epsilon 1.0 → 0.1 over the
first 80% of the budget. Training samples saved training pairs only. The final
budget checkpoint is retained; no checkpoint/configuration search occurred.

| Policy | Success / 928 | Collision | Timeout | Success rate | SPL | Mean return |
|---|---:|---:|---:|---:|---:|---:|
| Q-learning | 77 | 348 | 503 | 8.30% | 0.081286 | -2.139073 |
| BFS, known map | 928 | 0 | 0 | 100% | 1.000000 | 9.915420 |
| Random, untrained | 22 | 906 | 0 | 2.37% | 0.019854 | -4.684483 |

All policies used the same complete validation scenarios and episode rules.
Successful mean route lengths were Q-learning 2.247 (77 successes), BFS 8.458
(928 successes) and random 3.045 (22 successes). These conditional means are
**not** evidence that Q-learning/random find better paths: most difficult episodes
fail. Q-learning succeeded on 74/258 short, 3/350 medium and 0/320 long pairs.
It has learned some short navigation, but it is not an adequate navigation policy.

Monitoring at 10k/20k/30k interactions produced 1/3/3 successes on the fixed 30-pair
subset; this differs from complete validation and is labeled separately in artifacts.
Training completed 6,525 episodes, including 373 successes and successful goal
discoveries for 72/79 goals. The saved state-action array contains 18,537 visited
entries and exactly 30,000 visits. Goal sampling frequencies include the unused
reset immediately after the final completed training episode (6,526 sampled starts).

857/928 validation initial coordinate states (92.35%) were visited as states during
training, even though their initial pairs were held out of reset sampling. This
measures held-out initial conditions on a familiar map; it is not strict unseen-state
generalization. This single seed provides no seed-level uncertainty or algorithm
ranking beyond a pilot observation.

Training-loop time was 1.048 s, about 28,622 interactions/s on this machine.
It excludes initial environment construction, monitoring, checkpoint I/O, final
validation and trajectory export. Complete Q-learning validation took 0.717 s
for 101,506 actions, about 141,566 actions/s; BFS 0.185 s / 7,849 actions; random
0.185 s / 3,720 actions. These use `perf_counter` with rendering disabled. The
microbenchmark averaged 1.878 microseconds per greedy Q-learning decision over
100 repeated queries; it is not end-to-end application latency.

Monitoring used 7,914 additional actions. Complete validation used 113,075 actions
across all three policies. Exporting full trajectories repeated those same frozen
episodes for another 113,075 actions, recorded separately. BFS also passed optimal
navigation on all 4,306 training pairs (36,259 actions), without opening the test
manifest. Validation and exports do not change Q values. The final inference table
is 28,938 compressed bytes with matching metadata and checksum.

## Verification and UI

- **16 pytest checks pass** (final run: 5.98 s). They cover environment/seeding,
  action-200 terminal precedence, truncation bootstrapping, hand-calculated updates,
  fixture learning, split determinism/disjointness, goal coverage, metrics,
  save/load, corruption/map/reward rejection, replay validation and playback.
- Gymnasium's environment checker passes.
- All 2,784 full validation trajectory records were verified against the actual
  environment: 113,075 saved actions, with states, rewards, flags and metrics
  reconciled. Saved Q scores/actions match the frozen checkpoint. Visitation and
  training CSV counts reconcile. These replay checks do not execute a test policy.
- Streamlit's AppTest harness exercises Next, speed, seek, Play/Pause, Reset,
  actual timed-advance branch/counters, saved learned collision replay, a successful
  custom BFS demo, and invalid custom-cell rejection. Unit tests also verify pause,
  resume and automatic stop at terminal frames. The UI reads saved progress and
  never trains.
- The BFS figure was visually inspected and its legend spacing corrected. Six
  local Matplotlib renders averaged 0.173 s after that correction. This supports
  rendering feasibility for a 2 actions/s target but does not measure browser
  responsiveness. Browser/network/Streamlit scheduling are excluded.
- The training execution's source snapshot is retained under
  `artifacts/runs/smoke-seed7/source/`; its source hash matches the run manifest.
  Only renderer-footer spacing and UI import order/cache behavior changed after
  the pilot. Environment, agents and training code are unchanged. Regenerated
  figures use the corrected renderer and original replay records.
- Original `.gitattributes`, root README and project plan hashes are unchanged.
  No energy project or `sources/` directory was present in this checkout, and no
  files in other projects were edited. See `preservation_check.json`.

## Open issues and next step

1. **Manual animation verification is still required.** The local server started
   successfully at `http://127.0.0.1:8517` (8501 was occupied). Browser access was
   unavailable. A screenshot of unrelated Chrome content was rejected by automatic
   approval review; that capture was not retried. Computer Use later ended because
   it could not establish Chrome's current URL confidently enough to enforce policy.
   No browser-visible start-to-goal playback or browser responsiveness claim is
   made. Open the local URL, select BFS success, Play/Pause/seek/reset, then inspect
   the Q-learning collision and timeout replays. Until checked, that UI acceptance
   criterion remains open despite passing automated wiring tests.
2. **Learning remains weak.** The next diagnostic is to inspect the saved timeout
   loops, Q values, visitation and undiscovered goals by distance, then predeclare
   an adequate learning-pilot budget/protocol before the main study. The fixture
   already learns correctly. Do not hide failures, add a planner fallback, tune on
   final-test pairs, or silently change the benchmark to raise the score.
3. Checkpoints are explicitly inference-only. Exact interrupted-training resume,
   multi-seed comparisons and later RL dependencies are not part of this pilot.

No SARSA, DQN, PPO, tuning, ensembles, unseen maps, moving obstacles, continuous
control, physical robot integration, deployment or full comparison dashboard was
added. All artifacts remain local.
