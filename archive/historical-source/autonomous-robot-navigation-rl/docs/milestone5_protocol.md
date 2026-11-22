# Milestone 5 protocol — integration and core robustness

Recorded before producing group results. No new training, hyperparameter search,
reward change, dynamics noise or final-test execution is authorized by this protocol.
All labels use the canonical map and the 928 saved validation pairs, independently
of learned outcomes. Existing manifests and experiment snapshots stay immutable.

## Fixed labels

- Distance: short 1–5, medium 6–10, long >=11, from saved shortest distance.
- Corridor: a canonical BFS reference route intersects a free cell with exactly
  two opposite free neighbors (vertical or horizontal); otherwise no corridor.
  Includes endpoints. The canonical shortest route uses the existing fixed move
  order. This is a reference-route property, not a guarantee every shortest route
  or the actual learned route has the same property.
- Wall proximity: the start has at least one blocked/out-of-bounds cardinal
  neighbor versus an open start. The boundary counts as a wall.
- Start region and goal region: NW/NE/SW/SE using row < height/2 and column < width/2.
- Region crossing: same start/goal quadrant versus different quadrants.

Each dimension partitions validation independently. Dimensions overlap; do not
sum groups from different dimensions or claim separate independent test sets.
Retain scenario IDs, counts, signatures and the label-file hash.

## Systems and aggregation

Use every saved validation metric for milestone-4 original-setting controls and
tuned Q-learning/SARSA, seeds 101/211/307/401/503 (four systems, five seeds each).
Use all five milestone-3 DQN seeds 7/19/43/67/91 as historical Part A context,
explicitly unpaired with Part B. Include the fixed milestone-4 vote as one system,
with no between-ensemble SD. No best-seed selection. Reconcile saved episode
aggregates with run summaries, verify checkpoint/task checksums and hash input
files. This reanalysis adds zero evaluation interactions to the recorded benchmark.

For every group report per-seed success/collision/timeout/SPL, mean successful
route length with its success count, return (same reward only), support and sample
SD across seeds. Five-seed means are descriptive stability evidence, not episode
confidence intervals. For Q/SARSA control versus tuned report seed-matched deltas
within each group. Keep missing/zero support explicit and retain weak groups.
Frozen policies replayed interactively use validation/custom demonstration only;
those displays never enter these benchmark summaries.

## Dashboard and comparisons

Six views: overview, simulator, learning analysis, agent comparison, optimization
and robustness, methodology. Keep the existing animated playback controls; add
prefix-only cell visits, optional planner reference, actual recorded Q scores or
vote proposals, event seeking, matched agent comparison and early-versus-selected
checkpoint replay. Synchronize by action index and hold each finished episode at
its terminal frame. Identity/signatures/scenario/evaluation mode must be visible.
No inference parameter changes or alternative environment transitions.

Archive the first periodic (10k) checkpoint for a predeclared exemplar from each
implemented family: m3-q-learning-seed7, m3-sarsa-seed7, m3-dqn-seed7, plus
m4-tuned-q-learning-seed101 and m4-tuned-sarsa-seed101. These are display examples,
not extra selection candidates. Existing selected checkpoints supply later frames.
Missing early files are an explicit limitation, never substituted with selected
weights. No full periodic checkpoint inventory is required by the UI.

## Acceptance evidence

Run focused group/aggregation and synchronized playback tests; exercise every
dashboard view, overlays, action scores/votes and event seeking through AppTest.
Audit matched displayed trajectories/actions against saved models. Inspect figures
and attempt an isolated local browser check of success/failure/comparison playback.
Record actual limitations if browser automation is unavailable. Verify all 3,041
protected tracked root/benchmark/earlier-artifact hashes. Save actual results and
costs locally. Final freezing and sealed test evaluation belong to milestone 6.
