# History-aware loop recovery: development protocol v1

Registered 17 November 2026, before the new controller is scored. Study ID `loop-recovery-dev-v1`. This is an exploratory development study, not another M12 final evaluation.

## Hypothesis and fixed design

Use the three existing M11 shared-linear SARSA checkpoints, training seeds 12101, 12203 and 12301. Compare each deterministic base policy with the same policy wrapped in bounded history-aware recovery on the same 192 routes across four development layouts. No weights, rewards, action cap, map roster or observation version change. No training and no access to M12 test layouts.

The wrapper observes the last five robot positions. Pattern A-B-A-B-A with distinct A/B triggers recovery if fewer than four interventions have occurred in the episode. Compute the base policy's four Q scores. Eligible candidates are all actions except its greedy choice whose score is within **0.25** of the maximum. Choose uniformly from those candidates using a seeded RNG. If none qualify, use the base action without consuming an intervention. No obstacle action mask, BFS, safety override, added reward or early termination is introduced. Consequently recovery can cause collision.

Clear history and the intervention counter at each episode reset. Use intervention RNG seeds **17011, 17021 and 17033**, each as one stream across the ordered route roster for each trained policy. Retain the distinction between training seed and intervention RNG. There are 9 wrapper runs / 1,728 wrapper-route cases. Evaluate each deterministic baseline once per training seed (576 baseline cases) and pair/reuse it across the three intervention streams. Reuse is not additional independent baseline replication.

## Measurements and acceptance rule

Primary result: equal-weight mean success change. Promotion requires at least **+5 percentage points**, no decrease in mean SPL and mean collision rate **at most 2%**. Report a missed target as a result and retain the original selected controller. Publish every training-seed/RNG row, paired timeout-to-success and timeout-to-collision counts, persistent suffix-cycle rate, actual interventions, base/wrapper decision latency, wall time and storage. No selection of the best RNG/route is permitted.

For a stochastic, history-aware controller, a repeated state need not imply repeated future actions. Keep M13's deterministic state/action consistency checks unchanged. The new diagnostic counts a timeout as a persistent suffix cycle only when at least two complete state/action repetitions exactly match the remaining suffix; it does not infer future behavior or label a nonrepeating trace as deterministic.

Preserve all complete trajectories, cap illustrative recordings at one per outcome per run, and generate a seed/RNG success/SPL figure from saved rows. Limits: **180 seconds** execution, **256 MiB** new artifacts. An exceeded limit retains an incomplete attempt and excludes it from a complete-study headline. Inputs, source and this protocol are hashed before scoring. Verification recomputes stored transitions, pair outcomes and aggregates without changing old artifacts.

Unit gates: a hand-solvable two-cell loop, an unaffected success path, recovery leading to collision, reset/RNG determinism, absent near-ties and the four-intervention cap. Existing fixed and unseen frozen releases must retain their seals after the study.
