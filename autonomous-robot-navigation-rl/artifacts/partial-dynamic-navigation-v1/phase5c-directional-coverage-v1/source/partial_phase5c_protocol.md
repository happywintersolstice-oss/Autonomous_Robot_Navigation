# Phase 5C: zero-training directional coverage analysis plan

Study `phase5c-directional-coverage-v1`, analysis `directional-coverage-audit-v1`.
Base is the published, reviewed Phase 5B commit
`ae3ae73bf7ba5cbe36dcf879125f15fd14c43f4e`, on new branch
`feature/phase5c-directional-coverage-v1`. Read the root work order
`CODEX_PHASE5C_DIRECTIONAL_COVERAGE_DIAGNOSIS.md`. No previous file is edited.
Commit/push require a further explicit request; no main merge is authorized.

## C0 first, before any possible intervention

Freeze this plan, config, analysis source, tests and legal probe manifest before
computing the new coverage outcomes. Check the old Git parent/tree, Phase 5A/5B
delivery pins and all existing file hashes. Use a minimal snapshot of only new
source/config/tests/brief, flat filenames, plus per-file hashes for dependencies;
do not duplicate old raw evidence or create long nested repository snapshots.

Read all eight Phase 5B training streams and every 0/2k/6k checkpoint, including
case streams, numerical logs, saved replay/rollout state and development replays.
Reconcile exact prefixes, resets, actions, goal rewards, flags and old summaries.
Mechanical replay executes only recorded actions; score probes execute no model
actions, call no trainer/update, and perform no new policy/environment rollout.
Report those replay counts separately from zero new Phase 5C training/assessment
actions. Required inherited tests may perform tiny mechanical optimizer tests;
they are not additional scientific training attempts.

## Denominators and categories

At each historical checkpoint report each prefix separately. The 2k prefix is
contained in the 6k prefix: never sum them as independent interactions. The
untrained checkpoint has zero training observations and undefined rates.

Reset counts include actual empty SB3 autoresets at a budget boundary. Separately
count completed episodes, nonterminal prefixes with actions and empty resets.
Every reset must match one of the eight original A cases. Compare its exact
case-ID sequence to seeded uniform sampling and the recorded case-stream JSON.

Current goal direction is exactly one of Up, Down, Left, Right, Up-Left,
Up-Right, Down-Left, Down-Right or At-goal. Diagonal decisions are separate,
never two disjoint axis samples. State identity is the SHA256 of all 409 pre-action
float32 values, not position alone. Keep episode-start direction separate from
current direction after wandering. Joint bins are direction x robot quadrant x
edge/interior x Manhattan distance (0/1/2/3-5/6+) x current-cell clipped visits
(first/revisited/saturated-five) x known-cell band (<25/25-74/75+).

For each marginal and observed joint bin save all five chosen actions, actual
distance reduction/preservation/increase, attempted collision, successful move,
Wait and contemporaneous greedy preference. Zero observed categories have
count zero and rate null. Unobserved Cartesian joint bins include impossible
combinations and do not establish zero probability or missing causal coverage.

Near-goal distance 1 has one immediate goal action. Distance 2 uses distance-
reducing actions; no cardinal action can immediately reach a distance-2 goal.
Distinguish selected action, membership in the maximum-score tie set and first
argmax. Epsilon is logged but its realized exploration coin is not: a non-greedy
choice is not proof of a particular epsilon event. PPO is stochastic in training.
For SARSA the carried action was sampled before the preceding update; its current
score ordering need not equal its ordering when sampled. Do not invent labels.

Goal signals are counted by actual goal-reaching action, reset case/distance and
position-goal pair. Inspect DQN stored rewards against raw transitions. For PPO,
only the last recorded rollout buffer is available at each saved checkpoint;
compare its goal rewards with the last 100 raw actions, distinguishing timeout
bootstrapping from actual positive goal rewards. Record optimizer steps, PPO
epochs, epsilon/entropy schedules and finite status without equating families.

## Predeclared legal probes

Use 24 fresh reset observations: robot positions (4,4), (2,2), (7,7), four
cardinal goals, distances 1 and 2. These establish fixed-position goal variation
and coordinated legal translations, not a pure intervention on one hidden
neural factor. Add all eight D4 transforms of each of two fixed script-history
observations (8 and 22 chronological scans) on the open grid. Construct history
through the unchanged sensor/memory encoder, without env.step or any model.
Rotate/mirror coordinates, all four 10x10 channels, N/S/E/W rays and action
mapping together. Tests compare with actual scripted mechanical trajectories.

Freeze the resulting 40 observations/hashes before scoring any checkpoint.
Score every saved family/arm/checkpoint without calling policy.act; derive
argmax mathematically and keep its scores/probabilities. Record transformation
equivariance errors and translation/distance sensitivity as descriptive probes,
not causal findings. Also retain all historical original/expanded first decisions
and explicit Right-success versus Right-failure examples where available.

## Sampling gate, fixed before new coverage counts

This is an engineering triage rule, not a significance test or a claim that
small samples prove undertraining. Assess only the historically selected B1 arm,
using its final 6k checkpoint, with all four families examined. A family/direction
is a credible reset-sampling candidate only when **all** hold:

1. At least 64 recorded episode starts. That direction has reset share <15%
   (nominal 25%), OR either of its two cases has reset share <6.25% (nominal 12.5%).
2. The same direction achieves at most 1/4 expanded development goals.
3. It has fewer than 50 current-axis distance-1 decisions OR fewer than 10 goal
   transitions arriving from that current direction. These are scarcity flags,
   not a declaration of causal absence.
4. A fixed no-learning simulation at proposed seed 23041 and the same number of
   episode starts improves that direction's share by >=5 percentage points OR
   the deficient case's share by >=3 points, using balanced shuffled blocks.

Inspect seeded schedule prefixes 8/32/64/128/256/512 and the observed episode
counts; do not select the most favorable prefix for the gate. Exact full blocks
contain each of the eight cases once, hence two starts per direction. Incomplete
blocks remain incomplete. Simulations do not equalize transition counts, goal
rewards or visits and do not simulate learned actions.

If no candidate passes, terminal verdict is
`DIAGNOSIS_ONLY_SAMPLING_NOT_JUSTIFIED`, with zero new scientific training actions.
Balanced resets together with substantial direction/goal signals are evidence
against this *specific first intervention*, not proof that ordering can never
matter or that state-action coverage is adequate everywhere. A different
optimizer/credit-assignment hypothesis requires a separately reviewed brief.

If a candidate passes, do not use this C0 plan as the training registration.
First implement/validate and separately freeze the permitted C0-uniform versus
C1-shuffled-block experiment, B1 features in both arms, seed23041, checkpoints
0/2k/4k and all exact work-order ceilings/screens. Generate/audit the new16-case
development panel at seed23042 only then. No automatic training call is exposed
by the read-only C0 command. A passing audit alone is not a learning outcome.

## Integrity, limits and reproducibility

C0 wall ceiling 900 seconds, new archive ceiling 128 MiB (stricter than the
conditional 512 MiB maximum); exceptions preserve a failure receipt and never
produce a green gate. Register no training budget or placeholder training result
when the sampling gate fails. Inference copies cannot mutate checkpoint bytes;
score passes must preserve parameter fingerprints and global RNG. Re-verify old
hashes after analysis. The source-evidence manifest records every historical
artifact read, all source dependencies and the exact runtime/commands.

Commands from the app root with its installed Python and PYTHONPATH=src:
`partial_phase5c validate`, `partial_phase5c prepare`, `partial_phase5c analyze`,
`partial_phase5c verify`, `partial_phase5c report`. Preserve existing outputs;
an exclusive analysis lock prevents overwriting an attempt. `verify` is read-only
over the completed audit and independently recounts its raw denominators.
Run full regressions again after analysis. Add no UI or historical file changes.
