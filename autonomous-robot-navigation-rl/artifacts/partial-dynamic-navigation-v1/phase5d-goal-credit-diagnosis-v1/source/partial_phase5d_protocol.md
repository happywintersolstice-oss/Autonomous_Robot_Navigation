# Phase 5D: preregistered goal-conditioning and credit diagnosis

Study `phase5d-goal-credit-diagnosis-v1`, version `goal-credit-read-only-v1`.
Fork the published Phase 5C commit
`c1505cc1dab0623b9c7eb86b7bfae9ac3216fa4d` onto
`feature/phase5d-goal-credit-diagnosis-v1`. Preserve all prior source/config/data.
The root Phase 5D work order is authoritative. Commit/push need a separate
explicit request. No training, policy/environment episode, final/test
observation, reward/feature/optimizer change, UI work or main merge is permitted.

## G0 and evidence availability before results

Run the inherited 399-test baseline and official Phase 5C read-only verification
on the exact parent before adding new modules to its strict source glob. Preserve
their receipts. Verify the recorded source lists and Phase 5A/B/C delivery pins;
new modules do not justify rewriting their old registration guards. Checksums
for preservation do not select, parse or score held-out final observations.

Inspect actual schemas before selecting the analysis. Linear logs contain actual
target, TD error, feature/update norms and SARSA's sampled next action. Every
training decision has contemporaneous pre-action Q values or PPO probabilities;
PPO also logs a pre-action state value. DQN optimizer logs contain update norms
and counts, not replay batch IDs or historical target tensors. Saved checkpoints
contain online/target networks; full target-network trajectories are absent.
PPO continuation archives contain the last rollout's values, action log-probs,
returns and advantages at 2k/6k, not the entire historical GAE stream. Determine
last-boundary bootstrap availability from the actual archive; never substitute
the post-update checkpoint critic for its pre-update collection-time critic.

Produce an evidence availability matrix with available, not-recorded,
reconstructable-contemporaneous and snapshot-only statuses. Hash every actual
input and all dependencies. Use a flat snapshot of only the new files, parent
Git commit/tree and per-file dependency hashes, not another repository copy.

## Frozen cohorts and observations

Include all eight Phase 5B arms and 0/2k/6k snapshots, in family, arm and numeric
checkpoint order. Read each 6k stream once; 2k is its nested prefix, not another
independent trial. Keep empty autoresets, incomplete prefixes, collisions and
timeouts distinct. Reuse all 40 exact Phase 5C legal probes and all 20 saved
development reset observations. Reuse the 960 Phase 5C score vectors and 480
historical case vectors under verified hashes; these are not new score calls.
No new probe family is generated.

For new historical-observation scoring, preselect at most 48 transitions per arm
from its entire 6k stream, reused unchanged at all three snapshots. Stratify by
current axis direction (four), and outcome (terminal positive goal versus other
distance-one decision). Deduplicate exact pre-action float32 observation hashes
within each stratum, retaining first occurrence. Select up to six equally spaced
indices from each sorted stratum using floor(linspace(0,n-1,k)). If support is
less than six, retain all and disclose it; do not substitute cases. This is a
deliberately balanced descriptive cohort, not a representative success rate.
Freeze IDs, source hashes and observation hashes before any new policy scoring.

Report exact full-409 observation overlap and exact position-goal overlap of
each development reset with training prefixes. Approximate support requires the
same current direction, Manhattan distance, robot quadrant, edge/interior,
current-cell visits bin, known-cell band, mean known-memory age band
(zero/(0,.25]/>.25) and elapsed band (0/1-19/20-99/100-199), plus raw409 mean
absolute difference <=.05. Count repeated transitions and distinct observations
separately. No fitting of this metric or threshold on evaluation results.
No support means unresolved, not no effect. Synthetic probes and development
pairs are inspected familiar-geometry observations, not unseen confirmation.

Goal-only comparisons use Phase 5C fresh-reset observations at identical robot
position and identical sensory/memory/history, changing only legal goal fields.
Reject a swapped goal with positive visit count or impossible coordinates.
D4 observations use the already verified jointly transformed coordinates,
N/S/E/W rays and all memory channels. Translations also translate absolute map
memory, so they cannot isolate pure position from occupancy-channel effects.
Different legal history probes do not isolate history if position also differs.

## Credit calculations and provenance

For every historical prefix, report goal/choice alignment, direction, unique
states, position-goal pairs, near-one-cell selected versus contemporaneous greedy
actions, ties and actual goal rewards. PPO probabilities are not Q values.
First argmax uses action order Up/Down/Left/Right/Wait; all maximizing actions
and ties are retained. Sensor ray order is N/S/E/W, not action order.

Linear actual-log verification is algebra, not replay optimization. Verify
error = logged_target - logged_current_Q(action), and
update_norm = alpha * abs(error) / ||phi||. For a same-episode nonterminal
continuation, back out the already recorded row increment from the next logged
post-update score vector:

`Q_before(s',a) = Q_after(s',a) - alpha*error*(phi(s) dot phi(s'))/(phi(s) dot phi(s))`

only for the row of the action just updated. Then use max for Q-learning or
the actual logged sampled next action for SARSA. Never apply an update to any
model, weight array or replay. Terminal target is the actual reward. At external
truncation or a last nonterminal record, a continuing-state pre-update score is
not independently reconstructable unless actually recorded: retain the actual
logged target, mark independent bootstrap verification unavailable. Do not fill
it with a later snapshot. Inspect alpha .1, gamma .99 and unchanged features.

For every linear probe/case/cohort, split all five action scores into bias,
position, goal, rays, occupancy, age, visits, streak, elapsed and relative-goal
components. Check their sum against the saved or newly scored Q vector. Keep
Right-minus-best-other components signed; large absolute sums are not causality.

For DQN cohort transitions, score frozen online(s), target(s), and target(s')
when not naturally terminated. Snapshot target is
`r + .99*(1-terminated)*max Q_target(s')`. Actual SB3 one-step replay masks
timeouts from dones, so external truncation still bootstraps. No Double-DQN
substitution is introduced. Every such target/residual is labelled snapshot,
even when checkpoint and transition counters coincide; optimizer batch/network
provenance is absent. Actual terminal goal reward9.99 remains a separate fact.

For PPO report probabilities, margins/log-odds, entropy and frozen V(s), never
Q(s,a). Verify saved last-rollout values and sampled log-probs against collection
logs, returns = advantages + values, and the interior GAE recurrence. If the last
rollout ends naturally, check its terminal boundary independently. Otherwise the
pre-update final bootstrap value is not recorded; do not derive it from the very
advantage being verified and call that an independent reconstruction. Earlier
rollouts' full GAE cannot be invented from post-update boundary values.
Distinguish the raw environment reward from SB3 collector timeout augmentation
in saved rollout rewards. A saved episode-end recurrence can have zero further
continuation while its saved reward already contains a timeout bootstrap. Audit
that recurrence, but do not claim independent recovery of that unsaved tensor.

## Limits, runtime guards and counting

Scientific actions0, executed policy/environment episodes0, gradient updates0.
No live trainer is created. An inference-only loader may construct the serialized
model without an environment; it never learns, samples replay or executes an
action. Runtime guards reject model learn/train, linear update, optimizer step,
policy.act and env.step. A source gate rejects those calls and network imports
in the scientific modules. Unit-only mechanics remain separate from this study.

Maximum5000 new forward head calls, including duplicates, each per observation
(not discounted by batching). Count online/Q or policy-probability prediction,
target prediction and PPO critic prediction separately. Count a complete linear
component calculation additionally and conservatively as one call. At most48
cohort states plus60 reused probe/development states per snapshot gives a worst
case4752 calls: 2592 linear predictions+decompositions, 1224 DQN online/target
predictions, and936 PPO policy/critic predictions. Terminal DQN states omit
unneeded next-target calls. Exact per-model/head observation caching and verified
Phase 5C reuse can reduce actual calls; retain a call journal. Verification and
reporting make zero new model calls. Stop before the ceiling, no hidden retry.

Wall ceiling1200 seconds, artifact ceiling128 MiB, excluding separately timed
baseline/postrun test processes. Preserve a failed receipt and partial journal
on any exception; never overwrite an existing attempt or silently drop evidence.
Fingerprint model weights, optimizer/buffer data and RNG around inference.
No source, input, cohort or threshold changes after registration.

## Interpretation and terminal decision

The primary outputs are descriptive, with all denominators and provenance.
No performance contest, significance threshold or favorable-checkpoint choice.
G0-G5 failures imply IMPLEMENTATION_BLOCKED. Otherwise explicitly assess whether
the data can distinguish the focal DQN/PPO credit mechanisms from different
observable states and optimization history. Missing actual per-update DQN
targets/batch identities, unavailable PPO boundary/GAE history or inadequate exact
state support can require DIAGNOSTIC_COMPLETED_INSUFFICIENT_HISTORICAL_SIGNAL
even when a plausible descriptive pattern is present. Use the narrow-hypothesis
verdict only with a clearly isolated, supported descriptive contrast and a single
specified future test; never established causality. Prefer precise future
instrumentation over an arbitrary bigger budget when mechanisms remain mixed.

Commands, with the recorded installed interpreter and PYTHONPATH=src:
`partial_phase5d validate`, `prepare`, `analyze`, `verify`, `report`.
Run full regressions after analysis, then final verification. No commit or push
unless separately requested. No next training experiment is authorized.
