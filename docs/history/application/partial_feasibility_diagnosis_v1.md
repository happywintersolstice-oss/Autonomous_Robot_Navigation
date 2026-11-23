# Phase 5A: gate verdict and post-hoc diagnosis v1

Measured/analysed 20 November 2026. This note interprets the immutable
`feasibility-pilot-v1` records; it does not amend the registered protocol.

**Verdict: `REQUIRES_REPRESENTATION_OR_TRAINING_DIAGNOSIS`.** No further training,
new final reserve, final evaluation or Phase 6 is authorized by this verdict.

## Required gate rubric

| Gate | Observed evidence | Result |
| --- | --- | --- |
| G1: engineering | 278 baseline regressions, 70 final focused partial gates, 305 final full regressions; sensing/memory/leakage, five actions, terminal/truncation, continuity, checkpoint/RNG and prefix checks | Pass |
| G2: basic learning | At 10,000 actions, Q-learning 1/4 A goals, SARSA 0/4, DQN 1/4, PPO 1/4; untrained counts were 1/4, 1/4, 1/4, 2/4 | Failed the registered improvement gate for every family |
| G3: obstacle reasoning | All mechanics scripts and the observation-only planner completed B/C controls; learners never passed A and received no B/C training/assessment | Not reached for learners; mechanics pass is not learning evidence |
| G4: wait/detour quality | D controls include both tradeoffs; scripts attain the clairvoyant action reference; online planner takes longer detours in the short-block cases | Controls measured; learned decision quality not assessed |
| G5: method integrity | One seed 23001 per family; exact 10,000 each / 40,000 total; 13,100 control/evaluation/composition actions; 324 episodes; all source/checkpoint/prefix/action/value/metric/budget audits passed | Pass |

All four attempts completed their registered A budget, with no retry or
interrupted model run. Each has one genuine incomplete final training episode
prefix at the budget boundary. These four prefixes were retained and excluded
from completed-episode rates; they were never relabelled as timeouts.

## Basic outcomes and competing explanations

| Method | Completed training goals | Collisions | Timeouts | Final A goals / 4 | Final A Wait % | Position revisits | Overlapping A-B-A-B windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q-learning | 126 | 89 | 20 | 1 | 26.79 | 398 | 369 |
| SARSA | 186 | 117 | 13 | 0 | 0.00 | 397 | 171 |
| DQN | 390 | 93 | 10 | 1 | 8.40 | 192 | 191 |
| PPO | 188 | 211 | 1 | 1 | 31.61 | 381 | 151 |

The final behavior is not the earlier functional smoke's universal all-Wait
collapse. It includes excessive waiting, oscillation and collisions. Revisit
and A-B-A-B window counts are position diagnostics, not proofs of an infinite
policy cycle in a memory/history-dependent observation process.

Ranked explanations to investigate, not causal findings:

1. **State-dependent learning and value/credit assignment.** Training produced
   many successful goals, but deterministic decisions on distinct, familiar-grid
   start-goal pairs did not improve over initialization. The generated learning
   curves and knowledge-before-action examples show this directly. Four cases
   are a small diagnostic, not a robust generalization estimate.
2. **Feature scaling, normalization and distribution.** Dense absolute memory
   channels coexist with much smaller relative-coordinate changes. The normalized
   linear update and neural optimization may learn nuisance/global preferences
   before useful goal-conditioned distinctions. This is a hypothesis; no scaling
   ablation was run. Training and diagnostic start-goal pairs differ on the same
   geometry, so memorization/limited transfer is a competing explanation.
3. **Finite interactions and exploration details.** Every method sampled all five
   actions. For example Q-learning's training counts were 1,918 Up, 1,909 Down,
   2,011 Left, 2,049 Right and 2,113 Wait. Missing actions are ruled out, but useful
   state-action coverage is not guaranteed. DQN used its declared continuation
   schedule and linear methods their declared global schedule; they are not
   identical exploration controls. More training was not attempted after the cap.
4. **Reward/cap incentives.** A 200-Wait route returns -2 undiscounted, an immediate
   collision -5.01, and an adjacent goal +9.99. Waiting can be safer than unskilled
   exploration, but arriving is strongly rewarded. Discounted examples are in
   `reward-sanity.json`. These comparisons do not prove the reward alone caused
   the learned behavior. SARSA's final zero-Wait failures particularly preclude
   an all-Wait-only account.

Simple interface explanations are weakened by the successful scripted mechanics,
exact action/reward replay, correct truncation tests, finite changed parameters
and checkpoint decisions matching every saved learned evaluation action. DQN's
final replay buffer contains all 10,000 transitions, including **390 positive
goal-reward transitions**, with 2,475 gradient updates. PPO completed 100 rollouts
and its SB3 counter records 500 epochs; these are not equivalent compute units.
This is not proof that every optimization choice is adequate.

## Analytic capacity qualification: zero extra learning

The separately labelled post-hoc calculation
[`affine-policy-capacity-v1`](../artifacts/partial-dynamic-navigation-v1/feasibility-pilot-v1/derived-analysis/affine-policy-capacity-v1/analysis.json)
uses the existing final two relative-goal features. Its score matrix sets
Up = -relative-row, Down = relative-row, Left = -relative-column,
Right = relative-column, Wait = 0. On each of the four adjacent-goal A
observations, its initial argmax points to the goal.

This is matrix arithmetic on four reset observations: **zero training updates,
zero executed environment actions, and no learned checkpoints loaded or changed**.
It is not an initialized learner, fallback, evaluation competitor or learned
success rate. It demonstrates representability of those policy decisions,
not of the exact optimal Bellman value function or guaranteed TD convergence.
The source, inputs and output have their own post-hoc manifest.

The pre-registered XOR/parallelogram probe remains a valid synthetic interaction
example, but its labels were never claimed optimal navigation actions. Together
the probes support a careful capacity critique, not a claim that raw affine
features necessarily make A impossible. Diagnose A's learning process before
assuming a larger feature set fixes it.

## Scenario-composition findings: explicit denominators

The unchanged online reference was run once on each of the 180 existing dynamic
training/development scenarios: **175 goals and 5 timeouts**, 2,898 actions.
This is a reference audit, not an RL comparison or new final evaluation.

| Measure | Actual count | Denominator / meaning |
| --- | --- | --- |
| Nominal events | 540 | Three per existing dynamic case |
| Initially on a base shortest route | 202 | Events, not guaranteed exposure |
| Unimpeded near-ray timing proxy | 64 | Events; counterfactual geometric proximity, ignoring delay/occlusion |
| Active at unimpeded route arrival | 28 | Events; counterfactual timing |
| Single-cell disconnecting impacts | 16 | Event-removal impacts, not episodes needing Wait |
| Observed at an executed decision | 75 | Distinct case/event observations before actual reference actions |
| Observed and base-route-relevant at a decision | 34 | Distinct case/events under the declared relevance definition |
| No relevant event observed | 147 | Cases out of 180 |
| Temporarily disconnected in a relevant window | 6 | Cases; not an invariant across other trajectories |
| Detour available in a relevant window | 33 | Cases; may overlap disconnection at another time |
| No legal movement in a relevant window | 0 | Cases for this reference; handcrafted C supplies this mechanism |
| Start at/after the reference route ends | 305 | Nominal events out of 540 |
| Actually applied events | 235 | Distinct case/events, trajectory dependent |
| Detected under existing metric, including final scan | 77 | Distinct case/events; differs from pre-decision count |
| Never observed under existing metric | 463 | Nominal events out of 540 |
| Suppressed activations | 11 | Active ticks across 11 distinct case/events |
| Recovery opportunities / recoveries | 34 / 31 | Conditional event denominator, not all cases |

Durations 3/4/5/6/7/8 occur 84/86/102/88/87/93 times. Nominal activation steps
range from 0 to 24; their complete histogram and per-case counts are retained.
These findings justify reviewing future mechanism balance, not rewriting v1.
They **cannot explain the A failures**, because A contains no dynamic events.

For the four D control cases, the clairvoyant time-expanded reference is
9, 11, 9, 11 actions. The information-restricted online planner succeeds in
9, 15, 9, 15; the mechanics scripts use 9, 11, 9, 11. The two four-action excesses
reflect an unknown-duration wait/detour tradeoff. They do not establish that
the online choices were irrational without future information. Failures have
undefined success-only excess, not a fabricated finite regret.

## Timing and resource scope

The complete primary run, including controls, composition and in-run checkpoint
audits/export, took **178.87 seconds**, under the registered 1,200-second cap.
Per-action monotonic guards enforced the 240-second algorithm limit. Observed
first-to-last durable reservation spans are 25.76 s Q-learning, 22.12 s SARSA,
32.43 s DQN and 25.92 s PPO. These spans exclude initialization and final export/
audit tails and include intervening evaluations: they are **not isolated pure
training times**. No missing finer timing is reconstructed as a measurement.

Recorded process peak working set was 1,149,194,240 bytes, including models,
recording and audits, not model-only memory. CPU user/system totals at run end
were 162.22/8.39 seconds. The protocol records Python, dependencies, CPU/platform
metadata and one Torch thread. Separate post-run audits/tests add no primary
learning actions and are not included in the primary-run wall figure.

## Narrow next step: proposal only

Keep the problem at A. Review the saved goal-bearing values and feature/update
scales, then request approval for a small A-only, separately registered
optimization/scaling control with fixed training and diagnostic pairs. Do not
increase map diversity or introduce a recovery controller to hide this failure.
The 644-feature robot-relative ablation and future v2 scenario-quota plan in the
[main results report](partial_feasibility_results.md) are proposals only. The
analytic witness means that feature enlargement should not be assumed necessary
for A. No new extractor, altered reward, extra seed or additional learning run
has been executed.
