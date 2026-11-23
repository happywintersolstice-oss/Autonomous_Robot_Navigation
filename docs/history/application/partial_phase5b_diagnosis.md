# Phase 5B: zero-learning preflight diagnosis

This report was written before the registered Phase 5B trial. It reads existing
[Phase 5A results](partial_feasibility_results.md) and
[Phase 5A diagnosis](partial_feasibility_diagnosis_v1.md) without modifying them.
It is not a Phase 5B learning outcome or evidence of generalization.

## Observed historical evidence

| Family | Mean 412-feature norm | Relative-goal energy fraction | Old training goals | Exact linear checkpoint reconstruction |
| --- | --- | --- | --- | --- |
| q-learning | 9.0143 | 0.2572% | 126 | [2000, 6000, 10000] |
| sarsa | 8.9493 | 0.2522% | 186 | [2000, 6000, 10000] |
| dqn | 9.128 | 0.2215% | 390 | not applicable |
| ppo | 8.8749 | 0.2540% | 188 | not applicable |

The historical source/checkpoint pins passed. Linear arithmetic/RNG replay
exactly reproduced the archived 2k, 6k and 10k weights. All five training actions
were sampled. All four old A reset decisions at 0/2k/6k/10k have saved values,
margins and component breakdowns. Historical optimizer moments are finite;
inference checkpoint files do not retain live gradients, so no historical gradient
measurement is invented. Live Phase 5B optimizer steps will be instrumented.

The full implementation gate passed **366 tests**, with zero failures,
errors or skips. Tests cover cardinal rotations, action/ray ordering, worked
Bellman targets, terminal/truncated bootstrapping and carried SARSA actions,
numpy/Torch parity, source boundaries, hidden-history equivalence, strict arm
loading, paired cases, RNG/optimizer continuity and injected failures.

## Worked targets and action alignment

Action IDs remain 0 Up, 1 Down, 2 Left, 3 Right, 4 Wait. Sensor rays remain
N/S/E/W; their third component is East, not action 2. Greedy ties choose the
first argmax; linear exploratory ties are sampled uniformly. The analytic
relative-goal witness chooses the four correct initial actions. It is a capacity
witness with no learning and no executed witness policy, never an initializer.

| Family | Synthetic transition | Analytic target | Real updated score |
| --- | --- | --- | --- |
| q-learning | nonterminal | 2.96 | 2.96 |
| q-learning | goal | 9.99 | 9.99 |
| q-learning | collision | -5.01 | -5.01 |
| q-learning | external_truncation | 2.96 | 2.96 |
| q-learning | two-step-backup-to-goal | 9.8801 | 9.8801 |
| sarsa | nonterminal | 1.97 | 1.97 |
| sarsa | goal | 9.99 | 9.99 |
| sarsa | collision | -5.01 | -5.01 |
| sarsa | external_truncation | 1.97 | 1.97 |
| sarsa | two-step-backup-to-goal | 9.8801 | 9.8801 |

Synthetic updates use alpha=1 to expose targets directly. A true goal/collision
does not bootstrap. External truncation keeps its real continuing observation
and bootstraps. The two-step backup is -0.01 + 0.99 * 9.99 = 9.8801. These tests
demonstrate implementation semantics, not that a learner acquires a good policy.

| Family | Second transition | Second target | First target |
| --- | --- | --- | --- |
| q-learning | nonterminal | 2.96 | 2.9204 |
| q-learning | goal | 9.99 | 9.8801 |
| q-learning | collision | -5.01 | -0.01 |
| q-learning | external_truncation | 2.96 | 2.9204 |
| sarsa | nonterminal | 2.96 | 2.9204 |
| sarsa | goal | 9.99 | 9.8801 |
| sarsa | collision | -5.01 | -4.9699 |
| sarsa | external_truncation | 2.96 | 2.9204 |

The full two-step synthetic check uses independent permitted feature vectors.
Q-learning takes a zero-valued alternative after the collision backup; SARSA
uses the declared sampled action. Both match the actual update function.

## Hypotheses and preflight decision

No deterministic test has established a baseline interface or optimization bug.
The original A goal directions are representable. Relative-goal energy is small
compared with dense memory channels, so a predeclared factor-four scale is a
plausible signal/conditioning probe. Feature energy alone is not a causal
explanation: normalized updates, state coverage, optimizer dynamics, the finite
budget and start/goal distribution remain alternatives. Rewards are unchanged;
dynamic events cannot explain A failures because A has none.

Proceed only to the frozen two-arm A-only protocol after G2 registration. Do not
enlarge features, change rewards/learning rates, train obstacles or select a new
multiplier. The successful and unsuccessful permitted-information examples,
training-only feature distributions, reset/exploration schedules and exact update
records are archived under the new study's `zero-learning-diagnosis/` directory.
