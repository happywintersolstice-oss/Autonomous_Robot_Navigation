# Phase 5A: bounded learning-feasibility protocol v1

Registered implementation plan, 20 November 2026. Study ID `feasibility-pilot-v1`.
The user brief is `CODEX_PHASE5_LEARNING_FEASIBILITY_PILOT.md`. This authorizes
a diagnostic pilot, not full Phase 5, tuning, a main comparison, final reserve
creation/evaluation, UI work, committing or pushing.

## Immutable baseline and validation

Work starts from feature commit `45f3f389518226d3ab054eb3fd321625f7b2636a`,
on isolated working branch `feature/partial-dynamic-feasibility-v1`. Preserve the
original dirty main workspace and every pre-existing tracked file on the feature
branch. Existing `partial_*` source, task config, smoke recordings, generated
dataset, historical protocols and results remain byte-identical. New modules,
tests, config, docs and evidence use distinct names.

The task stays 10x10, four two-cell occupancy rays, 409 float32 values and the
same 412 affine features for every learner. Actions are Up/Down/Left/Right/Wait;
rewards stay -0.01/action, +10/goal, -5/collision, no timeout penalty; the 200-step
cap remains external truncation. No action mask, planner fallback, hidden map,
remaining obstacle lifetime, coverage reward or anti-Wait override is added.

Run existing partial validation, dataset audit and a fresh manual trace first;
run the complete existing regressions. New fixture/budget/replay/leakage/RNG/
diagnostic tests must pass before registration. `partial_commands validate`
hashes current partial source/config/tests; the pilot additionally hashes its
own YAML, this protocol and full fixture suite. Registration is exclusive and
execution is locked once; no failed attempt can be rerun for free. Freeze source,
config, fixtures, tests and dataset snapshots. Record exact commands, runtime,
dependency versions and all exceptions. The machine-readable master protocol
and each algorithm protocol are written before their first training action.

## Fixture matrix and controls

`phase5a-handwritten-fixtures-v1` is separate from generated research splits.
Every case is connected in base geometry and has a handwritten successful
mechanics trace. All policies receive only the observation; fixture IDs and
hidden schedules remain evaluator/training-management metadata.

| Category | Training cases | Fixed diagnostic cases | Purpose |
| --- | --- | --- | --- |
| A | 8 | 4 | Open grid; balanced cardinal directions, one/two-step training goals; held-out adjacent-goal pairs |
| B | 4 | 4 | Unknown permanent wall with a detour; four rotations |
| C | 4 | 4 | Sole corridor, adjacent blockage active at t=0, clear after 3/5 training or 4/7 diagnostic steps |
| D | 4 | 4 | Two short-detour/long-block and two long-detour/short-block cases, rotated; different training/evaluation goals and durations |
| E | 4 | 4 | Optional exploration branch; blocked/free cell changes out of range along the verification script |

There are 24 training cases and 20 diagnostic cases. Geometry/orientations are
reused between training and diagnostics; start-goal pairs differ. This is
familiar-fixture diagnosis, not unseen-layout generalization. E's script proves
stale-memory revision, but a learner may wait in range or finish before a remote
change; successful E delivery alone does not establish memory reasoning.

Before learning conclusions, run AlwaysWait, Random (seed 23011 + fixed case
index), the unchanged observation-only OnlinePlanner, and handwritten mechanics
scripts on all 20 diagnostic cases. AlwaysWait must time out with 200 Waits;
all scripts must arrive; a violation stops the pilot. Random/planner failures
remain valid observations. Scripts know their pre-written action sequence and
are privileged mechanics checks, never learned or fair operational baselines.

For D only as a post-hoc diagnostic, time-expanded BFS uses the complete nominal
schedule and occupied-cell suppression to compute a clairvoyant minimum action
count within 200. No policy receives this value, route or future information.
Success-only action excess relative to it is information-advantaged hindsight,
not a same-information optimality claim. Failure excess is undefined.

## Finite primary budget and learning gates

One primary training seed, **23001**, per family; no confirmation seed is used.
Maximum **10,000 actions per family**, **40,000 total**, including all stages
and attempted retries. Fixed A checkpoints are **0, 2,000, 6,000, 10,000**,
stopping A at the first checkpoint satisfying the declared gate. Time cap:
**1,200 seconds for the entire pilot** (controls/composition/learning/evaluation),
**240 seconds per algorithm**, measured with a monotonic clock. Total evaluation,
control and composition action reservations cap at **100,000**. Artifact ceiling
is 512 MiB, checked at checkpoint boundaries. No large-study CLI or extra seeds.

- A basic gate: at least 2/4 diagnostic goals, and at least one more success than
  that algorithm's untrained checkpoint. A single initial tie-favored direction
  cannot pass. If A still fails at 10,000, stop that algorithm without B/C/D/E.
- If A passes and at least 2,000 actions remain: train B for 2,000 actions, then
  assess A+B (eight episodes). Require at least 1/4 B successes and 2/4 A retention.
- If that passes and 2,000 actions remain: train equally sampled C+D for 2,000,
  assess A+B+C+D (16 episodes), require at least one C and one D success and 2/4 A.
- If that passes and 2,000 actions remain: train E for 2,000, assess all 20 cases,
  then stop. There is no extra consolidation stage. Unused ceiling is not spent.

Every A checkpoint assesses only four fixed A cases. An A-failing method is not
quietly exposed to difficult C/D training or repeated exploratory evaluations.
The largest learned evaluation schedule is 52 episodes per algorithm, plus 80
control/script episodes and 180 composition-reference episodes. Even at 200
actions each this is under the 100,000-action cap. Inference sees the same
legal information as training; greedy evaluation and exploratory training have
separate ledgers and action distributions. Retain all checkpoints, not just a
best checkpoint. Gate adaptation is predeclared, not hyperparameter selection.

Budget reservations are flushed before execution. Actual stored actions are
audited independently; a crashed/incomplete operation is never relabelled as
a completed episode or reused as free budget. Limits abort and preserve raw
prefixes and checkpoints. Functional exceptions stop subsequent training;
per-algorithm limit aborts are explicitly incomplete, not completed budgets.

## Model and continuation settings

`configs/partial_feasibility_v1.yaml` contains all settings. Linear alpha .1,
gamma .99, epsilon 1 to .1 over the first 80% of the 10,000-action ceiling. A
continuous local RNG and pre-update SARSA continuation are preserved across A
checkpoints; truncation samples a bootstrap behavior action but discards it on
reset. A curriculum transition starts a new episode, marking any prior unfinished
prefix incomplete rather than inventing an environment terminal.

DQN/PPO use the existing builder and shared extractor: 32x32 networks, learning
rate .0003, gamma .99. DQN: replay 10,000, learning starts 100, batch 50, train
frequency four, one gradient step, target interval 250, epsilon 1 to .1 with
exploration_fraction .8. PPO: rollout 100, batch 50, five epochs; other optimizer
defaults are SB3 2.9.2 defaults, captured in checkpoints. Rollouts/frequencies
divide every permitted block exactly, so the action cap cannot be overshot.

Neural continuation uses public `learn(..., reset_num_timesteps=False)` and
`set_env(..., force_reset=True)` at curriculum boundaries; DQN replay persists.
SB3's DQN epsilon schedule is relative to each learn call's cumulative endpoint,
so it can rise when an A continuation endpoint extends. This is declared and
its actual epsilon is logged per action; it is not the linear family's fixed
global schedule. No custom hidden exploration override is inserted. These
family-specific learning rules prevent causal attribution to one factor here.
Each new curriculum environment has a declared seed offset (A 0, B 100, CD 200,
E 300). Save/load and evaluation restore Python/NumPy/Torch RNG state so they
cannot reseed the ongoing learner. Use sequential CPU runs, one Torch thread.

## Diagnostics, composition and conclusions

Save every actual action, observation pair, sensor/memory snapshot, nominal and
realized event, reward and terminal/truncation flag. Keep incomplete training
prefixes separate from complete-episode success rates. Record five-action counts,
Wait fractions/streaks, visits/oscillations, discoveries, episode-length histograms,
recovery opportunities/null denominators, numerical finiteness, updates, replay
goal rewards, rollout counts, checkpoint hashes and runtime/resource measurements.
Record state-specific Q values/probabilities and knowledge-before-action examples.
Replay all records and independently reconcile summaries and reserved budgets.

Controls precede learning. Composition analysis then runs the unchanged online
reference once on each of the existing 180 dynamic training/development cases;
no final data is accessed. Report nominal durations/start times, single-cell
impact statistics, actual observed/relevant events, current disconnections and
detours, route endings before activation and realized suppression. A geometric
near-ray timing proxy on the unimpeded original path is explicitly counterfactual
and ignores dynamic delays/occlusion. Its counts are not actual sensor detections.
The 16/540 statistic has event-impact denominator, not episodes needing Wait.

Show undiscounted and discounted reward examples under unchanged rewards. Test
that adjacent and farther obstacles are distinguishable. A four-state synthetic
XOR/parallelogram probe illustrates affine interaction limits only; its labels
are not asserted optimal navigation actions. Any robot-relative feature or
reward changes are proposals only, requiring a new ID and matched protocol.

Verdict is `IMPLEMENTATION_BLOCKED` for remaining functional/replay/test failure;
`READY_FOR_PHASE5_PROTOCOL_REVIEW` only if all four meet the basic A gate and
engineering/method gates pass; otherwise
`REQUIRES_REPRESENTATION_OR_TRAINING_DIAGNOSIS`. None authorizes further work.
