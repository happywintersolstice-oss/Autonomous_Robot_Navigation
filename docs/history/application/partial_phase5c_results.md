# Phase 5C: directional coverage diagnosis

**DIAGNOSIS_ONLY_SAMPLING_NOT_JUSTIFIED**. Sampling intervention justified:
**False**. New Phase 5C training actions:
**0**. New executed model/environment assessment actions: **0**.
No conditional training protocol, trial ledger, new development panel or model
checkpoint was created. Historical checkpoint scoring is not a new learning run.

## Scope, provenance and inherited facts

Base `ae3ae73bf7ba5cbe36dcf879125f15fd14c43f4e`, tree `071ba00f2784107c8e419f319914068c9679120e`; working branch
`feature/phase5c-directional-coverage-v1`. Historical [Phase 5B results](partial_phase5b_results.md) and
[Phase 5A diagnosis](partial_feasibility_diagnosis_v1.md) remain unchanged.
Phase 5B's single-seed B1 DQN expanded direction outcomes were 3/4 Up, 4/4 Down,
3/4 Left and 0/4 Right. Those are inherited outcomes, not Phase 5C performance.
The full historical comparison is retained in the CSVs and decision records.

The [analysis plan](partial_phase5c_protocol.md) fixed numerical gate criteria
and all 40 legal probes before computing coverage. Snapshot only new files;
old evidence is referenced by path, Git tree and SHA256. The
[source/evidence manifest](../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/source-and-evidence-manifest.json) lists every
historical file read. No final maps, new reward or feature transformation was used.

## Exact historical denominators

Unique historical training actions reconciled: **48000**.
All 24 checkpoint prefixes are reported (0/2k/6k x eight arms). The 2k prefix is
contained in 6k; do not sum overlapping prefixes as independent samples.
Historical assessment replay: **41705 actions /
480 episodes**. These are recorded actions
replayed mechanically, not newly selected model actions. Score-only synthetic
probe calls: **960**, with no env.step.

Reset counts include empty autoresets, reported separately from completed
episodes and nonterminal prefixes with actions. States hash the entire 409-value
pre-action float32 vector; position-goal pairs and episode-start case IDs are
different denominators. Marginal tables reuse the same cohort; diagonals occupy
their own disjoint category. An unobserved bin has null rates, not zero probability.
See the [data dictionary](../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/data-dictionary.json).

## Final-prefix exposure, both feature arms

| Family | Arm | Resets | Start U | Start D | Start L | Start R | Current R decisions | Near1 R decisions | Near1 selected Right | Goal arrivals Right | Expanded Right /4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q-learning | B0-original | 191 | 50 | 53 | 45 | 43 | 345 | 110 | 31 | 31 | 2 |
| q-learning | B1-relative-x4 | 195 | 52 | 54 | 45 | 44 | 283 | 118 | 21 | 21 | 0 |
| sarsa | B0-original | 203 | 54 | 54 | 49 | 46 | 283 | 89 | 20 | 20 | 1 |
| sarsa | B1-relative-x4 | 223 | 58 | 60 | 56 | 49 | 365 | 129 | 29 | 29 | 3 |
| dqn | B0-original | 245 | 65 | 66 | 62 | 52 | 409 | 98 | 28 | 28 | 0 |
| dqn | B1-relative-x4 | 506 | 149 | 123 | 130 | 104 | 401 | 210 | 88 | 88 | 0 |
| ppo | B0-original | 217 | 58 | 56 | 54 | 49 | 409 | 139 | 20 | 20 | 0 |
| ppo | B1-relative-x4 | 252 | 67 | 68 | 62 | 55 | 381 | 118 | 20 | 20 | 0 |

Training cases already have two examples per direction. Starts are episode
counts; current Right decisions exclude diagonal goals and can follow wandering
from a differently directed reset. Goal arrivals are classified by the final
approach action. Episode-start directions, near-goal choices and rewards therefore
need not have matching proportions. None alone identifies a causal mechanism.

Full reset, action, near-goal and joint coverage for each family/arm/checkpoint is
in [coverage-audit.json](../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/coverage-audit.json),
[reset exposure](../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/reset-exposure.csv),
[direction exposure](../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/direction-exposure.csv), and
[near-goal exposure](../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/near-goal-exposure.csv). Joint bins also condition on
quadrant, edge/interior, distance, visits and discovered-map history.

## Preregistered sampling gate, B1 only at 6k

| Family | Direction | Resets | Share % | Near1 decisions | Goal arrivals | Expanded goals /4 | Reset deficit | Sparse signal | Candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q-learning | Up | 52 | 26.67 | 164 | 34 | 2 | False | False | False |
| q-learning | Down | 54 | 27.69 | 115 | 41 | 1 | False | False | False |
| q-learning | Left | 45 | 23.08 | 136 | 28 | 4 | False | False | False |
| q-learning | Right | 44 | 22.56 | 118 | 21 | 0 | False | False | False |
| sarsa | Up | 58 | 26.01 | 148 | 43 | 1 | False | False | False |
| sarsa | Down | 60 | 26.91 | 146 | 30 | 3 | False | False | False |
| sarsa | Left | 56 | 25.11 | 149 | 32 | 0 | False | False | False |
| sarsa | Right | 49 | 21.97 | 129 | 29 | 3 | False | False | False |
| dqn | Up | 149 | 29.45 | 307 | 108 | 3 | False | False | False |
| dqn | Down | 123 | 24.31 | 214 | 117 | 4 | False | False | False |
| dqn | Left | 130 | 25.69 | 263 | 131 | 3 | False | False | False |
| dqn | Right | 104 | 20.55 | 210 | 88 | 0 | False | False | False |
| ppo | Up | 67 | 26.59 | 145 | 30 | 4 | False | False | False |
| ppo | Down | 68 | 26.98 | 143 | 29 | 1 | False | False | False |
| ppo | Left | 62 | 24.6 | 155 | 27 | 0 | False | False | False |
| ppo | Right | 55 | 21.83 | 118 | 20 | 0 | False | False | False |

The exact criteria and seeded schedule simulations are in the plan and audit.
A sampling candidate needs a reset deficit, failure in that direction, sparse
useful signal and a material planned schedule change simultaneously. The planned
balanced blocks guarantee equal *starts*, not equal transitions, rewards or
visitation. A no-learning schedule simulation cannot guarantee learned coverage.
The gate is engineering triage, not a test of statistical significance.

## Permitted score probes and goal conditioning

| Family | Arm | Goal-directed reset argmax | Reset probes | D4 argmax agreement | Nonidentity D4 probes | Maximum D4 score discrepancy |
| --- | --- | --- | --- | --- | --- | --- |
| q-learning | B0-original | 6 | 24 | 11 | 14 | 0.827928 |
| q-learning | B1-relative-x4 | 14 | 24 | 14 | 14 | 0.666287 |
| sarsa | B0-original | 6 | 24 | 4 | 14 | 1.512962 |
| sarsa | B1-relative-x4 | 9 | 24 | 14 | 14 | 1.3048 |
| dqn | B0-original | 6 | 24 | 2 | 14 | 2.057128 |
| dqn | B1-relative-x4 | 7 | 24 | 14 | 14 | 1.505381 |
| ppo | B0-original | 0 | 24 | 7 | 14 | 0.116245 |
| ppo | B1-relative-x4 | 6 | 24 | 6 | 14 | 0.22269 |

Probe scores are Q values or PPO probabilities, not comparable numerical scales
across families. D4 agreement is descriptive learned symmetry, not a functional
pass requirement. Coordinates, rays, all memory/history grids and action mappings
were transformed consistently; tests compare with real scripted mechanical
histories. Translation probes change absolute location and the legally translated
memory together. They do not isolate a causal location variable.
See [all counterfactual scores](../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/counterfactual-scores.csv),
[probe observations](../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/probe-manifest.json) and
[Right success/failure comparisons](../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/right-goal-cases.csv).

## Historical decision quality and paired feature conditions

| Family | Arm | Original goals /4 | Expanded goals /16 | Up /4 | Down /4 | Left /4 | Right /4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q-learning | B0-original | 1 | 6 | 2 | 2 | 0 | 2 |
| q-learning | B1-relative-x4 | 1 | 7 | 2 | 1 | 4 | 0 |
| sarsa | B0-original | 1 | 5 | 3 | 0 | 1 | 1 |
| sarsa | B1-relative-x4 | 2 | 7 | 1 | 3 | 0 | 3 |
| dqn | B0-original | 1 | 2 | 0 | 1 | 1 | 0 |
| dqn | B1-relative-x4 | 3 | 10 | 3 | 4 | 3 | 0 |
| ppo | B0-original | 0 | 0 | 0 | 0 | 0 | 0 |
| ppo | B1-relative-x4 | 1 | 5 | 4 | 1 | 0 | 0 |

All 0/2k/6k original and expanded cases retain first Q/probabilities, margins,
outcomes, actions, Wait, revisits and oscillations in
[historical decision quality](../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/historical-decision-quality.csv).
[Exact paired cases](../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/historical-paired-cases.csv) compare B0/B1 within a
family on identical cases, not identical training trajectories. Their distributions
differ after actions diverge. No family superiority or new generalization result
is claimed. The old four original and sixteen expanded pairs are already inspected.

## Learning signals, integrity and limitations

Each checkpoint's coverage JSON includes exact goal arrivals by reset case and
pre-action position/goal pair, optimizer counts, recorded epsilon schedules and
stored DQN replay / PPO last-rollout checks. Only the final saved PPO rollout is
available per checkpoint, not its entire rollout history. Positive timeout
bootstrap rewards are not labelled goal rewards. Actual gradient/update records
were read and checked; PPO epochs are not DQN or linear update counts.

Selected actions versus current greedy preference do not reveal the realized
epsilon coin. PPO training is stochastic. SARSA carries an action sampled before
the previous update. Probe sensitivity and coverage associations do not prove
that scaling, reward, sampling or representation caused a failure.

Audit wall: 187.191 seconds; process resource scope:
`{'cpu_user_seconds': 173.140625, 'cpu_system_seconds': 8.671875, 'working_set_bytes': 1415458816, 'peak_working_set_bytes': 2048778240}`. Independent [replay audit](../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/replay-audit.json)
passed, with zero new learning. Historical files remained byte-identical.
Full test counts and original dirty-workspace preservation are recorded in final
delivery verification. Required inherited unit tests can perform tiny mechanical
optimizer checks; they are separate from the zero-action Phase 5C study.

## Decision and one next proposal

No family/direction meets the complete predeclared case-sampling justification. Stop at diagnosis: do not train the conditional C0/C1 arms. This does not prove that ordering never matters or that every state has adequate coverage. It means the specified episode-balanced scheduler is not a defensible first intervention for this evidence. Propose a separately reviewed A-only goal-conditioning/credit-assignment diagnostic; do not execute another optimization or training experiment under this brief.

## Reproduce

From the app directory with the recorded installed Python and PYTHONPATH=src:
`python -m robot_navigation.partial_phase5c verify` checks the completed audit
without training; `report` regenerates these tables. A fresh `prepare` refuses an
existing namespace, and `analyze` refuses an existing lock. Do not delete outputs
to obtain a retry. Windows historical paths may require
`git -c core.longpaths=true`. No commit, push, UI change or main merge is performed
without the separate authorization required by the brief.

## Reviewed interpretation of the Right failures

The preregistered rule found no reset deficit or scarce near-goal/arrival signal
in any B1 direction. This is a refusal of the specified sampling intervention,
not a claim that all state-action coverage is adequate.

DQN B1 had 506 resets: Up149, Down123, Left130 and Right104. Its two Right-start
cases were selected 57 and 47 times, and produced 49 and 38 completed goals
respectively (87 completed goals from those reset cases). Separately, across all
reset cases, there were 401 current-axis Right decisions and 210 decisions with
the goal exactly one cell to the right. Right was selected on 88 of those 210;
the contemporaneous greedy preference was Right on 95. There were 88 actual goal
arrivals from the Right approach direction across all reset cases. These counts
have different denominators and are not interchangeable success rates.

Despite these learning signals, the saved final policy failed all four expanded
Right cases. Their first Right-direction margins were negative. The original
Right case eventually succeeded after 31 actions, also with a negative initial
Right margin. Missing Right labels, a wrong action-ID mapping or a universal
all-Wait policy do not fit these observations. They do not prove any particular
optimization or credit-assignment mechanism caused the failures.

At the fixed illustrative length of 506 episode starts and proposed seed23041,
uniform sampling gives 133 Right starts; balanced blocks give 127. The simulation
would not supply the proposed boost in Right starts at that prefix. This is not
a prediction of how many episodes a new 4,000-action learner would complete.

PPO B1 recorded 118 near-one-cell Right decisions and 20 Right goal arrivals,
but its contemporaneous greedy preference was Right on zero of those decisions.
Stochastic goal-bearing experience and a reliably goal-conditioned greedy policy
are therefore distinct evidence requirements. SARSA and Q-learning likewise have
nonzero direction/arrival signals alongside development failures.

The one next proposal is a separately reviewed A-only goal-conditioning/credit-
assignment diagnosis of how Right-action targets and value contributions change
on permitted states. Do not silently change rewards, optimizers, features or
samplers, and do not start another learning experiment under this brief.

## Final validation and preservation

The inherited baseline passed **366** tests. The corrected new focused suite passed **33**;
full preaudit and postaudit suites each passed **399** with zero failures or skips.
Two earlier failures belonged to the same incomplete synthetic failure-injection
fixture; both receipts and the correction are retained in validation-history.
No baseline source or historical experiment was changed to make those tests pass.

Independent recount/replay passed for **48,000** unique historical training actions,
24 overlapping checkpoint prefixes, **41,705** historical assessment actions and **480**
historical episodes. **960** synthetic score probes executed no environment actions.
New Phase 5C scientific training and policy/environment assessment actions are **0**.

All **9,900** inventoried files present on the reviewed feature branch and all **9,645** inventoried present files
in the original dirty workspace remain byte-identical. Its branch, HEAD, index
and uncommitted status were preserved. Historical evidence used for aggregation
is enumerated in source-and-evidence-manifest; preservation-only hash reads are
also fully enumerated in base-inventory. The new archive contains a minimal flat
source snapshot rather than a copied repository tree.
