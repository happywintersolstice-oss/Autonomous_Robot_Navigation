# Phase 5C: diagnosis-only decision

**DIAGNOSIS_ONLY_SAMPLING_NOT_JUSTIFIED**. New Phase 5C training actions: **0**.
Sampling intervention justified: **False**.

The eight training cases already contain two cases per direction. The predeclared
gate requires a material reset deficit, same-direction failure, sparse useful
near-goal/arrival signal, and a meaningful planned schedule correction together.
It does not assume that equal episode starts produce equal state-action coverage.

| Family | Direction | Resets | Share % | Near1 decisions | Goal arrivals | Expanded /4 | Reset deficit | Sparse signal | Candidate |
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

See [complete results](partial_phase5c_results.md), the
[frozen analysis plan](partial_phase5c_protocol.md),
[raw coverage audit](../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/coverage-audit.json),
[schedule simulation](../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/schedule-simulations.json), and
[replay verification](../artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/replay-audit.json).

This is a single-seed historical association and score-probe audit, not a causal
exclusion of sampling effects. Unobserved states have undefined rates. A
direction-balanced reset schedule cannot directly enforce near-goal actions,
useful updates or transfer across absolute positions/history.

No conditional experiment is run when the gate is false. The one next proposal
is a separately reviewed A-only goal-conditioning/credit-assignment diagnostic;
no optimization or training experiment is executed under this diagnosis.

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
