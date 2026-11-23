# Phase 5D read-only findings — 2026-11-21


Verdict: **DIAGNOSTIC_COMPLETED_INSUFFICIENT_HISTORICAL_SIGNAL**. Missing DQN actual optimizer sample/target trajectory, incomplete PPO historical GAE, and development-state support limit unique causal attribution. Descriptive frozen sensitivity is not causal credit assignment.


Branch: `feature/phase5d-goal-credit-diagnosis-v1`. Reviewed HEAD `c1505cc1dab0623b9c7eb86b7bfae9ac3216fa4d`; tree `e6ae906c7736dd0862d5e7ecb066ad31c8660461`. The Phase 5D implementation is uncommitted at report generation. No main merge or push was performed. Historical Phase 5A commit `d3822423ffd6fa3c83f9a26fc15808fa9103e36c`, Phase 5B `ae3ae73bf7ba5cbe36dcf879125f15fd14c43f4e`, Phase 5C parent as above.


This study measures existing Fixture A evidence. It makes no performance improvement claim. Historical assessments are deterministic episodes previously executed; probe/cohort rows are frozen scores, not new navigation results. The fixed outcome-balanced cohort is deliberately selected from distinct near-one-cell historical observations, not a population success estimate.


## Phase 5D read-only findings table


Expanded success counts are ordered Up/Down/Left/Right, each out of four. Probe alignment is a first-argmax Manhattan-reducing choice across 40 legal observations (including D4 histories); cohort alignment uses the registered fixed sample. Exact/approximate support counts refer to the 20 development reset observations in each historical training prefix. Prefixes 0/2k/6k overlap and are never independent repetitions.


| Family | Arm | Checkpoint | Original goals | Expanded goals | Expanded U/D/L/R | Probe aligned | Cohort aligned | Exact supported | Approx supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q-learning | B0-original | 0 | 1/4 | 4/16 | 4/0/0/0 | 14/40 | 12/48 | 0/20 | 0/20 |
| q-learning | B0-original | 2000 | 1/4 | 4/16 | 0/0/0/4 | 19/40 | 12/48 | 0/20 | 7/20 |
| q-learning | B0-original | 6000 | 1/4 | 6/16 | 2/2/0/2 | 19/40 | 26/48 | 0/20 | 7/20 |
| q-learning | B1-relative-x4 | 0 | 1/4 | 4/16 | 4/0/0/0 | 14/40 | 12/48 | 0/20 | 0/20 |
| q-learning | B1-relative-x4 | 2000 | 2/4 | 8/16 | 3/1/3/1 | 25/40 | 21/48 | 0/20 | 7/20 |
| q-learning | B1-relative-x4 | 6000 | 1/4 | 7/16 | 2/1/4/0 | 30/40 | 29/48 | 0/20 | 7/20 |
| sarsa | B0-original | 0 | 1/4 | 4/16 | 4/0/0/0 | 14/40 | 12/48 | 0/20 | 0/20 |
| sarsa | B0-original | 2000 | 2/4 | 7/16 | 3/4/0/0 | 16/40 | 19/48 | 0/20 | 7/20 |
| sarsa | B0-original | 6000 | 1/4 | 5/16 | 3/0/1/1 | 17/40 | 21/48 | 0/20 | 7/20 |
| sarsa | B1-relative-x4 | 0 | 1/4 | 4/16 | 4/0/0/0 | 14/40 | 12/48 | 0/20 | 0/20 |
| sarsa | B1-relative-x4 | 2000 | 1/4 | 5/16 | 1/0/0/4 | 24/40 | 20/48 | 0/20 | 7/20 |
| sarsa | B1-relative-x4 | 6000 | 2/4 | 7/16 | 1/3/0/3 | 25/40 | 34/48 | 0/20 | 7/20 |
| dqn | B0-original | 0 | 1/4 | 4/16 | 0/0/0/4 | 14/40 | 12/48 | 0/20 | 0/20 |
| dqn | B0-original | 2000 | 2/4 | 4/16 | 1/0/2/1 | 19/40 | 23/48 | 0/20 | 7/20 |
| dqn | B0-original | 6000 | 1/4 | 2/16 | 0/1/1/0 | 18/40 | 29/48 | 0/20 | 7/20 |
| dqn | B1-relative-x4 | 0 | 1/4 | 4/16 | 0/0/0/4 | 14/40 | 12/48 | 0/20 | 0/20 |
| dqn | B1-relative-x4 | 2000 | 2/4 | 5/16 | 2/1/1/1 | 23/40 | 24/48 | 0/20 | 7/20 |
| dqn | B1-relative-x4 | 6000 | 3/4 | 10/16 | 3/4/3/0 | 23/40 | 33/48 | 0/20 | 7/20 |
| ppo | B0-original | 0 | 1/4 | 4/16 | 0/0/4/0 | 8/40 | 10/48 | 0/20 | 0/20 |
| ppo | B0-original | 2000 | 0/4 | 0/16 | 0/0/0/0 | 0/40 | 0/48 | 0/20 | 7/20 |
| ppo | B0-original | 6000 | 0/4 | 0/16 | 0/0/0/0 | 4/40 | 2/48 | 0/20 | 7/20 |
| ppo | B1-relative-x4 | 0 | 1/4 | 4/16 | 0/0/4/0 | 7/40 | 11/48 | 0/20 | 0/20 |
| ppo | B1-relative-x4 | 2000 | 2/4 | 6/16 | 3/3/0/0 | 19/40 | 11/48 | 0/20 | 7/20 |
| ppo | B1-relative-x4 | 6000 | 1/4 | 5/16 | 4/1/0/0 | 16/40 | 14/48 | 0/20 | 7/20 |


## Historical behavior and goal rewards


Reset starts, decisions, distinct observations, exact position-goal pairs and arrivals have different denominators. The following table uses all one-cell axis decisions in each nested prefix. Selected reducing/away/Wait reflect exploration behavior; greedy means contemporaneous logged first argmax. The CSV additionally preserves exact state/pair counts and all phase counters. Empty autoresets follow the original Phase 5C accounting.


| Family | Arm | CP | Direction | Starts | Near1 | Reduce/away/Wait | Goal arrivals | Greedy reducing | Ties |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q-learning | B0-original | 0 | Up | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| q-learning | B0-original | 0 | Down | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| q-learning | B0-original | 0 | Left | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| q-learning | B0-original | 0 | Right | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| q-learning | B0-original | 2000 | Up | 24 | 57 | 14/32/11 | 14 | 18 | 0 |
| q-learning | B0-original | 2000 | Down | 21 | 37 | 8/22/7 | 8 | 2 | 0 |
| q-learning | B0-original | 2000 | Left | 27 | 68 | 10/36/22 | 10 | 16 | 3 |
| q-learning | B0-original | 2000 | Right | 22 | 48 | 11/25/12 | 11 | 6 | 0 |
| q-learning | B0-original | 6000 | Up | 50 | 93 | 27/51/15 | 27 | 28 | 0 |
| q-learning | B0-original | 6000 | Down | 53 | 124 | 20/84/20 | 20 | 15 | 0 |
| q-learning | B0-original | 6000 | Left | 45 | 119 | 26/53/40 | 26 | 32 | 3 |
| q-learning | B0-original | 6000 | Right | 43 | 110 | 31/48/31 | 31 | 34 | 0 |
| q-learning | B1-relative-x4 | 0 | Up | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| q-learning | B1-relative-x4 | 0 | Down | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| q-learning | B1-relative-x4 | 0 | Left | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| q-learning | B1-relative-x4 | 0 | Right | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| q-learning | B1-relative-x4 | 2000 | Up | 18 | 44 | 10/26/8 | 10 | 13 | 0 |
| q-learning | B1-relative-x4 | 2000 | Down | 18 | 41 | 8/26/7 | 8 | 6 | 0 |
| q-learning | B1-relative-x4 | 2000 | Left | 24 | 56 | 10/33/13 | 10 | 18 | 3 |
| q-learning | B1-relative-x4 | 2000 | Right | 20 | 46 | 6/26/14 | 6 | 6 | 0 |
| q-learning | B1-relative-x4 | 6000 | Up | 52 | 164 | 34/92/38 | 34 | 33 | 0 |
| q-learning | B1-relative-x4 | 6000 | Down | 54 | 115 | 41/56/18 | 41 | 40 | 0 |
| q-learning | B1-relative-x4 | 6000 | Left | 45 | 136 | 28/80/28 | 28 | 34 | 3 |
| q-learning | B1-relative-x4 | 6000 | Right | 44 | 118 | 21/74/23 | 21 | 20 | 0 |
| sarsa | B0-original | 0 | Up | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| sarsa | B0-original | 0 | Down | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| sarsa | B0-original | 0 | Left | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| sarsa | B0-original | 0 | Right | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| sarsa | B0-original | 2000 | Up | 23 | 43 | 10/23/10 | 10 | 12 | 0 |
| sarsa | B0-original | 2000 | Down | 19 | 47 | 9/22/16 | 9 | 10 | 0 |
| sarsa | B0-original | 2000 | Left | 27 | 51 | 11/33/7 | 11 | 18 | 3 |
| sarsa | B0-original | 2000 | Right | 21 | 36 | 8/25/3 | 8 | 7 | 0 |
| sarsa | B0-original | 6000 | Up | 54 | 102 | 33/55/14 | 33 | 46 | 0 |
| sarsa | B0-original | 6000 | Down | 54 | 108 | 24/54/30 | 24 | 35 | 0 |
| sarsa | B0-original | 6000 | Left | 49 | 89 | 18/61/10 | 18 | 22 | 3 |
| sarsa | B0-original | 6000 | Right | 46 | 89 | 20/52/17 | 20 | 18 | 0 |
| sarsa | B1-relative-x4 | 0 | Up | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| sarsa | B1-relative-x4 | 0 | Down | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| sarsa | B1-relative-x4 | 0 | Left | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| sarsa | B1-relative-x4 | 0 | Right | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| sarsa | B1-relative-x4 | 2000 | Up | 23 | 57 | 16/35/6 | 16 | 29 | 0 |
| sarsa | B1-relative-x4 | 2000 | Down | 20 | 52 | 7/30/15 | 7 | 5 | 0 |
| sarsa | B1-relative-x4 | 2000 | Left | 27 | 62 | 14/38/10 | 14 | 24 | 3 |
| sarsa | B1-relative-x4 | 2000 | Right | 22 | 48 | 9/28/11 | 9 | 15 | 0 |
| sarsa | B1-relative-x4 | 6000 | Up | 58 | 148 | 43/87/18 | 43 | 66 | 0 |
| sarsa | B1-relative-x4 | 6000 | Down | 60 | 146 | 30/90/26 | 30 | 32 | 0 |
| sarsa | B1-relative-x4 | 6000 | Left | 56 | 149 | 32/96/21 | 32 | 43 | 3 |
| sarsa | B1-relative-x4 | 6000 | Right | 49 | 129 | 29/71/29 | 29 | 38 | 0 |
| dqn | B0-original | 0 | Up | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| dqn | B0-original | 0 | Down | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| dqn | B0-original | 0 | Left | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| dqn | B0-original | 0 | Right | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| dqn | B0-original | 2000 | Up | 24 | 34 | 12/18/4 | 12 | 5 | 0 |
| dqn | B0-original | 2000 | Down | 20 | 28 | 7/18/3 | 7 | 5 | 0 |
| dqn | B0-original | 2000 | Left | 27 | 54 | 19/15/20 | 19 | 23 | 0 |
| dqn | B0-original | 2000 | Right | 22 | 33 | 9/24/0 | 9 | 9 | 0 |
| dqn | B0-original | 6000 | Up | 65 | 103 | 37/43/23 | 37 | 29 | 0 |
| dqn | B0-original | 6000 | Down | 66 | 102 | 40/45/17 | 40 | 52 | 0 |
| dqn | B0-original | 6000 | Left | 62 | 118 | 54/31/33 | 54 | 65 | 0 |
| dqn | B0-original | 6000 | Right | 52 | 98 | 28/57/13 | 28 | 26 | 0 |
| dqn | B1-relative-x4 | 0 | Up | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| dqn | B1-relative-x4 | 0 | Down | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| dqn | B1-relative-x4 | 0 | Left | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| dqn | B1-relative-x4 | 0 | Right | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| dqn | B1-relative-x4 | 2000 | Up | 26 | 46 | 14/26/6 | 14 | 8 | 0 |
| dqn | B1-relative-x4 | 2000 | Down | 21 | 57 | 17/27/13 | 17 | 24 | 0 |
| dqn | B1-relative-x4 | 2000 | Left | 27 | 48 | 17/23/8 | 17 | 18 | 0 |
| dqn | B1-relative-x4 | 2000 | Right | 23 | 66 | 12/33/21 | 12 | 12 | 0 |
| dqn | B1-relative-x4 | 6000 | Up | 149 | 307 | 108/137/62 | 108 | 119 | 0 |
| dqn | B1-relative-x4 | 6000 | Down | 123 | 214 | 117/78/19 | 117 | 142 | 0 |
| dqn | B1-relative-x4 | 6000 | Left | 130 | 263 | 131/76/56 | 131 | 141 | 0 |
| dqn | B1-relative-x4 | 6000 | Right | 104 | 210 | 88/93/29 | 88 | 95 | 0 |
| ppo | B0-original | 0 | Up | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| ppo | B0-original | 0 | Down | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| ppo | B0-original | 0 | Left | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| ppo | B0-original | 0 | Right | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| ppo | B0-original | 2000 | Up | 16 | 37 | 8/22/7 | 8 | 2 | 0 |
| ppo | B0-original | 2000 | Down | 18 | 54 | 5/35/14 | 5 | 0 | 0 |
| ppo | B0-original | 2000 | Left | 23 | 51 | 6/34/11 | 6 | 22 | 0 |
| ppo | B0-original | 2000 | Right | 17 | 33 | 5/20/8 | 5 | 0 | 0 |
| ppo | B0-original | 6000 | Up | 58 | 135 | 27/76/32 | 27 | 2 | 0 |
| ppo | B0-original | 6000 | Down | 56 | 145 | 22/80/43 | 22 | 0 | 0 |
| ppo | B0-original | 6000 | Left | 54 | 137 | 12/91/34 | 12 | 22 | 0 |
| ppo | B0-original | 6000 | Right | 49 | 139 | 20/80/39 | 20 | 0 | 0 |
| ppo | B1-relative-x4 | 0 | Up | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| ppo | B1-relative-x4 | 0 | Down | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| ppo | B1-relative-x4 | 0 | Left | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| ppo | B1-relative-x4 | 0 | Right | 0 | 0 | 0/0/0 | 0 | 0 | 0 |
| ppo | B1-relative-x4 | 2000 | Up | 20 | 53 | 8/32/13 | 8 | 22 | 0 |
| ppo | B1-relative-x4 | 2000 | Down | 19 | 56 | 11/31/14 | 11 | 8 | 0 |
| ppo | B1-relative-x4 | 2000 | Left | 24 | 62 | 11/43/8 | 11 | 22 | 0 |
| ppo | B1-relative-x4 | 2000 | Right | 20 | 38 | 5/25/8 | 5 | 0 | 0 |
| ppo | B1-relative-x4 | 6000 | Up | 67 | 145 | 30/79/36 | 30 | 73 | 0 |
| ppo | B1-relative-x4 | 6000 | Down | 68 | 143 | 29/86/28 | 29 | 33 | 0 |
| ppo | B1-relative-x4 | 6000 | Left | 62 | 155 | 27/103/25 | 27 | 24 | 0 |
| ppo | B1-relative-x4 | 6000 | Right | 55 | 118 | 20/74/24 | 20 | 0 | 0 |


## Actual updates and checkpoint credit


All 24,000 recorded linear updates passed TD-error and normalized update-norm checks. 23,974 targets could also be independently recovered from terminal rewards or adjacent contemporaneous scores after analytically removing the already-recorded increment. No weights were updated by this calculation. Remaining truncated/end-of-stream bootstraps are explicitly unavailable for independent verification; their actual logged targets remain preserved. The mechanics are alpha=0.1, gamma=0.99, increment alpha×error×phi/(phi·phi). SARSA uses its logged sampled next action, never max Q. Every linear probe, case and cohort score decomposes into all ten feature groups and reconstructs Q within tolerance.


DQN online and target snapshots were compared at all checkpoints on the fixed cohort and all existing probes/cases. Cohort residuals use r + .99×(1−terminated)×max target-Q(next) − online-Q(chosen). SB3 masks replay timeout dones, so external truncation bootstraps. Every resulting target/residual is snapshot-only, even if its counters coincide. Positive terminal goal reward 9.99 is observed separately; it does not establish historical replay credit propagation or a greedy Right preference. Historical per-batch sample IDs, target tensors and intermediate target-network weights are absent.


PPO rows contain probabilities, entropy, Right log odds and V(s); no Q(s,a). All pre-action collection probabilities and V are logged. Saved last-rollout values/log probabilities were checked against those logs; returns=advantages+values and interior GAE recurrence were checked independently. Only the final 100 transitions at 2k and 6k are retained per arm. An unsaved pre-update final bootstrap cannot be replaced by the post-update critic or inferred from the advantage under verification. Full GAE history and batch-normalized advantages are unavailable.


PPO timeout rollout rewards may include the historical collector bootstrap augmentation. Raw environment reward and saved buffer reward are kept distinct in buffer-audit.json; an episode-end zero-continuation recurrence check does not independently recover the unsaved timeout bootstrap tensor.


| Family | Arm | CP | Saved rollout/replay records | Boundary / missingness |
| --- | --- | --- | --- | --- |
| dqn | B0-original | 0 | 0 | not recorded |
| dqn | B0-original | 2000 | 2000 | not recorded |
| dqn | B0-original | 6000 | 6000 | not recorded |
| dqn | B1-relative-x4 | 0 | 0 | not recorded |
| dqn | B1-relative-x4 | 2000 | 2000 | not recorded |
| dqn | B1-relative-x4 | 6000 | 6000 | not recorded |
| ppo | B0-original | 0 | 0 | not-observed |
| ppo | B0-original | 2000 | 100 | pre-update last bootstrap value not recorded |
| ppo | B0-original | 6000 | 100 | pre-update last bootstrap value not recorded |
| ppo | B1-relative-x4 | 0 | 0 | not-observed |
| ppo | B1-relative-x4 | 2000 | 100 | pre-update last bootstrap value not recorded |
| ppo | B1-relative-x4 | 6000 | 100 | pre-update last bootstrap value not recorded |


## All DQN B1 final development cases


The complete expanded panel below includes all four Right failures and the other twelve cases; original cases are retained as well. Scores use action order Up, Down, Left, Right, Wait. Margin is best reducing score minus best nonreducing score. All 480 family/arm/checkpoint cases, full 409-value pre-action observations, raw rays ordered North/South/East/West, known-cell counts, historical first actions, outcomes and trace references are in case-decisions.csv.


| Panel | Case | Direction | Distance | Position→goal | Scores U/D/L/R/W | Argmax | Goal margin | Old first action | Historical result/actions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | phase5a-handwritten-fixtures-v1-A-evaluation-00 | Up | 1 | [3,3,2,3] | [5.342010974884033,5.17048978805542,5.353381156921387,4.845742225646973,4.894524097442627] | 2 | -0.01137 | 2 | timeout/200 |
| original | phase5a-handwritten-fixtures-v1-A-evaluation-01 | Down | 1 | [6,6,7,6] | [4.845763683319092,5.407363414764404,5.163649559020996,4.650794982910156,4.76205587387085] | 1 | 0.24371 | 1 | success/1 |
| original | phase5a-handwritten-fixtures-v1-A-evaluation-02 | Left | 1 | [6,3,6,2] | [4.561523914337158,4.7238030433654785,4.893834114074707,4.513018608093262,4.495910167694092] | 2 | 0.17003 | 2 | success/1 |
| original | phase5a-handwritten-fixtures-v1-A-evaluation-03 | Right | 1 | [3,6,3,7] | [4.1964192390441895,4.0902605056762695,4.4573163986206055,4.048254013061523,4.054013729095459] | 2 | -0.40906 | 2 | success/31 |
| expanded | phase5b-open-development-pairs-v1-00 | Up | 1 | [4,3,3,3] | [5.190838813781738,5.151675224304199,5.07490348815918,4.781816005706787,4.807971477508545] | 0 | 0.039164 | 0 | success/1 |
| expanded | phase5b-open-development-pairs-v1-01 | Up | 1 | [1,7,0,7] | [4.971711158752441,5.263279438018799,5.155325889587402,4.754491806030273,4.765232563018799] | 1 | -0.29157 | 1 | timeout/200 |
| expanded | phase5b-open-development-pairs-v1-02 | Up | 2 | [7,4,5,4] | [5.867574691772461,5.924413204193115,6.049056053161621,5.347897052764893,5.401106834411621] | 2 | -0.18148 | 2 | success/12 |
| expanded | phase5b-open-development-pairs-v1-03 | Up | 2 | [9,6,7,6] | [6.55000638961792,6.462389945983887,6.453848838806152,5.822038650512695,5.936952114105225] | 0 | 0.087616 | 0 | success/2 |
| expanded | phase5b-open-development-pairs-v1-04 | Down | 1 | [3,6,4,6] | [4.392143726348877,4.451632976531982,4.851754665374756,4.1874494552612305,4.265254974365234] | 2 | -0.40012 | 2 | success/163 |
| expanded | phase5b-open-development-pairs-v1-05 | Down | 1 | [6,0,7,0] | [5.5360798835754395,5.902746200561523,5.679421424865723,5.323403835296631,5.320037364959717] | 1 | 0.22332 | 1 | success/1 |
| expanded | phase5b-open-development-pairs-v1-06 | Down | 2 | [6,9,8,9] | [6.520893096923828,7.19391393661499,6.971085548400879,5.886634349822998,6.140740871429443] | 1 | 0.22283 | 1 | success/2 |
| expanded | phase5b-open-development-pairs-v1-07 | Down | 2 | [0,2,2,2] | [6.308879375457764,6.715605735778809,6.467911720275879,5.781352996826172,5.89371395111084] | 1 | 0.24769 | 1 | success/2 |
| expanded | phase5b-open-development-pairs-v1-08 | Left | 1 | [7,1,7,0] | [5.841541767120361,5.848417282104492,6.339439868927002,5.487010955810547,5.4828715324401855] | 2 | 0.49102 | 2 | success/1 |
| expanded | phase5b-open-development-pairs-v1-09 | Left | 1 | [5,7,5,6] | [5.980350017547607,6.342552185058594,6.213550567626953,5.337522506713867,5.522459983825684] | 1 | -0.129 | 1 | success/17 |
| expanded | phase5b-open-development-pairs-v1-10 | Left | 2 | [0,4,0,2] | [6.304074287414551,6.612359046936035,6.521188735961914,5.742966651916504,5.8667168617248535] | 1 | -0.09117 | 1 | success/8 |
| expanded | phase5b-open-development-pairs-v1-11 | Left | 2 | [0,7,0,5] | [5.364162921905518,6.071475028991699,5.839334964752197,5.257678031921387,5.301231384277344] | 1 | -0.23214 | 1 | timeout/200 |
| expanded | phase5b-open-development-pairs-v1-12 | Right | 1 | [5,6,5,7] | [4.760051727294922,5.066084861755371,4.9632649421691895,4.487276077270508,4.575011730194092] | 1 | -0.57881 | 1 | timeout/200 |
| expanded | phase5b-open-development-pairs-v1-13 | Right | 1 | [2,0,2,1] | [5.879965782165527,5.884148597717285,5.465553283691406,5.373702049255371,5.3826680183410645] | 1 | -0.51045 | 1 | timeout/200 |
| expanded | phase5b-open-development-pairs-v1-14 | Right | 2 | [0,5,0,7] | [5.166350841522217,5.776038646697998,5.531229019165039,5.260280609130859,5.1500773429870605] | 1 | -0.51576 | 1 | timeout/200 |
| expanded | phase5b-open-development-pairs-v1-15 | Right | 2 | [8,3,8,5] | [5.428630352020264,5.053633213043213,5.151712894439697,5.1277689933776855,4.976288318634033] | 0 | -0.30086 | 0 | timeout/200 |


The DQN B1 Right training reset count is 104; current-axis near-Right decisions are 210; Right-arrival goals are 88. These are distinct counts, not interchangeable success rates. The four final expanded Right cases are 0/4, despite actual reward-bearing Right actions during training. Training start cases and development positions/history differ. PPO stochastic reward arrivals likewise cannot be read as deterministic argmax success.


## Legal goal sensitivity and symmetry


All 24 reset probes and 16 D4 histories are exactly the Phase 5C observations. Goal-only pairs preserve position and all history/ray fields; goal cells must be unvisited and different from current position. Translations and D4 transformations also change absolute position and memory; they cannot identify a pure isolated memory effect. No arbitrary memory-cell edits or new histories were constructed. D4 disagreements preserve ties and action permutations. The table reports means within each family/arm/checkpoint only; Q magnitudes and PPO probabilities are never compared across families.


| Family | Arm | CP | Comparison | Pairs | Mean absolute score difference | Argmax changes / equivariant tie sets |
| --- | --- | --- | --- | --- | --- | --- |
| q-learning | B0-original | 0 | legal goal-only | 84 | 0 | 0 |
| q-learning | B0-original | 0 | D4 legal history | 16 | 0 | 16 |
| q-learning | B0-original | 2000 | legal goal-only | 84 | 0.014017 | 0 |
| q-learning | B0-original | 2000 | D4 legal history | 16 | 0.30897 | 3 |
| q-learning | B0-original | 6000 | legal goal-only | 84 | 0.02096 | 0 |
| q-learning | B0-original | 6000 | D4 legal history | 16 | 0.26426 | 13 |
| q-learning | B1-relative-x4 | 0 | legal goal-only | 84 | 0 | 0 |
| q-learning | B1-relative-x4 | 0 | D4 legal history | 16 | 0 | 16 |
| q-learning | B1-relative-x4 | 2000 | legal goal-only | 84 | 0.086892 | 21 |
| q-learning | B1-relative-x4 | 2000 | D4 legal history | 16 | 0.21775 | 16 |
| q-learning | B1-relative-x4 | 6000 | legal goal-only | 84 | 0.10792 | 56 |
| q-learning | B1-relative-x4 | 6000 | D4 legal history | 16 | 0.18649 | 16 |
| sarsa | B0-original | 0 | legal goal-only | 84 | 0 | 0 |
| sarsa | B0-original | 0 | D4 legal history | 16 | 0 | 16 |
| sarsa | B0-original | 2000 | legal goal-only | 84 | 0.0099451 | 0 |
| sarsa | B0-original | 2000 | D4 legal history | 16 | 0.2412 | 9 |
| sarsa | B0-original | 6000 | legal goal-only | 84 | 0.019896 | 0 |
| sarsa | B0-original | 6000 | D4 legal history | 16 | 0.41276 | 6 |
| sarsa | B1-relative-x4 | 0 | legal goal-only | 84 | 0 | 0 |
| sarsa | B1-relative-x4 | 0 | D4 legal history | 16 | 0 | 16 |
| sarsa | B1-relative-x4 | 2000 | legal goal-only | 84 | 0.070693 | 22 |
| sarsa | B1-relative-x4 | 2000 | D4 legal history | 16 | 0.21386 | 15 |
| sarsa | B1-relative-x4 | 6000 | legal goal-only | 84 | 0.12095 | 19 |
| sarsa | B1-relative-x4 | 6000 | D4 legal history | 16 | 0.28341 | 16 |
| dqn | B0-original | 0 | legal goal-only | 84 | 0.0013124 | 0 |
| dqn | B0-original | 0 | D4 legal history | 16 | 0.099338 | 4 |
| dqn | B0-original | 2000 | legal goal-only | 84 | 0.016275 | 7 |
| dqn | B0-original | 2000 | D4 legal history | 16 | 0.25237 | 9 |
| dqn | B0-original | 6000 | legal goal-only | 84 | 0.20112 | 0 |
| dqn | B0-original | 6000 | D4 legal history | 16 | 0.9487 | 4 |
| dqn | B1-relative-x4 | 0 | legal goal-only | 84 | 0.0027937 | 0 |
| dqn | B1-relative-x4 | 0 | D4 legal history | 16 | 0.10178 | 4 |
| dqn | B1-relative-x4 | 2000 | legal goal-only | 84 | 0.025322 | 7 |
| dqn | B1-relative-x4 | 2000 | D4 legal history | 16 | 0.11171 | 15 |
| dqn | B1-relative-x4 | 6000 | legal goal-only | 84 | 0.2532 | 7 |
| dqn | B1-relative-x4 | 6000 | D4 legal history | 16 | 0.50257 | 16 |
| ppo | B0-original | 0 | legal goal-only | 84 | 3.3252e-05 | 15 |
| ppo | B0-original | 0 | D4 legal history | 16 | 0.00073147 | 6 |
| ppo | B0-original | 2000 | legal goal-only | 84 | 0.00049186 | 0 |
| ppo | B0-original | 2000 | D4 legal history | 16 | 0.021311 | 16 |
| ppo | B0-original | 6000 | legal goal-only | 84 | 0.0008414 | 0 |
| ppo | B0-original | 6000 | D4 legal history | 16 | 0.032102 | 9 |
| ppo | B1-relative-x4 | 0 | legal goal-only | 84 | 0.00011051 | 15 |
| ppo | B1-relative-x4 | 0 | D4 legal history | 16 | 0.00077176 | 5 |
| ppo | B1-relative-x4 | 2000 | legal goal-only | 84 | 0.0020417 | 16 |
| ppo | B1-relative-x4 | 2000 | D4 legal history | 16 | 0.018409 | 8 |
| ppo | B1-relative-x4 | 6000 | legal goal-only | 84 | 0.010463 | 0 |
| ppo | B1-relative-x4 | 6000 | D4 legal history | 16 | 0.043545 | 8 |


The full score records include Manhattan distance, quadrant, boundary, visits, known cells, elapsed and memory-age matching bins. Exact equality hashes all 409 float32 values. Exact position-goal matches do not imply identical visible history. Approximate matches require identical registered coarse bins and mean raw observation L1≤0.05; these are not exchangeable randomized controls. Zero support means unresolved due to support, never evidence of no effect. Repeated transitions and distinct observation counts are both retained.


## Limits and next step


Before further learning, register instrumentation for replay sample IDs, online/target fingerprints and actual TD targets; PPO collection bootstrap values, rollout IDs, returns/advantages and batch membership. No new training authorized. This is an instrumentation recommendation, not authorization for training or selection of a reward/feature intervention. Association between frozen scores and rewards cannot isolate representation, sampling, optimization or credit assignment. Multiple exploratory contrasts have no confirmatory significance claim. No obstacle/dynamic generalization or final/test performance is measured.


## Validation, preservation and resources


All G0–G5 gates passed. Inherited baseline: 399 tests. New full regression run: 424 tests, 0 failures, 0 errors. Test-only fixtures are mechanical validation, not scientific learning evidence; inherited tests may run their original tiny mechanics. The first focused run had two synthetic-fixture dtype failures (23 passed); both were fixed before registration, followed by 25 passing focused tests. Validation history preserves the failed receipt. Both original main working-tree inventory and the reviewed feature baseline are preserved by byte hashes; historical delivery pins remain unchanged.


| Quantity | Actual |
| --- | --- |
| historical_unique_training_actions | 48000 |
| historical_assessment_actions | 41705 |
| historical_assessment_episodes | 480 |
| historical_phase5c_probe_calls_reused | 960 |
| historical_case_vectors_reused | 480 |
| new_forward_calls | 3786 |
| new_training_actions | 0 |
| new_policy_environment_actions | 0 |
| new_probes | 0 |
| Analysis wall seconds | 61.418 |
| Score-call ceiling | 5000 |
| Analysis wall ceiling seconds | 1200 |
| Artifact ceiling bytes | 134217728 |


Baseline/postrun test timing is recorded separately in JUnit/receipts, outside the scientific analysis wall. Weights, optimizer/buffer content and RNG were protected during each inference context; all original checkpoint files remain byte-identical. Independent verification recomputes arithmetic/hashes and consumes zero new score calls.


## Evidence and reproduction


[Registered protocol](partial_phase5d_protocol.md). Evidence namespace: [`phase5d-goal-credit-diagnosis-v1`](../artifacts/partial-dynamic-navigation-v1/phase5d-goal-credit-diagnosis-v1/diagnostic-summary.json). Inspect analysis-plan.json/seal, source-and-evidence-manifest.json, evidence-availability.json, credit-metrics.csv, linear-credit-audit.csv, matched-states.csv, case-decisions.csv, direction-coverage.csv, counterfactual-comparisons.csv, scores.json, buffer-audit.json, forward-calls.jsonl, validation-gates.json and output-hashes.json.


Run from this application root with its installed interpreter and `PYTHONPATH=src`: `python -m robot_navigation.partial_phase5d verify` and `... report` are read-only rescoring-free checks. `prepare`/`analyze` are exclusive one-shot commands and reject an existing namespace/lock; do not rerun or overwrite the registered study. The raw historical trace/checkpoint paths and hashes are in the sealed evidence manifest.



## Fixed historical cohort by direction


The same selected observations are scored at all three snapshots. Counts below are cohort support, not independent episodes or representative success rates. Right margin is Right minus best other; goal margin is goal-reducing minus best other. DQN residuals are frozen counterfactual calculations only.


| Family | Arm | CP | Current direction | States | Observed goal records | Snapshot aligned | Mean Right margin | Mean goal margin | Mean snapshot TD residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q-learning | B0-original | 0 | Up | 12 | 6 | 12 | 0 | 0 | unobserved |
| q-learning | B0-original | 0 | Down | 12 | 6 | 0 | 0 | 0 | unobserved |
| q-learning | B0-original | 0 | Left | 12 | 6 | 0 | 0 | 0 | unobserved |
| q-learning | B0-original | 0 | Right | 12 | 6 | 0 | 0 | 0 | unobserved |
| q-learning | B0-original | 2000 | Up | 12 | 6 | 0 | 0.28875 | -0.40303 | unobserved |
| q-learning | B0-original | 2000 | Down | 12 | 6 | 0 | 0.15874 | -0.73041 | unobserved |
| q-learning | B0-original | 2000 | Left | 12 | 6 | 5 | -0.009552 | -0.36423 | unobserved |
| q-learning | B0-original | 2000 | Right | 12 | 6 | 7 | 0.14823 | 0.14823 | unobserved |
| q-learning | B0-original | 6000 | Up | 12 | 6 | 5 | -0.21382 | -0.036547 | unobserved |
| q-learning | B0-original | 6000 | Down | 12 | 6 | 9 | -0.22432 | 0.10961 | unobserved |
| q-learning | B0-original | 6000 | Left | 12 | 6 | 5 | -0.23017 | -0.32597 | unobserved |
| q-learning | B0-original | 6000 | Right | 12 | 6 | 7 | 0.012721 | 0.012721 | unobserved |
| q-learning | B1-relative-x4 | 0 | Up | 12 | 6 | 12 | 0 | 0 | unobserved |
| q-learning | B1-relative-x4 | 0 | Down | 12 | 6 | 0 | 0 | 0 | unobserved |
| q-learning | B1-relative-x4 | 0 | Left | 12 | 6 | 0 | 0 | 0 | unobserved |
| q-learning | B1-relative-x4 | 0 | Right | 12 | 6 | 0 | 0 | 0 | unobserved |
| q-learning | B1-relative-x4 | 2000 | Up | 12 | 6 | 3 | -0.29482 | -0.082604 | unobserved |
| q-learning | B1-relative-x4 | 2000 | Down | 12 | 6 | 1 | -0.30786 | -0.17516 | unobserved |
| q-learning | B1-relative-x4 | 2000 | Left | 12 | 6 | 10 | -0.63412 | 0.27413 | unobserved |
| q-learning | B1-relative-x4 | 2000 | Right | 12 | 6 | 7 | -0.0078432 | -0.0078432 | unobserved |
| q-learning | B1-relative-x4 | 6000 | Up | 12 | 6 | 7 | -0.49775 | 0.1021 | unobserved |
| q-learning | B1-relative-x4 | 6000 | Down | 12 | 6 | 5 | -0.44116 | -0.048987 | unobserved |
| q-learning | B1-relative-x4 | 6000 | Left | 12 | 6 | 10 | -0.69285 | 0.12846 | unobserved |
| q-learning | B1-relative-x4 | 6000 | Right | 12 | 6 | 7 | -0.13598 | -0.13598 | unobserved |
| sarsa | B0-original | 0 | Up | 12 | 6 | 12 | 0 | 0 | unobserved |
| sarsa | B0-original | 0 | Down | 12 | 6 | 0 | 0 | 0 | unobserved |
| sarsa | B0-original | 0 | Left | 12 | 6 | 0 | 0 | 0 | unobserved |
| sarsa | B0-original | 0 | Right | 12 | 6 | 0 | 0 | 0 | unobserved |
| sarsa | B0-original | 2000 | Up | 12 | 6 | 7 | -0.23072 | -0.054648 | unobserved |
| sarsa | B0-original | 2000 | Down | 12 | 6 | 5 | -0.34153 | -0.0071153 | unobserved |
| sarsa | B0-original | 2000 | Left | 12 | 6 | 3 | -0.28316 | -0.11698 | unobserved |
| sarsa | B0-original | 2000 | Right | 12 | 6 | 4 | -0.13861 | -0.13861 | unobserved |
| sarsa | B0-original | 6000 | Up | 12 | 6 | 12 | -1.5081 | 0.75513 | unobserved |
| sarsa | B0-original | 6000 | Down | 12 | 6 | 4 | -1.1921 | -0.51928 | unobserved |
| sarsa | B0-original | 6000 | Left | 12 | 6 | 3 | -1.3624 | -0.23357 | unobserved |
| sarsa | B0-original | 6000 | Right | 12 | 6 | 2 | -0.89065 | -0.89065 | unobserved |
| sarsa | B1-relative-x4 | 0 | Up | 12 | 6 | 12 | 0 | 0 | unobserved |
| sarsa | B1-relative-x4 | 0 | Down | 12 | 6 | 0 | 0 | 0 | unobserved |
| sarsa | B1-relative-x4 | 0 | Left | 12 | 6 | 0 | 0 | 0 | unobserved |
| sarsa | B1-relative-x4 | 0 | Right | 12 | 6 | 0 | 0 | 0 | unobserved |
| sarsa | B1-relative-x4 | 2000 | Up | 12 | 6 | 2 | -0.36887 | -0.29613 | unobserved |
| sarsa | B1-relative-x4 | 2000 | Down | 12 | 6 | 1 | 0.11933 | -0.255 | unobserved |
| sarsa | B1-relative-x4 | 2000 | Left | 12 | 6 | 8 | -0.44361 | 0.082168 | unobserved |
| sarsa | B1-relative-x4 | 2000 | Right | 12 | 6 | 9 | 0.32701 | 0.32701 | unobserved |
| sarsa | B1-relative-x4 | 6000 | Up | 12 | 6 | 8 | -0.54954 | 0.30351 | unobserved |
| sarsa | B1-relative-x4 | 6000 | Down | 12 | 6 | 10 | -0.40598 | 0.26697 | unobserved |
| sarsa | B1-relative-x4 | 6000 | Left | 12 | 6 | 6 | -0.50618 | -0.093724 | unobserved |
| sarsa | B1-relative-x4 | 6000 | Right | 12 | 6 | 10 | 0.19695 | 0.19695 | unobserved |
| dqn | B0-original | 0 | Up | 12 | 6 | 0 | 0.063491 | -0.29262 | 5.1502 |
| dqn | B0-original | 0 | Down | 12 | 6 | 0 | 0.060121 | -0.072648 | 5.0722 |
| dqn | B0-original | 0 | Left | 12 | 6 | 0 | 0.054502 | -0.16049 | 5.0791 |
| dqn | B0-original | 0 | Right | 12 | 6 | 12 | 0.058547 | 0.058547 | 5.0385 |
| dqn | B0-original | 2000 | Up | 12 | 6 | 6 | -0.1163 | -0.026715 | 4.649 |
| dqn | B0-original | 2000 | Down | 12 | 6 | 8 | -0.087791 | 0.018615 | 4.7102 |
| dqn | B0-original | 2000 | Left | 12 | 6 | 5 | -0.36228 | 0.12897 | 4.546 |
| dqn | B0-original | 2000 | Right | 12 | 6 | 4 | -0.025131 | -0.025131 | 4.7964 |
| dqn | B0-original | 6000 | Up | 12 | 6 | 3 | -0.19741 | -0.12342 | 3.2647 |
| dqn | B0-original | 6000 | Down | 12 | 6 | 8 | -0.24178 | -0.0098649 | 3.9105 |
| dqn | B0-original | 6000 | Left | 12 | 6 | 9 | -0.82689 | 0.52815 | 2.1641 |
| dqn | B0-original | 6000 | Right | 12 | 6 | 9 | 0.054955 | 0.054955 | 4.0221 |
| dqn | B1-relative-x4 | 0 | Up | 12 | 6 | 0 | 0.068183 | -0.29864 | 5.1449 |
| dqn | B1-relative-x4 | 0 | Down | 12 | 6 | 0 | 0.059909 | -0.073853 | 5.052 |
| dqn | B1-relative-x4 | 0 | Left | 12 | 6 | 0 | 0.066235 | -0.15844 | 5.0993 |
| dqn | B1-relative-x4 | 0 | Right | 12 | 6 | 12 | 0.063287 | 0.063287 | 5.0515 |
| dqn | B1-relative-x4 | 2000 | Up | 12 | 6 | 9 | -0.080035 | 0.042488 | 4.7297 |
| dqn | B1-relative-x4 | 2000 | Down | 12 | 6 | 7 | -0.086988 | 0.048264 | 4.7706 |
| dqn | B1-relative-x4 | 2000 | Left | 12 | 6 | 3 | -0.12818 | -0.00048732 | 4.7028 |
| dqn | B1-relative-x4 | 2000 | Right | 12 | 6 | 5 | 0.0085964 | 0.0085964 | 4.747 |
| dqn | B1-relative-x4 | 6000 | Up | 12 | 6 | 7 | -0.55044 | 0.19322 | 3.4183 |
| dqn | B1-relative-x4 | 6000 | Down | 12 | 6 | 11 | -0.70735 | 0.28645 | 3.153 |
| dqn | B1-relative-x4 | 6000 | Left | 12 | 6 | 12 | -0.98232 | 0.56683 | 2.6524 |
| dqn | B1-relative-x4 | 6000 | Right | 12 | 6 | 3 | -0.071032 | -0.071032 | 3.4149 |
| ppo | B0-original | 0 | Up | 12 | 6 | 0 | -0.001408 | -0.0020729 | unobserved |
| ppo | B0-original | 0 | Down | 12 | 6 | 1 | -0.0013754 | -0.0007962 | unobserved |
| ppo | B0-original | 0 | Left | 12 | 6 | 9 | -0.0012934 | 0.00022737 | unobserved |
| ppo | B0-original | 0 | Right | 12 | 6 | 0 | -0.0011514 | -0.0011514 | unobserved |
| ppo | B0-original | 2000 | Up | 12 | 6 | 0 | -0.096204 | -0.076167 | unobserved |
| ppo | B0-original | 2000 | Down | 12 | 6 | 0 | -0.093938 | -0.057084 | unobserved |
| ppo | B0-original | 2000 | Left | 12 | 6 | 0 | -0.098037 | -0.094523 | unobserved |
| ppo | B0-original | 2000 | Right | 12 | 6 | 0 | -0.090723 | -0.090723 | unobserved |
| ppo | B0-original | 6000 | Up | 12 | 6 | 0 | -0.077777 | -0.061659 | unobserved |
| ppo | B0-original | 6000 | Down | 12 | 6 | 0 | -0.070182 | -0.068124 | unobserved |
| ppo | B0-original | 6000 | Left | 12 | 6 | 0 | -0.074058 | -0.10346 | unobserved |
| ppo | B0-original | 6000 | Right | 12 | 6 | 2 | -0.060025 | -0.060025 | unobserved |
| ppo | B1-relative-x4 | 0 | Up | 12 | 6 | 0 | -0.0013158 | -0.00171 | unobserved |
| ppo | B1-relative-x4 | 0 | Down | 12 | 6 | 1 | -0.0012493 | -0.00074947 | unobserved |
| ppo | B1-relative-x4 | 0 | Left | 12 | 6 | 10 | -0.0015642 | 0.00025154 | unobserved |
| ppo | B1-relative-x4 | 0 | Right | 12 | 6 | 0 | -0.0015253 | -0.0015253 | unobserved |
| ppo | B1-relative-x4 | 2000 | Up | 12 | 6 | 2 | -0.053085 | -0.0081758 | unobserved |
| ppo | B1-relative-x4 | 2000 | Down | 12 | 6 | 9 | -0.054214 | 0.013354 | unobserved |
| ppo | B1-relative-x4 | 2000 | Left | 12 | 6 | 0 | -0.055401 | -0.035365 | unobserved |
| ppo | B1-relative-x4 | 2000 | Right | 12 | 6 | 0 | -0.051073 | -0.051073 | unobserved |
| ppo | B1-relative-x4 | 6000 | Up | 12 | 6 | 10 | -0.17405 | 0.088958 | unobserved |
| ppo | B1-relative-x4 | 6000 | Down | 12 | 6 | 4 | -0.12793 | -0.057954 | unobserved |
| ppo | B1-relative-x4 | 6000 | Left | 12 | 6 | 0 | -0.1584 | -0.15275 | unobserved |
| ppo | B1-relative-x4 | 6000 | Right | 12 | 6 | 0 | -0.11349 | -0.11349 | unobserved |


## Linear contribution and checkpoint preference changes


Mean signed contributions below are Right minus the best competing action on all 40 legal probes. Each row uses its own winning competitor. Absolute contributions can cancel and are descriptive algebra, not a learned causal importance estimate.


| Family | Arm | CP | Probes | Bias | Position | Goal | Rays | Occupancy | Age | Visits | Streak | Elapsed | Relative goal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q-learning | B0-original | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| q-learning | B0-original | 2000 | 40 | 0.011226 | -0.039392 | 0.0094451 | -0.0021709 | 0.20361 | 0.019865 | 0.00010698 | 0 | 8.9219e-05 | -0.010134 |
| q-learning | B0-original | 6000 | 40 | 0.0059544 | -0.057794 | 0.020541 | -0.0051744 | -0.31813 | 0.027221 | -0.0036209 | 0 | -5.1719e-05 | -0.01951 |
| q-learning | B1-relative-x4 | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| q-learning | B1-relative-x4 | 2000 | 40 | -0.000775 | -0.03781 | 0.0010773 | -0.0027607 | -0.26953 | -0.0045353 | 0.0017765 | 0 | 1.8787e-05 | -0.16488 |
| q-learning | B1-relative-x4 | 6000 | 40 | -0.0023893 | -0.045945 | -0.01369 | -0.0032717 | -0.39256 | 0.0020641 | -0.00088508 | 0 | 0.00012046 | -0.23716 |
| sarsa | B0-original | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sarsa | B0-original | 2000 | 40 | 0.0077882 | -0.013793 | 0.0088002 | -0.0065652 | -0.36283 | 0.023616 | -0.0011733 | 0 | 0.00013007 | -0.0081411 |
| sarsa | B0-original | 6000 | 40 | -0.0080876 | -0.049883 | 0.059185 | -0.00087757 | -1.5196 | -0.0078543 | -0.0060229 | 0 | -3.0627e-05 | -0.013123 |
| sarsa | B1-relative-x4 | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sarsa | B1-relative-x4 | 2000 | 40 | 0.0067034 | -0.0092436 | 0.018801 | -0.00070514 | 0.10073 | -0.014546 | -0.0047022 | 0 | -8.1391e-05 | -0.15449 |
| sarsa | B1-relative-x4 | 6000 | 40 | 0.0056901 | 0.00067952 | 0.018975 | -0.0025359 | -0.33452 | -0.0072729 | -0.013885 | 0 | -7.5915e-05 | -0.23584 |


Checkpoint preference changes below use the same ordered probes and fixed cohort states (not episodes). A tie at zero Right margin is retained as nonpositive; ordinary argmax changes use the original first-index semantics.


| Family | Arm | Change | States | Argmax changed | Aligned→unaligned | Unaligned→aligned | Right nonpositive→positive | Right positive→nonpositive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q-learning | B0-original | 0→2000 | 88 | 83 | 19 | 24 | 67 | 0 |
| q-learning | B0-original | 2000→6000 | 88 | 56 | 10 | 24 | 1 | 47 |
| q-learning | B1-relative-x4 | 0→2000 | 88 | 64 | 11 | 31 | 20 | 0 |
| q-learning | B1-relative-x4 | 2000→6000 | 88 | 28 | 4 | 17 | 0 | 5 |
| sarsa | B0-original | 0→2000 | 88 | 53 | 11 | 20 | 5 | 0 |
| sarsa | B0-original | 2000→6000 | 88 | 45 | 12 | 15 | 1 | 4 |
| sarsa | B1-relative-x4 | 0→2000 | 88 | 70 | 12 | 30 | 38 | 0 |
| sarsa | B1-relative-x4 | 2000→6000 | 88 | 40 | 8 | 23 | 3 | 21 |
| dqn | B0-original | 0→2000 | 88 | 79 | 16 | 32 | 0 | 79 |
| dqn | B0-original | 2000→6000 | 88 | 30 | 7 | 12 | 12 | 2 |
| dqn | B1-relative-x4 | 0→2000 | 88 | 77 | 13 | 34 | 0 | 77 |
| dqn | B1-relative-x4 | 2000→6000 | 88 | 28 | 6 | 15 | 2 | 6 |
| ppo | B0-original | 0→2000 | 88 | 61 | 18 | 0 | 0 | 4 |
| ppo | B0-original | 2000→6000 | 88 | 14 | 0 | 6 | 5 | 0 |
| ppo | B1-relative-x4 | 0→2000 | 88 | 84 | 16 | 28 | 0 | 4 |
| ppo | B1-relative-x4 | 2000→6000 | 88 | 40 | 11 | 11 | 0 | 0 |


## Delivery validation and known renderer limitation

The complete report above was rendered from unchanged scientific records. The first renderer attempt failed on CSV boolean parsing; the separately recorded adapter corrected literal True/False parsing and rendered these reports. Its final Git postcondition then encountered sandbox ownership restrictions and a mismatch between collapsed and expanded untracked-file status formats. A separate read-only check using the original --untracked-files=all options confirmed all 9,646 original files, main HEAD, index and working-tree status unchanged. The renderer and scientific analysis were not retried. All 432 final full regressions pass, including eight delivery-only checks. All original scientific source, protocol/seal, cohorts, scores and the 3,786-call journal remain byte-identical. The frozen original report CLI retains its parsing/status-check limitations; this completed report is the delivery. See [delivery details](partial_phase5d_delivery_completion.md), [parser correction](partial_phase5d_delivery_note.md) and the separately hashed delivery manifests. No new training, navigation actions or model scoring occurred during delivery.
