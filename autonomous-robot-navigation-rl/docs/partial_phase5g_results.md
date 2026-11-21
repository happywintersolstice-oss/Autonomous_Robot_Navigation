# Phase 5G measured longer-horizon learning diagnosis

Engineering verdict: **PHASE5G_MEASUREMENT_VALIDATED**.

Parent `6fb6aac124067b00016f5514ebbc66ef777dea33`; branch `feature/phase5g-longer-horizon-diagnosis-v1`. Fresh B1 DQN then PPO, seed23121, one original 2000-action learning call each. Checkpoints 0/500/1000/2000. Only the existing eight training, four original development and sixteen expanded development cases were used.

All numerical, identity, source and measurement gates must pass independently of navigation readiness. These familiar static open-grid development panels are not held-out generalization evidence. No full training, tuning, extra seed, dynamic-map learning, final/unseen evaluation or Phase6 was authorized or run.

## Fixed checkpoint results

| Model | Training actions | Original success | Expanded success | U / D / L / R expanded successes | First aligned combined | Original / expanded alignment | Right first margin (5 Right cases) | Wait actions / actual actions | Collisions / timeouts |
|---|---:|---:|---:|---|---:|---|---:|---|---|
| DQN | 0 | 1/4 | 5/16 | 0/4 / 2/4 / 0/4 / 3/4 | 6/20 | 1/4 ; 5/16 | 0.0003433 | 0/91 | 14 / 0 |
| DQN | 500 | 1/4 | 5/16 | 2/4 / 2/4 / 1/4 / 0/4 | 5/20 | 1/4 ; 4/16 | -0.0506588 | 1981/2025 | 4 / 10 |
| DQN | 1000 | 1/4 | 7/16 | 2/4 / 3/4 / 1/4 / 1/4 | 5/20 | 1/4 ; 4/16 | -0.0564155 | 2225/2514 | 1 / 11 |
| DQN | 2000 | 1/4 | 10/16 | 2/4 / 3/4 / 1/4 / 4/4 | 7/20 | 1/4 ; 6/16 | -0.0189562 | 1345/2003 | 0 / 9 |
| PPO | 0 | 1/4 | 3/16 | 1/4 / 0/4 / 1/4 / 1/4 | 4/20 | 1/4 ; 3/16 | -0.0004864 | 1359/1427 | 10 / 6 |
| PPO | 500 | 0/4 | 0/16 | 0/4 / 0/4 / 0/4 / 0/4 | 0/20 | 0/4 ; 0/16 | -0.0265042 | 4000/4000 | 0 / 20 |
| PPO | 1000 | 0/4 | 0/16 | 0/4 / 0/4 / 0/4 / 0/4 | 0/20 | 0/4 ; 0/16 | -0.0529724 | 4000/4000 | 0 / 20 |
| PPO | 2000 | 1/4 | 4/16 | 4/4 / 0/4 / 0/4 / 0/4 | 5/20 | 1/4 ; 4/16 | -0.1359639 | 0/79 | 15 / 0 |

DQN margins are Q units. PPO margins are action-probability differences; its separate Right log odds are preserved below and in verification.json. First alignment means a legal strict Manhattan-distance reduction, not eventual success. Wait, invalid first moves and exact maximal-score ties are reported separately. Neither panels nor earlier checkpoints select a new training regime.

## DQN readiness and first-action detail

**BASIC_NAVIGATION_SCREEN_NOT_MET**. Expanded success gain from its own initialized checkpoint: 5/16. Frozen checks: `{"each_direction_at_least_2": false, "expanded_at_least_12": false, "expanded_gain_at_least_4": true}`.

| Actions | Combined U/D/L/R alignment (each /5) | Expanded U/D/L/R alignment (each /4) | First Wait / invalid / ties | Mean goal margin | Right log odds | Moving revisits / ABAB loops |
|---:|---|---|---|---:|---|---|
| 0 | 0/5 / 3/5 / 0/5 / 3/5 | 0/4 / 2/4 / 0/4 / 3/4 | 0 / 1 / 0 | -0.0615430 | not_applicable | 0 / 0 |
| 500 | 1/5 / 3/5 / 1/5 / 0/5 | 1/4 / 2/4 / 1/4 / 0/4 | 8 / 1 / 0 | -0.0279726 | not_applicable | 6 / 0 |
| 1000 | 1/5 / 4/5 / 0/5 / 0/5 | 1/4 / 3/4 / 0/4 / 0/4 | 8 / 0 / 0 | -0.0371075 | not_applicable | 226 / 154 |
| 2000 | 1/5 / 4/5 / 0/5 / 2/5 | 1/4 / 3/4 / 0/4 / 2/4 | 5 / 0 / 0 | -0.0122764 | not_applicable | 586 / 481 |

## DQN all case-level measured outcomes

Full trajectories, initial public observation and first-action score vectors are preserved in the linked development files. Case rows below are recomputed from those actual paths; moving revisits exclude stationary Wait/collision steps. Shortest-route inefficiency is steps/Manhattan distance for successful cases only.

| Training actions | Panel | Case ID | Direction | Outcome | Steps | First action | Aligned | Wait | Revisit | ABAB | Successful step inefficiency |
|---:|---|---|---|---|---:|---:|---|---:|---:|---:|---|
| 0 | original | phase5a-handwritten-fixtures-v1-A-evaluation-00 | Up | collision | 7 | 1 | False | 0 | 0 | 0 | not_applicable |
| 0 | original | phase5a-handwritten-fixtures-v1-A-evaluation-01 | Down | success | 1 | 1 | True | 0 | 0 | 0 | 1.0 |
| 0 | original | phase5a-handwritten-fixtures-v1-A-evaluation-02 | Left | collision | 7 | 3 | False | 0 | 0 | 0 | not_applicable |
| 0 | original | phase5a-handwritten-fixtures-v1-A-evaluation-03 | Right | collision | 7 | 1 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-00 | Up | collision | 7 | 3 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-01 | Up | collision | 4 | 1 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-02 | Up | collision | 3 | 1 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-03 | Up | collision | 1 | 1 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-04 | Down | success | 1 | 1 | True | 0 | 0 | 0 | 1.0 |
| 0 | expanded | phase5b-open-development-pairs-v1-05 | Down | collision | 10 | 3 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-06 | Down | success | 2 | 1 | True | 0 | 0 | 0 | 1.0 |
| 0 | expanded | phase5b-open-development-pairs-v1-07 | Down | collision | 8 | 3 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-08 | Left | collision | 8 | 3 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-09 | Left | collision | 6 | 3 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-10 | Left | collision | 6 | 3 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-11 | Left | collision | 3 | 3 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-12 | Right | collision | 5 | 1 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-13 | Right | success | 1 | 3 | True | 0 | 0 | 0 | 1.0 |
| 0 | expanded | phase5b-open-development-pairs-v1-14 | Right | success | 2 | 3 | True | 0 | 0 | 0 | 1.0 |
| 0 | expanded | phase5b-open-development-pairs-v1-15 | Right | success | 2 | 3 | True | 0 | 0 | 0 | 1.0 |
| 500 | original | phase5a-handwritten-fixtures-v1-A-evaluation-00 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | original | phase5a-handwritten-fixtures-v1-A-evaluation-01 | Down | success | 1 | 1 | True | 0 | 0 | 0 | 1.0 |
| 500 | original | phase5a-handwritten-fixtures-v1-A-evaluation-02 | Left | collision | 4 | 1 | False | 0 | 0 | 0 | not_applicable |
| 500 | original | phase5a-handwritten-fixtures-v1-A-evaluation-03 | Right | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-00 | Up | success | 3 | 1 | False | 0 | 1 | 0 | 3.0 |
| 500 | expanded | phase5b-open-development-pairs-v1-01 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-02 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-03 | Up | success | 2 | 0 | True | 0 | 0 | 0 | 1.0 |
| 500 | expanded | phase5b-open-development-pairs-v1-04 | Down | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-05 | Down | success | 1 | 1 | True | 0 | 0 | 0 | 1.0 |
| 500 | expanded | phase5b-open-development-pairs-v1-06 | Down | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-07 | Down | success | 2 | 1 | True | 0 | 0 | 0 | 1.0 |
| 500 | expanded | phase5b-open-development-pairs-v1-08 | Left | collision | 3 | 1 | False | 0 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-09 | Left | success | 1 | 2 | True | 0 | 0 | 0 | 1.0 |
| 500 | expanded | phase5b-open-development-pairs-v1-10 | Left | collision | 7 | 1 | False | 0 | 3 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-11 | Left | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-12 | Right | timeout | 200 | 2 | False | 184 | 1 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-13 | Right | timeout | 200 | 1 | False | 197 | 1 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-14 | Right | collision | 1 | 0 | False | 0 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-15 | Right | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-00 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-01 | Down | success | 1 | 1 | True | 0 | 0 | 0 | 1.0 |
| 1000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-02 | Left | timeout | 200 | 1 | False | 198 | 0 | 0 | not_applicable |
| 1000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-03 | Right | collision | 39 | 1 | False | 0 | 27 | 19 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-00 | Up | success | 159 | 1 | False | 0 | 156 | 117 | 159.0 |
| 1000 | expanded | phase5b-open-development-pairs-v1-01 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-02 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-03 | Up | success | 2 | 0 | True | 0 | 0 | 0 | 1.0 |
| 1000 | expanded | phase5b-open-development-pairs-v1-04 | Down | success | 1 | 1 | True | 0 | 0 | 0 | 1.0 |
| 1000 | expanded | phase5b-open-development-pairs-v1-05 | Down | success | 1 | 1 | True | 0 | 0 | 0 | 1.0 |
| 1000 | expanded | phase5b-open-development-pairs-v1-06 | Down | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-07 | Down | success | 2 | 1 | True | 0 | 0 | 0 | 1.0 |
| 1000 | expanded | phase5b-open-development-pairs-v1-08 | Left | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-09 | Left | success | 66 | 4 | False | 65 | 0 | 0 | 66.0 |
| 1000 | expanded | phase5b-open-development-pairs-v1-10 | Left | timeout | 200 | 1 | False | 197 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-11 | Left | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-12 | Right | success | 43 | 1 | False | 0 | 29 | 6 | 43.0 |
| 1000 | expanded | phase5b-open-development-pairs-v1-13 | Right | timeout | 200 | 1 | False | 172 | 14 | 12 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-14 | Right | timeout | 200 | 1 | False | 193 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-15 | Right | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 2000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-00 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 2000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-01 | Down | success | 1 | 1 | True | 0 | 0 | 0 | 1.0 |
| 2000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-02 | Left | timeout | 200 | 0 | False | 0 | 192 | 124 | not_applicable |
| 2000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-03 | Right | timeout | 200 | 1 | False | 0 | 192 | 181 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-00 | Up | success | 179 | 1 | False | 0 | 176 | 173 | 179.0 |
| 2000 | expanded | phase5b-open-development-pairs-v1-01 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-02 | Up | timeout | 200 | 4 | False | 199 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-03 | Up | success | 2 | 0 | True | 0 | 0 | 0 | 1.0 |
| 2000 | expanded | phase5b-open-development-pairs-v1-04 | Down | success | 1 | 1 | True | 0 | 0 | 0 | 1.0 |
| 2000 | expanded | phase5b-open-development-pairs-v1-05 | Down | success | 1 | 1 | True | 0 | 0 | 0 | 1.0 |
| 2000 | expanded | phase5b-open-development-pairs-v1-06 | Down | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-07 | Down | success | 2 | 1 | True | 0 | 0 | 0 | 1.0 |
| 2000 | expanded | phase5b-open-development-pairs-v1-08 | Left | timeout | 200 | 0 | False | 196 | 2 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-09 | Left | success | 3 | 0 | False | 0 | 0 | 0 | 3.0 |
| 2000 | expanded | phase5b-open-development-pairs-v1-10 | Left | timeout | 200 | 1 | False | 150 | 23 | 3 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-11 | Left | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-12 | Right | success | 5 | 1 | False | 0 | 1 | 0 | 5.0 |
| 2000 | expanded | phase5b-open-development-pairs-v1-13 | Right | success | 5 | 1 | False | 0 | 0 | 0 | 5.0 |
| 2000 | expanded | phase5b-open-development-pairs-v1-14 | Right | success | 2 | 3 | True | 0 | 0 | 0 | 1.0 |
| 2000 | expanded | phase5b-open-development-pairs-v1-15 | Right | success | 2 | 3 | True | 0 | 0 | 0 | 1.0 |

[DQN development records at 0](../artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1/dqn/development/0000.json); checkpoint SHA256 `8f8ede78bc18aa4671a72a9218cdab0c416060d13b4718a0a13506552c1c2570`.

[DQN development records at 500](../artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1/dqn/development/0500.json); checkpoint SHA256 `6c8dd254ef879c99cccac8635232947be8b3dca1791d4fe2bff00d6e2ca0c5a6`.

[DQN development records at 1000](../artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1/dqn/development/1000.json); checkpoint SHA256 `3f1272e818e55bf5de4b46c6493b83b6a07ffbd522249aca38e49dcce973ac90`.

[DQN development records at 2000](../artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1/dqn/development/2000.json); checkpoint SHA256 `a76fe9e3c3e1a518b9f1383f20e2f4447bc87d5bea2254fb33eaa8a2f4e2ee99`.

## PPO readiness and first-action detail

**BASIC_NAVIGATION_SCREEN_NOT_MET**. Expanded success gain from its own initialized checkpoint: 1/16. Frozen checks: `{"each_direction_at_least_2": false, "expanded_at_least_12": false, "expanded_gain_at_least_4": false}`.

| Actions | Combined U/D/L/R alignment (each /5) | Expanded U/D/L/R alignment (each /4) | First Wait / invalid / ties | Mean goal margin | Right log odds | Moving revisits / ABAB loops |
|---:|---|---|---|---:|---|---|
| 0 | 1/5 / 0/5 / 2/5 / 1/5 | 1/4 / 0/4 / 1/4 / 1/4 | 6 / 2 / 0 | -0.0004870 | -0.0024303196826417103 | 1 / 0 |
| 500 | 0/5 / 0/5 / 0/5 / 0/5 | 0/4 / 0/4 / 0/4 / 0/4 | 20 / 0 / 0 | -0.0198204 | -0.1305403318859451 | 0 / 0 |
| 1000 | 0/5 / 0/5 / 0/5 / 0/5 | 0/4 / 0/4 / 0/4 / 0/4 | 20 / 0 / 0 | -0.0393942 | -0.2584980206976428 | 0 / 0 |
| 2000 | 5/5 / 0/5 / 0/5 / 0/5 | 4/4 / 0/4 / 0/4 / 0/4 | 0 / 4 / 0 | -0.0686199 | -0.5609255786845393 | 0 / 0 |

## PPO all case-level measured outcomes

Full trajectories, initial public observation and first-action score vectors are preserved in the linked development files. Case rows below are recomputed from those actual paths; moving revisits exclude stationary Wait/collision steps. Shortest-route inefficiency is steps/Manhattan distance for successful cases only.

| Training actions | Panel | Case ID | Direction | Outcome | Steps | First action | Aligned | Wait | Revisit | ABAB | Successful step inefficiency |
|---:|---|---|---|---|---:|---:|---|---:|---:|---:|---|
| 0 | original | phase5a-handwritten-fixtures-v1-A-evaluation-00 | Up | collision | 7 | 2 | False | 0 | 0 | 0 | not_applicable |
| 0 | original | phase5a-handwritten-fixtures-v1-A-evaluation-01 | Down | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 0 | original | phase5a-handwritten-fixtures-v1-A-evaluation-02 | Left | success | 1 | 2 | True | 0 | 0 | 0 | 1.0 |
| 0 | original | phase5a-handwritten-fixtures-v1-A-evaluation-03 | Right | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-00 | Up | collision | 10 | 2 | False | 0 | 1 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-01 | Up | success | 1 | 0 | True | 0 | 0 | 0 | 1.0 |
| 0 | expanded | phase5b-open-development-pairs-v1-02 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-03 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-04 | Down | collision | 170 | 4 | False | 163 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-05 | Down | collision | 1 | 2 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-06 | Down | collision | 8 | 2 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-07 | Down | collision | 8 | 3 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-08 | Left | success | 1 | 2 | True | 0 | 0 | 0 | 1.0 |
| 0 | expanded | phase5b-open-development-pairs-v1-09 | Left | collision | 6 | 0 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-10 | Left | collision | 6 | 3 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-11 | Left | collision | 1 | 0 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-12 | Right | timeout | 200 | 2 | False | 196 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-13 | Right | success | 1 | 3 | True | 0 | 0 | 0 | 1.0 |
| 0 | expanded | phase5b-open-development-pairs-v1-14 | Right | collision | 6 | 2 | False | 0 | 0 | 0 | not_applicable |
| 0 | expanded | phase5b-open-development-pairs-v1-15 | Right | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | original | phase5a-handwritten-fixtures-v1-A-evaluation-00 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | original | phase5a-handwritten-fixtures-v1-A-evaluation-01 | Down | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | original | phase5a-handwritten-fixtures-v1-A-evaluation-02 | Left | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | original | phase5a-handwritten-fixtures-v1-A-evaluation-03 | Right | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-00 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-01 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-02 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-03 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-04 | Down | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-05 | Down | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-06 | Down | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-07 | Down | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-08 | Left | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-09 | Left | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-10 | Left | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-11 | Left | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-12 | Right | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-13 | Right | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-14 | Right | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 500 | expanded | phase5b-open-development-pairs-v1-15 | Right | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-00 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-01 | Down | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-02 | Left | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-03 | Right | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-00 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-01 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-02 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-03 | Up | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-04 | Down | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-05 | Down | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-06 | Down | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-07 | Down | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-08 | Left | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-09 | Left | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-10 | Left | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-11 | Left | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-12 | Right | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-13 | Right | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-14 | Right | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 1000 | expanded | phase5b-open-development-pairs-v1-15 | Right | timeout | 200 | 4 | False | 200 | 0 | 0 | not_applicable |
| 2000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-00 | Up | success | 1 | 0 | True | 0 | 0 | 0 | 1.0 |
| 2000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-01 | Down | collision | 7 | 0 | False | 0 | 0 | 0 | not_applicable |
| 2000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-02 | Left | collision | 7 | 0 | False | 0 | 0 | 0 | not_applicable |
| 2000 | original | phase5a-handwritten-fixtures-v1-A-evaluation-03 | Right | collision | 4 | 0 | False | 0 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-00 | Up | success | 1 | 0 | True | 0 | 0 | 0 | 1.0 |
| 2000 | expanded | phase5b-open-development-pairs-v1-01 | Up | success | 1 | 0 | True | 0 | 0 | 0 | 1.0 |
| 2000 | expanded | phase5b-open-development-pairs-v1-02 | Up | success | 2 | 0 | True | 0 | 0 | 0 | 1.0 |
| 2000 | expanded | phase5b-open-development-pairs-v1-03 | Up | success | 2 | 0 | True | 0 | 0 | 0 | 1.0 |
| 2000 | expanded | phase5b-open-development-pairs-v1-04 | Down | collision | 4 | 0 | False | 0 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-05 | Down | collision | 7 | 0 | False | 0 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-06 | Down | collision | 7 | 0 | False | 0 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-07 | Down | collision | 1 | 0 | False | 0 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-08 | Left | collision | 8 | 0 | False | 0 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-09 | Left | collision | 6 | 0 | False | 0 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-10 | Left | collision | 1 | 0 | False | 0 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-11 | Left | collision | 1 | 0 | False | 0 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-12 | Right | collision | 6 | 0 | False | 0 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-13 | Right | collision | 3 | 0 | False | 0 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-14 | Right | collision | 1 | 0 | False | 0 | 0 | 0 | not_applicable |
| 2000 | expanded | phase5b-open-development-pairs-v1-15 | Right | collision | 9 | 0 | False | 0 | 0 | 0 | not_applicable |

[PPO development records at 0](../artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1/ppo/development/0000.json); checkpoint SHA256 `90400f4f17fab66aa3918ec62c644cc2b2abd776236fa584d9d2653c4f2fe0e4`.

[PPO development records at 500](../artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1/ppo/development/0500.json); checkpoint SHA256 `b039c1018eb0c3690445aab8a5ffccd824ce35ddec40d6fb230a00a6f17aeb81`.

[PPO development records at 1000](../artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1/ppo/development/1000.json); checkpoint SHA256 `71139022c12058defaaa7e2efcc77744d6de143d5f39bb4b4d01029cc345c30e`.

[PPO development records at 2000](../artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1/ppo/development/2000.json); checkpoint SHA256 `0e4c8396a5531f43ef3fd5d21ec66d6bae2342f88d5bc613826373f8f531b014`.

## Actual optimizer reward use

| Algorithm | Training actions | Actual optimizer steps | Goal experiences | Distinct goals sampled | Goal sample draws | Actual updates with goals | All sample draws / distinct experiences |
|---|---:|---:|---:|---:|---:|---:|---|
| DQN | 2000 | 475 | 27 | 27 | 379 | 257 | 23750 / 1867 |
| PPO | 2000 | 200 | 20 | 20 | 100 | 64 | 10000 / 2000 |

## DQN captured arithmetic and directional exposure

Independent actual-tensor arithmetic: `{"DQN": 475, "PPO": 0, "GAE": 0, "linear": 0}`. Required missing fields: `[]`. Completed training episodes 57; trailing incomplete prefix 31 actions; raw outcomes `{"collision": 27, "running": 1943, "success": 27, "timeout": 3}`.

```json
{
  "actual_target": {
    "n": 23750,
    "min": -5.010000228881836,
    "max": 9.989999771118164,
    "mean": 0.2740281443083757,
    "std": 1.4237362142153005
  },
  "residual_snapshot_reconstructed": {
    "n": 23750,
    "min": -5.351212978363037,
    "max": 10.067707061767578,
    "mean": 0.06987644111145484,
    "std": 1.3938548554734331
  },
  "target_syncs": 8,
  "profile_counts": {
    "selected_actions": 2000,
    "replay_sample_calls": 475,
    "replay_consumptions": 475,
    "target_forwards": 475,
    "online_forwards": 1573,
    "native_optimizer_steps": 475
  }
}
```

| Cumulative training actions | Direction | Goal experiences | Distinct sampled goals | Goal sample draws | Actual updates using goals |
|---:|---|---:|---:|---:|---:|
| 0 | Up | 0 | 0 | 0 | 0 |
| 0 | Down | 0 | 0 | 0 | 0 |
| 0 | Left | 0 | 0 | 0 | 0 |
| 0 | Right | 0 | 0 | 0 | 0 |
| 500 | Up | 0 | 0 | 0 | 0 |
| 500 | Down | 4 | 4 | 39 | 28 |
| 500 | Left | 2 | 2 | 6 | 6 |
| 500 | Right | 3 | 3 | 30 | 26 |
| 1000 | Up | 1 | 1 | 3 | 3 |
| 1000 | Down | 8 | 8 | 90 | 66 |
| 1000 | Left | 3 | 3 | 19 | 17 |
| 1000 | Right | 5 | 5 | 71 | 61 |
| 2000 | Up | 1 | 1 | 12 | 11 |
| 2000 | Down | 11 | 11 | 176 | 138 |
| 2000 | Left | 5 | 5 | 63 | 55 |
| 2000 | Right | 10 | 10 | 128 | 111 |

Actual use within each update-time interval (experiences may have occurred in an earlier interval):

| Interval | New goal experiences | Distinct goals consumed | Goal draws | Updates using goals |
|---|---:|---:|---:|---:|
| [0, 500] | 9 | 9 | 75 | 51 |
| [500, 1000] | 8 | 17 | 108 | 74 |
| [1000, 2000] | 10 | 27 | 196 | 132 |

Earliest directional goal-to-actual-update examples, including explicit absent events:

```json
{
  "Up": {
    "transition_id": 929,
    "case_id": "phase5a-handwritten-fixtures-v1-A-train-00",
    "direction": "Up",
    "experienced_at": 929,
    "used_at": 960,
    "update_id": 215,
    "sample_row": 42,
    "trace_sequence": 2606,
    "raw_goal_reward": 9.99,
    "actual_target": 9.989999771118164,
    "chosen_Q": 0.19658663868904114,
    "residual_snapshot_reconstructed": 9.793413162231445
  },
  "Down": {
    "transition_id": 152,
    "case_id": "phase5a-handwritten-fixtures-v1-A-train-05",
    "direction": "Down",
    "experienced_at": 152,
    "used_at": 168,
    "update_id": 17,
    "sample_row": 19,
    "trace_sequence": 395,
    "raw_goal_reward": 9.99,
    "actual_target": 9.989999771118164,
    "chosen_Q": -0.060632288455963135,
    "residual_snapshot_reconstructed": 10.05063247680664
  },
  "Left": {
    "transition_id": 291,
    "case_id": "phase5a-handwritten-fixtures-v1-A-train-06",
    "direction": "Left",
    "experienced_at": 291,
    "used_at": 404,
    "update_id": 76,
    "sample_row": 38,
    "trace_sequence": 1055,
    "raw_goal_reward": 9.99,
    "actual_target": 9.989999771118164,
    "chosen_Q": 0.1287444829940796,
    "residual_snapshot_reconstructed": 9.861255645751953
  },
  "Right": {
    "transition_id": 24,
    "case_id": "phase5a-handwritten-fixtures-v1-A-train-07",
    "direction": "Right",
    "experienced_at": 24,
    "used_at": 104,
    "update_id": 1,
    "sample_row": 38,
    "trace_sequence": 215,
    "raw_goal_reward": 9.99,
    "actual_target": 9.989999771118164,
    "chosen_Q": 0.05085461586713791,
    "residual_snapshot_reconstructed": 9.9391450881958
  }
}
```

[Actual compressed DQN learning traces](../artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1/dqn/trace/). Sequence/update/transition IDs in the examples resolve to those immutable shards. Every checkpoint's network/optimizer hashes are joined to its completed update boundary in verification.json; no saved model is scored by verification.

## PPO captured arithmetic and directional exposure

Independent actual-tensor arithmetic: `{"DQN": 0, "PPO": 200, "GAE": 20, "linear": 0}`. Required missing fields: `[]`. Completed training episodes 61; trailing incomplete prefix 100 actions; raw outcomes `{"collision": 41, "running": 1939, "success": 20}`.

```json
{
  "advantages": {
    "n": 2000,
    "min": -6.727189064025879,
    "max": 13.684968948364258,
    "mean": 0.06250224859220907,
    "std": 3.10336663243064
  },
  "returns": {
    "n": 2000,
    "min": -5.010000228881836,
    "max": 9.989999771118164,
    "mean": -1.5863660853151231,
    "std": 3.1384512352220435
  },
  "timeout_bootstraps": 0,
  "rollouts": 20,
  "epochs": 100,
  "Q_values": "not_applicable",
  "profile_counts": {
    "actor_forwards": 2000,
    "critic_forwards": 20,
    "rollout_consumptions": 200,
    "rollout_permutations": 100,
    "policy_training_evaluations": 200,
    "native_optimizer_steps": 200
  }
}
```

| Cumulative training actions | Direction | Goal experiences | Distinct sampled goals | Goal sample draws | Actual updates using goals |
|---:|---|---:|---:|---:|---:|
| 0 | Up | 0 | 0 | 0 | 0 |
| 0 | Down | 0 | 0 | 0 | 0 |
| 0 | Left | 0 | 0 | 0 | 0 |
| 0 | Right | 0 | 0 | 0 | 0 |
| 500 | Up | 1 | 1 | 5 | 5 |
| 500 | Down | 3 | 3 | 15 | 9 |
| 500 | Left | 1 | 1 | 5 | 5 |
| 500 | Right | 1 | 1 | 5 | 5 |
| 1000 | Up | 1 | 1 | 5 | 5 |
| 1000 | Down | 6 | 6 | 30 | 24 |
| 1000 | Left | 1 | 1 | 5 | 5 |
| 1000 | Right | 1 | 1 | 5 | 5 |
| 2000 | Up | 4 | 4 | 20 | 16 |
| 2000 | Down | 9 | 9 | 45 | 36 |
| 2000 | Left | 2 | 2 | 10 | 10 |
| 2000 | Right | 5 | 5 | 25 | 25 |

Actual use within each update-time interval (experiences may have occurred in an earlier interval):

| Interval | New goal experiences | Distinct goals consumed | Goal draws | Updates using goals |
|---|---:|---:|---:|---:|
| [0, 500] | 6 | 6 | 30 | 15 |
| [500, 1000] | 3 | 3 | 15 | 15 |
| [1000, 2000] | 11 | 11 | 55 | 34 |

Earliest directional goal-to-actual-update examples, including explicit absent events:

```json
{
  "Up": {
    "transition_id": 191,
    "case_id": "phase5a-handwritten-fixtures-v1-A-train-04",
    "direction": "Up",
    "experienced_at": 191,
    "used_at": 200,
    "update_id": 12,
    "sample_row": 37,
    "trace_sequence": 655,
    "raw_goal_reward": 9.99,
    "rollout_id": 2,
    "row": 90,
    "epoch": 0,
    "minibatch_id": 2,
    "raw_advantage": 9.284309387207031,
    "normalized_advantage": 3.8049895763397217,
    "return_value": 9.989999771118164,
    "action_probability_at_update": 0.19724825024604797
  },
  "Down": {
    "transition_id": 71,
    "case_id": "phase5a-handwritten-fixtures-v1-A-train-05",
    "direction": "Down",
    "experienced_at": 71,
    "used_at": 100,
    "update_id": 1,
    "sample_row": 20,
    "trace_sequence": 313,
    "raw_goal_reward": 9.99,
    "rollout_id": 1,
    "row": 70,
    "epoch": 0,
    "minibatch_id": 1,
    "raw_advantage": 9.721185684204102,
    "normalized_advantage": 1.578626275062561,
    "return_value": 9.989999771118164,
    "action_probability_at_update": 0.2002723515033722
  },
  "Left": {
    "transition_id": 8,
    "case_id": "phase5a-handwritten-fixtures-v1-A-train-06",
    "direction": "Left",
    "experienced_at": 8,
    "used_at": 100,
    "update_id": 1,
    "sample_row": 10,
    "trace_sequence": 313,
    "raw_goal_reward": 9.99,
    "rollout_id": 1,
    "row": 7,
    "epoch": 0,
    "minibatch_id": 1,
    "raw_advantage": 9.91561222076416,
    "normalized_advantage": 1.618405818939209,
    "return_value": 9.989999771118164,
    "action_probability_at_update": 0.19981642067432404
  },
  "Right": {
    "transition_id": 74,
    "case_id": "phase5a-handwritten-fixtures-v1-A-train-03",
    "direction": "Right",
    "experienced_at": 74,
    "used_at": 100,
    "update_id": 2,
    "sample_row": 40,
    "trace_sequence": 316,
    "raw_goal_reward": 9.99,
    "rollout_id": 1,
    "row": 73,
    "epoch": 0,
    "minibatch_id": 2,
    "raw_advantage": 9.711954116821289,
    "normalized_advantage": 1.674462914466858,
    "return_value": 9.989999771118164,
    "action_probability_at_update": 0.19910122454166412
  }
}
```

[Actual compressed PPO learning traces](../artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1/ppo/trace/). Sequence/update/transition IDs in the examples resolve to those immutable shards. Every checkpoint's network/optimizer hashes are joined to its completed update boundary in verification.json; no saved model is scored by verification.

## Resource, isolation and validation evidence

Measured run accounting: `{"actual_optimizer_steps": 675, "assessment_actions": 16139, "assessment_cases": 160, "assessment_wall_seconds": 1151.8514767000015, "final_unseen_accesses": 0, "new_synthetic_optimizer_steps": 0, "new_synthetic_transitions": 0, "status": "completed", "training_actions": 4000, "training_wall_seconds": 755.2297159000009}`. New synthetic physical transitions and optimizer operations: **0 / 0**, preregistered caps0/0. Legacy regression mechanics are separate: `{"child_fixture_mechanics": {"optimizer_steps": 108, "transitions": 480, "wall_seconds": 0}, "classification": "unchanged_parent_regression", "combined_environment_transitions": 968352, "combined_neural_optimizer_steps": 2016, "exit_code": 0, "new_Phase5G_synthetic_actions": 0, "parent_process": {"classification": "legacy_regression_mechanics", "exit_code": 0, "operations": {"completed_environment_transitions": 967872, "completed_linear_updates": 175, "completed_neural_optimizer_steps": 1908, "failed_environment_step_calls": 11}, "research_evaluation_episodes": 0, "scientific_actions": 0, "scope": "Complete unchanged parent test list, separately accounted by explicit user authorization", "wall_seconds": 512.9573343999946}, "partial_linear_updates": 175, "scientific_actions": 0, "wall_seconds": 515.9676962999947}`.

Legacy tests `{"tests": 479, "failures": 0, "errors": 0, "skipped": 0}`; targeted post tests `{"tests": 3, "failures": 0, "errors": 0, "skipped": 0}`. All pre/post attempts and genuine errors are preserved. A local Git ownership setup failure was corrected with process-local safe-directory configuration; no shared/global Git setting or scientific source/evidence was changed. The first plain F verifier also identified only the newly supplied G instruction file in main's untracked set. Its scoped pass excludes only that file for the old status comparison; G's own inventory includes it. Full receipts are in the evidence namespace.

Padded preregistered aggregate storage estimate 403440329 bytes, cap536870912. Namespace bytes before final report/inventory write: 120015092. Training wall time includes logging and checkpoint saves, capped1800 per arm/3600 combined; actual per-arm seconds: `{"dqn": 275.8008387000009, "ppo": 479.1525552000094}`. Assessment and full legacy times are separate. Extra instrumentation overhead and live bitwise on/off equivalence are **not_observed**, because an uninstrumented matched research run is unauthorized; E's synthetic equivalence remains inherited evidence only.

All 160 fixed case episodes used detached checkpoint copies. Native step counts equal assessment ledger debits, native optimizer calls are zero, each child checkpoint model/optimizer and child global RNG stayed unchanged during assessment, and parent global RNG/training ledger/receipt/checkpoint bytes were unchanged across children. The full before/after isolation receipt is assessment-parent-isolation.json. Checkpoint saves also preserve training network/optimizer/buffer/RNG fingerprints exactly.

DQN actual TD targets, consumed masks and Huber losses are measured optimizer tensors; residual summaries are snapshot_reconstructed algebra on those same pre-update tensors. PPO reports actual collected probabilities/log probabilities/V, GAE, returns and batch advantages/losses; Q-values are not_applicable. Goal sample draws include repeated draws/epochs; neither draws nor training goals are development successes. Timeout bootstraps are reported only when actually observed, and never counted as goal rewards.

Zero final/unseen research accesses. Original dirty main, historical branch refs and all 10,901 parent files are byte-checked. Output hashes, frozen source/library/case pins, raw ledgers, checkpoint copies and JUnit/log receipts are retained. Read-only reproduction: PYTHONPATH=src python -m robot_navigation.partial_phase5g verify. register/run are one-shot; do not remove locks to rerun.

F and G are separate fresh studies: G uses a different DQN schedule horizon. The four G checkpoint rows describe one G trajectory only. No association establishes causality, fixes credit assignment or ranks algorithms. Stop after Phase5G.

## Delivery-only appendix

The [complete preflight correction history](../artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1/setup-history.json) records the Git ownership checks and the CRLF/LF inventory assertion as well as the scoped historical-status exception. These occurred before registration and research. The [hashed delivery appendix](../artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1/delivery-appendix.json) records this report-only addition; no frozen scientific code, tests, case data, checkpoints, traces or assessment records changed.

Verification includes the preserved original dirty-workspace inventory and pinned runtime paths, so run it from this preserved worktree. Root instruction-brief hashes are byte-sensitive; automatic line-ending conversion fails closed. In PowerShell from the application folder with the pinned Python environment active: `$env:PYTHONPATH="src"; python -m robot_navigation.partial_phase5g verify`. Do not invoke the one-shot study again.
