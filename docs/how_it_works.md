## MediNav-RL in one page

**Problem.** A hospital delivery robot must carry medicine from the pharmacy to a ward bed. Wards are separated by walls with doorways, cluttered with beds and trolleys, and nurses and patients walk around unpredictably. The robot must arrive quickly, never collide, and keep a comfortable social distance (0.7 m) from people.

**Why not just A\*?** A\* finds the shortest path on a *known, static* map. It achieves 100% success on our maps when nobody is around, but fails in many episodes once people move (it does not see them). Reinforcement learning learns a *reactive* policy from LiDAR, so it can avoid things that are not on any map.

**The robot.** It's a differential-drive robot (like a TurtleBot) with a 360° LiDAR of 24 rays and 3 m range, plus the distance and direction to the goal. Each 0.2 s it chooses either one of 5 moves (discrete) or a linear + angular velocity (continuous).

**Reward.**

- **Progress:** the robot is rewarded for getting closer to the goal along the walkable route (geodesic distance).
- **Goal:** +10 when it arrives.
- **Collision:** −10, and the episode ends.
- **Time:** a small penalty every step.
- **Social:** a penalty for entering a person's 0.7 m comfort zone.
- **Clearance:** a penalty for driving too close to walls.

**Fair testing.** Maps are generated from seeds. Training, validation (model selection and tuning) and test (final results) maps come from separate seed ranges, so final scores are measured on floors the agents have never seen.

**Algorithms (one family per member).**

| Member | Family | Algorithms |
|---|---|---|
| 1 | Tabular | Q-learning, SARSA on a 1,944-state discretisation |
| 2 | Value-based deep RL | DQN, Double DQN, Dueling DQN |
| 3 | On-policy actor-critic | PPO, A2C, Recurrent PPO (LSTM) |
| 4 | Off-policy continuous control | SAC, TD3 |

**Part B refinements.**

- Optuna tuning: TPE compared with random search, with median pruning.
- Five strategies for missing LiDAR returns, plus frame stacking and an LSTM.
- Curriculum learning.
- A reward-shaping ablation.
- A hybrid A\* + RL agent.
- Four ensemble types.
- Nine robustness stress tests, including a non-stationary world where people start walking mid-episode.

**Iteration: environment v1 → v2 (found by the error analysis).** In v1, people walked their routes without noticing the robot. When we broke the person-collisions down by who moved into whom, almost all of them (A\*: 16 of 17; PPO: 13 of 13) were people walking into the robot, often while it was standing still. No policy can avoid that, so it capped every agent near 50–66% and hid the real differences between them. In v2, people **yield**: they wait when their next step would touch the robot, and after 10 blocked steps they walk somewhere else. Re-testing the *unchanged* v1 models in v2 cut TD3's person collisions from 59 to 3. The remaining failures are now the agents' own (walls, getting stuck). The v1 results are kept in `runs/env_v1/`. The old behaviour is still a stress test (`careless_people`).
