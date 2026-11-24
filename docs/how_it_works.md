## MediNav in one page

**Problem.** A delivery robot carries medicine from the **pharmacy** to a numbered **bed in a ward** of a hospital wing. On the way it must get out of the pharmacy, follow the main corridor past parked trolleys and walking staff, find the right ward door and stop at the right bed. It must arrive quickly, never collide, and keep a comfortable distance (0.7 m) from people.

**The hospital.** Every floor is a procedurally generated 24 m × 16 m wing with a main corridor. On both sides of it are rooms with doorways: a Pharmacy (shelves and counter), 2–6 Wards with numbered beds (A-1, A-2, …), a Nurse station, and, on larger floors, a Lobby and a Supply room. Room sizes, corridor width and position, door positions, which room is the pharmacy, trolleys and people all change from floor to floor. A mission reads like *"Pharmacy → Ward C, bed C-3"*.

**Busy-ness levels (difficulty).**

| Level | Name | People | Trolleys | Target bed |
|---|---|---|---|---|
| 0 | Quiet night | 0 | 0 | one of the closest |
| 1 | Normal shift | 2 | 1 | closer half |
| 2 | Busy day (training) | 4 | 3 | any |
| 3 | Visiting hours | 7 | 5 | farther half |

**Why not just A\*?** The robot knows the building's floor plan, so a classical A\* planner can find the shortest walking route. But the floor plan contains no people and no trolleys. A\* that just follows the route (our baseline) bumps into them. Reinforcement learning learns a *reactive* controller from LiDAR that avoids what is not on any plan.

**The robot's senses.** It's a differential-drive robot (like a TurtleBot) with a 360° LiDAR of 24 rays and 3 m range. It also gets the distance and direction to the target bed, plus the direction of the **next waypoint on the planned route**, 1.5 m ahead. Without that route hint, no algorithm learned to leave the pharmacy and find the right ward door within our training budget. Part B measures this as an ablation ("no route guidance"). Every 0.2 s the robot chooses either one of 5 moves (discrete) or a forward speed + turn rate (continuous).

**Reward.**

- **Progress:** a reward for getting closer to the bed along the walking route (geodesic distance, which goes around walls, not through them).
- **Goal:** +10 on arrival.
- **Collision:** −10, and the delivery ends.
- **Time:** a small penalty every step.
- **Social:** a penalty for entering a person's comfort zone.
- **Clearance:** a penalty for scraping walls.

**Fair testing.** Floors are generated from seeds. Training, validation (model selection and tuning) and test (final results) floors come from separate seed ranges, so final scores are measured in buildings the agents have never seen.

**Algorithms (one family per member).**

| Member | Family | Algorithms |
|---|---|---|
| 1 | Tabular | Q-learning, SARSA on a 1,944-state discretisation |
| 2 | Value-based deep RL | DQN, Double DQN, Dueling DQN |
| 3 | On-policy actor-critic | PPO, A2C |
| 4 | Off-policy continuous control | SAC, TD3 |

**Part B refinements.**

- Optuna tuning: random search compared with TPE, with median pruning.
- A "no route guidance" ablation.
- Strategies for 20% missing LiDAR returns.
- Curriculum learning.
- A walking-route vs straight-line reward ablation.
- Longer training.
- Ensembles and a rule-based safety shield.
- Ten robustness stress tests, including a bigger wing, visiting hours, careless people, and people who start walking mid-delivery.

**How the environment evolved (the iteration story, see the archived results).**

1. **v1: 10 m room, people ignore the robot.** Error analysis showed that almost all collisions with people (A\*: 16 of 17; PPO: 13 of 13) were people walking *into* the robot, often while it stood still. No policy can avoid that. → **v2: people yield** (they wait, then re-route). Re-testing the unchanged v1 models cut TD3's person collisions from 59 to 3. Archived in `runs/env_v1/`; the old behaviour is still the `careless_people` stress test.
2. **v2: 10 m room.** Realistic behaviour, but not a hospital: there was no pharmacy, no wards, and trips of only 2–11 m. → **v3: hospital wing** with named rooms and beds, and trips of 7–30 m. Archived in `runs/env_v2_room/`.
3. **v3: hospital wing without the floor plan.** No agent learned (0% after 120k steps). → **The robot gets the route waypoint**, which is how real hospital robots work: a known map plus a learned local controller.
4. **Part B config drift.** The first Part B run forgot `guidance: true`, and every refined model looked broken (≤ 5%). After retraining with the same base world as Part A, the refinements worked: SAC with 2× training reached 95%. A test now guards this.

**Known limitation.** People stop about 5 cm from the robot when they yield, so close passes are common for every agent (median closest approach about 0.07 m, A\* included). A wider yielding distance is future work.
