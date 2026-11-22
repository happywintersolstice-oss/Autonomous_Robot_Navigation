# Warehouse navigation: user and code guide

**Project title:** Autonomous Warehouse Robot Navigation: A Comparative Study of Q-learning, SARSA, DQN and PPO.

## What the project does

A simulated delivery robot moves between named warehouse stations while avoiding racks, a packing bench, pallets and outer walls. The question is how four learning algorithms compare under shared training conditions. There are 12 independently trained models: four algorithms, each with seeds 19411, 19421 and 19431. Every model receives 600,000 training actions.

The 13 by 17 floor contains 130 connected traversable cells. Three racks have parallel access aisles, an upper circulation aisle and a main cross-aisle. Receiving, packing, dispatch and charging are labelled. Seven preset jobs cover put-away, picking, shipping and returning to the dock. A pickup beside a rack is a navigable aisle cell.

## Start and navigate the app

Double-click **Start Warehouse Navigation.cmd**, or run `python run_app.py` from the repository root. Open **http://127.0.0.1:8517** and leave the terminal open. The dark control-room interface has three destinations: **Simulator**, **Results** and **Guide**.

1. In **Simulator > Watch the robot**, choose an **Algorithm**, **Training seed** and **Delivery job**. Press **Play**. R marks the robot, S its start and G its destination; the trail shows completed moves. Dark cells block movement.
2. Use **Pause**, **Next step**, **Reset** or **Replay position** to inspect decisions. A collision or 200-move timeout is a genuine recorded failure. Playback does not repair routes or update models.
3. Open **Compare robots** to view Q-learning, SARSA, DQN and PPO on the same job and seed. They advance on one clock; finished robots hold their terminal position. A single route is an example, rather than the aggregate score.
4. Expand **Browse every assessed route** and enable **Choose from all 188 assessment routes** to inspect all recorded cases, including failures. Changing the route or seed resets playback.
5. Open **Learning history** and select a training recording. These are real exploratory actions sampled at registered stages, rather than a final policy pretending to learn. Each run saves two episodes near the start, 200,000 moves and 400,000 moves. The learning-curve expander shows monitored performance.
6. Open **Choose a route**, set **Pickup point** and **Delivery point**, then click **Create delivery route**. Use the selected trained algorithm or the explicitly labelled BFS shortest-route planner. Changing a station takes effect when you create the route. A new algorithm or seed requires a new route.
7. Open **Results** for completion, collision, timeout and route-efficiency comparisons across all three seeds. Expand the learning curves or per-seed table and download the CSV. **Guide** explains the same controls inside the app.

**Temporary 3D map preview** projects the same map, fixtures and visible route into an interactive scene. Drag to rotate and scroll to zoom. It requires WebGL; switching it off restores 2D. It adds no vertical movement or extra robot capability.

**Historical maps and earlier studies** contains the original abstract grid and four structured maps. The expandable historical results section retains their original studies. Their policies and results are separate from warehouse models.

## Why these four algorithms?

| Algorithm | What it learns | Reason for inclusion |
| --- | --- | --- |
| Q-learning | A table of move values; updates toward the best estimated next action | Simple, inspectable off-policy value-learning reference for a finite grid |
| SARSA | A move-value table; updates using the next action actually chosen during exploration | Comparison with an on-policy tabular learning rule |
| DQN | A neural approximation to move values, trained with replay memory and a target network | Tests neural value learning against explicit tables |
| PPO | A policy over moves and a value estimate, using clipped policy updates | Tests a neural policy-learning approach |

No method was chosen as the warehouse winner beforehand. Q-learning appears first by display order. Results report measured performance for every declared seed, including weak runs. BFS knows the floor and provides a shortest-route reference for custom routes; it is not a fifth learned algorithm.

## What the robot sees and how rewards work

Every learned method receives robot row/column and goal row/column. Tabular methods index these coordinates; neural methods use fixed coordinate scaling. They have no obstacle sensor channel, action masking or planner fallback. This studies learning on a familiar floor rather than generalizing to arbitrary new warehouses.

Moves are up, down, left and right. Every attempted move costs -0.05. Reaching the destination adds +10. A collision adds -5 and ends the episode, so the last-step totals are +9.95 for arrival and -5.05 for collision. A route ends after 200 attempted moves if no terminal event occurs. Training bootstraps through time-limit truncation, while actual terminal transitions stop bootstrapping. Assessment uses frozen deterministic policies.

The same 715 training start-goal pairs and 188 development pairs are shared across methods. Their initial pairs are disjoint; training trajectories can visit overlapping states. All seven named jobs are development cases. A fixed 35-case subset is monitored every 100,000 training moves. Monitoring does not select the final checkpoint; all comparisons use the checkpoint at 600,000 moves.

## Read results correctly

- **Completion rate:** share of routes reaching the destination.
- **Collision rate:** share ending in a wall or boundary attempt.
- **Timeout rate:** share using the entire move budget without arriving.
- **SPL:** shortest-path length divided by actual route length for a success, and zero for a failure; averaged over all routes. Higher is better, up to one.
- **Seed SD:** sample standard deviation between the three independent training runs. It is not a confidence interval or a significance test.
- **Looping timeouts:** timeouts whose recorded positions repeat; the counter does not identify the cause of learning failure.

The current [results report](../autonomous-robot-navigation-rl/docs/warehouse_comparison_results.md) has the measured numbers and their interpretation. Equal action budgets do not mean equal compute or equal hyperparameter tuning. The methods retain their own learning rules and registered settings.

## How the code works

`run_app.py` starts the installed local Python environment and Streamlit entry point `app/streamlit_app.py`. `navigation_ui.py` owns map and replay-mode selection; `warehouse_ui.py` loads a checked model/run and its saved route recordings. `warehouse_results_ui.py` reads the checked comparison publication. `user_guide_ui.py` supplies the plain-language help. The original dark styles are in `app_theme.py`.

`warehouse.py` loads the floor and creates the task from `data/warehouse/warehouse-delivery-v1/` and its pinned environment settings. `core.py` defines the grid, navigation environment, signatures and transition rules. `agents.py` implements tabular policies and BFS; `dqn.py` and `ppo.py` wrap Stable-Baselines3 models. `training.py`, `part_a.py` and `training_traces.py` implement learning ledgers and actual action recording. `warehouse_study.py` freezes the protocol and inputs, runs the twelve jobs, reloads final models, assesses all routes, audits recorded transitions and publishes seed-level aggregates.

`episodes.py` validates recordings and controls single/shared-clock playback. `interface.py` draws maps and metrics from the visible prefix. `warehouse_scene.py` adds semantic fixtures/stations; `preview_3d.py` renders that same geometry. Opening a saved route does not train. Custom-route creation performs frozen inference and checks that learned state has not changed.

The study preserves exact engine/config/map/protocol hashes and checksummed output files. The project inventory checks current app dependencies separately. Frozen historical studies stay byte-identical and do not acquire warehouse labels or new scores.

## Scope

This is an actual registered, trained and measured simulation study. Its 188 development routes were already inspected during earlier work. A fresh confirmatory result would require new reserved cases after freezing the methods. The simulation does not model parcel loading, robot dimensions, continuous motion, live sensors, people, moving obstacles or battery consumption, so it cannot establish physical-robot readiness.
