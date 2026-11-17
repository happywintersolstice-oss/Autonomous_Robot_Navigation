# Robot Navigation user guide

Double-click **Start Robot Navigation.cmd** in the project folder. Keep its command window open. The app opens at **http://127.0.0.1:8517**. If it is already running, open that address directly. To stop it, press Ctrl+C in the command window.

## Watch the robot

1. Choose **Demo** at the top of the app.
2. Choose a **Navigation map**. The original bottleneck uses the original selected model. The four newer maps offer dead ends, loops, rooms and longer wrong turns.
3. Leave **Watch the robot** selected and press **Play**.
4. Use **Pause**, **Next step**, **Reset** or the replay timeline to inspect its moves. **Speed** changes animation speed.

The blue circle **R** is the robot, **S** is its start and green **G** is its destination. Dark squares are walls. The blue trail shows the route travelled. The map's row and column numbers identify individual squares; numbering begins at zero.

**Moves used** shows the move count and the limit. **Distance travelled** counts successful square-to-square movements. **Distance to goal** ignores walls, so the actual route can be longer.

The status explains whether the robot reached the goal, collided or ran out of moves. A replay can fail: Play does not retrain the robot or help it escape a loop. Successful examples are illustrative, not proof that every route succeeds.

## Choose your own route

Open **Choose a route**. Select a **Starting square** and **Destination** from the lists of open squares. You can preview the selected squares before generating. The numbers correspond to the map axes. For precise manual entry, expand **Enter exact coordinates (optional)**. Manual entries are checked and may be rejected if they select a wall, the same square or an unreachable goal.

Choose the trained robot, the **Shortest-path planner (BFS)** or **Random (untrained)**. Press **Generate demonstration**, then **Play**. Changing squares does not update an existing replay until you generate again. These attempts do not overwrite saved study results.

## Compare and learn

**Compare robots** advances models on the same route together. A finished robot stays at its final position while the other continues. The original map supports two models or four algorithm families. The structured maps compare Q-learning reward settings and independently trained versions. Original-map models are not reused on incompatible maps.

**Learning history** shows actual recorded training actions where available. The newer structured maps saved learning curves, but did not record chronological training actions; the app explains this instead of inventing a replay.

**Robot settings** exposes the saved model library or reward settings. Defaults are ready to use. **Scores and decision details** explains rewards and exposes the recorded decisions. These are optional.

## Understand the results

**Results** starts with the original selected model's saved final assessment: 90.5% completed routes, 0.8% collisions and 8.7% timeouts across 928 held-out routes on the familiar map. The example in Demo is a separate validation route.

Expand **Research results and comparisons** for original-map studies and previously unseen maps. Cohorts remain separate. New-map reward results appear under the map details in Demo. The stronger time cost improved mean success modestly; potential shaping increased collisions and is retained as an unsuccessful comparison. Repetition is not completely solved.

**Guide** explains the controls, outcomes, model choice and terminology inside the app. Research methods and an installation check are available in its expandable sections.

## Why this model?

The original selected controller is tuned **Q-learning**. It stores a score for each move at each robot-and-goal position, then chooses the best-scoring move after training. A table suits this small discrete grid: decisions are fast, scores are inspectable, and the original development comparison supported the selected tuned family. Selection was made before the saved final assessment.

Each structured map has separately trained Q-learning models. This does not establish that Q-learning is universally best or reliable on new geometry. SARSA, DQN and PPO are alternative learning methods studied in the project. BFS knows the whole map and is a planning reference, while Random is untrained.

This app simulates a robot on a grid. It does not control a physical robot.
