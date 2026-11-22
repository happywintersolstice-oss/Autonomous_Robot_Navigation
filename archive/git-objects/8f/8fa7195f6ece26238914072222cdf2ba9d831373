# Using the navigation dashboard

The sidebar is for changing research pages. Replay controls and model choices
are on the main screen, next to the content they affect.

## Choose a replay tab

| Tab | What it does |
| --- | --- |
| Trained replay | Plays a saved example of a model acting after training. Choose the model and example, then Play. |
| Learning replay | Plays actual actions recorded during training, including exploration. Choose a training run and recorded episode. |
| Compare agents | Gives different agents the same start and goal. Choose two models or the four-algorithm baseline lineup. |
| Try a route | Lets you choose start and goal cells and run a fixed trained model once. This does not retrain it or add benchmark results. |

Changing tabs pauses playback. Only the active tab runs its animation timer.
These tabs do not run live training.

“Trained” means a saved model produced by training. Its selected checkpoint
can precede the last training step. The replay library uses validation episodes;
it is separate from the archived final-test results on the research pages.

## Read the controls and map

- **Play / Pause:** start or stop the recording.
- **Next step:** advance one robot action.
- **Reset:** return to the beginning and pause.
- **Playback speed:** requested robot steps per second. Rendering can slow it.
- **Replay position:** move to a recorded step; seeking pauses playback.
- **S / G / R:** start, goal and the robot's current position. Dark blocks are walls.
- **Actions:** the number of displayed robot steps, up to the task's 200-step cap.
- **Cumulative reward:** the total score earned so far, not a success percentage.
- **Revisited cells:** returns to positions already visited in this episode.

The episode can still be “In progress” while playback is paused. A collision
shows the attempted cell; the robot stays at its last valid position. In a
comparison, an agent that finishes holds its last frame while the others continue.
The four-agent step table describes the displayed episodes, not overall rankings.

**Filter** narrows the model list by algorithm or experiment. A **seed** identifies
an independent training run. **Baseline** means original settings; **Tuned** means
the run used the completed tabular tuning study. The DQN input variants and PPO
baseline are not labeled as tuned models.

Map options, action scores, step history and model identifiers are available
below the main controls or inside expandable details. Q scores are not
probabilities; PPO displays action probabilities instead.

## Limits of learning replay

The new **Recorded training demo** runs cover Q-learning, SARSA, DQN and PPO.
Choose the algorithm in **Training run**, then choose a **Training episode**.
The episode label shows its stage: 0%, 25%, 50%, 75% or 95% of the training
budget, together with the actual training-step number. Each stage contains the
first two ordinary episodes starting at or after its boundary, regardless of
outcome. These are ten real training episodes per algorithm from a fresh seed,
not a complete history of all 200,000 training steps. Later episodes may still fail.

The older DQN input study recordings also remain available; those runs saved
only their first three training episodes. Original Q-learning, SARSA, DQN and
PPO runs did not record their histories. The new demo runs do not reconstruct
those histories, replace the archived models, or enter the original benchmark.
Use **Learning analysis** for saved monitoring curves. No missing historical
actions or decision scores are generated for this dashboard.

Recording checks verify the manifest declaration, checksum, training seed,
canonical training scenario and environment transitions. If a recording ends at
a training cutoff, the viewer labels that cutoff rather than inventing success,
collision or timeout. Trained and learning replays remain separate.

## Neural tuning results

Choose **Neural tuning** from **Dashboard view** for the separate
DQN/PPO studies. Use **Study** to switch between **Five-seed extension** and
**Original three-seed study**. The first table compares baselines with selected
settings across the displayed seed count. Success, collision and timeout are shown
as percentages. Paired changes use percentage points. SPL ranges from zero to one.

Use **Compare DQN models** or **Compare PPO models** to open the first declared
confirmation seed in **Compare agents**. Both models face the same route with
synchronized action steps. Change **Start and goal** to inspect other validation
routes. The initial seed is fixed before results and is not chosen for its score.

For other seeds, open **Filter**, choose **Neural tuning**, then select the
desired **Fresh baseline** or **Selected settings** model. Those runs also contain
bounded actual training episodes in **Learning replay**. The page displays the
verified study after its integrity checks finish and reports progress beforehand.

Expand **Two additional seeds separately** to inspect the added-seed averages.
Their results differ from the earlier three-seed pattern. The expansion note
records that the extra seeds were declared after the original outcomes.

At narrow widths, model and route selectors stack. Collapse the sidebar for
more reading space. Wide tables keep internal horizontal scrolling. Run names
ending in **attempt2** identify the completed PPO recovery cohort. Their earlier
interrupted artifacts are retained separately in the research record.
