# Navigation dashboard: comparable UI research

Research date: 9 November 2026.

## Scope and evidence

The brief is a simulator-first interface with animation, controls and algorithm
comparisons in a dark robotics control-room style. This research compares
published implementations, not just concept designs. Their availability is not
proof that every feature works: none of the external applications was installed
or executed during this review.

I visually inspected the authors' published PathBench, ReinforceUI Studio and
Husarion screenshots. Arena-Web was reviewed through its published paper and
figure descriptions; its screenshots could not be inspected through the web
tool. The supplementary warehouse example was checked against its application
source. Recommendations below are design judgments, not measured usability results.

## Relevant implementations

### 1. PathBench: simulator workspace

PathBench is a platform for classical and learned path planners. Its published
interface places a large scene between configuration and view-editor panels,
with a compact map/algorithm/position/state overlay. Configuration includes map,
algorithm, animation and start/goal; view settings expose layer visibility and
appearance. This is a useful reference for giving the simulation visual priority
while keeping debugging and appearance settings secondary.

Sources: [repository and interface documentation](https://github.com/perfectly-balanced/PathBench),
[inspected screenshot](https://raw.githubusercontent.com/perfectly-balanced/PathBench/master/readme_files/visualiser_full.png).

Its 3D controls and extensive color editor are unnecessary for our fixed 2D task.

### 2. ReinforceUI Studio: algorithm monitoring and comparison

The published multi-algorithm screenshot separates settings on the left from a
large shared training chart. Algorithms have separate curve colors and status
tabs; the footer shows steps, episode information and progress. A separate
control switches training and evaluation curves. This offers a clear reference
for organizing our existing saved learning evidence and distinguishing curve
types.

Sources: [repository](https://github.com/dvalenciar/ReinforceUI-Studio),
[inspected comparison screenshot](https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/multiple_selection_main_window.png).

This is an RL workflow application rather than a grid-navigation dashboard. Its
training controls should not imply that our replay viewer is training a policy.

### 3. Arena-Web: guided research workflow

The RSS 2023 paper describes Tools, Training and Evaluation as separate areas.
Setup proceeds through explicit choices rather than exposing every setting at
once. Its scenario editor combines a visual preview with numeric configuration.
The paper also distinguishes implemented CSV export and external plotting from
future integrated web plotting. This is a strong reference for task organization,
but not evidence of a currently working hosted service or live web charts.

Source: [Arena-Web paper, especially Figures 2, 6 and 8](https://www.roboticsproceedings.org/rss19/p088.pdf).

For our project, apply this to experiment selection and custom demonstrations;
map generation, dynamic obstacles and robot configuration remain deferred.

### 4. Husarion ROSbot XL with Foxglove: navigation console

The vendor's screenshot gives most space to the navigation scene, with layer
settings on the left and logs plus directional controls underneath. The useful
pattern is a central spatial view surrounded by compact inspectors and an event
history. This is a hardware/navigation reference, not an RL comparison project.

Sources: [vendor tutorial](https://husarion.com/tutorials/ros-projects/rosbotxl-foxglove-nav2/),
[inspected screenshot](https://husarion.com/assets/images/rosbotxl_foxglove_ui-fdbbcd9c47a72c639dbcd1969f11362a.png).

Treat this as a published layout reference, not a recommendation to adopt its
historical deployment stack. We do not need cameras, ROS topics or a joystick.

### Supplementary example: Streamlit warehouse navigation

The application source uses a sidebar, a 3:1 grid/metrics layout, comparison
charts and replay controls. However, `render_training_curve()` plots a rolling
mean of session `reward_history` under a training/learning heading and labels it
cumulative reward. It is insufficient evidence of training progress. This makes
the layout a useful reference with a specific scientific-labeling caveat.

Sources: [repository](https://github.com/TechySan031/Reinforcement-Learning-in-Warehouse-Routing),
[reviewed application source](https://github.com/TechySan031/Reinforcement-Learning-in-Warehouse-Routing/blob/main/app.py).

## Recommended direction for this dashboard

Keep the existing dark style and Streamlit application. The current code already
provides separate evidence pages, playback controls, synchronized pair comparison,
telemetry, loop warnings and recorded decisions. The next UI pass should improve
their organization and make four-algorithm comparison easier to discover.

Suggested simulator structure:

```text
Navigation Lab        Simulator | Compare | Learning | Study evidence
Experiment / algorithm / run       Scenario / episode       Replay status

Compact setup panel   Large map or matched map panels       Inspector
                      Playback + timeline                  Action / reward
                                                           Position / outcome

                      Events | Recorded decision | Run details
```

1. **Keep the map and playback together.** Use one compact toolbar immediately
   beside or below the map. Keep its spacing stable when labels or status change.
   Put detailed diagnostics in expandable panels instead of enlarging every card.
2. **Offer explicit comparison layouts.** Provide Focus, Pair and Four-agent
   views. Use equal square map areas and one shared scenario/action clock. A 2x2
   layout suits four agents on a wide display; stack panels or show one focused
   agent with an accessible switcher on narrow screens. Keep a compact four-agent
   outcome summary visible. Hold terminal frames and show each agent's actual
   action count, as the current pair comparison already does.
3. **Make model identity understandable.** Select experiment family, algorithm,
   then seed/checkpoint. Show readable names first and retain the raw run ID in
   details. Explicitly label baseline, tuned, representation control/variant and
   frozen final evidence. Do not label DQN or PPO tuned unless a completed tuning
   study supports that label.
4. **Separate playback state from episode state.** For example, show `Paused`
   beside `Action 42/200`, while the episode outcome remains `Running`. Display
   success, collision, timeout and observed loop warnings in plain language.
   Explain repeated actions without modifying the frozen policy to escape them.
5. **Use charts for the question being asked.** Comparison should emphasize
   success/collision/timeout, path efficiency, seed variability and training cost.
   Learning should show saved training/evaluation logs with labeled axes and
   units. A single replay's reward trace belongs in the episode inspector.
6. **Use animation to clarify actual actions.** Any optional movement tween is a
   visual transition between recorded cells; state, reward and counters advance
   only on recorded actions. Never display future trajectory as traveled history.

These are proposed improvements, not implemented changes. They preserve the
fixed map, task rules, historical artifacts, frozen results and existing energy
project. Hardware, SLAM, new maps and dynamic obstacles are outside this UI pass.

## Verification needed before implementation is considered complete

- Inspect focus, pair and four-agent layouts at desktop and narrow widths.
- Exercise play, pause, step, reset, seek and speed changes without layout jumps.
- Confirm shared scenario/clock, equal map geometry and terminal-frame holding.
- Confirm labels distinguish saved replay, custom demo and aggregate evidence.
- Recheck the existing M8 live-dashboard verification item separately; this
  external research does not resolve it.

No application code, policies, training runs or benchmark outputs were changed
for this research.
