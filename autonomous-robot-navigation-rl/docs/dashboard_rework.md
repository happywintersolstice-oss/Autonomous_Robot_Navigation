# Simulator interface rework

The requested direction is simulator first: animated navigation, playback
controls and side-by-side comparisons, in a dark robotics control-room style.

## Changes

- Mission setup, replay selection, custom scenarios and map layers live in the
  sidebar. The main view prioritizes the task, controls, map and live telemetry.
- The default run is the predeclared selected system,
  `m4-tuned-q-learning-seed101`. Historical runs and recorded failures remain
  selectable. Run labels explain baseline/refined stage and training seed;
  raw run IDs remain visible.
- Comparison defaults to refined Q-learning versus its seed-101 baseline.
  Both maps share one action clock. Completed agents hold their terminal frame;
  counters and recorded decisions use each agent's clamped action index.
- The existing scientific map renderer supplies actual prefix positions,
  collision attempts, optional visit counts and labeled BFS reference. The UI
  recolors and caches its PNG output; archived report figures remain unchanged.
- A warning recognizes two identical observed blocks of state/action/next-state
  transitions, with periods of 1–12 actions, for learned policies only. It uses
  no future actions, predicts no outcome and adds no recovery or exploration.
- Recorded decisions, event seeking, custom demonstrations, training summaries
  and all other evidence pages remain available.

## Verification

The full local suite passed: **69 tests**. The Gymnasium environment checker
passed. Automated coverage includes step/play/pause/reset/seek, a due animation
tick, custom scenario validation, all evidence pages, overlay state preservation,
early checkpoint comparison, prefix-only loop detection and terminal-frame
holding with synchronized per-agent counters. Event-log seeking is disabled
during playback, and its callback ignores stale events while playing. This
prevents an evolving event selector from rewinding an active animation.

Live Chrome verification observed continuous playback to success on validation
scenario `r0c0-r0c3`: 3 actions, position `[0, 3]`, return 9.97 and SPL 1.
Matched playback reached shared index 3 with the refined agent at success and
the baseline held at its one-action boundary collision (return −5.01, actual
position `[0, 0]`, attempted position `[-1, 0]`). The dark maps, markers,
trails and collision arrow were visually inspected in the dedicated local
preview. Native Chrome was used because the browser automation host was
unavailable. Desktop preview was inspected at 1536 and 1038 pixels wide;
phone and touch-device testing were not performed.

An initial spacing recheck on 7 November 2026 verified map/header alignment,
but missed the playback buttons touching the scenario panel and the uneven
speed-control row, as the user's screenshot subsequently demonstrated.
Both playback modes now use a keyed toolbar container with 18 px top padding
and 12 px bottom padding. Native columns vertically center the buttons beside
the taller speed slider. Padding is applied to the actual Streamlit container,
rather than relying on the scenario HTML's bottom margin.

The correction passed all 23 dashboard, playback and interface tests. The
7,249 protected-file hashes and 184 archived frozen inputs were reverified
unchanged. No training or evaluation was rerun.

### Map-image distortion correction

A subsequent user screenshot showed one SARSA panel with a shrunken grid and
overlapping map graphics. Both episodes use the same validated task; these are
not different obstacle maps. The UI renderer previously resized a figure with
automatic constrained layout enabled. It now disables automatic layout, sets
identical figure dimensions and explicit axes bounds, and serializes the full
Matplotlib draw/save/close operation across dashboard session threads. Figure
cleanup runs even if image generation fails. Restarting the app cleared the
previously cached PNGs.

The exact original corruption was not reproduced in a 24-render parallel probe
before the change, so its precise trigger remains unconfirmed. After the change,
24 parallel renders also matched their serial references. A regression test
checks render isolation, identical axes bounds and identical initial map pixels
for SARSA seeds 503 and 211. The full suite passed all 69 tests. Live inspection
of that exact pair at action 0 showed equal grids without overlapping graphics.
All 7,249 protected files and 184 archived frozen inputs remained unchanged.

SHA-256 preservation checks covered **7,249 existing protected files**, with no
changes. All **184 original frozen snapshot inputs** and the final freeze seal
verified. These checks include existing run artifacts, checkpoints, replay files,
scenario data, study results and root project files. No training, evaluation or
benchmark regeneration was performed for this rework.

## Reproducibility and remaining limits

This is a post-release UI revision. The live `app/streamlit_app.py` and
`src/robot_navigation/dashboard_ui.py` intentionally differ from the original
frozen bytes. Keep the original freeze and snapshots unchanged. To prepare the
exact scientific release in a new isolated directory, use:

```powershell
.\.venv\Scripts\python.exe scripts/replay_frozen_release.py --workspace .cache/frozen-ui-review
```

The command verifies the archived version without executing final-test policies.
Do not apply the old freeze to the revised working dashboard or rewrite its
hashes to match the new UI.

The map still uses discrete cell steps. Playback speed is a requested cadence;
rendering and browser scheduling can slow it. Weak frozen policies still loop
until their original episode termination condition. Deferred features remain
deferred, including unseen maps, dynamics noise and hardware integration. PPO was
subsequently implemented as the owner-authorized fourth algorithm; the current
four-family tables and verification are documented in
[the analysis results](four_way_analysis_results.md).

## Live playback correction — 8 November 2026

With the local dashboard active in Chrome, a new live check reproduced Play
rewinding to frame zero despite passing automated tests. Temporary callback
logging showed `control play` at index 0, followed by a delayed timeline seek
to 0 after the player reached index 1; stale event-log callbacks could then
bounce between frames. Diagnostic logging was removed after identifying this.

Timeline widgets now use frame-specific keys and ignore events from superseded
widgets. Duplicate current-frame events do not pause Play. Event seeking accepts
only the currently displayed event widget, and remains disabled during playback.
Paused viewers do not schedule timer reruns; reaching a terminal frame performs
one full rebuild with the timer disabled. Both single and matched replay use the
same seek guard. The server was restarted to clear an old imported module.

All 96 tests passed in 44.06 seconds in the fresh environment, including two new
delayed-event regressions. Native Chrome observed PPO seed91 advance to action 5
and finish at action 9, `[9,0]`, return 9.91, SPL 1.000. Rewinding to action 2
paused at `[2,0]`/-.02; expanding probabilities preserved the frame. Reset cleared
the old decision table at index 0. PPO seed7 versus control Q-learning101 on the
same three-step scenario reached shared index 3 with PPO at success/9.97 and the
control held at its one-action boundary collision/-5.01. The map and terminal
markers were visually inspected.

At approximately 854 pixels wide, the old toolbar clipped labels. Controls now
form two button rows with a full-width speed slider; the single replay map and
telemetry stack vertically. Full labels and the larger map were inspected live.
The four-family comparison/cost tables and distance/corridor group selectors
were also inspected at desktop width. No phone/touch or measured cadence claim
is made. This remains a presentation-only revision with no research retraining
or changes to benchmark rules, learned checkpoints or frozen final results.


## Task-based replay tabs ? 9 November 2026

The simulator now uses Trained replay, Learning replay, Compare agents and Try a
route tabs. The sidebar contains only research-page navigation and short help;
model, episode and comparison choices are in the main screen. Experiment filters
use a popover. Map options, identifiers and detailed recording information are
secondary controls rather than a persistent sidebar stack. Technical Focus/Pair
workspace labels were removed, and playback and episode states have separate,
plain-language labels.

Trained replay defaults to an example from the selected learned model rather
than a BFS reference. Custom routes default to the chosen trained model; BFS and
Random remain explicit references. The four-family baseline view uses the same
registered training seed and budget for Q-learning, SARSA, DQN and PPO, with one
shared action clock and held terminal frames. It is labeled as exploratory
baseline evidence, not four tuned agents or aggregate performance.

Learning replay reads only manifest-declared training traces. It verifies checksums,
training identity and recorded transitions and does not load a trained model to
reconstruct past behavior. Complete and cutoff recordings have distinct handling.
Only the first three chronological episodes from each M8 pilot/main DQN run exist;
older algorithms have no recoverable training replay. The screen and user guide
explain that limitation. Tab changes pause hidden playback, and inactive views do
not run animation fragments. The existing learning/evaluation evidence pages remain.

Validation: all **118 tests passed in 35.81 seconds**, including training/trained
tab separation, no model queries during learning replay, cutoff handling, tamper
rejection, four-agent terminal holding, training-seed changes, empty filters,
existing controls and research pages. All **8,816 protected files** were byte-checked
unchanged. All **184 archived frozen inputs** and the original freeze seal verified.
No training, benchmark regeneration or new final-test policy execution occurred.
Streamlit 1.67 is now the minimum because the tabs require tracked active state;
the existing dependency lock already pins that version.

Live Chrome verification is now complete for this UI revision at approximately
1536 px and 854 px window widths, including the expanded navigation sidebar.
The first open page was an older server on port 8518 with stale imported modules;
restarting that project process and reloading the page resolved its import error.
Both local preview ports 8518 and 8520 now serve the updated application.

The live check found small comparison grids at narrow width and a clipped Filter
label when the sidebar was expanded. Comparison rows now stack below 1100 px;
four-agent maps use the larger display bound. Model/episode selectors also stack
at narrow width, and the filter button has reserved space. The desktop inspector's
reward label clipped with the sidebar open, so its two metric cards now stack.
All corrections were visually rechecked. The sidebar contains navigation and help
only; controls, selectors and map/inspector alignment were inspected in Chrome.

Observed behavior: the selected Q-learning replay reached [0,3] in three actions
with return 9.97. A learning replay's first recorded action moved to [3,9] for
-0.01, matching the saved training trace. Pair playback held the control at its
one-action collision/-5.01 while the selected policy finished. Four-agent playback
paused consistently at shared index 27, with Q-learning/SARSA/PPO held at their
three-action successes and DQN at actual action 27, [1,0], return -0.27. The custom
form generated a fixed-model [0,0] to [9,9] demonstration with shortest route 18.

The full suite passed again: **118 tests in 51.45 seconds**. After the final
inspector-card presentation adjustment, **22 related UI tests passed in 20.31
seconds**. Protected-file and archived-freeze checks remained valid. This is a
desktop/narrow-browser check, not phone/touch testing or a measured frame-rate
claim. It does not substitute for visual review of every research-evidence page.
Instructions for the revised interface are in docs/dashboard_ui_guide.md.

## Five-seed research-page native check on 9 November 2026

Verified the Neural tuning page in native Chrome at 1536×912 and 854×722.
The original/five-seed selector restores the appropriate values and scope text.
The five-seed table, post-results expansion note, paired/monitoring figures,
two-added-seed section and per-seed cost table were inspected. Wide tables retain
their internal scrolling at narrow widths. Both scientific images were also
inspected as standalone artifacts.

The narrow check found a squeezed second-model selector beside the route
selector with the sidebar open. Those controls now stack at narrow widths.
The full-width second-model control was verified after refreshing the preview.

The matched PPO seed 7307 button opened the correct baseline/selected models.
An illustrative saved validation route r8c1-r5c0 showed selected settings reach
[5,0] in four actions with return 9.96 and SPL 1.000, while baseline held its
one-action collision at [8,1] and return -5.01. Seeking to shared frame 2 showed
selected [7,0]/-.02 and held the baseline terminal state. Reset restored both
to action 0 and zero return. A separate Next step on r0c0-r0c3 moved both models
to [1,0] for -0.01. These display checks are excluded from benchmark aggregation.
Manual Pause is covered by the automated tests. Phone/touch, screen-reader
usability and quantitative frame-rate measurements remain untested.

All 158 tests passed in 68.21 seconds after the completed recovery run was used
for the late-stage learning replay test. See docs/neural_confirmation_results.md
and docs/neural_confirmation_verification.json for data, recovery and limits.
