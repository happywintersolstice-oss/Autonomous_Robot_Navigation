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
deferred, including unseen maps, dynamics noise, PPO and hardware integration.
