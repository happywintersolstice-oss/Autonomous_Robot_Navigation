# Use the navigation application

Launch **Start Robot Navigation.cmd** from the repository root, or run `py -3 run_app.py`, and open http://127.0.0.1:8517. Work in the root **Robot Navigation.code-workspace** when editing.

- **Robot Simulator:** select a route, Static/Temporary obstacles and a native checkpoint. Run journey records real actions once. Play/Pause/Step/Back/Seek/Reset/End display saved frames. Human-only truth and robot memory stay separate. An honest collision or timeout is retained.
- **Train Models:** inspect checkpoint versions, training counts and integrity. Start begins a bounded background job; Refresh reads actual progress; Cancel requests a safe stop. Loading a page never trains. Quick runs can still navigate poorly.
- **Compare & Export:** inspect saved Phase2/Phase3 development evidence, the single frozen product holdout, or the separately labeled ensemble development comparison. Held-out viewing is read-only. Download CSV/JSON/replay ZIP and open a recorded failure in Simulator.
- **Guide:** read the public sensor/memory/action/reward explanation and the checked four-person presentation script.
- **Archives:** read historical reports or follow ZIP export/verification instructions. Earlier source and scientific results are reference material, not another app mode.

Phase3 selected models use public-only decision history; Phase2 baselines and old native starters keep their original feature signatures. Every robot still chooses among Up/Down/Left/Right/Wait. The optional vote and observed-map replanner are explicitly separate controllers. There is no hidden planner substitution when an RL model fails.

Read [recorded results](results.md), [the demo script](team-demo-script.md) and [the code walkthrough](code-walkthrough.md) for the actual measurements, remaining failures and implementation details. Academic submission/individual-contribution requirements are listed separately in [rubric status](rubric-completion-status.md).
