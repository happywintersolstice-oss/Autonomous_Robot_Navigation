# Development log — 6 November 2026

Implemented Section 12 in its own project directory with Codex assistance.
The existing root README, plan, attributes and Git files were left in place.
No energy files or `sources/` folder were present in this checkout at inspection.
Original-file SHA-256 verification is recorded at handoff.

Scope: fixed grid, deterministic manifests, random/BFS, Q-learning smoke pilot,
metrics/tests, local animated replay. SARSA/DQN belong to later milestones. PPO,
new maps, moving obstacles, continuous control, robots, tuning, voting and the
comparison dashboard were not implemented. No external AI API/cloud is used.

Python 3.14.10 was the installed interpreter. The isolated environment successfully
installed Gymnasium, NumPy, Matplotlib, Streamlit, PyYAML and pytest; exact versions
are captured in `requirements-lock.txt`. Python 3.14 compatibility was tested for
these milestone dependencies only; later SB3/PyTorch compatibility is unverified.
The sandbox blocked the first PyPI connection; the approved dependency install
then succeeded. The editable package needed setuptools installed explicitly.

Implementation choices: use a small built-in tested time-limit counter rather than
stacking wrappers, keeping goal/collision priority on action 200. Use Matplotlib
prefix rendering inside a Streamlit fragment; playback state never calls a policy.
Custom demonstrations execute frozen inference through the same rollout, then
become read-only replays. Checkpoints are inference-only snapshots; exact training
resume is unsupported. The smoke saves the final-budget model without selection.
Each run uses a unique directory and refuses to overwrite prior runs.

Before pilot: 30,000 training interactions, seed 7, fixed 30-scenario validation
monitoring every 10,000, complete validation at the end; no final-test execution.
No navigation-success target is set for this diagnostic pilot, and it is not a
main Part A comparison. Provisional viewer target: comfortable 2 actions/second
playback on this machine, with manual pause/seek responsive within about one second.
Actual outcomes and limitations are recorded in `milestone_results.md`.

Final verification: 16 tests passed; Gymnasium checker passed; every saved validation
trajectory was replay-validated. The run's original source snapshot is retained and
hash-verified. Visual inspection found overlapping renderer footer text; spacing was
fixed and the original trajectories re-rendered. UI import order now selects the
project-local Matplotlib cache before importing pyplot. No learning code changed.
Computer Use ended at its URL-confidence policy gate, leaving browser-visible
animation verification open. Original-file hashes matched at handoff.

Sources used for API contracts (consulted 6 November 2026):
- [Gymnasium custom environment](https://gymnasium.farama.org/introduction/create_custom_env/)
- [Farama step API](https://farama.org/Gymnasium-Terminated-Truncated-Step-API)
- [Streamlit fragments](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment)

Code was written for this project; no external navigation repository was copied.

## Milestone 2 — 6 November 2026

Added SARSA, a shared Q-learning/SARSA training loop, independent evaluation CLI,
algorithm-tagged checkpoints with legacy loading, periodic inference snapshots,
failure/interruption status, single-pass compressed evaluation trajectory export,
and algorithm-aware replay/custom demos. Preserved the original smoke bundle,
benchmark inputs and root files. No energy project or `sources/` was present here.

Recorded the development protocol before running three seeds per algorithm at
200k interactions plus one seed-7 repeat each. All eight runs completed. Mean
validation success: Q-learning 66.45%, SARSA 25.68%; repeated seeds reproduced
tables, training logs and replays exactly. All saved validation actions and 160
periodic checkpoints were audited. Final test remains sealed. Full results and
limitations are in `milestone2_results.md`; this is not the full Part A study.

23 tests and the Gymnasium checker pass. The inherited browser blocker was resolved
with an isolated local Chrome profile; the existing user browser was left alone.
Restarted the owned Streamlit server after its old import cache caused an ImportError.
Observed BFS and saved Q-learning start-to-goal replay in-browser. Fixed dark-mode
metric contrast and narrow-window clipping after inspection. Run source snapshots
preserve the training revision; post-run changes affect UI styling/layout and runner
output-name options. No additional dependencies, cloud calls or public publication.
