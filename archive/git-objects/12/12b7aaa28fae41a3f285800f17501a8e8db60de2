# Plan closure and remaining scope

Updated 9 November 2026 after completion of the preliminary DQN/PPO tuning study
and owner acceptance of the preceding dashboard visuals. This page records current status.
historical milestone records remain dated evidence of their original releases.

## Implemented and measured

- All six core milestones: versioned environment, manifests, references, tabular
  and DQN baselines, bounded refinement, reward ablation, voting, fixed-map
  robustness, animated dashboard, original frozen final evaluation and report.
- Owner-authorized fourth algorithm: PPO, five independent seeds with exactly
  200,000 training interactions each. Its separate exploratory validation study
  passed verification. PPO tuning was added separately in M9 below.
- Four-family comparison includes success/collision/timeout/SPL/return, conditional
  successful route lengths and counts, recorded training/validation runtime,
  decision latency and checkpoint file size, with seed means, sample SD and
  downloadable per-seed records.
- All six predeclared geometry dimensions are available for the four baseline
  families in Optimization and robustness. The new analysis reconciles 18,560
  saved episodes and 340 group/seed rows, using 104 unchanged input files. It
  performs zero policy evaluations or training and leaves original studies intact.
- Plan, README, report addendum, development log and release/UI notes distinguish
  current PPO support from the historical three-family frozen release.
- The subsequent owner-authorized milestone 8 completed the controlled DQN
  representation study, with five fresh paired seeds, chronological training
  traces and a passed full replay audit. Mean success was 5.97% absolute versus
  6.51% absolute plus relative, without consistent seed-level improvement.
  All 110 tests passed at the M8 release. See [milestone 8 results](milestone8_results.md).

Analysis protocol, actual costs and PPO group outcomes are in
[four-family analysis results](four_way_analysis_results.md). Primary evidence:
[verification](../artifacts/studies/four-rl-analysis-v1/verification.json),
[cost records](../artifacts/studies/four-rl-analysis-v1/costs.csv) and
[group records](../artifacts/studies/four-rl-analysis-v1/groups.csv).

## Earlier automated and setup verification

All **96 tests passed in 44.06 seconds** in a fresh isolated Python 3.14.10
environment with system site packages disabled. All 58 pinned distributions
installed and match the unchanged lock file; the project installed in editable
mode with build isolation disabled. `pip check`, PyTorch/pandas/SB3/Streamlit
imports and the Gymnasium checker passed. Exact frozen-release preparation also
passed from this fresh environment, without policy execution.

The initial restricted install could not resolve the first pinned requirement.
Retrying with approved explicit PyPI access succeeded; pip reused cached wheels.
This verifies a fresh environment on this Windows machine, not a cold-cache
download test or cross-machine/hardware portability. The environment remains
local and ignored under `.cache/fresh-setup-v1`.

Reproduce setup verification using a fresh environment name:

```powershell
python -m venv .cache/fresh-setup-repeat
.\.cache\fresh-setup-repeat\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements-lock.txt
.\.cache\fresh-setup-repeat\Scripts\python.exe -m pip install -e . --no-build-isolation --no-deps
.\.cache\fresh-setup-repeat\Scripts\python.exe -m pip check
.\.cache\fresh-setup-repeat\Scripts\python.exe -m robot_navigation.cli check
.\.cache\fresh-setup-repeat\Scripts\python.exe -m pytest -q
```

Preservation checks verified all 7,736 pre-existing artifact/data files unchanged
and all 184 original frozen inputs against their seal. The current analysis is a
new study; no older evidence was rewritten. See the
[machine-readable verification](plan_closure_verification.json).

## Earlier live visual verification

Passed in native Chrome after the owner opened the local preview tab. The in-app
browser host remains unavailable, but direct observation of the active local tab
worked. Inspected the four-family route/cost tables and distance/corridor results,
PPO probabilities, automatic success playback, timeline seeking, Reset, and
matched PPO/baseline terminal holding. Desktop and approximately 854-pixel-wide
layouts were inspected; phone/touch and quantitative frame-rate/input-latency
benchmarks were not performed. The preview is `http://127.0.0.1:8518`.

The live check exposed delayed slider events that rewound and paused animation;
old event-log callbacks could reinforce the reset. Timeline widgets now have
frame-specific identities and reject stale events; redundant seeks do not pause
Play. Old event-log events are also ignored, paused timers are disabled, and
terminal frames rebuild once with the timer stopped. The new tests cover delayed
events for both single and comparison playback. Narrow controls use a two-by-two
button layout with a full-width speed slider; the single map and telemetry stack.

Observed PPO seed91 on `r0c0-r9c0`: automatic progression through action 5 to
9-action success at `[9,0]`, return 9.91, SPL 1.000. Seeking to action 2 produced
`[2,0]`, return -.02 and paused playback; opening probabilities preserved that
frame. Reset returned to action 0 and removed the old decision table. PPO seed7
versus Q-learning control101 on `r0c0-r0c3` reached shared index 3: PPO success at
`[0,3]`/9.97, control held at its one-action collision `[0,0]`/-5.01. These are
display checks excluded from benchmark summaries, not new final evaluations.

## Remaining research and limits

1. **Historical training trajectories:** historical runs retain episode CSVs,
   visitation and full evaluation traces, but selected training episodes were
   not saved transition by transition. Those historical traces cannot be
   reconstructed from summaries. Milestone 8 now implements bounded prospective
   capture and retains 36 actual early training records across its pilots/main
   runs. Subsequent four-family demos add 40 staged recordings, and the verified
   M9 suite adds 220 staged recordings. Historical actions remain unavailable.
2. **Representation scope:** milestone 8 tested normalized versus absolute-plus-
   relative DQN inputs. It addresses that portion of RQ2 under one fixed setup,
   with no consistent improvement. Other representations and capacity-matched
   neural comparisons remain untested.
3. **Preliminary neural tuning:** M9 screened four configurations per algorithm
   and confirmed DQN/PPO c01 on three fresh matched seeds. Success increased from
   5.60% to 7.11% for DQN and 16.99% to 24.35% for PPO. Both remain weak and miss
   all declared performance targets. Five-seed confirmation remains future work.
4. **Independent four-family final evaluation:** PPO was added after the original
   holdout had been inspected. The extension is exploratory development validation.
   A separately designed evaluation is required for a new confirmatory claim;
   the old final results and predeclared selected policy remain unchanged.
5. **Scope and portability:** unseen maps, dynamics noise, moving obstacles,
   continuous movement, sensors, ROS and physical hardware remain deferred.
   Bitwise repeatability across machines and exact training resume are unsupported.
   Performance misses and residual collisions/timeouts are reported limitations.

The earlier integration did not initiate new research. Milestones 8 and 9 were
authorized separately afterward. Neural tuning remains separate from the input
study and the original final evaluation.

## Milestone 9 completion

The bounded suite completed 22 runs and 3.22 million training steps in 17.89
minutes, within its declared limits. One interrupted setup attempt added 20,000
training steps and was preserved separately. All 9,371 pre-study artifact/data
files retained their hashes. Independent checks verified the original 8,816
protected files and 184 sealed archived inputs.

The audit replayed 1,751,597 validation actions and 3,464 actual training actions.
All 148 tests passed after the final figure adjustment. The dashboard has a
separate Neural tuning page with matched model comparisons. Prior dashboard
visuals were owner-verified. The new research page passed the application harness.
Its native-browser appearance has not been independently inspected.

See [M9 measured results and limits](milestone9_results.md) and
[the registered protocol](milestone9_protocol.md).
