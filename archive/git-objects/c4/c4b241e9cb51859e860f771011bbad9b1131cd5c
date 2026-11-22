# PPO extension status

Requested by the owner on 8 November 2026: add PPO to Q-learning, SARSA and DQN
to compare four trained reinforcement-learning algorithms.

## Current execution

The earlier Windows Application Control blocker has been resolved by the owner.
The same project environment now imports PyTorch 2.15.0+cpu, pandas 3.1.1 and
Stable-Baselines3 2.9.2. All **90 tests** passed, including PPO learning on a tiny
fixture, timeout bootstrapping, checkpoint integrity, exact training budgets,
probability diagnostics and replay/custom/matched-comparison UI wiring.

The independent PPO pilot (seed 101) completed 10,000 training interactions in
12.33 seconds including monitoring/export. It reached 40 of 928 validation goals
(4.31%). This pilot is excluded from the main comparison and did not trigger any
hyperparameter changes. Its throughput estimated the five-seed matrix at about
568 seconds, below the registered limit.

The main matrix completed seeds 7, 19, 43, 67, 91 at exactly 200,000 training
interactions each (1,000,000 total), in 654.21 seconds including monitoring and
export. Verification passed for all 20 comparison runs, including all 100 PPO
monitoring checkpoints and 241,569 recorded PPO validation actions. All 128
historical comparison inputs match their registered hashes.

The workspace preservation check examined 7,249 protected files and found no
unauthorized changes. All 184 archived frozen inputs still match their seal.
The root project-plan edit records the owner's authorized PPO scope extension.

The dashboard's **Agent comparison** view now publishes the four-family table;
**Learning analysis** includes four-family monitoring curves. PPO averaged
15.22% validation success across seeds (sample SD 13.46 percentage points).
See [the measured results](ppo_comparison_results.md) for all four algorithms,
per-seed PPO outcomes and interpretation limits.

Automated replay, custom demonstration and matched-comparison checks passed.
Live visual verification remains unresolved: Windows computer-use automation
stopped because it could not reliably determine Chrome's current URL. No visual
verification is claimed.

## Implementation

- Vanilla PPO with separate actor/value networks and coordinate-only inputs.
- Deterministic frozen inference, categorical action probabilities and state value.
- ZIP checkpoints with checksums and task compatibility validation.
- Rollout-aligned budgets, post-update monitoring and full validation exports.
- A separate pilot/matrix runner, historical-file hashes and runtime/storage limits.
- A four-way validation analyzer, seed statistics, learning curves and audited results.
- Simulator, custom demonstrations, matched playback and early-checkpoint support.

PPO is a configured additional baseline; it is not hyperparameter-tuned. The
comparison reuses the original Part A baseline Q-learning, SARSA and DQN runs.
The old tuned tabular runs and frozen final evaluation remain separate and unchanged.

## Commands

From the project directory:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/run_ppo_comparison.py pilot
.\.venv\Scripts\python.exe scripts/run_ppo_comparison.py run
.\.venv\Scripts\python.exe scripts/analyze_ppo_comparison.py
```

The pilot and matrix refuse to overwrite existing runs/studies. Their output is
`artifacts/studies/four-rl-v1` and `artifacts/runs/m7-ppo-*`. The dashboard table
requires a passed verification record. No final-test policy execution is added.

See [the predeclared protocol](ppo_comparison_protocol.md) for configuration,
budget, checkpoint-selection and interpretation details.
