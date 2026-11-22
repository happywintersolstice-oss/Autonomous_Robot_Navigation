# Four-family route, cost and robustness integration

This post-release analysis uses saved development-validation evidence only. It
adds no training, policy execution or final-test evaluation. Original runs,
`four-rl-v1`, milestone-5 analyses and the frozen final archive are unchanged.

The new [study](../artifacts/studies/four-rl-analysis-v1/summary.json) reconciles
18,560 episode records across 20 runs, checks 104 input files and publishes 340
group/seed rows. All original four-family success means remain identical.
The [protocol](four_way_analysis_protocol.md) fixes the existing geometry labels
and states the aggregation and timing limits.

| Baseline family | Successful route actions, seed mean | Recorded training seconds, seed mean | Decision microseconds, seed mean | Selected checkpoint bytes, seed mean |
| --- | ---: | ---: | ---: | ---: |
| Q-learning | 6.845 | 6.203 | 1.882 | 143,352.8 |
| SARSA | 3.882 | 5.553 | 1.788 | 133,760.8 |
| DQN | 4.047 | 95.862 | 59.057 | 102,046.2 |
| PPO | 3.791 | 114.453 | 145.895 | 143,253.0 |

Route means are conditional on success, averaged equally over seeds with at least
one success. Short successful routes do not offset failure rates. Undefined means
in zero-success groups stay empty, with contributing seed counts and outcome
counts retained. The dashboard shows mean, sample SD, per-seed records and CSV
downloads. Checkpoint bytes are compressed file sizes excluding metadata, not RAM.
Training excludes monitoring/I/O/export; recorded validation time includes
reset/recording/export. Latency is the historical 100-query mean. These separate
runs are descriptive local measurements, not a hardware-isolated speed ranking.

## PPO robustness

All six historical dimensions are now available for all four baseline families.
Each dimension partitions the same 928 validation pairs per seed, across five
training seeds. Dimensions overlap. Corridor labels describe the canonical BFS
reference, and geometry/distance confound group comparisons.

| PPO scenario group | Mean success |
| --- | ---: |
| Short routes | 40.00% |
| Medium routes | 9.31% |
| Long routes | 1.69% |
| Reference corridor | 8.96% |
| No reference corridor | 44.81% |
| Near-wall start | 15.28% |
| Open start | 14.78% |
| Same quadrant | 32.43% |
| Different quadrants | 9.80% |

Start- and goal-quadrant breakdowns, sample seed SD, collisions, timeouts, SPL,
conditional route means and supports are retained in the
[full summary](../artifacts/studies/four-rl-analysis-v1/summary.json) and
[per-seed group CSV](../artifacts/studies/four-rl-analysis-v1/groups.csv).
PPO remains weak, particularly on long/corridor/cross-region cases. No tuning or
winner promotion follows these findings. These are neither causal effects nor
unseen-map results. Tuned tabular and historical final results remain separate.

To reproduce from saved metrics into a fresh output directory:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_four_way_details.py --suite-id four-rl-analysis-repeat
```

The command refuses overwrite. Dashboard pages use the verified canonical
`four-rl-analysis-v1` outputs and check their recorded output hashes before loading.
Original PPO trajectories and all previous results are preserved.
Scoped Git attributes preserve the new study's exact bytes across checkout so
newline conversion does not invalidate these output checksums.

## Verification

All 96 tests passed in a fresh environment installed from the unchanged lock
file, including all six new robustness selectors, cost tables, zero-success
handling, output-tamper rejection and reconciliation of every group's support
and outcome counts. Dependency consistency, neural/UI imports, the Gymnasium
checker and exact frozen archive preparation passed. All 7,736 existing artifact/
data files and 184 frozen inputs verified unchanged.

Live native Chrome verification passed after the owner opened the local tab.
Four-family metrics/costs and distance/corridor tables were inspected, together
with PPO animation, probability diagnostics, seeking/reset and matched terminal
holding. A delayed-slider-event issue found live was fixed and regression-tested.
The in-app browser host remains unavailable; no quantitative cadence claim is
made. Details and deferred research are tracked in [plan closure](plan_closure_status.md).
