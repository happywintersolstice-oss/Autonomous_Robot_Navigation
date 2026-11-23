# Phase 5D report delivery note

The scientific analysis completed once, consuming 3,786 new forward calls and
zero training or environment actions. All 424 full regressions and independent
data verification passed. The first report-rendering command then failed with
`TypeError: unsupported operand type(s) for +: 'int' and 'str'`: Python's CSV
writer had serialized booleans as `True` / `False`, which the JSON-only CSV
decoder treated as strings. No report was written by that failed command.

The scientific source, source snapshot, analysis plan/seal, cohorts, score
records and call journal are preserved unchanged. There is no rerun, new score,
new observation, outcome selection or training retry. The original report
source retains this known parsing defect to preserve its registration.

A separate delivery adapter at `developer/commands/phase5d_delivery.py` adds
only literal boolean decoding while invoking the frozen renderer. It has eight
additional mechanical tests at `developer/tests/test_phase5d_delivery.py`.
These files are registered separately in `delivery-plan.json`, after scoring;
they are delivery source, not preregistered experimental source. The failed
report attempt is preserved in `report-failure.json`. The first rendered report
and its hashes are archived before adding this explicit correction notice.

From the application root with the installed interpreter and `PYTHONPATH=src`,
the delivery was executed with `python developer/commands/phase5d_delivery.py`.
It is exclusive and refuses another delivery attempt. Do not rerun scientific
`prepare` or `analyze`. Verify the completed delivery with
`python -m robot_navigation.partial_phase5d verify`, and verify the additional
delivery source pins recorded in `output-hashes.json` / `delivery-plan.json`.

This correction does not remove the scientific limitations: historical DQN
replay sample/target trajectories and most PPO GAE/boundary history are absent;
no exact development reset observations or position-goal pairs occur in the
training traces. The terminal scientific verdict remains
`DIAGNOSTIC_COMPLETED_INSUFFICIENT_HISTORICAL_SIGNAL`.
