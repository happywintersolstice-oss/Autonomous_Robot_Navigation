# Autonomous Warehouse Robot Navigation: A Comparative Study of Q-learning, SARSA, DQN and PPO

Current technical report, 19 November 2026.

This project compares Q-learning, SARSA, DQN and PPO for delivery navigation around storage racks, receiving, packing, dispatch and charging on a shared warehouse floor. The study uses three independent seeds per method, 600,000 training actions per run and fixed final checkpoints. Twelve runs consume 7.2 million training actions. Each reloaded checkpoint is assessed on the same 188 development routes.

The [user and code guide](../../docs/user-guide.md) explains the problem, algorithm rationale, task/reward contract, implementation and UI. The [registered protocol](warehouse_comparison_protocol.md) records the matrix before training. The [measured results report](warehouse_comparison_results.md) contains the comparison, per-seed evidence and limitations. The [current evidence index](evidence_index.md) identifies the authoritative artifacts and checks.

The assessment routes were inspected in earlier warehouse work. The reported measurements are a controlled familiar-floor development comparison, rather than fresh final or unseen-warehouse evidence. The earlier single-seed pilot is excluded from the comparison. Physical parcel handling, continuous movement, sensors and dynamic obstacles are outside this grid simulation.

Earlier fixed-grid and unseen-map research remains in the [historical technical report](historical_technical_report.md) and [historical evidence index](historical_evidence_index.md). Those findings and sealed releases retain their original meaning.
