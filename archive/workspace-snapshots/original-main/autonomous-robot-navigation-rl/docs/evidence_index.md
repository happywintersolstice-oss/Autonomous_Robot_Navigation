# Warehouse comparison evidence index

**Autonomous Warehouse Robot Navigation: A Comparative Study of Q-learning, SARSA, DQN and PPO** - evidence cut: 19 November 2026.

| Question | Authoritative source |
| --- | --- |
| What was registered before training? | [Protocol](warehouse_comparison_protocol.md), [settings](../configs/warehouse_comparison.yaml) |
| What warehouse and routes were used? | [Floor](../data/warehouse/warehouse-delivery-v1/map.json), [fixed split and jobs](../data/warehouse/warehouse-delivery-v1/routes.json) |
| What code ran? | `artifacts/warehouse/warehouse-four-algorithm-v1/snapshot/`, with `freeze.json` and `seal.json` |
| Where are the twelve models and action recordings? | `artifacts/warehouse/warehouse-four-algorithm-v1/runs/warehouse-{algorithm}-seed{seed}/` |
| What are the measured results? | [Report](warehouse_comparison_results.md), [summary](../artifacts/warehouse/warehouse-four-algorithm-v1/summary.json), [CSV](../artifacts/warehouse/warehouse-four-algorithm-v1/comparison.csv) |
| How are outputs checked? | Each run's `outputs.json`; publication hashes in `published.json`; [project inventory](evidence/warehouse-project-v1.json) |
| How does a person use the app? | [User and code guide](../../docs/user-guide.md) |
| How are installation and tests verified? | [Developer guide](../../docs/developer-guide.md) |
| Where are earlier results? | [Historical index](historical_evidence_index.md), [earlier warehouse pilot](warehouse_delivery_pilot.md) |

The study has 4 methods x 3 seeds x 600,000 actions. Its 188 cases are inspected development routes on one familiar floor, with a 35-case monitoring subset. Fixed final checkpoints prevent selecting a favorable training stage. Recorded routes include collisions and timeouts. The shared-clock UI is an inspection tool; the all-route, three-seed summary is the comparison.

Historical fixed final, unseen final, diagnostic and map/reward records remain separate. No historical final cohort was rerun for this warehouse study.

The [verification record](evidence/warehouse-comparison-verification-v1.json) records the 250 regression checks, 2,256 assessment-episode audits, asset checks, preserved historical hashes and the browser-review limitation.
