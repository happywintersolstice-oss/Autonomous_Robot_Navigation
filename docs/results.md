# Recorded product results

These are saved measurements from the completed studies, displayed here without running policies again. Phase3 selected checkpoints and Phase2 references were evaluated on the same 64-case product holdout after development-only selection. This is one product panel/seed, not a claim of universal reliability.

| Family | Phase2 success /64 | Phase3 success /64 | Phase3 timeouts /64 | Phase3 Wait / all actions |
| --- | ---: | ---: | ---: | --- |
| q-learning | 48 | 59 | 5 | 209/2012 (10.39%) |
| sarsa | 52 | 64 | 0 | 85/1066 (7.97%) |
| dqn | 38 | 58 | 6 | 1161/2499 (46.46%) |
| ppo | 43 | 64 | 0 | 113/1118 (10.11%) |

All eight RL rows have zero collisions on that panel. Q-learning and DQN retain long-route timeouts; DQN still over-waits. See successful-only denominators, strata and failures in the [summary CSV](../autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/heldout-comparison/summary.csv), [case CSV](../autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/heldout-comparison/case-results.csv), [comparison metadata](../autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/heldout-comparison/manifest.json) and [replay ZIP](../autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/heldout-comparison/results-and-replays.zip). The heldout suite/version and code/checkpoint hashes are recorded there.

The optional four-model vote is measured separately on development/manual scenes, not this frozen holdout. See [rubric implementation status](rubric-completion-status.md) for the actual ensemble costs/limits and remaining human evidence.

Original reports retain their original text and recorded Git context: [Phase2](history/reports/PRODUCT_PHASE2_RESULTS.md), [Phase3](history/reports/PRODUCT_PHASE3_RESULTS.md), and [initial application delivery](history/reports/USABLE_APP_DELIVERY.md). Some original relative links describe their historical tree; use the current CSV/JSON/ZIP links above for the current layout.
