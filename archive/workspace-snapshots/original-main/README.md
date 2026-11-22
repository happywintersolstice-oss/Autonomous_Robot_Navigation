# Autonomous Warehouse Robot Navigation: A Comparative Study of Q-learning, SARSA, DQN and PPO

This project compares four reinforcement-learning algorithms for a delivery robot navigating a warehouse. Storage racks, receiving, packing, dispatch and charging give each route an operational purpose.

**Start:** double-click **Start Warehouse Navigation.cmd**, or run `python run_app.py`. Open **http://127.0.0.1:8517** and keep the terminal open.

Use **Simulator** to select an algorithm, training seed and delivery job. **Compare robots** shows all four on the same route; **Learning history** plays actions recorded during training. **Results** compares every assessed route across three seeds. **Guide** explains the controls and metrics in plain language. Saved-model playback needs no API key or new training.

The registered study gives each algorithm **600,000 training actions per seed**, using three seeds, one floor, one reward contract and the same route split: **12 runs and 7.2 million training actions**. Final-budget checkpoints are assessed on the same 188 development routes. These previously inspected routes support a familiar-floor development comparison; they are not a fresh final holdout or a physical-robot validation.

- [Measured warehouse comparison](autonomous-robot-navigation-rl/docs/warehouse_comparison_results.md)
- [How the code works, why these methods and how to use the UI](docs/user-guide.md)
- [Registered experiment protocol](autonomous-robot-navigation-rl/docs/warehouse_comparison_protocol.md)
- [Installation, verification and reproduction](docs/developer-guide.md)
- [Project organization](docs/project-structure.md)
- [Historical research evidence](autonomous-robot-navigation-rl/docs/historical_evidence_index.md)

Regression tests remain in `autonomous-robot-navigation-rl/developer/tests/`. Earlier studies retain their original maps, model IDs and sealed evidence; their scores are separate from this warehouse comparison.
