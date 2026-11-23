"""MediNav-RL: hospital delivery robot navigation with reinforcement learning.

Package layout (each module starts with a short "why" comment for the report/Q&A):

    medinav.env          - the simulated hospital world (Gymnasium environment)
    medinav.agents       - baselines (random, A*), tabular RL, deep RL, ensembles
    medinav.training     - training loops, callbacks, curriculum learning
    medinav.evaluation   - metrics, robustness tests, interpretability, report figures
    medinav.tuning       - Optuna hyper-parameter search (TPE vs random)
    medinav.cli          - one command line entry point that automates everything
"""

__version__ = "3.0.0"
