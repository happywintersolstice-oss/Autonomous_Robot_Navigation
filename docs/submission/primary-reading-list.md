# Primary sources for the team to read

These references support concept clarification and source checking. They are not a generated literature-review argument or proof that a student has read the papers. Record the pages and claims each member actually consults, then write the team's own comparison.

- Watkins, C. J. C. H., and Dayan, P. (1992). Technical Note Q-Learning. *Machine Learning*, 8, 279–292. [Author-hosted paper](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf), [DOI](https://doi.org/10.1007/BF00992698). Read its discrete-state/action sampling assumptions before citing a convergence guarantee; do not transfer that guarantee to this clipped linear, partially observed application without justification.
- Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, second edition. MIT Press. ISBN 9780262039246. [Publisher record](https://mitpress.mit.edu/9780262039246/reinforcement-learning/). Use the text to check temporal-difference control and the distinction between the behavior action used by SARSA and off-policy targets. Confirm the exact section/page used in the submitted citation.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533. DOI 10.1038/nature14236. [Author organization paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). Compare the replay/target mechanisms with this code; do not claim that this public-feature MLP is the paper's pixel-based Atari convolutional agent.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv 1707.06347. [Authors' preprint](https://arxiv.org/abs/1707.06347). Check the clipped objective and rollout/minibatch setting before explaining what the implemented PPO learner actually optimizes.

Source metadata and paper links were checked on 23 November 2026. A publisher sign-in interrupted the Nature landing page; the author-organization paper above was readable. The authors' book site was unavailable during this check, so use the verified publisher record and a legitimate copy of the book. No bibliography entry implies that the team has completed its required critical discussion.

Questions for your own reading notes:

1. Which assumptions differ between a fully observed Markov task and this occupancy-memory/history input?
2. How are the implemented algorithms related to the original methods, and what has been modified?
3. What does each measured panel support, and what would be an unsupported claim about unseen warehouses?
4. Which comparison actually controls an optimization setting, and which changes several factors together?
5. Why can voting increase inference cost without improving on the strongest single policy?
