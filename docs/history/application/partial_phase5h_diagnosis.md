# Phase 5H diagnosis

Verdict: **INCONCLUSIVE_SMALL_PILOT**.

The paired comparison changes one objective plus its necessary132-parameter head. It tests navigation outcomes, not just whether a network can decode an already visible direction. Direction accuracy, goal-reward optimizer membership and navigation success are separate observations. A single seed on familiar development pairs cannot identify a causal mechanism or establish a model ranking.

One follow-up recommendation: preregister a matched-capacity negative-control replication (same added head and parameter count, with a non-goal auxiliary target) before interpreting any favorable difference as goal-specific. This is outside the authorized pilot and was not run.

Stopped after Phase5H. No automatic continuation, Phase5I, full training, hidden/final assessment or Phase6.

Measured interpretation: H1 finished at 7/16 expanded successes versus H0 9/16, a difference of -2 cases and difference-in-differences -2. Expanded first alignment was 7/16 versus 6/16, below the required +3 advantage. H1 direction successes were {'Up': 0, 'Down': 0, 'Left': 4, 'Right': 3} versus H0 {'Up': 2, 'Down': 3, 'Left': 2, 'Right': 2}; Up and Down were lost despite the training-only auxiliary accuracy of 62.67% (14884/23750 repeated replay draws). This is not evidence of robust improved navigation.

Wait fraction fell from 30.58% to 6.47%, while timeouts rose from 10/20 to 13/20 and ABAB loop windows from 859 to 2111. Low Wait use therefore did not remove failure: moving revisits were 1289 versus 2404. The auxiliary arm is not promising under the frozen criteria and should not replace the control based on this result. All eight focused tests passed; the full legacy suite remains deferred.
