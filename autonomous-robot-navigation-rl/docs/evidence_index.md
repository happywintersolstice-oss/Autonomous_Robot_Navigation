# Current evidence and demonstration index

For the current interface and file locations, see the [user guide](../../docs/user-guide.md) and [project organization](../../docs/project-structure.md).

Evidence cut: 17 November 2026. This is the group-facing technical index. The original project plan and dated milestone reports describe their historical scope; they do not supersede the current benchmark boundaries below. Member names and accepted-proposal conformance remain for the group to confirm.

## Five-minute saved-data rehearsal

1. Launch from the [repository landing page](../../README.md). In **Demo → Watch the robot**, press Play on the selected Q-learning success recording; inspect reward, goal and terminal SPL.
2. Open **Compare robots → Two agents**. Keep the matched seed-101 control/refined pair, step once, then play. Finished agents hold their terminal frames.
3. Open **Results → Fixed layout → Compare results**. Read the separate development-validation and inspected M6 final tables; identify their denominators and predeclared model choice.
4. Open **Results → Unseen layouts → Results**. Keep the frozen M12 final cohort, then inspect Failure analysis and select Replay this event. Its caption identifies the cohort, seed, layout and event action.
5. Open **Guide**. Explain coordinate-only versus map-aware observations, the BFS reference, SPL, and why an inspected final cohort cannot be reused as a fresh holdout.

No rehearsal step starts training. Expert learning recordings, custom routes, neural tuning and robustness remain available one level below the three primary routes.

The shared **Demo → Navigation map** selector includes four structured layouts and separately retrained Q-learning models. See [map/reward protocol](navigation_lab_protocol.md) and [measured results](navigation_lab_results.md). It is a familiar-map development comparison with reserved pairs unscored, separate from both historical final cohorts. The current [demo asset inventory v3](evidence/demo-v3.json) covers the unified simulator. Earlier [v2](evidence/demo-v2.json) and [v1](evidence/demo-v1.json) inventories remain historical records.

## Requirement-to-evidence crosswalk

Owner numbers below refer to the proposed responsibilities in the [audit plan](../../docs/archive/audit-remediation-plan.md), not verified individual authorship.

| Requirement / question | Implementation | Cohort and measured evidence | Proposed owner; demo |
|---|---|---|---|
| Define navigation and reference methods | `core.py`, `benchmark.py`, `agents.py` | [Environment](environment_specification.md); original fixed layout, coordinate-only learned observations; BFS receives map | Member 1; Demo success/collision/timeout |
| Compare diverse learners and select shortlist | `part_a.py`, `dqn.py`, later `ppo.py` | [M3](milestone3_results.md): Q-learning 65.24%, SARSA 21.94%, DQN 5.32%; five seeds, 928 validation routes each. Q-learning/SARSA shortlisted. [PPO](ppo_comparison_results.md) is later development evidence | Members 1/3; Fixed layout comparison |
| Optimize under matched controls | `refinement.py`, training scripts | [M4](milestone4_results.md): refined Q-learning 89.96% versus 64.74% matched validation control, five paired seeds; reward-only ablation retained | Member 2; control/refined replay |
| Combine models and interpret tradeoff | `voting.py` | [M4 vote](milestone4_results.md): five fewer validation goals than seed-101 single, higher SPL; one ensemble, no ensemble-seed uncertainty | Member 2; vote recording and table |
| Evaluate frozen fixed-map selection | `release.py`, separate evidence pin | [M6](milestone6_results.md): seed 101 reaches 840/928 final goals, SPL .867870. Family mean 90.43%; final pairs now inspected | Members 1/3; fixed final table |
| Investigate representation and neural tuning | `representation.py`, `neural_tuning.py`, `neural_confirmation.py` | [M8](milestone8_results.md): +0.54 pp inconsistent paired effect; [five-seed tuning](neural_confirmation_results.md): mixed effects and missed targets | Members 2/3; Results → Fixed layout → Neural tuning |
| Evaluate multiple layouts | `unseen_benchmark.py`, shared feature/policy modules, `unseen_final.py` | [M11](milestone11_results.md), [M12](milestone12_results.md): 12/4/10 layouts; three seeds per learned family, 480 final routes per policy. Final success 16.81%, 62.08%, 29.10%, 64.10%; BFS 100% | Member 3; Unseen layouts final results |
| Diagnose failures and test one intervention | `unseen_diagnostics.py`, `loop_intervention.py` | [M13](milestone13_results.md): 96.59% of 4,434 timeouts are two-cell cycles. [Registered development intervention](loop_intervention_protocol.md) preserves the deterministic historical diagnosis | Member 2; failure event and paired development results |
| Present and automate evidence reliably | navigation registry, guarded readers, `demo_assets.py`, regression tests | [Remediation status](remediation_status.md); [demo manifest](evidence/demo-v1.json); three primary routes and explicit cohort labels | Member 4; missing-asset test and five-minute rehearsal |

## Cohorts and commands

Activate the documented virtual environment or use `.\.venv\Scripts\python.exe` from the application directory. Run metadata-writing historical verifiers in a disposable checkout or scratch workspace.

| Evidence | Protocol / results | Reproduction entry point |
|---|---|---|
| Original selection | [M3](milestone3_results.md) | `developer/commands/run_part_a.py`, `developer/commands/analyze_part_a.py` |
| Search, controls, reward and vote | [M4](milestone4_results.md) | `developer/commands/run_part_b.py`, `developer/commands/analyze_part_b.py` |
| Frozen fixed-map release | [M6 protocol](milestone6_protocol.md), [results](milestone6_results.md) | `developer/commands/replay_frozen_release.py --workspace .runtime/new-frozen-check` (preparation/hash check; fresh destination) |
| Later four-family comparison | [PPO results](ppo_comparison_results.md), [detail analysis](four_way_analysis_results.md) | `developer/commands/run_ppo_comparison.py`, `developer/commands/analyze_four_way_details.py` |
| Input/neural refinements | [M8](milestone8_results.md), [M9](milestone9_results.md), [confirmation](neural_confirmation_results.md) | `developer/commands/run_representation_study.py`, `developer/commands/run_neural_tuning.py`, `developer/commands/extend_neural_confirmation.py` |
| Shared-feature development | [M11 protocol](milestone11_protocol.md), [results](milestone11_results.md) | `python -m robot_navigation.unseen_main verify` |
| Frozen unseen-layout evaluation | [M12 protocol](milestone12_protocol.md), [results](milestone12_results.md) | `python -m robot_navigation.unseen_final verify` |
| Post hoc diagnosis | [M13 protocol](milestone13_protocol.md), [results](milestone13_results.md) | `python -m robot_navigation.unseen_diagnostics verify` |
| New development-only wrapper | [Protocol](loop_intervention_protocol.md) | `python -m robot_navigation.loop_intervention run` then `verify` |
| Saved-demo assets | [Manifest](evidence/demo-v1.json) | `python -m robot_navigation.demo_assets verify` |

## Claims the group must still defend

M10 to M11 changed representation, value-function implementation, budget and seeds. That comparison is descriptive. A causal feature-only claim needs the separately proposed paired raw-input/control experiment; no such claim is made here. The existing M8 ablation already documents a controlled representation refinement.

The fully known static simulator supplies complete observations for each task version. Missing sensors, continuous motion, physical hardware and uncertain observations are not implemented requirements unless the accepted proposal separately demands them. BFS solves known grids optimally; RL here studies learning and generalization. Related-system literature, originality claims and each member's assessed academic interpretation need the group's own cited discussion and review.

The ten M12 layouts and M6 final pairs are inspected evidence. The new wrapper uses development layouts only. A confirmatory result requires a new versioned, duplicate-checked holdout after controller/configuration freeze.

Confirm member names, accepted proposal, tutor-approved deviations, institutional AI disclosure, independent second-machine installation and the final presentation with the group. Technical changes cannot verify those facts.
