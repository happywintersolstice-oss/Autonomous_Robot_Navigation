# Milestone 8 status: controlled DQN inputs

The owner selected this next research milestone on 8 November 2026. The six core
milestones, PPO extension and saved-evidence/dashboard verification are complete.

The new input encoder and inference adapter support four normalized absolute
coordinates and those same coordinates plus two relative goal offsets. Old
checkpoints retain their existing encoding and task checks. The canonical raw
observation and environment are unchanged. The separate registered protocol is
[representation_protocol.md](representation_protocol.md).

Bounded trace capture records the first three chronological training episodes
when requested by the new configuration, including actual exploratory actions,
states, rewards and endings. Incomplete prefixes are explicitly labeled and have
no fabricated terminal metrics. Historical training traces remain unavailable.

Mechanics tests passed for transition parity, timeout bootstrapping with six
features, saved-model compatibility and trace capture that leaves seeded neural
weights/replay and tabular actions/updates unchanged. All 108 tests passed before
the independent pilots and registered main matrix.

Both 10k pilots completed and passed chronological trace/ledger checks (three
episodes and 11 actual actions per pilot). Reloaded checkpoints report the
correct feature encodings, with 4,740 versus 4,868 online parameters. The measured
main estimate was 1,831.51 seconds, below the predeclared 2,400-second cap. Pilot
validation success was 24/928 control and 18/928 relative; these are software/
throughput pilots, not part of the main comparison. No settings changed.

The main study completed all ten runs: five fresh paired seeds
[101,211,307,401,503], two encodings, 200,000 interactions each. Matrix wall time
was 1,938.53 seconds (32.31 minutes), below the 2,400-second cap; new artifacts
occupied 53.36 MiB at completion, below 1 GiB. Including pilots, the study added
2,020,000 training interactions. No hyperparameter search or final-test execution
was performed.

Mean validation success was 5.97% control versus 6.51% relative. The paired change
was +0.54 percentage points, sample paired-seed SD 2.88 pp. Two seeds improved,
two worsened and one tied. Relative inputs reduced average collisions but
increased timeouts. Both systems remain weak; the study does not establish a
consistent improvement or justify replacing the selected controller.

All 110 tests passed in 39.40 seconds after publication. The audit replayed
1,187,423 validation actions, loaded all 200 periodic checkpoints, and reconciled
78 captured main training actions with chronological ledgers. All 7,746 protected
historical artifact/data files remained unchanged, and all 184 archived frozen
inputs still match their seal. Both scientific figures were visually inspected.
Native dashboard verification is the remaining delivery check.

See [measured results, all seeds and limitations](milestone8_results.md) and
[the audit record](../artifacts/studies/representation-v1/verification.json).
