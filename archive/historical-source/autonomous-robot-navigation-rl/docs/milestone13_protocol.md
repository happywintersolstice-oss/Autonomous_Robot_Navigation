# Milestone 13. Saved-trajectory failure diagnosis

Registered 11 November 2026 before the full diagnostic calculation. The default
next milestone follows the completed M12 evaluation and the original objective
to explain navigation failures. It analyzes saved evidence and adds replay
navigation. Initial bounded inspection found two-state cycles in one development
timeout per learned family. This is a post hoc descriptive study.

## Sources and scope

Use all twelve M11 main-policy validation recordings and all twelve M12 final
recordings. Support is 2,304 development and 5,760 final policy–route episodes.
Keep those cohorts separate. All four algorithms have three training seeds.
References retain their existing result tables and fall outside this learned-
policy diagnosis. Record file hashes, verified source identities and the M12
seal. Preserve existing input files and frozen outcomes.

Replay stored transitions through the core environment and reconcile every
per-run outcome/interaction aggregate with its verified source. The calculation
uses saved actions, observations and diagnostics, without querying a model.
Known-map shortest distances provide retrospective geometric annotations.

## Diagnostic definitions

An optimal success has recorded length equal to the saved shortest route.
A detour success is longer. A collision is a boundary attempt when the attempted
cell lies outside the grid, otherwise a wall attempt. Collision terminal states
are excluded from cycle classification.

For timeouts, find the first repeated full robot/goal observation. Its earlier
index is the cycle start, and their index difference is the candidate period.
A persistent cycle requires at least two full repetitions and exact repetition
of the entire remaining state/action suffix. Classify period one as stationary,
period two as a two-cell cycle and larger periods as longer cycles. Other
timeouts remain non-periodic under this definition. For learned frozen policies,
require the saved action for each repeated full observation to agree throughout
the episode. These are trace descriptions and do not identify training causes.

Report cycle start, first return, period, distinct visited cells, first closest
approach, minimum/final shortest goal distance and actions after the first
return. The latter counts transitions after the first closed traversal and is
not an estimate of a tested recovery improvement. Full observations include the
fixed goal, and the map remains fixed within an episode.

Provide exclusive diagnostic counts, per-seed means/sample SD, distance-band
success/failure rates and per-layout counts. Seed means have equal route support.
Routes within one policy/layout are dependent. One failure-location observation
is counted per failed episode. Use the cycle entry cell for persistent cycles,
the last valid robot cell for collisions and the final cell for other timeouts.
Location counts describe that rule and do not estimate obstacle risk exposure.

## UI, verification and acceptance

Add a Failure analysis tab within the existing development/final study entries.
Show diagnostic composition, route-difficulty tables and a selectable failure-
location map. Use existing registered replay examples, with one bounded example
per run/diagnostic category, taking the first available source example in its
declared order. Examples illustrate individual cases. Link to their first
cycle return or terminal collision in the existing centered replay player.
Preserve playback controls and pause on view/study changes.

Store source hashes, per-episode/per-seed/group records, bounded example
references, a standalone comparison figure and a measured report. Check exact
source support, duplicate identities, count/interaction reconciliation, cycle
definitions, geometric annotations and example membership/checksums. Verify
saved outputs by recomputing from the fixed inputs. Use synthetic cycle, detour,
wall/boundary and non-periodic cases, plus normal dashboard entry-point tests.
Finish with native desktop/narrow checks and pre-existing-file preservation.

The study has a 900-second calculation limit and a 64 MiB derived-artifact limit.
Verification is separate. Completed compatible results are reused. Inputs and
complete outputs are fingerprinted. Incomplete publication is retained in its
own pending directory and is never accepted as a completed analysis.

Performance changes require a separately registered training study and a fresh
independent holdout. M13 reports failure patterns in already inspected data.
