# Audit remediation implementation record

Date: 17 November 2026. Baseline: `d1ed065a6d76240226f1c671664978ca623ab188`. The [audit plan](../../docs/archive/audit-remediation-plan.md) is the specification; this record distinguishes implemented software from acceptance work requiring the group or another machine.

## Changes delivered

| Priority | Implementation | Verification / remaining acceptance |
|---|---|---|
| P0.1 Entry point | Root landing page explicitly enters the application directory before setup; links current evidence, report and reproduction. Inner README preserves historical commands with a current-navigation notice | Both README link checks run locally. Independent second-machine/cold-install acceptance remains external |
| P0.2 Evidence boundaries | Methods reads no optional studies; Unseen layouts does not load the fixed task or validation. Optional studies have panel-level recovery notices; candidate unseen studies are validated before discovery. Expected missing/parse/schema/signature faults have explicit unavailable/unverified states; unexpected exceptions remain visible | Regression fixtures cover absent, malformed, unverified, wrong-signature and changed-value evidence, plus unseen results/replay with fixed validation absent |
| P0.2 Fixed final integrity | `docs/evidence/fixed-final-v1.json` separately pins the unchanged M6 summary to the known freeze digest. Published and audited summaries matched; per-run episode aggregates and family means/sample SD were recomputed before pinning | Arbitrary completed bodies and a changed metric under the same task signature are rejected. Original snapshot/seal files were not rewritten |
| P0.3 Algorithm joins | Four-family combined columns join by canonical family IDs, checking cohort, label, seed support and missing/extra IDs; incompatible detail evidence leaves baseline-only rows | Synthetic different values for all four families remain attached correctly when detail order is reversed |
| P1.1 Evidence/report alignment | Current report opening distinguishes the completed fixed and unseen cohorts. Methods explains both contracts and inspected final status. Evidence index maps requirements, implementation, cohorts, suggested owners and demo actions. Original plan/review guide are marked historical | Real member names, accepted/tutor-agreed proposal, related-system academic discussion and individual ownership remain group verification tasks |
| P1.2 Navigation | One registry owns Demo / Results / Methods; Results groups Fixed layout and Unseen layouts. All existing expert views remain reachable. Replay clock and stale-event contracts retained; new history-recovery replay has its own paused clock | Automated navigation and playback checks pass. Fresh native desktop/narrow visual acceptance is blocked as described below |
| P2.1 Asset inventory | `demo_assets` builds/verifies exact references, hashes, sizes, task/feature versions, cohorts and seeds. Sidebar exposes Check saved demo assets. Original source/snapshot trees remain in place | Current bundle: 83,561,697 bytes (about 79.69 MiB), with 16 shared content hashes between the historical release input sets. Missing/changed dependencies report their exact paths. This is a reference inventory, not an archive migration |
| Controlled follow-up | Registered and executed bounded history-aware SARSA recovery on four development layouts, three pretrained policies and three recovery RNG streams each | [Measured result](loop_intervention_results.md): +4.40 pp success, improved SPL, 1.04% collision; misses +5 pp promotion requirement. Original selected controller unchanged |

## Executed checks

- New defect fixtures were introduced before the corresponding fixes. The arbitrary-summary acceptance failure was reproduced on the old loader.
- Targeted audit/dashboard run: **41 passed in 25.52 seconds**.
- Full suite after the main fixes and intervention: **218 passed in 113.19 seconds**. The final expanded suite passed **221 tests in 123.55 seconds** after the asset and recovery UI additions.
- Focused audit/asset/recovery/tab run: **22 passed in 14.34 seconds**.
- Dependency consistency: no broken requirements. Gymnasium checker: passed.
- `developer/commands/replay_frozen_release.py --workspace .cache/remediation-frozen-20261117`: restored and verified the exact 184-input original archive; preparation only, no final policy execution.
- Existing unseen final seal retains 114 inputs. Diagnostic source checks retain their historical contracts and cohort records.
- New recovery audit checked **246,867 actions**, including repeated verification of reused deterministic baselines; 576 unique baseline and 1,728 wrapper cases. It added zero training and zero final-layout evaluations. Execution took **21.50 seconds** against a 180-second cap; initial study footprint **4,595,068 bytes** against 256 MiB.
- Demo manifest build and verification passed. It inventories the saved rehearsal and recovery results, not every historical training checkpoint. Frozen source copies and outputs were not moved or normalized.

Use the tested Windows/Python 3.14.10 environment and locked versions. Current dependency checks used the existing installation; no independent clean installation is claimed. Run `python -m robot_navigation.demo_assets verify` before rehearsal. Rebuilding the demo inventory after intentional active-source changes is separate from modifying any frozen experiment.

## Native UI verification limitation

The native browser inventory identified the local Navigation Control Room. Its initial accessible state still showed the old server imports. The server was restarted with the new implementation. Computer Use then stopped the reload action because it could not determine Chrome's current URL confidently enough to enforce its policy. No further native browser input was attempted. Desktop/narrow screenshots and a timed human rehearsal are therefore **not claimed as passed**. Automated Streamlit interaction tests exercise the new routes and recorded playback.

## Work deliberately outside the measured claim

The controlled recovery study stops at development validation, as permitted by the plan. The M6 pairs and M12 layouts remain inspected historical evidence; no fresh holdout or confirmatory improvement is claimed. A raw-input versus shared-feature PPO experiment is conditional on making a representation-only causal claim, which the current report does not make.

The four-member allocation is a proposed ownership map. A second machine, accepted proposal/tutor confirmation, actual named contributions, disclosure form, final report formatting/submission and presentation/Q&A require the group. No external messages, submissions or publication were performed.

Final checks: both README link sets resolved (6 root links, 82 inner links), new index/result/status links resolved, Git diff whitespace checks passed, and no pre-existing tracked data/artifact paths appear in the change list. The final reference manifest contains **1,006 files / 83,561,697 bytes**.

## Subsequent map and reward revision

The later structured map/reward lab adds four measured layouts and 36 new matched development runs. See [results](navigation_lab_results.md). Its full regression run passed 229 tests; the final focused map/UI run passed 22. The current demo-v2 manifest contains 1256 files / 102216715 bytes. Earlier counts above describe the remediation baseline. No old frozen map, reward configuration or model was repurposed for the new layouts.

## Unified Demo interface

The original simulator and structured-map lab now share **Demo > Navigation map**, with all five maps and one set of Watch the robot / Learning replay / Compare robots / Try a route tabs. The standalone lab route and back/explore buttons are removed; sessions on the former route migrate into Demo. Geometry-specific models remain distinct, and synchronized reward comparisons explicitly allow reward-only differences while checking identical dynamics, scenario, replay seed and genuine signatures. Structured custom routes use the selected reward environment; no training runs during demonstration. Where training actions were not retained, the Learning replay tab clearly displays only saved monitoring curves.

The merged implementation passed **231 tests in 103.43 seconds** (plus 26 targeted checks). Historical release seals and M13 inputs/outputs verify. Current demo inventory v3 contains 1259 files / 102225870 bytes. [Verification record](evidence/unified-demo-verification-v1.json).
