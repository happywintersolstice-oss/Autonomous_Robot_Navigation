# BMDS2114 assignment and rubric compliance audit

Reviewed on 23 November 2026 against the actual four-page BMDS2114 assignment PDF and the separate assessment-rubric DOCX in the user's Downloads. This is a technical evidence audit, not a student-authored final report, a mark prediction, or a certificate that every assignment requirement is met.

**Conclusion: substantial technical requirements have evidence, but full assignment compliance is not established.** The current app provides four genuinely learned algorithms, a defined partial-observation environment, bounded optimization, automated comparisons, one separate product test, multiple metrics and inspectable failures. The current model-combination implementation and its development-only measurement are now available. Team/proposal verification, the formal report/submission package and AI-use compliance remain unresolved. See [the current completion status](rubric-completion-status.md).

## Sources and scope

- `C:/Users/User/Downloads/Assignment specification (v1).pdf`: all four pages extracted and visually checked. Page 1 defines the problem/checklist and Part A; page 2 continues Part A and starts Part B; page 3 continues Part B and gives AI/assessment rules; page 4 defines final-report/submission requirements.
- `C:/Users/User/Downloads/Assessment rubric (v1).docx`: the complete OOXML table content was reviewed, preserving criterion names, weights and individual/team labels. Source document layout was not rendered or changed.
- [Source-file provenance](assignment-source-provenance.json) records SHA256 values for these exact source documents. They remain unmodified in Downloads and are not republished as repository artifacts.
- Implementation/evidence: [Phase3 results](history/reports/PRODUCT_PHASE3_RESULTS.md), [Phase2 results](history/reports/PRODUCT_PHASE2_RESULTS.md), active source, the locked Phase3 selection, all eight training receipts, saved development/final comparison metadata, and [the checked demo script](team-demo-script.md).
- The historical [project plan](history/planning/project-plan.md) and [earlier audit](history/planning/audit-remediation-plan.md) provide context only. The old plan calls itself a private project and defers features that the current system later implemented. It is not verified tutor-approved scope and does not override the institutional brief.

The original cleanup/audit performed no training, parameter search or frozen/academic final test. The subsequent registered rubric-support study adds development-only inference; it also performs zero training or frozen-test re-execution. Existing results remain historical evidence of their recorded source revisions. No estimated grades are assigned: individual programming, authorship and live explanations cannot be inferred from a repository alone.

Status meanings: **Evidence present** means inspectable technical evidence exists; **Partial** means limits or missing components remain; **Gap** means no current implementation/evidence was found in scope; **Unverified** requires team, tutor or submission evidence.

## Assignment checklist and Part A

| Requirement | Status | Evidence and limit |
| --- | --- | --- |
| Define a practical problem and approach | Partial | Warehouse delivery navigation is a practical motivation, with explicitly simulated movement/obstacles. The team still needs its own critical comparison of existing technologies and a defensible research question. Simulation is not evidence of deployment on a physical robot. |
| Define the environment and its dynamics | Evidence present | 10x10, five actions, cardinal range-two sensing, local memory, official rewards, collision/goal endings and 200-action cap. Static walls versus temporary initial blockages are explicit; this is not unrestricted moving-traffic simulation. |
| Prepare state representations | Evidence present | Strict 409-value public observations; F2 has 480 features; F3 has a 431-value internal state and 520 derived features. Unknown occupancy is explicitly encoded; there is no need to invent supervised-data imputation or classification labels for this RL task. |
| A1: train diverse suitable methodologies | Evidence present | Q-learning, SARSA, DQN and PPO have actual independent training receipts, weight changes and native checkpoint files. The observed-map replanner is separately non-RL. |
| A2: validation and relevant performance comparison | Evidence present, limited | Development-only selection followed by one frozen64-case product assessment. Success, collision, timeout, official return, successful-only steps/efficiency, Waits, revisits, loops and recovery are measured on matched cases. Cross-validation/precision-recall are examples in the brief, not mandatory substitutes for task-appropriate RL evaluation. Only one training seed/product panel is supported; no statistical superiority is established. |
| A3: patterns and influencing factors | Evidence present, limited | Condition/distance/topology breakdowns, vote disagreement, descriptive family-cluster intervals and1,024 public-history sensitivity probes now exist. Empty-history probes may be out of distribution and do not establish causal feature importance or performance effects. One training seed remains a limitation. |
| A4: errors, inefficiency and generalisation | Evidence present, limited | Recorded long-route timeouts, DQN over-waiting, Q-learning movement loops, last decisions and accessible failure replays. Generalisation is bounded to the new product generators, not arbitrary real warehouses. |
| A5: refine representations/learning configurations | Evidence present | Versioned public history, curriculum stages, linear TD stability, actual DQN replay/targets and PPO rollout optimization, plus optional public-only train feedback. Old checkpoints retain their original signatures. |
| A6: incorporate measured refinements and validation | Evidence present | Fixed unshaped/liveness candidates were compared on development with all attempts retained. Historical phases remain separate. No post-test tuning or relabeling of reused outcomes as unseen. |
| A7: shortlist promising models | Evidence present | A locked checkpoint ID/hash for each family selected by registered development criteria. This keeps one candidate per family for the four-algorithm comparison; it does not prove a universally best model. |
| Automate repetitive evaluation/selection | Evidence present | Configuration/manifest-driven bounded study, ranking, actual counters, immutable CSV/JSON/replays and ZIP inventories. The app can explicitly compare development choices; saved held-out viewing is passive. |

## Part B

| Requirement | Status | Evidence and limit |
| --- | --- | --- |
| B1: adjust learning settings and optimize from performance | Evidence present, limited | Two predetermined 40k-action candidates per family, explicit settings/objectives, real optimizer/TD counters and development scores. Q-learning/SARSA/DQN selected liveness; PPO selected unshaped. This is a bounded comparison, not a broad independently replicated hyperparameter search. The team must explain actual settings and outcomes rather than say only that defaults were tuned. |
| B2: combine models to improve reliability | Implemented and measured on development | A fixed equal-vote ensemble of the four current checkpoints completed36/36 previously inspected development/demo cases, versus Q-learning34/36 and the other three native models36/36. No new training, fitted vote weights, planner or held-out evaluation. Its decision cost and route length were higher than the strongest individual policies; there is no superiority or unseen claim. [Actual evidence](rubric-completion-status.md). |
| B3: test new situations and changing information | Evidence present, limited | The 64 new cases contain 28 static/36 dynamic and 32 short/32 long, with forced Wait and optional detours. Distinct structural generators and overlap checks are saved. Blockers start occupied and have finite clearance; there is no claim of stochastic pedestrian traffic, new obstacle arrivals or physical-world robustness. |
| B4: automate and improve efficiency/scalability | Evidence present, limited | Automated runs, selection, exports, bounded background workers and passive replay avoid repeated work. CPU jobs are bounded to 50k/model and one hour. Ten-by-ten simulation and bounded CPU evidence are not a large-scale deployment or big-data benchmark. |
| Proper comments/remarks explaining outcomes | Improved technical support | Standard readable formatting, focused page modules, named registry helpers, learner/trainer docstrings and [code walkthrough](code-walkthrough.md). Each student still has to explain and modify their own assigned code in person. |

The combination heading in the specification is explicit. The rubric's optimization examples also mention evaluating ensembles, but the rubric's B2 criterion is named **System implementation**, not **Ensemble**. Keep these two sources distinct; do not claim a guaranteed mark deduction or exemption without the tutor's interpretation.

## Rubric mapping

Part A 40% plus Part B 40% plus Output/Presentation 20% is separate from the specification's 40% team/60% individual allocation.

| Criterion | Marks / assessment | Evidence | Remaining condition |
| --- | --- | --- | --- |
| A Problem statement |10 / team | Clear simulated task, information limits and warehouse framing | Team-authored critical evaluation/related systems, originality and agreed project scope are unverified. |
| A Programming |20 / individual | Four real learners, automation, error handling, regression checks; readable code | Named individual ownership and ability to explain, justify and modify code must be demonstrated. |
| A Degree of completion |10 / team | Five app pages, training, simulation, comparison, exports, visible failures | Current ensemble measurement is now available; final demo reliability and actual human explanations still require rehearsal. An RL timeout is an honestly measured outcome, not automatically a software bug. |
| B Model optimization |10 / individual | Actual candidates/settings, before/after development evidence, successful-only and all-action metrics | Each student must defend an authentic optimization contribution. Current ensemble development evidence exists; no multi-seed causal/component study is claimed. |
| B System implementation |15 / team | Current implemented partial dynamic warehouse system and documented contracts | Compare against the actual submitted/tutor-accepted proposal and record agreed scope changes. That approval document was not verified. |
| B Results / performance measurement |10 / individual | Matched-panel results, denominators, strata, failures and downloadable raw evidence | Student-owned interpretation must be written and defended; one panel/seed does not justify superiority claims. |
| B Organization |5 / team | Separate code guide, evidence report, script and compliance audit | A coherent final academic paper, correct format and references were not verified. Repository research notes are not the final paper. |
| Output |10 / individual | Maps, memory/truth separation, charts, status, CSV/JSON/ZIP and recorded failures | Actual visual polish/accessibility and usefulness should be reviewed during human rehearsal. Automated AppTest is not a manual visual assessment. |
| Presentation |10 / individual | Checked four-role trajectory script and code explanations | Attendance, individual competence, concise delivery and Q&A cannot be certified by code tests. The short demo script does not establish the tutor's required presentation duration. |

## Final-report and submission checklist

| Requirement from the source | Status |
| --- | --- |
| Three-to-four-person team with unique title | Unverified: role labels exist, but a verified member/ID list and originality approval were not found. |
| Proposed title and brief problem/solution description | Technical material exists; the team must own and finalize its academic argument. |
| Explain how the solution works without omitting important details | [Technical walkthrough](code-walkthrough.md) and versioned source provide evidence; final report integration is unverified. |
| Explain evaluation outcomes/results in clear English | Actual tables and failures exist in the product reports; student-authored interpretation and final compilation remain unverified. |
| Task allocation and project plan with individual work | Partial: a four-role demo script exists; named actual contributions, code/tests/results ownership and individual work evidence were not verified. A planned allocation is not proof of completed contributions. |
| One group-compiled final report | No finished submission document verified in the reviewed workspaces. |
| Times New Roman 11pt, single spacing, cover page | Not verified; Markdown technical reports do not meet this document-format requirement by themselves. |
| Around 40 pages and below 50 | Not verified; no final report pagination was checked. |
| One completed assessment rubric as a separate file | Source rubric is separate but its title/group/name/ID fields are blank. Do not merge it into the final report. |
| Submit final report and source to Google Classroom | Unverified and not submitted by this task. |
| Final report Week 6; presentation/Q&A from Week 7 practical classes | Stated in brief; actual class calendar/date and submission status require team/tutor confirmation. No deadline date was invented. |
| Originality, attribution and plagiarism rules | Unverified; cannot be inferred from executable results. |

## AI-use compliance is unresolved

Page 3 classifies the assignment as **Yellow / Limited AI**. It permits assistance such as clarification, syntax/technical support and language refinement, while restricting generated core arguments/analysis/original academic content and complete solutions/reports. It requires the prescribed AI Usage Disclosure Form and each student's ability to explain, justify, modify and demonstrate submitted work.

This conversation/repository includes substantial AI-assisted implementation and generated technical reports. It would be inaccurate to certify that merely adding a disclosure now makes all prior work compliant. The team should show the actual assistance history to the tutor, establish what work may be submitted or must be independently redone, complete the prescribed disclosure accurately and retain real contribution evidence. Do not relabel AI-generated content as student-authored analysis or fabricate individual contributions. This audit does not replace that review and is not a ready-to-submit academic paper.

## What can be concluded now

The implementation provides strong inspectable technical support for diverse modelling, environment/state preparation, automation, bounded optimization, evaluation and error analysis. Full assignment compliance remains **not established**. The previously missing current-system model combination is now implemented and development-evaluated. The remaining submission gaps concern the final paper/package, verified approved scope, individual work and presentation, and actual AI-use permission/disclosure. The team should resolve these before making an all-requirements-met claim.
