# BMDS2114 robot navigation audit and remediation plan

**Audit date** 17 November 2026
**Repository snapshot** `main`, [`d1ed065a6d76240226f1c671664978ca623ab188`][commit]
**Course references** *BMDS2114 Machine Learning Assignment, Assignment specification (v1)*, pp. 1–4, and *Assessment rubric (v1)*, Parts A and B and Output and Presentation. The [group proposal conversation][proposal] provides scope context. Confirm the submitted or tutor-agreed proposal with the group before signing off design conformance.

**First priority.** Repair the repository entry point and make the dashboard resilient to incomplete evidence before adding experiments. Most of the assignment's modelling work already has measured results. The substantive open experiment is whether a targeted change reduces recurring unseen-layout failures while preserving navigation quality. The repository also needs a clear path from the group proposal to the evidence each member can defend.

## 1. What the prototype already delivers

| Evidence cohort | Work already completed | Interpretation for the assignment |
| --- | --- | --- |
| M3 fixed-layout Part A | Q-learning, SARSA and vanilla SB3 DQN, five independent training seeds per family and 200,000 interactions per run. Full validation uses 928 fixed-layout start–goal pairs. Random and map-aware BFS are references. Q-learning 65.24%, SARSA 21.94%, DQN 5.32% mean success. | Original validation-only shortlist is Q-learning and SARSA. M7 adds PPO as a later four-family development comparison. Preserve that chronology. [M3 results][m3], [PPO comparison][ppo]. |
| M4–M6 fixed-layout Part B | Bounded search, five paired confirmation seeds per shortlisted family, reward-only ablation, one frozen three-member Q-learning vote, and stratified robustness. On M4 validation, tuned Q-learning reached 89.96% versus 64.74% matched control. On M6 final, the corresponding family means were 90.43% versus 65.52%. | The declared single controller is the seed-101 policy, 840/928 final goals and SPL .867870. The M4 vote lost five validation goals against that single but raised SPL. M6 vote reached 848/928 without a final-data promotion. The 928 final pairs are already inspected. [M4 results][m4], [M6 results][m6]. |
| M8–M9 and neural follow-up | Paired DQN coordinate/offset ablation and bounded DQN/PPO search with five-seed same-settings confirmations. DQN representation change averaged +0.54 percentage points on development validation. Five-seed neural tuning averaged 6.88% to 7.37% for DQN and 19.44% to 23.69% for PPO. | Keep the mixed paired effects and missed targets visible. The neural tables answer within-family questions under their own search budgets. [M8 results][m8], [neural confirmations][neural]. |
| M10–M12 unseen layouts | Separate suite with 12 training, four development and ten final layouts. M12 evaluated 480 final routes per frozen policy, three trained seeds per algorithm. Mean success was 16.81% Q-learning, 62.08% SARSA, 29.10% DQN and 64.10% PPO. BFS reached 100% with access to the layout. | The four learned families missed the declared 80% success and .75 SPL targets. The benchmark itself exists and was independently evaluated before post hoc diagnosis. Treat the ten final layouts as inspected evidence for future experiments. [M11 protocol][m11p], [M12 results][m12], [final targets][finalconfig]. |
| M13 failure analysis | Saved-trajectory replay and diagnosis over 8,064 learned-policy cases and 908,298 actions. There were 4,434 timeouts, of which 4,283 (96.59%) were two-cell cycles. | Part A error evidence and motivation for a Part B intervention. The counts describe deterministic behavior in these recorded cohorts. The training cause and effect of a fix remain open questions. [M13 definitions][m13p], [M13 results][m13]. |

Use separate table labels for familiar-layout validation, familiar-layout frozen test, unseen-layout development, unseen-layout frozen test and post hoc analysis. Label each plot with task/observation version, split, checkpoint selection rule, training seeds, scenario support and aggregation unit. SPL is success-weighted shortest-path efficiency, with zero for failures. Report successful path length alongside the number of successes. BFS sees the map in the fixed-layout benchmark and is the planner reference. Be explicit about when a learned agent receives layout data.

## 2. Ranked defects and acceptance tests

### P0.1 Fix the root README and the first-run path

**Confirmed.** The repository root contains a copied project README. Its links address root-level `docs/` and `data/`, and its setup assumes `requirements-lock.txt`, `scripts/` and `app/` are in the current directory. Those targets live inside `autonomous-robot-navigation-rl/`. A filesystem check found 65 unresolved relative-link occurrences (59 unique targets) in the root README and zero in the inner README under the same check. The root also stops its opening summary at older studies while the inner README describes M9–M13. [Root README][rootreadme], [actual project README][innerreadme].

**Change.** Replace the root copy with a short landing page. Put the explicit `cd autonomous-robot-navigation-rl` before setup, link to the inner README and evidence index, and show the one command that launches the saved demo. Keep detailed installation, troubleshooting and reproduction in the inner README. Document tested Windows/Python and dependency versions from a fresh installation. Keep historical experimental commands accessible from the evidence index.

**Accept when** all relative Markdown links resolve in both READMEs and a teammate on a separate machine follows the root instructions from a fresh checkout to dependency check, environment check and a loaded saved replay. Record OS, Python version, commands, outcomes and whether dependencies were cached.

### P0.2 Isolate unavailable evidence so unrelated pages still work

**Confirmed failure paths.** `dashboard_ui.render_page` handles faults from the representation and detail readers, then loads and checks the four-family summary and fixed final summary without the same protection. A four-family status error can block even Methodology, which does not consume its results. `unseen_ui.render_unseen` parses candidate main/final summaries to populate the dropdown before the guarded load. An invalid candidate summary can abort the entire unseen page before fallback. In `app/streamlit_app.py`, fixed-layout task and validation data load before page dispatch, so even Unseen maps depends on the fixed-layout validation file. [Dashboard dispatch][dashload], [four-family reader][fourload], [unseen discovery and load][unseenload], [app entry point][app].

**Confirmed integrity gap.** The fixed-layout final aggregate loader trusts `final_evaluation_complete` in `status.json` and returns the summary body without verifying it. A temporary fixture with that flag and an arbitrary summary body was accepted by the loader. Its caller checks the task signature, which identifies the task rather than the accuracy of result values. The original final runner audits and freezes inputs, then publishes the summary and status separately. The unseen final reader already validates a summary hash. [Fixed reader][release], [fixed publication][fixedpublish], [unseen final checks][unseenload].

**Change.** Dispatch by page and declare its evidence dependencies. Validate optional summaries before adding their options. Catch expected file, parse, schema and signature errors at the corresponding panel boundary, with a recovery message and the affected study identifier. Separate unavailable, unverified and successful states. Keep established hash/seal checks strict. Pin the fixed-layout final summary hash in a separately versioned evidence manifest tied to the known frozen release after reconciling it with the audited output. Move the fixed-layout load behind the pages that need it and avoid reading optional extended evidence for Methodology. Keep unexpected exceptions visible to maintainers.

**Accept when** Streamlit AppTest cases cover an absent file, malformed JSON, unverified status, mismatched task signature and a changed metric value with the same task signature. Unaffected panels still render. The affected panel shows its own recovery message and displays no unverified metrics. With fixed-layout validation missing and unseen evidence intact, the unseen results and one saved replay still open. The normal complete-artifact path shows the same metrics and saved trajectory frames.

### P0.3 Join four-family metrics by algorithm ID

**Confirmed latent data-label defect.** `dashboard_ui.py` builds baseline rows from one JSON object's values and appends results from another using `zip`. This relies on matching object order and silently drops any unmatched tail. The committed inputs have matching family order. When a fixture reversed only the detail mapping, the Q-learning row showed PPO's 3.7910 mean successful-route actions in place of Q-learning's 6.8446. The same join supplies success count and mean return. [Positional join][dashjoin], [baseline producers][fourload], [detail reader][details].

**Change.** Join the two mappings using the canonical family IDs already present. Validate required family IDs and sufficient metadata before displaying combined columns. Display baseline-only rows with an evidence notice if the extension cannot be joined. Keep the existing study schemas unless a separately versioned schema is needed.

**Accept when** the identical chart values result from reversing just one input object's order. Missing and extra family IDs are reported. A fixture with deliberately different synthetic values for all four families demonstrates that route length, seed support, success count and return remain attached to the correct family.

### P1.1 Align the dashboard, report and proposal with the current evidence

**Confirmed.** The fixed-layout Methodology page says unseen maps remain deferred, while the application has an unseen-layout final-study page. Its text also makes it easy to confuse an unopened test manifest during a replay with final results that already exist and have been inspected. The technical report opens with 158 tests and the unseen evaluation still upcoming, then describes M12 and M13 later. The root development plan frames a solo private project and defers unseen maps. The proposal conversation describes a team project and multiple environments. These contradictions make assessors reconstruct the research sequence from milestone files. No tutor-approved version or individual ownership evidence was verified. [Methodology][dashmethod], [unseen views][unseenload], [report opening][reporttop], [later results][reportend], [historical development plan][privateplan], [group proposal][proposal].

**Change.** Create a current group-facing design-to-evidence table with an implementation location, result cohort, owner and demo action for each agreed requirement. Mark old plans and dated results as historical references. In the report, give each benchmark its own methods and results, use a dated evidence cut with final-layout status, and link to detailed artifacts. Explain validation versus inspected final evidence in the UI. Keep fixed-map and unseen-map claims distinct. Explain why the project uses RL as a learning study when BFS solves the known-map simulator optimally, and tie originality claims to the group's own work and cited systems.

**Accept when** a teammate can locate the original shortlist, tuned vs matched controls, vote trade-off, independent unseen evaluation, and M13 limitations via links from the landing page. Dashboard and report show the same split, denominators and selection history. The team reconciles deviations against its accepted proposal and verifies every report headline against a committed result.

### P1.2 Simplify the demonstration and remove duplicate navigation paths

**Confirmed design friction.** The sidebar declares seven page buttons plus a select box over eight pages, with Optimization and robustness reachable only via the Research library. The evidence pages share a large dispatch function and the live-map, two-agent comparison and unseen replay paths each manage playback state. Discovery and identity information are spread across string keys, filename prefixes, UI constants and manifests. [Sidebar][appnav], [page list][dashpages], [page dispatch][dashload], [run selector][selector], [unseen replay][unseenplay].

**Change.** Provide three visible first-level routes for the assessed workflow: **Demo**, **Results** and **Methods**. Within Results use explicit **Fixed layout**, **Unseen layouts** and cohort labels. Keep saved learning playback, expert comparisons and detailed diagnostics one level deeper. Let a single page list own navigation. Retain the existing `Playback`, `ComparisonPlayback` and event-seek logic. Consolidate state responsibilities only after tests describe the behaviors used by all views.

**Accept when** a new user can launch, play a goal-reaching replay, open a matched comparison, see a held-out aggregate and find a failed unseen-layout example in a five-minute rehearsal. Run AppTest plus manual desktop and narrow-width checks. Assert that a tab or study switch pauses playback, a stale seek cannot move a different episode, final replay uses recorded actions, and failure cards show cohort and frame identity. Confirm that the group can demo from saved data without launching training.

### P2.1 Reduce the evidence footprint and clarify what source is editable

**Confirmed.** The pinned Git tree has 8,838 tracked files and about 674 MiB of uncompressed blob content. `artifacts/` accounts for 8,623 files and 669.43 MiB, including 4,099 files and 42.59 MiB under artifact `source/` or `snapshot/` directories. The active package has 42 tracked files. Docs has 91, including screenshots and verification records. Per-run copies preserve reproducibility and create many search results for older copies of active modules. `part_a.py` and `unseen_study.py` copy source trees. Release code hashes snapshot bytes, so moving or normalizing old files can invalidate recorded verification. [Pinned tree][tree], [per-run source copies][partsnap], [unseen snapshot code][unseensnap], [release snapshot checks][release], [ignore rules][ignore].

**Change.** Build a manifest for the assessed demo and its transitive dependencies, with task/feature versions, cohort, seed, file hashes and exact replay/model references. Keep old frozen paths and byte hashes through submission. Direct editors to `src/`, `app/`, `configs/` and `tests/`. Group historical document links in an evidence index. For new studies, consider a single source snapshot per verified source identity where the runner can resolve it. Move historical bulk evidence only in a later, separately tested migration if needed.

**Accept when** a trusted second checkout can verify both historical releases and open the assessed demo from the documented evidence paths. Compare exact hashes before and after structural changes. A missing required demo asset produces a recovery message with its path. Record the measured bundle size and identify dependencies shared by both releases before any archive changes.

## 3. Rubric crosswalk and remaining submissions

The rubric assigns **Part A 40, Part B 40 and Output/Presentation 20 points**. Across those sections, **40 points are team-assessed and 60 individual**. The assignment counts for **36% of the subject grade**. The evidence below addresses what the team can demonstrate against each row, without predicting a grade.

| Rubric row | Present evidence | Remaining action and test |
| --- | --- | --- |
| A1 Problem statement, 10 team | Simulation and information contract in the environment spec and technical report. | Team writes a critical discussion of related navigation systems, practical assumptions, why learning is being evaluated against BFS and the group's specific contribution. Review primary sources and check that each claim about another system has a citation. [Environment spec][env], [technical report][reporttop]. |
| A2 Programming, 20 individual | Gymnasium task, tabular and neural agents, automated runs, validation, audits and replay tests. | Give each member a meaningful code path, a regression test and an automation or analysis output. Member explains the flow, handles one expected error and demonstrates a small change. [Core CLI][cli], [Part A runner][partsnap], [test tree][tests]. |
| A3 Degree of completion, 10 team | Functioning simulator and multiple metrics across both benchmarks. | Pass P0 fixes and the fresh-checkout demo. Demonstrate success, collision/timeout, SPL and trajectory on clearly labelled cohorts. [Main package README][innerreadme], [M13 examples][m13]. |
| B1 Model optimization, 10 individual | Search, matched controls, reward/input ablations and a measured vote already exist. | Each member contributes a defensible result to the optimization story. Reuse existing experiments and run the bounded intervention in Section 4 if capacity permits. Explain the vote's success/SPL/runtime trade-off. [M4][m4], [neural confirmations][neural]. |
| B2 System implementation, 15 team | Substantial overlap with the proposal conversation's simulated agents, equivalent task comparison and multiple layouts. | Check against the accepted proposal and attach one result or demo action per promised behavior. Ask the tutor how to record any scope changes where needed. [Proposal conversation][proposal], [unseen final][m12]. |
| B3 Results, 10 individual | Seed-level means, paired comparisons, outcome metrics, layout intervals, cost and replay error evidence. | Every member can regenerate their table or figure and explain its denominator, appropriate uncertainty unit, largest failure mode and conclusion. [M4][m4], [M12][m12], [M13][m13]. |
| B4 Organization, 5 team | Consolidated technical report plus milestone protocols and results. | Team compiles the submission report in problem → methods → Part A → Part B → frozen evaluation → limitations order. Make tables and figures answer the stated questions. Link to detailed historical evidence in an appendix. [Technical report][reporttop]. |
| Output, 10 individual | Interactive replay, saved comparisons, charts and maps. | Each member provides one legible output, verifies displayed metrics against the saved source, and demonstrates where to find it. [Dashboard][dashload], [unseen page][unseenload]. |
| Presentation, 10 individual | Available saved replays and comparisons. Presentation readiness is unverified. | Rehearse using one short segment per member. Each person explains and defends their design and results, answers a peer's question and shows how they would modify their own code. |

**Submission checks from the brief.** Final report in Week 6 and practical-class demo/Q&A from Week 7. Submit the group report and source through Google Classroom with one separate assessment rubric. The report needs a cover page, Times New Roman 11 pt, single spacing, around 40 pages and fewer than 50, plus named allocations and a project plan. Record AI assistance using the prescribed disclosure form. Use this file as an internal technical checklist under the brief's Yellow limited-AI terms. Members verify the evidence, write their own academic analysis and explain their own work.

## 4. Remaining experiment work

### Part A. Reconcile existing studies, then isolate the representation effect if needed

1. **Required evidence repair.** Publish a traceability table from the original question through the M3 selection rule and initial shortlist to the subsequent four-family development comparison and unseen-layout experiment. Include the M8 paired DQN result and the observed M13 failure patterns. Use one figure per claim. A suggested figure is a seed-level success/SPL plot, with the three original Part A families framed as the selection cohort and PPO identified as a later baseline. A second figure can show the two-cell cycle rate by algorithm and route band on unseen development layouts with counts. Choose supporting trajectories after fixing the report's selection rule for examples.
2. **Open controlled comparison.** The M10→M11 change alters representation, the linear value-function implementation, training budget and seeds. The repository explicitly treats it as descriptive. If the group intends to claim that the shared features themselves help, run raw 104-input PPO versus the 1,234-feature PPO pipeline on the same task, training layouts, PPO settings, three declared paired seeds, 200,000 interactions each and the same checkpoint monitor. Reuse the existing extractor and runner in `shared_neural.py` and `unseen_main.py`. If first-layer parameter counts differ, disclose the difference and call the result a comparison of input pipelines. Match parameter counts with a predeclared design if the team needs an effect attributed solely to the representation. The M8 ablation already covers the general representation-refinement criterion. [M8 results][m8], [M10 protocol][m10p], [M11 limitations][m11limit], [shared neural code][shared], [main runner][mainrunner].

### Part B. Test one change motivated by the diagnosed failures

1. **Declare one hypothesis.** For example, test whether randomized action selection among near-tied policy actions after detecting a repeated two-cell transition reduces persistent timeouts. Start with the existing shared-feature SARSA family or one preselected PPO family. Document the extra state/history used by the proposed controller, legal action policy, trigger, candidate actions, tie behavior and seeded RNG. Cap recovery attempts and keep collision and timeout rules identical. Introduce a distinct controller ID and label the method as a history-aware controller. Old M13 classifications assume consistent deterministic actions per state. Update the evaluator for an intervention that deliberately changes actions at repeat visits, and preserve the original diagnostic contract for historical runs. [Original diagnostic definitions][m13p], [classification code][diagnose], [M12 protocol][m12p].
2. **Run matched controls.** Use the same pretrained checkpoints, route roster, step cap, rewards and observation version for the base controller and the wrapper. Review the implementation on a hand-solvable two-cell fixture, an unaffected successful path and a fixture where avoiding a cycle leads to a collision. Then compare on the existing four development layouts, keeping model seed separate from intervention RNG seed. Predeclare the primary metric (success with SPL and collision guardrails), support and resource cap. Example promotion rule for the group to agree before execution: success improves by at least 5 points, mean SPL does not decrease and mean collisions stay at or below 2%. Explicitly track paired `timeout→success` and `timeout→collision` counts, persistent cycle rate, extra decision time and all seed-level outcomes. A missed target can be the result.
3. **Use a new holdout only after freezing.** The M6 pairs and M12 layouts are already inspected. For a confirmatory claim, reserve a new versioned layout roster outside the 12/4/10 layouts, check for duplicate layouts, declare routes and difficulty strata, pin the controller, checkpoints, RNG policy and analysis config, then score the roster once. Keep development and final results distinct. A smaller study can stop at development validation and report that scope. Bootstrap by layout where sensible, report training-seed variation separately and treat routes from one layout/policy as repeated observations. [Final freeze runner][unseenfinal], [original holdout and limitations][m12].
4. **Consider a second ensemble only if it answers a specific question.** The existing Q-learning vote satisfies the combine-model experiment. Its uncertainty is limited to one assembly. If the group proposes combining different families, predeclare a rule using compatible policies on the same task and compare with each member and the selected single under matched scenarios. Report agreement, tie frequency, costs and collision/timeout changes. A fixed-map Q-table and unseen-layout feature model use different task signatures. [Voting implementation][voting], [M4 results][m4].

The current simulator supplies complete deterministic observations for its respective task versions. In the report, state that the assignment's incomplete-information example has no missing-data case in this design. If the agreed proposal explicitly includes missing sensors or uncertain inputs, create a separate versioned experiment with a declared corruption mechanism and a matched complete-input control. Choose the loop intervention first if time covers only one follow-up.

**Experiment acceptance.** Every included result has a checked config, code and task identifiers, committed or retrievable checkpoint identity, per-seed rows, matched scenarios, exact run command and a plot generated from those rows. A clean reviewer run reconciles outcome counts, means and SPL. The analysis states what changed, how checkpoints were selected, the uncertainty unit and whether success improved at the expense of other outcomes. A run with incomplete support is attributed and excluded from any complete-run headline with the reason recorded.

## 5. Target structure and staged refactor

Keep the existing package name and user-facing commands during the submission sprint. Organize by responsibility before moving files. The paths below show a first-pass layout, with the existing dated docs and artifacts accessible through indexes.

```text
README.md                            repository landing page and entry command
autonomous-robot-navigation-rl/
  README.md                          setup, saved demo and reproduction
  app/streamlit_app.py                page registration and dispatch
  src/robot_navigation/
    core.py, benchmark.py             task rules and familiar split
    unseen_benchmark.py              layout roster and task identities
    agents.py, dqn.py, ppo.py
    shared_agents.py, shared_neural.py
    voting.py                        policy implementation and inference
    episodes.py, workflow.py          metrics, rollout and playback state
    part_a.py, refinement.py          selection and refinement logic
    release.py, unseen_final.py       frozen evaluation contracts
    dashboard.py                     evidence loading and comparison helpers
    dashboard_ui.py, simulator_ui.py
    unseen_ui.py, interface.py        render and handle user interaction
  configs/                           versioned study inputs
  data/                              scenario manifests and maps
  scripts/                           reproduction commands and verification
  docs/
    evidence_index.md                 rubric, cohort and command links
    technical_report.md               technical reference for the group
  artifacts/
    runs/                            saved runs and policy checkpoints
    studies/                         comparisons and frozen evidence
  tests/                             unit, benchmark and viewer checks
```

`unseen_study.py`, `unseen_main.py`, `unseen_diagnostics.py` and the neural study modules remain the corresponding study entry points in the first pass. Evidence readers supply typed or explicitly validated data to render functions. UI owns selection and playback state. Policy and environment code keep Streamlit out of training and evaluation.

| Stage | Work and handoff | Merge gate |
| --- | --- | --- |
| 0. Baseline, half day | Record audit commit, exact test environment and verification commands. Agree on one demo path and confirm the group proposal used for marking. Add failing fixtures for P0.2/P0.3. | Baseline metrics and replays recorded. Regression scenarios reproduce the targeted behavior. |
| 1. Correctness, days 1–2 | Fix root README, isolate artifact loading and switch the positional join to algorithm IDs. Keep patches small. Preserve current identifiers, relative paths and on-disk schemas. | P0 tests pass and baseline artifacts render with the same values. One teammate follows the README in a fresh checkout. |
| 2. Presentation and evidence, days 2–4 | Centralize navigation, explain benchmark scope, update the current technical reference and build the cohort/ownership index. Add a demo manifest. Start from saved validation and inspected final tables. | Desktop and narrow rehearsal pass. Headline values and claims reconcile with source results. Each team member can demonstrate their output. |
| 3. Controlled follow-up, days 3–5 if resources allow | Run the loop intervention against its paired controller on development layouts; plan the extra holdout only after freezing. Add the representation control only if the group makes the causal representation claim. | Registered config, output rows, paired summary, cost, limits and reproducible command. Preserved earlier evidence verifies. |
| 4. Submission rehearsal, final session | Cross-review report and proposal conformance, run the saved demo, then rehearse Q&A and collect individual evidence. | Confirm all graded rows, source/report/rubric submission materials and AI usage disclosure. Verify 1–2 claims and one code modification per person. |

**Refactor boundary.** Frozen snapshots include explicit byte checks. Before editing a path, find its entry in both freeze manifests. Run historical verification against the preserved snapshot using the corresponding runner. Treat verified metric values, chart cohort labels, task signatures and replay event identity as regression fixtures. For the fixed release, `scripts/replay_frozen_release.py` restores sealed inputs to a separate workspace. Use a separate branch or checkout for future study changes and version outputs with new IDs. [Fixed-release replay helper][replayfrozen], [fixed release checks][release], [unseen release checks][unseenfinal].

**Suggested verification from the project directory** using the documented virtual environment. Run verifiers that write metadata in a disposable checkout or scratch workspace.

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python -m pip check
& $python -m robot_navigation.cli check
& $python -m pytest -q
& $python scripts/replay_frozen_release.py --workspace .cache/audit-frozen-replay
& $python -m robot_navigation.unseen_diagnostics verify
& $python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Use a fresh workspace name if the fixed-release replay destination already exists. Pair automation with a manual test of launch, navigation labels, playback speed and failure display at desktop and narrow widths.

## 6. Four-member allocation

Use actual names in the submission. Estimates are hands-on time for the remediation sprint and exclude unattended training. Everyone owns code, an experiment or measurement, a report section and an individual demo.

| Member | Code and verification | Individual evidence and demo | Estimate and review |
| --- | --- | --- | --- |
| 1. Environment and fixed-layout evaluation | Own P0.1, fixed-layout loading boundary in `app/streamlit_app.py`, environment/split checks and the clean-checkout reproduction path. Add the regression case proving the unseen page can load without the fixed-layout validation file. | Explain simulation, observations, random/BFS context and the M3 shortlist. Regenerate the fixed-layout comparison and show a valid route with goal/collision/timeout rules. | 8–10 h. Reviews Member 2's matched protocol. |
| 2. Optimization and intervention | Own the paired controller study, config, rollout integration and hand-solvable loop/collision regression tests. Build on existing implementations and produce an analysis command for paired outcomes. | Explain M4 parameter/reward effects and the voting result. Demonstrate one matched optimization replay or show a recorded comparison with the experimental decision and cost. | 8–10 h. Reviews Member 3's metric and split definitions. |
| 3. Unseen-layout evaluation and evidence integrity | Own P0.2 evidence validation for unseen studies and P0.3 algorithm-ID join in `dashboard_ui.py`. Test schema/signature/body changes, evaluation scope and evidence linkage. Verify M12/M13 and generate the seed/layout results table. | Explain unseen layout support, failure rates and uncertainty. Show an inspected final result and trace one metric back to the preserved artifact. | 8–10 h. Reviews Member 4's dashboard semantics and demo path. |
| 4. Dashboard and report integration | Own P1.2 navigation and proposed demo manifest. Add AppTest for playback/seek/selection and automate the demo-bundle asset check. Maintain the scope-to-evidence index and lead the timed rehearsal. | Show synchronized replay, explain displayed SPL and failure-location evidence, and demonstrate the unavailable-asset state. Report a measured usability or evidence-loading result from their own tests. | 8–10 h. Reviews Member 1's setup and independent launch instructions. |

Member 3 owns evidence validation/join code, while Member 4 owns the presentation of these results. Sequence the two changes through a shared fixture and separate pull requests. Members 1 and 4 agree on page boundaries before editing the app entry point. For each contribution keep a small record with owner, code/commit, test or run command, resulting artifact, interpretation and peer reviewer. Everyone drafts their own assessed analysis. The group shares final report assembly and submission checks.

## 7. Audit verification and limits

The audit compared the GitHub default-branch commit with matching local Git objects and read a disposable archive of that exact tree. It inspected the core task, CLI, dashboard and replay paths, saved studies and protocols, assignment specification and original Word rubric.

- **Passed.** Windows/Python 3.14.10 dependency consistency and Gymnasium environment checks. The fixed-release replay helper restored and verified the sealed workspace. The M13 verifier reconciled 8,064 recorded cases and 908,298 actions with zero policy queries.
- **Passed.** 96 tests from test modules that do not use Streamlit AppTest. A separate targeted run also passed 28 tests across the environment, benchmark splits, unseen release and unseen diagnostics. Temporary fixtures reproduced malformed unseen summary handling, an unverified four-family study interrupting Methodology, acceptance of an unverified fixed summary body, and the route-length mislabel after reversing one mapping.
- **Current UI check incomplete.** This audit's Windows runtime stalled inside local socket setup for Streamlit AppTest, including with a one-widget app. The repository records 202 passing tests and desktop/narrow checks for the pinned milestone. A fresh end-to-end UI run is a release gate for the team.

Existing study numbers are cited as repository evidence. The new experiment designs, effort ranges and team allocation are proposals for the group. The tutor-approved proposal, individual contributions, a cold dependency installation on a second machine and a live presentation remain team verification items.

[commit]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/commit/d1ed065a6d76240226f1c671664978ca623ab188
[proposal]: chatgpt-conversation://6ae851c0-fa20-83ec-9f7f-3e7be7121090
[m3]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/milestone3_results.md#L9-L84
[ppo]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/ppo_comparison_results.md#L3-L42
[m4]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/milestone4_results.md#L5-L90
[m6]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/milestone6_results.md#L1-L34
[m8]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/milestone8_results.md
[neural]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/neural_confirmation_results.md
[m11p]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/milestone11_protocol.md#L8-L74
[m12]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/milestone12_results.md
[m13]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/milestone13_results.md
[m13p]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/milestone13_protocol.md#L9-L51
[finalconfig]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/configs/unseen_final.yaml#L1-L15
[rootreadme]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/README.md#L1-L85
[innerreadme]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/README.md
[dashload]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/dashboard_ui.py#L45-L124
[fourload]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/four_way.py#L14-L56
[unseenload]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/unseen_ui.py#L34-L103
[app]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/app/streamlit_app.py#L48-L65
[dashjoin]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/dashboard_ui.py#L73-L90
[details]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/four_way_analysis.py#L162-L183
[dashmethod]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/dashboard_ui.py#L126-L162
[reporttop]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/technical_report.md#L1-L45
[reportend]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/technical_report.md#L169-L205
[privateplan]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/Autonomous_Robot_Navigation_Codex_Project_Plan.md#L3-L60
[appnav]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/app/streamlit_app.py#L16-L45
[dashpages]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/dashboard.py#L10-L27
[selector]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/interface.py#L83-L139
[unseenplay]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/unseen_ui.py#L231-L283
[partsnap]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/part_a.py#L98-L119
[unseensnap]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/unseen_study.py#L346-L369
[release]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/release.py#L90-L127
[fixedpublish]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/scripts/run_final_evaluation.py#L187-L204
[ignore]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/.gitignore#L6-L42
[env]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/environment_specification.md
[cli]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/cli.py#L10-L52
[m10p]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/milestone10_protocol.md#L8-L61
[m11limit]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/milestone11_protocol.md#L68-L84
[shared]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/shared_neural.py
[mainrunner]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/unseen_main.py#L29-L112
[diagnose]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/unseen_diagnostics.py#L31-L79
[m12p]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/docs/milestone12_protocol.md
[unseenfinal]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/unseen_final.py
[voting]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/src/robot_navigation/voting.py#L11-L41
[replayfrozen]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/blob/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/scripts/replay_frozen_release.py#L18-L48
[tree]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/tree/d1ed065a6d76240226f1c671664978ca623ab188
[tests]: https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/tree/d1ed065a6d76240226f1c671664978ca623ab188/autonomous-robot-navigation-rl/tests
