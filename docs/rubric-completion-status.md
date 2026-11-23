# Rubric implementation and remaining human evidence

Updated 23 November 2026. The remaining **technical model-combination gap is implemented and measured**. Full assignment compliance is still **not established**, because the required team-authored paper, actual individual contribution evidence, approved scope, AI-use permission/disclosure, rehearsal and submission have not been provided or verified.

## Completed technical work

- The original four real algorithms and every historical checkpoint/result remain preserved. No new training, optimizer steps or previously frozen held-out policy execution occurred.
- Added an explicit equal-vote ensemble of the four selected policies. Ties use a rule fixed before evaluation; every member remembers the action actually executed. No planner fallback, mask, fitted vote weights or obstacle identity is used.
- Registered and committed the new protocol before inference at `6097c72db6e66b2634b00fba7cf3786202d58480`. Ran seven controllers on 32 previously inspected development cases plus 4 existing manual scenes:252 complete episodes, 4,092 environment actions, within the 50,400-action/one-hour cap. All attempts/outcomes were retained; there was no retry or post-result retuning.
- The ensemble achieved 36/36 goals, versus Q-learning 34/36 and SARSA/DQN/PPO 36/36. It did not beat the strongest individual models; its longer paths and higher local inference cost are reported. It remains optional rather than silently replacing an individual model.
- Added matched CSV/JSON/replays/ZIP, actual voting traces, timing, eight-family descriptive bootstrap intervals and 1,024 offline public-history sensitivity probes. Sensitivity and development results are not causal importance or unseen claims.
- Simulator offers the ensemble separately. Compare & Export has a read-only Ensemble development scope and an explicit optional ensemble checkbox for new Phase3 development comparisons. The original held-out view/selection stays unchanged.
- Prepared evidence figures, a verified PDF technical appendix, a fillable report outline, contribution/disclosure support and a readiness checker that reports missing evidence without inventing a passing grade.

## Matched development results

All rows use the same 36 cases and environment seed 0. This is not the original 64-case heldout or a new unseen test. Wait fractions below include failed episodes. The raw export separately includes successful-only Wait/step denominators and condition/topology/outcome strata.

| Controller | Goals | Collisions | Timeouts | Waits / all actions | Mean successful steps | Local ms / decision |
| --- | --- | --- | --- | --- | ---: | ---: |
| Q-learning |34/36|0/36|2/36|75/837|12.85|0.86|
| SARSA |36/36|0/36|0/36|30/506|14.06|0.90|
| DQN |36/36|0/36|0/36|247/765|21.25|0.94|
| PPO |36/36|0/36|0/36|51/471|13.08|1.01|
| Equal-vote ensemble |36/36|0/36|0/36|34/628|17.44|3.90|
| Observed-map replanner, non-RL |34/36|0/36|2/36|31/803|11.85|0.20|
| Random, non-RL |0/36|36/36|0/36|18/82|N/A|0.06|

Each ensemble member retains its original 40,000-action checkpoint. The ensemble has **zero new training actions and no new weight file**; a member-set hash identifies its constituent saves. It is a combination of learned policies, not a fifth independently trained RL family. Its four proposals per decision explain some of the increased cost; the timing is one local run, not a deployment benchmark.

There were 395 unanimous votes, 45 tied decisions and 34 chosen Waits among 628 ensemble decisions. The fixed tie rule was not adjusted after seeing these results. No weight/selection promotion was performed. Descriptive family-cluster intervals include zero for the improvement over Q-learning; degenerate zero intervals against the three perfect individual policies only describe this observed development sample, not unknown future cases.

## Actual requirement status

| Requirement group | Technical evidence now available | What still requires the team |
| --- | --- | --- |
| Part A problem and appropriate algorithms | Defined warehouse simulation, four learned families, public state/memory and automation | Team's own critical problem argument, related-system evaluation, verified originality and accepted scope |
| Part A measurement, patterns and errors | Matched panels, strata, failures, history-sensitivity probes, real learning receipts and selection | Student interpretation and understanding; no universal generalisation or causal claim is justified |
| Part B optimization | Prior bounded candidates/settings, measured official metrics, actual updates | Each member must defend an authentic optimization contribution |
| Part B model combination | Current public-only ensemble implemented and compared, including cost and limits | Do not claim it beats the strongest single model or has an unseen score |
| System implementation and programming | Integrated working app, explicit exceptions, preserved signatures, focused tests | Actual proposal conformance and individual ability to explain/modify code |
| Output and organization | Clear figures, maps, replays, downloadable evidence, technical appendix and outline | Complete final student paper around 40/below 50 pages, formatting/content review, separate completed rubric |
| Presentation and individual marks | Checked software trajectories and code walkthrough | Named real contributions, rehearsal and individual Q&A/modification competence |
| AI-use and submission | Actual assistance notes and machine checklist | Prescribed form, permitted-use resolution, authentic declarations and actual Classroom receipt |

The school's rubric is not an acceptance-test score. A code test can demonstrate software behavior; it cannot establish a person's authorship, competence, tutor approval, novelty or actual submission. No completed signatures or contribution claims are fabricated.

## Evidence and reproduction

Study: `autonomous-robot-navigation-rl/runs/rubric-readiness/20261123T072303Z-6bec5a6f/`.

- [Protocol](../autonomous-robot-navigation-rl/runs/rubric-readiness/20261123T072303Z-6bec5a6f/protocol.json)
- [Actual completion receipt](../autonomous-robot-navigation-rl/runs/rubric-readiness/20261123T072303Z-6bec5a6f/complete.json)
- [Comparison JSON](../autonomous-robot-navigation-rl/runs/rubric-readiness/20261123T072303Z-6bec5a6f/comparison/manifest.json)
- [Case CSV](../autonomous-robot-navigation-rl/runs/rubric-readiness/20261123T072303Z-6bec5a6f/comparison/case-results.csv)
- [Summary CSV](../autonomous-robot-navigation-rl/runs/rubric-readiness/20261123T072303Z-6bec5a6f/comparison/summary.csv)
- [Replay ZIP and inventory](../autonomous-robot-navigation-rl/runs/rubric-readiness/20261123T072303Z-6bec5a6f/comparison/results-and-replays.zip)
- [Vote diagnostics](../autonomous-robot-navigation-rl/runs/rubric-readiness/20261123T072303Z-6bec5a6f/vote-diagnostics.json)
- [Submission support files](submission/README.md) and [current missing-evidence result](submission/readiness-result.json)

The pre-result mechanical suite passed 6 checks. The final integrated technical acceptance suite passed **39 checks in 22.31 seconds**, including the real manual ensemble journey, saved evidence, public-only history, model compatibility and all five pages. The readiness fixtures were additionally rechecked after removing a dependency on the real team manifest, so future manual completion will not break regression tests. These are focused product checks, not historical scientific regressions. Final receipts are in `docs/rubric-evidence/`; packaging, compilation, formatting, root/health HTTP200, HTML edit/download and all three PDF-page checks passed. The normal app remains on 8517; its temporary 8521 smoke process was stopped.

From the application directory:

```powershell
python -m pytest developer/tests/test_product_phase3.py developer/tests/test_cleanup_integration.py developer/tests/test_product_phase2.py developer/tests/test_ensemble.py -q -o cache_dir=.runtime/rubric-cache --basetemp=.runtime/rubric-acceptance --junitxml=.runtime/rubric-acceptance.xml
python -m pytest developer/tests/test_submission_readiness.py -q -o cache_dir=.runtime/rubric-cache --basetemp=.runtime/rubric-submission-check --junitxml=.runtime/rubric-submission-check.xml
python developer/commands/build_rubric_figures.py
```

The registered study is deliberately one-shot in its folder. No learner is invoked by the figure or document builders. The document builder requires ReportLab and Windows Times New Roman fonts; this task used the bundled document Python. Native Word export stalled, so no unverified DOCX is delivered. The PDF appendix was independently generated, rendered and reviewed; it is not the requested 40-page final academic report.

## Manual team items as requested

The user has confirmed that the team will complete these manual items themselves. They do not block completion of the technical implementation. The team should supply the real 3-4 member names/IDs and actual contribution evidence, the accepted proposal/scope record, the team's authored report/draft, the prescribed AI disclosure and tutor guidance, the completed separate rubric, rehearsal evidence and the actual submission receipt when available. Fill only truthful fields in `docs/submission/submission-manifest.json`, then rerun its checker. The currently blank official cover and rubric found in Downloads do not establish those facts.
