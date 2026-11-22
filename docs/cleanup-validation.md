# Code cleanup and integration validation

Completed on 23 November 2026 on `feature/phase3-cleanup-assignment-audit-v1`, based on Phase3 delivery `bd13ae2889642b3634e1738720424bb1119cb19c`. This work integrates the unfinished presentation cleanup from the separate Documents workspace with the delivered Phase3 backend. It does not merge into main, overwrite that source workspace or change historical results.

## What changed

- The compressed active Python is expanded into conventional formatting. The final formatter check covers 46 active Python files, excluding archived source and experiment evidence.
- The single `dynamic_demo_ui` entry point routes to focused page modules in `ui/`. Shared navigation/cards, CSS and help text have one owner. The existing dark warehouse presentation from the clean workspace is retained.
- Checkpoint-format detection, selection matching and scope labels use small named helpers; loading uses explicit branches instead of nested conditional expressions. All 24 existing registry records are preserved, including IDs, selected flags, feature versions, training counts and weight hashes.
- Public-history state, TD updates, training boundaries and evaluator separation have explanatory docstrings. Saved model class/module names remain stable.
- Invalid comparison scopes and mismatched algorithm slots fail before evaluation. Custom start/goal choices now match the preview. The Guide again explains the task, observations, genuine algorithms, rewards and page navigation.
- The launcher can reuse a configured existing environment without repointing its editable install. Packaged UI assets are declared in setuptools. Playback tests write fixtures to temporary output rather than adding test journeys to product results.
- [Code walkthrough](code-walkthrough.md) explains how to read and modify the system. [Assignment audit](assignment-compliance-audit.md) maps the actual PDF/DOCX criteria to evidence and unresolved requirements; it does not certify final submission compliance.

## Actual checks

**27 distinct tests passed across focused runs.** The main invocation passed 25 checks in 23.28 seconds. A targeted invocation passed three checks in 3.42 seconds, including one already counted navigation check plus the new launcher and simulator-control checks. After routing that simulator fixture to temporary output, its targeted rerun passed in 2.97 seconds. There were no failing assertions in these cleanup test runs.

The initial command disabled pytest's cache plugin and produced one non-functional unknown `cache_dir` configuration warning. The later commands use an explicit cache location. Original output/JUnit evidence is retained; no failed result was hidden or recast as a pass.

From `autonomous-robot-navigation-rl/` with a compatible installed interpreter:

```powershell
python -m pytest developer/tests/test_product_phase3.py developer/tests/test_cleanup_integration.py developer/tests/test_product_phase2.py -q -p no:cacheprovider --basetemp=.runtime/cleanup-pytest --junitxml=.runtime/cleanup/acceptance.xml
python -m pytest developer/tests/test_cleanup_integration.py::test_page_navigation_and_saved_replays_never_start_training_or_evaluation developer/tests/test_cleanup_integration.py::test_launcher_reuses_the_configured_environment_without_repointing_install developer/tests/test_dynamic_app.py::test_only_new_ui_and_playback_controls -q -o cache_dir=.runtime/cleanup-pytest-cache --basetemp=.runtime/cleanup-controls --junitxml=.runtime/cleanup/controls.xml
python -m pytest developer/tests/test_dynamic_app.py::test_only_new_ui_and_playback_controls -q -o cache_dir=.runtime/cleanup-pytest-cache --basetemp=.runtime/cleanup-controls-temporary --junitxml=.runtime/cleanup/temporary-controls.xml
python -m pip wheel --no-deps --no-build-isolation --wheel-dir .runtime/cleanup/wheels .
```

The 28 environment/learner/training/study modules covered by the executable-syntax guard have the same executable AST as the Phase3 base, excluding comments/docstrings/formatting. This is a source-preservation check, not a mathematical proof of all runtime behavior. Actual old/new checkpoint loads and inference, the full published replay/export checks, and UI interaction tests complement it. There was **no substantive training, no optimizer update and no held-out policy re-execution**. The stored final-assessment use count remains one.

The wheel built successfully and contains the stylesheet, Markdown guide and all page modules; it contains no archive, model-run or historical-artifact payload. Compilation and the final Black 26.10.1 `--check` passed. Formatting tools were isolated in an ignored temporary folder rather than installed into the application's environment.

The launcher import check passed using the ignored machine-local interpreter pointer. Its HTTP smoke used port **8521** to avoid interfering with another process; the normal default remains **8517**. `/` and `/_stcore/health` both returned HTTP200, with health body `ok`. The task's smoke process was stopped afterward. **No manual visual browser assessment was performed.** AppTest covers all five pages, saved development/held-out scope selection, downloads, failure-to-Simulator navigation and passive playback controls.

## Preservation and integration evidence

- [Validation summary](cleanup-evidence/summary.json)
- [Original UI integration source/hashes](cleanup-evidence/integration-source.json)
- [Preservation checks](cleanup-evidence/preservation.json)
- [Main test output](cleanup-evidence/acceptance-output.txt)
- [Targeted controls output](cleanup-evidence/controls-output.txt)
- [Temporary-fixture controls output](cleanup-evidence/temporary-controls-output.txt)
- [Packaging receipt](cleanup-evidence/packaging.json)
- [Formatting receipt](cleanup-evidence/format-check.json)
- [HTTP receipt](cleanup-evidence/health.json)

Git changed-path checks against the Phase3 base show no modifications to archive/, artifacts/, data/, existing runs/checkpoints/replays, or the Phase2/Phase3 results reports. No large archive rehash was performed. Checksums of all 19 adopted source-workspace files still match the originals in Documents; this task did not modify that workspace. The original dirty Desktop main checkout is also preserved. Existing recorded source hashes still describe their original experiment revisions, not the newly formatted active files.

## Open and launch the integrated checkout

Open `Robot Navigation.code-workspace` in this cleanup feature checkout. The existing local worktree is `C:/Users/User/Desktop/Autonomous_Robot_Navigation/autonomous-robot-navigation-rl/.runtime/newapp`. The Documents source workspace still contains its original unfinished changes; open the integrated checkout when reviewing this delivery.

From the integrated repository root:

```powershell
py -3 run_app.py --check
py -3 run_app.py
```

Double-clicking `Start Robot Navigation.cmd` at the root or `Start Navigation.cmd` inside the application uses the same launcher. This machine's ignored `.runtime/python-path.txt` reuses the existing compatible Python. A fresh clone should follow [Windows installation](windows-install.md); an absolute interpreter path can also be provided with `--python`.

The app/code consolidation is complete. Assignment gaps in the separate audit remain real: in particular current model combination, the formal paper/submission package, verified individual/proposal evidence and AI-use compliance. Code organization and passing tests do not resolve those requirements.
