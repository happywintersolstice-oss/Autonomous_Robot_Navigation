# Active application development

Read [the code walkthrough](code-walkthrough.md) first. It follows a user action from the launcher to its page, service, policy, environment and saved replay. Open `Robot Navigation.code-workspace` to work on this checkout. Install using [the Windows guide](windows-install.md).

The sole UI entry is `app/streamlit_app.py` -> `dynamic_demo_ui.render` -> `ui/` page modules. Edit presentation in `ui/`, shared labels/navigation in `ui/components.py`, and CSS in `ui/styles.css`. Guide text is a packaged Markdown resource. Simulator, Train, Compare, Guide and Archives share that router; there is no classic-app runtime.

Application services and strict checkpoint loading live in `app_service.py` and `app_registry.py`. The environment/sensing/memory/reward contract is in `partial_*`. Versioned F2/F3 learners must keep their original module/class names because serialized neural policies refer to them. Formatting and explanatory comments do not require retraining; observation/feature/reward changes need a deliberate versioned experiment. Historical source hashes describe the original runs, not the newly formatted source.

From the application directory, using a compatible installed Python (the launcher also accepts `--python`):

```powershell
python -m pytest developer/tests/test_product_phase3.py developer/tests/test_cleanup_integration.py developer/tests/test_product_phase2.py -q
```

This focused command loads models, checks public-state/version contracts, reads saved product evidence, validates the integrated UI and exercises mechanical fixtures. It does not perform optimizer training or execute the frozen test policies again. The separate `test_dynamic_app.py` contains training fixtures; do not run its full file when no new learner activity is intended. Its UI-only test can be selected explicitly.

Training is an explicit separate action, such as `python -m robot_navigation.app_training --features product-v3 --preset quick --models all`. It creates a new immutable run. Do not use training merely to check a page renders. View the published test through Compare & Export; never remove a use marker to obtain another supposedly unseen assessment.

Archive scripts stream historical ZIP exports and verify inventories when explicitly invoked. Archived source/research is not installed as another selectable app mode. Main and unrelated dirty workspaces are preserved; integration changes are isolated on the cleanup feature branch.
