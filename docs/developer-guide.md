# Active application development

Read [the code walkthrough](code-walkthrough.md) first. It follows a user action from the launcher to its page, service, policy, environment and saved replay. Open `Robot Navigation.code-workspace` to work on this checkout. Install using [the Windows guide](windows-install.md).

The sole UI entry is `app/streamlit_app.py` -> `dynamic_demo_ui.render` -> `ui/` page modules. Edit presentation in `ui/`, shared labels/navigation in `ui/components.py`, and CSS in `ui/styles.css`. Guide text is a packaged Markdown resource. Simulator, Train, Compare, Guide and Archives share that router; there is no classic-app runtime.

Application services and strict checkpoint loading live in `app_service.py` and `app_registry.py`. The environment/sensing/memory/reward contract is in `partial_*`. Versioned F2/F3 learners must keep their original module/class names because serialized neural policies refer to them. Formatting and explanatory comments do not require retraining; observation/feature/reward changes need a deliberate versioned experiment. Historical source hashes describe the original runs, not the newly formatted source.

Automated test source, test-only fixtures, pytest configuration and its development dependency were removed at the owner's request. Historical validation outputs are retained as evidence of prior runs. The removed source is preserved in a verified archive; see [maintenance notes](maintenance.md).

From the repository root, run `py -3 run_app.py --check` for a basic import check. Review Simulator, training controls, saved comparisons and replay manually after future changes. This import check is not a replacement claim for the removed regression suite.

Training is an explicit separate action, such as `python -m robot_navigation.app_training --features product-v3 --preset quick --models all`. It creates a new immutable run. Do not use training merely to check a page renders. View the published test through Compare & Export; never remove a use marker to obtain another supposedly unseen assessment.

Archive scripts stream historical ZIP exports and verify inventories when explicitly invoked. Archived source/research is not installed as another selectable app mode. Work in the canonical Desktop main checkout; source, data and uncommitted changes from the older checkout were preserved before consolidation.
