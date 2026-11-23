# Maintenance notes

The owner requested removal of all active automated test files on 23 November 2026. The current application's six test modules, two test-only JSON fixtures and root pytest configuration were archived, then removed. The older Desktop checkout's remaining 46 test source/helper files and its pytest configuration were treated the same way.

A verified [archive manifest](../archive/source-snapshots/removed-active-tests-20261123T101901Z/manifest.json) and [source ZIP](../archive/source-snapshots/removed-active-tests-20261123T101901Z/source.zip) preserve all 56 source/configuration/fixture files with original paths and SHA256 values. Existing validation reports, historical archives, training data, product evaluation panels, checkpoints and measured results were not deleted or rewritten. Evaluation/test panels are application research data, not automated test source.

Pytest configuration, the optional test dependency, test-only lock entries and editor discovery settings were removed. Current installation/maintenance guides no longer direct users to deleted files. This does not uninstall packages from an existing environment.

No replacement test suite was created. The current smoke check is `py -3 run_app.py --check`; compilation and strict loading of the four selected checkpoints were also checked after removal. No training, optimizer updates or frozen-test policy execution is needed for this cleanup. Prior passing test results remain statements about their original runs, not evidence that a now-removed suite was rerun.

Archived test sources and recorded validation outputs are reference material. Do not run automatic test discovery against archived research trees.
