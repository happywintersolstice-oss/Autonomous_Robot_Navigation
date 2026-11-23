# Maintenance notes

The owner requested removal of all active automated test files on 23 November 2026. The current application's six test modules, two test-only JSON fixtures and root pytest configuration were archived, then removed. The older Desktop checkout's remaining 46 test source/helper files and its pytest configuration were treated the same way.

A verified [archive manifest](../archive/source-snapshots/removed-active-tests-20261123T101901Z/manifest.json) and [source ZIP](../archive/source-snapshots/removed-active-tests-20261123T101901Z/source.zip) preserve all 56 source/configuration/fixture files with original paths and SHA256 values. Existing validation reports, historical archives, training data, product evaluation panels, checkpoints and measured results were not deleted or rewritten. Evaluation/test panels are application research data, not automated test source.

Pytest configuration, the optional test dependency, test-only lock entries and editor discovery settings were removed. Current installation/maintenance guides no longer direct users to deleted files. This does not uninstall packages from an existing environment.

No replacement test suite was created. The current smoke check is `py -3 run_app.py --check`; compilation and strict loading of the four selected checkpoints were also checked after removal. No training, optimizer updates or frozen-test policy execution is needed for this cleanup. Prior passing test results remain statements about their original runs, not evidence that a now-removed suite was rerun.

Archived test sources and recorded validation outputs are reference material. Do not run automatic test discovery against archived research trees.


## Canonical main workspace consolidation

The Desktop repository itself is now the current `main` checkout. Its launcher and editor workspace use `autonomous-robot-navigation-rl/` directly; no nested forwarding launcher is needed. The previous implementation briefs, product reports and application research notes are organized under `docs/history/`. Model and dataset bytes were retained; saved-model module/class paths were not renamed.

Before replacing the old Desktop tree, 250 source/configuration/document files were saved in a checksum-verified source ZIP together with the old Git index and uncommitted patches. The original source and data directories were also moved intact into local recovery storage. Original data/artifact counts are preserved: 14,638 artifact files (1,149,859,557 bytes) and 22 data files (5,947,772 bytes). Six prior linked checkouts were grouped with `git worktree move` under ignored `.runtime/history-worktrees/`, preserving their working files.

Recovery source: [snapshot manifest](../archive/source-snapshots/desktop-before-consolidation-20261123T103531Z/manifest.json). Raw local recovery: `autonomous-robot-navigation-rl/.runtime/recovery/desktop-before-consolidation-20261123T103531Z/`. To inspect earlier source, extract its ZIP into a separate folder and use the recorded Git context.

Validation from the actual Desktop location passed: current import root/version 2.0.0, normal launcher, 24-record CLI registry, strict loading of four selected checkpoints, syntax checks, all five passive AppTest pages, all 95 relocated document hashes and all 14,529 historical catalog paths/file sizes. Relocation-aware ZIP export was checked on a small synthetic source after correcting a missing required metadata field in that fixture. No historical data was altered by that fixture. The full archive was not rehashed. No training, optimizer update or frozen-evaluation execution occurred. No manual visual browser review is claimed.

Two obsolete previews serving the same repository from the Documents linked checkout were stopped. Port 8517 now serves the canonical Desktop `main`; both HTTP root and health endpoints returned 200. That Documents checkout's files remain untouched.

Detailed receipts: [Desktop validation](history/maintenance/canonical-workspace/desktop-main-validation.json), [source/data preservation](history/maintenance/canonical-workspace/preserved-directories.json), [safe checkout relocation](history/maintenance/canonical-workspace/historical-worktrees.json), [materialized main](history/maintenance/canonical-workspace/materialization.json), and [HTTP verification](history/maintenance/canonical-workspace/http-main.json).
