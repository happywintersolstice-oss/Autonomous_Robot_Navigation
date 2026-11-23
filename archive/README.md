# Research history archive — read-only

The four-action, coordinate-only application is retired. The only installed application is now the partially observed, five-action dynamic-navigation product. This archive never offers an old simulator or executes historical policies.

`INDEX.csv` lists unique preserved file versions, original and current backing paths, sizes, SHA-256, Git blob identity, category and source version. `manifests/original-path-map.csv` includes all same-content aliases and their origins. `manifests/pre-migration.json` is the complete pre-change inventory; `manifests/integrity.json` records the post-move full hash check. `manifests/source-versions.json` explains commit/branch and local uncommitted snapshot labels. Git history and previous branches remain intact.

- **legacy-classic/**: original classic data, maps, models, training curves, learning replays, logs and result assets, with original inner relative paths.
- **historical-source/**: retired application/source/tests, launchers and reproduction material. It is outside the installed `src` package and is not imported by app startup/training.
- **workspace-snapshots/**: unique historical files and local uncommitted source/output versions from the original workspace and prior worktrees, including failed engineering evidence. The original dirty workspace was read, never moved or edited.
- **git-objects/**: earlier or deleted Git file versions not represented by the on-disk snapshot. Original path and introducing commit are in the inventory. These are real file bytes, not placeholder references.
- **partial-research references**: sealed five-action data remains byte-identical at `autonomous-robot-navigation-rl/artifacts/partial-dynamic-navigation-v1/` and `data/partial/`, with a complete path/hash manifest in `manifests/partial-research.json`. This includes feasibility/Phase 5A, 5B–5H, failures, checkpoints, optimizer/rollout/replay traces, code/config snapshots, reports and original briefs.

Every file version is accessible at its recorded `archive_path`. Identical content can have several recorded original paths and one backing file. Archived `.gitignore` and `.gitattributes` files have a `.archived` suffix on disk so they cannot change archive Git behavior; ZIPs use their original names. Historical reports were not edited to repair old relative links. Use the path map, the exported original directory hierarchy, or the original Git commit when reproducing old work. Source changes needed by the new app do not replace their archived historical versions.

## Export real ZIP files

From the repository root, with Python 3.11 or newer:

```powershell
python scripts/export_history_archives.py --list
python scripts/export_history_archives.py --output-dir history-exports
python scripts/export_history_archives.py --output-dir history-exports --verify
```

The fresh output directory receives **legacy-classic-history.zip**, **partial-navigation-research-history.zip** and **archive-inventory.json**. ZIP entries use fixed timestamps and streamed copies. The inventory supplies each member's path/size/SHA-256/category/source version and each ZIP checksum. `--verify` streams and checks all ZIP members and coverage; `--verify-sources` separately rechecks every recorded local backing file. Choose a new output directory for another export. Generated duplicate ZIPs are intentionally not committed to Git.

Each ZIP's `original/` contains the corresponding original snapshot hierarchy; `workspace/` preserves unique local variants and `git-versions/` preserves older versions. `ARCHIVE-ORIGINS.json` explains deduplicated origins and source versions. Shared reproduction source can appear in both ZIP collections. The archive is not a promise that old experiments run under today's libraries; use their pinned protocols and original commits.

LFS pointers are distinguished from payloads. Missing payloads cause an explicit failure and file list; recover with `git lfs fetch --all` and `git lfs pull`, then regenerate/verify the archive. No missing data is silently replaced. Large ZIPs require sufficient disk space and may take time; exporting never deletes originals.

Historical final/unseen results are reference-only, not new generalization evidence. Four-action checkpoints cannot be converted into this app's 409-observation/five-action models. The live registry contains new native app training runs only; archived scientific scores never enter its comparison tables.


## Current document locations

Implementation briefs, prior product reports, Phase5 notes and earlier validation records now live under `docs/history/`. Their bytes and original source catalog remain preserved. `archive/relocations.json` maps the original backing paths to their current storage locations; the export/verification tools use that map while retaining original ZIP member names and checksums.

`archive/source-snapshots/` also contains verified recovery ZIPs for removed active test source and the earlier dirty Desktop source. Local original data/artifact directories are preserved under the Desktop application's ignored `.runtime/recovery/` folder; these local recovery copies are separate from the committed historical catalog.
