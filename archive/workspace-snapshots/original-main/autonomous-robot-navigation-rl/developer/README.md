# Developer tools

Run commands from the application directory, one level above this folder.

- `tests/`: automated regression tests. Run `python -m pytest -q` using the application virtual environment.
- `commands/`: real analysis, study, reproduction and release tools. Follow each study protocol and use fresh output IDs.

See the [developer guide](../../docs/developer-guide.md). Scratch outputs belong under `.runtime/`; published evidence belongs in versioned `artifacts/` and `docs/evidence/` locations.
