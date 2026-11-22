# Native partial/dynamic application

This is the only active application. See the repository-root [README](../README.md), [Windows installation](../docs/windows-install.md) and [delivery report](../USABLE_APP_DELIVERY.md).

Install `python -m pip install -e ".[test]"` in a local virtual environment. Start from the repository root with `py -3 run_app.py`, or here with `python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8517`.

Train all four real five-action algorithms with `python -m robot_navigation.app_training --preset quick --models all`; `--list-models` lists the live registry. New runs/checkpoints/replays/comparisons live in `runs/`, background progress/cancellation in `.runtime/jobs/`, separate from historical scientific evidence.

The original partial environment/encoders remain authoritative. `core.py` re-exports neutral GridMap/JSON utilities; `release.py` supplies only a checksum, with no old runtime imports. Historical code/tests/data remain in `archive/` or retained sealed evidence paths. No classic routes are installed.

Focused checks: `python -m pytest developer/tests/test_dynamic_app.py -q`. The old research suite is archived, not rerun for this product. Quick models may fail; references are never hidden fallbacks.

Phase3 adds a versioned public-history learner, four new demo presets, and saved read-only held-out comparisons with failure replay access. See [Phase3 measured results](../PRODUCT_PHASE3_RESULTS.md) and [checked teammate script](../docs/team-demo-script.md). Phase2 and archived evidence remain preserved.
