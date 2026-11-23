"""Team walkthrough and read-only research history."""

import re
from pathlib import Path
import streamlit as st
from ..core import ROOT, read_json
from .components import page_header


def guide():
    page_header(
        "The field guide",
        "Guide",
        "A clear walkthrough for your team and your presentation.",
        "DEMO READY",
    )
    st.markdown(Path(__file__).with_name("guide.md").read_text(encoding="utf-8"))
    team = ROOT.parent / "docs/team-demo-script.md"
    if team.is_file():
        st.download_button(
            "Download demonstration script",
            team.read_bytes(),
            file_name=team.name,
            mime="text/markdown",
        )
        with st.expander("Four-person teammate demonstration"):
            text = re.sub(
                "\\[([^\\]]+)\\]\\(\\.\\./[^)]+\\)",
                "\\1",
                team.read_text(encoding="utf-8"),
            )
            st.markdown(text)
        results = ROOT.parent / "docs/history/reports/PRODUCT_PHASE3_RESULTS.md"
        if results.is_file():
            st.download_button(
                "Download Phase 3 results report",
                results.read_bytes(),
                file_name=results.name,
                mime="text/markdown",
            )


def archives():
    page_header(
        "Research library",
        "Archives — read-only reference",
        "Earlier studies, checkpoints and reports, preserved as reference material.",
        "READ ONLY",
    )
    root = ROOT.parent
    receipt = root / "archive/manifests/integrity.json"
    if receipt.is_file():
        r = read_json(receipt)
        st.success(
            f"Preservation verified: {r['file_versions']:,} unique file versions, {r['missing_files']} missing files, {r['missing_lfs']} missing LFS payloads."
        )
    st.write(
        "Historical runs, checkpoints, negative results, source versions and Phase 5A–5H reports are indexed in archive/INDEX.csv. Historical models and scientific scores are not loaded into this app registry."
    )
    st.code(
        "python scripts/export_history_archives.py --output-dir history-exports\npython scripts/export_history_archives.py --output-dir history-exports --verify",
        language="powershell",
    )
    st.caption(
        "Run from the repository root. ZIP exports stream to disk, preserve originals, include inventories and fail if data is missing. Choose a fresh output folder."
    )
    guide = root / "archive/README.md"
    if guide.is_file():
        with st.expander("Archive guide and export instructions"):
            st.markdown(guide.read_text(encoding="utf-8"))
    docs = list(
        (ROOT.parent / "docs/history/application").glob("partial_phase5*_results.md")
    )
    if docs:
        selected = st.selectbox(
            "Read an archived research report", [p.name for p in docs]
        )
        p = next((p for p in docs if p.name == selected))
        st.download_button(
            "Download historical report",
            p.read_bytes(),
            file_name=p.name,
            mime="text/markdown",
        )
        with st.expander("View historical text (not app results)"):
            st.markdown(p.read_text(encoding="utf-8"))
