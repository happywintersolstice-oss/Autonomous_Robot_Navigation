"""Saved results, explicit development comparisons, and verified exports."""

from pathlib import Path
import streamlit as st
from ..core import ROOT
from .. import phase3_ui
from ..app_registry import registry
from ..app_service import compare
from .components import LABELS, page_header


def comparison():
    page_header(
        "Results & evidence",
        "Compare & Export",
        "Compare the four robots on the same routes. Keep the evidence, including failures.",
        "SAVED RESULTS",
    )
    available = phase3_ui.SCOPES
    default = (
        0
        if (ROOT / "runs/phase3-publication.json").is_file()
        else 1 if (ROOT / "runs/phase3-selection.json").is_file() else 2
    )
    scope = st.selectbox("Evaluation scope", available, index=default)
    if st.session_state.get("comparison_scope") != scope:
        st.session_state.comparison_scope = scope
        st.session_state.comparison = phase3_ui.saved(scope)
    if st.button("Load saved comparison"):
        st.session_state.comparison = phase3_ui.saved(scope)
    held = scope == phase3_ui.SCOPES[0]
    read_only = held or scope == "Ensemble development"
    if held:
        st.info(
            "Frozen product held-out results are read-only. This page never reruns them or uses them to tune checkpoints. Phase3 may retain a Phase2 checkpoint when development selection favored it."
        )
    elif scope == "Ensemble development":
        st.info(
            "Registered development-only ensemble evidence. These cases were previously inspected; this is not unseen accuracy. Loading this result never reruns policies or changes the four selected checkpoints."
        )
    if not read_only:
        items = registry()
        selected = {}
        for algorithm in LABELS:
            candidates = [i for i in items if i["algorithm"] == algorithm]
            candidates.sort(
                key=lambda i: not (
                    i.get("phase3_selected")
                    if scope == phase3_ui.SCOPES[1]
                    else i.get("phase2_baseline")
                )
            )
            if not candidates:
                st.info(LABELS[algorithm] + " has no checkpoint")
                continue
            label = st.selectbox(
                LABELS[algorithm] + " checkpoint",
                [i["label"] for i in candidates],
                key="compare_" + scope + algorithm,
            )
            selected[algorithm] = next((i for i in candidates if i["label"] == label))
        include_random = st.checkbox("Include Random as an additional non-RL reference")
        include_ensemble = (
            st.checkbox("Include a separately labeled four-model voting ensemble")
            if scope == phase3_ui.SCOPES[1]
            else False
        )
        if st.button(
            "Run fair comparison", type="primary", disabled=len(selected) != 4
        ):
            try:
                with st.spinner(
                    "Explicit development evaluation; no model updates or held-out cases"
                ):
                    st.session_state.comparison = (
                        phase3_ui.development_compare(
                            selected, include_random, include_ensemble
                        )
                        if scope == phase3_ui.SCOPES[1]
                        else compare(selected, include_random=include_random)
                    )
            except Exception as exc:
                st.error("Comparison could not finish: " + str(exc))
    result = st.session_state.get("comparison")
    if not result:
        st.info("No saved result is available for this scope yet.")
        return
    manifest = result["manifest"]
    if manifest.get("ensemble"):
        with st.expander("How the ensemble was evaluated"):
            st.write(
                "Unweighted votes from the same four independently trained models. All members remember the actual executed actions. No action mask, planner, hidden events, fitted voting weights or new training is used. The individual models remain separately available."
            )
            st.json(manifest["ensemble"])
            st.json(manifest.get("public_history_sensitivity", {}))
    st.caption(manifest.get("purpose", "Recorded product comparison"))
    if manifest.get("holdout_suite_hash"):
        st.code(
            "Holdout: "
            + manifest["holdout_version"]
            + " | "
            + manifest["holdout_suite_hash"]
        )
    overview = [row for row in manifest["summary"] if row["Subset"] == "all"]
    if held:
        cards = st.columns(4)
        for col, algorithm in zip(cards, LABELS):
            result_row = next(
                (r for r in overview if r["Policy"] == "Phase3 " + algorithm), None
            )
            if result_row:
                col.metric(
                    LABELS[algorithm],
                    f"{result_row['Success %']:.2f}%",
                    help=f"{result_row['Successes']} successes from {result_row['Cases']} saved cases.",
                )
    st.dataframe(
        overview,
        column_order=[
            "Policy",
            "Cases",
            "Successes",
            "Success %",
            "Timeouts",
            "Wait %",
        ],
        width="stretch",
        hide_index=True,
    )
    import pandas as pd

    st.bar_chart(pd.DataFrame(overview).set_index("Policy")[["Success %"]])
    st.caption(
        "Counts keep their denominators. Successful-only Wait and step metrics exclude failed episodes. Empty values mean N/A. Observed alternatives are safe next cells, not a guaranteed whole detour or knowledge of obstacle permanence."
    )
    with st.expander("Condition, distance, topology and outcome breakdowns"):
        st.dataframe(manifest["summary"], width="stretch", hide_index=True)
    with st.expander("All case outcomes and checkpoint provenance"):
        st.dataframe(result["rows"], width="stretch", hide_index=True)
        st.json(manifest.get("training_provenance", {}))
    folder = Path(result["folder"])
    a, b, c, d = st.columns(4)
    a.download_button(
        "Summary CSV",
        (folder / "summary.csv").read_bytes(),
        file_name="summary.csv",
        mime="text/csv",
    )
    b.download_button(
        "Case-level CSV",
        (folder / "case-results.csv").read_bytes(),
        file_name="case-results.csv",
        mime="text/csv",
    )
    c.download_button(
        "Comparison JSON",
        (folder / "manifest.json").read_bytes(),
        file_name="comparison.json",
        mime="application/json",
    )
    d.download_button(
        "Results + all replays ZIP",
        (folder / "results-and-replays.zip").read_bytes(),
        file_name="results-and-replays.zip",
        mime="application/zip",
    )
    if (folder / "failure-traces.json").is_file():
        st.download_button(
            "Failure trace JSON",
            (folder / "failure-traces.json").read_bytes(),
            file_name="failure-traces.json",
            mime="application/json",
        )
    kind = st.radio(
        "Recorded journeys", ["Failures", "Successes", "All"], horizontal=True
    )
    rows = [
        r
        for r in result["rows"]
        if kind == "All" or (r["outcome"] == "success") == (kind == "Successes")
    ]
    if rows:
        labels = [
            f"{i}: {row['policy']} | {row['case_id']} | {row['outcome']} ({row['steps']} actions)"
            for i, row in enumerate(rows)
        ]
        chosen = st.selectbox("Choose a recorded journey", labels)
        row = rows[labels.index(chosen)]
        path = (folder / row["replay"]).resolve()
        if path.is_relative_to(folder.resolve()):
            st.button(
                "Open recorded journey in Simulator",
                on_click=phase3_ui.open_replay,
                args=(str(path),),
            )
