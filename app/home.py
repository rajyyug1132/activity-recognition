"""Owner: Yashaswini. Phone recording page; app/streamlit_app.py routes to it.

Controls on the left, the result on the right. Summary tab: what happened, in order.
Details tab: confidence, per-window predictions, download.

Presentation only: all data and inference logic is unchanged (see DESIGN.md).
"""
import os
import sys

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, "src")
from inference import ACTS, DummyModel, predict_csv, smooth
from validation import validate_csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from components.ui import (
    act,
    coalesce,
    confidence_chart,
    empty_state,
    facts,
    footer,
    headline,
    ledger,
    listing,
    load_css,
    mmss,
    notice,
    page_header,
    rail_mark,
    section,
    summarize,
    tape,
)

MODEL = "models/cnn_uci.keras"
DEMO = "app/demo/uci_test_person_12.csv"  # real UCI HAR test person, never trained on

st.set_page_config(page_title="Phone recording · Activity Recognition", page_icon=":material/upload_file:", layout="wide")
load_css()


@st.cache_resource(show_spinner="Loading the model…")
def load_model():
    if os.path.exists(MODEL):
        from tensorflow import keras
        return keras.models.load_model(MODEL), "1D-CNN (UCI HAR)"
    return DummyModel(), "dummy (always Standing)"


# ---------------------------------------------------------------------------
# Header and controls (left rail)
# ---------------------------------------------------------------------------
cfg = yaml.safe_load(open("config/config.yaml"))
page_header(
    "Phone recording",
    "Upload a recording from a phone worn at the waist. The model labels each 2.56-second window "
    "as walking, stairs, sitting, standing or lying.",
)
rail, canvas = st.columns([1, 2.4], gap="large")

with rail:
    rail_mark()
    up = st.file_uploader("Phone recording (CSV)", type="csv")
    if up is not None:
        st.session_state.demo = False
    elif not st.session_state.get("demo"):
        if st.button("Try the sample recording", type="primary", icon=":material/play_arrow:", width="stretch"):
            st.session_state.demo = True
            st.rerun()
        st.caption("Two minutes of a real person from the UCI HAR test set, whom the model never trained on.")
    else:
        notice("Showing the sample: UCI HAR test person 12, not used in training.")
        if st.button("Clear sample", width="stretch"):
            st.session_state.demo = False
            st.rerun()
    source = up if up is not None else (DEMO if st.session_state.get("demo") else None)

    with st.expander("Analysis settings"):
        smooth_on = st.checkbox(
            "Smooth predictions",
            value=True,
            help=f"Majority vote over {cfg['smooth_window']} windows removes single-window flicker.",
        )
        cfg["window_samples"] = st.slider(
            "Window length (samples)", min_value=32, max_value=512,
            value=int(cfg["window_samples"]), step=32,
            help="128 samples = 2.56 s at 50 Hz, what the model was trained on.",
        )
        step_fraction = st.slider(
            "Step size (fraction of window)", min_value=0.1, max_value=1.0,
            value=float(1 - cfg["overlap"]), step=0.05,
            help="0.5 = windows overlap by half. Smaller steps give more, overlapping windows.",
        )
        cfg["overlap"] = 1 - step_fraction
        if cfg["window_samples"] != 128:
            notice("The CNN was trained on 128-sample windows; other lengths may reduce prediction quality.", "warn")

# ---------------------------------------------------------------------------
# Result (right canvas)
# ---------------------------------------------------------------------------
with canvas:
    if source is None:
        empty_state(
            "No recording yet",
            "Upload a CSV from a phone worn at the waist, or try the sample recording. "
            "Columns: timestamp (s), acc_x/y/z (g), gyro_x/y/z (rad/s), 50 Hz or faster.",
        )
        footer()
        st.stop()

    try:
        df = pd.read_csv(source)
    except Exception as e:
        st.error(
            f"Couldn't read this file as a CSV ({type(e).__name__}). "
            "Export it from the logging app as CSV and upload it again."
        )
        st.stop()

    rep = validate_csv(df, cfg)
    issues, warns = rep["issues"], rep.get("warnings", [])
    if not rep["ok"]:
        st.error(
            "This file didn't pass the checks:\n\n"
            + "\n".join(f"- {m}" for m in issues)
            + "\n\nFix the recording or its conversion, then upload it again."
        )
        st.stop()

    model, model_name = load_model()
    pred = predict_csv(df, model, cfg)
    if smooth_on:
        pred = smooth(pred, cfg["smooth_window"])

    if pred.empty:
        st.warning("Too short: record at least a few seconds so one full window fits.")
        st.stop()

    step = cfg["window_samples"] * (1 - cfg["overlap"]) / cfg["sampling_rate_hz"]
    dur = df["timestamp"].max() - df["timestamp"].min()
    share = pred["activity"].value_counts().reindex(ACTS, fill_value=0)
    low = int((pred["confidence"] < 0.6).sum())
    hz = f"{rep['est_hz']:.0f} Hz" if rep["est_hz"] else "n/a"

    summary, details = st.tabs(["Summary", "Details"])

    with summary:
        title, sub = summarize(share.to_dict())
        headline(title, sub)

        if (pred["confidence"] < 0.6).mean() > 0.5:
            notice(
                "The model is unsure about most of this recording. Check that the phone was at the waist "
                "and the units are g and rad/s. Per-window confidence is in the Details tab.",
                "warn",
            )

        tape(coalesce(pred["activity"]), step=step, total=dur)

        left, right = st.columns([3, 2], gap="large")
        with left:
            section("Time per activity")
            seen = share[share > 0].sort_values(ascending=False)
            ledger((a, c / len(pred), mmss(c * step), f"{c / len(pred):.0%}") for a, c in seen.items())
            unseen = [act(a).label.lower() for a in ACTS if share[a] == 0]
            if unseen:
                st.caption(f"Not detected: {listing(unseen)}.")
        with right:
            section("This recording")
            facts(
                [
                    ("Length", mmss(dur)),
                    ("Sampling rate", hz),
                    ("Average confidence", f"{pred['confidence'].mean():.0%}"),
                    ("Model", model_name),
                ],
                col=True,
            )
            for w in warns:
                notice(w, "warn")
        footer()

    with details:
        facts(
            [
                ("Status", "Valid"),
                ("Windows", str(len(pred))),
                ("Low confidence (< 60%)", f"{low} of {len(pred)}"),
                ("Window / step", f"{cfg['window_samples'] / cfg['sampling_rate_hz']:.2f} s / {step:.2f} s"),
            ]
        )

        section("Confidence over time", model_name)
        confidence_chart(pred["start_time"], pred["confidence"])

        section("Confidence by activity")
        by_act = pred.groupby("activity")["confidence"].mean().reindex(ACTS).dropna()
        ledger((a, v, f"{v:.0%}", "") for a, v in by_act.items())

        section("Every window")
        st.dataframe(
            pred.assign(activity=pred["activity"].map(lambda a: act(a).label)),
            hide_index=True,
            width="stretch",
            height=320,
            column_config={
                "start_time": st.column_config.NumberColumn("Start (s)", format="%.2f"),
                "activity": st.column_config.TextColumn("Activity"),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )
        st.download_button("Download predictions CSV", pred.to_csv(index=False), "predictions.csv", "text/csv")
        footer()
