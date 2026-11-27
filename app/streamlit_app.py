"""Owner: Yashaswini. Run from the repo root: streamlit run app/streamlit_app.py

Mobile-first. Story tab: activity hero, timeline, ranked activities.
Details tab: config, validation, per-window predictions.

UI redesign: Essor-inspired dark theme (see docs/ui_redesign_design.md).
Presentation only — all data/inference logic is unchanged.
"""
import html as _html
import os
import sys

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, "src")
from inference import ACTS, DummyModel, predict_csv, smooth
from validation import validate_csv

# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from components.ui import (
    activity_hero,
    activity_card,
    empty_state,
    footer,
    load_css,
    mmss,
    section_label,
    status_chip,
    timeline_bar,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = "models/cnn_uci.keras"

# New colorblind-safe activity palette (bar_color, bar_text, display_name)
STYLE = {
    "WALKING":            ("#4ADE80", "#000000", "Walking"),
    "WALKING_UPSTAIRS":   ("#60A5FA", "#000000", "Walking upstairs"),
    "WALKING_DOWNSTAIRS": ("#3B82F6", "#ffffff", "Walking downstairs"),
    "SITTING":            ("#FBBF24", "#000000", "Sitting"),
    "STANDING":           ("#FB923C", "#000000", "Standing"),
    "LAYING":             ("#F87171", "#000000", "Lying"),
}

# ---------------------------------------------------------------------------
# Page setup — must come before any other st call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Activity Recognition",
    page_icon=":material/directions_run:",
    layout="centered",
)
load_css()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model…")
def load_model():
    if os.path.exists(MODEL):
        from tensorflow import keras
        return keras.models.load_model(MODEL), "1D-CNN (UCI HAR)"
    return DummyModel(), "dummy (always Standing)"


# ---------------------------------------------------------------------------
# Helpers (keep existing logic, only rename for clarity)
# ---------------------------------------------------------------------------
def listing(words):
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def headline_text(share):
    """'Mostly walking, with short stretches of standing.' from activity window counts."""
    main = share.idxmax()
    rest = [
        STYLE[a][2].lower()
        for a in share.sort_values(ascending=False).index
        if a != main and share[a] > 0
    ]
    s = f"Mostly <em>{STYLE[main][2].lower()}</em>"
    return s + (f", with short stretches of {listing(rest)}." if rest else ".")


# ---------------------------------------------------------------------------
# Sidebar: smoothing toggle lives here for every page
# ---------------------------------------------------------------------------
cfg = yaml.safe_load(open("config/config.yaml"))

st.sidebar.markdown(
    '<p class="section-label" style="margin-bottom:4px;">Smoothing</p>',
    unsafe_allow_html=True,
)
smooth_on = st.sidebar.checkbox(
    "Smooth predictions",
    value=True,
    help=f"Majority vote over {cfg['smooth_window']} windows removes single-window flicker.",
)
st.sidebar.caption(
    "On: short bursts are merged into the dominant activity. "
    "Off: raw prediction per window."
)
cfg["window_samples"] = st.sidebar.slider(
    "Window length (samples)",
    min_value=32,
    max_value=512,
    value=int(cfg["window_samples"]),
    step=32,
)
step_fraction = st.sidebar.slider(
    "Step size (fraction of window)",
    min_value=0.1,
    max_value=1.0,
    value=float(1 - cfg["overlap"]),
    step=0.05,
    help="Smaller steps create more overlap between windows.",
)
cfg["overlap"] = 1 - step_fraction

# ---------------------------------------------------------------------------
# File uploader (above tabs so it persists across both)
# ---------------------------------------------------------------------------
up = st.file_uploader("Upload a phone recording (CSV)", type="csv")
story, details = st.tabs(["Story", "Details"])

# Details controls (must render even with no file so layout doesn't jump)
with details:
    if cfg["window_samples"] != 128:
        st.warning(
            "The CNN was trained on 128-sample windows; "
            "other lengths may reduce prediction quality. Adjust the window "
            "and step in the sidebar."
        )

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
if not up:
    with story:
        empty_state(
            icon="+",
            heading="No recording yet",
            body=(
                "Drop a CSV above, or try the replay demo on the Live page. "
                "Columns: timestamp (s), acc_x/y/z (g), gyro_x/y/z (rad/s), 50 Hz or faster."
            ),
        )
        footer()
    st.stop()

# ---------------------------------------------------------------------------
# Parse & validate
# ---------------------------------------------------------------------------
try:
    df = pd.read_csv(up)
except Exception as e:
    story.error(
        f"Couldn't read this file as a CSV ({type(e).__name__}). "
        "Export it from the logging app as CSV and upload it again."
    )
    st.stop()

rep = validate_csv(df, cfg)
issues, warns = rep["issues"], rep.get("warnings", [])

with details:
    col_a, col_b = st.columns(2)
    col_a.metric("Sampling rate", f"{rep['est_hz']:.1f} Hz" if rep["est_hz"] else "n/a")
    col_b.metric("Status", "Valid" if rep["ok"] else "Rejected")
    for msg in issues:
        st.error(msg)
    for msg in warns:
        st.warning(msg)

if not rep["ok"]:
    story.error(
        "This file didn't pass the checks:\n\n"
        + "\n".join(f"- {m}" for m in issues)
        + "\n\nFix the recording or its conversion, then upload it again."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
model, model_name = load_model()
pred = predict_csv(df, model, cfg)
if smooth_on:
    pred = smooth(pred, cfg["smooth_window"])

if pred.empty:
    story.warning("Too short: record at least a few seconds so one full window fits.")
    st.stop()

step = cfg["window_samples"] * (1 - cfg["overlap"]) / cfg["sampling_rate_hz"]
dur = df["timestamp"].max() - df["timestamp"].min()
share = pred["activity"].value_counts().reindex(ACTS, fill_value=0)
main = share.idxmax()
main_conf = float(pred.loc[pred["activity"] == main, "confidence"].mean())
main_dur = mmss(share[main] * step)

# ---------------------------------------------------------------------------
# Story Tab
# ---------------------------------------------------------------------------
with story:
    # Hero — dominant activity
    activity_hero(
        activity_name=main,
        confidence=main_conf,
        duration=f"{mmss(dur)} recording",
        model=model_name,
    )

    st.markdown(
        f'<p class="story-copy">{headline_text(share)}</p>',
        unsafe_allow_html=True,
    )

    # Low-confidence warning
    if (pred["confidence"] < 0.6).mean() > 0.5:
        st.markdown(
            '<div class="info-card">The model is unsure about most of this recording. '
            "Check that the phone was at the waist and the units are g and rad/s. "
            "Per-window confidence is in the Details tab.</div>",
            unsafe_allow_html=True,
        )

    # Timeline
    section_label("Activity Timeline")
    runs_df = pred.groupby(
        (pred["activity"] != pred["activity"].shift()).cumsum()
    )["activity"].agg(["first", "size"])
    runs_list = [
        {"activity": row["first"], "count": int(row["size"])}
        for _, row in runs_df.iterrows()
    ]

    # Axis labels
    tl_label = "Timeline: " + ", then ".join(
        f"{STYLE[a][2]} {mmss(n * step)}" for a, n in runs_df.itertuples(index=False)
    )
    timeline_bar(runs_list, len(pred))
    st.caption(f"0:00 · {mmss(dur / 2)} · {mmss(dur)}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Time per activity + file checks
    left, right = st.columns([3, 2], gap="large")

    with left:
        section_label("Time per Activity")
        seen = share[share > 0].sort_values(ascending=False)
        for act, count in seen.items():
            pct = count / len(pred)
            bar_color, _, display_name = STYLE[act]
            escaped_name = _html.escape(display_name)
            escaped_time = _html.escape(mmss(count * step))
            st.markdown(
                f'<div style="margin-bottom:14px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">'
                f'<span style="font-size:.9rem;color:#F5F5F5;font-family:Inter,sans-serif;">{escaped_name}</span>'
                f'<span style="font-size:.9rem;font-weight:600;font-variant-numeric:tabular-nums;color:#F5F5F5;font-family:Inter,sans-serif;">{escaped_time}</span>'
                f'</div>'
                f'<div style="height:8px;border-radius:4px;background:#2A2B2D;overflow:hidden;">'
                f'<div style="width:{pct:.0%};height:100%;border-radius:4px;background:{bar_color};"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        unseen = [STYLE[a][2].lower() for a in ACTS if share[a] == 0]
        if unseen:
            st.caption(f"Not detected: {listing(unseen)}.")

    with right:
        section_label("File Checks")
        hz_text = f"{rep['est_hz']:.0f} Hz sampled"
        st.markdown(
            f'<div style="display:flex;flex-direction:column;gap:8px;">'
            f'<span style="font-size:.85rem;padding:6px 12px;border-radius:999px;'
            f'background:#17181A;border:1px solid #2A2B2D;color:#F5F5F5;'
            f'font-family:Inter,sans-serif;">{_html.escape(hz_text)}</span>',
            unsafe_allow_html=True,
        )
        if not issues:
            st.markdown(
                '<span style="font-size:.85rem;padding:6px 12px;border-radius:999px;'
                'background:#17181A;border:1px solid #2A2B2D;color:#4ADE80;'
                'font-family:Inter,sans-serif;">No issues found</span>',
                unsafe_allow_html=True,
            )
        for w in warns:
            escaped_w = _html.escape(w)
            st.markdown(
                f'<span style="font-size:.85rem;padding:6px 12px;border-radius:999px;'
                f'background:#2d2200;border:1px solid #FBBF24;color:#FBBF24;'
                f'font-family:Inter,sans-serif;">{escaped_w}</span>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    footer()

# ---------------------------------------------------------------------------
# Details Tab
# ---------------------------------------------------------------------------
low = int((pred["confidence"] < 0.6).sum())

with details:
    st.markdown("<br>", unsafe_allow_html=True)

    col_x, col_y = st.columns(2)
    col_x.metric("Windows", len(pred))
    col_y.metric("Low confidence (< 60%)", low)

    section_label("Confidence Over Time")
    st.caption(
        f"{model_name} · one row per "
        f"{cfg['window_samples'] / cfg['sampling_rate_hz']:.2f} s window, "
        f"every {step:.2f} s"
    )
    st.line_chart(
        pred.set_index("start_time")["confidence"],
        height=180,
        color="#D4FF00",
        x_label="time (s)",
        y_label="confidence",
    )

    section_label("Confidence by Activity")
    confidence_by_activity = (
        pred.groupby("activity")["confidence"]
        .mean()
        .reindex(ACTS)
        .dropna()
        .rename("Mean confidence")
        .to_frame()
    )
    st.bar_chart(confidence_by_activity, color="#D4FF00", y="Mean confidence")

    section_label("Activity Breakdown")
    breakdown_cols = st.columns(3)
    for i, act in enumerate(ACTS):
        count = int(share[act])
        pct = count / len(pred)
        with breakdown_cols[i % len(breakdown_cols)]:
            activity_card(
                [(STYLE[act][2], mmss(count * step))],
                sub=f"{pct:.0%} of this recording",
            )

    section_label("Per-Window Predictions")
    st.dataframe(
        pred.assign(activity=pred["activity"].map(lambda a: STYLE[a][2])),
        hide_index=True,
        use_container_width=True,
        height=320,
        column_config={
            "start_time": st.column_config.NumberColumn("Start (s)", format="%.2f"),
            "activity": st.column_config.TextColumn("Activity"),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )
    st.download_button(
        "Download predictions CSV",
        pred.to_csv(index=False),
        "predictions.csv",
        "text/csv",
        use_container_width=True,
        type="primary",
    )
    footer()
