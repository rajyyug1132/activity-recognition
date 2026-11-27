"""Live activity from the Phyphox app over Wi-Fi, or a replay of the sample CSV.
Phone setup: docs/live_demo.md. Run from the repo root: streamlit run app/streamlit_app.py

UI redesign: Essor-inspired dark theme (see docs/ui_redesign_design.md).
Presentation only — all data/inference logic is unchanged.
"""
import html as _html
import os
import sys
import time
from pathlib import Path

import streamlit as st
import yaml

sys.path.insert(0, "src")
from inference import DummyModel
from live import LiveClassifier, PhyphoxSource, ReplaySource

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from components.ui import (
    activity_hero,
    footer,
    load_css,
    section_label,
    setup_steps,
    status_chip,
)

MODEL = "models/cnn_uci.keras"
SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample_synthetic.csv"

# New colorblind-safe activity palette (bar_color, bar_text, display_name)
STYLE = {
    "WALKING":            ("#4ADE80", "#000000", "Walking"),
    "WALKING_UPSTAIRS":   ("#60A5FA", "#000000", "Walking upstairs"),
    "WALKING_DOWNSTAIRS": ("#3B82F6", "#ffffff", "Walking downstairs"),
    "SITTING":            ("#FBBF24", "#000000", "Sitting"),
    "STANDING":           ("#FB923C", "#000000", "Standing"),
    "LAYING":             ("#F87171", "#000000", "Lying"),
}
HISTORY = 30  # predictions kept in the strip (one per second)

st.set_page_config(
    page_title="Live · Activity Recognition",
    page_icon=":material/sensors:",
    layout="centered",
)
load_css()


@st.cache_resource(show_spinner="Loading the model…")
def load_model():
    if os.path.exists(MODEL):
        from tensorflow import keras
        return keras.models.load_model(MODEL)
    return DummyModel()


cfg = yaml.safe_load(open("config/config.yaml"))
ss = st.session_state
ss.setdefault("live", None)
ss.setdefault("history", [])
ss.setdefault("error", None)

# ---------------------------------------------------------------------------
# Header row: title + status chip
# ---------------------------------------------------------------------------
title_col, chip_col = st.columns([3, 1])
title_col.title("Live")

with chip_col:
    st.markdown("<div style='padding-top:14px;'>", unsafe_allow_html=True)
    if ss.live is not None:
        mode_val = st.session_state.get("_live_mode", "Replay mode")
        if "phone" in mode_val.lower():
            status_chip("Phone connected", "connected")
        else:
            status_chip("Replay mode", "replay")
    else:
        if ss.history:
            status_chip("Stopped", "idle")
        else:
            status_chip("Idle", "idle")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Source selector & controls
# ---------------------------------------------------------------------------
mode = st.radio(
    "Source",
    ["Phone (Phyphox)", "Replay sample"],
    horizontal=True,
    disabled=ss.live is not None,
)
# Store mode for status chip label
st.session_state["_live_mode"] = mode

ip = st.text_input(
    "Phone address shown in Phyphox",
    placeholder="192.168.0.42:8080",
    disabled=ss.live is not None or mode != "Phone (Phyphox)",
)

sample_available = SAMPLE.is_file()
if mode == "Replay sample" and not sample_available:
    st.warning(
        "Replay is unavailable because data/sample_synthetic.csv is missing. "
        "Add the replay CSV to the data folder, or choose Phone (Phyphox)."
    )

c1, c2 = st.columns(2)

if c1.button(
    "Start",
    type="primary",
    use_container_width=True,
    disabled=ss.live is not None or (mode == "Replay sample" and not sample_available),
):
    try:
        src = PhyphoxSource(ip) if mode.startswith("Phone") else ReplaySource(SAMPLE)
        src.start()
        ss.live = (src, LiveClassifier(load_model(), cfg))
        ss.history = []
        ss.error = None
    except FileNotFoundError:
        if mode == "Replay sample":
            ss.error = (
                "The replay sample became unavailable. Add "
                "data/sample_synthetic.csv to the data folder, or choose Phone (Phyphox)."
            )
        else:
            ss.error = "Could not start: the requested file was not found."
    except Exception as e:
        ss.error = f"Could not start: {e}"
    st.rerun()

if c2.button("Stop", use_container_width=True, disabled=ss.live is None):
    try:
        ss.live[0].stop()
    except Exception:
        pass
    ss.live = None
    st.rerun()

# ---------------------------------------------------------------------------
# Setup steps (shown when phone mode and not running)
# ---------------------------------------------------------------------------
if mode.startswith("Phone") and ss.live is None:
    setup_steps([
        "Install Phyphox on your phone.",
        'Open the "Acceleration without g" experiment (or the HAR live experiment).',
        "Enable Remote Access in the Phyphox menu.",
        "Type the IP address and port Phyphox shows into the field above.",
        "Press Start.",
    ])


# ---------------------------------------------------------------------------
# Live panel (polls every second)
# ---------------------------------------------------------------------------
@st.fragment(run_every=1.0)
def panel():
    if ss.live is not None:
        src, clf = ss.live
        try:
            clf.push(src.read())
            ss.error = None
        except Exception as e:
            ss.error = f"Lost contact with the phone: {e}"
        if (r := clf.predict()) is not None:
            ss.history = (ss.history + [r[:2]])[-HISTORY:]

    if ss.error:
        st.error(ss.error)

    if not ss.history:
        # Waiting / idle state
        waiting = ss.live is not None
        activity_hero(
            activity_name="STANDING",  # placeholder, shown as "—"
            confidence=None,
            duration=None,
            model=None,
        )
        # Override the placeholder with a waiting message
        msg = "Collecting the first 2.56 s of motion…" if waiting else "Press Start above."
        st.markdown(
            f'<div style="margin-top:-12px;padding:0 4px 16px;'
            f'font-size:.9rem;color:#9CA3AF;font-family:Inter,sans-serif;">{_html.escape(msg)}</div>',
            unsafe_allow_html=True,
        )
        return

    act, conf = ss.history[-1]
    state_note = "" if ss.live else " (stopped)"
    activity_hero(
        activity_name=act,
        confidence=conf,
        duration=state_note.strip() if state_note else None,
        model=None,
    )

    # Rolling strip
    section_label(f"Recent Activity — last {len(ss.history)} updates")
    cells = "".join(
        f'<div style="flex:1;background:{STYLE[a][0]};" '
        f'title="{_html.escape(STYLE[a][2])} {c:.0%}" aria-label="{_html.escape(STYLE[a][2])}"></div>'
        for a, c in ss.history
    )
    st.markdown(
        f'<div role="img" aria-label="Recent activities" '
        f'style="display:flex;gap:2px;height:36px;border-radius:8px;overflow:hidden;margin-bottom:4px;">'
        f'{cells}</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:.8rem;'
        f'color:#9CA3AF;font-family:Inter,sans-serif;margin-bottom:16px;">'
        f'<span>earlier</span><span>now</span></div>',
        unsafe_allow_html=True,
    )


panel()
footer()
