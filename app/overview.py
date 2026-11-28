"""Overview: what the project does, with the real sample recording as the centrepiece.

Presentation only. Motion is CSS (see theme.css) and respects prefers-reduced-motion.
"""
import html
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from components.ui import PHONE_ACCURACY, act, footer, icon, kinetic, load_css, section, tape_html

st.set_page_config(page_title="Overview · Activity Recognition", page_icon=":material/home:", layout="wide")
load_css()

SAMPLE = pathlib.Path(__file__).resolve().parent / "demo" / "uci_test_person_12.csv"
# Seconds per activity in the sample, as the phone model labels it (see the Phone recording page).
SEGMENTS = [("WALKING", 21), ("WALKING_DOWNSTAIRS", 19), ("WALKING_UPSTAIRS", 17),
            ("SITTING", 21), ("STANDING", 21), ("LAYING", 19)]
CAPTIONS = {
    "WALKING": "A steady beat, about two steps a second.",
    "WALKING_UPSTAIRS": "The same stride, lifting on every step.",
    "WALKING_DOWNSTAIRS": "A firmer landing on every step.",
    "SITTING": "Still, with the phone tipped back a little.",
    "STANDING": "Still, with the phone upright.",
    "LAYING": "Still, with gravity along another axis.",
}


def _md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_sample(points: int = 1200, width: int = 1200, height: int = 176) -> tuple[str, float] | None:
    """(SVG path of the sample's acceleration magnitude, length in seconds). 1 g sits mid-height.
    None if the file is missing."""
    try:
        df = pd.read_csv(SAMPLE, usecols=["timestamp", "acc_x", "acc_y", "acc_z"])
    except (OSError, ValueError):
        return None
    mag = np.linalg.norm(df[["acc_x", "acc_y", "acc_z"]].to_numpy(), axis=1)
    picks = np.linspace(0, len(mag) - 1, points).astype(int)
    y = np.clip(0.5 - (mag[picks] - 1.0) / 3.0, 0.02, 0.98) * height
    x = np.linspace(0, width, points)
    path = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    return path, float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])


def reel() -> str:
    slides = "".join(
        f'<div class="slide {act(k).cls}" style="--i:{i}" aria-hidden="true">{icon(act(k).icon)}'
        f'<div><span class="word"><span>{html.escape(act(k).short)}</span></span>'
        f'<p class="cap">{html.escape(CAPTIONS[k])}</p></div></div>'
        for i, k in enumerate(CAPTIONS))
    names = ", ".join(act(k).label.lower() for k in CAPTIONS)
    return f'<div class="reel" role="img" aria-label="The six activities the model recognises: {names}">{slides}</div>'


def scan_figure() -> str:
    sample = load_sample()
    if sample is None:
        return ""
    path, seconds = sample
    bands = "".join(f'<i class="{act(k).cls}" style="--w:{n}"></i>' for k, n in SEGMENTS)
    svg = ('<svg viewBox="0 0 1200 176" preserveAspectRatio="none" role="img" '
           'aria-label="Acceleration magnitude of the sample recording over two minutes">'
           f'<path class="scan-trace" d="{path}"/></svg>')
    return (f'<figure class="scan reveal" style="--dur:14s"><div class="scan-plot">'
            f'<div class="scan-sig"><div class="scan-bands" aria-hidden="true">{bands}</div>{svg}'
            f'<div class="scan-head" aria-hidden="true"></div></div>'
            f'{tape_html(SEGMENTS, step=1.0, total=seconds)}</div>'
            f'<figcaption>The line is the phone\'s acceleration, straight from the sample recording. '
            f'The colours are what the model made of it, 2.56 seconds at a time.</figcaption></figure>')


# ---------------------------------------------------------------------------
# Hero: kinetic headline on the left, the six activities wiping past on the right
# ---------------------------------------------------------------------------
copy, visual = st.columns([1.15, 1], gap="large", vertical_alignment="center")
with copy:
    headline_text = "Your phone can tell what you're doing."
    _md('<div class="hero-copy">'
        f'<h1 class="kinetic" aria-label="{html.escape(headline_text)}">{kinetic(headline_text)}</h1>'
        '<p>Wear it at your waist. A neural network reads the accelerometer and gyroscope and labels '
        'each 2.56-second window as walking, stairs, sitting, standing or lying.</p></div>')
    if st.button("Try the sample recording", type="primary", icon=":material/play_arrow:"):
        st.session_state.demo = True
        st.switch_page("home.py")
    st.page_link("pages/1_Live.py", label="Or watch it live", icon=":material/sensors:")
with visual:
    _md(reel())

# ---------------------------------------------------------------------------
# How it reads a recording
# ---------------------------------------------------------------------------
section("How it reads a recording", big=True)
_md(scan_figure())

# ---------------------------------------------------------------------------
# Proof, in one honest sentence
# ---------------------------------------------------------------------------
section("How well it works", big=True)
_md(f'<p class="proof reveal">Right <span class="num count" style="--to:{PHONE_ACCURACY}"></span>% of the time '
    'on <span class="num count" style="--to:9"></span> people it had never seen.</p>')
st.caption("Sitting and standing are the hardest pair to tell apart. The camera model is experimental; "
           "its numbers are on the Video page.")

# ---------------------------------------------------------------------------
# Three ways in
# ---------------------------------------------------------------------------
section("Three ways in", big=True)
for page, label, glyph, note in [
    ("home.py", "Phone recording", ":material/upload_file:",
     "Upload a CSV from a phone worn at the waist and get a labelled timeline."),
    ("pages/1_Live.py", "Live", ":material/sensors:",
     "Stream a phone over Wi-Fi with Phyphox, or replay the demo recording."),
    ("pages/2_Video.py", "Video", ":material/videocam:",
     "Upload a video or use the webcam. Experimental: walking, standing and stairs only."),
]:
    st.page_link(page, label=label, icon=glyph)
    _md(f'<p class="entry-note">{html.escape(note)}</p>')

_md('<p class="crew">Built by Nihal, Shravya, Yash, Yashaswini and Yyug.</p>')
footer()
