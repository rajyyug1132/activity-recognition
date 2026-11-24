"""Live activity from the Phyphox app over Wi-Fi, or a replay of the sample CSV.
Phone setup: docs/live_demo.md. Run from the repo root: streamlit run app/streamlit_app.py
"""
import os
import sys
import time

import streamlit as st
import yaml

sys.path.insert(0, "src")
from inference import DummyModel
from live import LiveClassifier, PhyphoxSource, ReplaySource

MODEL, SAMPLE = "models/cnn_uci.keras", "data/sample_synthetic.csv"
# same palette as the main page: (colour, text colour, display name)
STYLE = {
    "WALKING": ("#34A853", "#FFFFFF", "Walking"),
    "WALKING_UPSTAIRS": ("#4285F4", "#FFFFFF", "Walking upstairs"),
    "WALKING_DOWNSTAIRS": ("#1A5FD0", "#FFFFFF", "Walking downstairs"),
    "SITTING": ("#FBBC05", "#1F2328", "Sitting"),
    "STANDING": ("#E37400", "#1F2328", "Standing"),
    "LAYING": ("#EA4335", "#FFFFFF", "Lying"),
}
HISTORY = 30  # predictions kept in the strip (one per second)

st.set_page_config(page_title="Live · Activity Recognition", page_icon=":material/sensors:", layout="centered")
st.markdown("""<style>
.block-container { padding-top: 1.5rem; max-width: 760px; }
.now { border-radius: 24px; padding: 22px; margin: .5rem 0 1rem; }
.now small { display: block; font-size: .85rem; opacity: .9; }
.now h2 { color: inherit; font-family: Newsreader, Georgia, serif !important; font-weight: 300;
  font-size: clamp(2rem, 8vw, 3rem); line-height: 1.1; margin: .3rem 0; padding: 0; }
.now.idle { background: #fff; border: 1px solid #e1dcd3; color: #4a4741; }
.strip { display: flex; gap: 3px; height: 40px; border-radius: 12px; overflow: hidden; }
.strip i { flex: 1; }
.axis { display: flex; justify-content: space-between; font-size: .8rem; color: #4a4741; margin: 4px 2px 1rem; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the model…")
def load_model():
    if os.path.exists(MODEL):
        from tensorflow import keras
        return keras.models.load_model(MODEL)
    return DummyModel()


cfg = yaml.safe_load(open("config/config.yaml"))
ss = st.session_state
ss.setdefault("live", None)  # (source, classifier) while running
ss.setdefault("history", [])
ss.setdefault("error", None)

st.title("Live")
mode = st.radio("Source", ["Phone (Phyphox)", "Replay sample"], horizontal=True, disabled=ss.live is not None)
ip = st.text_input("Phone address shown in Phyphox", placeholder="192.168.0.42:8080",
                   disabled=ss.live is not None or mode != "Phone (Phyphox)")
c1, c2 = st.columns(2)

if c1.button("Start", type="primary", use_container_width=True, disabled=ss.live is not None):
    try:
        src = PhyphoxSource(ip) if mode.startswith("Phone") else ReplaySource(SAMPLE)
        src.start()
        ss.live, ss.history, ss.error = (src, LiveClassifier(load_model(), cfg)), [], None
    except Exception as e:  # bad IP, phone unreachable, remote access off, missing CSV
        ss.error = f"Could not start: {e}"
    st.rerun()

if c2.button("Stop", use_container_width=True, disabled=ss.live is None):
    try:
        ss.live[0].stop()
    except Exception:
        pass  # phone already gone; nothing to undo
    ss.live = None
    st.rerun()

if mode.startswith("Phone") and ss.live is None:
    st.caption("Load the HAR live experiment in Phyphox, enable remote access, "
               "and type the address it shows. Setup steps: docs/live_demo.md.")


@st.fragment(run_every=1.0)
def panel():
    if ss.live is not None:
        src, clf = ss.live
        try:
            clf.push(src.read())
            ss.error = None
        except Exception as e:  # keep polling; Wi-Fi hiccups are common
            ss.error = f"Lost contact with the phone: {e}"
        if (r := clf.predict()) is not None:
            ss.history = (ss.history + [r[:2]])[-HISTORY:]

    if ss.error:
        st.error(ss.error)
    if not ss.history:
        msg = "Collecting the first 2.56 s of motion…" if ss.live else "Press Start."
        st.markdown(f'<div class="now idle"><small>Now</small><h2>—</h2>{msg}</div>', unsafe_allow_html=True)
        return
    act, conf = ss.history[-1]
    bg, fg, name = STYLE[act]
    state = "" if ss.live else " (stopped)"
    st.markdown(f'<div class="now" style="background:{bg};color:{fg}"><small>Now{state}</small>'
                f'<h2>{name}</h2>{conf:.0%} confident</div>', unsafe_allow_html=True)
    cells = "".join(f'<i style="background:{STYLE[a][0]}" title="{STYLE[a][2]} {c:.0%}"></i>'
                    for a, c in ss.history)
    st.markdown(f'<div class="strip" role="img" aria-label="Recent activities">{cells}</div>'
                f'<div class="axis"><span>last {len(ss.history)} updates</span><span>now</span></div>',
                unsafe_allow_html=True)


panel()
