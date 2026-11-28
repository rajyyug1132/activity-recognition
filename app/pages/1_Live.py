"""Live activity from the Phyphox app over Wi-Fi, or a replay of the demo recording (app/demo).
Phone setup: docs/live_demo.md. Run from the repo root: streamlit run app/streamlit_app.py

Controls on the left, the status board on the right. Presentation only: the polling and
prediction logic is unchanged (see DESIGN.md).
"""
import os
import sys
from pathlib import Path

import streamlit as st
import yaml

sys.path.insert(0, "src")
from inference import DummyModel
from live import LiveClassifier, PhyphoxSource, ReplaySource

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from components.ui import (
    board,
    coalesce,
    footer,
    load_css,
    notice,
    page_header,
    rail_mark,
    section,
    setup_steps,
    tape,
)

MODEL = "models/cnn_uci.keras"
# Real UCI HAR recording of test person 12 (never trained on): walking, down, up, sit, stand, lie
SAMPLE = Path(__file__).resolve().parents[1] / "demo" / "uci_test_person_12.csv"
HISTORY = 30  # predictions kept in the strip (one per second)
PHONE, REPLAY = "Phone", "Demo replay"

st.set_page_config(page_title="Live · Activity Recognition", page_icon=":material/sensors:", layout="wide")
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

page_header(
    "Live",
    "Wear your phone at the waist and see what you are doing, updated every second. "
    "No phone? Play the demo recording.",
)
rail, canvas = st.columns([1, 2.4], gap="large")

# ---------------------------------------------------------------------------
# Controls (left rail)
# ---------------------------------------------------------------------------
with rail:
    rail_mark()
    mode = st.segmented_control("Source", [PHONE, REPLAY], default=PHONE, required=True,
                                disabled=ss.live is not None)
    ss["_live_mode"] = mode  # the status chip reads it

    if mode == PHONE:
        ip = st.text_input("Phone address shown in Phyphox", key="phone_ip", placeholder="192.168.0.42:8080",
                           disabled=ss.live is not None)
    else:
        ip = ss.get("phone_ip", "")
        ss.phone_ip = ip  # keep what was typed while the field is hidden

    sample_available = SAMPLE.is_file()
    if mode == REPLAY and not sample_available:
        notice(
            "Replay is unavailable because app/demo/uci_test_person_12.csv is missing. "
            "Restore it from git, or choose Phone.",
            "warn",
        )

    if ss.live is None:
        if st.button("Start", type="primary", width="stretch", icon=":material/play_arrow:",
                     disabled=mode == REPLAY and not sample_available):
            try:
                src = PhyphoxSource(ip) if mode == PHONE else ReplaySource(SAMPLE)
                src.start()
                ss.live = (src, LiveClassifier(load_model(), cfg))
                ss.history = []
                ss.error = None
            except FileNotFoundError:
                if mode == REPLAY:
                    ss.error = (
                        "The replay sample became unavailable. Restore "
                        "app/demo/uci_test_person_12.csv from git, or choose Phone."
                    )
                else:
                    ss.error = "Could not start: the requested file was not found."
            except Exception as e:
                ss.error = f"Could not start: {e}"
            st.rerun()
    else:
        if st.button("Stop", width="stretch", icon=":material/stop:"):
            try:
                ss.live[0].stop()
            except Exception:
                pass
            ss.live = None
            st.rerun()

    if mode == PHONE and ss.live is None:
        section("Connect the phone")
        setup_steps([
            "Install Phyphox on your phone.",
            'Open the "HAR live (acc + gyro)" experiment (docs/har_live.phyphox). '
            "Built-in experiments lack the buffers the model needs.",
            "Enable Remote Access in the Phyphox menu.",
            "Type the IP address and port Phyphox shows into the field above.",
            "Press Start.",
        ])


# ---------------------------------------------------------------------------
# Status board (polls every second)
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

    running = ss.live is not None
    if running:
        chip = ("Phone connected" if ss.get("_live_mode") == PHONE else "Demo recording", "live")
    else:
        chip = ("Stopped" if ss.history else "Idle", "idle")

    if not ss.history:
        # Nothing measured yet: the board must not name an activity.
        board(
            None,
            headline_text="Listening…" if running else "Not running",
            hint="The first label appears after about 3 seconds." if running else "Press Start to begin.",
            chip=chip,
        )
        return

    act_key, conf = ss.history[-1]
    if ss.get("_last_act") != act_key:  # bump so the name re-plays its entrance
        ss["_pop"] = ss.get("_pop", 0) + 1
        ss["_last_act"] = act_key
    board(act_key, confidence=conf, chip=chip, pop=ss.get("_pop", 0))

    section("Recent activity", f"last {len(ss.history)} updates")
    tape(coalesce(a for a, _ in ss.history), step=1.0, axis=("earlier", "now"))


with canvas:
    panel()
    footer()
