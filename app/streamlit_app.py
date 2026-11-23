"""Owner: Yashaswini. Run from the repo root: streamlit run app/streamlit_app.py

Story tab: one-line summary, timeline, time per activity (design B).
Details tab: config, validation cards, per-window predictions (design A).
"""
import os
import sys

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, "src")
from inference import ACTS, DummyModel, predict_csv
from validation import validate_csv

MODEL = "models/cnn_uci.keras"
# activity -> (bar colour, text on bar, tile background, tile label colour, display name)
STYLE = {
    "WALKING": ("#34A853", "#FFFFFF", "#E6F4EA", "#1E7E34", "Walking"),
    "WALKING_UPSTAIRS": ("#4285F4", "#FFFFFF", "#E8F0FE", "#1A5FD0", "Walking upstairs"),
    "WALKING_DOWNSTAIRS": ("#1A5FD0", "#FFFFFF", "#DCE6F9", "#0B3F99", "Walking downstairs"),
    "SITTING": ("#FBBC05", "#1F2328", "#FEF7E0", "#8A5A00", "Sitting"),
    "STANDING": ("#E37400", "#1F2328", "#FDEBD9", "#8A4500", "Standing"),
    "LAYING": ("#EA4335", "#FFFFFF", "#FCE8E6", "#B3261E", "Lying"),
}

st.set_page_config(page_title="Activity Recognition", layout="wide")
st.markdown("""<style>
.brand{display:flex;height:6px;margin-bottom:8px}.brand span{flex:1}
.tl{display:flex;gap:4px;height:110px}
.tl div{border-radius:12px;min-width:6px;display:flex;align-items:flex-end;padding:10px;
  box-sizing:border-box;overflow:hidden;font-size:13px;font-weight:600;white-space:nowrap}
.tile{border-radius:20px;padding:18px 20px;margin-bottom:16px}
.tile b{display:block;font-size:30px;color:#1F2328}
.tile small{color:#444C56}
</style>
<div class="brand"><span style="background:#4285F4"></span><span style="background:#EA4335"></span>
<span style="background:#FBBC05"></span><span style="background:#34A853"></span></div>""",
            unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if os.path.exists(MODEL):
        from tensorflow import keras
        return keras.models.load_model(MODEL), "1D-CNN (UCI)"
    return DummyModel(), "dummy (always Standing)"


def mmss(sec):
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def headline(share):
    """'Mostly walking, with short stretches of standing.' from activity shares."""
    main = share.idxmax()
    rest = [STYLE[a][4].lower() for a in share.index if a != main and share[a] > 0]
    s = f"Mostly {STYLE[main][4].lower()}"
    if rest:
        s += ", with short stretches of " + (" and ".join(rest) if len(rest) < 3
                                             else ", ".join(rest[:-1]) + " and " + rest[-1])
    return s + "."


cfg = yaml.safe_load(open("config/config.yaml"))
st.title("Activity Recognition")
up = st.file_uploader("Phone CSV: timestamp (s), acc_x..z (g), gyro_x..z (rad/s)", type="csv")
story, details = st.tabs(["Story", "Details"])

with details:
    c1, c2, _ = st.columns([1, 1, 2])
    cfg["window_samples"] = c1.number_input("Window length (samples)", 32, 512, cfg["window_samples"], 32)
    cfg["overlap"] = c2.slider("Overlap", 0.0, 0.9, float(cfg["overlap"]), 0.05)
    if cfg["window_samples"] != 128:
        st.warning("The CNN was trained on 128-sample windows.")

if not up:
    story.info("Upload a CSV to see what you were doing.")
    st.stop()

df = pd.read_csv(up)
rep = validate_csv(df, cfg)
issues, warns = rep["issues"], rep.get("warnings", [])

with details:
    m = st.columns(4)
    m[0].metric("Sampling rate", f"{rep['est_hz']:.1f} Hz" if rep["est_hz"] else "n/a")
    m[1].metric("Status", "Valid" if rep["ok"] else "Rejected")
    m[2].metric("Issues", len(issues))
    m[3].metric("Warnings", len(warns))
    for msg in issues:
        st.error(msg)
    for msg in warns:
        st.warning(msg)

if not rep["ok"]:
    story.error("This file didn't pass the checks. Open the Details tab to see why.")
    st.stop()

model, name = load_model()
pred = predict_csv(df, model, cfg)
if pred.empty:
    story.warning("Too short: need at least one full window of data.")
    st.stop()
step = cfg["window_samples"] * (1 - cfg["overlap"]) / cfg["sampling_rate_hz"]
dur = df["timestamp"].max() - df["timestamp"].min()
share = pred["activity"].value_counts().reindex(ACTS, fill_value=0)

with story:
    st.caption(f"{up.name} · {mmss(dur)} · {name}")
    st.header(headline(share))
    runs = pred.groupby((pred["activity"] != pred["activity"].shift()).cumsum())["activity"].agg(["first", "size"])
    bars = "".join(
        f'<div style="flex:{n};background:{STYLE[a][0]};color:{STYLE[a][1]}">'
        f'{STYLE[a][4] if n / len(pred) > 0.12 else ""}</div>' for a, n in runs.itertuples(index=False))
    st.markdown(f'<div class="tl" role="img" aria-label="Activity timeline">{bars}</div>', unsafe_allow_html=True)
    st.caption(f"0:00 → {mmss(dur)}")
    cols = st.columns(3)
    for i, a in enumerate(ACTS):
        bar, _, bg, ink, label = STYLE[a]
        cols[i % 3].markdown(f'<div class="tile" style="background:{bg}"><span style="color:{ink};font-weight:600">'
                             f'{label}</span><b>{mmss(share[a] * step)}</b><small>{share[a]} windows</small></div>',
                             unsafe_allow_html=True)
    checks = [f"✓ {rep['est_hz']:.0f} Hz, resampled to {cfg['sampling_rate_hz']} Hz"] + [f"! {w}" for w in warns]
    st.caption("File checks: " + " · ".join(checks))

with details:
    st.subheader(f"Predictions ({name})")
    st.bar_chart(share.rename(lambda a: STYLE[a][4]))
    st.dataframe(pred, hide_index=True, width="stretch")
    st.download_button("Download predictions CSV", pred.to_csv(index=False), "predictions.csv", "text/csv")
