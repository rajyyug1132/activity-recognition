"""Owner: Yashaswini. Run from the repo root: streamlit run app/streamlit_app.py

Mobile-first. Story tab: summary card, timeline, ranked activities.
Details tab: config, validation, per-window predictions.
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
# activity -> (bar colour, text on bar, soft background, deep shade for text/cards, display name)
STYLE = {
    "WALKING": ("#34A853", "#FFFFFF", "#E6F4EA", "#1E7E34", "Walking"),
    "WALKING_UPSTAIRS": ("#4285F4", "#FFFFFF", "#E8F0FE", "#1A5FD0", "Walking upstairs"),
    "WALKING_DOWNSTAIRS": ("#1A5FD0", "#FFFFFF", "#DCE6F9", "#0B3F99", "Walking downstairs"),
    "SITTING": ("#FBBC05", "#1F2328", "#FEF7E0", "#8A5A00", "Sitting"),
    "STANDING": ("#E37400", "#1F2328", "#FDEBD9", "#8A4500", "Standing"),
    "LAYING": ("#EA4335", "#FFFFFF", "#FCE8E6", "#B3261E", "Lying"),
}

st.set_page_config(page_title="Activity Recognition", page_icon=":material/directions_run:", layout="centered")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&display=swap');
html, body, .stApp, h1, h2, h3, p, label, button, input, textarea { font-family: Figtree, system-ui, sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 4rem; max-width: 760px; }
h1, h2, h3 { text-wrap: balance; letter-spacing: -0.01em; }
.brand { display: flex; gap: 6px; align-items: center; margin-bottom: .25rem; }
.brand i { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.brand b { font-size: 1.05rem; margin-left: 6px; color: #1F2328; }
.hero { border-radius: 24px; padding: 22px 22px 18px; color: #fff; margin: .5rem 0 1.25rem; }
.hero small { display: block; font-size: .85rem; opacity: .92; }
.hero h2 { color: #fff; font-size: clamp(1.6rem, 6vw, 2.3rem); font-weight: 800; line-height: 1.1; margin: .35rem 0 .5rem; padding: 0; }
.hero p { margin: 0; font-size: .95rem; }
.tl { display: flex; gap: 3px; height: 64px; border-radius: 14px; overflow: hidden; }
.tl div { min-width: 4px; display: flex; align-items: flex-end; padding: 8px; box-sizing: border-box;
  overflow: hidden; font-size: .8rem; font-weight: 600; white-space: nowrap;
  transform-origin: left; animation: grow .45s cubic-bezier(.22, 1, .36, 1) both; }
@keyframes grow { from { transform: scaleX(.6); opacity: .4; } to { transform: none; opacity: 1; } }
@media (prefers-reduced-motion: reduce) { .tl div { animation: none; } }
.axis { display: flex; justify-content: space-between; font-size: .8rem; color: #57606A; margin: 6px 2px 1.5rem; }
.acts { display: flex; flex-direction: column; gap: 14px; margin-bottom: 1.25rem; }
.act { display: grid; grid-template-columns: 1fr auto; gap: 4px 12px; align-items: baseline; }
.act span { font-weight: 600; }
.act b { font-size: 1.35rem; font-variant-numeric: tabular-nums; color: #1F2328; }
.act .bar { grid-column: 1 / -1; height: 10px; border-radius: 5px; background: #EEF1F5; overflow: hidden; }
.act .bar i { display: block; height: 100%; border-radius: 5px; }
.none { font-size: .9rem; color: #57606A; margin: 0 0 1rem; }
.checks { display: flex; flex-wrap: wrap; gap: 8px; }
.checks span { font-size: .85rem; padding: 6px 12px; border-radius: 999px; background: #EEF1F5; color: #1F2328; }
.checks .warn { background: #FEF7E0; color: #8A5A00; }
.howto { border: 1px solid #E3E7EE; border-radius: 18px; padding: 18px 20px; }
.howto code { font-size: .85rem; }
</style>
<div class="brand"><i style="background:#4285F4"></i><i style="background:#EA4335"></i><i style="background:#FBBC05"></i><i style="background:#34A853"></i><b>Activity Recognition</b></div>""",
            unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the model…")
def load_model():
    if os.path.exists(MODEL):
        from tensorflow import keras
        return keras.models.load_model(MODEL), "1D-CNN (UCI)"
    return DummyModel(), "dummy (always Standing)"


def mmss(sec):
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def listing(words):
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def headline(share):
    """'Mostly walking, with short stretches of standing.' from activity window counts."""
    main = share.idxmax()
    rest = [STYLE[a][4].lower() for a in share.sort_values(ascending=False).index if a != main and share[a] > 0]
    s = f"Mostly {STYLE[main][4].lower()}"
    return s + (f", with short stretches of {listing(rest)}." if rest else ".")


cfg = yaml.safe_load(open("config/config.yaml"))
up = st.file_uploader("Upload a phone recording (CSV)", type="csv")
story, details = st.tabs(["Story", "Details"])

with details:
    c1, c2 = st.columns(2)
    cfg["window_samples"] = c1.number_input("Window length (samples)", 32, 512, cfg["window_samples"], 32)
    cfg["overlap"] = c2.slider("Overlap", 0.0, 0.9, float(cfg["overlap"]), 0.05)
    if cfg["window_samples"] != 128:
        st.warning("The CNN was trained on 128-sample windows; other lengths will predict poorly.")

if not up:
    story.markdown("""<div class="howto"><p><b>See what you were doing, minute by minute.</b></p>
<p>Record with the phone at your waist (see the data collection protocol), export a CSV and upload it above.
Columns: <code>timestamp</code> (s), <code>acc_x..z</code> (g), <code>gyro_x..z</code> (rad/s), 50 Hz or faster.</p></div>""",
                   unsafe_allow_html=True)
    st.stop()

df = pd.read_csv(up)
rep = validate_csv(df, cfg)
issues, warns = rep["issues"], rep.get("warnings", [])

with details:
    m = st.columns(2)
    m[0].metric("Sampling rate", f"{rep['est_hz']:.1f} Hz" if rep["est_hz"] else "n/a")
    m[1].metric("Status", "Valid" if rep["ok"] else "Rejected")
    for msg in issues:
        st.error(msg)
    for msg in warns:
        st.warning(msg)

if not rep["ok"]:
    story.error("This file didn't pass the checks:\n\n" + "\n".join(f"- {m}" for m in issues)
                + "\n\nFix the recording or its conversion, then upload it again.")
    st.stop()

model, name = load_model()
pred = predict_csv(df, model, cfg)
if pred.empty:
    story.warning("Too short: record at least a few seconds so one full window fits.")
    st.stop()
step = cfg["window_samples"] * (1 - cfg["overlap"]) / cfg["sampling_rate_hz"]
dur = df["timestamp"].max() - df["timestamp"].min()
share = pred["activity"].value_counts().reindex(ACTS, fill_value=0)
main = share.idxmax()

with story:
    st.markdown(
        f'<div class="hero" style="background:{STYLE[main][3]}"><small>{mmss(dur)} recording · {name}</small>'
        f'<h2>{headline(share)}</h2><p>{STYLE[main][4]} for {mmss(share[main] * step)}, '
        f'{share[main] / len(pred):.0%} of the time.</p></div>', unsafe_allow_html=True)

    runs = pred.groupby((pred["activity"] != pred["activity"].shift()).cumsum())["activity"].agg(["first", "size"])
    bars = "".join(
        f'<div style="flex:{n};background:{STYLE[a][0]};color:{STYLE[a][1]};animation-delay:{i * 40}ms">'
        f'{STYLE[a][4] if n / len(pred) > 0.2 else ""}</div>'
        for i, (a, n) in enumerate(runs.itertuples(index=False)))
    st.markdown(f'<div class="tl" role="img" aria-label="Activity timeline">{bars}</div>'
                f'<div class="axis"><span>0:00</span><span>{mmss(dur / 2)}</span><span>{mmss(dur)}</span></div>',
                unsafe_allow_html=True)

    seen = share[share > 0].sort_values(ascending=False)
    rows = "".join(
        f'<div class="act"><span style="color:{STYLE[a][3]}">{STYLE[a][4]}</span><b>{mmss(n * step)}</b>'
        f'<div class="bar"><i style="width:{n / len(pred):.0%};background:{STYLE[a][0]}"></i></div></div>'
        for a, n in seen.items())
    unseen = [STYLE[a][4].lower() for a in ACTS if share[a] == 0]
    st.markdown(f'<div class="acts">{rows}</div>'
                + (f'<p class="none">Not detected: {listing(unseen)}.</p>' if unseen else ""),
                unsafe_allow_html=True)

    chips = [f"<span>{rep['est_hz']:.0f} Hz, resampled to {cfg['sampling_rate_hz']} Hz</span>",
             "<span>No missing values</span>" if not issues else ""] + \
            [f'<span class="warn">{w}</span>' for w in warns]
    st.markdown(f'<div class="checks">{"".join(chips)}</div>', unsafe_allow_html=True)

with details:
    st.subheader(f"Predictions ({name})")
    st.bar_chart(share.rename(lambda a: STYLE[a][4]), horizontal=True)
    st.dataframe(pred, hide_index=True, width="stretch")
    st.download_button("Download predictions CSV", pred.to_csv(index=False), "predictions.csv", "text/csv",
                       width="stretch")
