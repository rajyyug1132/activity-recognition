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
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,300;1,6..72,300&display=swap');
html { touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
html, body, .stApp, h1, h2, h3, p, label, button, input, textarea { font-family: 'Hanken Grotesk', system-ui, sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 4rem; max-width: 760px; }
@media (min-width: 1024px) { .block-container { max-width: 1080px; } }
h3.sec { font-size: .75rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #4a4741; margin: 0 0 1rem; padding: 0; }
h1, h2, h3 { text-wrap: balance; letter-spacing: -0.01em; }
.stApp h1, .stApp h2, .stApp h3, [data-testid="stMarkdownContainer"] { font-family: 'Hanken Grotesk', system-ui, sans-serif !important; } /* beat Streamlit's heading font */
.brand { display: flex; gap: 6px; align-items: center; margin-bottom: .25rem; }
.brand i { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.brand b { font-size: 1.05rem; margin-left: 6px; color: #4a4741; font-weight: 600; }
.hero { border-radius: 24px; padding: 22px 22px 18px; color: #fff; margin: .5rem 0 1.25rem; }
.hero small { display: block; font-size: .85rem; opacity: .92; }
.hero h2 { color: #fff; font-family: Newsreader, Georgia, serif !important; font-size: clamp(1.9rem, 7vw, 2.8rem); font-weight: 300; line-height: 1.12; margin: .35rem 0 .5rem; padding: 0; }
.hero p { margin: 0; font-size: .95rem; }
.tl { display: flex; gap: 3px; height: 64px; border-radius: 14px; overflow: hidden; }
.tl div { min-width: 4px; display: flex; align-items: flex-end; padding: 8px; box-sizing: border-box;
  overflow: hidden; font-size: .8rem; font-weight: 600; white-space: nowrap; }
.tl { transform-origin: left center; animation: grow .5s cubic-bezier(.22, 1, .36, 1) both; }
@keyframes grow { from { transform: scaleX(.85); opacity: .35; } to { transform: none; opacity: 1; } }
@media (prefers-reduced-motion: reduce) { .tl { animation: none; } }
.axis { font-variant-numeric: tabular-nums; display: flex; justify-content: space-between; font-size: .8rem; color: #4a4741; margin: 6px 2px 1.5rem; }
.acts { display: flex; flex-direction: column; gap: 14px; margin-bottom: 1.25rem; }
.act { display: grid; grid-template-columns: 1fr auto; gap: 4px 12px; align-items: baseline; }
.act span { font-weight: 600; }
.act b { font-size: 1.35rem; font-weight: 600; font-variant-numeric: tabular-nums; color: #1c1b1a; }
.act .bar { grid-column: 1 / -1; height: 10px; border-radius: 5px; background: #e1dcd3; overflow: hidden; }
.act .bar i { display: block; height: 100%; border-radius: 5px; }
.none { font-size: .9rem; color: #4a4741; margin: 0 0 1rem; }
.checks { display: flex; flex-wrap: wrap; gap: 8px; }
.checks span { overflow-wrap: anywhere; font-size: .85rem; padding: 8px 14px; border-radius: 9999px; background: #fff; border: 1px solid #e1dcd3; color: #1c1b1a; }
.checks .warn { background: #f3e6c8; border-color: #ddbe78; color: #684d07; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: .25rem 0 1rem; }
.stats div { background: #fff; border: 1px solid #e1dcd3; border-radius: 24px; padding: 16px 18px; }
.stats small { display: block; font-size: .8rem; color: #4a4741; }
.stats b { font-size: 1.3rem; font-weight: 600; font-variant-numeric: tabular-nums; color: #1c1b1a; }
.unsure { background: #f3e6c8; color: #684d07; border-radius: 24px; padding: 16px 18px; font-size: .9rem; margin: -.5rem 0 1.25rem; }
.howto { background: #fff; border: 1px solid #e1dcd3; border-radius: 24px; padding: 22px 24px; }
hr, [data-baseweb=tab-border] { background-color: #e1dcd3 !important; }
.howto code { font-size: .85rem; }
</style>
<div class="brand"><span aria-hidden="true" style="display:contents"><i style="background:#4285F4"></i><i style="background:#EA4335"></i><i style="background:#FBBC05"></i><i style="background:#34A853"></i></span><b>Activity Recognition</b></div>""",
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
    s = f"Mostly <em>{STYLE[main][4].lower()}</em>"
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

try:
    df = pd.read_csv(up)
except Exception as e:  # not a CSV at all, or unreadable encoding
    story.error(f"Couldn't read this file as a CSV ({type(e).__name__}). Export it from the logging app as CSV and upload it again.")
    st.stop()
rep = validate_csv(df, cfg)
issues, warns = rep["issues"], rep.get("warnings", [])

def stats(pairs):
    """Compact label/value grid: 2 across on a phone, more on desktop."""
    cells = "".join(f'<div><small>{k}</small><b>{v}</b></div>' for k, v in pairs)
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


with details:
    stats([("Sampling rate", f"{rep['est_hz']:.1f} Hz" if rep["est_hz"] else "n/a"),
           ("Status", "Valid" if rep["ok"] else "Rejected")])
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
    if (pred["confidence"] < 0.6).mean() > 0.5:
        st.markdown('<p class="unsure">The model is unsure about most of this recording. Check that the phone '
                    'was at the waist and the units are g and rad/s; per-window confidence is in Details.</p>',
                    unsafe_allow_html=True)

    runs = pred.groupby((pred["activity"] != pred["activity"].shift()).cumsum())["activity"].agg(["first", "size"])
    bars = "".join(
        f'<div style="flex:{n};background:{STYLE[a][0]};color:{STYLE[a][1]}">'
        f'{STYLE[a][4] if n / len(pred) > 0.2 else ""}</div>'
        for a, n in runs.itertuples(index=False))
    tl_label = "Timeline: " + ", then ".join(f"{STYLE[a][4]} {mmss(n * step)}" for a, n in runs.itertuples(index=False))
    st.markdown(f'<div class="tl" role="img" aria-label="{tl_label}">{bars}</div>'
                f'<div class="axis"><span>0:00</span><span>{mmss(dur / 2)}</span><span>{mmss(dur)}</span></div>',
                unsafe_allow_html=True)

    # phone: stacked; desktop: activities beside the file checks
    left, right = st.columns([3, 2], gap="large")
    seen = share[share > 0].sort_values(ascending=False)
    rows = "".join(
        f'<div class="act"><span style="color:{STYLE[a][3]}">{STYLE[a][4]}</span><b>{mmss(n * step)}</b>'
        f'<div class="bar" aria-hidden="true"><i style="width:{n / len(pred):.0%};background:{STYLE[a][0]}"></i></div></div>'
        for a, n in seen.items())
    unseen = [STYLE[a][4].lower() for a in ACTS if share[a] == 0]
    left.markdown('<h3 class="sec">Time per Activity</h3>'
                  f'<div class="acts">{rows}</div>'
                  + (f'<p class="none">Not detected: {listing(unseen)}.</p>' if unseen else ""),
                  unsafe_allow_html=True)

    chips = [f"<span>{rep['est_hz']:.0f} Hz, resampled to {cfg['sampling_rate_hz']} Hz</span>",
             "<span>No missing values</span>" if not issues else ""] + \
            [f'<span class="warn">{w}</span>' for w in warns]
    right.markdown(f'<h3 class="sec">File Checks</h3><div class="checks">{"".join(chips)}</div>',
                   unsafe_allow_html=True)

low = int((pred["confidence"] < 0.6).sum())
with details:
    st.subheader("Predictions")
    st.caption(f"{name} · one row per {cfg['window_samples'] / cfg['sampling_rate_hz']:.2f} s window, "
               f"every {step:.2f} s")
    stats([("Windows", len(pred)), ("Unsure (under 60%)", low)])
    st.line_chart(pred.set_index("start_time")["confidence"], height=180, color="#4285F4",
                  x_label="time (s)", y_label="confidence")
    st.dataframe(pred.assign(activity=pred["activity"].map(lambda a: STYLE[a][4])), hide_index=True, width="stretch", height=320, column_config={
        "start_time": st.column_config.NumberColumn("Start (s)", format="%.2f"),
        "activity": st.column_config.TextColumn("Activity"),
        "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
    })
    st.download_button("Download predictions CSV", pred.to_csv(index=False), "predictions.csv", "text/csv",
                       width="stretch", type="primary")
