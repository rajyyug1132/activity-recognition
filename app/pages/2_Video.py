"""Activity from a phone video: MediaPipe body keypoints -> hand-built features -> Random Forest.
Experimental: 3 classes, trained on 5 people. Run from the repo root: streamlit run app/streamlit_app.py
"""
import os
import sys
import tempfile

import streamlit as st
import yaml

sys.path.insert(0, "src")
from inference import smooth
from video_pose import MODEL, predict_video

# activity -> (bar colour, text on bar, display name); same hues as the phone pages
STYLE = {
    "walking": ("#34A853", "#FFFFFF", "Walking"),
    "standing": ("#E37400", "#1F2328", "Standing"),
    "stairs": ("#4285F4", "#FFFFFF", "Stairs"),
}

st.set_page_config(page_title="Video · Activity Recognition", page_icon=":material/videocam:", layout="centered")
st.markdown("""<style>
.block-container { padding-top: 1.5rem; max-width: 760px; }
.hero { border-radius: 24px; padding: 22px; margin: .5rem 0 1rem; }
.hero small { display: block; font-size: .85rem; opacity: .9; }
.hero h2 { color: inherit; font-family: Newsreader, Georgia, serif !important; font-weight: 300;
  font-size: clamp(1.8rem, 7vw, 2.6rem); line-height: 1.15; margin: .3rem 0; padding: 0; }
.tl { display: flex; height: 44px; border-radius: 12px; overflow: hidden; font-size: .8rem; font-weight: 600; }
.tl div { display: flex; align-items: center; justify-content: center; white-space: nowrap; overflow: hidden; }
.axis { display: flex; justify-content: space-between; font-size: .8rem; color: #4a4741; margin: 4px 2px 1rem; }
.note { font-size: .85rem; color: #4a4741; border-left: 3px solid #e1dcd3; padding-left: 10px; }
</style>""", unsafe_allow_html=True)


def mmss(sec):
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


@st.cache_resource(show_spinner="Loading the model…")
def load_bundle():
    import joblib
    return joblib.load(MODEL)


st.title("Video")
st.markdown('<p class="note"><b>Experimental.</b> Tells walking, standing and stairs apart from a phone video of one '
            'person. Trained on 5 people; on people it has not seen it gets about 6 windows in 10 right '
            '(macro F1 0.63). No sitting or lying. The video is processed on this computer and not kept.</p>',
            unsafe_allow_html=True)

if not os.path.exists(MODEL):
    st.error("No video model yet. Run notebooks/08_video_features.ipynb to create models/video_rf.joblib.")
    st.stop()

up = st.file_uploader("Upload a video (one person, whole body in frame)", type=["mp4", "mov"])
if not up:
    st.stop()

cfg = yaml.safe_load(open("config/config.yaml"))
key = (up.name, up.size)
if st.session_state.get("video_key") != key:
    suffix = os.path.splitext(up.name)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(up.getbuffer())
        path = f.name
    try:
        with st.spinner("Finding the body in each frame… (about as long as the video)"):
            st.session_state.video_pred = predict_video(path, load_bundle(), cfg)
    except Exception as e:  # unreadable video, missing pose model file
        st.error(f"Couldn't process this video ({type(e).__name__}: {e}).")
        st.stop()
    finally:
        os.remove(path)
    st.session_state.video_key = key

pred = st.session_state.video_pred
if pred.empty:
    st.warning("No full 2 s stretch with a person in frame. Use a longer clip with the whole body visible.")
    st.stop()
if st.sidebar.checkbox("Smooth predictions", value=True,
                       help=f"Majority vote over {cfg['smooth_window']} windows removes single-window flicker."):
    pred = smooth(pred, cfg["smooth_window"])

v = cfg["video"]
step = v["window_frames"] * (1 - v["overlap"]) / v["fps"]
dur = pred["start_time"].max() + v["window_frames"] / v["fps"]
share = pred["activity"].value_counts()
main = share.idxmax()
bg, fg, name = STYLE[main]
st.markdown(f'<div class="hero" style="background:{bg};color:{fg}"><small>{mmss(dur)} video · Random Forest, '
            f'hand-built pose features</small><h2>Mostly {name.lower()}</h2>'
            f'{share[main] / len(pred):.0%} of the time · {pred["confidence"].mean():.0%} average confidence</div>',
            unsafe_allow_html=True)

runs = pred.groupby((pred["activity"] != pred["activity"].shift()).cumsum())["activity"].agg(["first", "size"])
bars = "".join(f'<div style="flex:{n};background:{STYLE[a][0]};color:{STYLE[a][1]}">'
               f'{STYLE[a][2] if n / len(pred) > 0.2 else ""}</div>' for a, n in runs.itertuples(index=False))
label = "Timeline: " + ", then ".join(f"{STYLE[a][2]} {mmss(n * step)}" for a, n in runs.itertuples(index=False))
st.markdown(f'<div class="tl" role="img" aria-label="{label}">{bars}</div>'
            f'<div class="axis"><span>0:00</span><span>{mmss(dur / 2)}</span><span>{mmss(dur)}</span></div>',
            unsafe_allow_html=True)

st.video(up)
with st.expander("Per-window predictions"):
    st.dataframe(pred.assign(activity=pred["activity"].map(lambda a: STYLE[a][2])), hide_index=True, width="stretch",
                 column_config={
                     "start_time": st.column_config.NumberColumn("Start (s)", format="%.1f"),
                     "activity": st.column_config.TextColumn("Activity"),
                     "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0,
                                                                   format="%.2f")})
    st.download_button("Download predictions CSV", pred.to_csv(index=False), "video_predictions.csv", "text/csv")
