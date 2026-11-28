"""Activity from a phone video: MediaPipe body keypoints -> hand-built features -> Random Forest.
Experimental: 3 classes, trained on 5 people. Run from the repo root: streamlit run app/streamlit_app.py

UI redesign: Essor-inspired dark theme (see docs/ui_redesign_design.md).
Presentation only — all data/inference logic is unchanged.
"""
import html as _html
import os
import sys
import tempfile

import streamlit as st
import yaml

sys.path.insert(0, "src")
from inference import smooth
from video_pose import MODEL, predict_live, predict_video

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from components.ui import (
    activity_hero,
    empty_state,
    footer,
    info_card,
    load_css,
    mmss,
    section_label,
)

# Video activity palette — subset of the shared palette (walking, standing, stairs)
# Stairs maps to WALKING_UPSTAIRS colour for visual consistency
STYLE = {
    "walking":  ("#4ADE80", "#000000", "Walking"),
    "standing": ("#FB923C", "#000000", "Standing"),
    "stairs":   ("#60A5FA", "#000000", "Stairs"),
}

st.set_page_config(
    page_title="Video · Activity Recognition",
    page_icon=":material/videocam:",
    layout="centered",
)
load_css()

st.title("Video")

# Experimental note as info card
info_card(
    "Experimental. Tells walking, standing and stairs apart from a phone video "
    "of one person. Trained on 5 people; on unseen people it gets about 6 windows "
    "in 10 right (macro F1 0.63). No sitting or lying. "
    "The video is processed on this computer and not kept."
)

# ---------------------------------------------------------------------------
# Model availability check
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL):
    st.error(
        "No video model yet. Run notebooks/08_video_features.ipynb "
        "to create models/video_rf.joblib."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading the model…")
def load_bundle():
    import joblib
    return joblib.load(MODEL)


cfg = yaml.safe_load(open("config/config.yaml"))


def run_live(bundle, cfg):
    """Webcam -> pose per frame -> label from the last 2 s, refreshed every 0.5 s.
    Needs the app to run on the computer the camera is plugged into."""
    import time
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions, vision

    cam = st.number_input("Camera number", 0, 5, 0, help="0 is usually the built-in webcam.")
    c1, c2 = st.columns(2)
    if c1.button("Start", type="primary", use_container_width=True):
        st.session_state.cam_on = True
    if c2.button("Stop", use_container_width=True):
        st.session_state.cam_on = False
    if not st.session_state.get("cam_on"):
        empty_state(
            icon="[camera]",
            heading="Camera off",
            body=(
                "Stand 2 to 3 m from the camera with your whole body in frame, then press Start. "
                "The label appears after 2 s and updates twice a second."
            ),
        )
        return

    label, frame_box = st.empty(), st.empty()
    cap = cv2.VideoCapture(int(cam))
    if not cap.isOpened():
        st.session_state.cam_on = False
        st.error(f"Can't open camera {cam}. Try another number, or allow camera access for the terminal.")
        return
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=cfg["video"]["pose_model"]),
        running_mode=vision.RunningMode.VIDEO,
    )
    times, kps, t0, last, ms = [], [], time.time(), -1.0, -1
    try:  # Stop reruns the script, which ends this loop; finally frees the camera
        with vision.PoseLandmarker.create_from_options(opts) as lm:
            while st.session_state.get("cam_on"):
                ok, bgr = cap.read()
                if not ok:
                    st.session_state.cam_on = False
                    st.error("The camera stopped sending frames. Press Start to try again.")
                    break
                bgr = cv2.resize(bgr, (640, int(bgr.shape[0] * 640 / bgr.shape[1])))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                t = time.time() - t0
                ms = max(int(t * 1000), ms + 1)  # MediaPipe needs strictly increasing timestamps
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
                kp = np.full((33, 4), np.nan)
                if res.pose_landmarks:
                    kp = np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]])
                    h, w = rgb.shape[:2]
                    for x, y in kp[:, :2]:
                        cv2.circle(rgb, (int(x * w), int(y * h)), 4, (74, 222, 128), -1)
                times.append(t)
                kps.append(kp)
                while times[0] < t - 3:  # keep 3 s; the model reads the last 2 s
                    times.pop(0)
                    kps.pop(0)
                if t - last >= 0.5:
                    last = t
                    r = predict_live(times, np.array(kps), bundle, cfg)
                    with label.container():
                        if r:
                            activity_hero(r[0].title(), r[1], model="Random Forest · live camera")
                        else:
                            st.info("Looking for a whole body in frame for 2 s…")
                frame_box.image(rgb, channels="RGB", width="stretch")
    finally:
        cap.release()


mode = st.radio("Source", ["Upload a video", "Live camera"], horizontal=True)
if mode == "Live camera":
    run_live(load_bundle(), cfg)
    footer()
    st.stop()

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
up = st.file_uploader(
    "Upload a video (one person, whole body in frame)",
    type=["mp4", "mov"],
)

if not up:
    empty_state(
        icon="[video]",
        heading="No video yet",
        body=(
            "Upload an MP4 or MOV recorded from a phone. "
            "Keep the whole body in frame. Activities: walking, standing, stairs."
        ),
    )
    footer()
    st.stop()

# ---------------------------------------------------------------------------
# Inference (cached by filename + size)
# ---------------------------------------------------------------------------
key = (up.name, up.size)

if st.session_state.get("video_key") != key:
    suffix = os.path.splitext(up.name)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(up.getbuffer())
        path = f.name
    try:
        with st.spinner("Finding the body in each frame… (about as long as the video)"):
            st.session_state.video_pred = predict_video(path, load_bundle(), cfg)
    except Exception as e:
        st.error(f"Couldn't process this video ({type(e).__name__}: {e}).")
        st.stop()
    finally:
        os.remove(path)
    st.session_state.video_key = key

pred = st.session_state.video_pred

if pred.empty:
    st.warning(
        "No full 2 s stretch with a person in frame. "
        "Use a longer clip with the whole body visible."
    )
    footer()
    st.stop()

# Smoothing (sidebar toggle, same as main page)
if st.sidebar.checkbox(
    "Smooth predictions",
    value=True,
    help=f"Majority vote over {cfg['smooth_window']} windows removes single-window flicker.",
):
    pred = smooth(pred, cfg["smooth_window"])

# ---------------------------------------------------------------------------
# Compute stats
# ---------------------------------------------------------------------------
v = cfg["video"]
step = v["window_frames"] * (1 - v["overlap"]) / v["fps"]
dur = pred["start_time"].max() + v["window_frames"] / v["fps"]
share = pred["activity"].value_counts()
main = share.idxmax()
main_conf = float(pred.loc[pred["activity"] == main, "confidence"].mean())

share = share.reindex(STYLE, fill_value=0)
video_col, analysis_col = st.columns([1, 1], gap="large")

with video_col:
    section_label("Video")
    st.video(up)

with analysis_col:
    activity_hero(
        activity_name=main.title(),
        confidence=main_conf,
        duration=f"{mmss(dur)} video",
        model="Random Forest · pose features",
    )

    section_label("Activity Timeline")

    runs_df = pred.groupby(
        (pred["activity"] != pred["activity"].shift()).cumsum()
    )["activity"].agg(["first", "size"])

    bars_parts = []
    legend_seen: set[str] = set()
    legend_parts = []
    for _, row in runs_df.iterrows():
        act = row["first"]
        n = int(row["size"])
        pct = n / len(pred) * 100
        color, _, display_name = STYLE.get(act, ("#9CA3AF", "#000000", act))
        safe_name = _html.escape(display_name)
        bars_parts.append(
            f'<div style="width:{pct:.2f}%;background:{color};height:100%;" '
            f'title="{safe_name} ({pct:.1f}%)" aria-label="{safe_name} {pct:.1f}%"></div>'
        )
        if act not in legend_seen:
            legend_seen.add(act)
            legend_parts.append(
                f'<span style="display:inline-flex;align-items:center;gap:6px;'
                f'font-size:.75rem;color:#9CA3AF;font-family:Inter,sans-serif;">'
                f'<span style="width:10px;height:10px;border-radius:2px;background:{color};display:inline-block;"></span>'
                f'{safe_name}</span>'
            )

    tl_label = "Timeline: " + ", then ".join(
        f"{STYLE.get(a, ('', '', a))[2]} {mmss(n * step)}"
        for a, n in runs_df.itertuples(index=False)
    )
    st.markdown(
        f'<div style="display:flex;height:28px;border-radius:8px;overflow:hidden;gap:2px;" '
        f'role="img" aria-label="{_html.escape(tl_label)}">'
        + "".join(bars_parts)
        + "</div>"
        + f'<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;">{"".join(legend_parts)}</div>'
        + f'<div style="display:flex;justify-content:space-between;font-size:.8rem;'
        f'color:#9CA3AF;font-family:Inter,sans-serif;margin-top:4px;margin-bottom:16px;">'
        f"<span>0:00</span><span>{mmss(dur / 2)}</span><span>{mmss(dur)}</span></div>",
        unsafe_allow_html=True,
    )

    section_label("Activity Breakdown")
    breakdown_cols = st.columns(3)
    for i, (activity, count) in enumerate(share.items()):
        _, _, display_name = STYLE[activity]
        pct = count / len(pred)
        with breakdown_cols[i % len(breakdown_cols)]:
            st.markdown(
                f'<div class="dark-card">'
                f'<p class="card-label">{_html.escape(display_name)}</p>'
                f'<p class="card-value">{mmss(count * step)}</p>'
                f'<p class="card-sub">{pct:.0%} of this video</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Per-window table (collapsed)
# ---------------------------------------------------------------------------
with st.expander("Per-window predictions"):
    st.dataframe(
        pred.assign(activity=pred["activity"].map(lambda a: STYLE.get(a, ("", "", a))[2])),
        hide_index=True,
        use_container_width=True,
        column_config={
            "start_time": st.column_config.NumberColumn("Start (s)", format="%.1f"),
            "activity": st.column_config.TextColumn("Activity"),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )
    st.download_button(
        "Download predictions CSV",
        pred.to_csv(index=False),
        "video_predictions.csv",
        "text/csv",
    )

footer()
