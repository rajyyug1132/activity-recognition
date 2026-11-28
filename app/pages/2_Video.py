"""Activity from a phone video: MediaPipe body keypoints -> hand-built features -> Random Forest.
Experimental: 3 classes, trained on 5 people. Run from the repo root: streamlit run app/streamlit_app.py

The video and its controls on the left, the result on the right. Presentation only: pose
estimation and prediction logic are unchanged (see DESIGN.md).
"""
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
    act,
    board,
    coalesce,
    empty_state,
    facts,
    footer,
    headline,
    ledger,
    load_css,
    mmss,
    page_header,
    rail_mark,
    section,
    summarize,
    tape,
)

# Classes of the camera model. "stairs" only appears with a model saved before notebook 09.
VIDEO_CLASSES = ["walking", "standing", "upstairs", "downstairs"]
DOT_FILL, DOT_RING = (255, 255, 255), (20, 33, 61)  # pose points drawn on the webcam frame (RGB)

st.set_page_config(page_title="Video · Activity Recognition", page_icon=":material/videocam:", layout="wide")
load_css()

VIDEO_NOTE = ("Camera model: on a person it never saw, 64% of windows right for walking / standing / stairs "
              "(macro F1 0.65); stairs direction right 85% of the time.")
page_header(
    "Activity from video",
    "Upload a video of one person, or use your webcam, to see when they walk, stand, "
    "or walk up or down stairs.",
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


def run_live(bundle, cfg, canvas):
    """Webcam -> pose per frame -> label from the last 2 s, refreshed every 0.5 s.
    Needs the app to run on the computer the camera is plugged into.
    Controls render where this is called (the rail); the label and frame go to `canvas`."""
    import time
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions, vision

    cam = st.number_input("Camera number", 0, 5, 0, help="0 is usually the built-in webcam.")
    c1, c2 = st.columns(2)
    if c1.button("Start", type="primary", width="stretch"):
        st.session_state.cam_on = True
    if c2.button("Stop", width="stretch"):
        st.session_state.cam_on = False

    with canvas:
        if not st.session_state.get("cam_on"):
            empty_state(
                "Camera off",
                "Stand 2 to 3 m from the camera with your whole body in frame, then press Start. "
                "The label appears after 2 s and updates twice a second.",
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
                            cv2.circle(rgb, (int(x * w), int(y * h)), 5, DOT_FILL, -1)
                            cv2.circle(rgb, (int(x * w), int(y * h)), 5, DOT_RING, 1)
                    times.append(t)
                    kps.append(kp)
                    while times[0] < t - 3:  # keep 3 s; the model reads the last 2 s
                        times.pop(0)
                        kps.pop(0)
                    if t - last >= 0.5:
                        last = t
                        r = predict_live(times, np.array(kps), bundle, cfg, size=rgb.shape[1::-1])
                        with label.container():
                            chip = ("Live camera", "live")
                            if r:
                                board(r[0], confidence=r[1], chip=chip)
                            else:
                                board(None, headline_text="Looking for a body",
                                      hint="Keep your whole body in frame for 2 seconds.", chip=chip)
                    frame_box.image(rgb, channels="RGB", width="stretch")
        finally:
            cap.release()


rail, canvas = st.columns([1, 2.4], gap="large")

with rail:
    rail_mark()
    mode = st.segmented_control("Source", ["Upload", "Webcam"], default="Upload", required=True)
    if mode == "Webcam":
        run_live(load_bundle(), cfg, canvas)
        with canvas:
            footer(VIDEO_NOTE)
        st.stop()

    up = st.file_uploader("Upload a video (one person, whole body in frame)", type=["mp4", "mov"])
    with st.expander("How accurate is it? (experimental)"):
        st.markdown(
            "- Trained on videos of **5 people**. Tested on each person after training on the other four.\n"
            "- Reads the 33 MediaPipe body points: undistorted positions, a 3D body frame from the depth "
            "estimate (so the camera angle matters less), and how the whole body moves.\n"
            "- Walking / standing / stairs: **64%** of 2-second windows right, **macro F1 0.65** "
            "(random guessing: 0.33).\n"
            "- Stairs direction, up or down: right **85%** of the time (80% on side views, "
            "where facing the camera can't give it away). Shuffled labels score 50%.\n"
            "- No sitting or lying. The video is processed on this computer and not kept."
        )

if not up:
    with canvas:
        empty_state(
            "No video yet",
            "Upload an MP4 or MOV recorded from a phone. "
            "Keep the whole body in frame. Activities: walking, standing, walking up or down stairs.",
        )
        footer(VIDEO_NOTE)
    st.stop()

# ---------------------------------------------------------------------------
# Inference (cached by filename + size)
# ---------------------------------------------------------------------------
key = (up.name, up.size)

with canvas:
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
        footer(VIDEO_NOTE)
        st.stop()

with rail:
    st.video(up)
    # Smoothing toggle, same as the phone page
    smooth_on = st.checkbox(
        "Smooth predictions",
        value=True,
        help=f"Majority vote over {cfg['smooth_window']} windows removes single-window flicker.",
    )
if smooth_on:
    pred = smooth(pred, cfg["smooth_window"])

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
v = cfg["video"]
step = v["window_frames"] * (1 - v["overlap"]) / v["fps"]
dur = pred["start_time"].max() + v["window_frames"] / v["fps"]
share = pred["activity"].value_counts()
classes = VIDEO_CLASSES + (["stairs"] if "stairs" in share else [])  # model saved before notebook 09
share = share.reindex(classes, fill_value=0)

with canvas:
    title, sub = summarize(share.to_dict(), "video")
    headline(title, sub)
    tape(coalesce(pred["activity"]), step=step, total=dur)

    section("Time per activity")
    seen = share[share > 0].sort_values(ascending=False)
    ledger((a, c / len(pred), mmss(c * step), f"{c / len(pred):.0%}") for a, c in seen.items())

    facts([
        ("Length", mmss(dur)),
        ("Average confidence", f"{pred['confidence'].mean():.0%}"),
        ("Model", "Random Forest · pose features"),
    ])

    with st.expander("Per-window predictions"):
        st.dataframe(
            pred.assign(activity=pred["activity"].map(lambda a: act(a).label)),
            hide_index=True,
            width="stretch",
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

    footer(VIDEO_NOTE)
