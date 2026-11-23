"""Owner: Yashaswini. Run from the repo root: streamlit run app/streamlit_app.py"""
import os
import sys
import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, "src")
from inference import ACTS, DummyModel, predict_csv
from validation import validate_csv

MODEL = "models/cnn_uci.keras"
cfg = yaml.safe_load(open("config/config.yaml"))

st.title("Activity Recognition")
st.sidebar.header("Config")
cfg["window_samples"] = st.sidebar.number_input("Window length (samples)", 32, 512, cfg["window_samples"], 32)
cfg["overlap"] = st.sidebar.slider("Overlap", 0.0, 0.9, float(cfg["overlap"]), 0.05)
if cfg["window_samples"] != 128:
    st.sidebar.warning("The CNN was trained on 128-sample windows.")


@st.cache_resource
def load_model():
    if os.path.exists(MODEL):
        from tensorflow import keras
        return keras.models.load_model(MODEL), "1D-CNN (UCI)"
    return DummyModel(), "dummy (always STANDING)"


up = st.file_uploader("Phone CSV: timestamp, acc_x..z (g), gyro_x..z (rad/s)", type="csv")
if not up:
    st.stop()
df = pd.read_csv(up)

rep = validate_csv(df, cfg)
st.subheader("Validation")
c1, c2 = st.columns(2)
c1.metric("Estimated rate", f"{rep['est_hz']:.1f} Hz" if rep["est_hz"] else "n/a")
c2.metric("Status", "OK" if rep["ok"] else "Rejected")
for m in rep["issues"]:
    st.error(m)
for m in rep.get("warnings", []):
    st.warning(m)
if not rep["ok"]:
    st.stop()

model, name = load_model()
pred = predict_csv(df, model, cfg)
st.subheader(f"Predictions ({name})")
pred["activity"] = pd.Categorical(pred["activity"], ACTS)
st.scatter_chart(pred.assign(code=pred["activity"].cat.codes), x="start_time", y="code", color="activity")
st.bar_chart(pred["activity"].value_counts().reindex(ACTS))
st.dataframe(pred, hide_index=True)
st.download_button("Download predictions CSV", pred.to_csv(index=False), "predictions.csv", "text/csv")
