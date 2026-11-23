"""Owner: Yashaswini."""
import numpy as np
import pandas as pd
from preprocess import to_windows, normalize

ACTS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]


class DummyModel:
    """Always predicts STANDING. Used until a trained model file exists."""
    def predict(self, X, **_):
        p = np.zeros((len(X), 6)); p[:, 4] = 1
        return p


def predict_csv(path, model, cfg) -> pd.DataFrame:
    """One row per window: start_time, activity, confidence.

    `path` can be a file path or a file-like object (Streamlit upload).
    `model.predict(X)` must return class probabilities [N, 6].
    """
    df = pd.read_csv(path) if not isinstance(path, pd.DataFrame) else path
    X = normalize(to_windows(df, cfg), cfg["norm_stats"])
    if len(X) == 0:
        return pd.DataFrame(columns=["start_time", "activity", "confidence"])
    p = model.predict(X, verbose=0) if hasattr(model, "compile") else model.predict(X)
    step = cfg["window_samples"] * (1 - cfg["overlap"]) / cfg["sampling_rate_hz"]
    return pd.DataFrame({"start_time": df["timestamp"].min() + np.arange(len(X)) * step,
                         "activity": [ACTS[i] for i in p.argmax(1)],
                         "confidence": p.max(1).round(3)})
