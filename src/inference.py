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


def smooth(pred_df, k=5) -> pd.DataFrame:
    """Centered rolling majority vote over `activity` (odd k, edges use a shorter window).

    Ties keep the original label if it is among the leaders. Returns a copy.
    """
    assert k % 2 == 1, "k must be odd"
    out, acts, h = pred_df.copy(), list(pred_df["activity"]), k // 2
    for i, orig in enumerate(acts):
        win = acts[max(0, i - h): i + h + 1]
        counts = {a: win.count(a) for a in set(win)}
        top = max(counts.values())
        if counts[orig] < top:
            out.iloc[i, out.columns.get_loc("activity")] = next(a for a in win if counts[a] == top)
    return out
