"""Owner: Yashaswini."""
import pandas as pd


def predict_csv(path, model, cfg) -> pd.DataFrame:
    """One row per window: start_time, activity, confidence."""
    raise NotImplementedError
