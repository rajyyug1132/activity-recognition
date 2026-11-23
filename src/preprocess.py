"""Owner: Nihal."""
import numpy as np


def to_windows(df, cfg) -> np.ndarray:
    """Resample to 50 Hz, filter, cut 128-sample windows at cfg overlap. Returns [N,128,9]."""
    raise NotImplementedError
