"""Owner: Nihal. Load the UCI HAR dataset in one call."""
import numpy as np

SIGNALS = ["body_acc_x", "body_acc_y", "body_acc_z",
           "body_gyro_x", "body_gyro_y", "body_gyro_z",
           "total_acc_x", "total_acc_y", "total_acc_z"]


def _raw(root, split):
    files = [f"{root}/{split}/Inertial Signals/{s}_{split}.txt" for s in SIGNALS]
    return np.stack([np.loadtxt(f) for f in files], axis=-1)   # [N, 128, 9]


def load_uci(root) -> dict:
    """Returns X_raw [N,128,9], X_feat [N,561], y [N] (0-5), subject [N] for train and test.

    Shape: {"train": {"X_raw", "X_feat", "y", "subject"}, "test": {...}}.
    `root` is the unzipped "UCI HAR Dataset" folder.
    """
    out = {}
    for split in ("train", "test"):
        out[split] = {
            "X_raw": _raw(root, split),
            "X_feat": np.loadtxt(f"{root}/{split}/X_{split}.txt"),
            "y": np.loadtxt(f"{root}/{split}/y_{split}.txt", dtype=int) - 1,
            "subject": np.loadtxt(f"{root}/{split}/subject_{split}.txt", dtype=int),
        }
    return out


def split_val(train, val_subjects):
    """Split the train dict by person (never by row). Returns (train, val)."""
    m = np.isin(train["subject"], val_subjects)
    return ({k: v[~m] for k, v in train.items()},
            {k: v[m] for k, v in train.items()})
