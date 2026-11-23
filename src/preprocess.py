"""Owner: Nihal. Turn continuous phone recordings into UCI-style windows."""
import numpy as np
from scipy.signal import butter, filtfilt, medfilt

COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def _lowpass(x, cutoff, fs):
    b, a = butter(3, cutoff, btype="low", fs=fs)
    return filtfilt(b, a, x, axis=0)


def to_windows(df, cfg) -> np.ndarray:
    """Resample to 50 Hz, filter, cut 128-sample windows at cfg overlap. Returns [N,128,9].

    Input columns: timestamp (s), acc_x/y/z (g, incl. gravity), gyro_x/y/z (rad/s).
    Channel order matches data_loader.SIGNALS: body_acc, body_gyro, total_acc.
    """
    fs, win = cfg["sampling_rate_hz"], cfg["window_samples"]
    step = int(win * (1 - cfg["overlap"]))
    df = df.dropna(subset=["timestamp"] + COLS).sort_values("timestamp")
    df = df.drop_duplicates("timestamp")
    t = df["timestamp"].to_numpy(float)
    grid = np.arange(t[0], t[-1], 1 / fs)
    if len(grid) < win:
        return np.empty((0, win, 9))
    # ponytail: linear interp also fills short gaps; validate_csv rejects long ones
    x = np.stack([np.interp(grid, t, df[c].to_numpy(float)) for c in COLS], axis=1)
    x = medfilt(x, kernel_size=(3, 1))
    x = _lowpass(x, cfg["noise_cutoff_hz"], fs)
    total, gyro = x[:, :3], x[:, 3:]
    body = total - _lowpass(total, cfg["gravity_cutoff_hz"], fs)
    sig = np.concatenate([body, gyro, total], axis=1)          # [T, 9]
    starts = range(0, len(sig) - win + 1, step)
    return np.stack([sig[s:s + win] for s in starts])


def fit_norm(X, path):
    """Per-channel mean/std from TRAINING windows only; saved for the app."""
    mean, std = X.mean(axis=(0, 1)), X.std(axis=(0, 1))
    np.savez(path, mean=mean, std=std)
    return mean, std


def normalize(X, path):
    s = np.load(path)
    return (X - s["mean"]) / s["std"]


if __name__ == "__main__":
    import pandas as pd
    cfg = {"sampling_rate_hz": 50, "window_samples": 128, "overlap": 0.5,
           "noise_cutoff_hz": 20, "gravity_cutoff_hz": 0.3}
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.uniform(0.009, 0.011, 6000))          # ~100 Hz, jittery, 60 s
    df = pd.DataFrame({"timestamp": t, "acc_x": 1 + 0.2 * np.sin(2 * np.pi * 2 * t),
                       "acc_y": 0.0, "acc_z": 0.0, "gyro_x": 0.0, "gyro_y": 0.0,
                       "gyro_z": 0.1 * np.sin(2 * np.pi * t)})
    W = to_windows(df, cfg)
    n = (int((t[-1] - t[0]) * 50) - 128) // 64 + 1
    assert W.shape == (n, 128, 9), W.shape
    mid = W[2:-2]                                               # skip filter edges
    assert abs(mid[..., 6].mean() - 1) < 0.02                   # total acc keeps 1 g
    assert abs(mid[..., 0].mean()) < 0.02                       # body acc has gravity removed
    assert to_windows(df.iloc[:100], cfg).shape == (0, 128, 9)  # < 2.56 s -> no windows
    print("ok", W.shape)
