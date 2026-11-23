"""Owner: Shravya."""
import numpy as np

COLS = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
MAX_ACC_G, MAX_GYRO = 8.0, 35.0     # sane sensor limits (g, rad/s)
SHORT_GAP, LONG_GAP = 0.1, 1.0      # s: gaps > SHORT_GAP are counted, > LONG_GAP rejected
HZ_TOL = 0.10                       # allowed relative error vs cfg rate


def validate_csv(df, cfg) -> dict:
    """Checks columns, sampling rate, missing readings. Returns {ok, issues, est_hz, warnings}; short gaps are warnings only."""
    issues, warnings, est_hz = [], [], None
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        return {"ok": False, "est_hz": None, "issues": [f"missing columns: {missing}"]}

    n_nan = int(df[COLS].isna().sum().sum())
    if n_nan:
        issues.append(f"{n_nan} NaN values")
    d = df.dropna(subset=COLS)
    t = d["timestamp"].to_numpy(float)
    if len(t) < 2 or not np.all(np.diff(t) > 0):
        return {"ok": False, "est_hz": None, "issues": issues + ["too few rows or timestamps not increasing"]}

    gaps = np.diff(t)
    est_hz = float(1 / np.median(gaps))
    want = cfg["sampling_rate_hz"]
    if abs(est_hz - want) > HZ_TOL * want:
        issues.append(f"sampling rate {est_hz:.1f} Hz, expected ~{want} Hz")
    if np.percentile(gaps, 95) > 2 * np.median(gaps):  # ponytail: crude jitter test
        issues.append("uneven sampling (high gap spread)")
    n_gaps = int((gaps > SHORT_GAP).sum())
    if n_gaps:
        warnings.append(f"{n_gaps} time gaps > {SHORT_GAP}s (largest {gaps.max():.2f}s)")
    if gaps.max() > LONG_GAP:
        issues.append(f"long gap > {LONG_GAP}s: reject file")

    need = cfg["window_samples"] / want
    if t[-1] - t[0] < need:
        issues.append(f"only {t[-1] - t[0]:.2f}s of data, need >= {need:.2f}s")

    acc, gyro = d[["acc_x", "acc_y", "acc_z"]].to_numpy(float), d[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(float)
    if np.abs(acc).max() > MAX_ACC_G:
        issues.append(f"|acc| up to {np.abs(acc).max():.1f} > {MAX_ACC_G} g (units?)")
    if np.abs(gyro).max() > MAX_GYRO:
        issues.append(f"|gyro| up to {np.abs(gyro).max():.1f} > {MAX_GYRO} rad/s (deg/s?)")
    if abs(np.median(np.linalg.norm(acc, axis=1)) - 9.8) < 1.0:
        issues.append("median |acc| ~9.8: looks like m/s^2, divide by 9.80665")

    return {"ok": not issues, "est_hz": est_hz, "issues": issues, "warnings": warnings}
