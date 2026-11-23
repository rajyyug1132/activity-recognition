import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from validation import validate_csv

CFG = {"sampling_rate_hz": 50, "window_samples": 128}


def make(n=500, scale=1.0, hz=50):
    t = np.arange(n) / hz
    rng = np.random.default_rng(0)
    d = {"timestamp": t}
    for c in ["acc_x", "acc_y", "acc_z"]:
        d[c] = rng.normal(0, 0.05, n) + (1.0 if c == "acc_x" else 0.0)
    for c in ["gyro_x", "gyro_y", "gyro_z"]:
        d[c] = rng.normal(0, 0.1, n)
    df = pd.DataFrame(d)
    df[["acc_x", "acc_y", "acc_z"]] *= scale
    return df


def test_validation():
    r = validate_csv(make(), CFG)
    assert r["ok"] and abs(r["est_hz"] - 50) < 1, r

    bad = make(scale=9.80665)                       # m/s^2 not converted
    bad["timestamp"] = np.where(bad.index >= 250, bad["timestamp"] + 2.0, bad["timestamp"])  # 2 s gap
    r = validate_csv(bad, CFG)
    text = " ".join(r["issues"])
    assert not r["ok"] and "m/s^2" in text and "long gap" in text, r

    short = make()
    short["timestamp"] += (short.index >= 250) * 0.3  # one 0.3 s gap: warning, still ok
    r = validate_csv(short, CFG)
    assert r["ok"] and r["warnings"], r

    assert not validate_csv(make().drop(columns="gyro_z"), CFG)["ok"]
    assert not validate_csv(make(n=100), CFG)["ok"]  # 2 s < 2.56 s


if __name__ == "__main__":
    test_validation()
    print("validation checks passed")
