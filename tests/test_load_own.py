import os, sys, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import load_own

CFG = {"sampling_rate_hz": 50, "window_samples": 128, "overlap": 0.5,
       "noise_cutoff_hz": 20, "gravity_cutoff_hz": 0.3}


def rec(seconds=10, hz=100):
    t = np.arange(0, seconds, 1 / hz)
    return pd.DataFrame({"timestamp": t, "acc_x": 1 + 0.1 * np.sin(2 * np.pi * 2 * t), "acc_y": 0.0,
                         "acc_z": 0.0, "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0})


def test_load_own():
    with tempfile.TemporaryDirectory() as d:
        rec().to_csv(f"{d}/p1_walking_01.csv", index=False)
        rec().to_csv(f"{d}/p4_lying_01.csv", index=False)
        out = load_own(d, CFG)
        n = len(out["y"]) // 2
        assert out["X"].shape[1:] == (128, 9)
        assert set(out["y"]) == {0, 5} and set(out["subject"]) == {"p1", "p4"}
        assert (out["y"][out["subject"] == "p4"] == 5).all() and n > 0

        rec().to_csv(f"{d}/p2_running_01.csv", index=False)   # unknown activity -> refuse
        try:
            load_own(d, CFG)
            raise AssertionError("bad file name accepted")
        except ValueError as e:
            assert "p2_running_01.csv" in str(e)


if __name__ == "__main__":
    test_load_own()
    print("load_own checks passed")
