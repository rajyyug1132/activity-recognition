import os, sys
from urllib.parse import parse_qsl
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from live import PhyphoxSource


class FakePhone(PhyphoxSource):
    """Answers /get like phyphox: values after the first reference-time above the threshold."""
    def __init__(self):
        super().__init__("192.168.0.42")
        t = np.arange(0, 2, 0.01)
        self.data = {"acc_time": t, "accX": 0 * t, "accY": np.full_like(t, 9.80665), "accZ": 0 * t,
                     "gyr_time": t + 0.005, "gyrX": 0 * t, "gyrY": t, "gyrZ": 0 * t}
        self.now = 1.0  # phone has measured this many seconds so far

    def _call(self, path):
        assert path.startswith("/get?")
        out = {}
        for name, v in parse_qsl(path[5:]):
            thr, _, ref = v.partition("|")
            tref = self.data[ref or name]
            keep = (tref > float(thr)) & (tref <= self.now)
            out[name] = {"buffer": self.data[name][keep].tolist()}
        return {"buffer": out}


def test_polls_incrementally_and_converts_units():
    p = FakePhone()
    assert p.base == "http://192.168.0.42:8080"
    a = p.read()
    assert len(a) == 101 and np.allclose(a["acc_x"], 1.0)                 # m/s² -> g
    p.now = 2.0
    b = p.read()
    assert b["timestamp"].iloc[0] > a["timestamp"].iloc[-1]              # no repeats
    assert np.allclose(b["gyro_x"], b["timestamp"] - 0.005, atol=1e-6)   # gyro interpolated onto acc clock
