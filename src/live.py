"""Live classification from the Phyphox app (Wi-Fi remote access) or a CSV replay.

Both sources return new samples as a DataFrame in the app's input format:
timestamp (s), acc_x..z (g, incl. gravity), gyro_x..z (rad/s).
"""
import json
import time
import urllib.request

import numpy as np
import pandas as pd

from inference import ACTS
from preprocess import COLS, normalize, to_windows

G = 9.80665
ACC = ["accX", "accY", "accZ"]          # buffer names in docs/har_live.phyphox
GYR = ["gyrX", "gyrY", "gyrZ"]
# Phone axis (index, sign) feeding UCI x, y, z, applied to acc and gyro alike.
# UCI x is vertical; a phone upright at the waist has its own y vertical, so swap x/y
# and flip z to stay right-handed. ponytail: untested on a real phone; check with the
# standing test in docs/live_demo.md and edit here if acc_x is not ~ +1 g.
AXES = [(1, 1), (0, 1), (2, -1)]


def _cols(buf, names):
    """[n, len(names)] array; buffers caught mid-update are cut to the shortest, NaN rows dropped."""
    cols = [np.asarray(buf[b]["buffer"], float) for b in names]
    n = min(len(c) for c in cols)
    x = np.stack([c[:n] for c in cols], 1)
    return x[~np.isnan(x).any(1)]


class PhyphoxSource:
    """Polls http://<ip>:8080/get for samples newer than the last ones seen."""

    def __init__(self, address, timeout=2.0):
        self.base = "http://" + address.strip().removeprefix("http://").rstrip("/")
        if ":" not in self.base[7:]:
            self.base += ":8080"
        self.timeout = timeout
        self.t_acc = self.t_gyr = -1.0
        self.last_gyr = None  # last gyro sample, keeps interpolation continuous across polls

    def _call(self, path):
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as r:
            return json.loads(r.read())

    def start(self):
        # No cmd=clear: on a real phone it shut down phyphox's web server. Skip old samples by
        # starting the thresholds at the newest timestamps already in the buffers.
        buf = self._call("/get?acc_time&gyr_time")["buffer"]
        last = lambda b: (buf.get(b, {}).get("buffer") or [None])[-1]
        self.t_acc = last("acc_time") if last("acc_time") is not None else -1.0
        self.t_gyr = last("gyr_time") if last("gyr_time") is not None else -1.0
        self._call("/control?cmd=start")

    def stop(self):
        self._call("/control?cmd=stop")

    def read(self):
        ta, tg = f"{self.t_acc:.6f}", f"{self.t_gyr:.6f}"
        q = "&".join([f"{b}={ta}|acc_time" for b in ACC] + [f"acc_time={ta}"]
                     + [f"{b}={tg}|gyr_time" for b in GYR] + [f"gyr_time={tg}"])
        buf = self._call("/get?" + q.replace("|", "%7C"))["buffer"]  # phyphox answers 400 to a raw |
        a, g = _cols(buf, ["acc_time"] + ACC), _cols(buf, ["gyr_time"] + GYR)
        if len(g):
            self.t_gyr = g[-1, 0]
            g = g if self.last_gyr is None else np.vstack([self.last_gyr, g])
            self.last_gyr = g[-1:]
        elif self.last_gyr is not None:
            g = self.last_gyr
        if not len(a) or not len(g):
            return pd.DataFrame(columns=["timestamp"] + COLS)
        self.t_acc = a[-1, 0]
        # gyro has its own clock; resample it onto the accelerometer timestamps
        gi = np.stack([np.interp(a[:, 0], g[:, 0], g[:, i]) for i in (1, 2, 3)], 1)
        acc = np.stack([s * a[:, 1 + i] / G for i, s in AXES], 1)
        gyr = np.stack([s * gi[:, i] for i, s in AXES], 1)
        return pd.DataFrame(np.hstack([a[:, :1], acc, gyr]), columns=["timestamp"] + COLS)


class ReplaySource:
    """Plays a CSV back at real time, looping, so the page works without a phone."""

    def __init__(self, path, speed=1.0):
        self.df = pd.read_csv(path).sort_values("timestamp").reset_index(drop=True)
        self.t = self.df["timestamp"].to_numpy(float) - self.df["timestamp"].iloc[0]
        self.speed = speed
        self.t0 = None
        self.pos = 0

    def start(self):
        self.t0, self.pos = time.monotonic(), 0

    def stop(self):
        self.t0 = None

    def read(self):
        if self.t0 is None:
            return self.df.iloc[:0]
        now = (time.monotonic() - self.t0) * self.speed
        dur = self.t[-1] + 1e-3
        lap, rel = divmod(now, dur)
        end = int(np.searchsorted(self.t, rel, "right"))
        if self.pos > end:  # wrapped around: finish the old lap first
            out = self.df.iloc[self.pos:].assign(timestamp=self.t[self.pos:] + (lap - 1) * dur)
            self.pos = 0
            return pd.concat([out, self.read()], ignore_index=True)
        out = self.df.iloc[self.pos:end].assign(timestamp=self.t[self.pos:end] + lap * dur)
        self.pos = end
        return out


class LiveClassifier:
    """Rolling sample buffer; classifies the newest 128-sample window."""

    # ponytail: 5 s, not 2.56 s, so the 0.3 Hz gravity filter has some context
    def __init__(self, model, cfg, seconds=5.0):
        self.model, self.cfg, self.seconds = model, cfg, seconds
        self.buf = pd.DataFrame(columns=["timestamp"] + COLS)

    def push(self, new):
        if len(new):
            self.buf = new if self.buf.empty else pd.concat([self.buf, new], ignore_index=True)
            self.buf = self.buf[self.buf["timestamp"] > self.buf["timestamp"].iloc[-1] - self.seconds]

    def predict(self):
        """(activity, confidence, probs) for the newest window, or None if < 2.56 s buffered."""
        if len(self.buf) < 2:
            return None
        W = to_windows(self.buf, self.cfg)
        if not len(W):
            return None
        X = normalize(W[-1:], self.cfg["norm_stats"])
        p = self.model.predict(X, verbose=0) if hasattr(self.model, "compile") else self.model.predict(X)
        p = np.asarray(p)[0]
        return ACTS[int(p.argmax())], float(p.max()), p


if __name__ == "__main__":
    import yaml
    from inference import DummyModel
    cfg = yaml.safe_load(open("config/config.yaml"))
    src = ReplaySource("data/sample_synthetic.csv", speed=50)  # 90 s file in ~1.8 s
    src.start()
    clf = LiveClassifier(DummyModel(), cfg)
    assert clf.predict() is None
    n, last = 0, -1.0
    while n < len(src.df) * 1.5:  # cross the loop point once
        chunk = src.read()
        if len(chunk):
            assert chunk["timestamp"].iloc[0] > last, "timestamps must keep increasing"
            last = chunk["timestamp"].iloc[-1]
        n += len(chunk)
        clf.push(chunk)
        time.sleep(0.05)
    span = clf.buf["timestamp"].iloc[-1] - clf.buf["timestamp"].iloc[0]
    assert 4.5 < span <= 5.0, span
    assert clf.predict()[0] == "STANDING"
    print("ok", n, "samples replayed")
