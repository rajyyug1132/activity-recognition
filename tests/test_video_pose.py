import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from video_pose import normalize_pose, to_windows, load_video

CFG = {"video": {"fps": 25, "window_frames": 50, "overlap": 0.5, "max_gap": 5,
                  "classes": ["walking", "standing", "stairs"]}}


def pose(n=120, seed=0):
    """Random skeleton, shoulders 2 above hips (torso = 2 after shift), scaled/offset per frame."""
    kp = np.random.default_rng(seed).normal(size=(n, 33, 4)).astype(np.float32)
    kp[:, 23, :2] = [0.4, 0.6]; kp[:, 24, :2] = [0.6, 0.6]
    kp[:, 11, :2] = [0.4, 0.2]; kp[:, 12, :2] = [0.6, 0.2]   # torso length 0.4
    return kp


def test_normalize_centres_and_scales():
    out = normalize_pose(pose())
    assert out.shape == (120, 33, 2)
    assert np.allclose(out[:, [23, 24]].mean(1), 0, atol=1e-6)
    assert np.allclose(np.linalg.norm(out[:, [11, 12]].mean(1), axis=1), 1, atol=1e-6)


def test_gaps():
    kp = pose()
    kp[10:13] = np.nan      # short gap
    kp[40:50] = np.nan      # long gap
    out = normalize_pose(kp, max_gap=5)
    assert not np.isnan(out[10:13]).any()
    assert np.isnan(out[40:50]).all() and not np.isnan(out[:10]).any() and not np.isnan(out[50:]).any()


def test_windows():
    kp = np.zeros((120, 33, 2))
    w = to_windows(kp, 50, 0.5)
    assert w.shape == (3, 50, 66)       # starts 0, 25, 50 (70 would overflow)
    kp[60] = np.nan                     # hits windows starting at 25 and 50
    assert to_windows(kp, 50, 0.5).shape == (1, 50, 66)


def test_load_video():
    with tempfile.TemporaryDirectory() as d:
        np.savez(f"{d}/p1_walking_01.npz", kp=pose(), fps=25)
        np.savez(f"{d}/p4_stairs_02.npz", kp=pose(seed=1), fps=25)
        out = load_video(d, CFG)
        assert out["X"].shape[1:] == (50, 66) and len(out["X"]) == len(out["y"]) == len(out["subject"])
        assert (out["y"][out["subject"] == "p1"] == 0).all() and (out["y"][out["subject"] == "p4"] == 2).all()
        np.savez(f"{d}/p2_running_01.npz", kp=pose(), fps=25)
        try:
            load_video(d, CFG)
            raise AssertionError("bad file name accepted")
        except ValueError as e:
            assert "p2_running_01.npz" in str(e)


def test_pose_features():
    from video_pose import pose_features, FEATURE_NAMES
    t = np.arange(50) / 25
    still = normalize_pose(np.repeat(pose()[:1], 50, axis=0))[..., :].reshape(50, 66)
    walk = normalize_pose(np.repeat(pose()[:1], 50, axis=0)).copy()
    walk[:, 27, 1] += 0.3 * np.sin(2 * np.pi * 1.0 * t)          # left ankle swings at 1 Hz
    walk[:, 28, 1] -= 0.3 * np.sin(2 * np.pi * 1.0 * t)
    a, b = pose_features(still), pose_features(walk.reshape(50, 66))
    assert a.shape == b.shape == (len(FEATURE_NAMES),) and np.isfinite(b).all()
    i = FEATURE_NAMES.index
    assert b[i("spd_ankle")] > a[i("spd_ankle")] + 0.5
    assert abs(b[i("ankle_dy_peak_hz")] - 1.0) < 0.2 and b[i("ankle_dy_peak_frac")] > 0.5


def test_predict_live():
    from video_pose import predict_live

    class Echo:  # fake model: probability of class 0 = mean of the first feature, kept in 0-1
        def predict_proba(self, X):
            p = 1 / (1 + np.exp(-X[:, :1]))
            return np.hstack([p, 1 - p])

    bundle = {"model": Echo(), "classes": ["walking", "standing"]}
    cfg = {"video": dict(CFG["video"])}
    kp = pose(90)
    t30 = np.arange(90) / 30 + np.random.default_rng(1).uniform(0, 0.005, 90)  # jittery 30 fps webcam
    assert predict_live(t30[:40], kp[:40], bundle, cfg) is None      # 1.3 s: shorter than the 2 s window
    act, conf = predict_live(t30, kp, bundle, cfg)
    assert act in ("walking", "standing") and 0.5 <= conf <= 1
    kp[-40:] = np.nan                                                  # person left the frame
    assert predict_live(t30, kp, bundle, cfg) is None


def test_stairs_become_up_or_down():
    from video_pose import _classify

    class Fixed:  # fake model: always the given probabilities
        def __init__(self, p): self.p = np.array(p)
        def predict_proba(self, X): return np.tile(self.p, (len(X), 1))

    F = np.zeros((2, 23))
    base = {"model": Fixed([0.1, 0.1, 0.8]), "classes": ["walking", "standing", "stairs"]}
    act, conf = _classify(F, base)                       # old bundle: unchanged
    assert list(act) == ["stairs", "stairs"] and np.allclose(conf, 0.8)
    base["direction"] = {"model": Fixed([0.25, 0.75]), "classes": ["upstairs", "downstairs"]}
    act, conf = _classify(F, base)
    assert list(act) == ["downstairs", "downstairs"] and np.allclose(conf, 0.8 * 0.75)
    base["model"] = Fixed([0.7, 0.2, 0.1])               # not stairs: direction model not used
    act, conf = _classify(F, base)
    assert list(act) == ["walking", "walking"] and np.allclose(conf, 0.7)


def test_body_point_features_shape_and_windows():
    from video_pose import window_features
    rng = np.random.default_rng(3)
    kp = pose(120)
    kp[:, :, :3] = kp[:, :, :3] * 0.02 + kp[:, :, :3].mean(0) * 0 + rng.uniform(0.3, 0.7, (33, 3))  # plausible, varied points
    kp[:, 23, :2] = [0.45, 0.6]; kp[:, 24, :2] = [0.55, 0.6]; kp[:, 11, :2] = [0.45, 0.3]; kp[:, 12, :2] = [0.55, 0.3]
    s_old, F_old = window_features(kp, {}, CFG)
    s_new, F_new = window_features(kp, {"cols": 42}, CFG)
    assert s_old == s_new == [0, 25, 50]
    assert F_old.shape == (3, 23) and F_new.shape == (3, 42) and np.isfinite(F_new).all()
    kp[60:70] = np.nan                                     # 10-frame gap (> max_gap): windows touching it drop
    assert window_features(kp, {"cols": 42}, CFG)[0] == [0]
