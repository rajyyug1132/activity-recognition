"""Camera pipeline: video -> MediaPipe Pose keypoints -> normalised windows for a Keras model."""
import argparse, csv, os, re, sys, time
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _cfg():
    with open(os.path.join(ROOT, "config", "config.yaml")) as f:
        return yaml.safe_load(f)


def extract_keypoints(video_path, fps=25) -> np.ndarray:
    """[frames, 33, 4] (x, y, z, visibility), resampled to `fps`; NaN rows where no person is found."""
    import cv2, mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    model = os.path.join(ROOT, _cfg()["video"]["pose_model"])
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=model),
                                        running_mode=vision.RunningMode.VIDEO)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_out = int(n_src / src_fps * fps)
    want = {min(round(k * src_fps / fps), n_src - 1): k for k in range(n_out)}  # source frame -> output index
    out = np.full((n_out, 33, 4), np.nan, np.float32)
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        for i in range(n_src):
            if i not in want:
                cap.grab()
                continue
            ok, bgr = cap.read()
            if not ok:
                break
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(img, int(want[i] * 1000 / fps))
            if res.pose_landmarks:
                out[want[i]] = [[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]]
    cap.release()
    return out


def normalize_pose(kp, max_gap=5) -> np.ndarray:
    """[frames, 33, 2]: x,y centred on hip midpoint, scaled by torso length; NaN gaps <= max_gap interpolated."""
    xy = np.asarray(kp, np.float64)[:, :, :2]
    hip = xy[:, [23, 24]].mean(1)
    torso = np.linalg.norm(xy[:, [11, 12]].mean(1) - hip, axis=1)
    torso[torso == 0] = np.nan
    out = (xy - hip[:, None]) / torso[:, None, None]
    bad = np.isnan(out).any((1, 2))
    good = np.flatnonzero(~bad)
    if len(good) == 0:
        return out
    flat = out.reshape(len(out), -1)
    t = np.arange(len(out))
    for c in range(flat.shape[1]):
        flat[bad, c] = np.interp(t[bad], good, flat[good, c], left=np.nan, right=np.nan)
    # re-blank gaps longer than max_gap
    i = 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j < len(bad) and bad[j]:
                j += 1
            if j - i > max_gap:
                flat[i:j] = np.nan
            i = j
        else:
            i += 1
    return flat.reshape(out.shape)


def to_windows(kp_norm, window, overlap) -> np.ndarray:
    """[N, window, 66]; windows containing NaN are dropped."""
    flat = np.asarray(kp_norm).reshape(len(kp_norm), -1)
    step = max(1, int(window * (1 - overlap)))
    w = [flat[s:s + window] for s in range(0, len(flat) - window + 1, step)]
    w = [x for x in w if not np.isnan(x).any()]
    return np.stack(w) if w else np.empty((0, window, flat.shape[1]))


def load_video(root, cfg) -> dict:
    """Read cached <person>_<activity>_<take>.npz keypoints from root -> {"X", "y", "subject"}."""
    v = cfg["video"]
    classes = v["classes"]
    pat = re.compile(rf"^([^_]+)_({'|'.join(classes)})_\d+\.npz$")
    X, y, subj = [], [], []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".npz"):
            continue
        m = pat.match(name)
        if not m:
            raise ValueError(f"{name}: expected <person>_<activity>_<take>.npz with activity in {classes}")
        kp = np.load(os.path.join(root, name))["kp"]
        w = to_windows(normalize_pose(kp, v["max_gap"]), v["window_frames"], v["overlap"])
        X.append(w)
        y += [classes.index(m.group(2))] * len(w)
        subj += [m.group(1)] * len(w)
    if not X:
        raise FileNotFoundError(f"no .npz files in {root}")
    return {"X": np.concatenate(X), "y": np.array(y, int), "subject": np.array(subj)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--labels")
    a = ap.parse_args()
    cfg = _cfg()["video"]
    fps, classes = cfg["fps"], cfg["classes"]
    os.makedirs(a.out_dir, exist_ok=True)
    if a.labels:
        takes, jobs = {}, []
        with open(a.labels, newline="") as f:
            for r in csv.DictReader(f):
                if r["activity"] not in classes:
                    print(f"skip {r['file']}: activity '{r['activity']}' not in {classes}")
                    continue
                k = (r["person"], r["activity"])
                takes[k] = takes.get(k, 0) + 1
                jobs.append((r["file"], f"{k[0]}_{k[1]}_{takes[k]:02d}"))
    else:
        jobs = [(f, os.path.splitext(f)[0]) for f in sorted(os.listdir(a.videos_dir))
                if f.lower().endswith((".mp4", ".mov"))]
    for f, name in jobs:
        dst = os.path.join(a.out_dir, name + ".npz")
        if os.path.exists(dst):
            print(f"exists, skip {dst}")
            continue
        np.savez(dst, kp=extract_keypoints(os.path.join(a.videos_dir, f), fps), fps=fps)
        print(f"{f} -> {dst}")


if __name__ == "__main__":
    main()


FEATURE_NAMES = (["spd_ankle", "spd_knee", "spd_wrist", "spd_nose", "spd_all", "spd_ankle_y"]
                 + [f"{s}_{k}" for s in ("ankle_dy", "ankle_dist", "knee_dy") for k in ("std", "peak_hz", "peak_frac")]
                 + ["knee_ang_mean", "knee_ang_std", "knee_ang_min", "knee_ang_range",
                    "ankle_y_range", "knee_y_mean", "knee_y_std", "torso_tilt"])


def _angle(a, b, c):
    """Angle at b (degrees) per frame, for points a-b-c [frames, 2]."""
    u, v = a - b, c - b
    cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _periodic(s, fps):
    """(std, dominant frequency in 0.5-4 Hz, share of the 0.5-4 Hz power at that peak) of a signal."""
    s = s - s.mean()
    f, a = np.fft.rfftfreq(len(s), 1 / fps), np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    m = (f >= 0.5) & (f <= 4)
    return s.std(), f[m][a[m].argmax()], a[m].max() / (a[m].sum() + 1e-9)


def pose_features(window, fps=25) -> np.ndarray:
    """Hand-built features of one normalised window [frames, 66] -> [len(FEATURE_NAMES)].

    Speeds are in torso lengths per second. Built to depend little on the camera angle: only
    speeds, vertical offsets, joint angles and regularity of the leg swing; no absolute x.
    """
    P = np.asarray(window, float).reshape(len(window), 33, 2)
    vel = np.diff(P, axis=0) * fps
    spd = lambda ids: np.linalg.norm(vel[:, ids], axis=2).mean()
    ank, kn = P[:, [27, 28]], P[:, [25, 26]]
    ang = np.stack([_angle(P[:, h], P[:, k], P[:, a]) for h, k, a in ((23, 25, 27), (24, 26, 28))], 1).mean(1)
    f = [spd([27, 28]), spd([25, 26]), spd([15, 16]), spd([0]), np.linalg.norm(vel, axis=2).mean(),
         np.abs(vel[:, [27, 28], 1]).mean()]
    for s in (ank[:, 0, 1] - ank[:, 1, 1], np.linalg.norm(ank[:, 0] - ank[:, 1], axis=1), kn[:, 0, 1] - kn[:, 1, 1]):
        f += list(_periodic(s, fps))
    sh = P[:, [11, 12]].mean(1)
    f += [ang.mean(), ang.std(), ang.min(), np.ptp(ang), np.ptp(ank[:, :, 1], axis=0).mean(),
          kn[:, :, 1].mean(), kn[:, :, 1].std(axis=0).mean(), np.abs(sh[:, 0]).mean()]
    return np.array(f)


def feature_windows(X, fps=25) -> np.ndarray:
    """[N, frames, 66] -> [N, len(FEATURE_NAMES)]."""
    return np.stack([pose_features(w, fps) for w in X])
