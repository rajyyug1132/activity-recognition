"""Stairs direction (up vs down) for the camera model. Run from the repo root:
    python notebooks/09_stairs_direction.py

Two stages: the existing 3-class model (notebook 08) says walking / standing / stairs; a second
Random Forest, same settings, says up or down for windows called stairs. Stage 1 is not retrained.

Direction labels were set by eye from video frames (not from any model output). "front/back" clips
were filmed from the top of the flight, so up = walking toward the camera: a model could learn
"toward the camera" instead of direction, so side-view clips are scored separately as the real test.
A (3) is too short; A (5) and D (6) are side views whose direction could not be told: left out.
"""
import glob, os, sys
import joblib, numpy as np, yaml
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, "src")
from video_pose import MODEL, feature_windows, normalize_pose, to_windows

# keypoint file -> [(direction, start s, end s, view)]; video file in the comment
SEG = {"p1_stairs_05": [("upstairs", 0, 9.0, "front/back"), ("downstairs", 10.5, 20.0, "front/back")],  # A (8)
       "p3_stairs_01": [("upstairs", 0, 9.5, "front/back"), ("downstairs", 10.5, 19.0, "front/back")],  # C (5)
       "p4_stairs_02": [("upstairs", 0, 6.0, "front/back"), ("downstairs", 6.5, 12.56, "front/back")],  # D (7)
       "p2_stairs_02": [("downstairs", 0, 4.6, "front/back")],                                          # B (5)
       "p2_stairs_03": [("upstairs", 0, 4.0, "front/back")],                                            # B (7)
       "p1_stairs_03": [("upstairs", 0, 14.76, "side")],                                                # A (4)
       "p2_stairs_01": [("upstairs", 0, 10.52, "side")],                                                # B (4)
       "p1_stairs_01": [("downstairs", 0, 14.44, "side")]}                                              # A (1)
DIRS = ["upstairs", "downstairs"]
RF = lambda: RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)

cfg = yaml.safe_load(open("config/config.yaml"))
v, fps = cfg["video"], cfg["video"]["fps"]
X, y, sub, view = [], [], [], []
for name, segs in SEG.items():
    kp = np.load(f"data/video/keypoints/{name}.npz")["kp"]
    for d, s, e, vw in segs:
        w = to_windows(normalize_pose(kp[int(s * fps):int(e * fps)], v["max_gap"]), v["window_frames"], v["overlap"])
        X.append(w); y += [DIRS.index(d)] * len(w); sub += [name[:2]] * len(w); view += [vw] * len(w)
F, y, sub, view = feature_windows(np.concatenate(X), fps), np.array(y), np.array(sub), np.array(view)
print(f"{len(y)} stairs windows: {(y == 0).sum()} up, {(y == 1).sum()} down")

# leave one person out: direction accuracy on the held-out person's true stairs windows
right = np.zeros(len(y), bool)
for p in sorted(set(sub)):
    te = sub == p
    right[te] = RF().fit(F[~te], y[~te]).predict(F[te]) == y[te]
    print(f"{p}: {right[te].mean():.2f} of {te.sum()} windows")
for vw in ("front/back", "side"):
    print(f"{vw:10s}: {right[view == vw].mean():.2f} of {(view == vw).sum()} windows")
rng, null = np.random.default_rng(0), []
for _ in range(10):  # same test with training labels shuffled: what chance looks like here
    r = np.zeros(len(y), bool)
    for p in sorted(set(sub)):
        te = sub == p
        r[te] = RF().fit(F[~te], rng.permutation(y[~te])).predict(F[te]) == y[te]
    null.append(r.mean())
print(f"all: {right.mean():.2f} (training labels shuffled: {np.mean(null):.2f} ± {np.std(null):.2f})")

bundle = joblib.load(MODEL)
bundle["direction"] = {"model": RF().fit(F, y), "classes": DIRS}
joblib.dump(bundle, MODEL)
print(f"added the direction model to {MODEL}")
