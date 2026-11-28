"""Camera model on better use of the body points. Run from the repo root:
    python notebooks/10_body_points.py

Candidates, fixed before any test result:
  F0  the 23 features of notebook 08 (today's model)
  F1  the same 23 on undistorted points (MediaPipe x is in widths, y in heights; videos are portrait)
  F2  F1 + 15 features in a 3D body frame (x, y, depth z; torso = vertical axis): knee and hip angles,
      foot and step height, so they do not depend on the camera angle
  F3  F2 + 4 whole-body movement features (travel and approach speed through the frame)
each with a Random Forest (notebook 08 settings) or gradient boosting.

Honest score: nested leave-one-person-out. For each test person, the candidate is chosen using only
the other people, then scored once on the test person. p5 (one clip) is training-only.
"""
import glob, os, sys
import joblib, numpy as np, yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, "src")
from video_pose import MODEL, TRAIN_SIZE, window_features

cfg = yaml.safe_load(open("config/config.yaml"))
C3, DIRS = cfg["video"]["classes"], ["upstairs", "downstairs"]
src = open("notebooks/09_stairs_direction.py", encoding="utf-8").read()
SEG = eval(src[src.index("SEG = ") + 6: src.index("DIRS = ")].strip())   # direction labels, set by eye
FPS = cfg["video"]["fps"]
SETS = {"F0": None, "F1": 23, "F2": 38, "F3": 42}
MODELS = {"RF": lambda: RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1),
          "HGB": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=42)}


def feats(kp, cols):
    return window_features(kp, {"cols": cols}, cfg, TRAIN_SIZE)[1]


# stage 1 data: every window of every clip, 3 classes
F = {k: [] for k in SETS}; y, who = [], []
for f in sorted(glob.glob("data/video/keypoints/*.npz")):
    p, act, _ = os.path.basename(f)[:-4].split("_")
    kp = np.load(f)["kp"]
    for k, c in SETS.items():
        X = feats(kp, c)
        if X is not None: F[k].append(X)
    n = len(window_features(kp, {}, cfg)[0])
    assert all(len(window_features(kp, {"cols": c}, cfg)[0]) == n for c in (23,)), "window sets differ"
    y += [C3.index(act)] * n; who += [p] * n
F = {k: np.concatenate(v) for k, v in F.items()}; y, who = np.array(y), np.array(who)
TEST = ["p1", "p2", "p3", "p4"]
print(f"{len(y)} windows: {dict(zip(C3, np.bincount(y)))}")


def lopo(X, mk, people, mask):
    P = np.full(len(y), -1)
    for t in people:
        tr, te = mask & (who != t), mask & (who == t)
        P[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    return P


def score(P, people):
    a = [accuracy_score(y[who == t], P[who == t]) for t in people]
    f = [f1_score(y[who == t], P[who == t], average="macro", labels=range(3)) for t in people]
    return np.mean(a), np.std(a), np.mean(f), a


print("\nplain leave-one-person-out (choosing the best of these afterwards would be optimistic):")
plain = {}
for k in SETS:
    for m, mk in MODELS.items():
        acc, sd, f1, per = score(lopo(F[k], mk, TEST, np.ones(len(y), bool)), TEST)
        plain[(k, m)] = f1
        print(f"  {k} {m:3s}: accuracy {acc:.2f} ± {sd:.2f}, macro F1 {f1:.2f}  per person {np.round(per, 2).tolist()}")

print("\nnested (honest):")
outer = []
for t in TEST:
    inner, m = [q for q in TEST if q != t], who != t
    best = max(((k, mm) for k in SETS for mm in MODELS),
               key=lambda km: score(lopo(F[km[0]], MODELS[km[1]], inner, m), inner)[2])
    te = who == t
    z = MODELS[best[1]]().fit(F[best[0]][m], y[m]).predict(F[best[0]][te])
    outer.append((accuracy_score(y[te], z), f1_score(y[te], z, average="macro", labels=range(3))))
    print(f"  test {t}: chose {best[0]} {best[1]} -> accuracy {outer[-1][0]:.2f}, macro F1 {outer[-1][1]:.2f}")
print(f"  NESTED: accuracy {np.mean([o[0] for o in outer]):.2f} ± {np.std([o[0] for o in outer]):.2f}, "
      f"macro F1 {np.mean([o[1] for o in outer]):.2f}")

k, m = max(plain, key=plain.get)
print(f"\nfinal model: {k} {m} (best plain score), trained on all 5 people")
stage1 = MODELS[m]().fit(F[k], y)

# stage 2: up / down on the direction-labelled stairs windows, same features
Xd, yd, wd = [], [], []
for name, segs in SEG.items():
    kp = np.load(f"data/video/keypoints/{name}.npz")["kp"]
    for d, s, e, _ in segs:
        X = feats(kp[int(s * FPS):int(e * FPS)], SETS[k])
        if X is not None:
            Xd.append(X); yd += [DIRS.index(d)] * len(X); wd += [name[:2]] * len(X)
Xd, yd, wd = np.concatenate(Xd), np.array(yd), np.array(wd)
right = np.concatenate([MODELS["RF"]().fit(Xd[wd != t], yd[wd != t]).predict(Xd[wd == t]) == yd[wd == t] for t in TEST])
print(f"stairs direction, leave one person out: {right.mean():.2f} of {len(right)} windows")
bundle = {"model": stage1, "classes": C3, "cols": SETS[k], "features": f"{k} {m}",
          "direction": {"model": MODELS["RF"]().fit(Xd, yd), "classes": DIRS}}
joblib.dump(bundle, MODEL)
print(f"saved {MODEL}")
