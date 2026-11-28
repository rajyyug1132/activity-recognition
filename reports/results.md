# Results

One row per experiment.

| Date | Model | Data | Split | Accuracy | Macro F1 | Ran by |
|---|---|---|---|---|---|---|
| 2026-11-23 | Random Forest (300 trees) | UCI 561 features | official (21 train / 9 test subjects) | 0.929 | 0.927 | Yyug |
| 2026-11-23 | 1D-CNN (best epoch 4 of 9, early stopping) | UCI raw windows, normalised | 18 train / 3 val (3, 22, 25) / 9 test subjects | 0.926 | 0.926 | Yyug |
| 2026-11-24 | LSTM (2x64, best epoch 5 of 10, early stopping) | UCI raw windows, normalised | 18 train / 3 val (3, 22, 25) / 9 test subjects | 0.881 | 0.875 | Yyug |
| 2026-11-24 | 1D-CNN retrained, now `models/cnn_uci.keras` (best epoch 2 of 7, early stopping) | UCI raw windows, normalised | 18 train / 3 val (3, 22, 25) / 9 test subjects | 0.910 | 0.909 | Yyug |
| 2026-11-24 | 1D-CNN baseline, mean of 3 seeds (42-44) | UCI raw windows, normalised | 18 train / 3 val (3, 22, 25) / 9 test subjects | 0.913 ± 0.004 | 0.912 | Yyug |
| 2026-11-24 | 1D-CNN + gravity-angle channels (12 ch), mean of 3 seeds | UCI raw windows + 3 gravity angles | same | 0.914 ± 0.002 | 0.914 | Yyug |
| 2026-11-24 | 1D-CNN + gravity-angle dense branch, mean of 3 seeds | UCI raw windows + window-mean gravity angles | same | 0.919 ± 0.006 | 0.919 | Yyug |
| 2026-11-24 | 1D-CNN + gravity-angle dense branch, best seed (44), `figures/sitstand_best_confusion.png` | same | same | 0.925 | 0.924 | Yyug |
| 2026-11-24 | Baseline CNN scored as 5 classes (sitting+standing = stationary), mean of 3 seeds | same | same | 0.968 | 0.961 | Yyug |
| 2026-11-24 | Dense-branch CNN scored as 5 classes, mean of 3 seeds | same | same | 0.977 | 0.972 | Yyug |

**Sitting vs standing (notebook 05):** no variant reduces sit↔stand errors (baseline 162, angle channels 159, dense branch 171 of 1,023 test sit/stand windows, means over 3 seeds). The dense branch's small 6-class gain comes from other classes and is within seed noise (the baseline itself moved 0.5 pt between two identical runs). Gravity-angle histograms for sitting and standing overlap heavily: a waist phone barely tilts between the two postures, and `total_acc` already carries that tilt. Merging them into one "stationary" class gives 0.97 accuracy.

## TFLite export (UCI test set, 9 unseen subjects)

| Model | Size (KB) | Accuracy | Macro F1 | ms/window |
|---|---|---|---|---|
| Keras (reference, incl. optimizer state) | 4820 | 0.910 | 0.909 | - |
| TFLite float32 | 1601 | 0.910 | 0.909 | 0.141 |
| TFLite dynamic-range | 410 | 0.910 | 0.909 | 0.051 |
| TFLite full int8 | 411 | 0.911 | 0.910 | 0.045 |

Notebook `06_tflite.ipynb`, run on the retrained `cnn_uci.keras` (saved 2026-11-24 23:46). Latency = mean per window, batch 1, 1 thread, TFLite interpreter on a Windows PC CPU (not a phone); it varies with machine load. Int8 calibrated on 200 random train windows. TFLite matches the Keras scores, so quantization costs nothing. Training is not bit-reproducible (same seed gave 0.919 vs 0.934 on two runs), so the Keras row can differ from other CNN rows.

## Camera pipeline (own videos, 3 classes: walking, standing, stairs)

Leave-one-person-out over p1-p4 (493 windows of 2 s in total). p5 has one walking clip, so p5 is training-only and never a test or validation person. Not comparable with the 6-class phone results above. Two baselines: random guessing scores about 0.33 macro F1; always answering "standing" (the largest class) scores 0.39 accuracy but only 0.18 macro F1. Accuracy alone flatters weak models here, so macro F1 is the main number. Notebooks `07_video.ipynb`, `08_video_features.ipynb`.

| Date | Model | Input | Accuracy | Macro F1 | Ran by |
|---|---|---|---|---|---|
| 2026-11-26 | 1D-CNN | raw keypoints x,y (mirror augmentation) | 0.456 ± 0.150 | 0.407 ± 0.071 | Yyug |
| 2026-11-26 | 1D-CNN | raw keypoints, height (y) only | 0.373 ± 0.094 | 0.340 ± 0.104 | Yyug |
| 2026-11-26 | Random Forest | raw keypoints x,y, window mean/std/min/max | 0.586 ± 0.159 | 0.589 ± 0.151 | Yyug |
| 2026-11-26 | Random Forest | raw keypoints, height (y) only | 0.459 ± 0.051 | 0.472 ± 0.085 | Yyug |
| 2026-11-26 | Random Forest | 23 hand-built features | 0.621 ± 0.150 | 0.627 ± 0.112 | Yyug |
| 2026-11-26 | Logistic regression | 23 hand-built features | 0.615 ± 0.170 | 0.609 ± 0.150 | Yyug |

± is the standard deviation over the 4 held-out people. Ablation on the hand-built features (Random Forest, macro F1): all 23 = 0.627, without `torso_tilt` = 0.585, without `torso_tilt` and `knee_y_mean` = 0.660. These differences are within noise at 4 test people. The 120 fps clips (A (4), B (8), C (7)) were checked: walking cadence 0.63-0.74 Hz, same as 30 fps clips (0.66-0.85 Hz), so they are not slow motion and are kept. The first run (before p5's clip was added) gave the same picture: CNN 0.36, raw RF 0.63, features RF 0.63.

## Camera pipeline: stairs direction (`notebooks/09_stairs_direction.py`)

A second stage on top of the 3-class model: for windows called stairs, a Random Forest with the same settings says up or down. The 3-class model is unchanged. Direction labels were set by eye from video frames, never from a model's output, and were not checked by a second person. Five clips were filmed from the top of the flight, so walking toward the camera is up. Three side-view clips were judged by which step the front foot lands on. A (5) and D (6) are side views whose direction could not be told, and A (3) is too short: all three are left out. 82 windows remain (46 up, 36 down), and each of p1-p4 has both directions.

| Date | Test (leave one person out) | Direction right | Chance | Ran by |
|---|---|---|---|---|
| 2026-11-28 | all stairs windows | 0.82 of 82 (p1 0.74, p2 0.80, p3 0.93, p4 1.00) | 0.50 ± 0.06 with training labels shuffled | Yyug |
| 2026-11-28 | side views only | 0.77 of 35 | 0.50 | Yyug |
| 2026-11-28 | front/back views only | 0.85 of 47 | 0.50 | Yyug |

Whole pipeline, 4 classes (walking, standing, upstairs, downstairs): macro F1 0.50 ± 0.12 (random 0.25). A single 4-class Random Forest scored 0.48 ± 0.13 and called more stairs windows walking, so the app uses the two stages. In the front/back clips "up" always means "toward the camera", so a model could learn that instead of direction. The side-view score rules this out, but it rests on 35 windows from 3 clips.

**Caveats.** (1) 0.63 is slightly optimistic: the 23 features, the RF and logistic-regression settings and the CNN input variant were all chosen while looking at the same 4 test people. (2) The app's Random Forest is trained on all 5 people and has no held-out score; 0.63 is the leave-one-person-out estimate of the same recipe. (3) The saved `models/video_cnn.keras` was early-stopped on a random 15% of windows, which overlap 50% with training windows, so its 0.87 validation accuracy is inflated and must not be quoted. It is not used by the app.
