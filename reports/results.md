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
