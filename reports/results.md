# Results

One row per experiment.

| Date | Model | Data | Split | Accuracy | Macro F1 | Ran by |
|---|---|---|---|---|---|---|
| 2026-11-23 | Random Forest (300 trees) | UCI 561 features | official (21 train / 9 test subjects) | 0.929 | 0.927 | Yyug |
| 2026-11-23 | 1D-CNN (best epoch 4 of 9, early stopping) | UCI raw windows, normalised | 18 train / 3 val (3, 22, 25) / 9 test subjects | 0.926 | 0.926 | Yyug |
| 2026-11-24 | LSTM (2x64, best epoch 5 of 10, early stopping) | UCI raw windows, normalised | 18 train / 3 val (3, 22, 25) / 9 test subjects | 0.881 | 0.875 | Yyug |
