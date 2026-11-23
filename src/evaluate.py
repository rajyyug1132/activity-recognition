"""Owner: Yash."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

from inference import ACTS


def report(y_true, y_pred, out_dir, name="model") -> dict:
    """Accuracy, F1 per activity, confusion matrix PNG saved to out_dir/<name>_confusion.png."""
    os.makedirs(out_dir, exist_ok=True)
    f1 = f1_score(y_true, y_pred, average=None, labels=range(6))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=ACTS, ax=ax,
                                            xticks_rotation=45, colorbar=False)
    ax.set_title(name); fig.tight_layout()
    fig.savefig(f"{out_dir}/{name}_confusion.png", dpi=120); plt.close(fig)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "f1_per_activity": dict(zip(ACTS, f1.round(4)))}


def plot_learning_curves(history, path):
    """Train vs validation loss and accuracy per epoch (the overfitting graph)."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, k in zip(axs, ["loss", "accuracy"]):
        ax.plot(history[k], label="train"); ax.plot(history["val_" + k], label="validation")
        ax.set_xlabel("epoch"); ax.set_title(k); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
