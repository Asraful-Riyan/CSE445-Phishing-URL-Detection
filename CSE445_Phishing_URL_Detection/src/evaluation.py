import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from config import RESULTS_DIR


def get_phishing_probability(model, X):
    """
    Dataset labels:
      0 = Phishing
      1 = Legitimate

    Return probability/score for phishing (class 0).
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        classes = list(model.classes_)
        idx = classes.index(0)
        return probs[:, idx]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Most sklearn binary decision functions score class 1 positively.
        return -np.asarray(scores)

    return None


def evaluate_predictions(name, y_true, y_pred, phishing_score=None, save_plots=True):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_Phishing": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Recall_Phishing": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "F1_Phishing": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }

    if phishing_score is not None:
        y_phish = (y_true == 0).astype(int)
        try:
            metrics["ROC_AUC"] = roc_auc_score(y_phish, phishing_score)
        except ValueError:
            metrics["ROC_AUC"] = np.nan
    else:
        metrics["ROC_AUC"] = np.nan

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Phishing (0)", "Legitimate (1)"],
        digits=4,
        zero_division=0,
    )

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )

    (RESULTS_DIR / f"{safe_name}_classification_report.txt").write_text(report)

    if save_plots:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Phishing", "Legitimate"],
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, values_format="d")
        ax.set_title(f"{name} - Confusion Matrix")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"{safe_name}_confusion_matrix.png", dpi=160)
        plt.close(fig)

        if phishing_score is not None and len(np.unique(y_true)) == 2:
            y_phish = (y_true == 0).astype(int)
            fig, ax = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_predictions(y_phish, phishing_score, ax=ax)
            ax.set_title(f"{name} - ROC Curve (Phishing Positive)")
            fig.tight_layout()
            fig.savefig(RESULTS_DIR / f"{safe_name}_roc_curve.png", dpi=160)
            plt.close(fig)

    return metrics, report
