"""
QuantumShield – Evaluation Metrics
Computes and logs all metrics needed for H1–H4.
"""

import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

log = logging.getLogger(__name__)


def compute_all_metrics(y_true, y_pred, y_prob=None, model_name="Model", save_dir="results"):
    """
    Compute and log full metric suite.

    Parameters
    ----------
    y_true     : array-like (n,)
    y_pred     : array-like (n,)
    y_prob     : array-like (n,) P(attack) – for AUROC; optional
    model_name : str for plot titles / file names
    save_dir   : directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    log.info("=== %s ===", model_name)
    log.info("  Accuracy : %.4f", acc)
    log.info("  Precision: %.4f", prec)
    log.info("  Recall   : %.4f", rec)
    log.info("  F1-Score : %.4f", f1)

    if y_prob is not None:
        auroc = roc_auc_score(y_true, y_prob)
        log.info("  AUROC    : %.4f", auroc)
    else:
        auroc = None

    log.info("\n%s", classification_report(y_true, y_pred, target_names=["Benign", "Attack"]))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Attack"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{model_name} – Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_cm.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    log.info("Confusion matrix saved to %s", cm_path)

    # ROC curve
    if y_prob is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=model_name)
        ax.set_title(f"{model_name} – ROC Curve")
        plt.tight_layout()
        roc_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_roc.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        log.info("ROC curve saved to %s", roc_path)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auroc": auroc}


def inference_latency(model_fn, X_tensor, n_runs=100):
    """
    Measure average inference time per sample (ms).
    model_fn: callable(X_tensor) → predictions
    """
    import time
    model_fn(X_tensor[:1])   # warm-up

    start = time.perf_counter()
    for _ in range(n_runs):
        model_fn(X_tensor[:1])
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    log.info("Inference latency: %.3f ms/sample (avg over %d runs)", elapsed, n_runs)
    return elapsed


def compare_models(results_dict, save_dir="results"):
    """
    Bar chart comparing F1 and AUROC across all models.
    results_dict: { "DNN": {"f1": 0.95, "auroc": 0.97}, ... }
    """
    os.makedirs(save_dir, exist_ok=True)
    names  = list(results_dict.keys())
    f1s    = [results_dict[n].get("f1",    0) for n in names]
    aurocs = [results_dict[n].get("auroc", 0) or 0 for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, f1s,    0.35, label="F1-Score",  color="#4C72B0")
    ax.bar(x + 0.2, aurocs, 0.35, label="AUROC",     color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("QuantumShield – Model Comparison (F1 & AUROC)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Comparison chart saved to %s", path)
    return path
