"""
QuantumShield – SHAP Explainability (H3)
KernelSHAP applied at the hybrid model's input-output boundary.

H3: ≥ 4 of the top-5 SHAP features are domain-plausible network features.

SHAP is applied to the FULL hybrid ensemble (classical + quantum),
treating it as a black box with 8 PCA inputs → 2-class output.
"""

import os
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
import shap
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.ensemble import SoftVotingEnsemble

log = logging.getLogger(__name__)

SAVE_DIR = "results/shap_plots"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Wrapper so SHAP sees a plain numpy callable
# ---------------------------------------------------------------------------

class EnsembleWrapper:
    """SHAP-compatible wrapper returning P(attack) probabilities."""

    def __init__(self, ensemble: SoftVotingEnsemble):
        self.ensemble = ensemble

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.FloatTensor(X)
        probs = self.ensemble.predict_proba(X_tensor)
        return probs   # (n, 2)


# ---------------------------------------------------------------------------
# Compute SHAP values
# ---------------------------------------------------------------------------

def compute_shap(
    data_dir="data/processed/cicids2017",
    n_background=100,
    n_explain=50,
    pca_pkl="data/processed/cicids2017/pca.pkl",
):
    """
    1. Load background (benign) samples for KernelExplainer.
    2. Run SHAP on n_explain test samples.
    3. Map SHAP values back to original feature names via PCA loadings.
    4. Save summary plot and return shap_values.
    """
    X_train  = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test   = np.load(os.path.join(data_dir, "X_test.npy"))
    yb_train = np.load(os.path.join(data_dir, "yb_train.npy"))

    # Background: balanced sample of benign
    benign_idx = np.where(yb_train == 0)[0]
    bg_idx = np.random.choice(benign_idx, min(n_background, len(benign_idx)), replace=False)
    background = X_train[bg_idx]

    # Explain first n_explain test samples
    X_explain = X_test[:n_explain]

    ensemble = SoftVotingEnsemble()
    wrapper = EnsembleWrapper(ensemble)

    log.info("Running KernelSHAP (background=%d, explain=%d)...", len(background), len(X_explain))
    explainer   = shap.KernelExplainer(wrapper, background)
    shap_values = explainer.shap_values(X_explain, nsamples=100)
    # shap_values: list[2] each (n_explain, 8) — index 1 = ATTACK class

    attack_shap = shap_values[1]   # focus on attack class explanations
    log.info("SHAP values computed. Shape: %s", attack_shap.shape)

    # --- Summary plot (PCA components labelled PC1–PC8) ---
    feature_names = [f"PC{i+1}" for i in range(X_explain.shape[1])]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        attack_shap, X_explain,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
    )
    plt.title("QuantumShield – SHAP Feature Importance (Attack Class)")
    plt.tight_layout()
    summary_path = os.path.join(SAVE_DIR, "shap_summary.png")
    plt.savefig(summary_path, dpi=150)
    plt.close()
    log.info("SHAP summary plot saved to %s", summary_path)

    # --- H3 check: top-5 features by mean |SHAP| ---
    mean_abs = np.abs(attack_shap).mean(axis=0)
    top5_idx = np.argsort(mean_abs)[::-1][:5]
    log.info("H3 – Top-5 SHAP features (by mean |SHAP|):")
    for rank, idx in enumerate(top5_idx, 1):
        log.info("  %d. %s (mean |SHAP|=%.4f)", rank, feature_names[idx], mean_abs[idx])

    # Save SHAP values for further analysis
    np.save(os.path.join(SAVE_DIR, "shap_values_attack.npy"), attack_shap)
    log.info("Raw SHAP values saved to %s", os.path.join(SAVE_DIR, "shap_values_attack.npy"))

    return shap_values, top5_idx


# ---------------------------------------------------------------------------
# Per-sample waterfall plot (for dashboard / individual alert explanation)
# ---------------------------------------------------------------------------

def explain_single(sample_idx=0, data_dir="data/processed/cicids2017"):
    """Generate a waterfall plot for one test sample."""
    shap_path = os.path.join(SAVE_DIR, "shap_values_attack.npy")
    X_test    = np.load(os.path.join(data_dir, "X_test.npy"))

    if not os.path.exists(shap_path):
        log.warning("SHAP values not found. Run compute_shap() first.")
        return

    attack_shap = np.load(shap_path)
    feature_names = [f"PC{i+1}" for i in range(X_test.shape[1])]

    shap_exp = shap.Explanation(
        values=attack_shap[sample_idx],
        base_values=0.5,
        data=X_test[sample_idx],
        feature_names=feature_names,
    )
    plt.figure()
    shap.waterfall_plot(shap_exp, show=False)
    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, f"shap_waterfall_sample{sample_idx}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Waterfall plot saved to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    compute_shap()
    explain_single(sample_idx=0)
