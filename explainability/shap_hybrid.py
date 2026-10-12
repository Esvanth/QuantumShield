"""
QuantumShield - SHAP explainability (H3).

H3: the features the ensemble relies on are domain-plausible intrusion
    indicators rather than incidental ones.

Method
------
The deployed model consumes 16 PCA components, not raw network features, so
KernelSHAP is applied at that boundary and the resulting attributions are then
projected back onto the 78 original CICIDS2017 features through the PCA
loadings.

The projection is an IMPORTANCE approximation, not exact Shapley values on the
raw features:

    importance(feature i) = sum_j  mean|phi_j| * |W[j, i]|

where phi_j is the SHAP value of component j and W = pca.components_. Because
PCA is a linear map, a component's influence distributes across the original
features in proportion to its loadings. Exact Shapley values on the 78 raw
features would require explaining a wrapper that re-runs scaler and PCA inside
the model, which is defensible but roughly five times the compute and needs the
raw feature matrix, which this pipeline does not retain. State the
approximation in the report rather than implying exact attributions.

The model explained here is the WEIGHTED soft-voting ensemble using the weights
fitted during hypothesis evaluation, so the explanations describe the same
model the H1 result came from.

Usage:
    python -m explainability.shap_hybrid
    python -m explainability.shap_hybrid --n-explain 60 --nsamples 300
"""

import os
import json
import pickle
import logging
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from models.classical.dnn import DNN
from models.classical.cnn import CNN1D
from models.classical.lstm import LSTMClassifier
from models.quantum.circuits import HybridQNN, N_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAVE_DIR = "results/shap_plots"
MODEL_ORDER = ["DNN", "CNN-1D", "LSTM", "HybridQNN"]

# Features the intrusion detection literature consistently associates with
# attack discrimination on flow-based datasets. Used only to flag which of the
# top-ranked features are expected ones; it does not decide H3 on its own.
REFERENCE_SUBSTRINGS = [
    "Flow Duration", "IAT", "Packet Length", "Packet Size",
    "Flow Bytes", "Flow Packets", "Flag Count", "Flag",
    "Fwd Packets", "Bwd Packets", "Destination Port",
    "Init_Win", "Active", "Idle", "Subflow",
]


# ---------------------------------------------------------------------------
# Weighted ensemble, matching the model evaluated in hypotheses.py
# ---------------------------------------------------------------------------

class WeightedEnsemble:
    """Soft-voting ensemble returning P(attack). Weights come from the
    validation-fitted values recorded during hypothesis evaluation."""

    def __init__(self, weights=None):
        self.models = {}

        dnn = DNN(input_dim=N_FEATURES)
        dnn.load_state_dict(torch.load("models/classical/dnn_weights.pth", map_location="cpu"))
        self.models["DNN"] = dnn

        cnn = CNN1D(input_dim=N_FEATURES)
        cnn.load_state_dict(torch.load("models/classical/cnn_weights.pth", map_location="cpu"))
        self.models["CNN-1D"] = cnn

        lstm = LSTMClassifier(input_dim=N_FEATURES)
        lstm.load_state_dict(torch.load("models/classical/lstm_weights.pth", map_location="cpu"))
        self.models["LSTM"] = lstm

        hqnn = HybridQNN()
        hqnn.load_state_dict(torch.load("models/saved/hybrid_qnn.pth", map_location="cpu"))
        self.models["HybridQNN"] = hqnn

        for m in self.models.values():
            m.eval()

        if weights is None:
            weights = {m: 1 for m in MODEL_ORDER}
        w = np.array([weights[m] for m in MODEL_ORDER], dtype=np.float64)
        self.w = w / w.sum()
        self.weights = weights
        log.info("ensemble weights %s", weights)

    def __call__(self, X):
        """X is (n, 16) PCA features. Returns P(attack) as (n,)."""
        X = np.asarray(X, dtype=np.float32)
        parts = []
        with torch.no_grad():
            Xt = torch.FloatTensor(X)
            for name in MODEL_ORDER:
                model = self.models[name]
                inp = model.scale(Xt) if name == "HybridQNN" else Xt
                parts.append(torch.softmax(model(inp), dim=1)[:, 1].numpy())
        return self.w @ np.stack(parts)


def load_fitted_weights(tag):
    path = f"results/{tag}_hypotheses.json"
    if not os.path.exists(path):
        log.warning("%s not found, falling back to equal weights", path)
        return None
    with open(path) as f:
        data = json.load(f)
    try:
        return data["H1"]["weighted"]["weights"]
    except (KeyError, TypeError):
        log.warning("no fitted weights in %s, falling back to equal weights", path)
        return None


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def as_single_output(shap_values):
    """
    Normalise SHAP's return shape across versions. The wrapper returns a
    single scalar per row, so we expect (n, n_features); older versions may
    still wrap it in a list.
    """
    if isinstance(shap_values, list):
        shap_values = shap_values[0] if len(shap_values) == 1 else shap_values[-1]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:            # (n, features, outputs)
        arr = arr[:, :, -1]
    return arr


def compute_shap(data_dir="data/processed/cicids2017",
                 n_background=30, n_explain=40, nsamples=200, seed=42):
    tag = os.path.basename(data_dir.rstrip("/\\"))
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(seed)

    X_train  = np.load(os.path.join(data_dir, "X_train.npy")).astype(np.float32)[:, :N_FEATURES]
    yb_train = np.load(os.path.join(data_dir, "yb_train.npy"))
    X_test   = np.load(os.path.join(data_dir, "X_test.npy")).astype(np.float32)[:, :N_FEATURES]
    yb_test  = np.load(os.path.join(data_dir, "yb_test.npy"))

    # Background: benign traffic, the reference the explanation is relative to
    benign_idx = np.where(yb_train == 0)[0]
    background = X_train[rng.choice(benign_idx, min(n_background, len(benign_idx)),
                                    replace=False)]

    # Explain actual attack samples, since H3 is about attack detection
    attack_idx = np.where(yb_test == 1)[0]
    explain_idx = rng.choice(attack_idx, min(n_explain, len(attack_idx)), replace=False)
    X_explain = X_test[explain_idx]

    ensemble = WeightedEnsemble(load_fitted_weights(tag))

    total_calls = n_background * nsamples * len(X_explain)
    log.info("KernelSHAP: background=%d, explain=%d attack samples, nsamples=%d",
             len(background), len(X_explain), nsamples)
    log.info("approximately %d model evaluations, expect several minutes", total_calls)

    explainer = shap.KernelExplainer(ensemble, background)
    raw = explainer.shap_values(X_explain, nsamples=nsamples)
    phi = as_single_output(raw)           # (n_explain, 16)
    log.info("SHAP values computed, shape %s", phi.shape)

    np.save(os.path.join(SAVE_DIR, "shap_values_components.npy"), phi)

    # --- component-level importance ----------------------------------------
    comp_names = [f"PC{i+1}" for i in range(phi.shape[1])]
    comp_importance = np.abs(phi).mean(axis=0)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(phi, X_explain, feature_names=comp_names,
                      show=False, plot_type="bar")
    plt.title("QuantumShield - SHAP importance by PCA component (attack class)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_components_bar.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(phi, X_explain, feature_names=comp_names, show=False)
    plt.title("QuantumShield - SHAP value distribution by PCA component")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_components_beeswarm.png"), dpi=150)
    plt.close()

    # --- project back onto the original network features --------------------
    with open(os.path.join(data_dir, "pca.pkl"), "rb") as f:
        pca = pickle.load(f)
    W = np.abs(pca.components_)                     # (16, 78)

    names_path = os.path.join(data_dir, "feature_names.npy")
    if not os.path.exists(names_path):
        raise SystemExit(f"{names_path} not found. Run data.recover_feature_names first.")
    feature_names = np.load(names_path, allow_pickle=True)

    if W.shape[0] != len(comp_importance) or W.shape[1] != len(feature_names):
        raise SystemExit(f"shape mismatch: loadings {W.shape}, "
                         f"components {len(comp_importance)}, names {len(feature_names)}")

    feat_importance = comp_importance @ W           # (78,)
    feat_importance = feat_importance / feat_importance.sum()

    order = np.argsort(feat_importance)[::-1]
    top20 = [(str(feature_names[i]), float(feat_importance[i])) for i in order[:20]]

    plt.figure(figsize=(10, 7))
    labels = [n for n, _ in top20][::-1]
    values = [v for _, v in top20][::-1]
    plt.barh(labels, values, color="#4C78A8")
    plt.xlabel("Relative importance (SHAP projected through PCA loadings)")
    plt.title("QuantumShield - top 20 original features by projected SHAP importance")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_features_projected.png"), dpi=150)
    plt.close()

    # --- H3 reporting -------------------------------------------------------
    def expected(name):
        return any(s.lower() in name.lower() for s in REFERENCE_SUBSTRINGS)

    top5 = [{"rank": r + 1,
             "feature": n,
             "importance": round(v, 6),
             "matches_reference_list": bool(expected(n))}
            for r, (n, v) in enumerate(top20[:5])]
    hits = sum(1 for f in top5 if f["matches_reference_list"])

    log.info("top 5 original features by projected SHAP importance:")
    for f in top5:
        log.info("  %d. %-40s %.4f  %s", f["rank"], f["feature"], f["importance"],
                 "expected" if f["matches_reference_list"] else "unexpected")
    log.info("%d of 5 match the reference list of commonly discriminative features", hits)

    summary = {
        "dataset": tag,
        "method": ("KernelSHAP on 16 PCA components, projected onto 78 original "
                   "features via sum_j mean|phi_j| * |pca.components_[j, i]|. "
                   "Approximate importance ranking, not exact Shapley values on "
                   "the raw features."),
        "model_explained": "weighted soft-voting ensemble",
        "ensemble_weights": ensemble.weights,
        "n_background": int(len(background)),
        "n_explained": int(len(X_explain)),
        "explained_class": "attack",
        "nsamples": int(nsamples),
        "component_importance": {c: float(v) for c, v in zip(comp_names, comp_importance)},
        "top_components": [comp_names[i] for i in np.argsort(comp_importance)[::-1][:5]],
        "top20_features": [{"feature": n, "importance": round(v, 6)} for n, v in top20],
        "top5_features": top5,
        "reference_list_hits": hits,
        "h3_verdict": None,
        "h3_note": ("Set h3_verdict to true or false after reviewing the top-5 "
                    "against domain expectation. reference_list_hits is an "
                    "indicator, not the test: every CICIDS2017 column is a network "
                    "feature, so the hypothesis should ask whether these are the "
                    "features an analyst would expect to discriminate attacks."),
    }

    with open("results/shap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("plots written to %s", SAVE_DIR)
    log.info("summary written to results/shap_summary.json")
    return summary


def explain_single(sample_idx=0, data_dir="data/processed/cicids2017"):
    """Waterfall plot for one explained sample, for the dashboard."""
    path = os.path.join(SAVE_DIR, "shap_values_components.npy")
    if not os.path.exists(path):
        log.warning("run compute_shap() first")
        return
    phi = np.load(path)
    X_test = np.load(os.path.join(data_dir, "X_test.npy")).astype(np.float32)[:, :N_FEATURES]
    comp_names = [f"PC{i+1}" for i in range(phi.shape[1])]

    exp = shap.Explanation(values=phi[sample_idx],
                           base_values=0.0,
                           data=X_test[sample_idx],
                           feature_names=comp_names)
    plt.figure()
    shap.plots.waterfall(exp, show=False)
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, f"shap_waterfall_sample{sample_idx}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("waterfall plot saved to %s", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed/cicids2017")
    ap.add_argument("--n-background", type=int, default=30)
    ap.add_argument("--n-explain", type=int, default=40)
    ap.add_argument("--nsamples", type=int, default=200)
    args = ap.parse_args()

    compute_shap(data_dir=args.data_dir,
                 n_background=args.n_background,
                 n_explain=args.n_explain,
                 nsamples=args.nsamples)
    explain_single(0, data_dir=args.data_dir)


if __name__ == "__main__":
    main()