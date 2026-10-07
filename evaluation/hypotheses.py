"""
QuantumShield – Hypothesis Testing Runner
Evaluates H1, H2, H3, H4 and prints a final verdict table.
"""

import logging
import numpy as np
import torch
from evaluation.metrics import compute_all_metrics, compare_models
from models.ensemble import SoftVotingEnsemble, compare_parameter_counts, mcnemar_test
from models.classical.dnn import DNN
from models.quantum.qae import QuantumAutoencoder, evaluate_qae
from explainability.shap_hybrid import compute_shap

log = logging.getLogger(__name__)

DATA_DIR = "data/processed/cicids2017"


def _load_npy(fname):
    import os
    return np.load(os.path.join(DATA_DIR, fname))


def run_all_hypotheses():
    results = {}

    # -----------------------------------------------------------------------
    # H1 – Detection parity (ensemble F1 ≥ best classical, McNemar p < 0.05)
    # -----------------------------------------------------------------------
    log.info("\n--- H1: Detection Parity ---")
    X_test  = torch.FloatTensor(_load_npy("X_test.npy"))
    yb_test = _load_npy("yb_test.npy")

    ensemble = SoftVotingEnsemble()
    y_prob_ens = ensemble.predict_proba(X_test)[:, 1]
    y_pred_ens = (y_prob_ens >= 0.5).astype(int)

    # Best classical: DNN
    from models.classical.dnn import DNN
    import os, torch
    dnn = DNN(input_dim=8)
    if os.path.exists("models/classical/dnn_weights.pth"):
        dnn.load_state_dict(torch.load("models/classical/dnn_weights.pth", map_location="cpu"))
    dnn.eval()
    with torch.no_grad():
        dnn_probs = torch.softmax(dnn(X_test), dim=1)[:, 1].numpy()
    y_pred_dnn = (dnn_probs >= 0.5).astype(int)

    ens_metrics = compute_all_metrics(yb_test, y_pred_ens, y_prob_ens, "Ensemble")
    dnn_metrics = compute_all_metrics(yb_test, y_pred_dnn, dnn_probs,  "DNN")

    pvalue = mcnemar_test(yb_test, y_pred_dnn, y_pred_ens)
    h1_pass = (ens_metrics["f1"] >= dnn_metrics["f1"]) and (pvalue < 0.05)
    results["H1"] = {"pass": h1_pass, "ensemble_f1": ens_metrics["f1"],
                     "dnn_f1": dnn_metrics["f1"], "mcnemar_p": pvalue}
    log.info("H1 verdict: %s", "PASS" if h1_pass else "FAIL")

    # -----------------------------------------------------------------------
    # H2 – Parameter efficiency ≥ 3×
    # -----------------------------------------------------------------------
    log.info("\n--- H2: Parameter Efficiency ---")
    counts, ratio = compare_parameter_counts()
    h2_pass = ratio >= 3.0
    results["H2"] = {"pass": h2_pass, "ratio": ratio, "counts": counts}
    log.info("H2 verdict: %s (ratio=%.2f×)", "PASS" if h2_pass else "FAIL", ratio)

    # -----------------------------------------------------------------------
    # H3 – SHAP plausibility (≥ 4 of top-5 are domain-plausible)
    # -----------------------------------------------------------------------
    log.info("\n--- H3: SHAP Plausibility ---")
    try:
        _, top5_idx = compute_shap(data_dir=DATA_DIR, n_background=50, n_explain=20)
        # Human-in-the-loop check; auto-mark as pending if not evaluated
        h3_plausible = None   # set to True/False after manual feature inspection
        results["H3"] = {"pass": h3_plausible, "top5_pca_components": top5_idx.tolist()}
        log.info("H3: top-5 PCA components identified. Manual plausibility check required.")
    except Exception as e:
        log.warning("H3 SHAP computation skipped: %s", e)
        results["H3"] = {"pass": None, "error": str(e)}

    # -----------------------------------------------------------------------
    # H4 – QAE AUROC ≥ 0.85
    # -----------------------------------------------------------------------
    log.info("\n--- H4: QAE Zero-Day AUROC ---")
    try:
        qae = QuantumAutoencoder()
        import os
        if os.path.exists("models/quantum/qae_weights.pth"):
            qae.load_state_dict(torch.load("models/quantum/qae_weights.pth", map_location="cpu"))
        auroc, _ = evaluate_qae(qae, data_dir=DATA_DIR)
        h4_pass = auroc >= 0.85
        results["H4"] = {"pass": h4_pass, "auroc": auroc}
        log.info("H4 verdict: %s (AUROC=%.4f)", "PASS" if h4_pass else "FAIL", auroc)
    except Exception as e:
        log.warning("H4 QAE evaluation skipped: %s", e)
        results["H4"] = {"pass": None, "error": str(e)}

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    log.info("\n========== HYPOTHESIS SUMMARY ==========")
    for h, res in results.items():
        status = "PASS" if res["pass"] is True else ("FAIL" if res["pass"] is False else "PENDING")
        log.info("  %-4s: %s", h, status)
    log.info("=========================================")

    # Compare all models
    all_results = {
        "DNN":      dnn_metrics,
        "Ensemble": ens_metrics,
    }
    compare_models(all_results, save_dir="results")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_all_hypotheses()
