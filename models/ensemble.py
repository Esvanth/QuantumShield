"""
QuantumShield – Soft-Voting Ensemble
Combines classical (DNN, CNN, LSTM) and quantum (HybridQNN) branch outputs
via soft-voting on class probabilities.

H1: ensemble detection parity (F1 ≥ best classical baseline, p < 0.05 McNemar).
H2: HybridQNN trainable parameters ≥ 3× fewer than best classical model.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

from models.classical.dnn  import DNN
from models.classical.cnn  import CNN1D
from models.classical.lstm import LSTMClassifier
from models.quantum.circuits import HybridQNN

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(cls, weight_path, **kwargs):
    model = cls(**kwargs)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        log.info("Loaded weights from %s", weight_path)
    else:
        log.warning("Weight file not found: %s — using random weights", weight_path)
    model.eval()
    return model

def _get_probs(model, X_tensor, batch_size=256):
    """Run model in eval mode and return softmax probabilities (n, n_classes)."""
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size)
    probs_list = []
    with torch.no_grad():
        for (x,) in loader:
            logits = model(x)
            probs_list.append(torch.softmax(logits, dim=1))
    return torch.cat(probs_list, dim=0).numpy()


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class SoftVotingEnsemble:
    """
    Soft-voting ensemble over four models.
    weights: per-model contribution (default: equal)
    """

    def __init__(self, weights=None):
        self.models = {}
        self.weights = weights   # dict: model_name -> float, or None for equal
        self.models["dnn"]  = _load_model(DNN,            "models/classical/dnn_weights.pth",  input_dim=8)
        self.models["cnn"]  = _load_model(CNN1D,          "models/classical/cnn_weights.pth",  input_dim=8)
        self.models["lstm"] = _load_model(LSTMClassifier, "models/classical/lstm_weights.pth", input_dim=8)
        self.models["qnn"]  = _load_model(HybridQNN,      "models/quantum/qnn_weights.pth")

    def predict_proba(self, X_tensor):
        """Return averaged softmax probabilities (n_samples, n_classes)."""
        names = list(self.models.keys())
        if self.weights is None:
            w = {n: 1.0 / len(names) for n in names}
        else:
            total = sum(self.weights.values())
            w = {n: self.weights[n] / total for n in names}

        avg_probs = None
        for name, model in self.models.items():
            p = _get_probs(model, X_tensor)
            avg_probs = p * w[name] if avg_probs is None else avg_probs + p * w[name]
        return avg_probs

    def predict(self, X_tensor):
        return self.predict_proba(X_tensor).argmax(axis=1)


# ---------------------------------------------------------------------------
# McNemar test (H1)
# ---------------------------------------------------------------------------

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    H0: model A and model B make errors on the same samples.
    H1 support: ensemble significantly better than best classical (p < 0.05).
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    # Contingency table: [[both right, only A right], [only B right, both wrong]]
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    table = np.array([[np.sum(correct_a & correct_b), b],
                      [c, np.sum(~correct_a & ~correct_b)]])
    result = mcnemar(table, exact=True)
    log.info("McNemar test: b=%d c=%d p=%.4f | %s",
             b, c, result.pvalue, "PASS" if result.pvalue < 0.05 else "FAIL")
    return result.pvalue


# ---------------------------------------------------------------------------
# H2: parameter count comparison
# ---------------------------------------------------------------------------

def compare_parameter_counts():
    dnn  = DNN(input_dim=8)
    cnn  = CNN1D(input_dim=8)
    lstm = LSTMClassifier(input_dim=8)
    qnn  = HybridQNN()

    counts = {
        "DNN":  sum(p.numel() for p in dnn.parameters()),
        "CNN":  sum(p.numel() for p in cnn.parameters()),
        "LSTM": sum(p.numel() for p in lstm.parameters()),
        "QNN":  qnn.count_params(),
    }
    best_classical = max(counts["DNN"], counts["CNN"], counts["LSTM"])
    ratio = best_classical / counts["QNN"]
    log.info("Parameter counts: %s", counts)
    log.info("H2 ratio (best classical / QNN): %.2f× | threshold ≥ 3×: %s",
             ratio, "PASS" if ratio >= 3 else "FAIL")
    return counts, ratio


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(data_dir="data/processed/cicids2017"):
    X_test  = torch.FloatTensor(np.load(os.path.join(data_dir, "X_test.npy")))
    yb_test = np.load(os.path.join(data_dir, "yb_test.npy"))

    ensemble = SoftVotingEnsemble()
    y_pred_ens = ensemble.predict(X_test)

    # Best classical (DNN)
    dnn = _load_model(DNN, "models/classical/dnn_weights.pth", input_dim=8)
    y_pred_dnn = _get_probs(dnn, X_test).argmax(axis=1)

    acc_ens = accuracy_score(yb_test, y_pred_ens)
    f1_ens  = f1_score(yb_test, y_pred_ens, average="weighted")
    f1_dnn  = f1_score(yb_test, y_pred_dnn, average="weighted")

    log.info("Ensemble accuracy: %.4f | F1: %.4f", acc_ens, f1_ens)
    log.info("DNN F1: %.4f | H1 parity met: %s", f1_dnn, "PASS" if f1_ens >= f1_dnn else "FAIL")
    log.info("\n%s", classification_report(yb_test, y_pred_ens, target_names=["Benign", "Attack"]))

    pvalue = mcnemar_test(yb_test, y_pred_dnn, y_pred_ens)
    compare_parameter_counts()

    return y_pred_ens, pvalue


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate()
