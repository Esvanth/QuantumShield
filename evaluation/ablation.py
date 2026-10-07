"""
QuantumShield – Ablation Study
Compares: classical-only vs quantum-only vs hybrid ensemble
to show the contribution of each component.
"""

import logging
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

from models.classical.dnn  import DNN
from models.classical.cnn  import CNN1D
from models.classical.lstm import LSTMClassifier
from models.quantum.circuits import HybridQNN
from models.ensemble import SoftVotingEnsemble, _get_probs, _load_model
from evaluation.metrics import compare_models

log = logging.getLogger(__name__)
DATA_DIR = "data/processed/cicids2017"


def run_ablation():
    X_test  = torch.FloatTensor(np.load(f"{DATA_DIR}/X_test.npy"))
    yb_test = np.load(f"{DATA_DIR}/yb_test.npy")

    results = {}

    # Individual models
    for name, cls, path in [
        ("DNN",  DNN,  "models/classical/dnn_weights.pth"),
        ("CNN",  CNN1D, "models/classical/cnn_weights.pth"),
        ("LSTM", LSTMClassifier, "models/classical/lstm_weights.pth"),
        ("QNN",  HybridQNN, "models/quantum/qnn_weights.pth"),
    ]:
        kwargs = {"input_dim": 8} if name != "QNN" else {}
        model = _load_model(cls, path, **kwargs)
        probs = _get_probs(model, X_test)
        preds = probs.argmax(axis=1)
        f1  = f1_score(yb_test, preds, zero_division=0)
        acc = accuracy_score(yb_test, preds)
        results[name] = {"f1": f1, "accuracy": acc, "auroc": None}
        log.info("%-6s | acc=%.4f | f1=%.4f", name, acc, f1)

    # Classical-only ensemble (DNN + CNN + LSTM)
    ens_classical = SoftVotingEnsemble(weights={"dnn": 1, "cnn": 1, "lstm": 1, "qnn": 0})
    p_cl = ens_classical.predict_proba(X_test)
    y_cl = p_cl.argmax(axis=1)
    results["Classical Ensemble"] = {
        "f1": f1_score(yb_test, y_cl, zero_division=0),
        "accuracy": accuracy_score(yb_test, y_cl),
        "auroc": None,
    }
    log.info("Classical Ensemble | f1=%.4f", results["Classical Ensemble"]["f1"])

    # Full hybrid ensemble
    ens_full = SoftVotingEnsemble()
    p_full = ens_full.predict_proba(X_test)
    y_full = p_full.argmax(axis=1)
    results["Hybrid Ensemble"] = {
        "f1": f1_score(yb_test, y_full, zero_division=0),
        "accuracy": accuracy_score(yb_test, y_full),
        "auroc": None,
    }
    log.info("Hybrid Ensemble    | f1=%.4f", results["Hybrid Ensemble"]["f1"])

    compare_models(results, save_dir="results")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_ablation()
