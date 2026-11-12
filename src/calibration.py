import json
import numpy as np
from pathlib import Path
from sklearn.metrics import brier_score_loss, log_loss

FILE_MAP = {
    "DNN":       "dnn",
    "CNN-1D":    "cnn1d",
    "LSTM":      "lstm",
    "HybridQNN": "hybridqnn",
}

WEIGHTS = {"DNN": 1, "CNN-1D": 4, "LSTM": 3, "HybridQNN": 1}


def ece(y_true, y_prob, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.digitize(y_prob, bins[1:-1])
    val = 0.0
    for i in range(n_bins):
        mask = ids == i
        if mask.sum() == 0:
            continue
        val += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(val)


def all_metrics(y_true, y_prob):
    y_clip = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return {
        "ece_15bin": ece(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "nll": float(log_loss(y_true, y_clip)),
    }


def load_corpus_probs(corpus):
    probs = {}
    for display, stem in FILE_MAP.items():
        path = Path(f"results/{corpus}_{stem}_test_probs.npy")
        if path.exists():
            probs[display] = np.load(path)
        else:
            print(f"missing: {path}")
    total = sum(WEIGHTS.values())
    ens = sum(WEIGHTS[m] * probs[m] for m in probs) / total
    probs["Ensemble"] = ens
    return probs


def run_corpus(corpus):
    y_true = np.load(f"data/processed/{corpus}/yb_test.npy")
    probs = load_corpus_probs(corpus)
    metrics = {}
    print(f"\n=== {corpus} ===")
    for m, y_prob in probs.items():
        metrics[m] = all_metrics(y_true, y_prob)
        print(f"{m:12s} ECE {metrics[m]['ece_15bin']:.4f}  "
              f"Brier {metrics[m]['brier']:.4f}  "
              f"NLL {metrics[m]['nll']:.4f}")
    out = Path(f"results/{corpus}_calibration_metrics.json")
    out.write_text(json.dumps(metrics, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    run_corpus("cicids2017")
    run_corpus("ciciot2023")