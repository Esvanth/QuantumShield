import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score

FILE_MAP = {
    "DNN":       "dnn",
    "CNN-1D":    "cnn1d",
    "LSTM":      "lstm",
    "HybridQNN": "hybridqnn",
}

WEIGHTS = {"DNN": 1, "CNN-1D": 4, "LSTM": 3, "HybridQNN": 1}
THRESHOLD = 0.468


def load_probs(corpus):
    probs = {}
    for display, stem in FILE_MAP.items():
        probs[display] = np.load(f"results/{corpus}_{stem}_test_probs.npy")
    return probs


def weighted(probs, weights):
    total = sum(weights.values())
    return sum(weights[m] * probs[m] for m in weights) / total


def run_corpus(corpus):
    y_true = np.load(f"data/processed/{corpus}/yb_test.npy")
    probs = load_probs(corpus)

    base_p = weighted(probs, WEIGHTS)
    base_f1 = f1_score(y_true, (base_p >= THRESHOLD).astype(int))
    base_auc = roc_auc_score(y_true, base_p)

    results = {"baseline": {"f1": float(base_f1), "auroc": float(base_auc)}}
    print(f"\n=== {corpus} ===")
    print(f"baseline (all four)      F1 {base_f1:.4f}  AUROC {base_auc:.4f}")

    for drop in FILE_MAP:
        w = {m: WEIGHTS[m] for m in FILE_MAP if m != drop}
        p = {m: probs[m] for m in FILE_MAP if m != drop}
        ens = weighted(p, w)
        f1 = f1_score(y_true, (ens >= THRESHOLD).astype(int))
        auc = roc_auc_score(y_true, ens)
        results[f"drop_{drop}"] = {
            "f1": float(f1),
            "auroc": float(auc),
            "f1_drop": float(base_f1 - f1),
            "auroc_drop": float(base_auc - auc),
        }
        print(f"drop {drop:12s}        F1 {f1:.4f} ({f1 - base_f1:+.4f})  "
              f"AUROC {auc:.4f} ({auc - base_auc:+.4f})")

    out = Path(f"results/{corpus}_h5_ablation.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    run_corpus("cicids2017")
    run_corpus("ciciot2023")