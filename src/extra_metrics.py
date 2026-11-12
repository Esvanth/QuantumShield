import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve

FILE_MAP = {"DNN": "dnn", "CNN-1D": "cnn1d", "LSTM": "lstm", "HybridQNN": "hybridqnn"}
WEIGHTS = {"DNN": 1, "CNN-1D": 4, "LSTM": 3, "HybridQNN": 1}


def load_ensemble(corpus):
    probs = {d: np.load(f"results/{corpus}_{s}_test_probs.npy")
             for d, s in FILE_MAP.items()}
    total = sum(WEIGHTS.values())
    return sum(WEIGHTS[m] * probs[m] for m in FILE_MAP) / total


def tpr_at_fpr(y_true, y_prob, target):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    idx = np.argmin(np.abs(fpr - target))
    return float(tpr[idx]), float(fpr[idx]), float(thr[idx])


def run_corpus(corpus):
    y_bin = np.load(f"data/processed/{corpus}/yb_test.npy")
    ens = load_ensemble(corpus)
    print(f"\n=== {corpus} ===")

    results = {}
    for t in (0.001, 0.01):
        tpr, fpr_actual, thr = tpr_at_fpr(y_bin, ens, t)
        results[f"tpr_at_fpr_{t}"] = {"tpr": tpr, "fpr_actual": fpr_actual,
                                      "threshold": thr}
        print(f"TPR at FPR={t:.4f}: {tpr:.4f} (actual FPR {fpr_actual:.5f}, "
              f"thr {thr:.4f})")

    y_multi_path = Path(f"data/processed/{corpus}/y_test.npy")
    if y_multi_path.exists():
        y_multi = np.load(y_multi_path)
        pred = (ens >= 0.468).astype(int)
        per_class = {}
        for cls in np.unique(y_multi):
            if cls == 0:
                continue
            mask = y_multi == cls
            if mask.sum() == 0:
                continue
            tp = int(((pred == 1) & mask).sum())
            fn = int(((pred == 0) & mask).sum())
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            per_class[int(cls)] = {"count": int(mask.sum()),
                                   "recall": float(recall)}
            print(f"class {cls:3d}: n={mask.sum():6d}  recall {recall:.4f}")
        results["per_class_recall"] = per_class
    else:
        print("multi-class y_test.npy not found, per-class skipped")

    Path(f"results/{corpus}_extra_metrics.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    run_corpus("cicids2017")
    run_corpus("ciciot2023")