import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

FILE_MAP = {
    "DNN":       "dnn",
    "CNN-1D":    "cnn1d",
    "LSTM":      "lstm",
    "HybridQNN": "hybridqnn",
}
WEIGHTS = {"DNN": 1, "CNN-1D": 4, "LSTM": 3, "HybridQNN": 1}

COLOURS = {
    "DNN":       "#0072B2",
    "CNN-1D":    "#009E73",
    "LSTM":      "#E69F00",
    "HybridQNN": "#CC79A7",
    "Ensemble":  "#D55E00",
}


def reliability(y_true, y_prob, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.digitize(y_prob, bins[1:-1])
    conf, acc = [], []
    for i in range(n_bins):
        mask = ids == i
        if mask.sum() == 0:
            continue
        conf.append(y_prob[mask].mean())
        acc.append(y_true[mask].mean())
    return conf, acc


def plot(corpus, pretty):
    y_true = np.load(f"data/processed/{corpus}/yb_test.npy")
    probs = {d: np.load(f"results/{corpus}_{s}_test_probs.npy")
             for d, s in FILE_MAP.items()}
    total = sum(WEIGHTS.values())
    probs["Ensemble"] = sum(WEIGHTS[m] * probs[m] for m in FILE_MAP) / total
    metrics = json.load(open(f"results/{corpus}_calibration_metrics.json"))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="perfect")
    for name, y_prob in probs.items():
        c, a = reliability(y_true, y_prob)
        ece = metrics[name]["ece_15bin"]
        ax.plot(c, a, "o-", color=COLOURS[name],
                label=f"{name} (ECE {ece:.4f})",
                linewidth=1.5, markersize=5)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction of positives")
    ax.set_title(f"Reliability on {pretty}")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(f"results/{corpus}_calibration.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    plot("cicids2017", "CICIDS2017")
    plot("ciciot2023", "CICIoT2023")