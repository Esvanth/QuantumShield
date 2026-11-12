"""
Noise-robustness sweep for HybridQNN (batched for default.mixed).

Processes test flows in small batches because default.mixed cannot hold
15,000 density matrices in memory at once. Takes longer per corpus but
runs end to end.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score

import models.quantum.circuits as C
from models.quantum.circuits import HybridQNN

WEIGHTS = {
    "cicids2017": "models/weights/cicids2017/hybrid_qnn.pth",
    "ciciot2023": "models/weights/ciciot2023/hybrid_qnn.pth",
}
NOISE_LEVELS = [0.0, 0.001, 0.01, 0.05]
PRETTY = {"cicids2017": "CICIDS2017", "ciciot2023": "CICIoT2023"}

# Subsample for speed. 2000 is statistically ample to show the noise trend.
TEST_SUBSET = 2000
BATCH_SIZE = 32   # default.mixed memory-safe batch


def evaluate(corpus, noise_p):
    C.NOISE_P = float(noise_p)
    model = HybridQNN()
    model.load_state_dict(torch.load(WEIGHTS[corpus], map_location="cpu"))
    model.eval()

    X = np.load(f"data/processed/{corpus}/X_test.npy")
    y = np.load(f"data/processed/{corpus}/yb_test.npy")

    # Stratified subsample so both classes are represented
    rng = np.random.default_rng(42)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_each = TEST_SUBSET // 2
    pos_sel = rng.choice(pos_idx, min(n_each, len(pos_idx)), replace=False)
    neg_sel = rng.choice(neg_idx, min(n_each, len(neg_idx)), replace=False)
    sel = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(sel)
    X, y = X[sel], y[sel]

    all_probs = []
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        X_scaled = model.scale(X_t)
        for i in range(0, len(X_scaled), BATCH_SIZE):
            batch = X_scaled[i:i + BATCH_SIZE]
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs)
    pred = (probs >= 0.5).astype(int)

    return {
        "noise_p": float(noise_p),
        "n_test": int(len(y)),
        "f1": float(f1_score(y, pred)),
        "auroc": float(roc_auc_score(y, probs)),
    }


def run_corpus(corpus):
    print(f"\n=== {corpus} ===")
    rows = []
    for p in NOISE_LEVELS:
        r = evaluate(corpus, p)
        print(f"p={p:.4f}:  F1 {r['f1']:.4f}  AUROC {r['auroc']:.4f}  "
              f"(n={r['n_test']})")
        rows.append(r)
    Path(f"results/{corpus}_noise_sweep.json").write_text(
        json.dumps(rows, indent=2))
    return rows


def plot_sweep(all_rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colours = {"cicids2017": "#0072B2", "ciciot2023": "#009E73"}
    for corpus, rows in all_rows.items():
        ps = [r["noise_p"] for r in rows]
        f1s = [r["f1"] for r in rows]
        aucs = [r["auroc"] for r in rows]
        ax1.plot(ps, f1s, "o-", color=colours[corpus],
                 label=PRETTY[corpus], linewidth=1.5, markersize=6)
        ax2.plot(ps, aucs, "o-", color=colours[corpus],
                 label=PRETTY[corpus], linewidth=1.5, markersize=6)
    for ax, title in ((ax1, "F1"), (ax2, "AUROC")):
        ax.set_xlabel("depolarising probability p")
        ax.set_ylabel(title)
        ax.set_title(f"HybridQNN {title} vs noise level")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path("results/noise_sweep.png")
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    results = {corpus: run_corpus(corpus) for corpus in WEIGHTS}
    plot_sweep(results)