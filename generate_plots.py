"""
QuantumShield - figure generation for the report.

Reads only artefacts already written by the training and evaluation runs, so it
never retrains anything and takes seconds. Every figure is saved at 300 dpi into
results/figures/ as PNG, plus PDF for vector-quality insertion into the report.

Palette is Okabe-Ito, validated for deuteranopia and tritanopia separation so the
figures survive both colourblind readers and greyscale printing. Series are also
distinguished by line style and marker, so colour is never the only encoding.

Usage:
    python generate_plots.py
    python generate_plots.py --tag ciciot2023
"""

import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, f1_score, average_precision_score)

OUT_DIR = "results/figures"

# Okabe-Ito, ordered so no adjacent pair falls below the CVD separation floor.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
GREY_INK = "#3B3B3B"
GREY_GRID = "#DADADA"

# Secondary encoding, so the figures still read in greyscale
STYLES = [("-", "o"), ("--", "s"), ("-.", "^"), (":", "D"), ("-", "v"), ("--", "P")]

MODELS = [
    ("DNN", "dnn"),
    ("CNN-1D", "cnn1d"),
    ("LSTM", "lstm"),
    ("HybridQNN", "hybridqnn"),
]


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": GREY_INK,
        "axes.labelcolor": GREY_INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GREY_GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "xtick.color": GREY_INK,
        "ytick.color": GREY_INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
    })


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def load_probs(tag):
    """Return {display name: P(attack)} for every model plus both ensembles."""
    probs = {}
    for display, slug in MODELS:
        p = f"results/{tag}_{slug}_test_probs.npy"
        if os.path.exists(p):
            probs[display] = np.load(p)
        else:
            print(f"  missing {p}, skipping {display}")
    for display, fname in [("Ensemble (equal)", f"results/{tag}_ensemble_equal_probs.npy"),
                           ("Ensemble (weighted)", f"results/{tag}_ensemble_weighted_probs.npy")]:
        if os.path.exists(fname):
            probs[display] = np.load(fname)
    return probs


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def history_path(stem, tag):
    """
    Prefer the dataset-tagged copy, fall back to the untagged one.

    The quantum training scripts write history files without a dataset tag, so
    whichever dataset ran last owns the untagged name. run_ciciot2023.py files
    tagged copies, which is what makes plotting either dataset possible at any
    time rather than only right after its own run.
    """
    tagged = os.path.join("results", f"{tag}_{stem}")
    if os.path.exists(tagged):
        return tagged
    return os.path.join("results", stem)



def fig_roc(y, probs):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, (name, p) in enumerate(probs.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ls, _ = STYLES[i % len(STYLES)]
        ax.plot(fpr, tpr, ls, color=PALETTE[i % len(PALETTE)],
                label=f"{name} (AUC {auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], color=GREY_GRID, linewidth=1, zorder=0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the held-out test set")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    save(fig, "fig01_roc_curves")

    # Zoomed view, because every curve is pinned to the top-left corner
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, (name, p) in enumerate(probs.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ls, _ = STYLES[i % len(STYLES)]
        ax.plot(fpr, tpr, ls, color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves, detail of the operating region")
    ax.set_xlim(0, 0.10); ax.set_ylim(0.85, 1.005)
    ax.legend(loc="lower right")
    save(fig, "fig02_roc_curves_zoom")


def fig_pr(y, probs):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, (name, p) in enumerate(probs.items()):
        prec, rec, _ = precision_recall_curve(y, p)
        ls, _ = STYLES[i % len(STYLES)]
        ax.plot(rec, prec, ls, color=PALETTE[i % len(PALETTE)],
                label=f"{name} (AP {average_precision_score(y, p):.4f})")
    base = y.mean()
    ax.axhline(base, color=GREY_GRID, linewidth=1, zorder=0)
    ax.annotate(f"no-skill baseline ({base:.3f})", (0.98, base + 0.02),
                ha="right", va="bottom", color=GREY_INK, fontsize=8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves on the held-out test set")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    save(fig, "fig03_precision_recall")


def fig_confusion(y, probs, results):
    """Confusion matrices for the best single model and the weighted ensemble."""
    wanted = [n for n in ("CNN-1D", "Ensemble (weighted)") if n in probs]
    if not wanted:
        return
    key = {"CNN-1D": "CNN-1D", "Ensemble (weighted)": "Ensemble-weighted"}

    fig, axes = plt.subplots(1, len(wanted), figsize=(5.2 * len(wanted), 4.4))
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, wanted):
        thr = results[key[name]]["threshold"]
        cm = confusion_matrix(y, (probs[name] >= thr).astype(int))
        # Sequential single hue for magnitude, never a rainbow
        ax.imshow(cm, cmap="Blues", aspect="equal")
        for r in range(2):
            for c in range(2):
                pct = 100 * cm[r, c] / cm[r].sum()
                ax.text(c, r, f"{cm[r, c]:,}\n{pct:.2f}%", ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else GREY_INK,
                        fontsize=10)
        ax.set_xticks([0, 1], ["Benign", "Attack"])
        ax.set_yticks([0, 1], ["Benign", "Attack"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{name}  (threshold {thr:.3f})")
        ax.grid(False)

    fig.suptitle("Confusion matrices at the validation-fitted thresholds", y=1.02)
    save(fig, "fig04_confusion_matrices")


def fig_model_comparison(results):
    names = [n for n in ["DNN", "CNN-1D", "LSTM", "HybridQNN",
                         "Ensemble-equal", "Ensemble-weighted"] if n in results]
    metrics = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1"]

    x = np.arange(len(names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        vals = [results[n][m] for n in names]
        # 2px surface gap between adjacent bars
        bars = ax.bar(x + (i - 1) * width, vals, width * 0.92,
                      label=lab, color=PALETTE[i], edgecolor="white", linewidth=1.5)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2, color=GREY_INK)

    ax.set_xticks(x, [n.replace("Ensemble-", "Ens. ") for n in names], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0.80, 1.0)
    ax.set_title("Test-set performance at validation-fitted thresholds")
    ax.legend(ncol=3, loc="upper left")
    save(fig, "fig05_model_comparison")


def fig_params_vs_f1(results):
    """H2 made visual: F1 against parameter count on a log scale."""
    names = [n for n in ["DNN", "CNN-1D", "LSTM", "HybridQNN"] if n in results]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    pvals = [results[n]["parameters"] for n in names]
    fvals = [results[n]["f1"] for n in names]

    # Direct labels carry identity, so no legend box is needed. Labels alternate
    # above and below the marker, ordered by parameter count, so neighbouring
    # points on the log axis cannot collide.
    order = sorted(range(len(names)), key=lambda k: pvals[k])

    for slot, i in enumerate(order):
        n, p, f1 = names[i], pvals[i], fvals[i]
        _, mk = STYLES[i % len(STYLES)]
        ax.scatter(p, f1, s=150, marker=mk, color=PALETTE[i % len(PALETTE)],
                   edgecolor="white", linewidth=1.5, zorder=3)

        below = (slot % 2 == 0)
        dy, va = (-26, "top") if below else (26, "bottom")
        if p == min(pvals):
            ha, dx = "left", 12
        elif p == max(pvals):
            ha, dx = "right", -12
        else:
            ha, dx = "center", 0
        ax.annotate(f"{n}\n{p:,} params", (p, f1), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va=va, fontsize=8, color=GREY_INK)

    ax.set_xscale("log")
    ax.set_xlim(min(pvals) / 3.5, max(pvals) * 3.5)
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Test F1")
    ax.set_title("Parameter efficiency: F1 against model size")
    spread = max(max(fvals) - min(fvals), 0.02)
    ax.set_ylim(min(fvals) - spread * 0.55, max(fvals) + spread * 0.55)
    save(fig, "fig06_parameters_vs_f1")


def fig_hybrid_convergence(tag):
    lp = history_path("hybrid_loss_history.npy", tag)
    fp = history_path("hybrid_f1_history.npy", tag)
    if not (os.path.exists(lp) and os.path.exists(fp)):
        print("  hybrid history missing, skipping convergence figure")
        return
    loss, f1 = np.load(lp), np.load(fp)
    ep = np.arange(1, len(loss) + 1)

    # Two stacked panels rather than a dual y-axis, which misleads by construction
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True)
    a1.plot(ep, loss, "-o", color=PALETTE[0], markersize=4)
    a1.set_ylabel("Weighted cross-entropy")
    a1.set_title("HybridQNN training convergence")

    a2.plot(ep, f1, "--s", color=PALETTE[1], markersize=4)
    a2.set_ylabel("Training F1")
    a2.set_xlabel("Epoch")
    a2.set_xticks(ep[::max(1, len(ep) // 15)])
    save(fig, "fig07_hybridqnn_convergence")


def fig_qae_convergence(tag):
    p = history_path("qae_loss_history.npy", tag)
    if not os.path.exists(p):
        print("  QAE loss history missing, skipping")
        return
    loss = np.load(p)
    ep = np.arange(1, len(loss) + 1)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(ep, loss, "-o", color=PALETTE[2], markersize=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Quantum autoencoder training convergence (benign traffic only)")
    ax.set_xticks(ep[::max(1, len(ep) // 15)])
    save(fig, "fig08_qae_convergence")


def fig_qae_scores(tag):
    sp = history_path("qae_scores.npy", tag)
    lp = history_path("qae_labels.npy", tag)
    if not (os.path.exists(sp) and os.path.exists(lp)):
        print("  QAE scores missing, skipping")
        return
    s, y = np.load(sp), np.load(lp)
    hi = np.quantile(s, 0.995)
    bins = np.linspace(0, hi, 90)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.hist(s[y == 0], bins=bins, color=PALETTE[0], alpha=0.75,
            label=f"Benign (mean {s[y == 0].mean():.4f})")
    ax.hist(s[y == 1], bins=bins, color=PALETTE[1], alpha=0.75,
            label=f"Attack (mean {s[y == 1].mean():.4f})")
    ax.set_yscale("log")
    ax.set_xlabel("Reconstruction error (anomaly score)")
    ax.set_ylabel("Samples (log scale)")
    ax.set_title("QAE anomaly score distribution")
    ax.legend()
    save(fig, "fig09_qae_score_distribution")

    fpr, tpr, _ = roc_curve(y, s)
    a = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.plot(fpr, tpr, color=PALETTE[2], label=f"QAE (AUROC {a:.4f})")
    ax.plot([0, 1], [0, 1], color=GREY_GRID, linewidth=1, zorder=0)
    ax.axhline(0, color="none")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("H4: unsupervised attack detection by the quantum autoencoder")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    save(fig, "fig10_qae_roc")


def fig_ensemble_comparison(results, h1):
    """H1 made visual: why equal weighting failed and weighting fixed it."""
    names, vals = [], []
    for n in ["DNN", "CNN-1D", "LSTM", "HybridQNN",
              "Ensemble-equal", "Ensemble-weighted"]:
        if n in results:
            names.append(n.replace("Ensemble-", "Ensemble, "))
            vals.append(results[n]["f1"])

    best = h1["best_single_f1"]
    colors = [PALETTE[5]] * len(names)
    for i, n in enumerate(names):
        if n.startswith("Ensemble, equal"):
            colors[i] = PALETTE[1]
        elif n.startswith("Ensemble, weighted"):
            colors[i] = PALETTE[2]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bars = ax.bar(names, vals, 0.6, color=colors, edgecolor="white", linewidth=1.5)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3, color=GREY_INK)
    ax.axhline(best, color=GREY_INK, linestyle="--", linewidth=1)
    lo = min(vals) - 0.015
    hi = max(vals) + 0.022
    ax.annotate(f"best single model: {h1['best_single_model']} {best:.4f}",
                (-0.45, hi - 0.004), ha="left", va="top", fontsize=8,
                color=GREY_INK)
    ax.set_ylabel("Test F1")
    ax.set_ylim(lo, hi)
    ax.set_title("H1: equal weighting underperforms the best member, fitted weighting does not")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=9)
    save(fig, "fig11_ensemble_comparison")


def fig_threshold_sensitivity(y, probs, results):
    """Shows why the threshold was tuned rather than left at 0.5."""
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    grid = np.linspace(0.02, 0.98, 97)

    for i, name in enumerate([n for n in ("HybridQNN", "Ensemble (weighted)")
                              if n in probs]):
        p = probs[name]
        f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
        ls, _ = STYLES[i % len(STYLES)]
        ax.plot(grid, f1s, ls, color=PALETTE[i], label=name)
        key = "Ensemble-weighted" if name.startswith("Ensemble") else name
        thr = results[key]["threshold"]
        ax.scatter([thr], [results[key]["f1"]], s=90, zorder=3,
                   color=PALETTE[i], edgecolor="white", linewidth=1.5)

    ax.axvline(0.5, color=GREY_GRID, linewidth=1, zorder=0)
    ax.annotate("default 0.5", (0.512, 0.055), fontsize=8, va="bottom",
                color=GREY_INK)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Test F1")
    ax.set_title("Threshold sensitivity, with the validation-fitted operating points marked")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    save(fig, "fig12_threshold_sensitivity")


def fig_ablation():
    p = "results/ablation_results.json"
    if not os.path.exists(p):
        print("  ablation results missing, skipping (run models.quantum.ablation)")
        return
    with open(p) as f:
        rows = json.load(f)

    names = [r["architecture"].split(" (")[0] for r in rows]
    f1s = [r["f1"] for r in rows]
    qp = [r["quantum_params"] for r in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    b1 = a1.bar(names, f1s, 0.55, color=PALETTE[:len(names)],
                edgecolor="white", linewidth=1.5)
    a1.bar_label(b1, fmt="%.4f", fontsize=8, padding=3, color=GREY_INK)
    a1.set_ylabel("Test F1")
    a1.set_ylim(min(f1s) - 0.05, max(f1s) + 0.03)
    a1.set_title("Test F1 by quantum architecture")

    b2 = a2.bar(names, qp, 0.55, color=PALETTE[:len(names)],
                edgecolor="white", linewidth=1.5)
    a2.bar_label(b2, fmt="%d", fontsize=8, padding=3, color=GREY_INK)
    a2.set_ylabel("Quantum circuit parameters")
    a2.set_title("Circuit parameter count")

    for ax in (a1, a2):
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)

    fig.suptitle("Quantum architecture ablation under identical training conditions",
                 y=1.03)
    save(fig, "fig13_architecture_ablation")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cicids2017")
    args = ap.parse_args()
    tag = args.tag

    global OUT_DIR
    OUT_DIR = os.path.join("results", "figures", tag)

    setup_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    hyp_path = f"results/{tag}_hypotheses.json"
    if not os.path.exists(hyp_path):
        raise SystemExit(f"{hyp_path} not found. Run evaluation.hypotheses first.")
    with open(hyp_path) as f:
        hyp = json.load(f)
    results = hyp["per_model"]

    labels_path = f"results/{tag}_test_labels.npy"
    y = np.load(labels_path)
    probs = load_probs(tag)
    print(f"generating figures for {tag}: {len(y):,} test rows, {len(probs)} models")

    fig_roc(y, probs)
    fig_pr(y, probs)
    fig_confusion(y, probs, results)
    fig_model_comparison(results)
    fig_params_vs_f1(results)
    fig_hybrid_convergence(tag)
    fig_qae_convergence(tag)
    fig_qae_scores(tag)
    fig_ensemble_comparison(results, hyp["H1"])
    fig_threshold_sensitivity(y, probs, results)
    fig_ablation()

    print(f"\nall figures written to {OUT_DIR}/ as PNG (300 dpi) and PDF")
    print("SHAP figures are already in results/shap_plots/")


if __name__ == "__main__":
    main()