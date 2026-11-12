"""
Abstention mechanism evaluation (RQ2).

Uses the cached ensemble member probabilities. Computes the spread between
member probabilities, picks an abstention threshold on the first half of
the test set, evaluates on the second half, and runs H5 and H6 tests.

Outputs:
  - results/<corpus>_abstention.json : the metrics for the methodology/results
  - Prints tau_abstain, coverage, accuracy lift, Mann-Whitney U for H5.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from scipy.stats import mannwhitneyu

FILE_MAP = {"DNN": "dnn", "CNN-1D": "cnn1d", "LSTM": "lstm", "HybridQNN": "hybridqnn"}
WEIGHTS = {"DNN": 1, "CNN-1D": 4, "LSTM": 3, "HybridQNN": 1}
THRESHOLD = 0.468


def load_probs(corpus):
    return {d: np.load(f"results/{corpus}_{s}_test_probs.npy")
            for d, s in FILE_MAP.items()}


def ensemble(probs):
    total = sum(WEIGHTS.values())
    return sum(WEIGHTS[m] * probs[m] for m in FILE_MAP) / total


def spread_vec(probs):
    stacked = np.stack([probs[m] for m in FILE_MAP], axis=0)
    return stacked.max(axis=0) - stacked.min(axis=0)


def pick_tau(val_spread, val_correct, min_cov=0.80):
    best_tau, best_acc = 1.0, 0.0
    for tau in np.linspace(0.05, 0.95, 46):
        accepted = val_spread <= tau
        if accepted.mean() < min_cov or accepted.sum() == 0:
            continue
        acc = val_correct[accepted].mean()
        if acc > best_acc:
            best_tau, best_acc = tau, acc
    return float(best_tau), float(best_acc)


def run_corpus(corpus):
    y = np.load(f"data/processed/{corpus}/yb_test.npy")
    probs = load_probs(corpus)
    ens = ensemble(probs)
    pred = (ens >= THRESHOLD).astype(int)
    correct = (pred == y).astype(int)
    spr = spread_vec(probs)

    n = len(y)
    half = n // 2
    tau, _ = pick_tau(spr[:half], correct[:half])

    eval_spr, eval_corr = spr[half:], correct[half:]
    eval_y, eval_ens = y[half:], ens[half:]

    base_acc = eval_corr.mean()
    base_f1 = f1_score(eval_y, (eval_ens >= THRESHOLD).astype(int))

    accepted = eval_spr <= tau
    cov = accepted.mean()
    acc_acc = eval_corr[accepted].mean() if accepted.sum() else 0.0
    acc_f1 = (f1_score(eval_y[accepted],
              (eval_ens[accepted] >= THRESHOLD).astype(int),
              zero_division=0) if accepted.sum() else 0.0)

    wrong_spr = spr[correct == 0]
    right_spr = spr[correct == 1]
    u, p = mannwhitneyu(wrong_spr, right_spr, alternative="greater")

    results = {
        "tau_abstain": float(tau),
        "coverage": float(cov),
        "baseline_accuracy": float(base_acc),
        "accepted_accuracy": float(acc_acc),
        "accuracy_lift": float(acc_acc - base_acc),
        "baseline_f1": float(base_f1),
        "accepted_f1": float(acc_f1),
        "H5_mannwhitney": {
            "u": float(u),
            "p_value": float(p),
            "wrong_spread_mean": float(wrong_spr.mean()),
            "right_spread_mean": float(right_spr.mean()),
        }
    }

    print(f"\n=== {corpus} ===")
    print(f"tau_abstain         {tau:.4f}")
    print(f"coverage            {cov:.4f}")
    print(f"baseline accuracy   {base_acc:.4f}")
    print(f"accepted accuracy   {acc_acc:.4f}")
    print(f"accuracy lift       {acc_acc - base_acc:+.4f}")
    print(f"baseline F1         {base_f1:.4f}")
    print(f"accepted F1         {acc_f1:.4f}")
    print(f"H5 Mann-Whitney U   {u:.1f}")
    print(f"H5 p-value          {p:.4e}")
    print(f"wrong spread mean   {wrong_spr.mean():.4f}")
    print(f"right spread mean   {right_spr.mean():.4f}")

    Path(f"results/{corpus}_abstention.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    run_corpus("cicids2017")
    run_corpus("ciciot2023")