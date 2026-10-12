"""
QuantumShield - Hypothesis evaluation.

H1  Detection parity        : the soft-voting ensemble beats the best single model
                              on F1, and McNemar's test confirms the difference is
                              statistically significant (p < 0.05)
H2  Parameter efficiency    : HybridQNN reaches comparable F1 with >= 3x fewer parameters
H3  Explanation plausibility: SHAP attributions agree with domain expectation
H4  Zero-day robustness     : QAE reaches AUROC >= 0.85 without supervised attack labels

Methodology notes
-----------------
Decision thresholds and ensemble weights are both fitted on the VALIDATION
split and then applied once to the test split. Fitting either on test would be
leakage, so the test set is used for measurement only.

Two ensembles are reported. The equal-weight ensemble is the naive baseline.
The weighted ensemble searches a small integer weight grid on validation, with
every member constrained to a weight of at least 1 so that no model is silently
dropped from the architecture.

Model probabilities are cached to results/ so that reruns do not repeat the
six minutes of quantum circuit simulation. Pass --refresh to recompute.

Usage:
    python -m evaluation.hypotheses
    python -m evaluation.hypotheses --refresh
    python -m evaluation.hypotheses --data-dir data/processed/ciciot2023
"""

import os
import json
import argparse
import logging
import itertools

import numpy as np
import torch
from scipy.stats import chi2
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from models.classical.dnn import DNN
from models.classical.cnn import CNN1D
from models.classical.lstm import LSTMClassifier
from models.quantum.circuits import HybridQNN, N_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# H2 criterion. The parameter ratio comes from the original specification.
# The performance floor is the fraction of the best classical F1 the quantum
# model must retain for "comparable" to mean anything. Set this on principle
# and record the reasoning in the specification document.
COMPARABLE_MARGIN = 0.90
MIN_PARAM_RATIO   = 3.0
H4_AUROC_TARGET   = 0.85
ALPHA             = 0.05

# Weight search: each member gets an integer weight from this set, so every
# model keeps a non-zero vote and the chosen weights stay interpretable.
WEIGHT_CHOICES = (1, 2, 3, 4)
WEIGHT_SEARCH_ROWS = 120000     # validation subsample used for the weight grid
WEIGHT_SEARCH_GRID = 60         # threshold resolution during the weight search

MODEL_ORDER = ["DNN", "CNN-1D", "LSTM", "HybridQNN"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models():
    """Load all four trained models. Classical ones take raw PCA features;
    the HybridQNN applies its own stored scaler internally."""
    models = {}

    dnn = DNN(input_dim=N_FEATURES)
    dnn.load_state_dict(torch.load("models/classical/dnn_weights.pth", map_location="cpu"))
    models["DNN"] = dnn

    cnn = CNN1D(input_dim=N_FEATURES)
    cnn.load_state_dict(torch.load("models/classical/cnn_weights.pth", map_location="cpu"))
    models["CNN-1D"] = cnn

    lstm = LSTMClassifier(input_dim=N_FEATURES)
    lstm.load_state_dict(torch.load("models/classical/lstm_weights.pth", map_location="cpu"))
    models["LSTM"] = lstm

    hqnn = HybridQNN()
    hqnn.load_state_dict(torch.load("models/saved/hybrid_qnn.pth", map_location="cpu"))
    models["HybridQNN"] = hqnn

    for name in MODEL_ORDER:
        models[name].eval()
        log.info("loaded %-10s %6d trainable parameters", name, param_count(models[name]))

    return models


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def slug(name):
    return name.replace("-", "").lower()


# ---------------------------------------------------------------------------
# Scoring and statistics
# ---------------------------------------------------------------------------

def predict_probs(name, model, X, batch_size=2048):
    """Return P(attack) for every row. Batched so the quantum simulator does
    not allocate one state tensor for the whole split."""
    out = []
    n_batches = (len(X) + batch_size - 1) // batch_size

    with torch.no_grad():
        Xt = torch.FloatTensor(X)
        if name == "HybridQNN":
            Xt = model.scale(Xt)        # the quantum model needs its own scaling
        for b in range(n_batches):
            batch = Xt[b * batch_size:(b + 1) * batch_size]
            out.append(torch.softmax(model(batch), dim=1)[:, 1])
            if name == "HybridQNN" and ((b + 1) % 40 == 0 or b + 1 == n_batches):
                log.info("  %s scoring %d/%d", name, b + 1, n_batches)

    return torch.cat(out).numpy()


def cached_probs(name, model, X, tag, split, refresh):
    """Score a split, reusing the cached array unless --refresh was passed."""
    path = f"results/{tag}_{slug(name)}_{split}_probs.npy"
    if not refresh and os.path.exists(path):
        cached = np.load(path)
        if len(cached) == len(X):
            log.info("  %s %s probabilities loaded from cache", name, split)
            return cached
        log.warning("  cached %s has %d rows but split has %d, recomputing",
                    path, len(cached), len(X))
    p = predict_probs(name, model, X)
    os.makedirs("results", exist_ok=True)
    np.save(path, p)
    return p


def tune_threshold(y, p, grid=200):
    """Pick the threshold that maximises F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, grid):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def search_weights(y_val, val_probs, seed=42):
    """
    Grid search integer weights on the validation split.

    Every member is constrained to a weight of at least 1, so the search tunes
    each model's influence rather than deciding whether it belongs in the
    architecture at all. A subsample keeps the search tractable; the chosen
    weights are re-thresholded on the full validation split afterwards.
    """
    rng = np.random.default_rng(seed)
    n = len(y_val)
    if n > WEIGHT_SEARCH_ROWS:
        idx = rng.choice(n, size=WEIGHT_SEARCH_ROWS, replace=False)
    else:
        idx = np.arange(n)

    y_s = y_val[idx]
    P_s = np.stack([val_probs[m][idx] for m in MODEL_ORDER])   # (4, rows)

    best_w, best_f1 = None, -1.0
    combos = list(itertools.product(WEIGHT_CHOICES, repeat=len(MODEL_ORDER)))
    log.info("searching %d weight combinations on %d validation rows",
             len(combos), len(idx))

    for combo in combos:
        w = np.array(combo, dtype=np.float64)
        w = w / w.sum()
        p = w @ P_s
        _, f1 = tune_threshold(y_s, p, grid=WEIGHT_SEARCH_GRID)
        if f1 > best_f1:
            best_f1, best_w = f1, combo

    weights = {m: int(w) for m, w in zip(MODEL_ORDER, best_w)}
    log.info("best weights %s (subsample val F1 %.4f)", weights, best_f1)
    return weights


def apply_weights(weights, probs):
    w = np.array([weights[m] for m in MODEL_ORDER], dtype=np.float64)
    w = w / w.sum()
    return w @ np.stack([probs[m] for m in MODEL_ORDER])


def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar's test on paired predictions, with continuity correction.
    b = A right and B wrong, c = A wrong and B right.
    Null hypothesis: the two classifiers have the same error rate.
    """
    a_ok = (pred_a == y_true)
    b_ok = (pred_b == y_true)
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return 1.0, b, c
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(chi2.sf(stat, df=1)), b, c


def metrics_at(y_true, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy":  float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall":    float(recall_score(y_true, pred, zero_division=0)),
        "f1":        float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, p)),
        "confusion": confusion_matrix(y_true, pred).tolist(),
    }


# ---------------------------------------------------------------------------
# H1 and H2
# ---------------------------------------------------------------------------

def evaluate_h1_h2(models, data_dir, tag, refresh):
    X_val  = np.load(os.path.join(data_dir, "X_val.npy")).astype(np.float32)[:, :N_FEATURES]
    y_val  = np.load(os.path.join(data_dir, "yb_val.npy")).astype(np.int64)
    X_test = np.load(os.path.join(data_dir, "X_test.npy")).astype(np.float32)[:, :N_FEATURES]
    y_test = np.load(os.path.join(data_dir, "yb_test.npy")).astype(np.int64)

    log.info("validation %d rows, test %d rows", len(y_val), len(y_test))

    val_probs, test_probs, test_preds, results = {}, {}, {}, {}

    for name in MODEL_ORDER:
        log.info("scoring %s", name)
        model = models[name]
        p_val  = cached_probs(name, model, X_val,  tag, "val",  refresh)
        p_test = cached_probs(name, model, X_test, tag, "test", refresh)
        val_probs[name], test_probs[name] = p_val, p_test

        t, val_f1 = tune_threshold(y_val, p_val)
        test_preds[name] = (p_test >= t).astype(int)
        results[name] = metrics_at(y_test, p_test, t)
        results[name]["parameters"] = param_count(model)
        results[name]["val_f1_at_threshold"] = round(val_f1, 4)

        log.info("  %-10s threshold=%.3f  test F1=%.4f  ROC-AUC=%.4f",
                 name, t, results[name]["f1"], results[name]["roc_auc"])

    total_params = sum(param_count(models[m]) for m in MODEL_ORDER)

    # --- equal-weight ensemble (naive baseline) -----------------------------
    p_val_eq  = np.mean([val_probs[m] for m in MODEL_ORDER], axis=0)
    p_test_eq = np.mean([test_probs[m] for m in MODEL_ORDER], axis=0)
    t_eq, val_f1_eq = tune_threshold(y_val, p_val_eq)
    test_preds["Ensemble-equal"] = (p_test_eq >= t_eq).astype(int)
    results["Ensemble-equal"] = metrics_at(y_test, p_test_eq, t_eq)
    results["Ensemble-equal"]["parameters"] = total_params
    results["Ensemble-equal"]["val_f1_at_threshold"] = round(val_f1_eq, 4)
    results["Ensemble-equal"]["weights"] = {m: 1 for m in MODEL_ORDER}
    log.info("  %-14s threshold=%.3f  test F1=%.4f  ROC-AUC=%.4f",
             "Ensemble-equal", t_eq, results["Ensemble-equal"]["f1"],
             results["Ensemble-equal"]["roc_auc"])

    # --- weighted ensemble (weights fitted on validation) -------------------
    weights = search_weights(y_val, val_probs)
    p_val_w  = apply_weights(weights, val_probs)
    p_test_w = apply_weights(weights, test_probs)
    t_w, val_f1_w = tune_threshold(y_val, p_val_w)
    test_preds["Ensemble-weighted"] = (p_test_w >= t_w).astype(int)
    results["Ensemble-weighted"] = metrics_at(y_test, p_test_w, t_w)
    results["Ensemble-weighted"]["parameters"] = total_params
    results["Ensemble-weighted"]["val_f1_at_threshold"] = round(val_f1_w, 4)
    results["Ensemble-weighted"]["weights"] = weights
    log.info("  %-14s threshold=%.3f  test F1=%.4f  ROC-AUC=%.4f",
             "Ensemble-weighted", t_w, results["Ensemble-weighted"]["f1"],
             results["Ensemble-weighted"]["roc_auc"])

    os.makedirs("results", exist_ok=True)
    np.save(f"results/{tag}_ensemble_equal_probs.npy", p_test_eq)
    np.save(f"results/{tag}_ensemble_weighted_probs.npy", p_test_w)
    np.save(f"results/{tag}_test_labels.npy", y_test)

    # --- H1 -----------------------------------------------------------------
    singles = {m: results[m]["f1"] for m in MODEL_ORDER}
    best_single = max(singles, key=singles.get)

    p_eq, b_eq, c_eq = mcnemar_test(y_test, test_preds[best_single],
                                    test_preds["Ensemble-equal"])
    p_w,  b_w,  c_w  = mcnemar_test(y_test, test_preds[best_single],
                                    test_preds["Ensemble-weighted"])

    h1 = {
        "hypothesis": ("H1: the soft-voting ensemble beats the best single model on F1, "
                       f"significant at p < {ALPHA}"),
        "best_single_model": best_single,
        "best_single_f1": singles[best_single],
        "equal_weight": {
            "f1": results["Ensemble-equal"]["f1"],
            "margin": results["Ensemble-equal"]["f1"] - singles[best_single],
            "mcnemar_p": p_eq,
            "discordant_ensemble_only": c_eq,
            "discordant_best_only": b_eq,
            "supported": bool(results["Ensemble-equal"]["f1"] > singles[best_single]
                              and p_eq < ALPHA),
        },
        "weighted": {
            "weights": weights,
            "f1": results["Ensemble-weighted"]["f1"],
            "margin": results["Ensemble-weighted"]["f1"] - singles[best_single],
            "mcnemar_p": p_w,
            "discordant_ensemble_only": c_w,
            "discordant_best_only": b_w,
            "supported": bool(results["Ensemble-weighted"]["f1"] > singles[best_single]
                              and p_w < ALPHA),
        },
    }
    h1["supported"] = h1["weighted"]["supported"]

    # --- H2 -----------------------------------------------------------------
    q_params = results["HybridQNN"]["parameters"]
    q_f1 = results["HybridQNN"]["f1"]
    classical = ["DNN", "CNN-1D", "LSTM"]
    best_classical = max(classical, key=lambda n: results[n]["f1"])
    ratios = {n: results[n]["parameters"] / q_params for n in classical}

    h2 = {
        "hypothesis": (f"H2: HybridQNN reaches >= {COMPARABLE_MARGIN:.0%} of the best "
                       f"classical F1 with at least {MIN_PARAM_RATIO:.0f}x fewer parameters"),
        "hybridqnn_parameters": q_params,
        "hybridqnn_f1": q_f1,
        "classical_parameters": {n: results[n]["parameters"] for n in classical},
        "parameter_ratios": {n: round(r, 1) for n, r in ratios.items()},
        "min_parameter_ratio": round(min(ratios.values()), 1),
        "best_classical_model": best_classical,
        "best_classical_f1": results[best_classical]["f1"],
        "f1_retained": round(q_f1 / results[best_classical]["f1"], 4),
        "parameter_criterion_met": bool(min(ratios.values()) >= MIN_PARAM_RATIO),
        "performance_criterion_met": bool(
            q_f1 >= COMPARABLE_MARGIN * results[best_classical]["f1"]),
    }
    h2["supported"] = bool(h2["parameter_criterion_met"]
                           and h2["performance_criterion_met"])

    return results, h1, h2


# ---------------------------------------------------------------------------
# H3 and H4
# ---------------------------------------------------------------------------

def evaluate_h3():
    """H3 depends on the SHAP run. Report it if the artefacts exist."""
    path = "results/shap_summary.json"
    if not os.path.exists(path):
        log.warning("H3: %s not found, run explainability.shap_hybrid first", path)
        return {"hypothesis": "H3: SHAP attributions are domain-plausible",
                "status": "pending", "supported": None}
    with open(path) as f:
        shap_summary = json.load(f)
    return {"hypothesis": "H3: SHAP attributions are domain-plausible",
            "status": "computed", "detail": shap_summary, "supported": None}


def evaluate_h4():
    """H4 reuses the QAE anomaly scores saved by models.quantum.qae."""
    s_path, l_path = "results/qae_scores.npy", "results/qae_labels.npy"
    if not (os.path.exists(s_path) and os.path.exists(l_path)):
        log.warning("H4: QAE scores not found, run models.quantum.qae first")
        return {"hypothesis": "H4: QAE reaches AUROC >= 0.85 without attack labels",
                "status": "pending", "supported": None}

    scores = np.load(s_path)
    labels = np.load(l_path)
    auroc = float(roc_auc_score(labels, scores))
    return {
        "hypothesis": f"H4: QAE reaches AUROC >= {H4_AUROC_TARGET} without attack labels",
        "auroc": round(auroc, 4),
        "benign_mean_error": float(scores[labels == 0].mean()),
        "attack_mean_error": float(scores[labels == 1].mean()),
        "separation_ratio": round(float(scores[labels == 1].mean() /
                                        scores[labels == 0].mean()), 2),
        "supported": bool(auroc >= H4_AUROC_TARGET),
    }


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed/cicids2017")
    ap.add_argument("--refresh", action="store_true",
                    help="recompute model probabilities instead of using the cache")
    args = ap.parse_args()
    tag = os.path.basename(args.data_dir.rstrip("/\\"))

    log.info("evaluating on %s", args.data_dir)
    models = load_models()
    results, h1, h2 = evaluate_h1_h2(models, args.data_dir, tag, args.refresh)
    h3 = evaluate_h3()
    h4 = evaluate_h4()

    out = {"dataset": tag, "per_model": results,
           "H1": h1, "H2": h2, "H3": h3, "H4": h4}

    os.makedirs("results", exist_ok=True)
    with open(f"results/{tag}_hypotheses.json", "w") as f:
        json.dump(out, f, indent=2)

    # --- readable summary ---------------------------------------------------
    log.info("=" * 82)
    log.info("%-18s %10s %9s %9s %9s %9s",
             "Model", "Params", "Prec", "Recall", "F1", "ROC-AUC")
    for name in MODEL_ORDER + ["Ensemble-equal", "Ensemble-weighted"]:
        r = results[name]
        log.info("%-18s %10d %9.4f %9.4f %9.4f %9.4f",
                 name, r["parameters"], r["precision"], r["recall"],
                 r["f1"], r["roc_auc"])

    log.info("-" * 82)
    log.info("H1 %s", "SUPPORTED" if h1["supported"] else "NOT SUPPORTED")
    log.info("     best single model: %s F1=%.4f",
             h1["best_single_model"], h1["best_single_f1"])
    eq, wt = h1["equal_weight"], h1["weighted"]
    log.info("     equal weights    : F1=%.4f  margin=%+.4f  McNemar p=%.3e  %s",
             eq["f1"], eq["margin"], eq["mcnemar_p"],
             "pass" if eq["supported"] else "fail")
    log.info("     fitted weights %s", wt["weights"])
    log.info("                      : F1=%.4f  margin=%+.4f  McNemar p=%.3e  %s",
             wt["f1"], wt["margin"], wt["mcnemar_p"],
             "pass" if wt["supported"] else "fail")
    log.info("     discordant (weighted): ensemble-only %d, best-single-only %d",
             wt["discordant_ensemble_only"], wt["discordant_best_only"])

    log.info("H2 %s", "SUPPORTED" if h2["supported"] else "NOT SUPPORTED")
    log.info("     HybridQNN %d params, F1 %.4f, retains %.1f%% of best classical F1 (%s %.4f)",
             h2["hybridqnn_parameters"], h2["hybridqnn_f1"],
             100 * h2["f1_retained"], h2["best_classical_model"],
             h2["best_classical_f1"])
    for n, r in h2["parameter_ratios"].items():
        log.info("     %-8s %.1fx more parameters than HybridQNN", n, r)
    log.info("     parameter criterion (>= %.0fx)   : %s",
             MIN_PARAM_RATIO, "met" if h2["parameter_criterion_met"] else "not met")
    log.info("     performance criterion (>= %.0f%%): %s",
             100 * COMPARABLE_MARGIN,
             "met" if h2["performance_criterion_met"] else "not met")

    if h4.get("supported") is None:
        log.info("H4 PENDING")
    else:
        log.info("H4 %s | AUROC %.4f | separation %.2fx",
                 "SUPPORTED" if h4["supported"] else "NOT SUPPORTED",
                 h4["auroc"], h4["separation_ratio"])
    log.info("H3 %s", h3.get("status", "pending"))
    log.info("saved to results/%s_hypotheses.json", tag)


if __name__ == "__main__":
    main()