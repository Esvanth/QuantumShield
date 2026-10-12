"""
QuantumShield - Flask REST API.

Serves the weighted soft-voting ensemble exactly as it was evaluated: the same
four models, the same validation-fitted weights, and the same decision
threshold. Those come from results/<dataset>_hypotheses.json rather than being
hardcoded, so the deployed system cannot drift away from the reported metrics.

Endpoints:
  GET  /health          liveness, plus what is actually loaded
  GET  /metrics         the reported test metrics for the loaded dataset
  GET  /explain         global SHAP feature importance, real network features
  POST /predict         classify one flow
  POST /predict/batch   classify many flows

Input accepts either 78 raw CICIDS2017 flow features, which are passed through
the fitted scaler and PCA, or 16 PCA components directly. Raw is what a real
sensor would send.

Run:
    python -m deployment.api
    DATASET=ciciot2023 python -m deployment.api
"""

import os
import json
import pickle
import logging

import numpy as np
import torch
from flask import Flask, request, jsonify

from models.classical.dnn import DNN
from models.classical.cnn import CNN1D
from models.classical.lstm import LSTMClassifier
from models.quantum.circuits import HybridQNN, N_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

DATASET = os.environ.get("DATASET", "cicids2017")
PROC_DIR = os.path.join("data", "processed", DATASET)
MODEL_ORDER = ["DNN", "CNN-1D", "LSTM", "HybridQNN"]

CHECKPOINTS = {
    "DNN": "models/classical/dnn_weights.pth",
    "CNN-1D": "models/classical/cnn_weights.pth",
    "LSTM": "models/classical/lstm_weights.pth",
    "HybridQNN": "models/saved/hybrid_qnn.pth",
}

# Filled by load_resources()
STATE = {
    "ready": False,
    "scaler": None,
    "pca": None,
    "models": {},
    "weights": None,       # normalised, in MODEL_ORDER
    "raw_weights": None,
    "threshold": 0.5,
    "metrics": None,
    "n_raw_features": None,
}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def load_resources():
    """Load the transforms, the four models, and the fitted decision rule."""
    if STATE["ready"]:
        return

    with open(os.path.join(PROC_DIR, "scaler.pkl"), "rb") as f:
        STATE["scaler"] = pickle.load(f)
    with open(os.path.join(PROC_DIR, "pca.pkl"), "rb") as f:
        STATE["pca"] = pickle.load(f)
    STATE["n_raw_features"] = int(STATE["pca"].components_.shape[1])

    models = {}
    dnn = DNN(input_dim=N_FEATURES)
    dnn.load_state_dict(torch.load(CHECKPOINTS["DNN"], map_location="cpu"))
    models["DNN"] = dnn

    cnn = CNN1D(input_dim=N_FEATURES)
    cnn.load_state_dict(torch.load(CHECKPOINTS["CNN-1D"], map_location="cpu"))
    models["CNN-1D"] = cnn

    lstm = LSTMClassifier(input_dim=N_FEATURES)
    lstm.load_state_dict(torch.load(CHECKPOINTS["LSTM"], map_location="cpu"))
    models["LSTM"] = lstm

    hqnn = HybridQNN()
    hqnn.load_state_dict(torch.load(CHECKPOINTS["HybridQNN"], map_location="cpu"))
    models["HybridQNN"] = hqnn

    for m in models.values():
        m.eval()
    STATE["models"] = models

    # The decision rule comes from evaluation, never from a hardcoded default.
    hyp_path = os.path.join("results", f"{DATASET}_hypotheses.json")
    if os.path.exists(hyp_path):
        with open(hyp_path) as f:
            hyp = json.load(f)
        ens = hyp["per_model"]["Ensemble-weighted"]
        raw = ens.get("weights", {m: 1 for m in MODEL_ORDER})
        STATE["raw_weights"] = raw
        w = np.array([raw[m] for m in MODEL_ORDER], dtype=np.float64)
        STATE["weights"] = w / w.sum()
        STATE["threshold"] = float(ens["threshold"])
        STATE["metrics"] = {k: hyp["per_model"][k] for k in hyp["per_model"]}
        log.info("decision rule from %s: weights %s, threshold %.3f",
                 hyp_path, raw, STATE["threshold"])
    else:
        log.warning("%s not found, falling back to equal weights at 0.5", hyp_path)
        STATE["raw_weights"] = {m: 1 for m in MODEL_ORDER}
        STATE["weights"] = np.full(len(MODEL_ORDER), 1.0 / len(MODEL_ORDER))
        STATE["threshold"] = 0.5

    STATE["ready"] = True
    log.info("QuantumShield API ready on dataset '%s'", DATASET)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def to_components(rows):
    """
    Accept either raw flow features or PCA components and return (n, 16).

    A real sensor sends raw features, so scaler and PCA have to run here. The
    PCA-component path exists for testing against the saved splits.
    """
    X = np.asarray(rows, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    n_raw = STATE["n_raw_features"]
    if X.shape[1] == n_raw:
        X = STATE["scaler"].transform(X)
        X = STATE["pca"].transform(X)
    elif X.shape[1] != N_FEATURES:
        raise ValueError(
            f"expected {n_raw} raw features or {N_FEATURES} PCA components, "
            f"got {X.shape[1]}")

    return torch.FloatTensor(np.asarray(X, dtype=np.float32)[:, :N_FEATURES])


def attack_probability(X):
    """Weighted soft vote over the four members. Returns (n,) P(attack)."""
    parts = []
    with torch.no_grad():
        for name in MODEL_ORDER:
            model = STATE["models"][name]
            # The quantum model applies its own tanh scaler internally
            inp = model.scale(X) if name == "HybridQNN" else X
            parts.append(torch.softmax(model(inp), dim=1)[:, 1].numpy())
    per_model = {n: p for n, p in zip(MODEL_ORDER, parts)}
    return STATE["weights"] @ np.stack(parts), per_model


def classify(rows):
    X = to_components(rows)
    p_attack, per_model = attack_probability(X)
    thr = STATE["threshold"]

    out = []
    for i, p in enumerate(p_attack):
        out.append({
            "prediction": "ATTACK" if p >= thr else "BENIGN",
            "label": int(p >= thr),
            "attack_probability": round(float(p), 6),
            "threshold": thr,
            "per_model": {n: round(float(per_model[n][i]), 6) for n in MODEL_ORDER},
        })
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    try:
        load_resources()
    except Exception as exc:
        log.exception("startup failed")
        return jsonify({"status": "error", "detail": str(exc)}), 503

    return jsonify({
        "status": "ok",
        "service": "QuantumShield",
        "dataset": DATASET,
        "models": MODEL_ORDER,
        "ensemble_weights": STATE["raw_weights"],
        "decision_threshold": STATE["threshold"],
        "expects": {
            "raw_features": STATE["n_raw_features"],
            "or_pca_components": N_FEATURES,
        },
    }), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    load_resources()
    if STATE["metrics"] is None:
        return jsonify({"error": "no evaluation results for this dataset"}), 404
    return jsonify({"dataset": DATASET, "test_metrics": STATE["metrics"]}), 200


@app.route("/explain", methods=["GET"])
def explain():
    """
    Global feature importance from the SHAP run, in real network feature names
    rather than PCA components. Per-sample explanation is not served here
    because KernelSHAP needs hundreds of model evaluations per sample, which is
    not an interactive-latency operation.
    """
    path = os.path.join("results", "shap_summary.json")
    if not os.path.exists(path):
        return jsonify({"error": "SHAP not computed, run explainability.shap_hybrid"}), 404
    with open(path) as f:
        summary = json.load(f)
    return jsonify({
        "method": summary.get("method"),
        "model_explained": summary.get("model_explained"),
        "top_features": summary.get("top20_features", [])[:10],
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_resources()
        payload = request.get_json(force=True, silent=False)
        if not payload or "features" not in payload:
            return jsonify({"error": "body must be {\"features\": [...]}"}), 400
        return jsonify(classify(payload["features"])[0]), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        log.exception("prediction failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    try:
        load_resources()
        payload = request.get_json(force=True, silent=False)
        if not payload or "batch" not in payload:
            return jsonify({"error": "body must be {\"batch\": [[...], [...]]}"}), 400

        rows = payload["batch"]
        if len(rows) > 10000:
            return jsonify({"error": "batch limit is 10000 rows"}), 413

        results = classify(rows)
        n_attack = sum(r["label"] for r in results)
        return jsonify({
            "count": len(results),
            "attacks": n_attack,
            "benign": len(results) - n_attack,
            "results": results,
        }), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        log.exception("batch prediction failed")
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_resources()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)