"""
QuantumShield – Flask REST API
Endpoints:
  POST /predict         – classify a single flow (binary)
  POST /predict/batch   – classify multiple flows
  GET  /health          – liveness check
  GET  /shap/<int:idx>  – retrieve SHAP explanation for a recent prediction
"""

import os
import logging
import numpy as np
import pickle
import torch
from flask import Flask, request, jsonify

from models.ensemble import SoftVotingEnsemble

log = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load resources at startup
# ---------------------------------------------------------------------------

CICIDS_PROC = "data/processed/cicids2017"

_scaler = None
_pca    = None
_ensemble = None


def _load_resources():
    global _scaler, _pca, _ensemble
    if _scaler is None:
        with open(os.path.join(CICIDS_PROC, "scaler.pkl"), "rb") as f:
            _scaler = pickle.load(f)
        with open(os.path.join(CICIDS_PROC, "pca.pkl"), "rb") as f:
            _pca = pickle.load(f)
        _ensemble = SoftVotingEnsemble()
        log.info("Resources loaded.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess(raw_features: list) -> torch.Tensor:
    """raw_features: list of dicts or list of float lists (already 8-dim PCA) """
    X = np.array(raw_features, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    # If raw (78-feature) vector supplied, scale + PCA
    if X.shape[1] != 8:
        X = _scaler.transform(X)
        X = _pca.transform(X)
    return torch.FloatTensor(X)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "QuantumShield"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    _load_resources()
    payload = request.get_json(force=True)
    try:
        features = payload["features"]   # list of floats (8 or 78)
        X = _preprocess(features)
        probs = _ensemble.predict_proba(X)[0]
        label = int(probs.argmax())
        return jsonify({
            "prediction": "ATTACK" if label == 1 else "BENIGN",
            "label": label,
            "confidence": float(probs[label]),
            "probabilities": {"benign": float(probs[0]), "attack": float(probs[1])},
        })
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        log.exception("Prediction error")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    _load_resources()
    payload = request.get_json(force=True)
    try:
        batch = payload["batch"]   # list of feature lists
        X = _preprocess(batch)
        probs = _ensemble.predict_proba(X)
        labels = probs.argmax(axis=1).tolist()
        return jsonify({
            "predictions": ["ATTACK" if l == 1 else "BENIGN" for l in labels],
            "labels": labels,
            "probabilities": probs.tolist(),
        })
    except Exception as e:
        log.exception("Batch prediction error")
        return jsonify({"error": str(e)}), 500


@app.route("/shap/<int:idx>", methods=["GET"])
def get_shap(idx):
    shap_path = os.path.join("results", "shap_plots", "shap_values_attack.npy")
    if not os.path.exists(shap_path):
        return jsonify({"error": "SHAP values not computed yet. Run shap_hybrid.py first."}), 404
    shap_vals = np.load(shap_path)
    if idx >= len(shap_vals):
        return jsonify({"error": f"Index {idx} out of range (max {len(shap_vals)-1})"}), 400
    feature_names = [f"PC{i+1}" for i in range(shap_vals.shape[1])]
    explanation = dict(zip(feature_names, shap_vals[idx].tolist()))
    return jsonify({"sample_index": idx, "shap_attack_class": explanation})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
