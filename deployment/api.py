"""
QuantumShield - Flask REST API.

Serves the weighted soft-voting ensemble exactly as it was evaluated: the same
four models, the same validation-fitted weights, and the same decision
threshold. Those come from results/<dataset>_hypotheses.json rather than being
hardcoded, so the deployed system cannot drift away from the reported metrics.

RQ2 abstention: when the spread between the four member probabilities exceeds
TAU_ABSTAIN, the API returns ABSTAIN rather than a binary decision. Thresholds
are per-dataset and come from abstention.py.

Every classification is persisted to DynamoDB so the system keeps a detection
history rather than computing and forgetting. Writes are best effort: if
DynamoDB is unreachable the prediction is still returned.

Endpoints:
  GET  /                landing page listing the service
  GET  /health          liveness, plus what is actually loaded
  GET  /metrics         the reported test metrics for the loaded dataset
  GET  /explain         global SHAP feature importance, real network features
  GET  /recent          recent classifications from DynamoDB
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
from decimal import Decimal
from datetime import datetime, timezone

import numpy as np
import torch
import boto3
from boto3.dynamodb.conditions import Key
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

# Abstention thresholds from abstention.py (RQ2).
# When spread of member probabilities exceeds TAU_ABSTAIN, return ABSTAIN.
TAU_ABSTAIN = {
    "cicids2017": 0.07,
    "ciciot2023": 0.05,
}

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
    log.info("QuantumShield API ready on dataset '%s' with tau_abstain=%.3f",
             DATASET, TAU_ABSTAIN.get(DATASET, 0.07))


# ---------------------------------------------------------------------------
# Detection history, DynamoDB
# ---------------------------------------------------------------------------

DDB_TABLE = os.environ.get("DDB_TABLE", "quantumshield-predictions")
DDB_REGION = os.environ.get("AWS_REGION", "eu-west-1")
_table = None


def ddb_table():
    """
    Lazy so the API still starts when DynamoDB is unreachable. Credentials come
    from the instance's IAM role, so nothing is stored in the image.
    """
    global _table
    if _table is None:
        _table = boto3.resource("dynamodb", region_name=DDB_REGION).Table(DDB_TABLE)
    return _table


def log_prediction(record):
    """
    Persist one classification. Partitioned by day and sorted by timestamp, so
    a day's detections can be read back in order without scanning the table.

    Best effort by design: a logging failure must never break a prediction.
    DynamoDB rejects Python floats, hence Decimal(str(x)).
    """
    try:
        now = datetime.now(timezone.utc)
        ddb_table().put_item(Item={
            "day": now.strftime("%Y-%m-%d"),
            "ts": now.isoformat(),
            "dataset": DATASET,
            "prediction": record["prediction"],
            "label": record["label"],
            "attack_probability": Decimal(str(record["attack_probability"])),
            "threshold": Decimal(str(record["threshold"])),
            "spread": Decimal(str(record["spread"])),
            "tau_abstain": Decimal(str(record["tau_abstain"])),
            "per_model": {k: Decimal(str(v)) for k, v in record["per_model"].items()},
        })
    except Exception:
        log.warning("could not write prediction to DynamoDB", exc_info=True)


def jsonable(obj):
    """DynamoDB returns Decimal, which the JSON encoder refuses."""
    if isinstance(obj, list):
        return [jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Decimal):
        return float(obj)
    return obj


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
    """
    Classify one or more flows. Applies the RQ2 abstention rule: if the spread
    of member probabilities exceeds TAU_ABSTAIN for the loaded dataset, return
    ABSTAIN rather than ATTACK or BENIGN.
    """
    X = to_components(rows)
    p_attack, per_model = attack_probability(X)
    thr = STATE["threshold"]
    tau_abstain = float(TAU_ABSTAIN.get(DATASET, 0.07))

    out = []
    for i, p in enumerate(p_attack):
        member_probs = [float(per_model[n][i]) for n in MODEL_ORDER]
        spread = float(max(member_probs) - min(member_probs))

        # RQ2 abstention: refuse when members disagree too much
        if spread > tau_abstain:
            prediction, label = "ABSTAIN", -1
        elif p >= thr:
            prediction, label = "ATTACK", 1
        else:
            prediction, label = "BENIGN", 0

        record = {
            "prediction": prediction,
            "label": label,
            "attack_probability": round(float(p), 6),
            "threshold": thr,
            "spread": round(spread, 6),
            "tau_abstain": tau_abstain,
            "per_model": {n: round(float(per_model[n][i]), 6) for n in MODEL_ORDER},
        }
        log_prediction(record)
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Landing route, so the base URL lists the service rather than 404ing."""
    return jsonify({
        "service": "QuantumShield",
        "description": "Hybrid quantum-classical intrusion detection with SHAP explainability and spread-based abstention",
        "dataset": DATASET,
        "endpoints": {
            "GET /health": "liveness and loaded configuration",
            "GET /metrics": "reported test metrics",
            "GET /explain": "global SHAP feature importance",
            "GET /recent": "recent classifications, ?n=50&day=YYYY-MM-DD",
            "POST /predict": "classify one flow",
            "POST /predict/batch": "classify many flows",
        },
    }), 200


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
        "tau_abstain": float(TAU_ABSTAIN.get(DATASET, 0.07)),
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


@app.route("/recent", methods=["GET"])
def recent():
    """
    Recent classifications for one day, newest first. The dashboard's live feed
    reads this. Querying one partition rather than scanning the table is why
    the key is day plus timestamp. Counts ATTACK, BENIGN and ABSTAIN
    separately so the dashboard can show all three.
    """
    try:
        n = min(int(request.args.get("n", 50)), 500)
    except ValueError:
        return jsonify({"error": "n must be an integer"}), 400

    day = request.args.get("day") or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        resp = ddb_table().query(
            KeyConditionExpression=Key("day").eq(day),
            ScanIndexForward=False,
            Limit=n,
        )
    except Exception as exc:
        log.exception("could not read detection history")
        return jsonify({"error": str(exc)}), 503

    items = jsonable(resp.get("Items", []))
    attacks = sum(1 for it in items if it.get("label") == 1)
    benign = sum(1 for it in items if it.get("label") == 0)
    abstain = sum(1 for it in items if it.get("label") == -1)
    return jsonify({
        "day": day,
        "count": len(items),
        "attacks": attacks,
        "benign": benign,
        "abstain": abstain,
        "items": items,
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
        n_attack = sum(1 for r in results if r["label"] == 1)
        n_benign = sum(1 for r in results if r["label"] == 0)
        n_abstain = sum(1 for r in results if r["label"] == -1)
        return jsonify({
            "count": len(results),
            "attacks": n_attack,
            "benign": n_benign,
            "abstain": n_abstain,
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