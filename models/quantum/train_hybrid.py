"""
QuantumShield – HybridQNN training and evaluation.

Hypothesis H1: the soft-voting ensemble beats the best single model on F1.
Hypothesis H2: the HybridQNN reaches comparable performance with >= 3x fewer
               trainable parameters than the classical baselines.

Trained on a stratified subset of the training split, because every forward
pass is a classical simulation of an 8-qubit circuit. Evaluated on the FULL
test set so the reported metrics are honest.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from models.quantum.circuits import HybridQNN, N_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAVE_PATH = "models/saved/hybrid_qnn.pth"


def stratified_subset(X, y, n_samples, seed=42):
    """
    Take a stratified subset that preserves the original class ratio, so the
    class prior the model learns matches the data it will be tested on.
    """
    rng = np.random.default_rng(seed)
    idx_parts = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        take = int(round(n_samples * len(cls_idx) / len(y)))
        take = max(1, min(take, len(cls_idx)))
        idx_parts.append(rng.choice(cls_idx, size=take, replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def train_hybrid(data_dir="data/processed/cicids2017", epochs=15, batch_size=128,
                 lr=5e-3, n_samples=20000):
    torch.manual_seed(42)
    np.random.seed(42)

    X_train = np.load(os.path.join(data_dir, "X_train.npy")).astype(np.float32)[:, :N_FEATURES]
    yb_train = np.load(os.path.join(data_dir, "yb_train.npy")).astype(np.int64)

    X_sub, y_sub = stratified_subset(X_train, yb_train, n_samples)
    log.info("HybridQNN training on %d samples (%d benign, %d attack)",
             len(y_sub), (y_sub == 0).sum(), (y_sub == 1).sum())

    model = HybridQNN()
    X_raw = torch.FloatTensor(X_sub)
    model.fit_scaler(X_raw)
    X = model.scale(X_raw)
    y = torch.LongTensor(y_sub)

    n_params = model.count_params()
    log.info("HybridQNN trainable parameters: %d", n_params)

    # Class weights, because CICIDS2017 is imbalanced and plain cross-entropy
    # would let the model score well by predicting the majority class.
    counts = np.bincount(y_sub, minlength=2).astype(np.float32)
    weights = torch.FloatTensor(counts.sum() / (2.0 * counts))
    log.info("class weights: benign=%.3f attack=%.3f", weights[0], weights[1])

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    n_batches = (len(X) + batch_size - 1) // batch_size
    best_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X))
        epoch_loss = 0.0
        all_preds = []
        all_true = []

        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb, yb = X[idx], y[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_preds.append(logits.argmax(dim=1).detach())
            all_true.append(yb)

        epoch_loss /= n_batches
        preds = torch.cat(all_preds).numpy()
        true = torch.cat(all_true).numpy()
        train_f1 = f1_score(true, preds, zero_division=0)
        history.append({"loss": epoch_loss, "f1": train_f1})

        log.info("Epoch %02d/%02d | loss=%.6f | train F1=%.4f",
                 epoch, epochs, epoch_loss, train_f1)

        if train_f1 > best_f1:
            best_f1 = train_f1
            torch.save(model.state_dict(), SAVE_PATH)

    os.makedirs("results", exist_ok=True)
    np.save("results/hybrid_loss_history.npy",
            np.array([h["loss"] for h in history]))
    np.save("results/hybrid_f1_history.npy",
            np.array([h["f1"] for h in history]))
    np.save("results/hybrid_param_count.npy", np.array([n_params]))

    log.info("HybridQNN training complete. Best train F1: %.4f", best_f1)
    return model


def evaluate_hybrid(model, data_dir="data/processed/cicids2017", batch_size=2048):
    X_raw  = torch.FloatTensor(np.load(os.path.join(data_dir, "X_test.npy"))[:, :N_FEATURES])
    yb_test = np.load(os.path.join(data_dir, "yb_test.npy")).astype(np.int64)

    model.eval()
    all_probs = []

    with torch.no_grad():
        X_test = model.scale(X_raw)
        n_batches = (len(X_test) + batch_size - 1) // batch_size
        for b in range(n_batches):
            batch = X_test[b * batch_size:(b + 1) * batch_size]
            logits = model(batch)
            all_probs.append(torch.softmax(logits, dim=1))
            if (b + 1) % 20 == 0 or b + 1 == n_batches:
                log.info("scoring batch %d/%d", b + 1, n_batches)

    probs = torch.cat(all_probs).numpy()
    preds = probs.argmax(axis=1)

    metrics = {
        "accuracy":  accuracy_score(yb_test, preds),
        "precision": precision_score(yb_test, preds, zero_division=0),
        "recall":    recall_score(yb_test, preds, zero_division=0),
        "f1":        f1_score(yb_test, preds, zero_division=0),
        "roc_auc":   roc_auc_score(yb_test, probs[:, 1]),
    }

    log.info("HybridQNN test metrics:")
    for k, v in metrics.items():
        log.info("  %-10s %.4f", k, v)
    log.info("confusion matrix (rows true, cols pred):\n%s",
             confusion_matrix(yb_test, preds))

    # Probabilities are what the soft-voting ensemble needs
    os.makedirs("results", exist_ok=True)
    np.save("results/hybrid_probs.npy", probs)
    np.save("results/hybrid_preds.npy", preds)
    np.save("results/test_labels.npy", yb_test)
    log.info("probabilities saved to results/hybrid_probs.npy")

    return metrics, probs


if __name__ == "__main__":
    if os.path.exists(SAVE_PATH):
        log.info("loading existing weights from %s", SAVE_PATH)
        model = HybridQNN()
        model.load_state_dict(torch.load(SAVE_PATH))
    else:
        model = train_hybrid()

    evaluate_hybrid(model)