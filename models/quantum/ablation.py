"""
QuantumShield - quantum architecture ablation.

Compares three variational circuit designs under identical conditions: the same
scaler, the same classical pre-encoder, the same classical head, the same
optimiser, learning rate, batch size, epoch count, random seed and training
subset. The ONLY thing that differs is the quantum block, so any difference in
F1 is attributable to the circuit design rather than to anything around it.

Produces the architecture comparison table for the evaluation chapter, which is
what turns the choice of StronglyEntanglingLayers from an arbitrary decision
into an evidenced one.

QSVM is deliberately excluded. Quantum kernel methods are quadratic in sample
count, so a 20,000-sample kernel matrix is 400 million entries and will not run
under the same conditions. Report it separately, on a reduced subset, with that
limitation stated.

Usage:
    python -m models.quantum.ablation
    python -m models.quantum.ablation --epochs 10 --n-samples 10000
"""

import os
import json
import time
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

N_FEATURES = 16
N_QUBITS   = 8
N_LAYERS   = 4
Z_DIVISOR  = 3.0

dev = qml.device("default.qubit", wires=N_QUBITS)


# ---------------------------------------------------------------------------
# Three circuits, all with the same signature: (batch, 8) angles -> 8 expvals
# ---------------------------------------------------------------------------

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnn_circuit(inputs, weights):
    """Baseline: StronglyEntanglingLayers. weights (N_LAYERS, N_QUBITS, 3)."""
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def qcnn_circuit(inputs, w_conv, w_pool):
    """
    Quantum CNN: alternating convolution and pooling blocks. Convolution
    entangles adjacent qubit pairs with local rotations; pooling uses controlled
    rotations to funnel information toward the even wires.
    w_conv (N_LAYERS, N_QUBITS), w_pool (N_LAYERS, N_QUBITS // 2).
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    for layer in range(N_LAYERS):
        for i in range(0, N_QUBITS - 1, 2):
            qml.CNOT(wires=[i, i + 1])
            qml.RY(w_conv[layer, i], wires=i)
            qml.RY(w_conv[layer, i + 1], wires=i + 1)
        for i in range(N_QUBITS // 2):
            qml.CRY(w_pool[layer, i], wires=[2 * i + 1, 2 * i])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def vqc_circuit(inputs, weights):
    """
    Variational Quantum Classifier: RY/RZ rotations with a CNOT ring.
    weights (N_LAYERS, N_QUBITS, 2).
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])      # wrap-around entanglement
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ---------------------------------------------------------------------------
# Quantum blocks, each (batch, N_QUBITS) -> (batch, N_QUBITS)
# ---------------------------------------------------------------------------

class QNNBlock(nn.Module):
    name = "HybridQNN (StronglyEntanglingLayers)"

    def __init__(self):
        super().__init__()
        shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
        self.weights = nn.Parameter(torch.randn(*shape) * 0.01)

    def forward(self, angles):
        return torch.stack(qnn_circuit(angles, self.weights), dim=-1).float()


class QCNNBlock(nn.Module):
    name = "QCNN (convolution + pooling)"

    def __init__(self):
        super().__init__()
        self.w_conv = nn.Parameter(torch.randn(N_LAYERS, N_QUBITS) * 0.01)
        self.w_pool = nn.Parameter(torch.randn(N_LAYERS, N_QUBITS // 2) * 0.01)

    def forward(self, angles):
        return torch.stack(qcnn_circuit(angles, self.w_conv, self.w_pool), dim=-1).float()


class VQCBlock(nn.Module):
    name = "VQC (RY/RZ + CNOT ring)"

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(N_LAYERS, N_QUBITS, 2) * 0.01)

    def forward(self, angles):
        return torch.stack(vqc_circuit(angles, self.weights), dim=-1).float()


# ---------------------------------------------------------------------------
# Identical shell around whichever quantum block is being tested
# ---------------------------------------------------------------------------

class HybridModel(nn.Module):
    def __init__(self, block_cls, num_classes=2):
        super().__init__()
        self.pre_encoder = nn.Linear(N_FEATURES, N_QUBITS)
        self.quantum = block_cls()
        self.head = nn.Sequential(
            nn.Linear(N_QUBITS, 16), nn.ReLU(),
            nn.Linear(16, num_classes),
        )
        self.register_buffer("x_mean", torch.zeros(N_FEATURES))
        self.register_buffer("x_std", torch.ones(N_FEATURES))

    def fit_scaler(self, X_raw):
        self.x_mean.copy_(X_raw.mean(dim=0))
        std = X_raw.std(dim=0)
        std[std < 1e-8] = 1.0
        self.x_std.copy_(std)

    def scale(self, X_raw):
        return torch.tanh((X_raw - self.x_mean) / self.x_std / Z_DIVISOR)

    def forward(self, x):
        return self.head(self.quantum(torch.tanh(self.pre_encoder(x))))

    def quantum_params(self):
        return sum(p.numel() for p in self.quantum.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Shared data handling, so every architecture sees exactly the same samples
# ---------------------------------------------------------------------------

def stratified_subset(X, y, n_samples, seed=42):
    rng = np.random.default_rng(seed)
    parts = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        take = int(round(n_samples * len(cls_idx) / len(y)))
        take = max(1, min(take, len(cls_idx)))
        parts.append(rng.choice(cls_idx, size=take, replace=False))
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def load_data(data_dir, n_samples):
    X_train = np.load(os.path.join(data_dir, "X_train.npy")).astype(np.float32)[:, :N_FEATURES]
    y_train = np.load(os.path.join(data_dir, "yb_train.npy")).astype(np.int64)
    X_sub, y_sub = stratified_subset(X_train, y_train, n_samples)

    X_test = np.load(os.path.join(data_dir, "X_test.npy")).astype(np.float32)[:, :N_FEATURES]
    y_test = np.load(os.path.join(data_dir, "yb_test.npy")).astype(np.int64)
    return X_sub, y_sub, X_test, y_test


# ---------------------------------------------------------------------------
# Train and evaluate one architecture
# ---------------------------------------------------------------------------

def run_one(block_cls, X_sub, y_sub, X_test, y_test, tag,
            epochs=15, batch_size=128, lr=5e-3, eval_batch=2048):

    # Reset the seed before each architecture so all three start from
    # equivalent initialisation conditions.
    torch.manual_seed(42)
    np.random.seed(42)

    model = HybridModel(block_cls)
    X_raw = torch.FloatTensor(X_sub)
    model.fit_scaler(X_raw)
    X = model.scale(X_raw)
    y = torch.LongTensor(y_sub)

    log.info("=" * 70)
    log.info("%s", block_cls.name)
    log.info("quantum params: %d | total params: %d",
             model.quantum_params(), model.total_params())

    counts = np.bincount(y_sub, minlength=2).astype(np.float32)
    weights = torch.FloatTensor(counts.sum() / (2.0 * counts))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_batches = (len(X) + batch_size - 1) // batch_size
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X))
        epoch_loss = 0.0
        preds, trues = [], []

        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb, yb = X[idx], y[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds.append(logits.argmax(dim=1).detach())
            trues.append(yb)

        train_f1 = f1_score(torch.cat(trues).numpy(),
                            torch.cat(preds).numpy(), zero_division=0)
        log.info("  epoch %02d/%02d | loss=%.6f | train F1=%.4f",
                 epoch, epochs, epoch_loss / n_batches, train_f1)

    train_time = time.time() - t0

    # Evaluate on the FULL test set
    model.eval()
    probs = []
    with torch.no_grad():
        Xt = model.scale(torch.FloatTensor(X_test))
        nb = (len(Xt) + eval_batch - 1) // eval_batch
        for b in range(nb):
            probs.append(torch.softmax(model(Xt[b * eval_batch:(b + 1) * eval_batch]), dim=1))
    probs = torch.cat(probs).numpy()
    pred = probs.argmax(axis=1)

    result = {
        "architecture":    block_cls.name,
        "quantum_params":  model.quantum_params(),
        "total_params":    model.total_params(),
        "precision":       float(precision_score(y_test, pred, zero_division=0)),
        "recall":          float(recall_score(y_test, pred, zero_division=0)),
        "f1":              float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc":         float(roc_auc_score(y_test, probs[:, 1])),
        "train_seconds":   round(train_time, 1),
    }

    log.info("  TEST  F1=%.4f  ROC-AUC=%.4f  precision=%.4f  recall=%.4f  (%.0fs)",
             result["f1"], result["roc_auc"],
             result["precision"], result["recall"], train_time)

    os.makedirs("results", exist_ok=True)
    short = block_cls.__name__.replace("Block", "").lower()
    np.save(f"results/{tag}_ablation_{short}_probs.npy", probs)

    return result


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed/cicids2017")
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    tag = os.path.basename(args.data_dir.rstrip("/\\"))

    X_sub, y_sub, X_test, y_test = load_data(args.data_dir, args.n_samples)
    log.info("ablation on %d training samples (%d benign, %d attack), %d test samples",
             len(y_sub), (y_sub == 0).sum(), (y_sub == 1).sum(), len(y_test))

    results = []
    for block_cls in (QNNBlock, QCNNBlock, VQCBlock):
        results.append(run_one(block_cls, X_sub, y_sub, X_test, y_test, tag,
                               epochs=args.epochs))

    os.makedirs("results", exist_ok=True)
    with open("results/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info("=" * 70)
    log.info("%-42s %8s %8s %8s %8s", "Architecture", "Q-param", "Total", "F1", "ROC-AUC")
    for r in results:
        log.info("%-42s %8d %8d %8.4f %8.4f",
                 r["architecture"], r["quantum_params"],
                 r["total_params"], r["f1"], r["roc_auc"])
    log.info("saved to results/ablation_results.json")


if __name__ == "__main__":
    main()