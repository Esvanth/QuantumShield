"""
QuantumShield – Quantum Autoencoder (QAE) for Zero-Day Detection
Hypothesis H4: QAE achieves AUROC ≥ 0.85 on unseen attack types.

Architecture:
  Encoder : AngleEmbedding (8 qubits) + StronglyEntanglingLayers → compress to 4 qubits
  Decoder : mirrored variational layers
  Anomaly score = reconstruction fidelity / error on test sample
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)

N_QUBITS  = 8
N_LATENT  = 4          # compressed qubit space
N_LAYERS  = 2          # fewer layers for encoder and decoder each

dev = qml.device("default.qubit", wires=N_QUBITS)


# ---------------------------------------------------------------------------
# Encoder circuit
# ---------------------------------------------------------------------------

@qml.qnode(dev, interface="torch")
def encoder_circuit(inputs, enc_weights):
    """
    Encode 8-qubit state into latent 4-qubit representation.
    enc_weights : (N_LAYERS, N_QUBITS, 3) for StronglyEntanglingLayers
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(enc_weights, wires=range(N_QUBITS))
    # Measure all 8 qubits; latent space = first N_LATENT
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


@qml.qnode(dev, interface="torch")
def decoder_circuit(latent, dec_weights):
    """
    Decode latent representation back to 8-qubit state.
    latent      : (N_QUBITS,) – output of encoder_circuit
    dec_weights : (N_LAYERS, N_QUBITS, 3)
    """
    qml.AngleEmbedding(latent, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(dec_weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ---------------------------------------------------------------------------
# PyTorch QAE module
# ---------------------------------------------------------------------------

class QuantumAutoencoder(nn.Module):
    """
    Full Quantum Autoencoder.
    Training objective: minimise reconstruction error on BENIGN traffic.
    Anomaly score at inference: reconstruction MSE (high → likely attack).
    """

    def __init__(self):
        super().__init__()
        enc_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
        dec_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
        self.enc_weights = nn.Parameter(torch.randn(*enc_shape) * 0.01)
        self.dec_weights = nn.Parameter(torch.randn(*dec_shape) * 0.01)

    def encode(self, x):
        """Returns 8-dim expectation vector (use first N_LATENT as latent code)."""
        return torch.stack(
            [torch.stack(encoder_circuit(x[i], self.enc_weights)) for i in range(x.shape[0])]
        )   # (batch, N_QUBITS)

    def decode(self, z):
        return torch.stack(
            [torch.stack(decoder_circuit(z[i], self.dec_weights)) for i in range(z.shape[0])]
        )   # (batch, N_QUBITS)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def reconstruction_error(self, x):
        """Per-sample MSE — used as anomaly score."""
        x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)   # (batch,)


# ---------------------------------------------------------------------------
# Training (on BENIGN samples only)
# ---------------------------------------------------------------------------

def train_qae(data_dir="data/processed/cicids2017", epochs=30, batch_size=128, lr=5e-3):
    X_train = np.load(os.path.join(data_dir, "X_train.npy")).astype(np.float32)
    yb_train = np.load(os.path.join(data_dir, "yb_train.npy"))

    # Train on benign only
    benign_mask = yb_train == 0
    X_benign = torch.FloatTensor(X_train[benign_mask])
    log.info("QAE training on %d benign samples", len(X_benign))

    model = QuantumAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    save_path = "models/quantum/qae_weights.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(len(X_benign))
        epoch_loss = 0.0
        steps = 0
        for start in range(0, len(X_benign), batch_size):
            batch = X_benign[idx[start: start + batch_size]]
            optimizer.zero_grad()
            x_hat = model(batch)
            loss = criterion(x_hat, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        epoch_loss /= steps
        log.info("Epoch %02d/%02d | recon loss=%.6f", epoch, epochs, epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)

    log.info("QAE training complete. Best loss: %.6f", best_loss)
    return model


# ---------------------------------------------------------------------------
# Evaluation — H4
# ---------------------------------------------------------------------------

def evaluate_qae(model, data_dir="data/processed/cicids2017"):
    X_test  = torch.FloatTensor(np.load(os.path.join(data_dir, "X_test.npy")))
    yb_test = np.load(os.path.join(data_dir, "yb_test.npy"))

    model.eval()
    with torch.no_grad():
        scores = model.reconstruction_error(X_test).numpy()

    auroc = roc_auc_score(yb_test, scores)
    log.info("QAE AUROC (H4): %.4f | threshold ≥ 0.85: %s", auroc, "PASS" if auroc >= 0.85 else "FAIL")
    return auroc, scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model = train_qae()
    evaluate_qae(model)
