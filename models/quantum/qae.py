"""
QuantumShield – Hybrid Quantum Autoencoder (QAE) for Zero-Day Detection
Hypothesis H4: QAE achieves AUROC >= 0.85 on unseen attack types.

Architecture:
  Scaler       : z-score on benign traffic, then tanh squash into (-1, 1).
                 tanh is monotonic and bounded, so extreme attack values
                 saturate instead of wrapping around the Bloch sphere.
  Pre-encoder  : classical Linear(16 -> 8) + tanh, produces the rotation angles
  Q-encoder    : AngleEmbedding (8 qubits) + StronglyEntanglingLayers
                 -> first 4 expectation values form the latent code
  Q-decoder    : latent zero-padded back to 8 wires + StronglyEntanglingLayers
  Post-decoder : classical Linear(8 -> 16), reconstructs all 16 features
  Anomaly score = per-sample reconstruction MSE
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)

N_FEATURES = 16        # all PCA components, not just the first 8
N_QUBITS   = 8
N_LATENT   = 4         # compressed qubit space
N_LAYERS   = 2         # layers for encoder and decoder each
Z_DIVISOR  = 3.0       # spreads benign z-scores across the useful tanh range

dev = qml.device("default.qubit", wires=N_QUBITS)


# ---------------------------------------------------------------------------
# Quantum circuits
# ---------------------------------------------------------------------------

@qml.qnode(dev, interface="torch", diff_method="backprop")
def encoder_circuit(inputs, enc_weights):
    """inputs is (batch, N_QUBITS) with values in (-1, 1)."""
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(enc_weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def decoder_circuit(latent, dec_weights):
    """latent is (batch, N_QUBITS), only the first N_LATENT carry signal."""
    qml.AngleEmbedding(latent, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(dec_weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ---------------------------------------------------------------------------
# Hybrid QAE module
# ---------------------------------------------------------------------------

class QuantumAutoencoder(nn.Module):
    """
    Hybrid quantum-classical autoencoder.
    Trained to minimise reconstruction error on BENIGN traffic only.
    Anomaly score at inference: reconstruction MSE (high -> likely attack).
    """

    def __init__(self):
        super().__init__()
        enc_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
        dec_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
        self.enc_weights = nn.Parameter(torch.randn(*enc_shape) * 0.01)
        self.dec_weights = nn.Parameter(torch.randn(*dec_shape) * 0.01)

        # Classical layers that bridge 16 features to the 8 available qubits
        self.pre_encoder  = nn.Linear(N_FEATURES, N_QUBITS)
        self.post_decoder = nn.Linear(N_QUBITS, N_FEATURES)

        # Scaler stats stored as buffers so they save and load with the weights
        self.register_buffer("x_mean", torch.zeros(N_FEATURES))
        self.register_buffer("x_std", torch.ones(N_FEATURES))

    def fit_scaler(self, X_raw):
        """Fit z-score statistics on benign training data only."""
        self.x_mean.copy_(X_raw.mean(dim=0))
        std = X_raw.std(dim=0)
        std[std < 1e-8] = 1.0
        self.x_std.copy_(std)

    def scale(self, X_raw):
        """
        z-score then tanh. Bounded and monotonic, so an attack sample sitting
        at z = 20 saturates near 1.0 instead of wrapping past 2*pi and
        masquerading as benign.
        """
        z = (X_raw - self.x_mean) / self.x_std
        return torch.tanh(z / Z_DIVISOR)

    def encode(self, x):
        """x is (batch, N_FEATURES), already scaled. Returns the latent code."""
        angles = torch.tanh(self.pre_encoder(x))
        out = torch.stack(encoder_circuit(angles, self.enc_weights), dim=-1).float()
        return out[:, :N_LATENT]        # (batch, N_LATENT) -- the bottleneck

    def decode(self, z):
        """Zero-pad the latent code back to N_QUBITS wires, decode, project out."""
        pad = torch.zeros(z.shape[0], N_QUBITS - N_LATENT, device=z.device)
        z_padded = torch.cat([z, pad], dim=1)
        q_out = torch.stack(decoder_circuit(z_padded, self.dec_weights), dim=-1).float()
        return self.post_decoder(q_out)

    def forward(self, x):
        return self.decode(self.encode(x))

    def reconstruction_error(self, x):
        """Per-sample MSE on scaled input -- used as anomaly score."""
        x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)   # (batch,)


# ---------------------------------------------------------------------------
# Training (on BENIGN samples only)
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
def train_qae(data_dir="data/processed/cicids2017", epochs=30, batch_size=128,
              lr=5e-3, n_samples=5000):
    X_train = np.load(os.path.join(data_dir, "X_train.npy")).astype(np.float32)
    yb_train = np.load(os.path.join(data_dir, "yb_train.npy"))

    # Train on benign only
    benign_mask = yb_train == 0
    X_raw = torch.FloatTensor(X_train[benign_mask][:n_samples, :N_FEATURES])
    log.info("QAE training on %d benign samples, %d features", len(X_raw), N_FEATURES)

    model = QuantumAutoencoder()
    model.fit_scaler(X_raw)
    X_benign = model.scale(X_raw)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("QAE trainable parameters: %d", n_params)

    save_path = "models/quantum/qae_weights.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_loss = float("inf")
    n_batches = (len(X_benign) + batch_size - 1) // batch_size
    history = []

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
        history.append(epoch_loss)
        log.info("Epoch %02d/%02d | recon loss=%.6f", epoch, epochs, epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)

    # Save the loss curve for the training convergence plot
    os.makedirs("results", exist_ok=True)
    np.save("results/qae_loss_history.npy", np.array(history))

    log.info("QAE training complete. Best loss: %.6f", best_loss)
    return model


# ---------------------------------------------------------------------------
# Evaluation — H4
# ---------------------------------------------------------------------------

def evaluate_qae(model, data_dir="data/processed/cicids2017", batch_size=2048):
    X_raw   = torch.FloatTensor(np.load(os.path.join(data_dir, "X_test.npy"))[:, :N_FEATURES])
    yb_test = np.load(os.path.join(data_dir, "yb_test.npy"))

    model.eval()
    all_scores = []

    with torch.no_grad():
        X_test = model.scale(X_raw)

        # Batched so the simulator does not allocate the whole state tensor
        # for every test sample at once.
        n_batches = (len(X_test) + batch_size - 1) // batch_size
        for b in range(n_batches):
            batch = X_test[b * batch_size:(b + 1) * batch_size]
            all_scores.append(model.reconstruction_error(batch))
            if (b + 1) % 20 == 0 or b + 1 == n_batches:
                log.info("scoring batch %d/%d", b + 1, n_batches)

    scores = torch.cat(all_scores).numpy()

    auroc = roc_auc_score(yb_test, scores)
    log.info("QAE AUROC (H4): %.4f | threshold >= 0.85: %s",
             auroc, "PASS" if auroc >= 0.85 else "FAIL")

    # Report the separation between the two score distributions, which is what
    # actually drives AUROC. Useful for the discussion section.
    benign_mean = scores[yb_test == 0].mean()
    attack_mean = scores[yb_test == 1].mean()
    log.info("mean recon error -- benign: %.6f | attack: %.6f | ratio: %.2fx",
             benign_mean, attack_mean, attack_mean / benign_mean)

    # Save for the ROC curve and score distribution plots later
    os.makedirs("results", exist_ok=True)
    np.save("results/qae_scores.npy", scores)
    np.save("results/qae_labels.npy", yb_test)
    log.info("scores saved to results/qae_scores.npy")

    return auroc, scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Architecture changed, so always retrain rather than loading stale weights
    model = train_qae()
    evaluate_qae(model)