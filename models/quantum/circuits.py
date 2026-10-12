"""
QuantumShield – Quantum Circuit Definitions
Uses PennyLane 0.44 with 8 qubits and 4 variational layers.

Architecture:
  Scaler        : z-score on training data, then tanh squash into (-1, 1).
                  Bounded so large feature values cannot wrap past 2*pi on the
                  Bloch sphere and masquerade as small ones.
  Pre-encoder   : classical Linear(16 -> 8) + tanh, produces the rotation angles
  Quantum layer : AngleEmbedding + StronglyEntanglingLayers (4 layers, shallow
                  to avoid barren plateaus) -> 8 PauliZ expectation values
  Head          : classical Linear(8 -> 16) -> Linear(16 -> 2)

Training runs on default.qubit with backprop, which vectorises the whole batch
in one state-vector simulation.
"""

import logging
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------
N_FEATURES = 16        # all PCA components
N_QUBITS   = 8
N_LAYERS   = 4
Z_DIVISOR  = 3.0       # spreads training z-scores across the useful tanh range

dev_train = qml.device("default.qubit", wires=N_QUBITS)


# ---------------------------------------------------------------------------
# Core QNN circuit
# ---------------------------------------------------------------------------

@qml.qnode(dev_train, interface="torch", diff_method="backprop")
def qnn_circuit(inputs, weights):
    """
    inputs  : tensor (batch, 8) – a batch of rotation angles in (-1, 1)
    weights : tensor (4, 8, 3) – StronglyEntanglingLayers params
    Returns : expectation value of PauliZ on every qubit
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def get_weight_shapes():
    """Return the weight shape required by StronglyEntanglingLayers."""
    return qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)


# ---------------------------------------------------------------------------
# QCNN circuit (simplified – alternating conv + pooling blocks)
# ---------------------------------------------------------------------------

@qml.qnode(dev_train, interface="torch", diff_method="backprop")
def qcnn_circuit(inputs, weights_conv, weights_pool):
    """
    Quantum Convolutional Neural Network.
    weights_conv : (n_conv_params,)
    weights_pool : (n_pool_params,)
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")

    # Convolutional layer: CNOT pairs + local rotations
    for i in range(0, N_QUBITS - 1, 2):
        qml.CNOT(wires=[i, i + 1])
        qml.RY(weights_conv[i], wires=i)
        qml.RY(weights_conv[i + 1], wires=i + 1)

    # Pooling layer: measure half qubits (represented by conditional rotation)
    for i in range(N_QUBITS // 2):
        qml.CRY(weights_pool[i], wires=[2 * i, 2 * i + 1])

    return qml.expval(qml.PauliZ(0))


# ---------------------------------------------------------------------------
# VQC – Variational Quantum Classifier
# ---------------------------------------------------------------------------

@qml.qnode(dev_train, interface="torch", diff_method="backprop")
def vqc_circuit(inputs, weights):
    """
    Simple VQC: RY encoding + parametric RY/RZ layers with entanglement.
    weights : (N_LAYERS, N_QUBITS, 2)
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")

    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])   # wrap-around entanglement

    return qml.expval(qml.PauliZ(0))


# ---------------------------------------------------------------------------
# PyTorch wrapper for the QNN
# ---------------------------------------------------------------------------

class QuantumLayer(nn.Module):
    """
    Single quantum layer that wraps qnn_circuit as a PyTorch module.
    Outputs one expectation value per qubit, used as the feature vector for
    the classical head.
    """

    def __init__(self):
        super().__init__()
        weight_shape = get_weight_shapes()
        self.weights = nn.Parameter(torch.randn(*weight_shape) * 0.01)

    def forward(self, x):
        # x: (batch, N_QUBITS) -- the whole batch goes through in one simulation
        out = qnn_circuit(x, self.weights)
        return torch.stack(out, dim=-1).float()   # (batch, N_QUBITS)


class HybridQNN(nn.Module):
    """
    Hybrid model: classical pre-encoder -> quantum feature extraction ->
    classical classification head.
    Hypothesis H2: parameter efficiency >= 3x vs classical baselines.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.pre_encoder = nn.Linear(N_FEATURES, N_QUBITS)
        self.quantum = QuantumLayer()
        self.head = nn.Sequential(
            nn.Linear(N_QUBITS, 16), nn.ReLU(),
            nn.Linear(16, num_classes),
        )

        # Scaler stats stored as buffers so they save and load with the weights
        self.register_buffer("x_mean", torch.zeros(N_FEATURES))
        self.register_buffer("x_std", torch.ones(N_FEATURES))

    def fit_scaler(self, X_raw):
        """Fit z-score statistics on the training split only."""
        self.x_mean.copy_(X_raw.mean(dim=0))
        std = X_raw.std(dim=0)
        std[std < 1e-8] = 1.0
        self.x_std.copy_(std)

    def scale(self, X_raw):
        """z-score then tanh. Bounded and monotonic, so no angle wrapping."""
        z = (X_raw - self.x_mean) / self.x_std
        return torch.tanh(z / Z_DIVISOR)

    def forward(self, x):
        """x is (batch, N_FEATURES), already scaled."""
        angles = torch.tanh(self.pre_encoder(x))   # (batch, N_QUBITS), in (-1, 1)
        q_out = self.quantum(angles)               # (batch, N_QUBITS)
        return self.head(q_out)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sample = torch.randn(4, N_FEATURES)
    model = HybridQNN()
    out = model(sample)
    log.info("HybridQNN output shape: %s | trainable params: %d",
             out.shape, model.count_params())
    log.info("QNN weight shape: %s", get_weight_shapes())