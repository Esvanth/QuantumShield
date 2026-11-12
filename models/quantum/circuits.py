"""
QuantumShield - Quantum Circuit Definitions
Uses PennyLane 0.44 with 8 qubits and 4 variational layers.

Noise robustness:
  Device changed to default.mixed to support DepolarizingChannel. NOISE_P
  controls the depolarising probability applied on every qubit after the
  entangling block. Set NOISE_P = 0.0 for the noise-free baseline, which
  reproduces the existing results exactly.
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
N_FEATURES = 16
N_QUBITS   = 8
N_LAYERS   = 4
Z_DIVISOR  = 3.0
NOISE_P    = 0.0       # depolarising probability per qubit after entangling block

dev_train = qml.device("default.mixed", wires=N_QUBITS)


@qml.qnode(dev_train, interface="torch", diff_method="backprop")
def qnn_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    if NOISE_P > 0:
        for q in range(N_QUBITS):
            qml.DepolarizingChannel(NOISE_P, wires=q)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def get_weight_shapes():
    return qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)


# qcnn_circuit and vqc_circuit unchanged (they are not used by HybridQNN)

@qml.qnode(dev_train, interface="torch", diff_method="backprop")
def qcnn_circuit(inputs, weights_conv, weights_pool):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    for i in range(0, N_QUBITS - 1, 2):
        qml.CNOT(wires=[i, i + 1])
        qml.RY(weights_conv[i], wires=i)
        qml.RY(weights_conv[i + 1], wires=i + 1)
    for i in range(N_QUBITS // 2):
        qml.CRY(weights_pool[i], wires=[2 * i, 2 * i + 1])
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev_train, interface="torch", diff_method="backprop")
def vqc_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])
    return qml.expval(qml.PauliZ(0))


# ---------------------------------------------------------------------------
# PyTorch wrapper for the QNN (unchanged below)
# ---------------------------------------------------------------------------

class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shape = get_weight_shapes()
        self.weights = nn.Parameter(torch.randn(*weight_shape) * 0.01)

    def forward(self, x):
        out = qnn_circuit(x, self.weights)
        return torch.stack(out, dim=-1).float()


class HybridQNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.pre_encoder = nn.Linear(N_FEATURES, N_QUBITS)
        self.quantum = QuantumLayer()
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
        z = (X_raw - self.x_mean) / self.x_std
        return torch.tanh(z / Z_DIVISOR)

    def forward(self, x):
        angles = torch.tanh(self.pre_encoder(x))
        q_out = self.quantum(angles)
        return self.head(q_out)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sample = torch.randn(4, N_FEATURES)
    model = HybridQNN()
    out = model(sample)
    log.info("HybridQNN output shape: %s | trainable params: %d",
             out.shape, model.count_params())
    log.info("QNN weight shape: %s", get_weight_shapes())