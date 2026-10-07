"""
QuantumShield – Quantum Circuit Definitions
Uses PennyLane 0.35 with 8 qubits and 4 variational layers.

Architecture:
  AngleEmbedding (8 PCA features → 8 qubits)
  StronglyEntanglingLayers (4 layers, shallow to avoid barren plateaus)
  Measurement: PauliZ on qubit 0 → scalar expectation value

Training runs on default.qubit (simulator).
Hardware validation uses qiskit.ibmq (ibm_brisbane, 127 qubits) — swap dev_train for dev_hw.
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
N_QUBITS = 8
N_LAYERS = 4

dev_train = qml.device("default.qubit", wires=N_QUBITS)

# IBM hardware device — uncomment and set your token for hardware validation
# from qiskit_ibm_runtime import QiskitRuntimeService
# QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN_HERE", overwrite=True)
# dev_hw = qml.device("qiskit.ibmq", wires=N_QUBITS, backend="ibm_brisbane", shots=1024)


# ---------------------------------------------------------------------------
# Core QNN circuit
# ---------------------------------------------------------------------------

@qml.qnode(dev_train, interface="torch")
def qnn_circuit(inputs, weights):
    """
    inputs  : tensor (8,)   – one PCA-reduced sample
    weights : tensor (4, 8, 3) – StronglyEntanglingLayers params
    Returns : expectation value of PauliZ on qubit 0
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


def get_weight_shapes():
    """Return the weight shape required by StronglyEntanglingLayers."""
    return qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)


# ---------------------------------------------------------------------------
# QCNN circuit (simplified – alternating conv + pooling blocks)
# ---------------------------------------------------------------------------

@qml.qnode(dev_train, interface="torch")
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

@qml.qnode(dev_train, interface="torch")
def vqc_circuit(inputs, weights):
    """
    Simple VQC: RY encoding + parametric RY/RZ layers with entanglement.
    weights : (N_LAYERS, N_QUBITS, 2)
    """
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

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
    Outputs a scalar expectation value → used as one feature in the hybrid model.
    """

    def __init__(self):
        super().__init__()
        weight_shape = get_weight_shapes()
        self.weights = nn.Parameter(torch.randn(*weight_shape) * 0.01)

    def forward(self, x):
        # x: (batch, 8)
        batch_out = torch.stack([qnn_circuit(x[i], self.weights) for i in range(x.shape[0])])
        return batch_out.unsqueeze(1)   # (batch, 1)


class HybridQNN(nn.Module):
    """
    Hybrid model: quantum expectation + classical post-processing head.
    Hypothesis H2: parameter efficiency ≥ 3× vs classical baselines.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.quantum = QuantumLayer()
        self.head = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        q_out = self.quantum(x)   # (batch, 1)
        return self.head(q_out)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sample = torch.randn(4, N_QUBITS)
    model = HybridQNN()
    out = model(sample)
    log.info("HybridQNN output shape: %s | trainable params: %d",
             out.shape, model.count_params())
    log.info("QNN weight shape: %s", get_weight_shapes())
    log.info("Circuit diagram:\n%s", qml.draw(qnn_circuit)(sample[0], model.quantum.weights))
