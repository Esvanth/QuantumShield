"""
QuantumShield – Quantum Support Vector Machine (QSVM)
Uses PennyLane's quantum kernel with scikit-learn SVC.
8 qubits, ZZFeatureMap-style encoding.
"""

import os
import logging
import pickle
import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

log = logging.getLogger(__name__)

N_QUBITS = 8
dev = qml.device("default.qubit", wires=N_QUBITS)


# ---------------------------------------------------------------------------
# Quantum kernel
# ---------------------------------------------------------------------------

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    """
    ZZ-feature-map style quantum kernel.
    K(x1, x2) = |<phi(x2)|phi(x1)>|^2
    """
    # Encode x1
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)
        qml.RZ(2 * x1[i], wires=i)
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(2 * (np.pi - x1[i]) * (np.pi - x1[i + 1]), wires=i + 1)
        qml.CNOT(wires=[i, i + 1])

    # Inverse encoding of x2 (adjoint)
    for i in range(N_QUBITS - 2, -1, -1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(-2 * (np.pi - x2[i]) * (np.pi - x2[i + 1]), wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
    for i in range(N_QUBITS - 1, -1, -1):
        qml.RZ(-2 * x2[i], wires=i)
        qml.Hadamard(wires=i)

    return qml.probs(wires=range(N_QUBITS))


def quantum_kernel(X1, X2):
    """Compute the full kernel matrix K[i,j] = K(X1[i], X2[j])."""
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            probs = kernel_circuit(X1[i], X2[j])
            K[i, j] = probs[0]   # probability of measuring |0...0> = overlap
        if (i + 1) % 50 == 0:
            log.info("Kernel: %d/%d rows done", i + 1, n1)
    return K


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_qsvm(data_dir="data/processed/cicids2017", max_train=1000):
    """
    NOTE: Full kernel computation on large datasets is expensive.
    max_train caps the training set for feasibility on CPU.
    """
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    yb_train = np.load(os.path.join(data_dir, "yb_train.npy"))

    # Sub-sample for kernel computation feasibility
    if len(X_train) > max_train:
        idx = np.random.choice(len(X_train), max_train, replace=False)
        X_train, yb_train = X_train[idx], yb_train[idx]
    log.info("QSVM training on %d samples", len(X_train))

    log.info("Computing training kernel matrix (%d × %d)...", len(X_train), len(X_train))
    K_train = quantum_kernel(X_train, X_train)

    svm = SVC(kernel="precomputed", C=1.0, random_state=42)
    svm.fit(K_train, yb_train)

    save_path = "models/quantum/qsvm.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"svm": svm, "X_train": X_train}, f)
    log.info("QSVM saved to %s", save_path)
    return svm, X_train


def evaluate_qsvm(data_dir="data/processed/cicids2017", max_test=500):
    save_path = "models/quantum/qsvm.pkl"
    with open(save_path, "rb") as f:
        obj = pickle.load(f)
    svm, X_sv = obj["svm"], obj["X_train"]

    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    yb_test = np.load(os.path.join(data_dir, "yb_test.npy"))

    if len(X_test) > max_test:
        idx = np.random.choice(len(X_test), max_test, replace=False)
        X_test, yb_test = X_test[idx], yb_test[idx]

    log.info("Computing test kernel matrix (%d × %d)...", len(X_test), len(X_sv))
    K_test = quantum_kernel(X_test, X_sv)
    y_pred = svm.predict(K_test)

    acc = accuracy_score(yb_test, y_pred)
    log.info("QSVM test accuracy: %.4f", acc)
    log.info("\n%s", classification_report(yb_test, y_pred, target_names=["Benign", "Attack"]))
    return acc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    svm, X_sv = train_qsvm()
    evaluate_qsvm()
