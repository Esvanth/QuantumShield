# QuantumShield 🛡️
**A Cloud-Deployed Hybrid Quantum-Classical Deep Learning Intrusion Detection System with Explainable AI**

> Esvanth Mohankumar (x24311073) ·  MSc AI, NCI Dublin · Practicum 

---

## Overview
QuantumShield detects network intrusions using a soft-voting ensemble of classical deep learning models (DNN, CNN, LSTM) and a hybrid quantum neural network (QNN built with PennyLane). SHAP explainability is applied at the ensemble boundary so every detection comes with a ranked feature explanation. The system is containerised and deployed on AWS EC2 with a Flask REST API and Streamlit dashboard.

---

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/quantumshield.git
cd quantumshield
pip install -r requirements.txt
```

### 2. Download Datasets
- **CICIDS2017** → https://www.unb.ca/cic/datasets/ids-2017.html  
  Place all 8 day-wise CSVs in `data/raw/cicids2017/`
- **CICIoT2023** → http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/  
  Place CSV files in `data/raw/ciciot2023/`

### 3. Preprocess
```bash
python data/preprocessing.py
```
Outputs saved to `data/processed/cicids2017/` and `data/processed/ciciot2023/`.

### 4. Train Classical Baselines
```bash
python models/classical/dnn.py
python models/classical/cnn.py
python models/classical/lstm.py
```

### 5. Train Quantum Models
```bash
python models/quantum/circuits.py   # HybridQNN
python models/quantum/qae.py        # Quantum Autoencoder (H4)
python models/quantum/qsvm.py       # QSVM (optional — slow on CPU)
```

### 6. Evaluate All Hypotheses
```bash
python -m evaluation.hypotheses
```

### 7. Generate SHAP Explanations
```bash
python -m explainability.shap_hybrid
```

### 8. Launch API + Dashboard
```bash
# Terminal 1
python deployment/api.py

# Terminal 2
streamlit run deployment/dashboard.py
```

---

## Architecture

```
Raw Traffic (78 features)
    ↓ StandardScaler + PCA (8 components)
    ├── DNN (classical)
    ├── CNN-1D (classical)
    ├── LSTM (classical)
    └── HybridQNN ← AngleEmbedding + StronglyEntanglingLayers (8 qubits, 4 layers)
           ↓
    Soft-Voting Ensemble
           ↓
    SHAP Explanation
           ↓
    Flask API / Streamlit Dashboard / AWS EC2
```

---

## Hypotheses

| ID | Claim | Metric | Threshold |
|----|-------|--------|-----------|
| H1 | Ensemble ≥ best classical | F1 + McNemar | p < 0.05 |
| H2 | QNN parameter efficiency | Param ratio | ≥ 3× fewer |
| H3 | SHAP plausibility | Expert review | ≥ 4/5 features |
| H4 | QAE zero-day detection | AUROC | ≥ 0.85 |

---

## Project Structure
```
quantumshield/
├── data/
│   ├── raw/cicids2017/       ← place CSVs here
│   ├── raw/ciciot2023/       ← place CSVs here
│   ├── processed/            ← auto-generated .npy + .pkl
│   └── preprocessing.py
├── models/
│   ├── classical/dnn.py, cnn.py, lstm.py
│   ├── quantum/circuits.py, qae.py, qsvm.py
│   └── ensemble.py
├── explainability/shap_hybrid.py
├── evaluation/metrics.py, hypotheses.py, ablation.py
├── deployment/api.py, dashboard.py, Dockerfile
├── notebooks/
├── tests/
├── results/shap_plots/
├── requirements.txt
└── README.md
```

---

## IBM Quantum (Hardware Validation)
Uncomment the IBM device lines in `models/quantum/circuits.py` and add your API token:
```python
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN", overwrite=True)
dev_hw = qml.device("qiskit.ibmq", wires=8, backend="ibm_brisbane", shots=1024)
```
Free tier available at: https://quantum.ibm.com/

---

## Docker Deployment
```bash
docker build -t quantumshield -f deployment/Dockerfile .
docker run -p 5000:5000 quantumshield
```

---

## References
- Bharathi et al. (2026) — Hybrid QNN for IoT IDS, EPJ Quantum Technology
- Abreu et al. (2024) — QML-IDS, IEEE ISCC
- Bellante et al. (2025) — QML in Cybersecurity, Computers & Security
- Havlíček et al. (2019) — Quantum-enhanced feature spaces, Nature
- Lundberg & Lee (2017) — SHAP, NeurIPS
- Schuld & Petruccione (2021) — Machine Learning with Quantum Computers, Springer
