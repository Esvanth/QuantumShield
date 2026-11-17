# QuantumShield

**A Cloud-Deployed Hybrid Quantum-Classical Deep Learning Intrusion Detection System with Explainable AI and Selective Abstention**

> Esvanth Mohankumar (24311073) · MSc Computing (Artificial Intelligence) · National College of Ireland · 2026


---

## Overview

QuantumShield detects network intrusions using a **weighted soft-voting ensemble** of three classical deep networks (DNN, CNN-1D, LSTM) and a **HybridQNN** (hybrid quantum neural network, 410 parameters, built with PennyLane). A **selective abstention gate** routes low-agreement flows to a human-review queue, so borderline predictions are never acted on blindly. Every acted-on verdict comes with a **SHAP explanation** projected back onto the 78 original CICFlowMeter features so the drivers are human-readable. An **unsupervised quantum autoencoder** runs alongside as a zero-shot auxiliary detector.

The whole pipeline is containerised with Docker and deployed on **AWS**: an EC2 t3.micro behind a Flask REST API, with DynamoDB persistence, CloudWatch telemetry and a Streamlit live-detection console.

---

## Live Deployment

| Service | URL |
|---|---|
| Health check | http://3.254.138.3:5000/health |
| Streamlit dashboard | http://3.254.138.3:8501 |
| Docker image | `esvanth7/quantumshield:latest` |

---

## Results Summary

| Hypothesis | Test | CICIDS2017 | CICIoT2023 | Verdict |
|---|---|---|---|---|
| **H1** Weighted ensemble beats best single model | McNemar paired test | F1 = 0.9537 vs 0.9528, p = 2.1 × 10⁻⁵ | Tied (saturated corpus) | Supported (CICIDS2017) |
| **H2** HybridQNN parameter efficiency (≥ 90% F1 at ≥ 3× fewer params) | F1 ratio + param ratio | 93.6% of CNN-1D's F1 with 55.9× fewer params | 99.9% of LSTM's F1 | **Supported** |
| **H3** SHAP top-5 features match classical attack signatures | Expert list match | 5/5 (Fwd PSH Flags, SYN Flag Count, Flow Packets/s, Fwd Packets/s, PSH Flag Count) | Supported by analogy | **Supported** |
| **H4** Quantum autoencoder zero-shot AUROC ≥ 0.85 | Unsupervised AUROC | 0.888 | 0.941 | **Supported** |
| **H5** Member spread separates correct from incorrect | Mann-Whitney U | 5.8× separation, p < 10⁻³⁰⁰ | 7.9× separation, p < 10⁻³⁰⁰ | **Supported** |

**Abstention gate in production** (CICIDS2017, τ_abstain = 0.07):
- Routes 18.2% of flows to human review
- Lifts accuracy on the acted-on fraction from **98.43% → 99.95%**

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Esvanth/QuantumShield.git
cd QuantumShield
python -m venv .venv
source .venv/bin/activate     # Linux / macOS
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt
```

Python 3.10 is pinned because `torch==2.0.1` only ships wheels up to `cp310`.

### 2. Download datasets

- **CICIDS2017** → https://www.unb.ca/cic/datasets/ids-2017.html
  Place the five day-wise CSV files in `data/raw/cicids2017/`
- **CICIoT2023** → https://www.unb.ca/cic/datasets/iotdataset-2023.html
  Place the CSV files in `data/raw/ciciot2023/`

### 3. Preprocess

```bash
python src/preprocessing.py --dataset cicids2017
python src/preprocessing.py --dataset ciciot2023
```

StandardScaler + PCA reduces 78 features to 16 components (captures approximately 83.67% variance on CICIDS2017). Outputs land in `data/processed/<dataset>/` plus `models/<dataset>/scaler.pkl` and `models/<dataset>/pca.pkl`.

### 4. Train the four learners

```bash
python src/train.py --model dnn   --dataset cicids2017
python src/train.py --model cnn1d --dataset cicids2017
python src/train.py --model lstm  --dataset cicids2017
python src/train.py --model qnn   --dataset cicids2017

# repeat with --dataset ciciot2023
```

Classical learners finish in a few minutes on CPU. The HybridQNN takes approximately 25 minutes per corpus on an 8-core laptop (PennyLane `default.qubit` simulator, no GPU, no quantum hardware required).

### 5. Train the quantum autoencoder

```bash
python src/qae.py --dataset cicids2017
python src/qae.py --dataset ciciot2023
```

The QAE is trained on BENIGN flows only and used as the H4 zero-shot baseline.

### 6. Fit the ensemble and the abstention gate

```bash
python src/ensemble.py   --dataset cicids2017
python src/abstention.py --dataset cicids2017
# repeat for ciciot2023
```

Weights, decision threshold τ and spread threshold τ_abstain are saved in `models/<dataset>/weights.json`.

### 7. Run all hypothesis tests

```bash
python src/evaluate_all.py --dataset cicids2017 --out results/cicids2017/
python src/evaluate_all.py --dataset ciciot2023 --out results/ciciot2023/
```

### 8. Generate SHAP explanations

```bash
python src/explain.py --dataset cicids2017 --background 30 --samples 40 --coalitions 200
```

### 9. Launch the API and dashboard locally

```bash
# Terminal 1
python deployment/api.py

# Terminal 2
streamlit run deployment/dashboard.py
```

---

## Architecture

```
Raw CICFlowMeter traffic (78 features)
         │
         ▼
  StandardScaler + PCA 78 → 16 components   (fit on train split only)
         │
         ├────────── DNN       (3,730 params)   ┐
         ├────────── CNN-1D    (22,914 params)  │
         ├────────── LSTM      (56,418 params)  │
         └────────── HybridQNN (410 params)     │   ← AngleEmbedding + 4 StronglyEntanglingLayers on 8 qubits
                                                │
                       Weighted soft-voting ensemble   (w = (1,4,4,1), τ = 0.468 on CICIDS2017)
                                                │
                                                ▼
                       Abstention gate   (spread > τ_abstain → defer to human review)
                                                │
                                                ▼
                       Verdict  +  SHAP explanation (projected onto 78 raw features)
                                                │
                                                ▼
                       Flask REST API  →  DynamoDB audit log  →  CloudWatch telemetry
                                                │
                                                ▼
                       Streamlit live-detection console on AWS EC2 t3.micro
```

An **unsupervised quantum autoencoder** (8 → 4 latent, 376 parameters) runs alongside, trained on BENIGN flows only and scored by reconstruction MSE as the H4 zero-shot anomaly detector.

---

## REST API

| Endpoint | Method | Payload | Returns |
|---|---|---|---|
| `/health` | GET | none | Model-loaded status, τ_abstain, model version |
| `/predict` | POST | JSON: 16 PCA features | Verdict ∈ {ATTACK, BENIGN, ABSTAIN}, per-model probabilities, spread |
| `/predict/batch` | POST | JSON: list of vectors | List of verdicts and metadata |
| `/recent` | GET | `?day=YYYY-MM-DD&n=N` | N most recent predictions from DynamoDB |
| `/explain` | POST | JSON: 16 PCA features | SHAP contributions per component |

---

## Project Structure

```
QuantumShield/
├── data/
│   ├── raw/cicids2017/        ← 5 day-wise CSVs (Monday to Friday)
│   ├── raw/ciciot2023/        ← CSV files
│   └── processed/             ← auto-generated .npy + .pkl
├── src/
│   ├── preprocessing.py       ← StandardScaler + PCA
│   ├── models/
│   │   ├── dnn.py cnn1d.py lstm.py
│   │   └── qnn.py qae.py
│   ├── train.py               ← training entry point (--model, --dataset)
│   ├── ensemble.py            ← weighted soft-voting
│   ├── abstention.py          ← spread threshold + Mann-Whitney U
│   ├── explain.py             ← SHAP KernelExplainer + PCA back-projection
│   ├── noise_sweep.py         ← depolarising noise sweep
│   └── evaluate_all.py        ← replays H1 to H5
├── deployment/
│   ├── app.py                 ← Flask REST API
│   ├── dashboard.py           ← Streamlit console
│   └── Dockerfile
├── models/                    ← serialised weights + pca.pkl + scaler.pkl
├── results/                   ← JSON artefacts + figures
├── notebooks/
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Docker Deployment

Build and run locally:

```bash
docker build -t esvanth7/quantumshield:latest -f deployment/Dockerfile .
docker run -d -p 5000:5000 -p 8501:8501 \
    -e DATASET=cicids2017 \
    -e AWS_REGION=eu-west-1 \
    esvanth7/quantumshield:latest
```

The `DATASET` environment variable selects which weights and thresholds are loaded at start-up:
- `DATASET=cicids2017` → weights (1,4,4,1), τ = 0.468, τ_abstain = 0.07
- `DATASET=ciciot2023` → weights (4,2,2,1), τ = 0.291, τ_abstain = 0.05

A single image serves either corpus without a rebuild.

---

## Smoke test against the live API

```bash
# Health
curl http://3.254.138.3:5000/health

# Classify a BENIGN-shaped PCA vector
curl -X POST http://3.254.138.3:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [0.5,-0.3,0.2,0.1,-0.4,0.3,-0.2,0.4,
                       0.1,-0.1,0.2,-0.3,0.4,-0.2,0.3,-0.1]}'

# Read the 5 most recent predictions from DynamoDB
curl "http://3.254.138.3:5000/recent?day=$(date +%Y-%m-%d)&n=5"
```

---

## Reproducibility

- Fixed stratified 70/15/15 train/val/test split, seed 42 across NumPy, Python `random`, PyTorch and `sklearn.model_selection`
- Pinned Python dependencies in `requirements.txt` (full lock in `requirements.lock`)
- Results reproduce to the fourth decimal place on the same CPU architecture; small differences across AVX vs non-AVX machines are expected
- Dockerfile pins the exact model weights, PCA basis and `StandardScaler` into the image

---

## References (selected)

- Bharathi et al. (2026) — A quantum-classical hybrid framework for cloud intrusion detection. *EPJ Quantum Technology*, 13(4).
- Abreu et al. (2024) — A quantum neural network for intrusion detection in IoT networks. *IEEE ISCC 2024*.
- Abreu et al. (2025) — Hybrid quantum-classical learning for network traffic classification. *Int. J. Network Management*, 35(2).
- Bellante et al. (2025) — Explainable AI for cybersecurity: a survey. *Computers & Security*, 148.
- Shahraki et al. (2020) — Boosting algorithms for network intrusion detection. *Journal of Cloud Computing*, 9(1).
- Hdaib et al. (2024) — Quantum neural networks for intrusion detection: a comparative study. *arXiv:2402.08176*.
- Havlíček et al. (2019) — Supervised learning with quantum-enhanced feature spaces. *Nature*, 567(7747).
- Schuld and Petruccione (2021) — *Machine Learning with Quantum Computers*. Springer.
- Lundberg and Lee (2017) — A unified approach to interpreting model predictions. *NeurIPS*.
- Geifman and El-Yaniv (2017) — Selective classification for deep neural networks. *NeurIPS*.
- Sharafaldin et al. (2018) — Toward generating a new intrusion detection dataset. *ICISSP*.
- Neto et al. (2023) — CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment. *Sensors*, 23(13).
- Chow (1970) — On optimum recognition error and reject tradeoff. *IEEE Trans. Info. Theory*, 16(1).

Full bibliography in the research report.

---

## License and acknowledgements

This project was submitted as the MSc Research Practicum at the National College of Ireland. Supervised by **Ade Fajemisin, PhD** (Assistant Professor in Computing, NCI). CICIDS2017 and CICIoT2023 are released by the Canadian Institute for Cybersecurity and used under their terms of use.

For issues, questions or collaboration, open an issue on [GitHub](https://github.com/Esvanth/QuantumShield) or reach out: esvanth7@gmail.com.
