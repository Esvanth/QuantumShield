# config.py – Central configuration for QuantumShield
PCA_N_COMPONENTS: int = 16
INPUT_DIM: int = PCA_N_COMPONENTS
BATCH_SIZE: int = 256
EPOCHS: int = 30
LR: float = 1e-3
RANDOM_SEED: int = 42

DATA_DIR_CICIDS = "data/processed/cicids2017"
DATA_DIR_CICIOT = "data/processed/ciciot2023"

N_QUBITS: int = 8
N_Q_LAYERS: int = 4
QAE_LATENT: int = 4