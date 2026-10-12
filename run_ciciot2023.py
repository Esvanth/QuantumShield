"""
QuantumShield - run the whole pipeline on CICIoT2023 without losing CICIDS2017.

Every training script writes to a fixed filename regardless of which dataset it
was given, so training on a second corpus overwrites the first corpus's weights
and its untagged result files. Rather than editing six scripts, this driver
handles it around them:

    1. snapshot every CICIDS2017 checkpoint and untagged result file
    2. train all five models on CICIoT2023
    3. file the new checkpoints and results under a ciciot2023 name
    4. leave the CICIoT2023 weights in the fixed locations so the evaluation
       script can read them

When you are done with CICIoT2023, put CICIDS2017 back with --restore.

Usage:
    python run_ciciot2023.py            # steps 1 to 4, about 2 hours
    python run_ciciot2023.py --restore  # put the CICIDS2017 weights back
"""

import os
import time
import shutil
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = "data/processed/ciciot2023"
SNAPSHOT = os.path.join("models", "weights", "cicids2017")
TARGET = os.path.join("models", "weights", "ciciot2023")

# The fixed paths every training script writes to, and the name to file them under
CHECKPOINTS = {
    "dnn.pth":        "models/classical/dnn_weights.pth",
    "cnn.pth":        "models/classical/cnn_weights.pth",
    "lstm.pth":       "models/classical/lstm_weights.pth",
    "hybrid_qnn.pth": "models/saved/hybrid_qnn.pth",
    "qae.pth":        "models/quantum/qae_weights.pth",
}

# Result files the scripts write WITHOUT a dataset tag, so they clobber too
UNTAGGED_RESULTS = [
    "qae_scores.npy", "qae_labels.npy", "qae_loss_history.npy",
    "hybrid_loss_history.npy", "hybrid_f1_history.npy",
    "hybrid_param_count.npy", "hybrid_probs.npy", "hybrid_preds.npy",
    "test_labels.npy",
]


def copy(src, dst):
    if not os.path.exists(src):
        log.warning("  not found, skipping: %s", src)
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    log.info("  %s -> %s", src, dst)
    return True


def snapshot_cicids2017():
    """Step 1. Preserve everything CICIDS2017 produced."""
    log.info("snapshotting CICIDS2017 checkpoints into %s", SNAPSHOT)
    for name, fixed in CHECKPOINTS.items():
        copy(fixed, os.path.join(SNAPSHOT, name))

    log.info("snapshotting untagged CICIDS2017 result files")
    for fname in UNTAGGED_RESULTS:
        copy(os.path.join("results", fname),
             os.path.join(SNAPSHOT, "results", fname))


def train_everything():
    """Step 2. Five training runs on CICIoT2023, slowest last."""
    from models.classical.dnn import run_training as train_dnn
    from models.classical.cnn import run_training as train_cnn
    from models.classical.lstm import run_training as train_lstm
    from models.quantum.train_hybrid import train_hybrid, evaluate_hybrid
    from models.quantum.qae import train_qae, evaluate_qae

    log.info("=" * 70)
    log.info("DNN on %s", DATA_DIR)
    train_dnn(data_dir=DATA_DIR)

    log.info("=" * 70)
    log.info("CNN-1D on %s", DATA_DIR)
    train_cnn(data_dir=DATA_DIR)

    log.info("=" * 70)
    log.info("LSTM on %s", DATA_DIR)
    train_lstm(data_dir=DATA_DIR)

    log.info("=" * 70)
    log.info("HybridQNN on %s, this one takes about 20 minutes", DATA_DIR)
    model = train_hybrid(data_dir=DATA_DIR)
    evaluate_hybrid(model, data_dir=DATA_DIR)

    log.info("=" * 70)
    log.info("Quantum autoencoder on %s, about 20 minutes", DATA_DIR)
    qae = train_qae(data_dir=DATA_DIR)
    evaluate_qae(qae, data_dir=DATA_DIR)


def file_ciciot2023():
    """Steps 3 and 4. File the new artefacts under a ciciot2023 name."""
    log.info("filing CICIoT2023 checkpoints into %s", TARGET)
    for name, fixed in CHECKPOINTS.items():
        copy(fixed, os.path.join(TARGET, name))

    log.info("tagging the untagged CICIoT2023 result files")
    for fname in UNTAGGED_RESULTS:
        copy(os.path.join("results", fname),
             os.path.join("results", f"ciciot2023_{fname}"))


def restore_cicids2017():
    """Put the CICIDS2017 checkpoints and result files back in place."""
    if not os.path.isdir(SNAPSHOT):
        raise SystemExit(f"no snapshot at {SNAPSHOT} - nothing to restore")

    log.info("restoring CICIDS2017 checkpoints")
    for name, fixed in CHECKPOINTS.items():
        copy(os.path.join(SNAPSHOT, name), fixed)

    log.info("restoring untagged CICIDS2017 result files")
    for fname in UNTAGGED_RESULTS:
        copy(os.path.join(SNAPSHOT, "results", fname),
             os.path.join("results", fname))

    log.info("CICIDS2017 is back in place")
    log.info("re-run 'python -m evaluation.hypotheses' if you need its numbers again")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true",
                    help="put the CICIDS2017 weights back and stop")
    args = ap.parse_args()

    if args.restore:
        restore_cicids2017()
        return

    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"{DATA_DIR} not found")

    t0 = time.time()
    snapshot_cicids2017()
    train_everything()
    file_ciciot2023()

    mins = (time.time() - t0) / 60
    log.info("=" * 70)
    log.info("done in %.0f minutes", mins)
    log.info("")
    log.info("CICIoT2023 weights are now in the fixed locations, so run:")
    log.info("    python -m evaluation.hypotheses --data-dir data/processed/ciciot2023")
    log.info("    python generate_plots.py --tag ciciot2023")
    log.info("")
    log.info("then put CICIDS2017 back with:")
    log.info("    python run_ciciot2023.py --restore")


if __name__ == "__main__":
    main()