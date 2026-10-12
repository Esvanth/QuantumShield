"""
QuantumShield - recover the original feature names dropped during preprocessing.

preprocessing.clean() builds its feature matrix with

    df.drop(columns=['Label']).select_dtypes(include=[np.number])

and takes .values, so the column ORDER of that DataFrame is exactly what the
columns of PCA.components_ correspond to. The names themselves were never
saved, which makes SHAP attributions uninterpretable (PC3 is not a network
feature). This script reproduces the same selection from the raw CSVs and
writes the names next to the processed splits.

It verifies the recovered count against PCA.components_.shape[1] and refuses
to save a mismatched list, because a silently misaligned name vector would
produce confident, wrong feature attributions.

Run once per dataset:
    python -m data.recover_feature_names
    python -m data.recover_feature_names --dataset ciciot2023
"""

import os
import glob
import pickle
import logging
import argparse

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def recover(raw_dir, out_dir, label_col="Label", nrows=20000, from_folder=False):
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    log.info("sampling up to %d rows from each of %d files", nrows, len(csv_files))

    frames = []
    for f in csv_files:
        df = pd.read_csv(f, encoding="utf-8", low_memory=False, nrows=nrows)
        df.columns = df.columns.str.strip()
        if from_folder:
            # CICIoT2023 gets its label from the parent folder, same as the loader
            df["Label"] = os.path.basename(os.path.dirname(f))
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    if label_col not in combined.columns:
        lower = {c.lower(): c for c in combined.columns}
        if label_col.lower() not in lower:
            raise ValueError(f"Label column not found. Available: {list(combined.columns)}")
        label_col = lower[label_col.lower()]

    # Exactly the selection preprocessing.clean() performs
    feature_df = combined.drop(columns=[label_col]).select_dtypes(include=[np.number])
    names = list(feature_df.columns)
    log.info("recovered %d numeric feature columns", len(names))

    with open(os.path.join(out_dir, "pca.pkl"), "rb") as fh:
        pca = pickle.load(fh)
    expected = pca.components_.shape[1]

    if len(names) != expected:
        log.error("MISMATCH: recovered %d names, PCA was fitted on %d features",
                  len(names), expected)
        log.error("A partial read can infer dtypes differently from the full read.")
        log.error("Retry with a larger --nrows, or 0 to read the files in full.")
        log.error("recovered: %s", names)
        raise SystemExit(1)

    out_path = os.path.join(out_dir, "feature_names.npy")
    np.save(out_path, np.array(names))
    log.info("saved %d feature names to %s", len(names), out_path)
    log.info("first 12: %s", names[:12])
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cicids2017",
                    choices=["cicids2017", "ciciot2023"])
    ap.add_argument("--nrows", type=int, default=20000,
                    help="rows to sample per file for dtype inference, 0 reads in full")
    args = ap.parse_args()

    raw_dir = f"data/raw/{args.dataset}"
    out_dir = f"data/processed/{args.dataset}"
    nrows = None if args.nrows == 0 else args.nrows

    recover(raw_dir, out_dir,
            nrows=nrows,
            from_folder=(args.dataset == "ciciot2023"))


if __name__ == "__main__":
    main()