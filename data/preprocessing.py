"""
QuantumShield - Data Preprocessing Pipeline
Handles CICIDS2017 and CICIoT2023 datasets.
"""

import os
import glob
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import PCA_N_COMPONENTS as _DEFAULT_N_COMPONENTS
except ImportError:
    _DEFAULT_N_COMPONENTS = 16

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_cicids2017(data_dir="data/raw/cicids2017"):
    """Concatenate all day-wise CICIDS2017 CSVs into one DataFrame."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    log.info("Loading %d CICIDS2017 files ...", len(csv_files))
    frames = []
    for f in csv_files:
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    log.info("CICIDS2017 raw shape: %s", combined.shape)
    return combined


def load_ciciot2023(data_dir="data/raw/ciciot2023"):
    """
    CICIoT2023 has no label column — derived from parent folder name.
    e.g. data/raw/ciciot2023/Backdoor_Malware/foo.csv  -> label = 'Backdoor_Malware'
         data/raw/ciciot2023/Benign_Final/foo.csv      -> label = 'BENIGN'
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    log.info("Loading %d CICIoT2023 files ...", len(csv_files))
    frames = []
    for f in csv_files:
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        df.columns = df.columns.str.strip()
        folder = os.path.basename(os.path.dirname(f))
        label = "BENIGN" if "benign" in folder.lower() else folder
        df["Label"] = label
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    log.info("CICIoT2023 raw shape: %s", combined.shape)
    return combined


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(df, label_col="Label"):
    """
    1. Drop duplicates.
    2. Keep only numeric feature columns.
    3. Replace inf / NaN with column median.
    4. Build binary label (0 = BENIGN, 1 = ATTACK).
    5. Build multi-class label via LabelEncoder.

    Returns
    -------
    X            : np.ndarray  (n_samples, n_features)
    y_binary     : np.ndarray  (n_samples,)
    y_multi      : np.ndarray  (n_samples,)
    class_names  : list[str]
    """
    log.info("Cleaning dataset (shape before: %s) ...", df.shape)
    df = df.drop_duplicates()

    # Separate label — try exact match first, then case-insensitive fallback
    if label_col not in df.columns:
        col_lower = {c.lower(): c for c in df.columns}
        if label_col.lower() in col_lower:
            label_col = col_lower[label_col.lower()]
        else:
            raise ValueError(f"Label column '{label_col}' not found. Available: {df.columns.tolist()}")
    labels = df[label_col].astype(str).str.strip()

    # Keep only numeric features
    feature_df = df.drop(columns=[label_col]).select_dtypes(include=[np.number])

    # Replace inf with NaN then fill with median
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    X = feature_df.values.astype(np.float32)

    # Binary label — BENIGN -> 0, all attacks -> 1
    # .contains("BENIGN") catches both "BENIGN" and "Benign_Final"
    y_binary = np.where(labels.str.upper().str.contains("BENIGN"), 0, 1).astype(np.int64)

    # Multi-class label
    le = LabelEncoder()
    y_multi = le.fit_transform(labels).astype(np.int64)
    class_names = list(le.classes_)

    log.info("Cleaned shape: %s | classes: %s", X.shape, class_names)
    log.info("Binary distribution — BENIGN: %d | ATTACK: %d",
             (y_binary == 0).sum(), (y_binary == 1).sum())
    return X, y_binary, y_multi, class_names


# ---------------------------------------------------------------------------
# Scaling + Dimensionality Reduction
# ---------------------------------------------------------------------------

def scale_and_reduce(X_train, X_val, X_test, n_components=_DEFAULT_N_COMPONENTS):
    """
    StandardScaler (fit on train only) -> PCA to n_components.

    Returns
    -------
    X_train_pca, X_val_pca, X_test_pca : np.ndarray
    scaler                              : fitted StandardScaler
    pca                                 : fitted PCA
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca   = pca.transform(X_val_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    explained = pca.explained_variance_ratio_.sum() * 100
    log.info("PCA explained variance (%d components): %.2f%%", n_components, explained)
    if explained < 80:
        log.warning("Explained variance %.2f%% < 80%% — consider increasing n_components.", explained)

    return X_train_pca, X_val_pca, X_test_pca, scaler, pca


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def build_pipeline(
    dataset_name,
    load_fn,
    data_dir,
    label_col="Label",
    n_components=_DEFAULT_N_COMPONENTS,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
    out_dir=None,
):
    """
    End-to-end: load -> clean -> split (70/15/15) -> scale -> PCA -> save.

    Saves to data/processed/<dataset_name>/:
        X_train.npy, X_val.npy, X_test.npy
        yb_train.npy, yb_val.npy, yb_test.npy   (binary)
        ym_train.npy, ym_val.npy, ym_test.npy   (multi-class)
        class_names.npy
        scaler.pkl, pca.pkl
    """
    if out_dir is None:
        out_dir = os.path.join("data", "processed", dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    df = load_fn(data_dir)
    X, y_binary, y_multi, class_names = clean(df, label_col=label_col)

    # Stratified 70 / 15 / 15 split
    X_train, X_tmp, yb_train, yb_tmp, ym_train, ym_tmp = train_test_split(
        X, y_binary, y_multi,
        test_size=(test_size + val_size),
        stratify=y_binary,
        random_state=random_state,
    )
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, yb_val, yb_test, ym_val, ym_test = train_test_split(
        X_tmp, yb_tmp, ym_tmp,
        test_size=(1 - rel_val),
        stratify=yb_tmp,
        random_state=random_state,
    )
    log.info("Split — train: %d | val: %d | test: %d", len(X_train), len(X_val), len(X_test))

    X_train_pca, X_val_pca, X_test_pca, scaler, pca = scale_and_reduce(
        X_train, X_val, X_test, n_components=n_components
    )

    np.save(os.path.join(out_dir, "X_train.npy"), X_train_pca)
    np.save(os.path.join(out_dir, "X_val.npy"),   X_val_pca)
    np.save(os.path.join(out_dir, "X_test.npy"),  X_test_pca)
    np.save(os.path.join(out_dir, "yb_train.npy"), yb_train)
    np.save(os.path.join(out_dir, "yb_val.npy"),   yb_val)
    np.save(os.path.join(out_dir, "yb_test.npy"),  yb_test)
    np.save(os.path.join(out_dir, "ym_train.npy"), ym_train)
    np.save(os.path.join(out_dir, "ym_val.npy"),   ym_val)
    np.save(os.path.join(out_dir, "ym_test.npy"),  ym_test)
    np.save(os.path.join(out_dir, "class_names.npy"), np.array(class_names))

    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)

    log.info("Saved processed data to %s", out_dir)
    return X_train_pca, X_val_pca, X_test_pca, yb_train, yb_val, yb_test, class_names


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Preprocessing CICIDS2017 ===")
    build_pipeline(
        dataset_name="cicids2017",
        load_fn=load_cicids2017,
        data_dir="data/raw/cicids2017",
        label_col="Label",
    )

    print("\n=== Preprocessing CICIoT2023 ===")
    build_pipeline(
        dataset_name="ciciot2023",
        load_fn=load_ciciot2023,
        data_dir="data/raw/ciciot2023",
        label_col="Label",
    )