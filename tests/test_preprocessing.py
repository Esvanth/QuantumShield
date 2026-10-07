"""Basic smoke tests for preprocessing pipeline."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.preprocessing import clean, scale_and_reduce

def make_dummy_df():
    import pandas as pd
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(200, 10), columns=[f"f{i}" for i in range(10)])
    df["Label"] = ["BENIGN"] * 100 + ["DDoS"] * 100
    return df

def test_clean_shape():
    df = make_dummy_df()
    X, yb, ym, classes = clean(df)
    assert X.shape == (200, 10)
    assert len(yb) == 200
    assert set(yb) == {0, 1}

def test_scale_and_reduce():
    df = make_dummy_df()
    X, yb, ym, _ = clean(df)
    split = 140
    Xtr, Xvl, Xte, sc, pca = scale_and_reduce(X[:split], X[split:170], X[170:], n_components=4)
    assert Xtr.shape[1] == 4
    assert Xvl.shape[1] == 4
    assert Xte.shape[1] == 4
