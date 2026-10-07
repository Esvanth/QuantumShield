"""Smoke tests for model forward passes."""
import torch, pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_dnn_forward():
    from models.classical.dnn import DNN
    model = DNN(input_dim=8)
    x = torch.randn(4, 8)
    out = model(x)
    assert out.shape == (4, 2)

def test_cnn_forward():
    from models.classical.cnn import CNN1D
    model = CNN1D(input_dim=8)
    x = torch.randn(4, 8)
    out = model(x)
    assert out.shape == (4, 2)

def test_lstm_forward():
    from models.classical.lstm import LSTMClassifier
    model = LSTMClassifier(input_dim=8)
    x = torch.randn(4, 8)
    out = model(x)
    assert out.shape == (4, 2)
