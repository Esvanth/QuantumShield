"""
QuantumShield – Classical Baseline: LSTM
Input: 8-dimensional PCA features treated as a 1-step sequence (batch, 1, 8)
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: (batch, input_dim) → (batch, 1, input_dim)  [single time-step]
        x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, batch, hidden)
        return self.classifier(h_n[-1])


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (out.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            loss = criterion(out, y_batch)
            total_loss += loss.item() * len(y_batch)
            correct += (out.argmax(1) == y_batch).sum().item()
            total += len(y_batch)
    return total_loss / total, correct / total


def run_training(data_dir="data/processed/cicids2017", epochs=20, batch_size=256, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_val   = np.load(os.path.join(data_dir, "X_val.npy"))
    yb_train = np.load(os.path.join(data_dir, "yb_train.npy"))
    yb_val   = np.load(os.path.join(data_dir, "yb_val.npy"))

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(yb_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.LongTensor(yb_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model = LSTMClassifier(input_dim=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    save_path = "models/classical/lstm_weights.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        log.info("Epoch %02d/%02d | train loss=%.4f acc=%.4f | val loss=%.4f acc=%.4f",
                 epoch, epochs, tr_loss, tr_acc, vl_loss, vl_acc)
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)
            log.info("  ✓ Saved best model (val_acc=%.4f)", best_val_acc)

    log.info("Training complete. Best val acc: %.4f", best_val_acc)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_training()
