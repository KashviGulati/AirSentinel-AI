# backend/ml/train_transformer.py

import os
import math
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------
# PATHS
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset", "cleaned_aqi_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# DATASET
# -----------------------
class AQITimeSeriesDataset(Dataset):
    """
    Builds sliding-window sequences from AQI_official.
    Example: seq_len=24, horizon=24
      Input:   t0..t23
      Target:  t24..t47
    """
    def __init__(self, series: np.ndarray, seq_len: int = 24, horizon: int = 24):
        self.seq_len = seq_len
        self.horizon = horizon
        self.series = series.astype(np.float32)

        self.samples = []
        max_idx = len(self.series) - (seq_len + horizon)
        for start in range(max_idx):
            x = self.series[start:start + seq_len]
            y = self.series[start + seq_len:start + seq_len + horizon]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # shape (seq_len, 1) for model
        return torch.from_numpy(x).unsqueeze(-1), torch.from_numpy(y)


# -----------------------
# POS ENCODING
# -----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return x


# -----------------------
# TRANSFORMER MODEL
# -----------------------
class AQITransformer(nn.Module):
    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        seq_len=24,
        horizon=24,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon

        # We only have 1 feature (AQI value), so input dim = 1
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # (seq_len, batch, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Predict horizon values from final time-step representation
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, horizon),
        )

    def forward(self, src):
        """
        src: (batch, seq_len, 1)
        """
        src = self.input_proj(src)                      # (batch, seq_len, d_model)
        src = src.permute(1, 0, 2)                      # (seq_len, batch, d_model)
        src = self.pos_encoder(src)                     # add positional encoding
        memory = self.transformer_encoder(src)          # (seq_len, batch, d_model)
        last_step = memory[-1]                          # (batch, d_model)
        out = self.fc_out(last_step)                    # (batch, horizon)
        return out


# -----------------------
# LOAD & PREPARE DATA
# -----------------------
def load_series():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    if "AQI_official" not in df.columns:
        raise ValueError("cleaned_aqi_data.csv must have 'AQI_official' column")

    # Time sort
    if "last_update" in df.columns:
        df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")
        df = df.sort_values("last_update")
    else:
        df = df.sort_index()

    # Clean AQI column
    df["AQI_official"] = pd.to_numeric(df["AQI_official"], errors="coerce")
    df = df.dropna(subset=["AQI_official"])

    series = df["AQI_official"].values
    return series


# -----------------------
# TRAIN LOOP
# -----------------------
def train_model(
    seq_len=24,
    horizon=24,
    batch_size=64,
    epochs=15,
    lr=1e-3,
):
    print("📌 Loading AQI time-series…")
    series = load_series()

    # Simple train/val split by time (80/20)
    split_idx = int(len(series) * 0.8)
    train_series = series[:split_idx]
    val_series = series[split_idx - (seq_len + horizon):]  # include context

    train_ds = AQITimeSeriesDataset(train_series, seq_len, horizon)
    val_ds = AQITimeSeriesDataset(val_series, seq_len, horizon)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    model = AQITransformer(
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        seq_len=seq_len,
        horizon=horizon,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)  # (batch, seq_len, 1)
            y = y.to(DEVICE)  # (batch, horizon)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_ds)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item() * x.size(0)
                val_mae += torch.abs(preds - y).mean().item() * x.size(0)

        val_loss /= len(val_ds)
        val_mae /= len(val_ds)

        print(
            f"Epoch {epoch:02d} | "
            f"Train MSE: {train_loss:.3f} | "
            f"Val MSE: {val_loss:.3f} | "
            f"Val MAE: {val_mae:.3f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model + config
    model_path = os.path.join(MODEL_DIR, "transformer_aqi.pt")
    torch.save(model.state_dict(), model_path)

    config = {
        "seq_len": seq_len,
        "horizon": horizon,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 128,
        "best_val_mae": float(best_val_mae),
    }
    cfg_path = os.path.join(MODEL_DIR, "transformer_config.pkl")
    joblib.dump(config, cfg_path)

    print("\n🎉 Transformer model trained & saved!")
    print(f"   Model: {model_path}")
    print(f"   Config: {cfg_path}")
    print(f"   Best Val MAE: {best_val_mae:.3f}")


if __name__ == "__main__":
    train_model()
