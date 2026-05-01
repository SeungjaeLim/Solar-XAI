"""PyTorch LSTM with L1 loss and multi-seed averaging."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class LSTMConfig:
    input_size: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    window: int = 48
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    patience: int = 6
    weight_decay: float = 0.0
    seed: int = 42


class LSTMRegressor(nn.Module):
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, window: int):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.window = window

    def __len__(self) -> int:
        return max(0, len(self.X) - self.window + 1)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, float]:
        x = self.X[idx : idx + self.window]
        y = self.y[idx + self.window - 1]
        return x, y


def make_windows(X: np.ndarray, y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Materialize sliding windows into arrays. Used for IG/permutation explanations."""
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    n = len(X) - window + 1
    if n <= 0:
        return np.empty((0, window, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
    Xw = np.stack([X[i : i + window] for i in range(n)], axis=0)
    yw = y[window - 1 :]
    return Xw, yw


def _train_one(
    cfg: LSTMConfig,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    device: str,
) -> tuple[LSTMRegressor, list[float]]:
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    train_ds = WindowDataset(X_tr, y_tr, cfg.window)
    val_ds = WindowDataset(X_va, y_va, cfg.window)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = LSTMRegressor(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.L1Loss()

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    bad = 0
    val_history: list[float] = []
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        vals: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                vals.append(float(torch.mean(torch.abs(pred - yb))))
        val_mae = float(np.mean(vals)) if vals else float("inf")
        val_history.append(val_mae)

        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model, val_history


def predict(model: LSTMRegressor, X: np.ndarray, window: int, device: str, batch_size: int = 256) -> np.ndarray:
    Xw, _ = make_windows(X, np.zeros(len(X)), window)
    if len(Xw) == 0:
        return np.full(len(X), np.nan, dtype=np.float32)
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(Xw), batch_size):
            xb = torch.from_numpy(Xw[i : i + batch_size]).to(device)
            pred = model(xb).cpu().numpy()
            out.append(pred)
    arr = np.concatenate(out, axis=0)
    # Pad initial window-1 positions with NaN so length matches input
    pad = np.full(window - 1, np.nan, dtype=arr.dtype)
    return np.concatenate([pad, arr], axis=0)


def fit_lstm_multi_seed(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: LSTMConfig,
    seeds: tuple[int, ...] = (42, 7, 123, 2024, 31337),
    device: str | None = None,
) -> tuple[list[LSTMRegressor], np.ndarray]:
    """Train one LSTM per seed and return models + averaged val predictions."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    preds: list[np.ndarray] = []
    for s in seeds:
        cfg_s = LSTMConfig(**{**cfg.__dict__, "seed": s})
        m, _ = _train_one(cfg_s, X_tr, y_tr, X_va, y_va, device)
        models.append(m)
        preds.append(predict(m, X_va, cfg.window, device))
    arr = np.stack(preds, axis=0)
    avg = np.nanmean(arr, axis=0)
    return models, avg


def predict_multi_seed(
    models: list[LSTMRegressor],
    X: np.ndarray,
    window: int,
    device: str | None = None,
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    preds = [predict(m, X, window, device) for m in models]
    return np.nanmean(np.stack(preds, axis=0), axis=0)


def lstm_objective_factory(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    epochs: int = 30,
):
    import optuna

    from ..metrics import mae

    def objective(trial: optuna.Trial) -> float:
        cfg = LSTMConfig(
            input_size=X_tr.shape[1],
            hidden_size=trial.suggest_int("hidden_size", 32, 128, step=32),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            window=trial.suggest_categorical("window", [24, 48, 72]),
            lr=trial.suggest_float("lr", 5e-4, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            epochs=epochs,
            patience=5,
            seed=42,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = _train_one(cfg, X_tr, y_tr, X_va, y_va, device)
        pred_va = predict(model, X_va, cfg.window, device)
        # Compare on the windowed portion only
        mask = ~np.isnan(pred_va)
        return mae(y_va[mask], pred_va[mask])

    return objective
