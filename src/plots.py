"""Pitch-ready figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def actual_vs_predicted(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    title: str = "Actual vs Predicted",
    n_days: int = 7,
) -> None:
    df = pd.DataFrame(
        {"ts": pd.to_datetime(timestamps), "actual": y_true, "predicted": y_pred}
    ).sort_values("ts")
    if len(df) > n_days * 24:
        df = df.tail(n_days * 24)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["ts"], df["actual"], label="Actual", color="#1f77b4", linewidth=2)
    ax.plot(df["ts"], df["predicted"], label="Predicted", color="#ff7f0e", linewidth=2, alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel("AC power")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def error_over_time(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    title: str = "Daily mean absolute error",
) -> None:
    df = pd.DataFrame(
        {"ts": pd.to_datetime(timestamps), "abs_err": np.abs(y_true - y_pred)}
    )
    daily = df.set_index("ts").resample("1D")["abs_err"].mean().dropna()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(daily.index, daily.values, color="#d62728", linewidth=2)
    ax.fill_between(daily.index, 0, daily.values, color="#d62728", alpha=0.25)
    ax.set_title(title)
    ax.set_ylabel("|error|")
    ax.set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    title: str = "Residual distribution",
) -> None:
    res = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(res, bins=60, kde=True, color="#2ca02c", ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Actual − Predicted")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def metric_bars(
    metrics_df: pd.DataFrame,
    out_path: str | Path,
    metric_cols: tuple[str, ...] = ("mae", "rmse", "mape"),
    title: str = "Model comparison",
) -> None:
    df = metrics_df.copy()
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 5))
    if len(metric_cols) == 1:
        axes = [axes]
    palette = sns.color_palette("Set2", n_colors=len(df))
    for ax, col in zip(axes, metric_cols):
        order = df.sort_values(col)
        ax.bar(order["model"], order[col], color=palette)
        ax.set_title(col.upper())
        ax.set_ylabel(col)
        for tick in ax.get_xticklabels():
            tick.set_rotation(35)
            tick.set_ha("right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def feature_importance_bar(
    names: list[str],
    importances: np.ndarray,
    out_path: str | Path,
    top_k: int = 20,
    title: str = "Feature importance",
) -> None:
    order = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(order))))
    ax.barh([names[i] for i in order][::-1], importances[order][::-1], color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
