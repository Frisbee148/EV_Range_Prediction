"""Publication-grade evaluation plots with multi-seed variance guards (PRD §16, §22.8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def assert_minimum_seeds(n_seeds: int, min_seeds: int = 3, context: str = "") -> None:
    """PRD §22.8: refuse to draw a comparison chart when fewer than 3 seeds are present."""
    if n_seeds < min_seeds:
        prefix = f"[{context}] " if context else ""
        raise ValueError(
            f"{prefix}Comparison plotting requires at least {min_seeds} seeds (PRD §22.8), got {n_seeds}."
        )


def plot_model_comparison_bars(
    seed_metrics_by_model: dict[str, list[dict[str, float]]],
    output_path: str | Path,
    metric_key: str = "range_mae_km",
    ylabel: str = "Range MAE (km)",
    min_seeds: int = 3,
) -> Path:
    """Draw a comparison bar chart with ±1 std error bars over seeds (PRD §22.8)."""
    if not seed_metrics_by_model:
        raise ValueError("seed_metrics_by_model cannot be empty.")

    models = list(seed_metrics_by_model.keys())
    means: list[float] = []
    stds: list[float] = []

    for m_name in models:
        runs = seed_metrics_by_model[m_name]
        assert_minimum_seeds(len(runs), min_seeds=min_seeds, context=f"Model {m_name}")
        vals = np.asarray([float(r[metric_key]) for r in runs], dtype=np.float64)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=0)))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    x = np.arange(len(models))
    ax.bar(x, means, yerr=stds, capsize=5, color="#2b5c8f", alpha=0.88, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Model Comparison — {ylabel} (mean ± std)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_data_ablation_curve(
    ablation_results: dict[float, dict[str, list[dict[str, float]]]],
    output_path: str | Path,
    metric_key: str = "range_mae_km",
    ylabel: str = "Range MAE (km)",
    min_seeds: int = 3,
) -> Path:
    """Plot Range MAE vs. training fraction for Model A and Model B with ±1 std bands (PRD §11.1, §16.2)."""
    fractions = sorted(ablation_results.keys())
    if not fractions:
        raise ValueError("ablation_results cannot be empty.")

    model_names = sorted(next(iter(ablation_results.values())).keys())
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)
    palette = {"A": "#d95f02", "B": "#1b9e77", "C": "#7570b3"}

    x_pct = np.asarray([f * 100.0 for f in fractions], dtype=np.float64)
    for m_name in model_names:
        means = []
        stds = []
        for frac in fractions:
            runs = ablation_results[frac][m_name]
            assert_minimum_seeds(
                len(runs), min_seeds=min_seeds, context=f"Ablation {frac:.2f} Model {m_name}"
            )
            vals = np.asarray([float(r[metric_key]) for r in runs], dtype=np.float64)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=0)))

        mean_arr = np.asarray(means)
        std_arr = np.asarray(stds)
        color = palette.get(m_name, None)
        ax.plot(x_pct, mean_arr, marker="o", linewidth=2.0, label=f"Model {m_name}", color=color)
        ax.fill_between(
            x_pct,
            mean_arr - std_arr,
            mean_arr + std_arr,
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Training Trips Fraction (%)")
    ax.set_ylabel(ylabel)
    ax.set_title("E2 Data Ablation — Range Error vs. Training Fraction")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_lambda_sweep_curve(
    sweep_results: dict[float, list[dict[str, float]]],
    output_path: str | Path,
    min_seeds: int = 3,
) -> Path:
    """Plot validation Range MAE and Physics Residual RMS across log-spaced `lambda_max` (PRD §16.7)."""
    lambdas = sorted(sweep_results.keys())
    if not lambdas:
        raise ValueError("sweep_results cannot be empty.")

    mae_means, mae_stds = [], []
    res_means, res_stds = [], []
    for lam in lambdas:
        runs = sweep_results[lam]
        assert_minimum_seeds(len(runs), min_seeds=min_seeds, context=f"Lambda {lam}")
        maes = np.asarray([float(r["range_mae_km"]) for r in runs], dtype=np.float64)
        ress = np.asarray([float(r["residual_rms"]) for r in runs], dtype=np.float64)
        mae_means.append(float(np.mean(maes)))
        mae_stds.append(float(np.std(maes, ddof=0)))
        res_means.append(float(np.mean(ress)))
        res_stds.append(float(np.std(ress, ddof=0)))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7.0, 4.5), dpi=150)
    x = np.asarray(lambdas, dtype=np.float64)

    ax1.errorbar(
        x, mae_means, yerr=mae_stds, marker="o", color="#1b9e77", capsize=4, label="Range MAE (km)"
    )
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\lambda_{\max}$")
    ax1.set_ylabel("Validation Range MAE (km)", color="#1b9e77")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.errorbar(
        x,
        res_means,
        yerr=res_stds,
        marker="s",
        linestyle="--",
        color="#d95f02",
        capsize=4,
        label="Residual RMS",
    )
    ax2.set_ylabel("Physics Residual RMS", color="#d95f02")

    fig.suptitle(r"E7 $\lambda_{\max}$ Regularisation Sweep")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_loss_curves(
    history: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot training and validation loss trajectories across epochs."""
    if not history:
        raise ValueError("history cannot be empty.")

    epochs = [int(r["epoch"]) for r in history]
    tr_tot = [float(r["train_loss_total"]) for r in history]
    va_tot = [float(r["val_loss_total"]) for r in history]
    tr_dat = [float(r["train_loss_data"]) for r in history]
    va_dat = [float(r["val_loss_data"]) for r in history]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=150)
    ax.plot(epochs, tr_tot, label="Train Total Loss", linewidth=1.8)
    ax.plot(epochs, va_tot, label="Val Total Loss", linewidth=1.8)
    ax.plot(epochs, tr_dat, linestyle="--", label="Train Data Loss", alpha=0.8)
    ax.plot(epochs, va_dat, linestyle="--", label="Val Data Loss", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss Curves")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
