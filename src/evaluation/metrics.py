"""Evaluation metrics, SOC-band stratification, multi-seed aggregation, and paired comparisons (PRD §16.10, §17)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.training.trainer import evaluate_predictions_dict

SOC_BANDS: tuple[tuple[str, float, float], ...] = (
    ("100-75%", 0.75, 1.0001),
    ("75-50%", 0.50, 0.75),
    ("50-25%", 0.25, 0.50),
    ("25-0%", -0.0001, 0.25),
)


def compute_all_metrics(
    soc_hat: np.ndarray,
    p_hat_w: np.ndarray,
    range_hat_m: np.ndarray,
    soc_true: np.ndarray,
    p_true_w: np.ndarray,
    range_true_m: np.ndarray,
    range_mask: np.ndarray,
    dt_s: np.ndarray,
    e_usable_j: np.ndarray,
    eps_m: float = 100.0,
) -> dict[str, float]:
    """Compute all PRD §17.1 accuracy and §17.2 physics-consistency metrics."""
    return evaluate_predictions_dict(
        soc_hat=soc_hat,
        p_hat_w=p_hat_w,
        range_hat_m=range_hat_m,
        soc_true=soc_true,
        p_true_w=p_true_w,
        range_true_m=range_true_m,
        range_mask=range_mask,
        dt_s=dt_s,
        e_usable_j=e_usable_j,
        eps_m=eps_m,
    )


def compute_soc_stratified_range_metrics(
    soc_true: np.ndarray,
    range_hat_m: np.ndarray,
    range_true_m: np.ndarray,
    range_mask: np.ndarray | None = None,
    eps_m: float = 100.0,
) -> dict[str, dict[str, float]]:
    """Compute range MAE, RMSE, and MAPE stratified by SOC quartile bands (PRD §17.3)."""
    soc = np.asarray(soc_true, dtype=np.float64)
    r_hat = np.asarray(range_hat_m, dtype=np.float64)
    r_true = np.asarray(range_true_m, dtype=np.float64)
    valid = (
        np.ones_like(soc, dtype=bool)
        if range_mask is None
        else np.asarray(range_mask, dtype=bool)
    )

    out: dict[str, dict[str, float]] = {}
    for band_name, low_ex, high_inc in SOC_BANDS:
        if band_name == "25-0%":
            in_band = (soc >= 0.0) & (soc <= high_inc) & valid
        else:
            in_band = (soc > low_ex) & (soc <= high_inc) & valid

        count = int(np.sum(in_band))
        if count == 0:
            out[band_name] = {
                "count": 0.0,
                "range_mae_km": float("nan"),
                "range_rmse_km": float("nan"),
                "range_mape_pct": float("nan"),
            }
            continue

        err_m = r_hat[in_band] - r_true[in_band]
        ref_m = r_true[in_band]
        out[band_name] = {
            "count": float(count),
            "range_mae_km": float(np.mean(np.abs(err_m)) / 1000.0),
            "range_rmse_km": float(np.sqrt(np.mean(err_m ** 2)) / 1000.0),
            "range_mape_pct": float(
                np.mean(np.abs(err_m) / np.maximum(np.abs(ref_m), eps_m)) * 100.0
            ),
        }
    return out


def aggregate_seed_metrics(
    seed_metrics: list[dict[str, float]],
) -> dict[str, dict[str, Any]]:
    """Compute mean, std, and formatted `mean ± std` across seeds (PRD §16.10, §17.4)."""
    if not seed_metrics:
        raise ValueError("Cannot aggregate an empty list of seed metrics.")
    keys = list(seed_metrics[0].keys())
    aggregated: dict[str, dict[str, Any]] = {}
    for k in keys:
        vals = np.asarray([float(m[k]) for m in seed_metrics], dtype=np.float64)
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals, ddof=0))
        precision = 6 if "residual" in k else 2
        aggregated[k] = {
            "mean": mean_v,
            "std": std_v,
            "n_seeds": len(vals),
            "formatted": f"{mean_v:.{precision}f} ± {std_v:.{precision}f}",
        }
    return aggregated


def paired_seed_comparison(
    metrics_a_by_seed: dict[int, dict[str, float]],
    metrics_b_by_seed: dict[int, dict[str, float]],
    metric_key: str = "range_mae_km",
) -> dict[str, Any]:
    """Compute paired per-seed difference `MAE_A - MAE_B` (PRD §16.10)."""
    common_seeds = sorted(set(metrics_a_by_seed.keys()) & set(metrics_b_by_seed.keys()))
    if not common_seeds:
        raise ValueError("No common seeds between Model A and Model B runs.")

    per_seed = {
        int(s): float(metrics_a_by_seed[s][metric_key] - metrics_b_by_seed[s][metric_key])
        for s in common_seeds
    }
    diffs = np.asarray(list(per_seed.values()), dtype=np.float64)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=0))
    return {
        "metric": metric_key,
        "n_seeds": len(common_seeds),
        "per_seed_diff": per_seed,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "formatted": f"{mean_diff:.3f} ± {std_diff:.3f}",
    }


def format_results_table(
    aggregated_by_model: dict[str, dict[str, dict[str, Any]]],
) -> str:
    """Format the PRD §17.4 Markdown comparison table (`mean ± std` over seeds)."""
    header = (
        "| Model | Range MAE (km) | Range RMSE (km) | Range MAPE (%) | "
        "SOC MAE (pp) | Power MAE (kW) | Residual RMS | Bound viol (%) | Mono viol (%) |"
    )
    sep = (
        "|---|---|---|---|---|---|---|---|---|"
    )
    rows = [header, sep]

    cols = (
        "range_mae_km",
        "range_rmse_km",
        "range_mape_pct",
        "soc_mae_pp",
        "power_mae_kw",
        "residual_rms",
        "soc_bound_viol_pct",
        "soc_mono_viol_pct",
    )

    for model_name, agg in aggregated_by_model.items():
        cells = []
        for col in cols:
            if col in agg:
                entry = agg[col]
                if isinstance(entry, dict) and "formatted" in entry:
                    cells.append(str(entry["formatted"]))
                elif isinstance(entry, dict) and "mean" in entry:
                    prec = 6 if "residual" in col else 2
                    cells.append(f"{entry['mean']:.{prec}f} ± {entry.get('std', 0.0):.{prec}f}")
                else:
                    cells.append(str(entry))
            else:
                cells.append("—")
        rows.append(f"| {model_name} | " + " | ".join(cells) + " |")

    return "\n".join(rows)
