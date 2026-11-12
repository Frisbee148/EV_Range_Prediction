"""Unified trainer entry point for Models A, B, and C with matched-budget enforcement (PRD §15, §18, §20)."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import (
    PreparedSplits,
    build_dataloaders,
    load_trip_splits,
    prepare_datasets,
)
from src.losses.physics_loss import PhysicsInformedLoss, consistency_residual_torch
from src.models.lstm import EVTrajectoryLSTM
from src.models.pinn import MODEL_SPECS, build_model
from src.training.seed import resolve_device, set_all_seeds


@dataclass
class TrainerConfig:
    """Hyperparameters and run settings matching `configs/training.yaml` and PRD §20."""

    model_id: str = "B"
    T_seq: int = 60
    stride: int = 5
    batch_size: int = 256
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    use_attention: bool = False
    learning_rate: float = 1e-3
    epochs: int = 100
    warmup_epochs: int = 10
    lambda_max: float = 1.0
    patience: int = 15
    use_range_consistency: bool = False
    c_r: float = 0.1
    seed: int = 0
    ablation_fraction: float = 1.0
    soc_k: int = 1
    noise_enabled: bool = False
    noise_seed: int = 42
    device: str = "auto"
    grad_clip_norm: float = 1.0

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> TrainerConfig:
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in raw.items() if k in known}
        if "seeds" in raw and "seed" not in overrides and "seed" not in filtered:
            seeds = raw["seeds"]
            if isinstance(seeds, list) and seeds:
                filtered["seed"] = int(seeds[0])
        filtered.update(overrides)
        return cls(**filtered)


@dataclass
class TrainResult:
    """Container returned by `Trainer.fit()` with history, test metrics, and saved artifact paths."""

    run_id: str
    model_id: str
    seed: int
    epochs_run: int
    best_epoch: int
    history: list[dict[str, float]]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    checkpoint_path: Path | None
    scaler_path: Path | None
    manifest_path: Path | None
    wall_clock_s: float


def _sha256_file(path: str | Path | None) -> str:
    if path is None:
        return "in_memory"
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def evaluate_predictions_dict(
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
    """Compute PRD §17 accuracy and physics-consistency metrics on de-normalised arrays."""
    r_mask = range_mask.astype(bool)
    if np.any(r_mask):
        r_err_m = range_hat_m[r_mask] - range_true_m[r_mask]
        r_true_valid = range_true_m[r_mask]
    else:
        r_err_m = (range_hat_m - range_true_m).ravel()
        r_true_valid = range_true_m.ravel()

    range_mae_km = float(np.mean(np.abs(r_err_m)) / 1000.0)
    range_rmse_km = float(np.sqrt(np.mean(r_err_m ** 2)) / 1000.0)
    range_mape_pct = float(
        np.mean(np.abs(r_err_m) / np.maximum(np.abs(r_true_valid), eps_m)) * 100.0
    )

    soc_err_pp = (soc_hat - soc_true) * 100.0
    soc_mae_pp = float(np.mean(np.abs(soc_err_pp)))
    soc_rmse_pp = float(np.sqrt(np.mean(soc_err_pp ** 2)))

    pow_err_kw = (p_hat_w - p_true_w) / 1000.0
    power_mae_kw = float(np.mean(np.abs(pow_err_kw)))
    power_rmse_kw = float(np.sqrt(np.mean(pow_err_kw ** 2)))

    # (L2) Physics consistency residual on de-normalised outputs
    dt_col = dt_s[:, None] if dt_s.ndim == 1 else dt_s
    eu_col = e_usable_j[:, None] if e_usable_j.ndim == 1 else e_usable_j
    residual = soc_hat[:, 1:] - (soc_hat[:, :-1] - p_hat_w[:, :-1] * dt_col / eu_col)
    residual_rms = float(np.sqrt(np.mean(residual ** 2)))

    soc_bound_viol_pct = float(np.mean((soc_hat < 0.0) | (soc_hat > 1.0)) * 100.0)
    discharging = p_hat_w[:, :-1] > 0.0
    mono_viol = (soc_hat[:, 1:] > soc_hat[:, :-1]) & discharging
    soc_mono_viol_pct = float(np.mean(mono_viol) * 100.0)
    range_neg_viol_pct = float(np.mean(range_hat_m < 0.0) * 100.0)

    return {
        "range_mae_km": range_mae_km,
        "range_rmse_km": range_rmse_km,
        "range_mape_pct": range_mape_pct,
        "soc_mae_pp": soc_mae_pp,
        "soc_rmse_pp": soc_rmse_pp,
        "power_mae_kw": power_mae_kw,
        "power_rmse_kw": power_rmse_kw,
        "residual_rms": residual_rms,
        "soc_bound_viol_pct": soc_bound_viol_pct,
        "soc_mono_viol_pct": soc_mono_viol_pct,
        "range_neg_viol_pct": range_neg_viol_pct,
    }


class Trainer:
    """Single unified trainer for Models A, B, and C (PRD §15, §18)."""

    def __init__(
        self,
        config: TrainerConfig,
        prepared: PreparedSplits,
        root_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.prepared = prepared
        self.root_dir = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[2]
        self.seed_info = set_all_seeds(config.seed)
        self.device = resolve_device(config.device)

        model_key = config.model_id.upper()
        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unknown model_id={config.model_id!r}")
        spec = MODEL_SPECS[model_key]

        use_attn = True if model_key == "C" else bool(config.use_attention)
        self.effective_lambda_max = float(config.lambda_max) if spec.use_physics else 0.0

        self.model: EVTrajectoryLSTM = build_model(
            model_id=model_key,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_attention_override=use_attn,
        ).to(self.device)
        self.model.set_target_scaler_stats(prepared.scalers.target_stats())

        self.criterion = PhysicsInformedLoss(
            w_soc=prepared.scalers.w_soc,
            w_pow=prepared.scalers.w_pow,
            w_rng=prepared.scalers.w_rng,
            lambda_max=self.effective_lambda_max,
            warmup_epochs=config.warmup_epochs,
            use_range_consistency=config.use_range_consistency,
            c_r=config.c_r,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=max(2, config.patience // 3)
        )
        self.loaders = build_dataloaders(
            prepared, batch_size=config.batch_size, seed=config.seed
        )

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _run_epoch(self, loader: DataLoader, epoch: int, train: bool) -> dict[str, float]:
        self.model.train(mode=train)
        totals: dict[str, float] = {
            "loss_total": 0.0,
            "loss_data": 0.0,
            "loss_soc": 0.0,
            "loss_pow": 0.0,
            "loss_rng": 0.0,
            "loss_consist": 0.0,
            "loss_bounds": 0.0,
            "loss_physics": 0.0,
            "residual_rms": 0.0,
            "lambda_t": 0.0,
        }
        n_batches = 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for raw_batch in loader:
                batch = self._move_batch(raw_batch)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                preds = self.model(batch["x_dyn"], batch["x_stat"])
                losses = self.criterion(preds, batch, epoch=epoch)

                if train:
                    losses["loss_total"].backward()
                    if self.config.grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.config.grad_clip_norm
                        )
                    self.optimizer.step()

                for k in totals:
                    totals[k] += float(losses[k].detach().item())
                n_batches += 1

        denom = max(1, n_batches)
        return {k: v / denom for k, v in totals.items()}

    @torch.no_grad()
    def evaluate_split(self, split_name: str = "test") -> dict[str, float]:
        """Run inference over `split_name` ('train', 'val', or 'test') and compute PRD §17 metrics."""
        self.model.eval()
        loader = self.loaders[split_name]

        soc_hats, p_hats, r_hats = [], [], []
        soc_trues, p_trues, r_trues = [], [], []
        r_masks, dts, e_usables = [], [], []

        for raw_batch in loader:
            batch = self._move_batch(raw_batch)
            preds = self.model(batch["x_dyn"], batch["x_stat"])
            soc_hats.append(preds["soc_hat"].cpu().numpy())
            p_hats.append(preds["p_hat_w"].cpu().numpy())
            r_hats.append(preds["range_hat_m"].cpu().numpy())

            soc_trues.append(batch["soc"].cpu().numpy())
            p_trues.append(batch["p_batt_w"].cpu().numpy())
            r_trues.append(batch["range_b_m"].cpu().numpy())
            r_masks.append(batch["range_mask"].cpu().numpy())
            dts.append(batch["dt_s"].cpu().numpy())
            e_usables.append(batch["e_usable_j"].cpu().numpy())

        return evaluate_predictions_dict(
            soc_hat=np.concatenate(soc_hats, axis=0),
            p_hat_w=np.concatenate(p_hats, axis=0),
            range_hat_m=np.concatenate(r_hats, axis=0),
            soc_true=np.concatenate(soc_trues, axis=0),
            p_true_w=np.concatenate(p_trues, axis=0),
            range_true_m=np.concatenate(r_trues, axis=0),
            range_mask=np.concatenate(r_masks, axis=0),
            dt_s=np.concatenate(dts, axis=0),
            e_usable_j=np.concatenate(e_usables, axis=0),
        )

    def fit(
        self,
        run_id: str | None = None,
        save_artifacts: bool = True,
        vehicle_config_path: str | Path | None = None,
        simulation_config_path: str | Path | None = None,
        split_file_path: str | Path | None = None,
    ) -> TrainResult:
        """Train for up to `config.epochs` with early stopping, checkpointing, and manifest emission."""
        start_t = time.perf_counter()
        if run_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"model_{self.config.model_id.upper()}_seed{self.config.seed}_{ts}"

        best_val_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []
        # For physics-enabled models (effective_lambda_max > 0), only start tracking the
        # best checkpoint once lambda(epoch) > 0 (i.e. epoch >= 1 when warmup_epochs > 0)
        # so Model B cannot silently select its epoch-0 (lambda=0) weights identical to Model A.
        min_ckpt_epoch = (
            1 if (self.effective_lambda_max > 0.0 and self.config.warmup_epochs > 0 and self.config.epochs > 1) else 0
        )

        for epoch in range(self.config.epochs):
            tr_stats = self._run_epoch(self.loaders["train"], epoch=epoch, train=True)
            va_stats = self._run_epoch(self.loaders["val"], epoch=epoch, train=False)

            val_criterion = va_stats["loss_data"]
            self.scheduler.step(val_criterion)

            row = {
                "epoch": float(epoch),
                **{f"train_{k}": v for k, v in tr_stats.items()},
                **{f"val_{k}": v for k, v in va_stats.items()},
            }
            history.append(row)

            if epoch < min_ckpt_epoch:
                continue

            if val_criterion < best_val_loss - 1e-6:
                best_val_loss = val_criterion
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        val_metrics = self.evaluate_split("val")
        test_metrics = self.evaluate_split("test")
        wall_clock_s = time.perf_counter() - start_t

        ckpt_path: Path | None = None
        scaler_path: Path | None = None
        manifest_path: Path | None = None

        if save_artifacts:
            ckpt_dir = self.root_dir / "models" / "checkpoints"
            scaler_dir = self.root_dir / "models" / "scalers"
            exp_dir = self.root_dir / "experiments" / run_id
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            scaler_dir.mkdir(parents=True, exist_ok=True)
            exp_dir.mkdir(parents=True, exist_ok=True)

            ckpt_path = ckpt_dir / f"{run_id}.pt"
            scaler_path = scaler_dir / f"{run_id}_scalers.pkl"
            manifest_path = exp_dir / "manifest.yaml"

            torch.save(
                {
                    "run_id": run_id,
                    "model_id": self.config.model_id.upper(),
                    "config": asdict(self.config),
                    "best_epoch": best_epoch,
                    "state_dict": self.model.state_dict(),
                    "target_stats": self.prepared.scalers.target_stats(),
                    "head_weights": self.prepared.scalers.head_weights_dict(),
                },
                ckpt_path,
            )
            self.prepared.scalers.save(scaler_path)

            manifest = {
                "run_id": run_id,
                "git_commit": _git_commit(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "seeds": self.seed_info,
                "hashes": {
                    "vehicle_config": _sha256_file(
                        vehicle_config_path or (self.root_dir / "configs" / "vehicles.yaml")
                    ),
                    "simulation_config": _sha256_file(
                        simulation_config_path or (self.root_dir / "configs" / "simulation.yaml")
                    ),
                    "split_file": _sha256_file(
                        split_file_path or (self.root_dir / "data" / "splits" / "trip_splits.json")
                    ),
                },
                "degradation": {
                    "ablation_fraction": self.config.ablation_fraction,
                    "soc_label_interval_k": self.config.soc_k,
                    "noise_enabled": self.config.noise_enabled,
                    "noise_seed": self.config.noise_seed,
                },
                "hyperparameters": {
                    "model_id": self.config.model_id.upper(),
                    "T_seq": self.config.T_seq,
                    "stride": self.config.stride,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "optimiser": "Adam",
                    "scheduler": "ReduceLROnPlateau",
                    "epochs_configured": self.config.epochs,
                    "epochs_run": len(history),
                    "early_stopping_epoch": best_epoch,
                    "patience": self.config.patience,
                    "hidden_dim": self.config.hidden_dim,
                    "num_layers": self.config.num_layers,
                    "dropout": self.config.dropout,
                    "use_attention": self.model.use_attention,
                },
                "physics_loss": {
                    "lambda_max": self.effective_lambda_max,
                    "warmup_epochs": self.config.warmup_epochs,
                    "components_enabled": (
                        ["L2_consist", "L3_bounds"]
                        + (["L4_range_c"] if self.config.use_range_consistency else [])
                        if self.effective_lambda_max > 0.0
                        else []
                    ),
                    "c_r": self.config.c_r,
                    "head_weights_L0": self.prepared.scalers.head_weights_dict(),
                },
                "artifacts": {
                    "checkpoint_path": str(ckpt_path.relative_to(self.root_dir)),
                    "scaler_path": str(scaler_path.relative_to(self.root_dir)),
                },
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "wall_clock_s": float(wall_clock_s),
                "device": str(self.device),
            }
            with manifest_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(manifest, fh, sort_keys=False)

        return TrainResult(
            run_id=run_id,
            model_id=self.config.model_id.upper(),
            seed=self.config.seed,
            epochs_run=len(history),
            best_epoch=best_epoch,
            history=history,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            checkpoint_path=ckpt_path,
            scaler_path=scaler_path,
            manifest_path=manifest_path,
            wall_clock_s=wall_clock_s,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EV Range LSTM / PINN (Models A, B, C).")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--dataset", type=str, default="data/processed/sumo_most_1hz.csv.gz")
    parser.add_argument("--splits", type=str, default="data/splits/trip_splits.json")
    parser.add_argument("--model", type=str, default="B", choices=["A", "B", "C"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lambda-max", type=float, default=None)
    parser.add_argument("--ablation-fraction", type=float, default=1.0)
    parser.add_argument("--soc-k", type=int, default=1)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    overrides: dict[str, Any] = {
        "model_id": args.model,
        "ablation_fraction": args.ablation_fraction,
        "soc_k": args.soc_k,
    }
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.warmup_epochs is not None:
        overrides["warmup_epochs"] = args.warmup_epochs
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.lambda_max is not None:
        overrides["lambda_max"] = args.lambda_max

    cfg = TrainerConfig.from_yaml(args.config, **overrides)
    df = pd.read_csv(args.dataset)
    splits = load_trip_splits(args.splits)
    prepared = prepare_datasets(
        df,
        splits,
        t_seq=cfg.T_seq,
        stride=cfg.stride,
        ablation_fraction=cfg.ablation_fraction,
        soc_k=cfg.soc_k,
        noise_enabled=cfg.noise_enabled,
        noise_seed=cfg.noise_seed,
        ablation_seed=cfg.seed,
    )
    trainer = Trainer(cfg, prepared)
    res = trainer.fit(run_id=args.run_id, save_artifacts=True)
    print(
        f"[{res.run_id}] model={res.model_id} epochs={res.epochs_run} "
        f"best_epoch={res.best_epoch} wall={res.wall_clock_s:.1f}s "
        f"Range MAE={res.test_metrics['range_mae_km']:.2f} km "
        f"SOC MAE={res.test_metrics['soc_mae_pp']:.2f} pp "
        f"Power MAE={res.test_metrics['power_mae_kw']:.2f} kW "
        f"Residual RMS={res.test_metrics['residual_rms']:.6f}"
    )


if __name__ == "__main__":
    main()
