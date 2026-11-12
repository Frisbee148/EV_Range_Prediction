"""Per-timestep multi-head LSTM and baseline estimators (PRD §12.1, §15)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from src.data.sequence_builder import DYNAMIC_FEATURES, STATIC_FEATURES, SequenceWindows
from src.models.attention import TemporalSelfAttention
from src.physics.trip import simulate_trip
from src.vehicles import PhysicsConstants, VehicleParams, load_all_vehicles, load_physics_constants

# Window-only Base-0 cap: 1000 km. The largest label range in the dataset is about 806 km.
_RANGE_CAP_M = 1.0e6


class EVTrajectoryLSTM(nn.Module):
    """Multi-task sequence-to-sequence LSTM predicting SOC, battery power, and range at every timestep.

    Architecture (PRD §12.1):
    - Dynamic inputs `[B, T_seq, F_dyn]` + broadcast static inputs `[B, T_seq, F_stat]`
    - Linear + ReLU input projection
    - Multi-layer LSTM (`batch_first=True`) returning output at every timestep `[B, T_seq, H]`
    - Optional causal temporal self-attention (`use_attention=True`)
    - Per-timestep output heads for SOC, battery power, and remaining range
    - Built-in de-normalisation buffers so `soc_hat`, `p_hat_w`, `range_hat_m` are in physical SI units.
    """

    def __init__(
        self,
        f_dyn: int = len(DYNAMIC_FEATURES),
        f_stat: int = len(STATIC_FEATURES),
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.f_dyn = int(f_dyn)
        self.f_stat = int(f_stat)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)
        self.use_attention = bool(use_attention)

        in_dim = self.f_dyn + self.f_stat
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_p if self.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention: TemporalSelfAttention | None = (
            TemporalSelfAttention(self.hidden_dim, dropout=self.dropout_p)
            if self.use_attention
            else None
        )

        self.soc_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.power_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.range_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        # Target de-normalisation buffers (mean and scale from training-only StandardScaler)
        self.register_buffer("soc_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("soc_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("pow_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("pow_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("rng_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("rng_scale", torch.tensor(1.0, dtype=torch.float32))

    def set_target_scaler_stats(self, stats: dict[str, tuple[float, float]]) -> None:
        """Set de-normalisation mean and scale from `ScalerBundle.target_stats()`."""
        soc_m, soc_s = stats["soc"]
        pow_m, pow_s = stats["p_batt_w"]
        rng_m, rng_s = stats["range_b_m"]
        self.soc_mean.fill_(float(soc_m))
        self.soc_scale.fill_(float(soc_s))
        self.pow_mean.fill_(float(pow_m))
        self.pow_scale.fill_(float(pow_s))
        self.rng_mean.fill_(float(rng_m))
        self.rng_scale.fill_(float(rng_s))

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_stat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning per-timestep predictions in both normalised and SI units."""
        _, t_seq, _ = x_dyn.shape
        if x_stat.ndim == 2:
            stat_bcast = x_stat.unsqueeze(1).expand(-1, t_seq, -1)
        elif x_stat.ndim == 3:
            stat_bcast = x_stat
        else:
            raise ValueError(f"Expected x_stat with 2 or 3 dims, got shape {tuple(x_stat.shape)}")

        x = torch.cat([x_dyn, stat_bcast], dim=-1)
        h_in = self.input_proj(x)
        h_seq, _ = self.lstm(h_in)

        attn_weights: torch.Tensor | None = None
        if self.attention is not None:
            h_seq, attn_weights = self.attention(h_seq)

        soc_norm_hat = self.soc_head(h_seq).squeeze(-1)
        p_norm_hat = self.power_head(h_seq).squeeze(-1)
        range_norm_hat = self.range_head(h_seq).squeeze(-1)

        # De-normalise to physical SI units for (L1) and (L2)-(L5) (PRD §12.3)
        soc_hat = soc_norm_hat * self.soc_scale + self.soc_mean
        p_hat_w = p_norm_hat * self.pow_scale + self.pow_mean
        range_hat_m = range_norm_hat * self.rng_scale + self.rng_mean

        out: dict[str, torch.Tensor] = {
            "soc_norm_hat": soc_norm_hat,
            "p_norm_hat": p_norm_hat,
            "range_norm_hat": range_norm_hat,
            "soc_hat": soc_hat,
            "p_hat_w": p_hat_w,
            "range_hat_m": range_hat_m,
        }
        if attn_weights is not None:
            out["attn_weights"] = attn_weights
        return out


class RidgeWindowBaseline:
    """Base-1 sanity floor: Ridge regression on flattened window features (PRD §15)."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.soc_reg = Ridge(alpha=self.alpha)
        self.pow_reg = Ridge(alpha=self.alpha)
        self.rng_reg = Ridge(alpha=self.alpha)

    @staticmethod
    def _flatten(windows: SequenceWindows) -> np.ndarray:
        n, t_seq, f_dyn = windows.x_dyn.shape
        x_dyn_flat = windows.x_dyn.reshape(n, t_seq * f_dyn)
        return np.concatenate([x_dyn_flat, windows.x_stat], axis=1)

    def fit(self, train_windows: SequenceWindows) -> RidgeWindowBaseline:
        x_flat = self._flatten(train_windows)
        self.soc_reg.fit(x_flat, train_windows.soc)
        self.pow_reg.fit(x_flat, train_windows.p_batt_w)
        self.rng_reg.fit(x_flat, train_windows.range_b_m)
        return self

    def predict(self, windows: SequenceWindows) -> dict[str, np.ndarray]:
        x_flat = self._flatten(windows)
        return {
            "soc_hat": self.soc_reg.predict(x_flat).astype(np.float32),
            "p_hat_w": self.pow_reg.predict(x_flat).astype(np.float32),
            "range_hat_m": self.rng_reg.predict(x_flat).astype(np.float32),
        }


class AnalyticPhysicsBaseline:
    """Base-0 upper bound: analytic physics estimator from E1-E14 on window inputs (PRD §15)."""

    def __init__(self, vehicles_yaml: str | Path) -> None:
        self.vehicles: dict[str, VehicleParams] = load_all_vehicles(vehicles_yaml)
        self.constants: PhysicsConstants = load_physics_constants(vehicles_yaml)

    def predict(
        self,
        windows: SequenceWindows,
        trip_frames: pd.DataFrame | None = None,
    ) -> dict[str, np.ndarray]:
        """Predict SOC, power and range for every window.

        With `trip_frames` (the full-trip rows of the same split), each trip is
        simulated once from its first SOC, exactly as the labels were built.
        Windows are then sliced from that simulation, so the range output is
        Definition B (E14), the same quantity as the label. This is the PRD §15
        upper bound.

        Without `trip_frames` the 60 s window is simulated alone. Range is then
        Definition A (E13), which is a different quantity from the label and
        can be unbounded when the smoothed energy rate is near zero. It is
        capped at the largest physically possible range for the vehicle.
        """
        n, t_seq = windows.soc.shape
        soc_hat = np.empty((n, t_seq), dtype=np.float32)
        p_hat_w = np.empty((n, t_seq), dtype=np.float32)
        range_hat_m = np.empty((n, t_seq), dtype=np.float32)

        if trip_frames is not None:
            if len(windows.start_idx) != n:
                raise ValueError("windows.start_idx is required when trip_frames is given.")
            trip_sims: dict[str, dict[str, np.ndarray | bool]] = {}
            for trip_id, group in trip_frames.groupby("trip_id", sort=False):
                trip = group.sort_values("timestamp", kind="mergesort")
                ts = trip["timestamp"].to_numpy(dtype=float)
                dt_s = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0
                trip_sims[str(trip_id)] = simulate_trip(
                    v_mps=trip["speed_mps"].to_numpy(dtype=float),
                    a_mps2=trip["accel_mps2"].to_numpy(dtype=float),
                    theta_rad=trip["grade_rad"].to_numpy(dtype=float),
                    t_amb_c=trip["ambient_temp_c"].to_numpy(dtype=float),
                    soc_0=float(trip["soc"].iloc[0]),
                    dt_s=dt_s,
                    vehicle=self.vehicles[str(trip["vehicle_id"].iloc[0])],
                    constants=self.constants,
                )
            for i in range(n):
                sim = trip_sims[windows.trip_ids[i]]
                lo = int(windows.start_idx[i])
                hi = lo + t_seq
                soc_hat[i] = np.asarray(sim["soc"])[lo:hi]
                p_hat_w[i] = np.asarray(sim["p_batt_w"])[lo:hi]
                range_hat_m[i] = np.asarray(sim["range_b_m"])[lo:hi]
        else:
            for i in range(n):
                vid = windows.vehicle_ids[i]
                vehicle = self.vehicles[vid]
                sim = simulate_trip(
                    v_mps=windows.speed_mps_raw[i],
                    a_mps2=windows.accel_mps2_raw[i],
                    theta_rad=windows.grade_rad_raw[i],
                    t_amb_c=windows.ambient_temp_c_raw[i],
                    soc_0=float(windows.soc[i, 0]),
                    dt_s=float(windows.dt_s[i]),
                    vehicle=vehicle,
                    constants=self.constants,
                )
                soc_hat[i] = sim["soc"].astype(np.float32)
                p_hat_w[i] = sim["p_batt_w"].astype(np.float32)
                range_hat_m[i] = np.clip(sim["range_a_m"], 0.0, _RANGE_CAP_M).astype(np.float32)

        return {
            "soc_hat": soc_hat,
            "p_hat_w": p_hat_w,
            "range_hat_m": range_hat_m,
        }
