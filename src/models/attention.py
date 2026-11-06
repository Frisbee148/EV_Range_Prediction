"""Causal temporal self-attention for sequence-to-sequence EV trajectory modelling (PRD §12.1)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TemporalSelfAttention(nn.Module):
    """Causal scaled dot-product self-attention preserving `[B, T_seq, H]` for per-timestep heads."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply causal temporal self-attention over `x` of shape `[B, T, H]`.

        Returns `(attended_x, attn_weights)` with shapes `([B, T, H], [B, T, T])`.
        """
        _, t_len, h_dim = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(h_dim))
        causal_mask = torch.triu(
            torch.ones((t_len, t_len), dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights_drop = self.dropout(weights)

        context = torch.matmul(weights_drop, v)
        out = self.norm(x + self.out_proj(context))
        return out, weights
