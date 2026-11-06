"""Deterministic seeding and device selection (PRD §20, §21)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int, configure_threads: bool = True) -> dict[str, int]:
    """Seed Python, NumPy, PyTorch, and CUDA RNGs deterministically (PRD §20)."""
    s = int(seed)
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)

    if configure_threads:
        cpu_count = os.cpu_count() or 4
        physical_est = max(1, cpu_count // 2)
        try:
            torch.set_num_threads(physical_est)
        except RuntimeError:
            pass

    return {
        "python": s,
        "numpy": s,
        "torch": s,
        "cuda": s if torch.cuda.is_available() else -1,
    }


def resolve_device(device_spec: str = "auto") -> torch.device:
    """Resolve `'auto'` to `'cuda'` when available, otherwise `'cpu'` (PRD §21)."""
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_spec)
