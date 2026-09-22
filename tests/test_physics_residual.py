"""L2 residual on a physically consistent trajectory (PRD §12.3)."""

import numpy as np

from src.losses.physics_loss import consistency_residual
from src.vehicles import KWH_TO_J


def test_consistency_residual_vanishes_on_consistent_soc_power():
    dt = 1.0
    e_usable = 60.0 * KWH_TO_J
    n = 200
    p_hat = np.linspace(5e3, 25e3, n)
    soc = np.empty(n)
    soc[0] = 0.8
    for t in range(n - 1):
        soc[t + 1] = soc[t] - p_hat[t] * dt / e_usable
    r = consistency_residual(soc, p_hat, dt, e_usable)
    assert float(np.max(np.abs(r))) < 1e-8
