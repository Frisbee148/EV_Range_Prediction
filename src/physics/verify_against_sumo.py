"""SUMO battery-device cross-check (PRD §9.4).

Not runnable until MoST FCD + battery-output exist. Compare `p_batt_w` from
`simulate_trip` to SUMO `device.battery` energy deltas; require agreement
within 2%.
"""

from __future__ import annotations


def compare_to_sumo_battery(*_args, **_kwargs) -> None:
    raise NotImplementedError(
        "SUMO battery-device verification waits on Tier-C MoST runs (PRD §9.4)."
    )
