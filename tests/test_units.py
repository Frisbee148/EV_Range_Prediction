"""SI unit conversion (PRD §22.1)."""

from pathlib import Path

import numpy as np

from src.physics.battery_model import integrate_soc
from src.vehicles import KW_TO_W, KWH_TO_J, VehicleParams, load_vehicle

ROOT = Path(__file__).resolve().parents[1]
VEHICLES = ROOT / "configs" / "vehicles.yaml"


def _synthetic(e_kwh: float) -> VehicleParams:
    return VehicleParams(
        vehicle_id="unit_test",
        display_name="unit",
        mass_kg=1500.0,
        drag_coefficient=0.3,
        frontal_area_m2=2.0,
        rolling_resistance=0.01,
        e_usable_j=e_kwh * KWH_TO_J,
        eta_dt=0.9,
        eta_rg=0.65,
        p_regen_max_w=50.0 * KW_TO_W,
        p_aux_base_w=700.0,
        lambda_rot=1.04,
        provenance={},
    )


def test_sixty_kwh_twenty_kw_one_hour():
    vehicle = _synthetic(60.0)
    dt = 1.0
    n = 3601  # 3600 updates of 1 s (SOC[t+1] uses P[t]; PRD §22.1)
    p = np.full(n, 20.0 * KW_TO_W)
    soc = integrate_soc(p, soc_0=1.0, dt_s=dt, vehicle=vehicle)
    np.testing.assert_allclose(soc[-1], 1.0 - 1.0 / 3.0, rtol=0, atol=1e-8)


def test_config_loader_converts_kwh_once():
    v = load_vehicle(VEHICLES, "bmw_i3_60ah")
    np.testing.assert_allclose(v.e_usable_j, 18.8 * KWH_TO_J)
    np.testing.assert_allclose(v.p_aux_base_w, 0.5 * KW_TO_W)
    np.testing.assert_allclose(v.p_regen_max_w, 50.0 * KW_TO_W)
    assert v.provenance["mass_kg"]["tier"] == "SOURCED"
