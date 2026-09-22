"""Energy bookkeeping and regen clamp (PRD §6.2, §22.6)."""

from pathlib import Path

import numpy as np

from src.physics.power_model import auxiliary_power, battery_power, wheel_power
from src.physics.vehicle_dynamics import traction_forces
from src.vehicles import load_all_vehicles, load_physics_constants, load_vehicle

ROOT = Path(__file__).resolve().parents[1]
VEHICLES = ROOT / "configs" / "vehicles.yaml"


def test_all_four_vehicles_load():
    vs = load_all_vehicles(VEHICLES)
    assert set(vs) == {
        "bmw_i3_60ah",
        "ford_focus_ev_2016",
        "tesla_model_x_75d_2016",
        "chevy_bolt_ev_2023",
    }


def test_regen_is_clamped():
    vehicle = load_vehicle(VEHICLES, "bmw_i3_60ah")
    constants = load_physics_constants(VEHICLES)
    n = 50
    v = np.full(n, 20.0)
    a = np.full(n, -8.0)
    theta = np.zeros(n)
    t_amb = np.full(n, 21.0)
    forces = traction_forces(v, a, theta, vehicle, constants)
    p_aux = auxiliary_power(t_amb, vehicle, constants)
    p_wheel = wheel_power(forces["f_trac_n"], v)
    assert np.any(p_wheel < 0)
    p_batt = battery_power(p_wheel, v, p_aux, vehicle, constants)
    assert float(np.min(p_batt)) >= -(vehicle.p_regen_max_w + 1e-6)
