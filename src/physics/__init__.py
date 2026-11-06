"""Physics engine E1–E14. Parameters come from configs, not literals (PRD §6–§8)."""

from src.physics.battery_model import integrate_soc, remaining_energy
from src.physics.power_model import auxiliary_power, battery_power, wheel_power
from src.physics.range_model import range_definition_a, range_definition_b
from src.physics.trip import kinematics_from_speed_elevation, simulate_trip
from src.physics.vehicle_dynamics import traction_forces
from src.physics.verify_against_sumo import (
    compare_to_sumo_battery,
    parse_sumo_battery_xml,
    sumo_battery_device_power,
)

__all__ = [
    "auxiliary_power",
    "battery_power",
    "compare_to_sumo_battery",
    "integrate_soc",
    "kinematics_from_speed_elevation",
    "parse_sumo_battery_xml",
    "range_definition_a",
    "range_definition_b",
    "remaining_energy",
    "simulate_trip",
    "sumo_battery_device_power",
    "traction_forces",
    "wheel_power",
]
