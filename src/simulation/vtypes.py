"""Generate SUMO vType additional file from vehicles.yaml (IDM, PRD §9.3)."""

from __future__ import annotations

from pathlib import Path

from src.vehicles import VehicleParams


def vtype_xml(vehicles: dict[str, VehicleParams]) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<additional>",
    ]
    for vid, v in vehicles.items():
        lines.append(
            f'  <vType id="{vid}" vClass="passenger" carFollowModel="IDM" '
            f'accel="{v.accel_mps2}" decel="{v.decel_mps2}" sigma="{v.sigma}" '
            f'maxSpeed="{v.max_speed_mps}" length="4.5" mass="{v.mass_kg}">'
        )
        lines.append(f'    <param key="airDragCoefficient" value="{v.drag_coefficient}"/>')
        lines.append(f'    <param key="frontSurfaceArea" value="{v.frontal_area_m2}"/>')
        lines.append(f'    <param key="rollDragCoefficient" value="{v.rolling_resistance}"/>')
        lines.append(f'    <param key="propulsionEfficiency" value="{v.eta_dt}"/>')
        lines.append(f'    <param key="recuperationEfficiency" value="{v.eta_rg}"/>')
        lines.append(f'    <param key="constantPowerIntake" value="{v.p_aux_base_w}"/>')
        lines.append("  </vType>")
    lines.append("</additional>")
    return "\n".join(lines) + "\n"


def write_vtypes(path: Path, vehicles: dict[str, VehicleParams]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(vtype_xml(vehicles), encoding="utf-8")
