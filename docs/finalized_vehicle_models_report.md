# Finalized four-vehicle models — EV Range PINN

**Project:** Physics-informed LSTM for EV range / SOC / power (PRD §8).  
**Config file:** `configs/vehicles.yaml`  
**EPA derivations:** `data/raw/epa/*_derived_params.json`  
**Date:** 2026-03-22

## Executive summary

Four production vehicle configurations span mass, aerodynamic load, and usable battery energy for SUMO Tier-C training. BMW i3 anchors Tier-A (TUM telemetry). Ford Focus EV and Chevrolet Bolt EV align with Argonne D3 for Tier-B validation. Tesla Model X 75D provides a heavy SUV reference without a D3 match.

| Config ID | PRD slot | ETW (kg) | Usable (kWh) | Cd·A (m²) | C_rr | D3 ID |
|-----------|----------|----------|--------------|-----------|------|-------|
| `bmw_i3_60ah` | V1 small / light | 1417.5 | 18.8 | 0.591 | 0.00733 | `bmw_i3_2014` + TUM |
| `ford_focus_ev_2016` | V2 compact | 1757.7 | 23.0 | 0.689 | 0.00595 | `ford_focus_ev_2013` |
| `tesla_model_x_75d_2016` | V3 SUV / high drag | 2494.8 | 72.5 | 0.778 | 0.00685 | — |
| `chevy_bolt_ev_2023` | V4 large pack | 1757.7 | 66.0 | 0.782 | 0.00787 | `chevy_bolt_2020` |

---

## Road-load methodology

EPA dynamometer target:

\[
F_\mathrm{lb} = A + B\,v_\mathrm{mph} + C\,v_\mathrm{mph}^2
\]

Project mapping (`docs/data_acquisition.md`):

| Derived quantity | Formula |
|----------------|---------|
| Rolling resistance | \(C_{rr} = (A_\mathrm{lbf} \times 4.44822) / (m_\mathrm{ETW} \times 9.81)\) |
| Drag area | \(\mathrm{Cd}\cdot A = 2 C_\mathrm{SI} / \rho\), \(C_\mathrm{SI} = C_\mathrm{lbf} \times 4.44822 / (0.44704)^2\), \(\rho = 1.225\,\mathrm{kg/m^3}\) |

The **B** coefficient is not mapped to a separate physics term; it is documented as absorbed into effective losses.

**Simulation mass:** EPA **equivalent test weight (ETW)** is used as `mass_kg` so derived \(C_{rr}\) matches target **A**. Kerb + 75 kg (PRD §5.2) is noted for BMW only as `kerb_mass_kg`.

**Cd and frontal area:** EPA provides Cd·A only. \(C_d\) is **ASSUMED** from literature; `frontal_area_m2 = drag_area_m2 / drag_coefficient` (**DERIVED**).

---

## Vehicle 1 — `bmw_i3_60ah`

| Parameter | Value | Tier |
|-----------|-------|------|
| Display name | BMW i3 60 Ah (2016 BEV) | — |
| EPA source | `16tstcar.csv`, I3 BEV | SOURCED |
| ETW | 3125 lb → 1417.5 kg | SOURCED |
| Target A / B / C (lbf, lbf/mph, lbf/mph²) | 22.9 / 0.346 / 0.01626 | SOURCED |
| `rolling_resistance` | 0.00733 | DERIVED |
| `drag_area_m2` | 0.5909 | DERIVED |
| `drag_coefficient` / `frontal_area_m2` | 0.29 / 2.038 | ASSUMED / DERIVED |
| `battery_capacity_kwh` | 18.8 usable | SOURCED |
| η_dt / η_rg | 0.90 / 0.65 | ASSUMED |
| P_regen_max / P_aux / λ_rot | 50 kW / 0.5 kW / 1.04 | ASSUMED |

**FuelEconomy.gov (2016 i3 BEV):** combE 27 kWh/100 mi, EPA range 81 mi.

**SUMO:** max_speed 33.3 m/s, accel 3.5 m/s², decel 4.5 m/s², sigma 0.5.

**Validation:** TUM trips (`data/raw/tum_bmw_i3/`); D3 `bmw_i3_2014`.

---

## Vehicle 2 — `ford_focus_ev_2016`

| Parameter | Value | Tier |
|-----------|-------|------|
| Display name | Ford Focus Electric (2016) | — |
| EPA source | `16tstcar.csv`, model **FOCUS**, fuel **Electricity** | SOURCED |
| ETW | 3875 lb → 1757.67 kg | SOURCED |
| Target A / B / C | 23.05 / 0.3647 / 0.01897 | SOURCED |
| `rolling_resistance` | 0.005946 | DERIVED |
| `drag_area_m2` | 0.689374 | DERIVED |
| `drag_coefficient` / `frontal_area_m2` | 0.295 / 2.34 | ASSUMED / DERIVED |
| `battery_capacity_kwh` | 23.0 usable | SOURCED |
| η_dt / η_rg | 0.90 / 0.65 | ASSUMED |
| P_regen_max / P_aux / λ_rot | 80 kW / 0.7 kW / 1.04 | ASSUMED |

**FuelEconomy.gov (2016):** combE 32 kWh/100 mi, range 76 mi.

**SUMO:** max_speed 35.8 m/s, accel 3.2 m/s², decel 4.5 m/s², sigma 0.5.

**Validation:** D3 `ford_focus_ev_2013`.

---

## Vehicle 3 — `tesla_model_x_75d_2016`

| Parameter | Value | Tier |
|-----------|-------|------|
| Display name | Tesla Model X 75D (2016) | — |
| EPA source | `16tstcar.csv`, Model X 75D | SOURCED |
| ETW | 5500 lb → 2494.76 kg | SOURCED |
| Target A / B / C | 37.68 / 0.0486 / 0.02140 | SOURCED |
| `rolling_resistance` | 0.006849 | DERIVED |
| `drag_area_m2` | 0.777681 | DERIVED |
| `drag_coefficient` / `frontal_area_m2` | 0.24 / 3.24 | ASSUMED / DERIVED |
| `battery_capacity_kwh` | 72.5 usable | SOURCED |
| η_dt / η_rg | 0.92 / 0.65 | ASSUMED |
| P_regen_max / P_aux / λ_rot | 120 kW / 0.9 kW / 1.04 | ASSUMED |

**SUMO:** max_speed 45.0 m/s, accel 2.5 m/s², decel 4.0 m/s², sigma 0.45 (heavier, slower acceleration).

**Validation:** Simulation-only for cross-mass / cross-drag (E5 held-out vehicle).

---

## Vehicle 4 — `chevy_bolt_ev_2023`

| Parameter | Value | Tier |
|-----------|-------|------|
| Display name | Chevrolet Bolt EV (2023) | — |
| EPA source | `23-testcar.xlsx`, BOLT EV | SOURCED |
| ETW | 3875 lb → 1757.67 kg | SOURCED |
| Target A / B / C | 30.51 / 0.0152 / 0.02151 | SOURCED |
| `rolling_resistance` | 0.007871 | DERIVED |
| `drag_area_m2` | 0.781678 | DERIVED |
| `drag_coefficient` / `frontal_area_m2` | 0.308 / 2.54 | ASSUMED / DERIVED |
| `battery_capacity_kwh` | 66.0 usable | SOURCED |
| η_dt / η_rg | 0.90 / 0.65 | ASSUMED |
| P_regen_max / P_aux / λ_rot | 100 kW / 0.7 kW / 1.04 | ASSUMED |

**FuelEconomy.gov (2023 Bolt EV):** combE 28.1 kWh/100 mi, range 259 mi.

**SUMO:** max_speed 40.0 m/s, accel 3.0 m/s², decel 4.5 m/s², sigma 0.5.

**Validation:** D3 `chevy_bolt_2020`.

**Note:** Focus and Bolt share ETW but differ in EPA **A** and **C**, pack size, and SUMO dynamics (PRD §9.3 trajectory diversity).

---

## Selection rationale (PRD §8.3)

| PRD target profile | Chosen vehicle | Why |
|--------------------|----------------|-----|
| V1 low m, low E_usable | BMW i3 60 Ah | Lightest ETW, smallest pack; TUM real data |
| V2 mid m, mid E, efficient compact | Ford Focus EV | Moderate pack; D3 match |
| V3 high m, high Cd·A | Model X 75D | Highest ETW in set; SUV-class aero load |
| V4 very high E_usable | Bolt EV 2023 | 66 kWh usable; D3 match |

**Not selected:** VW e-Golf (lowest Cd·A, redundant drag slot); Nissan Leaf Plus (pack size overlaps Bolt); Kia Soul EV (insufficient mass for V3).

See `docs/vehicle_selection.md` for the short PRD checklist entry.

---

## Dataset / training implications

Static columns to join on `vehicle_id` (from config loader):  
`mass_kg`, `drag_coefficient`, `frontal_area_m2`, `rolling_resistance`, `battery_capacity_kwh`, `drivetrain_efficiency`, `regen_efficiency`, `regen_power_max_kw`, `aux_power_kw`, `rotating_mass_factor`.

Range labels (`range_b_m`, etc.) remain **DERIVED** after physics engine E1–E14 is implemented.

---

## File index

| File | Purpose |
|------|---------|
| `configs/vehicles.yaml` | Authoritative parameters + SUMO vType hints |
| `data/raw/epa/bmw_i3_derived_params.json` | BMW EPA derivation record |
| `data/raw/epa/ford_focus_ev_2016_derived_params.json` | Focus EPA derivation |
| `data/raw/epa/tesla_model_x_75d_2016_derived_params.json` | Model X EPA derivation |
| `data/raw/epa/chevy_bolt_ev_2023_derived_params.json` | Bolt EPA derivation |
| `docs/assumptions.md` | ASSUMED parameter log |
| `docs/vehicle_selection.md` | PRD selection record |
