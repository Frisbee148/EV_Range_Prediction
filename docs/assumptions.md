# Assumptions log

Every `[ASSUMED]` parameter from `configs/vehicles.yaml` must appear here with
literature range and justification (PRD §8.2 / deliverable §24 #1).

PRD plausible ranges:

| Parameter | Range | Midpoint |
|-----------|-------|----------|
| C_rr (passenger EV) | 0.008 – 0.012 | 0.010 (EPA-derived values used where available) |
| η_dt | 0.85 – 0.93 | 0.90 |
| η_rg | 0.60 – 0.70 | 0.65 |
| P_aux | 0.3 – 1.5 kW | 0.7 kW |
| λ_rot | 1.03 – 1.05 | 1.04 |

## BMW i3 60 Ah (`bmw_i3_60ah`)

| Parameter | Value | Tier | Notes |
|---|---|---|---|
| ETW mass | 1417.5 kg | SOURCED | EPA 2016 Test Car List |
| Cd·A | 0.5909 m² | DERIVED | from EPA target-C |
| C_rr | 0.00733 | DERIVED | from EPA target-A; B absorbed |
| Usable battery | 18.8 kWh | SOURCED | literature |
| Cd | 0.29 | ASSUMED | split for SUMO |
| η_dt | 0.90 | ASSUMED | PRD midpoint |
| η_rg | 0.65 | ASSUMED | PRD midpoint |
| P_regen_max | 50 kW | ASSUMED | |
| P_aux | 0.5 kW | ASSUMED | calibrate from TUM HVAC |
| λ_rot | 1.04 | ASSUMED | PRD default |

## Ford Focus Electric (`ford_focus_ev_2016`)

| Parameter | Value | Tier | Notes |
|---|---|---|---|
| ETW mass | 1757.7 kg | SOURCED | EPA 2016 |
| Cd·A | 0.689 m² | DERIVED | target-C |
| C_rr | 0.00595 | DERIVED | target-A |
| Usable battery | 23.0 kWh | SOURCED | Ford pack spec |
| Cd | 0.295 | ASSUMED | published Focus EV |
| η_dt / η_rg | 0.90 / 0.65 | ASSUMED | PRD midpoints |
| P_regen_max | 80 kW | ASSUMED | below motor limit |
| P_aux | 0.7 kW | ASSUMED | PRD midpoint |
| λ_rot | 1.04 | ASSUMED | |

## Tesla Model X 75D (`tesla_model_x_75d_2016`)

| Parameter | Value | Tier | Notes |
|---|---|---|---|
| ETW mass | 2494.8 kg | SOURCED | EPA 2016 |
| Cd·A | 0.778 m² | DERIVED | target-C |
| C_rr | 0.00685 | DERIVED | target-A |
| Usable battery | 72.5 kWh | SOURCED | 75 kWh class literature |
| Cd | 0.24 | ASSUMED | SUV literature |
| η_dt | 0.92 | ASSUMED | upper PRD range |
| η_rg | 0.65 | ASSUMED | PRD midpoint |
| P_regen_max | 120 kW | ASSUMED | large SUV |
| P_aux | 0.9 kW | ASSUMED | larger HVAC |
| λ_rot | 1.04 | ASSUMED | |

## Chevrolet Bolt EV (`chevy_bolt_ev_2023`)

| Parameter | Value | Tier | Notes |
|---|---|---|---|
| ETW mass | 1757.7 kg | SOURCED | EPA 2023 |
| Cd·A | 0.782 m² | DERIVED | target-C |
| C_rr | 0.00787 | DERIVED | target-A |
| Usable battery | 66.0 kWh | SOURCED | EPA-era usable |
| Cd | 0.308 | ASSUMED | published Bolt |
| η_dt / η_rg | 0.90 / 0.65 | ASSUMED | PRD midpoints |
| P_regen_max | 100 kW | ASSUMED | |
| P_aux | 0.7 kW | ASSUMED | PRD midpoint |
| λ_rot | 1.04 | ASSUMED | |
