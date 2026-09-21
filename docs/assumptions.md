# Assumptions log

Every `[ASSUMED]` parameter from `configs/vehicles.yaml` must appear here with
literature range and justification (PRD §8.2 / deliverable §24 #1).

## BMW i3 60 Ah

| Parameter | Value | Tier | Notes |
|---|---|---|---|
| ETW mass | 1417.5 kg | SOURCED | EPA 2016 Test Car List |
| Cd·A | 0.5909 m² | DERIVED | from EPA target-C |
| C_rr | 0.00733 | DERIVED | from EPA target-A; B absorbed |
| Usable battery | 18.8 kWh | SOURCED | literature / ev-database |
| η_dt | 0.90 | ASSUMED | pending literature range |
| η_rg | 0.70 | ASSUMED | pending |
| P_regen_max | 50 kW | ASSUMED | pending |
| P_aux | 0.5 kW | ASSUMED | calibrate from TUM heater/AirCon when possible |
| λ_rot | 1.04 | ASSUMED | PRD default |

## Vehicles 2–4

TBD after EPA selection.
