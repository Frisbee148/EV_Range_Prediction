# Four-vehicle selection (PRD §8.3)

Selection finalised 2026-03-22 after EPA Test Car List verification (`data/raw/epa/`).

## Final vehicles

| ID | Vehicle | PRD role |
|----|---------|----------|
| `bmw_i3_60ah` | BMW i3 60 Ah BEV (2016 EPA) | V1 — light / small battery; Tier A (TUM) |
| `ford_focus_ev_2016` | Ford Focus Electric (2016 EPA) | V2 — compact / medium energy |
| `tesla_model_x_75d_2016` | Tesla Model X 75D (2016 EPA) | V3 — heavy SUV / high drag |
| `chevy_bolt_ev_2023` | Chevrolet Bolt EV (2023 EPA) | V4 — large usable pack |

Full parameter tables: `docs/finalized_vehicle_models_report.md`.  
Machine-readable config: `configs/vehicles.yaml`.

## Candidates considered

| Candidate | Outcome |
|-----------|---------|
| VW e-Golf | Lowest Cd·A; did not fill a distinct PRD slot |
| Nissan Leaf / Leaf Plus | Strong D3; pack size too close to Bolt for V4 |
| Kia Soul EV | D3 match; mass too low for V3 “heavy SUV” |
| Mercedes B-Class EV | Less mass/drag spread than Model X |

## Tier alignment

| Tier | Source | Vehicles |
|------|--------|----------|
| A | TUM BMW i3 | `bmw_i3_60ah` only |
| B | Argonne D3 | i3, Focus, Bolt (`d3_vehicle_id` in yaml) |
| C | SUMO + MoST | All four |

## Parameter tiers summary

- **SOURCED:** ETW, EPA target A/B/C, usable pack sizes (manufacturer / EPA metadata).
- **DERIVED:** C_rr, Cd·A, frontal area from Cd·A ÷ Cd.
- **ASSUMED:** Cd split, η_dt, η_rg, P_regen_max, P_aux, λ_rot (PRD §8.2 ranges; E9 sensitivity).
