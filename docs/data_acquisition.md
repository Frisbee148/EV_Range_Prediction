# Data acquisition and layout

This project uses **four data layers**. Raw blobs live under `data/raw/` and are
gitignored when large. Provenance and licensing are mandatory in the report.

## Status (as of reorg)

| Layer | Source | Local path | Status |
|---|---|---|---|
| Vehicle params | EPA Test Car List + FE.gov | `data/raw/epa/` | **Downloaded** |
| Real telemetry (Tier A) | TUM BMW i3 (Steinstraeter et al.) | `data/raw/tum_bmw_i3/` | **Downloaded** (70 trips via public GitHub mirror of Kaggle/IEEE set) |
| Multi-vehicle lab (Tier B) | Argonne D3 | `data/raw/argonne_d3/` | Manual — see below |
| Traffic + elevation (Tier C) | MoST (preferred) / LuST | `data/raw/sumo_scenarios/` | **Downloaded** |
| Certification cycles | EPA UDDS / HWFET / US06 | `data/raw/drive_cycles/` | **Downloaded** |
| Reference SUMO CSVs | original repo | `archive/original_repo/Simulation/` | Archived — **do not train on** |

## Critical finding: reference network is flat

```text
map_with_tls.net.xml → z_count = 0  (no elevation)
MoST most.net.xml    → z in [-0.08, 616.3] m   ← use this for training
LuST lust.net.xml    → no z (also flat)
```

Use **MoST** for the SUMO training set. Keep `simulation/networks/map_with_tls.net.xml`
only as a day-one scaffold.

## Layer 1 — EPA road-load coefficients

Source: https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy

Road load: `F = A + B v + C v²` (coastdown / dynamometer targets).

```text
C_rr  = A / (m * g)          # A in N
Cd·A  = 2 * C / rho          # C in N/(m/s)²
```

`B` has no clean physical analogue — document as absorbed into rolling losses.

**BMW i3 BEV (2016 Test Car List)** — derived file
`data/raw/epa/bmw_i3_derived_params.json`:

| Quantity | Value | Tier |
|---|---|---|
| ETW | 3125 lb → 1417.5 kg | SOURCED |
| Target A / B / C | 22.9 / 0.346 / 0.01626 | SOURCED |
| C_rr | 0.00733 | DERIVED |
| Cd·A | 0.5909 m² | DERIVED |

Compare to archived SUMO vType (`mass=1320`, `Cd=0.29`, `A=2.5` → Cd·A=0.725,
C_rr=0.01): do **not** reuse those labels.

Also present: `vehicles.csv` from FuelEconomy.gov for capacity / range metadata.

## Layer 2 — TUM BMW i3 telemetry (Tier A)

- Authors: Steinstraeter, Buberger, Trifonov (TUM)
- DOI: 10.21227/6jr9-5235 (IEEE DataPort)
- Kaggle mirror: `atechnohazard/battery-and-heating-data-in-real-driving-cycles`
- Local copy: 70 trip CSVs under `data/raw/tum_bmw_i3/` (mirrored from a Kaggle
  re-host; prefer citing IEEE DOI + Kaggle in the report)

TripA01 sanity check (charge-positive current → discharge power `P = -V·I`):

- Duration ~1009 s, SOC 86.9% → 81.5%
- Live elevation (std ≈ 5.6 m)
- Integrated energy ≈ 1.29 kWh → implied pack ≈ 23.9 kWh (order-of-magnitude OK
  vs ~22–27 kWh nominal 60 Ah pack; refine sign/aux accounting in physics validation)

**Role:** held-out real test set; validate physics engine `(E7)–(E9)` against measured `I×V`.

Re-download via Kaggle (authoritative mirror):

```bash
# ~/.kaggle/kaggle.json required
./scripts/download_tum_kaggle.sh
```

## Layer 3 — Argonne D3 (Tier B) — manual

https://www.anl.gov/taps/d3-downloadable-dynamometer-database

- Multi-EV chassis-dyno tests (Leaf etc.), pack current/voltage, 0% grade
- Attribution string required by D3 terms — paste into `docs/assumptions.md`
- Place downloaded files in `data/raw/argonne_d3/`

## Layer 4 — SUMO scenarios (Tier C)

| Scenario | Path | Elevation | Notes |
|---|---|---|---|
| **MoST (Monaco)** | `data/raw/sumo_scenarios/MoST/` | **Yes** | Preferred training network |
| LuST (Luxembourg) | `data/raw/sumo_scenarios/LuST/` | No | Better traffic validation; needs DEM if used |

Cite LuST/MoST papers from their `README.md` / BibTeX.

Refresh:

```bash
./scripts/download_sumo_scenarios.sh
```

## Certification cycles

`data/raw/drive_cycles/{udds,hwfet,us06}.txt` — EPA 1 Hz traces.
Use for controlled physics checks and alignment with D3/EPA tests.

## Recommended experiment tiers

| Tier | Source | Role |
|---|---|---|
| A | TUM BMW i3 | Real held-out; sim-to-real for RQ2 |
| B | Argonne D3 | Cross-vehicle power model, no grade |
| C | SUMO + MoST + 4 configs | Main training (~150k steps) |

Make BMW i3 one of the four simulated vehicles so Tier A is an apples-to-apples
transfer test.

## First-week order of work

1. ~~Elevation check on reference net~~ → flat; switch to MoST
2. ~~Download TUM~~ → integrate `I×V` vs SOC (TripA01 done)
3. Implement `(E1)`–`(E11)` on one TUM speed/elevation trace; compare to measured power
4. Fill `configs/vehicles.yaml` from EPA for four candidates
