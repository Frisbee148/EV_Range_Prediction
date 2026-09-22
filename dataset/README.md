# Project dataset (cleaned)

Built by `scripts/build_project_dataset.py`.

## Layout

- `processed/d3_1hz/` — Argonne D3, cleaned, ~1 Hz, per vehicle
- `processed/tum_1hz/` — TUM BMW i3 real trips, cleaned, ~1 Hz
- `processed/project_unified_1hz.csv.gz` — concatenated modelling table
- `reports/attribute_provenance.md` — **DATA vs DERIVED** attribute report
- `reports/build_summary.json` — row/trip counts

## Cleaning applied

- Dropped charge-only files
- Dropped t < 0 pre-buffers
- Dropped duplicate timestamps within a trip
- Dropped non-finite values; clipped accel and SOC
- SI units for speed (m/s), power (W), SOC (0–1)
- `p_batt_w` = V×I with discharge-positive convention

## Not included yet

- Tesla Model 3 (incomplete download)
- Static EPA vehicle params (join separately)
- Range labels (need physics engine)
