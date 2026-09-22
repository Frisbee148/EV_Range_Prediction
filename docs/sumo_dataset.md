# MoST multi-vehicle dataset (Tier C)

- Engine: **idm_most** (`sumo` on PATH: False)
- Rows: 144,000 at dt=1.0s; trips: 240
- Vehicles: bmw_i3_60ah, chevy_bolt_ev_2023, ford_focus_ev_2016, tesla_model_x_75d_2016
- Scenarios: arterial, highway, urban
- Elevation std: **170.88 m** (PRD gate > 1.0 m)
- Max |corr| of trip speed traces: 0.331 (gate < 0.9)
- Splits (trips): {'train': 168, 'val': 36, 'test': 36}
- Dropped net-negative energy trips: 0

Network: MoST `most.net.xml` (Codeca & Härri). Physics labels: `src/physics/`.
If SUMO is installed later, re-run this script; FCD+z replaces IDM speeds
but elevation still comes from the same net.

Outputs: `data/processed/sumo_most_1hz.csv.gz`, `data/splits/trip_splits.json`.
