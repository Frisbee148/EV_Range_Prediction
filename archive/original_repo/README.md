# Archive — original reference repository

Everything under this directory comes from the cloned reference repo
(`Frisbee148/EV-Range-prediction-using-PINNS-ref`, itself derived from
`kamaladhi/EV-Range-prediction-...` / Jeevakamal K R). License: MIT.

## Why it is archived

Per PRD §19 and the data plan:

- CSVs under `Simulation/output/` are **one vehicle, one flat network, one
  scenario** — they cannot support leave-one-vehicle-out (E5) or multi-config
  ablation (E2).
- Vehicle mass / Cd·A in various scripts disagree with EPA coastdown values;
  mass-dependent labels inherit that error.
- `map_with_tls.net.xml` has **no z-coordinates** (grade term is dead).
- Edge/ONNX/quantize demos are out of scope (PRD §4.2).

## What was promoted to the active tree

| Archived path | Active path |
|---|---|
| `Simulation/config/map_with_tls.net.xml` | `simulation/networks/` (scaffold only) |
| `Simulation/config/*.rou.xml`, `*.trips.xml` | `simulation/routes/` |
| `Simulation/config/ev_types.add.xml` | `simulation/vehicles/` |
| `Simulation/config/simulation.sumocfg` | `simulation/scenarios/reference_scaffold.sumocfg` |

Logic still to extract later (do not re-run as-is):

- `Simulation/data_preprocess.ipynb` → `src/data/preprocessing.py`
- `Scripts/pinns_model_new.ipynb` → `src/models/`
- `Scripts/validate_pinn_with_sumo.ipynb` → `src/evaluation/`

## Do not use as training data

- `Simulation/output/ev_sumo_dataset*.csv`
- `Simulation/filtered_sumo_data.csv`
- `Scripts/15_07_model.pth`, `training_history.pkl`, plot folders
