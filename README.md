# EV Range Prediction — Physics-Informed LSTM

Multi-vehicle EV energy / SOC / remaining-range prediction with a
physics-consistency-regularised LSTM. Spec: [`PRD.md`](PRD.md).

This tree was reorganised from a single-vehicle reference clone. Reusable SUMO
inputs were kept; everything else is under [`archive/original_repo/`](archive/original_repo/).

## Data (do not use archived CSVs)

| Need | Where |
|---|---|
| How to get / cite datasets | [`docs/data_acquisition.md`](docs/data_acquisition.md) |
| BMW i3 EPA-derived params | `data/raw/epa/bmw_i3_derived_params.json` |
| TUM real trips (Tier A) | `data/raw/tum_bmw_i3/` (gitignored CSVs; download scripts below) |
| MoST network **with elevation** | `data/raw/sumo_scenarios/MoST/` |
| Reference SUMO scaffold (flat) | `simulation/` |

**Reference `map_with_tls.net.xml` is flat (no z).** Training traffic should use MoST.

## Quick setup

```bash
# Refresh public datasets (EPA, cycles, LuST/MoST)
chmod +x scripts/*.sh
./scripts/download_epa_and_cycles.sh
./scripts/download_sumo_scenarios.sh

# TUM BMW i3 via Kaggle (needs ~/.kaggle/kaggle.json)
./scripts/download_tum_kaggle.sh
```

Argonne D3 is manual — see `docs/data_acquisition.md`.

## Layout

```text
configs/          vehicles.yaml (BMW i3 filled from EPA; 3 placeholders)
simulation/       kept SUMO XMLs (scaffold)
src/              physics / data / models / losses / training / evaluation
data/raw/         Tier A–C inputs
archive/          original reference repo (CSVs, notebooks, checkpoints)
docs/             data_acquisition.md, assumptions (TBD)
PRD.md            full specification
```

## Next implementation steps

1. Implement `(E1)`–`(E11)` in `src/physics/` and validate on one TUM trip vs `I×V`
2. Complete four vehicles in `configs/vehicles.yaml` from EPA
3. Build MoST-based multi-vehicle dataset (PRD §9–§10)
4. Extract LSTM/preprocess logic from `archive/original_repo/Scripts/`

## License

MIT (see `LICENSE`). Respect third-party dataset terms (IEEE / Kaggle / ANL D3 / SUMO scenarios).
