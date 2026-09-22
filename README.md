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
configs/          vehicles.yaml (four EPA-backed EV configs; see docs/)
simulation/       kept SUMO XMLs (scaffold)
src/              physics / data / models / losses / training / evaluation
data/raw/         Tier A–C inputs
archive/          original reference repo (CSVs, notebooks, checkpoints)
docs/             data_acquisition.md, finalized_vehicle_models_report.md
PRD.md            full specification
```

## Next implementation steps

1. Build MoST-based multi-vehicle dataset (PRD §9–§10)
2. Extract LSTM/preprocess logic from `archive/original_repo/Scripts/`

## License

MIT (see `LICENSE`). Respect third-party dataset terms (IEEE / Kaggle / ANL D3 / SUMO scenarios).
