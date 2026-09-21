# EV Range Prediction using Physics‑Informed Neural Networks (PINNs) — *with Real‑Traffic Simulation*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)]()
[![SUMO](https://img.shields.io/badge/Simulator-SUMO-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()



---

## Table of Contents
- [EV Range Prediction using Physics‑Informed Neural Networks (PINNs) — *with Real‑Traffic Simulation*](#ev-range-prediction-using-physicsinformed-neural-networks-pinns--with-realtraffic-simulation)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [🧩 Architecture](#-architecture)
  - [What’s New — Simulation](#whats-new--simulation)
  - [Model \& Physics](#model--physics)
  - [Data Schema](#data-schema)
  - [Quickstart](#quickstart)
    - [1️⃣ Setup](#1️⃣-setup)
    - [2️⃣ Run SUMO Simulation](#2️⃣-run-sumo-simulation)
    - [3️⃣ Preprocess Data](#3️⃣-preprocess-data)
    - [4️⃣ Train the PINN](#4️⃣-train-the-pinn)
    - [5️⃣ Validate with SUMO Data](#5️⃣-validate-with-sumo-data)
  - [Training \& Evaluation](#training--evaluation)
  - [Visualization](#visualization)
  - [Project Structure](#project-structure)
  - [Configuration](#configuration)
  - [Reproducibility](#reproducibility)
  - [Roadmap](#roadmap)
  - [Future Improvements](#future-improvements)
  - [Requirements](#requirements)
  - [License](#license)
  - [Author](#author)
  - [Summary](#summary)

---

## Overview

Conventional EV range estimators fail under **non‑stationary real‑world traffic**.  
This project combines **sequence modeling** with a **physics‑constrained loss** so predictions obey vehicle dynamics.  

Targets:
- **SOC (0–1)**
- **Traction/Aux Power (kW)**
- **Remaining Range (km)**

Key ideas:
- **PINN Loss** enforces power balance, SOC bounds, and energy conservation.  
- **SUMO Simulation** generates traffic‑accurate drive cycles (urban/arterial/highway).  
- **Attention** provides interpretability across time steps.

---
## 🧩 Architecture

```text
Input: [velocity, acceleration, elevation, ambient temperature, ...] → (time-series sequence)

            ┌───────────────────────────────┐
            │   Input Projection Layer      │
            └─────────────┬─────────────────┘
                          │
                   ┌──────▼──────┐
                   │   LSTM      │
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │ Attention   │
                   └──────┬──────┘
          ┌────────────────────────────────┐
          │                                │
   ┌──────▼──────┐                 ┌───────▼──────┐
   │ SOC Head    │                 │ Power Head   │
   └──────┬──────┘                 └──────┬───────┘
          │                                │
          └──────────────┬─────────────────┘
                         ▼
                 Range Estimator + Physics Layer
                         │
                   Final Outputs:
         SOC (0–1), Power (kW), Range (km)
```

---


## What’s New — Simulation

Real‑traffic simulation is now part of the pipeline:

1. **Network & Routes**: SUMO network created using provided XMLs (trips, routes, and vehicle types).  
2. **Traffic Profiles**: Realistic speed and acceleration patterns under traffic.  
3. **Telemetry Export**: Records speed, acceleration, elevation/grade, and stop events.  
4. **Data Preprocessing**: Generates enriched feature datasets for model training.  
5. **PINN Training**: Combines data-driven and physics-driven losses.

---

## Model & Physics

**Architecture**: LSTM → Attention → Dual heads (**SOC**, **Power**) → **Range head**  

**Physics constraint:**

$$
P(t) \approx \frac{(F_{drag} + F_{roll} + F_{grade} + F_{accel}) \, v}{\eta} + P_{aux}
$$

Where:

- Drag: \( F_{drag} = \tfrac{1}{2} \rho C_d A v^2 \)  
- Rolling: \( F_{roll} = m g C_r \)  
- Grade: \( F_{grade} = m g \sin(\theta) \)  
- Accel: \( F_{accel} = m a \)  

**SOC update (discrete):**

$$
SOC_{t+1} = SOC_t - \frac{P(t) \, \Delta t}{C_{bat}} \quad \text{s.t.} \; 0 \leq SOC \leq 1
$$


---

## Data Schema

**Raw SUMO outputs:** CSV and XML (speed, accel, position, trip info).  
**Processed datasets:** merged and feature-enriched CSVs.  

**Example columns:**
- `time_s, speed_mps, accel_mps2, lat, lon, elevation_m, grade, stop_flag, ambient_temp_C`
- Optional: `soc_meas, power_kw_meas` (if measured data exists)

---

## Quickstart

### 1️⃣ Setup
```bash
git clone https://github.com/kamaladhi/EV-Range-prediction-using-PINNS-under-real-traffic-condition.git
cd EV-Range-prediction-using-PINNS-under-real-traffic-condition
pip install -r requirements.txt
```

Install **SUMO** if not already:
```bash
# Linux
sudo apt install sumo sumo-tools

# Windows
# Download from: https://sumo.dlr.de/docs/Downloads.php
# Add SUMO/bin to PATH
```

### 2️⃣ Run SUMO Simulation
```bash
sumo -c Simulation/simulation.sumocfg
```
This generates raw telemetry under `Simulation/output/`

### 3️⃣ Preprocess Data
Open and execute:  
`Simulation/data_preprocess.ipynb`  
Outputs:  
- `filtered_sumo_data.csv`  
- `ev_sumo_dataset.csv`  
- `ev_sumo_dataset_16features.csv`  

### 4️⃣ Train the PINN
Open and run:  
`Scripts/pinns_model_new.ipynb`  
- Loads preprocessed data  
- Trains LSTM + Physics-Informed model  
- Saves model: `Scripts/15_07_model.pth`  
- Logs metrics: `training_history.pkl`  

### 5️⃣ Validate with SUMO Data
Open and run:  
`Scripts/validate_pinn_with_sumo.ipynb`  
- Uses SUMO-generated dataset for validation  
- Evaluates SOC, Power, and Range predictions  
- Plots stored in: `Scripts/ev_pinn_plots/`  

---

## Training & Evaluation

**Training notebook:** `Scripts/pinns_model_new.ipynb`  
**Validation notebook:** `Scripts/validate_pinn_with_sumo.ipynb`  

Metrics:
- **MAE / RMSE** for SOC, Power, and Range  
- **Physics residual** (lower is better)  
- **Percent within SOC bounds (0–1)**  

---

## Visualization

All visual outputs are saved in `Scripts/ev_pinn_plots/`:

- `training_validation_physics_loss.png`  
- `prediction_accuracy_errors_combined.png`  
- `physics_analysis.png`  
- `range_prediction_analysis.png`  
- `attention_heatmap.png`  

---

## Project Structure

```
EV-Range-prediction-using-PINNS-under-real-traffic-condition/
├── Scripts/
│   ├── ev_pinn_plots/                 # Output plots
│   ├── plots/                         # Additional figures
│   ├── 15_07_model.pth                # Trained model checkpoint
│   ├── training_history.pkl           # Loss and metric logs
│   ├── pinns_model_new.ipynb          # Main training notebook
│   └── validate_pinn_with_sumo.ipynb  # Simulation validation pipeline
│
├── Simulation/
│   ├── config/
│   │   └── bmw_i3/                    # Vehicle config (mass, Cd, etc.)
│   ├── ev_route_new1.rou.xml          # Route configuration
│   ├── ev_trips_new1.trips.xml        # Trip definitions
│   ├── ev_types.add.xml               # Vehicle types
│   ├── map_with_tls.net.xml           # SUMO network file
│   ├── simulation.sumocfg             # SUMO simulation config
│   ├── output/
│   │   ├── ev_sumo_dataset.csv
│   │   ├── ev_sumo_dataset_16features.csv
│   │   ├── filtered_sumo_data.csv
│   │   ├── summary.xml
│   │   ├── tripinfo.xml
│   │   └── simulation_summary.xml
│   └── data_preprocess.ipynb          # Preprocessing script
│
└── README.md
```

---

## Configuration

All hyperparameters and physics parameters are defined **inside the notebooks**:  
- `Scripts/pinns_model_new.ipynb` → Model, training, and loss configuration  
- `Simulation/data_preprocess.ipynb` → Vehicle physics and dataset features

Typical vehicle parameters:
```yaml
battery_capacity_kwh: 60.0
vehicle_mass_kg: 1750
drag_coeff: 0.28
frontal_area_m2: 2.2
rolling_resistance: 0.010
air_density: 1.225
gravity: 9.80665
drivetrain_efficiency: 0.92
aux_power_kw: 0.8
```

---

## Reproducibility

- Deterministic seeds for all runs  
- `requirements.txt` specifies frozen versions  
- Checkpoint (`15_07_model.pth`) included  
- SUMO input XMLs version-controlled for consistent simulation

---

## Roadmap

- Battery ageing integration (capacity fade & internal resistance)
- Weather and temperature profile coupling
- Deployment on Jetson / Raspberry Pi with CAN integration
- CARLA coupling for perception‑driven route simulation

---

##  Future Improvements

- Incorporate **battery degradation physics** for aging-aware prediction  
- Add **multi-modal fusion** (GPS, weather, and driver data)  
- Extend to **real-time deployment on edge hardware** (NVIDIA Jetson / Raspberry Pi)  
- Integrate **reinforcement learning** for adaptive eco-driving suggestions 
   
---

##  Requirements

- torch>=2.0.0
- numpy
- pandas
- matplotlib
- scikit-learn
- sumolib
- traci

---

##  License

This project is licensed under the **MIT License** — free for research, development, and educational use.

---

##  Author

**Jeevakamal K R**  

📧 [jeevakamal2005@gmail.com](mailto:jeevakamal2005@gmail.com)  
🌐 GitHub: [github.com/kamaladhi](https://github.com/kamaladhi)

---

##  Summary

> **EV-Range-Prediction-PINN**  
> blends **deep learning and vehicle physics**  
> to deliver **accurate, interpretable, and deployable** range estimation for the next generation of smart electric vehicles.

