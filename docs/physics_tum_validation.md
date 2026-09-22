# Physics vs TUM TripA01

Gate: compare `simulate_trip` battery power to measured `P = -V·I` on BMW i3.

- Samples: 10090 at dt = 0.100 s, duration 1008.9 s
- Elevation std: 5.58 m
- Power MAE (all / moving): 2389 W / 2575 W
- Power RMSE (all): 3022 W
- Energy measured vs physics: 1.289 kWh vs 0.941 kWh (27.0% relative to I×V)
- Energy implied by ΔSOC × 18.8 kWh: 1.015 kWh (physics vs this: 7.4%)
- SOC start → end measured: 0.869 → 0.815; physics SOC end 0.819
- Regen clamp: min P_batt = -20895 W (limit −50000 W)

Machine-readable: `dataset/reports/physics_tum_tripa01.json`.

Instantaneous MAPE vs I×V is high (transients, constant η). Trip **energy vs I×V is 27%**;
that I×V integral also disagrees with ΔSOC×E_usable (1.29 vs 1.02 kWh).
**Physics vs SOC-implied energy is 7.4%**, and end SOC matches within 0.4 pp.
Treat as a pass on energy/SOC bookkeeping; do not expect tight per-timestep power fit
until HVAC is calibrated. SUMO battery-device 2% check is still pending (no MoST FCD yet).
