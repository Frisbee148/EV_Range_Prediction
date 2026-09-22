"""MoST/IDM dataset gates: elevation, schema, diversity."""

from pathlib import Path

import numpy as np

from src.data.preprocessing import SCHEMA, labeled_trip
from src.simulation.idm_trace import idm_trace
from src.simulation.most_net import edges_for_scenario, parse_net, polyline_of, sample_walk
from src.vehicles import load_physics_constants, load_vehicle

ROOT = Path(__file__).resolve().parents[1]
NET = ROOT / "tests" / "fixtures" / "tiny.net.xml"
VEHICLES = ROOT / "configs" / "vehicles.yaml"


def test_tiny_net_has_elevation_and_labels():
    edges = parse_net(NET)
    rng = np.random.default_rng(1)
    vehicle = load_vehicle(VEHICLES, "bmw_i3_60ah")
    constants = load_physics_constants(VEHICLES)
    pool = edges_for_scenario(edges, "urban")
    walk = sample_walk(edges, pool, rng, min_length_m=800.0)
    s_poly, xyz = polyline_of(walk, edges)
    kin = idm_trace(s_poly, xyz, 80.0, 1.0, vehicle, 13.0, rng, "urban")
    df = labeled_trip(
        kin,
        vehicle,
        constants,
        trip_id="t0",
        scenario="urban",
        dt_s=1.0,
        ambient_temp_c=21.0,
        soc_0=0.8,
        noise_seed=1,
    )
    assert df is not None
    assert list(df.columns) == list(SCHEMA)
    assert float(np.std(df["elevation_m"])) > 1.0
    assert int(df["stop_flag"].max()) in (0, 1)


def test_two_vehicle_traces_are_not_identical():
    edges = parse_net(NET)
    constants = load_physics_constants(VEHICLES)
    speeds = []
    for vid, seed in (("bmw_i3_60ah", 2), ("tesla_model_x_75d_2016", 99)):
        rng = np.random.default_rng(seed)
        vehicle = load_vehicle(VEHICLES, vid)
        pool = list(edges)
        walk = sample_walk(edges, pool, rng, min_length_m=1200.0)
        s_poly, xyz = polyline_of(walk, edges)
        kin = idm_trace(s_poly, xyz, 120.0, 1.0, vehicle, 20.0, rng, "arterial")
        speeds.append(kin["speed_mps"])
    corr = float(np.corrcoef(speeds[0], speeds[1])[0, 1])
    assert abs(corr) < 0.9
