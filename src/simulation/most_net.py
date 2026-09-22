"""Parse SUMO net.xml edges, lanes, connections, and elevation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Edge:
    edge_id: str
    speed_mps: float
    length_m: float
    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray
    succs: list[str] = field(default_factory=list)


def parse_net(path: Path | str) -> dict[str, Edge]:
    tree = ET.parse(path)
    root = tree.getroot()
    edges: dict[str, Edge] = {}
    for el in root.findall("edge"):
        if el.get("function") == "internal":
            continue
        lane = el.find("lane")
        if lane is None:
            continue
        shape = lane.get("shape") or ""
        pts = []
        for tok in shape.split():
            parts = tok.split(",")
            if len(parts) < 2:
                continue
            x, y = float(parts[0]), float(parts[1])
            z = float(parts[2]) if len(parts) >= 3 else 0.0
            pts.append((x, y, z))
        if len(pts) < 2:
            continue
        arr = np.asarray(pts, dtype=float)
        eid = el.get("id")
        if not eid:
            continue
        edges[eid] = Edge(
            edge_id=eid,
            speed_mps=float(lane.get("speed", "13.89")),
            length_m=float(lane.get("length", "1")),
            xs=arr[:, 0],
            ys=arr[:, 1],
            zs=arr[:, 2],
        )
    for el in root.findall("connection"):
        a, b = el.get("from"), el.get("to")
        if a in edges and b in edges and b not in edges[a].succs:
            edges[a].succs.append(b)
    return edges


def edges_for_scenario(edges: dict[str, Edge], scenario: str) -> list[str]:
    ids = list(edges)
    if scenario == "urban":
        return [i for i in ids if edges[i].speed_mps <= 14.0]
    if scenario == "arterial":
        mid = [i for i in ids if 14.0 < edges[i].speed_mps <= 22.0]
        return mid or [i for i in ids if edges[i].speed_mps <= 22.0]
    hwy = [i for i in ids if edges[i].speed_mps > 22.0]
    return hwy or ids


def sample_walk(
    edges: dict[str, Edge],
    start_ids: list[str],
    rng: np.random.Generator,
    min_length_m: float,
) -> list[str]:
    start = str(rng.choice(start_ids))
    walk = [start]
    length = edges[start].length_m
    guard = 0
    while length < min_length_m and guard < 4000:
        guard += 1
        cur = edges[walk[-1]]
        nxts = [s for s in cur.succs if s in edges]
        if not nxts:
            start = str(rng.choice(start_ids))
            walk.append(start)
            length += edges[start].length_m
            continue
        nxt = str(rng.choice(nxts))
        walk.append(nxt)
        length += edges[nxt].length_m
    return walk


def polyline_of(walk: list[str], edges: dict[str, Edge]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys, zs = [], [], []
    for eid in walk:
        e = edges[eid]
        xs.extend(e.xs.tolist())
        ys.extend(e.ys.tolist())
        zs.extend(e.zs.tolist())
    pts = np.column_stack([xs, ys, zs])
    dxy = np.sqrt(np.diff(pts[:, 0]) ** 2 + np.diff(pts[:, 1]) ** 2)
    s = np.concatenate([[0.0], np.cumsum(dxy)])
    return s, pts
