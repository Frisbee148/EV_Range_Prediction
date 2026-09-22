# Elevation check (reference network)

Command equivalent (sumolib not required):

```python
# parsed all shape="x,y" / "x,y,z" vertices in map_with_tls.net.xml
z_count = 0   # NO_Z_COORDINATES
```

| Network | z range | Decision |
|---|---|---|
| `simulation/networks/map_with_tls.net.xml` | none | Scaffold only |
| `data/raw/sumo_scenarios/MoST/.../most.net.xml` | [-0.08, 616.3] m | **Use for training** |
| MoST IDM dataset FCD-equivalent (`sumo_most_1hz`) | std(z) ≈ 171 m | Gate `std(z) > 1.0` **pass** |
| `data/raw/sumo_scenarios/LuST/.../lust.net.xml` | none | Needs DEM if used |
