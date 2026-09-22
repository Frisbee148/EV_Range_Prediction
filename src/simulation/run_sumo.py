"""SUMO CLI wrapper. Raises if `sumo` is not on PATH."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def sumo_available() -> bool:
    return shutil.which("sumo") is not None


def run_sumo(sumocfg: Path, extra: list[str] | None = None) -> None:
    if not sumo_available():
        raise FileNotFoundError("sumo binary not found; using IDM/MoST kinematics fallback")
    cmd = ["sumo", "-c", str(sumocfg)]
    if extra:
        cmd.extend(extra)
    subprocess.run(cmd, check=True)
