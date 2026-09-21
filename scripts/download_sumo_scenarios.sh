#!/usr/bin/env bash
# Refresh LuST / MoST SUMO scenarios from upstream GitHub.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/raw/sumo_scenarios"
mkdir -p "$OUT"
cd "$OUT"

curl -fL -o LuSTScenario.zip "https://github.com/lcodeca/LuSTScenario/archive/refs/heads/master.zip"
rm -rf LuST
unzip -q LuSTScenario.zip && mv LuSTScenario-master LuST && rm LuSTScenario.zip

curl -fL -o MoSTScenario.zip "https://github.com/lcodeca/MoSTScenario/archive/refs/heads/master.zip"
rm -rf MoST
unzip -q MoSTScenario.zip && mv MoSTScenario-master MoST && rm MoSTScenario.zip

echo "LuST -> $OUT/LuST (flat network — needs DEM if used for grade)"
echo "MoST -> $OUT/MoST (has elevation — preferred)"
