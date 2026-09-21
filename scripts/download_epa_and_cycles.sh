#!/usr/bin/env bash
# Download EPA Test Car Lists + FuelEconomy.gov vehicles dump + drive cycles.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

EPA="$ROOT/data/raw/epa"
CYC="$ROOT/data/raw/drive_cycles"
mkdir -p "$EPA" "$CYC"

curl -fsSL -o "$EPA/26-testcar.xlsx" "https://www.epa.gov/system/files/documents/2026-01/26-testcar-2026-01-21.xlsx"
curl -fsSL -o "$EPA/25-testcar.xlsx" "https://www.epa.gov/system/files/documents/2026-01/25-testcar-2026-01-21.xlsx"
curl -fsSL -o "$EPA/24-testcar.xlsx" "https://www.epa.gov/system/files/documents/2025-05/24-testcar-2025-05.xlsx"
curl -fsSL -o "$EPA/16tstcar.csv" "https://www.epa.gov/sites/default/files/2016-07/16tstcar.csv"
curl -fsSL -o "$EPA/15tstcar.csv" "https://www.epa.gov/sites/default/files/2016-07/15tstcar.csv"
curl -fsSL -o "$EPA/14tstcar.csv" "https://www.epa.gov/sites/default/files/2016-07/14tstcar.csv"

curl -fsSL -o "$EPA/vehicles.csv.zip" "https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip"
unzip -o "$EPA/vehicles.csv.zip" -d "$EPA" && rm "$EPA/vehicles.csv.zip"

curl -fsSL -o "$CYC/udds.txt" "https://www.epa.gov/sites/default/files/2015-10/uddscol.txt"
curl -fsSL -o "$CYC/hwfet.txt" "https://www.epa.gov/sites/default/files/2015-10/hwycol.txt"
curl -fsSL -o "$CYC/us06.txt" "https://www.epa.gov/sites/default/files/2015-10/us06col.txt"

echo "EPA + drive cycles refreshed under data/raw/"
