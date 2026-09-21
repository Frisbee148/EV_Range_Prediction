#!/usr/bin/env bash
# Download TUM BMW i3 dataset from Kaggle (authoritative public mirror of IEEE DataPort).
# Requires: pip install kaggle && ~/.kaggle/kaggle.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/raw/tum_bmw_i3"
mkdir -p "$OUT"
kaggle datasets download -d atechnohazard/battery-and-heating-data-in-real-driving-cycles -p "$OUT" --unzip
echo "Downloaded to $OUT"
echo "Cite: Steinstraeter et al., DOI 10.21227/6jr9-5235; Kaggle atechnohazard/battery-and-heating-data-in-real-driving-cycles"
