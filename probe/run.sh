#!/usr/bin/env bash
# The hand-over's line at D/h 64 over the long periods (3.5 to 4 diameters): an arm INTERIOR_LINE (0p for "0.") runs
# probe/line_cap.py periodic on 5 draws.
set -euo pipefail
mkdir -p probe/out
export JAX_PLATFORMS=cpu
IFS=_ read -r interior line <<< "$PROBE_ARM"
interior="${interior/0p/0.}"; line="${line/0p/0.}"
python probe/line_cap.py periodic "$interior" 64 "$line" 5 "probe/out/${PROBE_ARM}.json" 3.5:4.0
