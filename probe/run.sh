#!/usr/bin/env bash
# The fix round of PR 267: the solid depth in the tube at stage 1's box, consistent given values against the cost
# probe's (the steady Stokeslets' negative under a shift).
set -euo pipefail
export JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p probe/out
python probe/tube_depth.py 2>&1 | tee "probe/out/log_${PROBE_ARM}.txt"
