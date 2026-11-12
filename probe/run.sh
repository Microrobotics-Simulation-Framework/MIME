#!/usr/bin/env bash
# The fix round of PR 267 at stage 1's operating point: reverse mode's memory and identities and a forward product's
# cost (B1, M2), the implemented solid depth on the plane (M2), gate 5 across the snap at the rule-2 cell (M4).
set -euo pipefail
export JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p probe/out
case "${PROBE_ARM}" in
  g5-*) python probe/gate5_op.py 2>&1 | tee "probe/out/log_${PROBE_ARM}.txt" ;;
  cost*|rev*) python probe/cost_op.py 2>&1 | tee "probe/out/log_${PROBE_ARM}.txt" ;;
  trunc) python probe/truncation_op.py 2>&1 | tee "probe/out/log_${PROBE_ARM}.txt" ;;
esac
