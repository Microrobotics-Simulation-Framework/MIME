#!/usr/bin/env bash
# The fix round of PR 267, its second rates probe: the column gate's control on its restated test, and gate 10 at its
# own scale (the history dropped) at three corners. Every target deterministic.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p probe/out
out="probe/out/detection_rate_${PROBE_ARM}.json"
case "${PROBE_ARM}" in
  m5-columns) python .github/probe/detection_rate.py --node "tests/patch/test_patch_columns.py::test_vmapped_over_seven_columns_the_reflection_is_the_loop_within_its_rounding" \
      --plant-var PATCH_PLANT --plant-value short_cap --runs 5 --clean-runs 5 --output "${out}" 2>&1 | tee "probe/out/log_${PROBE_ARM}.txt" ;;
  g10-*)
    export JAX_ENABLE_X64=1
    case "${PROBE_ARM}" in g10-l497) key=m50-dt0.5-g0.15 ;; g10-l248) key=m200-dt0.5-g0.15 ;; g10-l079) key=m200-dt5-g0.15 ;; esac
    python .github/probe/detection_rate.py --script probe/gate10_control.py --plant-var PATCH_PLANT --plant-value history_dropped \
        --runs 2 --clean-runs 2 --timeout 7200 --output "${out}" -- "${key}" 2>&1 | tee "probe/out/log_${PROBE_ARM}.txt" ;;
esac
