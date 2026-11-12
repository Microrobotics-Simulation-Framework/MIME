#!/usr/bin/env bash
# The fix round of PR 267: each new control's detection rate (detection_rate.py, fresh processes; every target is
# deterministic, so the runs show the plant keeps its margin and the clean arm stays quiet, not a draw distribution).
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p probe/out
rate() {  # rate <plant> <runs> <clean runs> --node <id> | --script <file> -- <args>
  local plant="$1" runs="$2" clean="$3"; shift 3
  python .github/probe/detection_rate.py "$@" --plant-var PATCH_PLANT --plant-value "${plant}" --runs "${runs}" \
      --clean-runs "${clean}" --timeout 7200 --output "probe/out/detection_rate_${PROBE_ARM}.json" \
      2>&1 | tee "probe/out/log_${PROBE_ARM}.txt"
}
case "${PROBE_ARM}" in
  m3-cell) rate cell_scale 5 5 --node "tests/patch/test_patch_plane.py::test_gate1_the_patchs_share_of_the_normal_entry_is_the_bands[cell]" ;;
  b1-identity) rate transpose_scaled 5 5 --node "tests/patch/test_patch_reverse.py::test_the_transpose_and_the_vjp_are_the_columns_transposes" ;;
  b1-memory) rate map_transposed 5 5 --node "tests/patch/test_patch_reverse.py::test_the_transpose_and_the_vjp_hold_no_more_than_the_forwards_temporaries" ;;
  m5-columns) rate short_cap 5 5 --node "tests/patch/test_patch_columns.py::test_vmapped_over_seven_columns_the_reflection_is_the_loop_within_its_rounding" ;;
  g5-snap) rate stale_given 5 5 --node "tests/patch/test_patch_motion.py::test_gate5_a_snap_moves_m_a_little_and_between_snaps_its_derivative_is_smooth" ;;
  g11-cell) rate read_stopped 5 5 --node "tests/patch/test_patch_motion.py::test_gate11_the_reflections_derivative_in_the_bodys_position_agrees_with_differences[cell]" ;;
  g10-l497) export JAX_ENABLE_X64=1; rate history_dropped 2 2 --script probe/gate10_control.py -- m50-dt0.5-g0.15 ;;
  g10-l248) export JAX_ENABLE_X64=1; rate history_dropped 2 2 --script probe/gate10_control.py -- m200-dt0.5-g0.15 ;;
  g10-l079) export JAX_ENABLE_X64=1; rate history_dropped 2 2 --script probe/gate10_control.py -- m200-dt5-g0.15 ;;
esac
