#!/usr/bin/env bash
# The fix round of PR 267: the new oracles' discriminating ranges (discriminating_range.py; each oracle deterministic,
# one draw a size) and gate 9's control rate on its rewritten test (the base column warmed from its history).
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p probe/out
range() {  # range <oracle> <lo> <hi>
  python probe/discriminating_range.py --oracle "ranges_fix:$1" --path probe --lo "$2" --hi "$3" --draws 1 \
      --points 8 --refine 4 --output-json "probe/out/range_${PROBE_ARM}.json" 2>&1 | tee "probe/out/log_${PROBE_ARM}.txt"
}
case "${PROBE_ARM}" in
  r-cell) range cell 1e-4 1e-1 ;;
  r-transpose) range transpose 1e-7 1e-1 ;;
  r-snap) range snap 1e-5 5e-2 ;;
  g9-warm) python .github/probe/detection_rate.py --node "tests/patch/test_patch_plane.py::test_gate9_a_warm_start_gives_the_cold_answer_and_iterates" \
      --plant-var PATCH_PLANT --plant-value accept_start --runs 5 --clean-runs 5 \
      --output "probe/out/detection_rate_${PROBE_ARM}.json" 2>&1 | tee "probe/out/log_${PROBE_ARM}.txt" ;;
esac
