#!/usr/bin/env bash
# The hand-over's gate 2 at the open vessel's line (its quiet open cases at the 0.1h interior, the forced case), and the
# named refusal of a held solve gone not finite (gate 7), each control's rate by detection_rate.py.
set -euo pipefail
mkdir -p probe/out
export JAX_PLATFORMS=cpu
T=tests/flow/test_porous_handover.py
Q="$T::test_a_driven_tube_a_gap_and_an_occluded_branch_are_not_flagged"
rate() { python .github/probe/detection_rate.py "$@" --output "probe/out/${PROBE_ARM}.json"; }
case "$PROBE_ARM" in
  g2_forced_off) rate --node "$T::test_a_fixed_inflow_through_a_plug_below_the_line_is_flagged" --plant-var MIME_PLANT_FORCED_OFF --runs 50 ;;
  g2_everywhere_pressure) rate --node "$Q[pressure]" --plant-var MIME_PLANT_FORCED_EVERYWHERE --runs 50 ;;
  g2_everywhere_gap) rate --node "$Q[gap]" --plant-var MIME_PLANT_FORCED_EVERYWHERE --runs 50 ;;
  g2_everywhere_twin) rate --node "$Q[twin]" --plant-var MIME_PLANT_FORCED_EVERYWHERE --runs 50 ;;
  not_finite_twin) rate --node "$T::test_a_held_solve_that_goes_not_finite_is_refused_by_name" --plant-var MIME_PLANT_NOT_FINITE_SILENT --runs 30 ;;
  not_finite_experiment) rate --node "$T::test_a_porous_flows_solve_gone_not_finite_is_refused_by_name_through_the_experiment" --plant-var MIME_PLANT_NOT_FINITE_SILENT --runs 10 ;;
esac
