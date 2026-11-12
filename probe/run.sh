#!/usr/bin/env bash
# The fix round's controls, each rate by detection_rate.py: the section-flux mutant (the review's M2) on gate 1 and
# gate 3; plugs in series left unrefused; the forced-flow test off on a closed valve and on parallel plugs; the topology
# answering "forced" everywhere on the quiet cases; the not-finite refusal left silent (the solve's own NaN, node and
# experiment) and the start read as finite (a NaN in the drive); a body's loss taken twice beside the plug.
set -euo pipefail
mkdir -p probe/out
export JAX_PLATFORMS=cpu
T=tests/flow/test_porous_handover.py
Q="$T::test_a_driven_tube_a_gap_and_an_occluded_branch_are_not_flagged"
V="$T::test_a_closed_valve_and_parallel_plugs_below_one_inflow_are_flagged"
N="$T::test_a_porous_flows_non_finite_run_is_named_by_its_cause_through_the_experiment"
G1="$T::test_the_hand_overs_reading_agrees_with_the_resolved_route_where_both_hold"
B="$T::test_a_plug_beside_a_resolved_body_closes_its_books_and_keeps_its_throughflow"
rate() { python .github/probe/detection_rate.py "$@" --output "probe/out/${PROBE_ARM}.json"; }
case "$PROBE_ARM" in
  section_g1_0p1h) rate --node "$G1[0.1h]" --plant-var MIME_PLANT_SECTION_HIGH --runs 10 ;;
  section_g1_0p01h) rate --node "$G1[0.01h]" --plant-var MIME_PLANT_SECTION_HIGH --runs 10 ;;
  section_g3) rate --node "$T::test_far_below_the_line_the_readings_error_is_constant_in_k[16]" --plant-var MIME_PLANT_SECTION_HIGH --runs 10 ;;
  series_silent) rate --node "$T::test_two_resolved_plugs_in_series_read_the_closed_form_and_one_below_the_line_is_refused" --plant-var MIME_PLANT_SERIES_SILENT --runs 10 ;;
  forced_off_valve) rate --node "$V[valve]" --plant-var MIME_PLANT_FORCED_OFF --runs 30 ;;
  forced_off_parallel) rate --node "$V[parallel]" --plant-var MIME_PLANT_FORCED_OFF --runs 30 ;;
  forced_off_inflow) rate --node "$T::test_a_fixed_inflow_through_a_plug_below_the_line_is_flagged" --plant-var MIME_PLANT_FORCED_OFF --runs 30 ;;
  everywhere_periodic) rate --node "$Q[periodic]" --plant-var MIME_PLANT_FORCED_EVERYWHERE --runs 30 ;;
  everywhere_pressure) rate --node "$Q[pressure]" --plant-var MIME_PLANT_FORCED_EVERYWHERE --runs 30 ;;
  everywhere_gap) rate --node "$Q[gap]" --plant-var MIME_PLANT_FORCED_EVERYWHERE --runs 30 ;;
  everywhere_twin) rate --node "$Q[twin]" --plant-var MIME_PLANT_FORCED_EVERYWHERE --runs 30 ;;
  silent_twin) rate --node "$T::test_a_held_solve_that_goes_not_finite_is_refused_by_name" --plant-var MIME_PLANT_NOT_FINITE_SILENT --runs 30 ;;
  silent_solve) rate --node "$N[solve]" --plant-var MIME_PLANT_NOT_FINITE_SILENT --runs 5 ;;
  any_drive) rate --node "$N[drive]" --plant-var MIME_PLANT_NOT_FINITE_ANY --runs 5 ;;
  body_twice_below) rate --node "$B[below]" --plant-var MIME_PLANT_BODY_TWICE --runs 3 ;;
esac
