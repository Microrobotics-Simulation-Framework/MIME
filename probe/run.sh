#!/usr/bin/env bash
# AP1a's controls, each measured with .github/probe/detection_rate.py: 200 runs a control, planted and clean at the same
# seeds, a heavy control split over arms by its seeds' base. PORT_PLANT names the plant (tests/port_cases.py).
set -euo pipefail
mkdir -p probe/out
STAGE_FILE=tests/ports/test_command_stage.py
CLASH=tests/ports/test_ports_build.py::test_a_port_named_as_another_ports_age_is_refused
MEASURE=tests/ports/test_measurement_ports.py::test_the_measure_stage_samples_and_delays_with_the_age_of_what_it_delivers
PROJECT=tests/sensing/test_camera_port.py::test_the_cameras_projection_is_opencvs_project_points
IDEAL=tests/ports/test_ports_build.py::test_no_port_an_empty_one_and_an_all_default_one_build_the_program_of_none
LOOP=tests/ports/test_a0_ports.py::test_the_ported_loop_builds_with_its_stage_between_writer_and_receiver
LEDGER=tests/ports/test_port_ledger.py::test_a_metered_receivers_command_is_booked_under_its_writer_through_a_port
A0=tests/ports/test_a0_ports.py::test_a_one_step_port_latency_on_a0_changes_its_program_and_moves_its_swim_speed
hypothesis=()
base=0
runs=200
case "$PROBE_ARM" in
  slew_before_bound|bound_before_delay|trip_unlatched) node=$STAGE_FILE; plant=$PROBE_ARM; hypothesis=(--hypothesis-seed) ;;
  latency_short|stale_age) node=$MEASURE; plant=$PROBE_ARM; hypothesis=(--hypothesis-seed) ;;
  project_undistorted|project_swapped_focal|project_inverse_rotation) node=$PROJECT; plant=$PROBE_ARM; hypothesis=(--hypothesis-seed) ;;
  stage_for_ideal|measure_for_ideal) node=$IDEAL; plant=$PROBE_ARM ;;
  clash_unchecked) node=$CLASH; plant=clash_unchecked ;;
  stage_after_receiver) node=$LOOP; plant=stage_after_receiver ;;
  ledger_blind_*) node=$LEDGER; plant=ledger_blind; runs=100; base=$(( ${PROBE_ARM##*_} * 100 )) ;;
  latency_dropped_*) node=$A0; plant=latency_dropped; runs=50; base=$(( ${PROBE_ARM##*_} * 50 )) ;;
  *) echo "unknown arm $PROBE_ARM" >&2; exit 2 ;;
esac
export OPENBLAS_NUM_THREADS=1
python .github/probe/detection_rate.py --node "$node" --plant-var PORT_PLANT --plant-value "$plant" --runs "$runs" \
    --seed-base "$base" "${hypothesis[@]}" --timeout 900 --output "probe/out/detection_rate_${PROBE_ARM}.json"
python - "probe/out/detection_rate_${PROBE_ARM}.json" <<'PY'
import json, sys
held = json.load(open(sys.argv[1]))
planted, clean = held.get("planted", {}), held.get("clean", {})
print(f"{sys.argv[1]}: caught {planted.get('caught')}/{planted.get('runs')}, clean flagged {clean.get('flagged')}/"
      f"{clean.get('runs')}, every_run {held.get('every_run')}")
PY
