#!/usr/bin/env bash
# F5 PR 1's fifth gates probe: gates 2 (the walls by the grid), 4 (a contact body one-pass), 7 (the read of a provided
# Poiseuille flow) and 10 (a rotating helix's far field): the cases and their durations, and each control's rate by
# detection_rate.py (planted with MIME_PLANT_SCHWARZ and clean at the same seeds, from 1000). One arm a runner.
set -u
mkdir -p probe/out
RUNS="${PROBE_RUNS:-3}"
F=tests/flow
WALL="$F/test_schwarz_walls.py::test_the_walls_by_the_grid_reproduce_the_band[0.2-2]"
CONTACT="$F/test_schwarz_contact.py::test_a_body_in_contact_folds_the_handed_response_into_its_step"
POISEUILLE="$F/test_schwarz_poiseuille.py::test_a_sphere_held_then_free_reads_the_tubes_poiseuille_flow[0.2-0.0]"
FAR="$F/test_schwarz_far_field.py::test_a_rotating_helixs_scattered_distribution_carries_its_far_field"
rate() {  # node plant output runs
  python .github/probe/detection_rate.py --node "$1" --plant-var MIME_PLANT_SCHWARZ --plant-value "$2" \
    --seed-var DETECTION_SEED --seed-base 1000 --runs "$4" --output "probe/out/$3.json" || status=1
}
cases() {  # output pytest-arguments...
  local out="$1"; shift
  python -m pytest -q -p no:cacheprovider -rA --durations=0 -s "$@" > "probe/out/$out.txt" 2>&1 || status=1
}
poses() { for p in "$@"; do echo "$F/test_schwarz_walls.py::test_the_walls_by_the_grid_reproduce_the_band[$p]"; done; }
lscpu | grep -E "Model name|^CPU\(s\)" > probe/out/cpu.txt
status=0
case "${PROBE_ARM:-}" in
  walls_a)    cases walls_a $(poses 0.1-1 0.1-2 0.1-4 0.2-1 0.2-2 0.2-4 0.2-8 0.2-axis) ;;
  walls_b)    cases walls_b $(poses 0.3-1 0.3-2 0.3-4 0.34-1 0.34-2 0.34-4) ;;
  walls32)    cases walls32 "$F/test_schwarz_walls.py::test_the_walls_by_the_grid_reproduce_the_band_in_float32" ;;
  offset)     rate "$WALL" offset walls_offset "$RUNS" ;;
  image)      rate "$WALL" image walls_image "$RUNS" ;;
  contact)    cases contact "$F/test_schwarz_contact.py"; rate "$CONTACT" diagonal contact_diagonal 5 ;;
  poiseuille) cases poiseuille "$F/test_schwarz_poiseuille.py" ;;
  unread)     rate "$POISEUILLE" unread poiseuille_unread "$RUNS" ;;
  far)        cases far "$F/test_schwarz_far_field.py"; rate "$FAR" net far_net "$RUNS" ;;
  *) echo "unknown arm ${PROBE_ARM:-}"; exit 2 ;;
esac
exit $status
