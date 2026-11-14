#!/usr/bin/env bash
# AP2's fix round: the derivative-in-every-input property's two matrix-path plants (predictions
# tests/control/predictions_ap2_fix1.json), 30 planted and 30 clean runs each in each dtype, in two arms of 15, each run
# in a fresh process with its own Hypothesis seed.
set -u
mkdir -p probe/out
export JAX_PLATFORMS=cpu
lscpu > "probe/out/lscpu_${PROBE_ARM}.txt" 2>&1
NODE=tests/control/test_exact_allocation.py::test_the_derivative_in_every_input_is_the_selected_sets_in_both_modes
rate() {  # rate <dtype> <plant> <runs> <seed base>
  python .github/probe/detection_rate.py --node "$NODE[$1]" --plant-var ALLOCATION_PLANT --plant-value "$2" \
    --runs "$3" --seed-base "$4" --hypothesis-seed --timeout 2400 --output "probe/out/rate_${PROBE_ARM}.json"
  python -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d.get(k) for k in ('planted','clean','every_run')})[:3000])" "probe/out/rate_${PROBE_ARM}.json"
}
case "${PROBE_ARM}" in
  stop_q_f64_a) rate float64 stop_q 15 0 ;;
  stop_q_f64_b) rate float64 stop_q 15 15 ;;
  stop_q_f32_a) rate float32 stop_q 15 0 ;;
  stop_q_f32_b) rate float32 stop_q 15 15 ;;
  stop_inverse_f64_a) rate float64 stop_inverse 15 0 ;;
  stop_inverse_f64_b) rate float64 stop_inverse 15 15 ;;
  stop_inverse_f32_a) rate float32 stop_inverse 15 0 ;;
  stop_inverse_f32_b) rate float32 stop_inverse 15 15 ;;
  *) echo "unknown arm ${PROBE_ARM}"; exit 2 ;;
esac
