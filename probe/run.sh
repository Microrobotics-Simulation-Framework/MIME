#!/usr/bin/env bash
# AP2's controls on the final test code (predictions tests/control/predictions_ap2.json): the swap test's plant (limits
# 1 % looser) over 60 seeds in four arms, and the derivative test's two plants over 30 seeds in two arms each, each
# with its clean arm at the same seeds.
set -u
mkdir -p probe/out
export JAX_PLATFORMS=cpu
lscpu > "probe/out/lscpu_${PROBE_ARM}.txt" 2>&1
NODE=tests/control/test_exact_allocation.py
SWAP=test_the_exact_allocation_meets_scipys_bvls_within_the_line
DERIVATIVE=test_the_derivative_is_the_selected_sets_in_both_modes
rate() {  # rate <test> <plant> <runs> <seed base>
  python .github/probe/detection_rate.py --node "$NODE::$1" --plant-var ALLOCATION_PLANT --plant-value "$2" \
    --runs "$3" --seed-base "$4" --hypothesis-seed --timeout 2400 --output "probe/out/rate_${PROBE_ARM}.json"
  python -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d.get(k) for k in ('planted','clean','every_run')})[:3000])" "probe/out/rate_${PROBE_ARM}.json"
}
case "${PROBE_ARM}" in
  swap_a) rate "$SWAP" limits 15 0 ;;
  swap_b) rate "$SWAP" limits 15 15 ;;
  swap_c) rate "$SWAP" limits 15 30 ;;
  swap_d) rate "$SWAP" limits 15 45 ;;
  free_tangent_a) rate "$DERIVATIVE" free_tangent 15 0 ;;
  free_tangent_b) rate "$DERIVATIVE" free_tangent 15 15 ;;
  stop_branch_a) rate "$DERIVATIVE" stop_branch 15 0 ;;
  stop_branch_b) rate "$DERIVATIVE" stop_branch 15 15 ;;
  *) echo "unknown arm ${PROBE_ARM}"; exit 2 ;;
esac
