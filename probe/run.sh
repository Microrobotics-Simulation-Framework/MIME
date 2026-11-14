#!/usr/bin/env bash
# AP2 on its final head (predictions tests/control/predictions_ap2.json): the switch scan's configurations and time, base
# and head on one runner, and the switch harness and solve walk on the head (the gates probe probe/ap2-gates ran the rest on
# a head whose src/ is this one's).
set -u
mkdir -p probe/out
export JAX_PLATFORMS=cpu
lscpu > "probe/out/lscpu_${PROBE_ARM}.txt" 2>&1
NODE=tests/control/test_exact_allocation.py
rate() {  # rate <node> <plant> <runs> <seed base>
  python .github/probe/detection_rate.py --node "$NODE::$1" --plant-var ALLOCATION_PLANT --plant-value "$2" \
    --runs "$3" --seed-base "$4" --hypothesis-seed --timeout 2400 --output "probe/out/rate_${PROBE_ARM}.json"
  python - "probe/out/rate_${PROBE_ARM}.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps({k: d.get(k) for k in ("planted", "clean", "every_run")}, indent=1)[:4000])
PY
}
base_tree() {  # the release tip as a worktree; its source run with this checkout's environment (PYTHONPATH first)
  git fetch -q --depth 1 origin release/v0.3.0 && git worktree add -q --detach base FETCH_HEAD
  git -C base log --oneline -1 | tee probe/out/base_commit.txt
}
SINGLE="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
status=0
case "${PROBE_ARM}" in
  det_limits) rate test_the_exact_allocation_meets_scipys_bvls_within_the_line limits 30 0 || status=1 ;;
  det_free_tangent_a) rate test_the_derivative_is_the_selected_sets_in_both_modes free_tangent 15 0 || status=1 ;;
  det_free_tangent_b) rate test_the_derivative_is_the_selected_sets_in_both_modes free_tangent 15 15 || status=1 ;;
  det_stop_branch_a) rate test_the_derivative_is_the_selected_sets_in_both_modes stop_branch 15 0 || status=1 ;;
  det_stop_branch_b) rate test_the_derivative_is_the_selected_sets_in_both_modes stop_branch 15 15 || status=1 ;;
  det_single_where)
    rate test_every_guard_of_the_law_is_reached_false_and_its_fallback_reaches_no_output_or_derivative single_where 30 0 \
      || status=1 ;;
  float32)
    XLA_FLAGS="$SINGLE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0 python probe/float32.py --count 3000 \
      --output probe/out/float32.json || status=1 ;;
  moves)
    base_tree || exit 1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_FLAGS="$SINGLE"
    (cd base && PYTHONPATH="$PWD/src" taskset -c 0 python ../probe/allocate_dump.py ../probe/out/allocate_base.npz) || status=1
    taskset -c 0 python probe/allocate_dump.py probe/out/allocate_head.npz || status=1
    python probe/allocate_moves.py probe/out/allocate_base.npz probe/out/allocate_head.npz probe/out/allocate_moves.json \
      || status=1 ;;
  share_a|share_b)
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_FLAGS="$SINGLE"
    for dtype in float64 float32; do
      for spec in navion:true navion:false cardiomag:true cardiomag:false; do
        name="${spec%%:*}_${spec##*:}_${dtype}"
        echo "=== $name"
        timeout 1500 taskset -c 0 python probe/node_measure.py --case "share_case:${spec%%:*}" --path probe \
          --arg "allocating=${spec##*:}" --steps 200 --repeats 3 --dtype "$dtype" --final controller.write_currents \
          --output-json "probe/out/share_${name}.json" > "probe/out/share_${name}.log" 2>&1 || { echo "$name failed"; status=1; }
        tail -2 "probe/out/share_${name}.log"
      done
    done
    python probe/share_collect.py probe/out > "probe/out/share_${PROBE_ARM}.json" || status=1
    cat "probe/out/share_${PROBE_ARM}.json" ;;
  scan)
    base_tree || exit 1
    for tree in base head base head; do
      if [ "$tree" = base ]; then dir=base; path="$PWD/base/src"; else dir=.; path=""; fi
      (cd "$dir" && PYTHONPATH="$path" python - <<'PY'
import sys
sys.path[:0] = ["tests", "tests/saved"]
from switch_configurations import configurations
print("configurations", len(configurations()))
PY
      ) | tee -a "probe/out/scan_${tree}.txt"
      start=$(date +%s.%N)
      (cd "$dir" && PYTHONPATH="$path" python -m pytest -q -p no:cacheprovider -n 4 tests/saved/test_switch_scan.py -W ignore \
        > "$OLDPWD/probe/out/scan_${tree}_$(date +%s).log" 2>&1) || { echo "scan $tree failed"; status=1; }
      end=$(date +%s.%N)
      echo "$tree seconds $(python -c "print(round($end - $start, 1))")" | tee -a probe/out/scan_times.txt
    done
    tail -3 probe/out/scan_*_*.log ;;
  walk)
    python -m pytest -q -p no:cacheprovider -n 4 tests/saved/test_switch_harness.py tests/saved/test_solve_walk.py \
      tests/solve_sites/test_solve_sites.py -W ignore > probe/out/walk.log 2>&1 || status=1
    tail -15 probe/out/walk.log ;;
  bits)
    set -- script_a:float64 script_a:float32 tracked_rpm:float64 tracked_rpm:float32 a0_graph:float64 a0_graph:float32 \
      coil_array:float64 coil_array:float32 gantry_magnet:float64 gantry_magnet:float32 sphere_settling:float64 \
      sphere_settling:float32
    python .github/probe/bit_compare.py --base origin/release/v0.3.0 --programs "$@" \
      --command 'python {head}/probe/dump.py {program} {out} {tree} {head}' --output probe/out/bit_compare_bits.json \
      || status=1 ;;
  bits_controls)
    python .github/probe/bit_compare.py --base HEAD --programs script_a:float64 a0_graph:float32 coil_array:float64 \
      --command 'python {head}/probe/dump.py {program} {out} {tree} {head}' --output probe/out/bit_compare_repeat.json \
      || status=1
    python .github/probe/bit_compare.py --base HEAD --programs script_a:float64:ulp a0_graph:float32:ulp \
      coil_array:float64:ulp --command 'python {head}/probe/dump.py {program} {out} {tree} {head}' \
      --output probe/out/bit_compare_one_ulp.json || status=1 ;;
  *) echo "unknown arm ${PROBE_ARM}"; exit 2 ;;
esac
for f in probe/out/bit_compare_*.json; do
  [ -f "$f" ] && python -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], 'all_equal', d.get('all_equal'))" "$f"
done
exit $status
