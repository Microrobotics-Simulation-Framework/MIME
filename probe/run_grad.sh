#!/usr/bin/env bash
# Camera spike R1, items 1, 2 and 4: the gradient gates, the controls' detection rates, the switch scan.
set -u
mkdir -p probe/out
export PYTHONHASHSEED=0
arm="${PROBE_ARM:-}"
lscpu | grep -i "model name" > probe/out/cpu.txt
grad() { python probe/r1_grad.py "$@"; }
case "$arm" in
  gate-160|gate-320)  # gate-320 runs on the local route (its reverse program is beyond a runner)
    width=${arm#gate-}; out=probe/out/$arm; mkdir -p $out
    for rc in 0 1; do for dt in float64 float32; do
      step_flag=""; [ "$width" = 160 ] && step_flag="--per-step"
      MIME_SPIKE_RENDER_CHECKPOINT=$rc grad --dtype $dt --width $width --out $out/$dt-rc$rc.json $step_flag 2>&1 | tail -3
    done; done
    python probe/r1_judge.py $out/*.json > $out/judged.json; echo "judge exit $?"
    for rc in 0 1; do MIME_SPIKE_RENDER_CHECKPOINT=$rc python probe/r1_scan.py $out/scan-rc$rc.json 2>&1 | tail -12; done ;;
  gate-cont)
    out=probe/out/$arm; mkdir -p $out
    for dt in float64 float32; do grad --dtype $dt --continued 6 --out $out/$dt-rc0.json --per-step 2>&1 | tail -3; done
    python probe/r1_judge.py $out/*.json > $out/judged.json; echo "judge exit $? (predicted 1: the continued run misses the cache's part)" ;;
  scan)
    for rc in 0 1; do
      MIME_SPIKE_RENDER_CHECKPOINT=$rc timeout 9000 python -m pytest -q -p no:cacheprovider tests/saved/test_switch_scan.py \
        > probe/out/test_switch_scan-rc$rc.txt 2>&1; echo "test_switch_scan rc$rc exit $?" | tee -a probe/out/scan_exit.txt
      tail -5 probe/out/test_switch_scan-rc$rc.txt
    done ;;
  det-*)
    rest=${arm#det-}; plant=${rest%-*}; index=${rest##*-}
    python .github/probe/detection_rate.py --script probe/r1_check.py --plant-var CAMERA_R1_PLANT \
      --plant-value $plant-stop --runs 5 --clean-runs 2 --seed-base $((1000 + 5 * index)) --timeout 2400 \
      --output probe/out/detection-$plant-$index.json; echo "detection_rate exit $?" ;;
  *) echo "unknown arm $arm"; exit 2 ;;
esac
