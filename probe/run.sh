#!/usr/bin/env bash
# AP1b's probes. PROBE_ARM `bits`: A0, script A, the tutorials and the AR4's servo, no port declared but the servo's,
# base AP1a's branch against this head in one job, one CPU, one thread; `moved` is the control and must differ. Every
# other arm is a control measured with .github/probe/detection_rate.py: 200 runs, planted and clean at the same seeds.
set -euo pipefail
mkdir -p probe/out
export OPENBLAS_NUM_THREADS=1
STAGE_FILE=tests/ports/test_command_stage.py
ARM=tests/ports/test_ports_build.py::test_an_arm_port_applies_its_latency_bounds_slew_and_lag_in_the_designs_order
hypothesis=()
case "$PROBE_ARM" in
  bits)
    python .github/probe/bit_compare.py --base origin/apparatus/ap1-ports \
        --programs a0_f32 a0_f64 script_a_f32 script_a_f64 tutorial_a_f32 tutorial_a0_f32 servo_f32 servo_f64 moved_f32 \
        --command 'python {head}/probe/dump.py {program} {out}' --output probe/out/bit_compare.json
    python - <<'PY'
import json
held = json.load(open("probe/out/bit_compare.json"))
for program, entry in held["programs"].items():
    print(program, entry.get("status"), entry.get("all_equal"))
PY
    exit 0 ;;
  respond_before_slew) node=$STAGE_FILE; plant=$PROBE_ARM; hypothesis=(--hypothesis-seed) ;;
  arm_slew_before_bound) node=$ARM; plant=slew_before_bound ;;
  arm_limits_after_slew) node=$ARM; plant=limits_after_slew ;;
  *) echo "unknown arm $PROBE_ARM" >&2; exit 2 ;;
esac
python .github/probe/detection_rate.py --node "$node" --plant-var PORT_PLANT --plant-value "$plant" --runs 200 \
    "${hypothesis[@]}" --timeout 900 --output "probe/out/detection_rate_${PROBE_ARM}.json"
python - "probe/out/detection_rate_${PROBE_ARM}.json" <<'PY'
import json, sys
held = json.load(open(sys.argv[1]))
planted, clean = held.get("planted", {}), held.get("clean", {})
print(f"{sys.argv[1]}: caught {planted.get('caught')}/{planted.get('runs')}, clean flagged {clean.get('flagged')}/"
      f"{clean.get('runs')}, every_run {held.get('every_run')}")
PY
