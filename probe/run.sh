#!/usr/bin/env bash
# AP1a's bits: A0, script A and the tutorials, no port declared, base release/v0.3.0 against this head in one job, one
# CPU, one thread; `moved` is the control (a one-step port on A0's arm on the head) and must differ.
set -euo pipefail
mkdir -p probe/out
python .github/probe/bit_compare.py --base origin/release/v0.3.0 \
    --programs a0_f32 a0_f64 script_a_f32 script_a_f64 tutorial_a_f32 tutorial_a0_f32 moved_f32 \
    --command 'python {head}/probe/dump.py {program} {out}' --output probe/out/bit_compare.json
python - <<'PY'
import json
held = json.load(open("probe/out/bit_compare.json"))
for program, entry in held["programs"].items():
    print(program, entry.get("status"), entry.get("all_equal"))
PY
