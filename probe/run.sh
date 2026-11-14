#!/usr/bin/env bash
# AP1a's per-step cost, used: the node count, the compiled step's operation count and the median chunk time, off and on
# alternated in this one job (one runner, one CPU model), 3 builds a feature, 20 steady runs a build, float32.
set -euo pipefail
mkdir -p probe/out
export OPENBLAS_NUM_THREADS=1
case "$PROBE_ARM" in
  a0) steps=40 ;;
  sphere) steps=200 ;;
  *) echo "unknown arm $PROBE_ARM" >&2; exit 2 ;;
esac
cd probe
for state in off on; do
  python count.py "${PROBE_ARM}_${state}" float32 > "out/count_${PROBE_ARM}_${state}.json"
done
for build in 1 2 3; do
  for state in off on; do
    python node_measure.py --case "probe_cases:${PROBE_ARM}_${state}" --path . --steps "$steps" --repeats 20 \
        --dtype float32 --output-json "out/measure_${PROBE_ARM}_${state}_${build}.json"
  done
done
python - "$PROBE_ARM" <<'PY'
import json, statistics, sys
from pathlib import Path
arm, out = sys.argv[1], Path("out")
summary = {}
for state in ("off", "on"):
    chunks = []
    passed = []
    for build in (1, 2, 3):
        held = json.loads((out / f"measure_{arm}_{state}_{build}.json").read_text())
        chunks += [entry["seconds"] for entry in held["phases"]["steady_run"]]
        passed.append(held["trace_check"]["passed"])
        cpu = held["environment"]["cpu_model"]
    count = json.loads((out / f"count_{arm}_{state}.json").read_text())
    summary[state] = {"median_chunk_s": statistics.median(chunks), "chunks": len(chunks), "trace_checks_passed": passed,
                      "nodes": count["nodes"], "hlo_instructions": count["hlo_instructions"],
                      "hlo_instructions_outside_fusions": count["hlo_instructions_outside_fusions"], "cpu": cpu}
summary["on_over_off"] = summary["on"]["median_chunk_s"] / summary["off"]["median_chunk_s"]
(out / f"summary_{arm}.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
