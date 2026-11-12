#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
mkdir -p probe/out
programs=$(python probe/list_programs.py)
echo "arm ${PROBE_ARM}: ${programs}" | tee probe/out/programs_${PROBE_ARM}.txt
python .github/probe/bit_compare.py --base origin/release/v0.3.0 --programs ${programs} \
    --command 'python {head}/probe/dump.py {program} {out}' --output probe/out/bit_compare_${PROBE_ARM}.json \
    2>&1 | tee probe/out/log_${PROBE_ARM}.txt
