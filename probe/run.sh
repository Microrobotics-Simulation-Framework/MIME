#!/usr/bin/env bash
# Spike w7bur: one arm a job (PROBE_ARM); numpy/scipy arms of probe/bur/arms.py, body arms of probe/bur/free.py (fr_*).
set -euo pipefail
mkdir -p probe/out
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 A_OUT=probe/out OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
grep -m1 "model name" /proc/cpuinfo > probe/out/cpu.txt || true
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)" | tee probe/out/versions.txt
case "${PROBE_ARM}" in
  L1-*) export BUR_ORDER=1; PROBE_ARM="${PROBE_ARM#L1-}" timeout 300m python probe/bur/arms.py 2>&1 | tail -n 50 | tee "probe/out/${PROBE_ARM}.log" ;;
  fr_*) timeout 300m python probe/bur/free.py 2>&1 | tail -n 400 | tee "probe/out/${PROBE_ARM}.log" ;;
  *) timeout 300m python probe/bur/arms.py 2>&1 | tail -n 50 | tee "probe/out/${PROBE_ARM}.log" ;;
esac
