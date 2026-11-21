#!/usr/bin/env bash
# w7ref: one arm of the near-wall finite-Re reference (NGSolve Taylor-Hood, steady Newton). PROBE_ARM names the arm in
# probe/w7ref/arms_spec.json. NGSolve, MKL and pypardiso go into an environment of their own, outside the project's.
set -u
mkdir -p probe/out
{ date -u; lscpu | head -20; free -g; nproc; } > "probe/out/${PROBE_ARM}.machine.txt" 2>&1
uv venv -q /tmp/ngenv --python 3.12
uv pip install -q --python /tmp/ngenv/bin/python ngsolve==6.2.2608 mkl==2026.2.1 pypardiso==0.4.7 numpy==2.5.5 scipy==1.18.1 \
  > "probe/out/${PROBE_ARM}.install.txt" 2>&1 || { echo "install failed"; cat "probe/out/${PROBE_ARM}.install.txt"; exit 1; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 REF_THREADS=4
/tmp/ngenv/bin/python -I probe/w7ref/ref_arm.py "$PROBE_ARM" probe/out > "probe/out/${PROBE_ARM}.log" 2>&1
rc=$?
echo "exit $rc" >> "probe/out/${PROBE_ARM}.log"
tail -5 "probe/out/${PROBE_ARM}.log"
exit $rc
