#!/bin/bash
# r39 probe: independent near-wall flow references (NGSolve Taylor-Hood). One arm = one spec file.
set -u
mkdir -p probe/out
uv pip install --quiet --python "$(command -v python)" ngsolve==6.2.2608 netgen-mesher==6.2.2608 mkl mpmath scipy > probe/out/install_${PROBE_ARM}.log 2>&1
python -I -c "import ngsolve; print('ngsolve', ngsolve.__version__)" >> probe/out/install_${PROBE_ARM}.log 2>&1
nproc > probe/out/nproc_${PROBE_ARM}.txt; free -g >> probe/out/nproc_${PROBE_ARM}.txt
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
while IFS= read -r line; do
  [ -z "$line" ] && continue
  python -I probe/code/run_lin.py "probe/out/${PROBE_ARM}.jsonl" "$line" >> "probe/out/${PROBE_ARM}.log" 2>&1
  echo "spec exit $?" >> "probe/out/${PROBE_ARM}.log"
done < "probe/specs/${PROBE_ARM}.specs"
echo ALLDONE >> "probe/out/${PROBE_ARM}.log"
