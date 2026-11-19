#!/usr/bin/env bash
# Spike W2F: one arm per job (PROBE_ARM), its configurations in probe/w2f/arms_table.txt. Kind g: the graph's step
# (probe/w2f/g.py, CORE spike 2's step on a held body); kind m: MIME's MAC resolved-body node (probe/w2f/mac.py).
set -euo pipefail
mkdir -p probe/out
export JAX_PLATFORMS=cpu W2F_OUT=probe/out
nproc > probe/out/nproc.txt
grep -m1 "model name" /proc/cpuinfo > probe/out/cpu.txt || true
python probe/w2f/refs.py > probe/out/refs.json
line="$(awk -v a="${PROBE_ARM}" '$1 == a {print $2, $3}' probe/w2f/arms_table.txt)"
kind="${line%% *}"; configs="${line#* }"
[ -n "$kind" ] || { echo "no arm ${PROBE_ARM}"; exit 2; }
case "$kind" in
  m)
    JAX_ENABLE_X64=1 W2F_CONFIGS="$configs" W2F_BUDGET_S=17500 timeout 310m python probe/w2f/mac.py 2>&1 | tee "probe/out/${PROBE_ARM}.log" ;;
  g)
    sudo apt-get install -y -qq libglu1-mesa libgl1 libxrender1 libxcursor1 libxft2 libxinerama1 libfontconfig1 > /dev/null 2>&1 || true
    uv pip install --python "$(command -v python)" gmsh==4.15.2 pyamg==5.4.1 > probe/out/pip.log 2>&1
    python -c "import gmsh, pyamg, jax, numpy, scipy; print('gmsh', gmsh.__version__, 'pyamg', pyamg.__version__, 'jax', jax.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)" | tee probe/out/versions.txt
    W2F_CONFIGS="$configs" W2F_BUDGET_S=17000 timeout 310m python probe/w2f/g.py 2>&1 | tee "probe/out/${PROBE_ARM}.log" ;;
esac
