#!/usr/bin/env bash
# w7fib arm A: sphere / punch / plant on a bond zone blended into a graded continuum block. One job per arm.
set -u
mkdir -p probe/out
A="${PROBE_ARM:-}"
echo "arm=$A start $(date -u +%FT%TZ)"; nproc; python -c "import jax; print(jax.__version__, jax.devices())"
P="python probe/w7fib/arm_a.py"
h_of() { case "$1" in h02) echo 0.2;; h014) echo 0.1414;; h01) echo 0.1;; esac; }
nu_of() { case "$1" in n25) echo 0.25;; n40) echo 0.4;; n45) echo 0.45;; esac; }
IFS=_ read -r kind f1 f2 f3 <<< "$A"
case "$kind" in
  s) $P --case sphere --h "$(h_of $f1)" --nu "$(nu_of $f2)" --corr "$f3" --out "probe/out/$A.json" ;;
  p) for c in fe none volume ghost skin; do $P --case punch --h "$(h_of $f1)" --nu 0.4 --corr $c --out "probe/out/${A}_$c.json"; done ;;
  bf) $P --case sphere --h 0.2 --nu 0.4 --corr fe --L "$f1" --out "probe/out/$A.json" ;;
  plant) $P --case plant --h 0.2 --nu 0.4 --corr none --seeds 10 --frac 0.02 --out "probe/out/$A.json" ;;
  zone3) $P --case sphere --h 0.2 --nu 0.4 --corr "$f1" --zone 3.0 --out "probe/out/$A.json" ;;
  pmb) $P --case sphere --h 0.2 --nu 0.4 --corr "$f1" --stencil pmb --out "probe/out/$A.json" ;;
  *) echo "unknown arm $A"; exit 2 ;;
esac
echo "end $(date -u +%FT%TZ)"
