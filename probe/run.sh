#!/usr/bin/env bash
# w7fib arms C (strip: toughness and strength against the horizon), B (blend residuals), D1 (affine network), E (MIME's
# body node pushed into the bond zone). One job per arm.
set -u
mkdir -p probe/out
A="${PROBE_ARM:-}"
echo "arm=$A start $(date -u +%FT%TZ)"; nproc
S="python -u probe/w7fib/strip2d.py --wall 18500 --protect 6"
dm_n() { [ "$1" = small ] && echo 2.0 || echo 20; }
dm_u() { [ "$1" = small ] && echo 5.0 || echo 45; }
IFS=_ read -r kind f1 f2 f3 f4 <<< "$A"
case "$kind" in
  cn) $S --regime "$f1" --h "$f2" --law "$f3" --notch 10.5 --dmax "$(dm_n $f1)" --steps 140 --out "probe/out/$A.json" ;;
  cu) $S --regime "$f1" --h "$f2" --law "$f3" --notch 0 --grip free_x --dmax "$(dm_u $f1)" --steps 120 --out "probe/out/$A.json" ;;
  ce) $S --regime "$f1" --h "$f2" --law elastic --notch "$f3" --steps 100 --dmax "$(dm_n $f1)" --out "probe/out/$A.json" ;;
  cp) pr="$f2"; [ "$pr" = degraded ] && pr=tension_degraded; $S --regime "$f1" --h 0.3 --law bilinear --notch 10.5 --nu 0.45 --pressure "$pr" --dmax "$(dm_n $f1)" --steps 140 --out "probe/out/$A.json" ;;
  cf) $S --regime "$f1" --h 0.3 --law bilinear --notch 10.5 --compressive 0.1 --dmax "$(dm_n $f1)" --steps 140 --out "probe/out/$A.json" ;;
  cg) $S --regime small --h 0.3 --law bilinear --notch 10.5 --gc_factor 1.5 --dmax 2.0 --steps 140 --out "probe/out/$A.json" ;;
  cge) $S --regime small --h 0.3 --law elastic --notch "$f1" --steps 70 --gc_factor 1.5 --out "probe/out/$A.json" ;;
  b) python -u probe/w7fib/blend3d.py --out "probe/out/$A.json" ;;
  d1) python -u probe/w7fib/affine.py --out "probe/out/$A.json" ;;
  e) python -u probe/w7fib/mime_e.py --mode "$f1" --steps 80 --out "probe/out/$A.json" ;;
  *) echo "unknown arm $A"; exit 2 ;;
esac
echo "end $(date -u +%FT%TZ)"
