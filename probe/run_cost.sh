#!/usr/bin/env bash
# Camera spike R1, item 3: the forward and reverse costs through the real node (node_measure.py), one configuration an
# arm, named <mode>-<dtype bits>-<width>-<steps>-r<render ckpt>s<sub-sample ckpt>c<cache-remake ckpt>[-p<period>]:
# mode fw (forward only), ck (MIME's chunked, checkpointed reverse run), pl (the plain reverse chunk).
set -u
mkdir -p probe/out
arm="${PROBE_ARM:?an arm}"
lscpu | grep -i "model name" > probe/out/cpu.txt
if [ "$arm" = eq-160 ]; then
  # The checkpoint placements give the unchecked gradient to rounding (T3's bound, 128 eps S), both dtypes, 160x120.
  out=probe/out/eq-160; mkdir -p $out
  for dt in float64 float32; do
    for v in "0 0 0" "0 1 1" "1 1 1"; do set -- $v
      MIME_SPIKE_RENDER_CHECKPOINT=$1 MIME_SPIKE_SUBSAMPLE_CHECKPOINT=$2 MIME_SPIKE_CACHE_CHECKPOINT=$3 \
        python probe/r1_grad.py --dtype $dt --out $out/$dt-r$1s$2c$3.json --no-ladder 2>&1 | tail -1
    done
  done
  python - <<'PY'
import json
for dt in ("float64", "float32"):
    base = json.load(open(f"probe/out/eq-160/{dt}-r0s0c0.json"))
    for v in ("r0s1c1", "r1s1c1"):
        other = json.load(open(f"probe/out/eq-160/{dt}-{v}.json"))
        rows = [(n, (other["g_vjp"][i] - base["g_vjp"][i]) / (base["eps"] * base["scale"][i]), other["g_vjp"][i] == base["g_vjp"][i])
                for i, n in enumerate(base["names"])]
        print(dt, v, "pass" if all(abs(r[1]) <= 128 for r in rows) else "FAIL", [(n, round(x, 2), b) for n, x, b in rows])
PY
  exit 0
fi
case "$arm" in
  bt-*)
    # R3: the camera batched under vmap (probe/r1_batch.py), forward then reverse, each in its own process.
    IFS=- read -r _ spec extra <<< "$arm"
    b=${spec%[st]}; kind=stacked; [ "${spec: -1}" = t ] && kind=theta
    period=5; case "${extra:-}" in p*) period=${extra#p} ;; esac
    for m in fwd rev; do
      python probe/r1_batch.py --batch $b --kind $kind --mode $m --period $period --out probe/out/$arm-$m.json 2>&1 | tail -1
    done
    exit 0 ;;
esac
IFS=- read -r mode bits width steps flags extra <<< "$arm"
r=${flags:1:1}; s=${flags:3:1}; c=${flags:5:1}
period=1; case "${extra:-}" in p*) period=${extra#p} ;; esac
args=(--case r1_core:cost_case --path probe --arg width=$width --arg steps=$steps --arg period_steps=$period
      --arg render_checkpoint=$r --arg subsample_checkpoint=$s --arg cache_checkpoint=$c
      --steps $steps --repeats 2 --dtype float$bits --output-json probe/out/$arm.json)
case "$mode" in
  fw) ;;
  ck) args+=(--reverse camera:.fy) ;;
  pl) args+=(--reverse camera:.fy --no-checkpoint) ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac
python probe/node_measure.py "${args[@]}"; echo "node_measure exit $?"
