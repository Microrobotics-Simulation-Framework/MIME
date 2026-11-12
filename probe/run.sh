#!/usr/bin/env bash
# F2b PR 4a: R-15's gates on decision (A) in product code (the Inflow's opening prescribed) under IMEX Euler, the bits
# with condition 3's exception, and the measurements (A)'s validity text states. Predictions:
# tests/flow/predictions_prescribed_inlet.json, committed before this probe.
set -u
mkdir -p probe/out
export OMP_NUM_THREADS=4
RELEASE_BASE=74d73542bbba3e0b8ddb082a8001a695498282e2
case_() { python probe/fb7node.py "$@"; }
fb9() {  # name, tilt, interior, extensions..., one process a reading
  local name=$1 tilt=$2 interior=$3; shift 3
  for e in "$@"; do
    JAX_ENABLE_X64=1 python probe/fb9.py --extensions "$e" --interiors "$interior" --tilt "$tilt" --output "probe/out/${name}_${interior}_${e}.jsonl"
  done
}
devlen() { JAX_ENABLE_X64=1 python probe/devlen.py --reynolds "$1" --cells "$2" --output "probe/out/devlen_$1_$2.json"; }
case "${PROBE_ARM}" in
  fb7_implicit_f64) JAX_ENABLE_X64=1 case_ --reynolds 300 --periods 2.25 --form implicit --output probe/out/fb7_implicit_f64.jsonl ;;
  fb7_implicit_f32) JAX_ENABLE_X64=0 case_ --reynolds 300 --periods 2.25 --form implicit --output probe/out/fb7_implicit_f32.jsonl ;;
  fb7_explicit_f64) JAX_ENABLE_X64=1 case_ --reynolds 300 --periods 2.25 --form explicit --output probe/out/fb7_explicit_f64.jsonl ;;
  fb7_explicit_f32) JAX_ENABLE_X64=0 case_ --reynolds 300 --periods 2.25 --form explicit --output probe/out/fb7_explicit_f32.jsonl ;;
  fb7_off_f64) JAX_ENABLE_X64=1 case_ --reynolds 300 --periods 2.25 --form off --output probe/out/fb7_off_f64.jsonl ;;
  fb7_off_f32) JAX_ENABLE_X64=0 case_ --reynolds 300 --periods 2.25 --form off --output probe/out/fb7_off_f32.jsonl ;;
  damp_implicit_f64) JAX_ENABLE_X64=1 case_ --reynolds 100 --periods 2 --form implicit --output probe/out/damp_implicit_f64.jsonl ;;
  damp_implicit_f32) JAX_ENABLE_X64=0 case_ --reynolds 100 --periods 2 --form implicit --output probe/out/damp_implicit_f32.jsonl ;;
  damp_explicit_f64) JAX_ENABLE_X64=1 case_ --reynolds 100 --periods 2 --form explicit --output probe/out/damp_explicit_f64.jsonl ;;
  damp_explicit_f32) JAX_ENABLE_X64=0 case_ --reynolds 100 --periods 2 --form explicit --output probe/out/damp_explicit_f32.jsonl ;;
  damp_off_f64) JAX_ENABLE_X64=1 case_ --reynolds 100 --periods 2 --form off --output probe/out/damp_off_f64.jsonl ;;
  damp_off_f32) JAX_ENABLE_X64=0 case_ --reynolds 100 --periods 2 --form off --output probe/out/damp_off_f32.jsonl ;;
  devlen_*) set -- $(echo "${PROBE_ARM#devlen_}" | tr _ ' '); devlen "$1" "$2" ;;
  residual) JAX_ENABLE_X64=1 python probe/residual.py --seeds 3000 3040 --output probe/out/residual.jsonl ;;
  flags_f64_*) i=${PROBE_ARM#flags_f64_}; JAX_ENABLE_X64=1 python probe/flags.py --lines $((i * 50)) $((i * 50 + 50)) --output "probe/out/flags_f64_$i.jsonl" ;;
  flags_f32_*) i=${PROBE_ARM#flags_f32_}; JAX_ENABLE_X64=0 python probe/flags.py --lines $((i * 50)) $((i * 50 + 50)) --output "probe/out/flags_f32_$i.jsonl" ;;
  rate_*)  # rate_<flag short>_<plant>_<f64|f32>
    set -- $(echo "${PROBE_ARM#rate_}" | sed 's/_f64$/ 1/; s/_f32$/ 0/'); which=$1; x64=$2
    flag=open_face_projection; plant=weights_uniform; line64=1e-14; line32=3e-6
    case "$which" in
      projection) ;;
      inconv) flag=opening_kinetic_flux; plant=kinetic_in_n_conv; line32=5e-6 ;;
      endhalf) flag=opening_kinetic_flux; plant=end_full_cell; line32=5e-6 ;;
      starthalf) flag=opening_kinetic_flux; plant=start_full_cell; line32=5e-6 ;;
    esac
    line=$line64; [ "$x64" = 0 ] && line=$line32
    JAX_ENABLE_X64=$x64 python .github/probe/detection_rate.py --script probe/flags.py --plant-var OPEN_PLANT \
      --plant-value "$plant" --runs 40 --seed-base 3000 --output "probe/out/${PROBE_ARM}.json" -- --flag "$flag" --line "$line" ;;
  f32rev_*)  # f32rev_<f64|f32>_<01|001>
    set -- $(echo "${PROBE_ARM#f32rev_}" | tr _ ' '); x64=1; [ "$1" = f32 ] && x64=0
    interior=0.1; [ "$2" = 001 ] && interior=0.01
    JAX_ENABLE_X64=$x64 python probe/f32rev.py --seeds 1 4 --interior "$interior" --steps 4 --output "probe/out/${PROBE_ARM}.jsonl" ;;
  fb9_tilted_01) fb9 fb9_tilted 30 0.1 0.25 2.0 ;;
  fb9_tilted_001) fb9 fb9_tilted 30 0.01 0.25 2.0 ;;
  fb9_aligned) fb9 fb9_aligned 0 0.1 0.25 2.0; fb9 fb9_aligned 0 0.01 0.25 2.0 ;;
  fb9_long) fb9 fb9_tilted 30 0.1 4.0 ;;
  fb9_steady)
    for cells in 8 16; do
      JAX_ENABLE_X64=1 python probe/fb9.py --extensions 0.25 --interiors 0.01 --tilt 30 --steady --cells "$cells" \
        --output "probe/out/fb9_steady_${cells}.jsonl"
    done ;;
  ptree_f32) python probe/ptree.py 32 > probe/out/ptree_f32.jsonl ;;
  ptree_f64) python probe/ptree.py 64 > probe/out/ptree_f64.jsonl ;;
  bits_release)
    programs=""
    for p in script_a script_a_metered hull_contact coupled_toy dynamic_rpm kinematic_rpm periodic_tube_flow \
             steady_tube_flow rk3_tube_flow prescribed_tube_flow tree_flow nudged_toy point_body_flow \
             resolved_body_flow porous_plug open_tube_loop closed_tube open_tube_flow; do
      programs="$programs $p@float64 $p@float32"
    done
    cp probe/dump.py /tmp/dump_programs.py
    python .github/probe/bit_compare.py --base "$RELEASE_BASE" --programs $programs \
      --command 'python /tmp/dump_programs.py {program} {out}' --output probe/out/bit_compare_release.json \
      > probe/out/bits_release.log 2>&1
    tail -40 probe/out/bits_release.log ;;
  *) echo "unknown arm ${PROBE_ARM}"; exit 2 ;;
esac
