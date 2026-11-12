#!/usr/bin/env bash
# The bits: every saved program but the porous plug (now below the line at D/h 8), and that plug above the line, base (the release tip) against the head,
# one job, one CPU, one thread; the repeat control (the head against itself) and the one-ulp control (the head's leaves
# one ulp up). The fields this PR adds are left out of the head's dump (probe/dump.py).
set -euo pipefail
mkdir -p probe/out
export JAX_PLATFORMS=cpu
A="script_a script_a_metered hull_contact coupled_toy dynamic_rpm kinematic_rpm periodic_tube_flow steady_tube_flow rk3_tube_flow prescribed_tube_flow"
B="nudged_toy open_tube_flow mrs_free_space mrs_near_field mrs_tube rft_free_space rft_tube coil_array gantry_magnet tracked_rpm"
C="full_toy walled_toy script_a_uncoupled probed_shapes tree_flow open_tube_loop closed_tube point_body_flow porous_plug_above resolved_body_flow"
D="mrs_near_field_gmres mrs_graded mrs_solids mrs_plane mrs_point_forces mrs_kept rft_kept bound_robot bound_robot_with_parts hull_contact_looped massive_rpm"
E="anchored_toy lab_bead rigid_explicit script_a_body script_a_kept script_a_ramped script_a_records script_a_unblended swing_magnet_kept twisted_toy"
programs() { for p in $1; do echo "$p:float64 $p:float32"; done; }
case "$PROBE_ARM" in
  bits_a) set -- $(programs "$A"); base=origin/release/v0.3.0 ;;
  bits_b) set -- $(programs "$B"); base=origin/release/v0.3.0 ;;
  bits_c) set -- $(programs "$C"); base=origin/release/v0.3.0 ;;
  bits_d) set -- $(programs "$D"); base=origin/release/v0.3.0 ;;
  bits_e) set -- $(programs "$E"); base=origin/release/v0.3.0 ;;
  repeat) set -- porous_plug_above:float64 porous_plug:float64 periodic_tube_flow:float64 open_tube_flow:float32 resolved_body_flow:float32; base=HEAD ;;
  one_ulp) set -- porous_plug_above:float64:ulp periodic_tube_flow:float64:ulp open_tube_flow:float32:ulp resolved_body_flow:float32:ulp; base=HEAD ;;
esac
python .github/probe/bit_compare.py --base "$base" --programs "$@" \
    --command 'python {head}/probe/dump.py {program} {out} {tree} {head}' --output "probe/out/bit_compare_${PROBE_ARM}.json"
python - "$PROBE_ARM" <<'PY'
import json, sys
arm = sys.argv[1]
d = json.load(open(f"probe/out/bit_compare_{arm}.json"))
print(arm, "all_equal", d.get("all_equal"))
PY
