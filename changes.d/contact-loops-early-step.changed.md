- **The contact loops' Newton property (`tests/oracles/test_contact_loops.py`) also compares the looped search's first one
  to four steps with 0.4.0's unrolled ones, on every group of the sphere, the capsule, the cylinder and the ellipsoid,
  against an estimate of two programs' rounding times a stated factor of 5: an estimate, not a derived bound.** The
  factor follows a rule committed before its run (the smallest whole number at least twice what was measured), and the
  margin is measured: on CI's runners 0.4.0's own steps run op by op, a legitimately different program, stood at most
  0.36 of the allowance, and one float32 program's own rounding at most 0.47. That op-by-op program is a new clean
  control, which must pass on every run in both dtypes. The comparison needs no end state, so it judges the groups 0.4.0
  does not resolve at its end. On the cylinder alone it raised the property's per-run detection of a stale carry from 1
  and 118 to 196 and 198 of 200 runs (float32, float64) and of a stop at 1e-5 of the hull's size from 55 and 194 to 196
  and 195; all but one of the runs it missed drew no cylinder. A loop five steps short is caught in 197 and 198 runs, but
  at the counts compared that loop takes no step at all. The realistic short loop, a fixed seven steps, is the
  comparison's blind spot: at four steps it is the clean loop, and only the end state sees it, in 3 and 6 runs of 200.
  **FL-9's screw is a gap**: it is left out of the early comparison, because its support refines from the search's
  point, which the estimate does not carry. The end state holds it, every plant is now pinned on a screw draw, and its
  screw-only per-run detection was measured at 187 to 200 of 200. The route to exact is a separate change of the law,
  the maintainer's to decide. A test change; the contact is unchanged.
