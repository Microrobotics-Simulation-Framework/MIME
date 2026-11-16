- **The contact loops' Newton property (`tests/oracles/test_contact_loops.py`) also compares the looped search's first one
  to four steps with 0.4.0's unrolled ones, on every group of the sphere, the capsule, the cylinder and the ellipsoid,
  within a bound derived from the step's rounding (never fitted).** It needs no end state, so it judges the groups 0.4.0
  does not resolve at its end. On the cylinder alone it raised the property's per-run detection of a loop five steps
  short from 2 to 194 and 198 of 200 runs (float32, float64), of a stale carry from 1 and 118 to 196 and 194, and of a
  stop at 1e-5 of the hull's size from 55 and 194 to 195; the runs it missed drew no cylinder. A loop capped at seven
  steps is still seen only at the end, about 1 run in 100. FL-9's screw is left out, because its support refines from the
  search's point, which the bound does not carry, and the end-state judgement holds it as before. A second Hypothesis
  target steers draws towards the groups whose search needs the most steps. A test change; the contact is unchanged.
