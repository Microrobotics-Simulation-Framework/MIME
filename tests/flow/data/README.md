# The FVM's stored reference data

- `golden_imex_step.npz`: one IMEX Euler step (backward Euler for viscosity and incompressibility, forward Euler for
  convection, the drive at the step's end) of an independent implementation of the monolithic projected CG, F0's
  tableau spike's, on a 16^3 periodic box of side 2 pi with no wall: `velocity_in` (a divergence-free state made of the
  manufactured solution's two fields at t = 0.3, projected), `force` (three random face fields, the drive `cos(t)
  force` evaluated at `t0 + dt`), `velocity_out`, and `nu = 0.1`, `dt = 0.02`, `t0 = 0.3`; float64, the solve to a
  relative true residual of 1e-13 (7.4e-14 reached, 8 Jacobi iterations). Its sha256 is pinned in
  `tests/flow/test_fvm_mms.py`.
