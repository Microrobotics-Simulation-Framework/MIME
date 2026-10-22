# Magnetic-swim debug — summary (overnight session)

## Goal
Make the de Jongh screw swim *through* the x-pipe under the rotating magnetic drive.

## What works (validated)
- **Magnetic chain**: field reaches the body, torque nonzero, varies with magnet strength (D1, D6, D7).
- **Swim mechanism**: the chirality coupling R_FΩ produces axial thrust that grows with
  rotation rate (D11: +48 nN @3Hz, +160 nN @10Hz on top of flow drag). The de-risk
  (`test_si_confined_bem`) confirms force-free swim = 3.27 mm/s. So **given rotation, the
  screw swims** — the coupled experiment's hydro is correct.
- **Frame-aware BEM** (ar4 x-axes): drag force AND torque frame-invariant to machine precision.

## Root cause of "spins but doesn't swim"
The screw does not **synchronise** to the field — it **whirls** (spins faster than the 3 Hz
drive, librating). Established:
- Not the drag-torque sign (reaction + -1 edge flip damps correctly — D3).
- Partly orientation-lag negative damping (staggered body.orientation→response edge); a
  coupling group removes the *monotonic* over-spin (D4) but not the whirl.
- **Not a numerical/dt artifact** — identical libration at dt 5e-4 vs 1.5e-4 (D12c). It's a
  genuine **under-damped** rotational ODE: tiny screw rotational inertia (~7e-11) + small
  axial rotational drag → second-order, underdamped → whirls instead of locking.
- The big implicit coupling group [fvm,bem,body,ext_magnet,magnet] (now added) doesn't fix it —
  the whirl is the rotational dynamics, not the coupling.

## The real fix (next session)
A microswimmer is **overdamped** (Re≪1, zero rotational inertia → first-order → locks, no
whirl). The RigidBodyNode overdamped mode (`use_inertial=False, use_analytical_drag=False`)
needs a **mobility** drag node (velocity out), but the BEM is a **resistance** node (force out)
→ NaNs at step 0 (D8). So the proper fix is a **mobility-based overdamped swimmer**:
  - Option A: have the BEM expose its resistance matrix R; the body solves [U;Ω]=R⁻¹·F_ext
    overdamped, coupling the background flow (the physically-correct model).
  - Option B (demo): prescribe the locked rotation Ω=drive-rate (valid below step-out) +
    emergent force-free translation — shows the swim, treats sync as the separate Problem-1 study.

Magnetic SYNC / step-out is itself the project's Problem 1 — so "does it lock" is a physics
question the project means to study, not only a bug.

## Infra added this session (committed, keep)
- `make_two_scale_coupling(..., extra_coupling_members=)` + `TwoScale(extra_coupling_members=)`
  — pulls body+magnetic chain into the Schwarz implicit group (attach TwoScale last).
- Composer `BODY_MODEL` = inertial | overdamped.
