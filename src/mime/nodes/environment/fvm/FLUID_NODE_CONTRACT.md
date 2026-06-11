# `FVMFluidNode` Contract

> For the interface **shared** with MIME's other fluid nodes — the single-body
> `drag_force`/`drag_torque` outputs and `body_*` inputs that make the fluid
> nodes graph-interchangeable — see [`../FLUID_NODE_CONTRACT.md`](../FLUID_NODE_CONTRACT.md).
> This document covers the FVM node's method-specific details.

This document records the interface decisions made during the M0 → M3
implementation rounds (and subsequent diagnose-first rounds R3 → R6) of
the graph-native FVM fluid node. It is the authoritative reference for
downstream consumers (rigid-body integrators, magnetic-actuation nodes,
GraphManager wiring).

## State pytree

The node's state is a flat dict of JAX arrays with static shape. All
fields are present for every step, regardless of whether the relevant
physics path is exercised.

| Key                | Shape                       | Meaning                                                   |
| ------------------ | --------------------------- | --------------------------------------------------------- |
| `u`                | `[N_cells, dim]`            | Cell-centred velocity. **`u_hom` when `lifting` is provided**, otherwise the physical velocity. |
| `u_pre_ibm`        | `[N_cells, dim]`            | Physical velocity *before* the post-projection Brinkman pass. Read by surface-integral and Brinkman force extractors. |
| `u_after_explicit` | `[N_cells, dim]`            | Physical velocity *after* explicit advection, *before* the pre-step Brinkman. Read by the `force_method="brinkman"` extractor (so the IBM penalty signal is not yet zeroed inside the body). |
| `p`                | `[N_cells]`                 | Pressure. **`p_hom` when `lifting` is provided** — the lifted pressure gradient (e.g., the Hagen-Poiseuille axial gradient that drives the lifted flow) is implicit in the lift balance and is *not* stored here. |
| `F`                | `[N_faces]`                 | Face mass flux of `u` (whichever frame `u` is in).        |
| `t`                | scalar                      | Simulation time.                                          |
| `i_step`           | int32 scalar                | Step index. Used to dynamic-index time-varying lifting fields. |
| `force_<body>`     | `[dim]`                     | Hydrodynamic force on each dynamic body (output flux).    |
| `torque_<body>`    | `[3]` (3D) or scalar (2D)   | Hydrodynamic torque on each dynamic body (output flux).   |

## Boundary inputs

Per dynamic body the node accepts `<name>_position`, `<name>_linear_velocity`,
and (3D) `<name>_angular_velocity`. The `IBMBody`'s SDF and rigid-body
velocity are rebuilt each step from these inputs so SDF/u_body gradients
with respect to pose are differentiable end-to-end.

## Boundary fluxes

Per dynamic body the node emits `force_<body>` (in N) and `torque_<body>`
(in N·m, 3D only). Force-extraction backends:

| `force_method`        | Source field           | Notes                                                                                                                                              |
| --------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"brinkman"`          | `u_after_explicit`     | Per-cell `α (u − u_body) χ_body` integrated over the body. Biased low at coarse IBM resolution (cpr ≲ 4); kept for backwards compatibility.        |
| `"surface_integral"`  | `u_pre_ibm`, `p`       | Cauchy stress integrated on a cell shell just outside the body (default 1.5–3.5 cells). Highly shell-location dependent on diffuse IBM (10× variation across (0.5,2.5) / (1.5,3.5) / (2.5,4.5)); shell (0.5, 2.5) — *inside* the IBM transition band — is closest to literature K_Happel for confined Stokes (within 30% at λ=0.3, cpr=6). The (1.5, 3.5) default sits in the post-IBM transition zone and under-reads by ~70%. Pass `p_lift_fn` when the lifting decomposition is in use. |
| `"momentum_deficit"`  | `u`, `p`               | Control-volume momentum balance (`F = ΔM + ΔP·A + ρ·f·V_CV − F_wall`). **Valid only at moderate-to-high Re** where wakes/pressure perturbations extend many diameters and reach the integration planes. **Invalid for Re ≪ 10**: the Stokes pressure dipole decays as 1/r² and is mostly localised within ~r_b of the body; integrating at sphere ± 5 r_b captures only ~3% of the dipole signal even when the IBM force is correct. Confirmed by drag-diagnostic sprint. Use `surface_integral_force` (with shell (0.5, 2.5) inside the IBM band) for Stokes validation. Requires `_pipe_radius` attribute on the node. |

## Lifting / homogenisation contract

The DST/DCT spectral basis used by the implicit-diffusion Helmholtz
solver enforces `u = 0` at Dirichlet walls regardless of any face-level
non-zero value passed through the convection / diffusion operators
(observed in R6 as a 14–29% Poiseuille profile error). The fix is the
classical field decomposition:

```
u(x, t)  =  u_lift(x, t)  +  u_hom(x, t)
```

where `u_lift` satisfies the non-zero Dirichlet BC analytically and
`u_hom` is zero at all walls. The PISO loop evolves `u_hom`; the
spectral basis sees a homogeneous problem and is exact.

### Source terms in the `u_hom` equation

Substituting the decomposition into incompressible Navier-Stokes gives

```
∂u_hom/∂t + (u_hom · ∇)u_hom
        =  -∇p_hom/ρ  +  ν ∇² u_hom
        +  f_lift  +  f_body
```

with

```
f_lift  =  − ∂u_lift/∂t
          − (u_hom · ∇) u_lift          (perturbation advected by lifted shear)
          − (u_lift · ∇) u_hom          (lifted flow advecting perturbation)
          + ν ∇² u_lift                 (lifted viscous diffusion)
```

The fourth term is **not** computed inside `compute_lifting_source` —
for steady Poiseuille it equals `-∂p_lift/∂z` (a constant axial body
force). It is folded into the lift's analytical pressure gradient,
which is *implicit* — the projection step only solves for `p_hom`. For
periodic-z setups this means **the lifted-pressure gradient is missing
from the actual solver state**, so an external body force must be added
back, OR the setup must use Dirichlet inlet/outlet (the recommended
configuration — see "When to use lifting" below).

### State convention with lifting

`state["u"]` always stores `u_hom`. The physical velocity is
reconstructed as `u_phys = state["u"] + u_lift_at(state["i_step"])`.

`state["u_pre_ibm"]` and `state["u_after_explicit"]` are stored in the
**physical frame** (with `u_lift` already added back) so external force
extractors are unaware of the decomposition.

`state["p"]` stores `p_hom` only. The lifted-pressure axial gradient is
analytical and never stored. Downstream consumers that need the *full*
physical pressure must add `p_lift` themselves; for Poiseuille this is
`p_lift(z) = -8μU_mean/R² · z`.

### Inlet velocity

When `lifting` is provided, all `VelocityBC.u_wall` entries should be
**zero**. The non-zero inlet velocity is enforced *implicitly* by the
lift, never by patching the spectral solver.

### When to use lifting

| Configuration                             | Use lifting? | Notes                                                                                       |
| ----------------------------------------- | ------------ | ------------------------------------------------------------------------------------------- |
| Lid-driven cavity                         | No           | Wall velocity is on the *transverse* face, the spectral basis enforces it correctly.        |
| Steady Poiseuille, periodic-z, body-force | No           | Body force drives the flow; no inlet to enforce.                                            |
| Steady Poiseuille, Dirichlet inlet/outlet | **Yes**      | Lift carries the parabolic profile; `u_hom = 0` is the steady solution.                     |
| Womersley, Dirichlet inlet/outlet         | **Yes**      | Time-varying lift built once at init; `du_lift_dt` precomputed analytically.                |
| IBM body in a lifted Poiseuille pipe      | **Yes**      | The sphere perturbs `u_hom`; the wake is captured by the projection on `u_hom`.             |

### Known caveat: `momentum_deficit_drag` with lifting

The drag estimator uses
`F = ΔM + ΔP·A + ρ·f·V_CV − F_wall(8πμU_in·L_CV)`. When the flow is
driven by a lifted pressure gradient, `state["p"]` does not include
that gradient and the `ΔP·A` term is missing the lifted contribution.
Pass `body_force = 8νU_mean/R²` (the equivalent driving rate per unit
mass for Hagen-Poiseuille) so the formula's `F_body` term restores the
analytical balance. For Womersley this becomes `body_force(t)` matching
the instantaneous driving rate.

## Vmap / scan / differentiability

* The node is a clean pytree — `jax.lax.scan` over coupled fluid +
  rigid-body integration runs without retracing (verified by
  `test_fvm_node_jax_lax_scan_integration`).
* `boundary_inputs["sphere_position"]` flows differentiably into the
  IBM body's SDF; `jax.grad(force_z, sphere_position)` works.
* `cfg` parameters (ν, ρ, IBM penalty) can be vmapped without retracing
  because none of them are baked into Python control flow inside the
  PISO step.

## Performance notes (RTX 2060 6GB, R4-P4 measurement)

* Dense O(N²) DCT/DST matmul beats cuFFT at all tested sizes (32³ → 128³).
  The default `transform_backend="auto"` therefore picks dense for any
  mesh with `N_cells < 256³`.
* IBM Brinkman update is fully fused into the PISO step; the per-step
  kernel launch count is ~5 (convection, helmholtz, projection ×2,
  Brinkman ×2). All inside `jax.lax.fori_loop` for `run_piso`.
* H100 estimates: at 256³ the FFT path becomes competitive (~2× faster
  than dense) and the recommended override is
  `transform_backend="fft"`. See `reference_xla_autotune_hopper` memory
  for autotune-cache-related compile-time pitfalls.
