---
bibliography: ../../bibliography.bib
---

# Stokeslet Fluid Node

**Module**: `mime.nodes.environment.stokeslet.fluid_node`
**Class**: `StokesletFluidNode`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-012`
**Version**: 1.0.0
**Verification Mode**: Mode 2 (Independent)

## Summary

Quasi-static regularised Stokeslet boundary-element method (BEM) fluid
solver for confined microrobot fluid-structure interaction. Computes
drag force and torque on a rigid body either via a precomputed 6×6
resistance matrix (standalone mode) or via LU-backsubstitution against
a background flow field (Schwarz coupling mode). Has no Mach-number
restriction, so it remains valid at the high rotation rates where
LBM-only solvers fail.

The node is one half of MIME's *hybrid* low-Reynolds solver: it
resolves the body drag exactly (no Mach constraint, no diffuse-band
bias), while [IBLBM](iblbm_fluid.md) and [FVM-IBM](fvm_fluid.md)
resolve the volumetric background flow with the body removed.

## Governing Equations

Quasi-static incompressible Stokes flow,

$$
\nabla p = \mu \nabla^2 \mathbf{u}, \qquad \nabla \cdot \mathbf{u} = 0,
$$

solved in boundary-integral form using the regularised Stokeslet
Green's function [@Cortez2005]:

$$
u_i(\mathbf{x}) = \frac{1}{8\pi\mu} \int_{\partial B}
   S^{\varepsilon}_{ij}(\mathbf{x}-\mathbf{y})\, f_j(\mathbf{y})\, dS_{\mathbf{y}},
$$

where the regularised kernel
$S^{\varepsilon}_{ij}(\mathbf{r})$ blends the singular Stokeslet with
a smoothing parameter $\varepsilon$ scaled to the surface-element
spacing.

Cylinder confinement uses the analytical Liron–Shahar image
correction $G_{\text{wall}}$ [@LironShahar1978]:

$$
A_{\text{conf}} = A_{\text{body}} + G_{\text{wall}}(R_{\text{cyl}}).
$$

## Discretization

* Surface mesh of $N_{\text{body}}$ collocation points with
  quadrature weights $w_k$ (regularised Stokeslet quadrature).
* System matrix $A$ is dense $3N_{\text{body}} \times 3N_{\text{body}}$.
  Standalone mode reduces it to a $6\times 6$ resistance matrix by
  solving six rigid-body forcing problems at init.
* Schwarz mode keeps the full $A$ factorisation; per-step cost is one
  LU solve.
* When `wall_table` is provided, the analytical $G_{\text{wall}}$ is
  baked into $A$ at init — there is no double-counting with a coupled
  LBM wall (LBM resolves the volumetric background; $G_{\text{wall}}$
  is an image correction on the body perturbation).

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---|---|---|
| Regularised Stokeslet kernel | `mime.nodes.environment.stokeslet.kernel` | JAX-vectorised |
| BEM system matrix $A$ | `mime.nodes.environment.stokeslet.bem.assemble_system_matrix` | |
| Liron–Shahar image $G_{\text{wall}}$ | `mime.nodes.environment.stokeslet.cylinder_wall_table.assemble_image_correction_matrix_from_table` | Precomputed table |
| 6×6 resistance matrix | `mime.nodes.environment.stokeslet.resistance.compute_resistance_matrix` | Standalone mode |
| Schwarz RHS / solve | `StokesletFluidNode._update_schwarz` | LU backsubstitution |
| Force/torque integration | `mime.nodes.environment.stokeslet.bem.compute_force_torque` | $\int \mathbf{f}\,dS$, $\int \mathbf{r}\times\mathbf{f}\,dS$ |

## Assumptions and Simplifications

1. Quasi-static Stokes flow on the body ($\partial_t \mathbf{u}$
   ignored on the BEM scale).
2. Newtonian, incompressible fluid; single $\mu$.
3. Rigid body — no surface deformation.
4. Confined mode assumes a long, straight, circular cylinder of
   radius $R_{\text{cyl}}$ and warns if the body centroid is more
   than $0.05\,R_{\text{cyl}}$ off-axis.
5. No double-counting check between the wall table and a coupled
   LBM background — those are documented to compute different things.

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|---|---|---|
| Body Reynolds | $\le 0.1$ | Stokes regime; rotation Re can be higher |
| $\kappa = a/R_{\text{cyl}}$ | $\le 0.3$ | Sphere benchmark to <4% error |
| Mesh density | mean-spacing-derived $\varepsilon$ | Robust on geodesic meshes |

## Known Limitations and Failure Modes

1. Standalone mode cannot represent pulsatile background flow or
   multi-robot wake coupling — those need Schwarz mode with LBM/FVM.
2. Off-axis bodies invalidate the cylindrical wall table; the node
   raises if the body extends outside the cylinder and warns at
   moderate offsets.
3. Dense $3N \times 3N$ memory cost limits standalone $N$ to a few
   thousand surface points on a single GPU.
4. The Force-Coupling-Method `body_force_density` flux is a stub
   that returns zeros until the Level-3 LBM coupling is wired.

## Stability Conditions

The BEM system is solved by direct LU factorisation, so there is no
explicit-timestep stability condition. Conditioning degrades for
$\varepsilon$ much smaller than the surface spacing; the node
defaults $\varepsilon = $ mean-spacing $/2$.

## State Variables

| Field | Shape | Units | Description |
|---|---|---|---|
| drag_force | (3,) | N | Hydrodynamic force on the body |
| drag_torque | (3,) | N·m | Hydrodynamic torque on the body |
| body_traction | (N, 3) | Pa | Surface traction (Schwarz mode only) |

## Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| mu | float | — | Pa·s | Dynamic viscosity |
| body_mesh | SurfaceMesh | — | — | Body surface mesh + weights |
| wall_mesh | SurfaceMesh \| None | None | — | Optional explicit wall mesh (standalone) |
| interface_mesh | SurfaceMesh \| None | None | — | Presence enables Schwarz mode |
| wall_table | WallTable \| None | None | — | Liron–Shahar image table |
| R_cyl | float \| None | None | m | Cylinder radius (required with `wall_table`) |
| epsilon | float \| None | spacing/2 | m | Regularisation length |

## Boundary Inputs

| Field | Shape | Default | Coupling Type | Description |
|---|---|---|---|---|
| body_velocity | (3,) | zeros | replacive | Body linear velocity [m/s] |
| body_angular_velocity | (3,) | zeros | replacive | Body angular velocity [rad/s] |
| body_orientation | (4,) | (1,0,0,0) | replacive | Body orientation quaternion |
| background_flow | (N, 3) | zeros | replacive | LBM velocity at body surface (Schwarz mode) |

## Boundary Fluxes (outputs)

| Field | Shape | Units | Description |
|---|---|---|---|
| drag_force | (3,) | N | Drag force on the body |
| drag_torque | (3,) | N·m | Drag torque on the body |
| body_traction | (N, 3) | Pa | Surface traction (Schwarz mode only) |
| body_force_density | (N, 3) | N/m³ | Stub for Level-3 FCM coupling |

## MIME-Specific Sections

### Anatomical Operating Context

| Compartment | Flow Regime | Re Range | Viscosity Range |
|---|---|---|---|
| Blood vessel (confined UMR) | quasi-Stokes | 0 – 0.1 | 1 – 4 mPa·s |
| Silicone vessel benchmarks | quasi-Stokes | 0 – 0.1 | 1 mPa·s (water) |

### Clinical Relevance

For confined helical UMRs, viscous drag dominates inertia and the
near-wall lubrication forces set the achievable swim speed. BEM
gives unbiased drag at the resolution needed for closed-loop
control — surface-method accuracy here is what lets the IDE's PID
trim work without retuning per geometry.

## References

- [@Cortez2005] Cortez et al. (2005). *The method of regularized Stokeslets in three dimensions.* Phys. Fluids 17(3).
- [@LironShahar1978] Liron & Shahar (1978). *Stokes flow due to a Stokeslet in a pipe.* J. Fluid Mech.
- [@Purcell1977] Purcell (1977). *Life at Low Reynolds Number.*

## Verification Evidence

- MIME-VER-stokeslet-001: sphere in unconfined flow vs. Stokes drag (<4%)
- MIME-VER-stokeslet-002: confined sphere vs. Liron–Shahar reference solution
- MIME-VER-stokeslet-003: rotational drag of helical UMR vs. de Jongh (2025)
- Unit tests: `tests/nodes/environment/stokeslet/`

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-04-10 | Initial implementation — standalone + Schwarz modes |
