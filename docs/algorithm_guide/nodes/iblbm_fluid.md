---
bibliography: ../../bibliography.bib
---

# IB-LBM Fluid Node

**Module**: `mime.nodes.environment.lbm.fluid_node`
**Class**: `IBLBMFluidNode`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-010`
**Version**: 1.0.0
**Verification Mode**: Mode 2 (Independent)

## Summary

3D immersed-boundary lattice Boltzmann ({term}`IB-LBM`) fluid solver for
confined microrobot flows. Wraps the existing D3Q19 LBM utilities
(`d3q19.py`, `bounce_back.py`, `helix_geometry.py`) as a MADDENING
{term}`SimulationNode` so the volumetric flow can be coupled into a
node-graph with rigid-body and magnetic-actuation nodes via edges.

The pipe wall uses simple bounce-back; the {term}`UMR` body uses simple
bounce-back or {term}`Bouzidi interpolated bounce-back <Bouzidi scheme>` depending on
`use_bouzidi`.

## Governing Equations

D3Q19 BGK lattice Boltzmann equation:

$$
f_i(\mathbf{x}+\mathbf{c}_i\Delta t,\, t+\Delta t)
= f_i(\mathbf{x},t) - \tfrac{1}{\tau}\big(f_i - f_i^{\text{eq}}\big),
$$

with macroscopic moments

$$
\rho = \sum_i f_i, \qquad \rho\,\mathbf{u} = \sum_i \mathbf{c}_i f_i,
$$

and the equilibrium expansion to $O(u^2)$ in the lattice speed.

Body coupling uses {term}`momentum-exchange <Momentum exchange>` [@Ladd1994] on the post-streaming
populations along boundary links, giving force and torque

$$
\mathbf{F}_b = \sum_{\text{links}} (f_i + f_{\bar i})\,\mathbf{c}_i,
\qquad
\mathbf{T}_b = \sum_{\text{links}} (\mathbf{r}-\mathbf{r}_{cm}) \times
        (f_i + f_{\bar i})\,\mathbf{c}_i.
$$

## Discretization

* D3Q19 stencil; BGK collision with relaxation time $\tau$.
* Two-pass bounce-back per step: static pipe wall first, then the
  rotating UMR body. The body mask is rebuilt each step from the SDF
  at the current rotation angle so the wall geometry stays
  differentiable in pose.
* Bouzidi interpolated bounce-back [@Bouzidi2001] uses sparse SDF
  $q$-values along each missing link; `max_boundary_links_per_dir`
  is pre-sized at construction to avoid silent truncation inside
  `jnp.nonzero`.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---|---|---|
| BGK collision + streaming | `mime.nodes.environment.lbm.d3q19.lbm_step_split` | |
| Equilibrium $f_i^{\text{eq}}$ | `mime.nodes.environment.lbm.d3q19.init_equilibrium` | |
| Pipe wall bounce-back | `mime.nodes.environment.lbm.bounce_back.apply_bounce_back` | First pass |
| Bouzidi IBB on UMR | `mime.nodes.environment.lbm.bounce_back.apply_bouzidi_bounce_back` | Second pass |
| Sparse $q$-values from SDF | `mime.nodes.environment.lbm.bounce_back.compute_q_values_sdf_sparse` | |
| Momentum-exchange force | `mime.nodes.environment.lbm.bounce_back.compute_momentum_exchange_force` | |
| Momentum-exchange torque | `mime.nodes.environment.lbm.bounce_back.compute_momentum_exchange_torque` | |
| LBM→SI conversion | `maddening.core.transforms.lbm_to_si_force/torque` | Edge transform |

## Assumptions and Simplifications

1. Incompressible flow: $\mathrm{Ma} \ll 1$ at every lattice node
   (validated for $\mathrm{Ma}_{\text{tip}} \le 0.1$).
2. Newtonian fluid; viscosity derived from $\tau$.
3. Rigid body, single rotation axis (z by default — only
   `body_angular_velocity[2]` is consumed).
4. Single-device execution: `requires_halo = True` blocks sharding.
5. The static pipe-wall mask is constructed once at init; only the
   dynamic body mask is rebuilt per step.

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|---|---|---|
| $\mathrm{Ma}_{\text{tip}}$ | 0 – 0.1 | Tip Mach number bound for incompressibility |
| $\tau$ | 0.55 – 1.5 | BGK relaxation stability window |
| $\mathrm{Re}_{\text{rot}}$ | 0 – ~30 | Tested across {term}`helical-UMR <Helical UMR>` replication cases |

## Known Limitations and Failure Modes

1. No multi-GPU support — halo exchange is not implemented and
   `requires_halo=True` blocks JAX sharding.
2. Per-step $q$-value recomputation costs ~0.1 s at $192^3$ on H100.
3. First step triggers JAX compilation (30–60 s at $192^3$).
4. If `max_boundary_links_per_dir` is undersized, `jnp.nonzero`
   silently truncates boundary links and accuracy degrades without
   error. The node uses a 1.5× margin over the at-init worst-case
   count.
5. Diffuse IB kernels (Peskin delta) would create
   direction-dependent velocity transfer errors at clinical
   frequencies; this node uses bounce-back instead.

## Stability Conditions

* $\tau \ge 0.55$ to keep BGK stable for the discrete equilibrium.
* $\mathrm{Ma} \le 0.1$ for the incompressible-limit expansion to
  hold pointwise.

## State Variables

| Field | Shape | Units | Description |
|---|---|---|---|
| f | (nx, ny, nz, 19) | lattice | D3Q19 populations |
| solid_mask | (nx, ny, nz) | bool | Pipe ∪ UMR mask |
| body_angle | () | rad | Accumulated body rotation |
| drag_force | (3,) | lattice | Last-step momentum exchange |
| drag_torque | (3,) | lattice | Last-step momentum exchange |

## Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| nx, ny, nz | int | — | cells | Lattice dimensions |
| tau | float | — | — | BGK relaxation time |
| vessel_radius_lu | float | — | lattice | Pipe radius in lattice units |
| body_geometry_params | dict | — | lattice | `create_umr_mask` kwargs |
| use_bouzidi | bool | False | — | Interpolated bounce-back on body |
| dx_physical | float | 1.0 | m | Physical lattice spacing (for SI conversion) |

## Boundary Inputs

| Field | Shape | Default | Coupling Type | Description |
|---|---|---|---|---|
| body_angular_velocity | (3,) | zeros | replacive | Angular velocity [rad/step] in lattice units |
| body_orientation | (4,) | (1,0,0,0) | replacive | Body orientation quaternion |

## Boundary Fluxes (outputs)

| Field | Shape | Units | Description |
|---|---|---|---|
| drag_force | (3,) | lattice | Momentum-exchange force (convert via edge) |
| drag_torque | (3,) | lattice | Momentum-exchange torque (convert via edge) |

## MIME-Specific Sections

### Anatomical Operating Context

| Compartment | Flow Regime | Re Range | Viscosity Range |
|---|---|---|---|
| Iliac artery (confined UMR) | quasi-stagnant | 0 – 0.1 | 3 – 4 mPa·s |

### Clinical Relevance

LBM gives the full volumetric flow that drives pulsatile drag and
near-wall lubrication on the UMR. In the hybrid solver, LBM
provides the background velocity that the
[Stokeslet BEM](stokeslet_fluid.md) uses as its {term}`Schwarz <Schwarz coupling>` boundary
condition — the LBM does *not* discretise the body itself, which
sidesteps Mach-number and IB-kernel pathologies at high rotation
rates.

## References

- [@Ladd1994] Ladd (1994). *Numerical simulations of particulate suspensions via a discretized Boltzmann equation.* J. Fluid Mech. 271, 285–309.
- [@Bouzidi2001] Bouzidi, Firdaouss & Lallemand (2001). *Momentum transfer of a Boltzmann-lattice fluid with boundaries.* Phys. Fluids 13(11).

## Verification Evidence

- MIME-VER-lbm-001: Poiseuille flow in cylinder (analytical)
- MIME-VER-lbm-002: Womersley unsteady pipe flow
- MIME-VER-lbm-003: Rotating-sphere torque vs. Stokes formula
- Unit tests: `tests/nodes/environment/lbm/`

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-02-18 | Initial implementation — D3Q19 BGK + Bouzidi IBB |
