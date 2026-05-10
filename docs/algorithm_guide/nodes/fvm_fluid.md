---
bibliography: ../../bibliography.bib
---

# FVM Fluid Node (with Diffuse-Penalty IBM)

**Module**: `mime.nodes.environment.fvm.fluid_node`
**Class**: `FVMFluidNode`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-020`
**Version**: 0.1.0
**Verification Mode**: Mode 2 (Independent)

## Summary

Graph-native finite-volume incompressible Navier–Stokes solver with a
diffuse-penalty immersed-boundary method (IBM). Wraps the FVM stack
(mesh + operators + PISO + IBM) as a MADDENING/MIME `SimulationNode`
so the volumetric flow can be composed in a `GraphManager` with
rigid-body, magnetic-actuation, and other MIME nodes.

The differentiable face-graph design — gather → compute → scatter on a
static-shape Cartesian mesh — follows the **DiFVM** formulation
[@DiFVM] (Du et al., 2026). DiFVM is the very recent (March 2026,
arXiv:2603.15920) reference design for a JAX-compatible
differentiable FVM; MIME's solver is structurally a DiFVM-style
gather/compute/scatter pipeline, with PISO time stepping
[@Issa1986] and Goldstein-style diffuse-penalty IBM [@Peskin2002]
layered on top.

The solver targets confined microrobot FSI at moderate $\mathrm{Re}$
where LBM's Mach-number ceiling bites and the Stokeslet BEM
quasi-static assumption breaks. It is the M3 deliverable described in
`ARCHITECTURE_PLAN.md`.

## Governing Equations

Incompressible Navier–Stokes,

$$
\partial_t \mathbf{u} + (\mathbf{u}\!\cdot\!\nabla)\mathbf{u}
   = -\tfrac{1}{\rho}\nabla p + \nu\,\nabla^2 \mathbf{u} + \mathbf{f},
\qquad
\nabla\!\cdot\!\mathbf{u} = 0,
$$

closed with a Goldstein-style diffuse-penalty IBM body force
[@Peskin2002]:

$$
\mathbf{f}_{\text{ibm}}(\mathbf{x}) =
   \alpha\, \chi_{\varepsilon}\!\big(\phi(\mathbf{x})\big)
   \big(\mathbf{u}_{\text{body}}(\mathbf{x}) - \mathbf{u}(\mathbf{x})\big),
$$

where $\phi$ is the body signed-distance field and
$\chi_{\varepsilon}$ is a smooth indicator of width $\varepsilon$.

## Discretization

* Cell-centred finite volume on a Cartesian face graph
  (gather → compute → scatter).
* PISO/projection time stepping [@Issa1986]; FFT/DST/DCT-diagonalised
  Poisson and Helmholtz solves.
* Rhie–Chow face flux to suppress pressure–velocity decoupling.
* IBM bodies are categorised as **static** (pipe wall, constructed
  once at node init) and **dynamic** (robot bodies whose SDF is
  rebuilt each step from boundary-input pose so SDF gradients with
  respect to pose remain differentiable).
* Three force-extraction modes are supported on the output flux:
  - `brinkman` — legacy per-cell penalty momentum sink (biased low
    at coarse IBM resolution).
  - `surface_integral` — Cauchy stress integrated on a shell of
    cells just outside the body. Preferred at moderate $\lambda$.
  - `momentum_deficit` — control-volume momentum balance over the
    pipe cross-section; recommended for confined flows
    ($\lambda \gtrsim 0.15$) where surface integration suffers from
    IBM-band gradient contamination.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---|---|---|
| Mesh + face graph | `mime.nodes.environment.fvm.mesh` | Static-shape pytree |
| Gather/compute/scatter operators | `mime.nodes.environment.fvm.operators` | |
| Pressure Poisson + Helmholtz | `mime.nodes.environment.fvm.pressure` | FFT/DST diagonalised |
| PISO loop | `mime.nodes.environment.fvm.piso.make_piso_step` | |
| Diffuse-penalty IBM force | `mime.nodes.environment.fvm.ibm.compute_ibm_forces` | Brinkman penalty |
| Surface-integral force | `mime.nodes.environment.fvm.ibm.surface_integral_force` | Cauchy stress on shell |
| Momentum-deficit drag | `mime.nodes.environment.fvm.ibm.momentum_deficit_drag` | Confined pipe |
| SDFs (sphere, cylinder, capsule) | `mime.nodes.environment.fvm.sdf` | Differentiable in pose |
| Lifting (Dirichlet inlet/outlet) | `mime.nodes.environment.fvm.lifting` | Optional |

## Assumptions and Simplifications

1. Incompressible, Newtonian fluid; single $\rho$ and $\nu$ per node.
2. No-slip walls (Dirichlet) or periodic boundary conditions only.
3. Cartesian mesh only — the unstructured-mesh extension is scoped
   in `ARCHITECTURE_PLAN.md` but not implemented.
4. Single-device execution: `requires_halo = True` blocks sharding.
5. IBM penalty $\alpha$ large enough that $\alpha\,\Delta t \gg 1$
   to enforce no-slip on the body.
6. The FVM does not discretise the body — drag is extracted via one
   of the three methods above so the body's bounding box never
   leaves the diffuse band.

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|---|---|---|
| $\mathrm{Re}_{\text{pipe}}$ | 0 – 500 | Tested up to ~200 in M1/M2 |
| Womersley number | 0 – 10 | Pulsatile Poiseuille verified at $\mathrm{Wo}=7$ |
| Confinement $\lambda = a/R$ | 0 – 0.30 | momentum_deficit recommended above 0.15 |

## Known Limitations and Failure Modes

1. Cartesian mesh only.
2. Body-force-driven flow only — inflow/outflow BCs are implemented
   in the operators but not wired through the node.
3. The IBM diffuse zone (~1.5 cells) shrinks the effective obstacle
   radius; drag is biased low at coarse resolution. Mesh refinement
   is the only fix.
4. Body-force-driven periodic flow takes several diffusion
   timescales ($R^2/\nu$) to reach periodic steady state. The
   initial transient is not the steady response and should not be
   reported as such.
5. `force_method="brinkman"` is kept for backwards compatibility but
   is biased low at moderate IBM resolution; prefer
   `surface_integral` or `momentum_deficit`.

## Stability Conditions

* PISO is implicit in diffusion; advective CFL still applies on the
  predictor step.
* Diagonalised pressure/Helmholtz solves are unconditionally stable.
* IBM penalty: $\alpha\,\Delta t \gtrsim 10$ recommended.

## State Variables

| Field | Shape | Units | Description |
|---|---|---|---|
| u | (Nc, dim) | m/s | Cell-centred velocity |
| p | (Nc,) | Pa | Cell-centred pressure |
| F | (Nf, dim) | m³/s | Face fluxes (Rhie–Chow) |
| t | () | s | Accumulated time |
| u_pre_ibm | (Nc, dim) | m/s | Velocity before IBM penalty (for force) |
| force_<body> | (dim,) | N | Hydrodynamic force per dynamic body |
| torque_<body> | (3,) or () | N·m | Hydrodynamic torque per dynamic body |

## Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| mesh | FVMMesh | — | — | Pre-built Cartesian mesh |
| bcs | dict[str, VelocityBC] | — | — | Patch-keyed boundary conditions |
| cfg | PisoConfig | — | — | $\nu$, $\rho$, IBM penalty, BC types |
| static_bodies | list[IBMBody] | () | — | Bodies that don't move (pipe wall) |
| dynamic_body_factories | list[(name, BodyFactory)] | () | — | Pose-driven bodies |
| body_force_fn | callable | None | m/s² | Time-dependent body force |
| lifting | LiftingFunction | None | — | Dirichlet inlet decomposition |
| force_method | str | "brinkman" | — | brinkman \| surface_integral \| momentum_deficit |
| force_shell | (float, float) | (1.5, 3.5) | dx | Shell location (surface_integral) |

## Boundary Inputs

Per dynamic body `name` (added by `dynamic_body_factories`):

| Field | Shape | Default | Coupling Type | Description |
|---|---|---|---|---|
| `<name>_position` | (dim,) | zeros | replacive | Body position [m] |
| `<name>_linear_velocity` | (dim,) | zeros | replacive | Body linear velocity [m/s] |
| `<name>_angular_velocity` | (3,) | zeros | replacive | Body angular velocity [rad/s], 3-D only |

## Boundary Fluxes (outputs)

Per dynamic body `name`:

| Field | Shape | Units | Description |
|---|---|---|---|
| `force_<name>` | (dim,) | N | Hydrodynamic force on the body |
| `torque_<name>` | (3,) or () | N·m | Hydrodynamic torque on the body |

## MIME-Specific Sections

### Anatomical Operating Context

| Compartment | Flow Regime | Re Range | Viscosity Range |
|---|---|---|---|
| Iliac artery (millibot) | oscillatory | 0 – 500 | 3 – 4 mPa·s |

### Clinical Relevance

At clinical actuation frequencies for millimeter-scale UMRs the
pulsatile boundary layer is the dominant control variable. FVM with
diffuse-penalty IBM resolves it directly without the Mach-number
ceiling that constrains LBM. Combined with the
[Stokeslet BEM](stokeslet_fluid.md) Schwarz coupling, FVM provides
the volumetric background flow while the BEM resolves the body —
the same hybrid architecture documented for IB-LBM, applied at the
$\mathrm{Re}$ regime where compressibility matters.

## References

- [@Issa1986] Issa (1986). *Solution of the implicitly discretised fluid flow equations by operator splitting.* J. Comput. Phys. 62.
- [@Peskin2002] Peskin (2002). *The immersed boundary method.* Acta Numerica 11.
- [@DiFVM] Du et al. (2026). *DiFVM: A differentiable finite volume method.* arXiv:2603.15920. — Foundational design that this node's gather/compute/scatter face-graph follows.
- [@Womersley1955] Womersley (1955). *Method for the calculation of velocity, rate of flow and viscous drag in arteries.* J. Physiol. 127:553–563.

## Verification Evidence

- MIME-VER-fvm-001: steady Poiseuille flow in a circular pipe
- MIME-VER-fvm-002: Womersley pulsatile pipe flow ($\mathrm{Wo} = 7$)
- MIME-VER-fvm-003: drag on a sphere vs. analytical Stokes /
  Schiller–Naumann at $\mathrm{Re} \in [1, 200]$
- Unit tests: `tests/nodes/environment/fvm/`

## Changelog

| Version | Date | Change |
|---|---|---|
| 0.1.0 | 2026-05-02 | Initial implementation — PISO + diffuse IBM with three force-extraction modes |
